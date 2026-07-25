// STTDownloadCoordinator.swift
// STTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

// In the SPM build `Qwen3ASR` is its own target; under xcodegen its sources
// compile into STTModule (already in scope). Same conditional the rest of the
// STT sources use — see `YoozSTTEngine.swift`.
#if canImport(Qwen3ASR)
    import Qwen3ASR
#endif

/// Explicit, per-backend STT model downloads (engine#291) — the speech-side
/// twin of the touch-up picker's `/v1/touchup/download`.
///
/// **Why this is separate from the load path.** `YoozSTTEngine.enqueueLoad`
/// has ONE load slot and cancels the in-flight load whenever the requested
/// language changes ("Different language while loading → cancel the prior
/// load"), and `setBackend` tears down the previous backend's state. So
/// driving a download through the load path means any engine/language switch
/// kills a multi-GB fetch in progress — the STT analogue of the eviction bug
/// engine#289 fixed for touch-up tiers. Downloading only needs bytes on
/// disk; it does not need the active backend, the load slot, or any weights
/// in memory. Keeping it in its own actor with a task PER BACKEND makes the
/// download independent of selection by construction rather than by
/// defensive guards.
///
/// Progress is published on the `stt` module with `modelId` set to the
/// backend's wire id, so a consumer can scope a progress row to the right
/// picker row instead of guessing which backend a module-wide fraction
/// belongs to (the misattribution bug whisper#338's review caught).
public actor STTDownloadCoordinator {
    public static let shared = STTDownloadCoordinator()

    /// `EngineEvent.module` key for STT rows — matches the STT picker's
    /// route family (`/v1/stt/engine`).
    public static let module = "stt"

    private let logger = Logger(
        subsystem: "live.yooz.engine", category: "STTDownloadCoordinator"
    )

    /// In-flight download per backend. A switch never touches another
    /// backend's entry; same-backend re-requests dedupe onto the running
    /// task. Entries remove themselves on completion.
    private var tasks: [STTBackendID: Task<Void, Never>] = [:]

    /// Latest fraction per backend while its download runs, for the
    /// `/v1/stt/engine` rows and the `/v1/state` snapshot. Cleared when the
    /// download settles so a finished backend never reports "downloading".
    private var fractions: [STTBackendID: Double] = [:]

    private init() {}

    /// Fraction-completed per backend wire id for any download currently in
    /// flight. Read by the row builders; empty when nothing is downloading.
    public func inFlightFractions() -> [String: Double] {
        var result: [String: Double] = [:]
        for (backend, fraction) in fractions where fraction > 0 && fraction < 1 {
            result[backend.rawValue] = fraction
        }
        return result
    }

    /// Whether `backend` has a download in flight right now.
    public func isDownloading(_ backend: STTBackendID) -> Bool {
        tasks[backend] != nil
    }

    /// Wire ids whose weights are already complete on disk (engine#291).
    ///
    /// The picker row builders need this to distinguish "downloaded but not
    /// active" from "not downloaded": before this, every non-active backend
    /// reported `.available`, so a consumer's Download affordance would
    /// appear for models the user already has. Kept here so both transports
    /// share one definition of "downloaded" and it can't drift.
    ///
    /// Pure disk inspection, no network: Qwen3 checks the fetcher's own
    /// integrity sentinel (`isModelDirReady` — required files plus the
    /// post-validation marker, so a half-staged dir reads as missing), and
    /// the MLX backends ask the HF cache for a local-only resolve.
    public func downloadedBackendIds(language: STTLanguage) async -> Set<String> {
        var downloaded: Set<String> = []
        for backend in STTBackendID.allCases where backend.requiresDownload {
            switch backend {
            case .qwen3ASRPreview:
                let fetcher = Qwen3ASRModelFetcher()
                if await fetcher.isModelDirReady(Qwen3ASRModelFetcher.defaultModelDir) {
                    downloaded.insert(backend.rawValue)
                }
            case .parakeet, .fastConformer:
                // Single-language probe, collapsed to a per-backend answer.
                // Accurate today (PR #295 review verified): Parakeet resolves
                // the SAME shared repo for all its languages, and
                // FastConformer's `huggingFaceID` is nil for all three of its
                // languages until engine#41 publishes MLX mirrors, so it
                // always — correctly — reports not-cached. REVISIT WITH #41:
                // if FastConformer ships PER-LANGUAGE repos, a snapshot
                // cached under one language would be missed when the probe
                // picks another, and this needs to become per-language state
                // rather than a single row flag.
                let resolvable = backend.supportedLanguages.contains(language)
                    ? language
                    : backend.supportedLanguages.first
                guard let probeLanguage = resolvable else { continue }
                if (try? await STTModelHFDownloader.snapshot(
                    for: probeLanguage, localFilesOnly: true
                )) != nil {
                    downloaded.insert(backend.rawValue)
                }
            case .appleSTT:
                continue
            }
        }
        return downloaded
    }

    /// Start (or join) a download for `backend`. Returns immediately: the
    /// fetch runs detached, and progress/outcome arrive as `downloadProgress`
    /// / `loadStateChanged` events on the `stt` module. Backends that need no
    /// download (Apple STT) are a no-op.
    public func requestDownload(_ backend: STTBackendID, language: STTLanguage) async {
        guard backend.requiresDownload else {
            logger.info(
                "requestDownload: \(backend.rawValue, privacy: .public) needs no download"
            )
            return
        }
        // Already complete on disk → nothing to fetch (PR #295 review),
        // matching `TouchUpEngine.requestDownload`'s early return. Worth the
        // probe: `STTModelHFDownloader.snapshot` always passes
        // `revision: "main"`, and swift-huggingface's "already fully cached"
        // fast path only triggers for a commit hash — so re-requesting a
        // cached backend would do a real network manifest fetch before
        // concluding there is nothing to do, and publish a redundant
        // `.cached` event on the way out.
        if await downloadedBackendIds(language: language).contains(backend.rawValue) {
            logger.info(
                "requestDownload: \(backend.rawValue, privacy: .public) already on disk"
            )
            return
        }
        guard tasks[backend] == nil else {
            logger.info(
                "requestDownload: \(backend.rawValue, privacy: .public) already in flight"
            )
            return
        }
        tasks[backend] = Task { [weak self] in
            await self?.runDownload(backend, language: language)
            await self?.clearTask(backend)
        }
    }

    /// Cancel an in-flight download for `backend`. No-op when it isn't
    /// downloading — never disturbs a settled or loaded backend.
    public func cancelDownload(_ backend: STTBackendID) {
        guard let task = tasks[backend] else { return }
        logger.info("cancelDownload: \(backend.rawValue, privacy: .public)")
        task.cancel()
        tasks[backend] = nil
        fractions[backend] = nil
    }

    // MARK: - Private

    private func clearTask(_ backend: STTBackendID) {
        tasks[backend] = nil
        fractions[backend] = nil
    }

    private func setFraction(_ fraction: Double, for backend: STTBackendID) async {
        // Publish on a 0.5% move or better; the STT fetchers report real
        // per-file BYTE counts (unlike the LLM path's per-file-only
        // granularity, engine#292), so this animates smoothly.
        let previous = fractions[backend] ?? -1
        // Monotonic (PR #294 review): the Parakeet/FastConformer path must
        // hand `STTModelHFDownloader.snapshot` a synchronous @MainActor
        // closure, so each ~100ms sampling tick spawns its own unstructured
        // Task into this actor — nothing orders them, and a stale smaller
        // fraction landing after a larger one would publish a bar that walks
        // backwards. Dropping a regressed sample costs nothing: the next tick
        // carries the current value.
        guard fraction >= previous else { return }
        fractions[backend] = fraction
        guard fraction > 0, fraction < 1, fraction - previous >= 0.005 else { return }
        await EngineEventBus.shared.publish(EngineEvent(
            kind: .downloadProgress, module: Self.module,
            modelId: backend.rawValue, progress: fraction
        ))
    }

    private func publishSettled(
        _ backend: STTBackendID, state: ModelLoadState, message: String? = nil
    ) async {
        await EngineEventBus.shared.publish(EngineEvent(
            kind: .loadStateChanged, module: Self.module,
            modelId: backend.rawValue, loadState: state, message: message
        ))
    }

    private func runDownload(_ backend: STTBackendID, language: STTLanguage) async {
        // Keep the hosting process alive for the whole fetch (engine#286):
        // launchd idle-exits a nested XPC service on transaction count, and
        // a detached download holds none.
        let keepAlive = ProcessInfo.processInfo.beginActivity(
            options: [.automaticTerminationDisabled, .suddenTerminationDisabled],
            reason: "STT model download: \(backend.rawValue)"
        )
        defer { ProcessInfo.processInfo.endActivity(keepAlive) }

        logger.info(
            "download start: \(backend.rawValue, privacy: .public) lang=\(language.rawValue, privacy: .public)"
        )
        do {
            switch backend {
            case .qwen3ASRPreview:
                try await runQwen3Download()
            case .parakeet, .fastConformer:
                // HF snapshot fetch, no memory load: `snapshot` resolves (and
                // downloads) the language's repo into the shared cache, which
                // is exactly what the lazy load path would later find.
                _ = try await STTModelHFDownloader.snapshot(
                    for: language,
                    localFilesOnly: false,
                    progress: { [weak self] fraction in
                        Task { await self?.setFraction(fraction, for: backend) }
                    }
                )
            case .appleSTT:
                return // guarded by `requiresDownload` above
            }
            await publishSettled(backend, state: .cached)
            logger.info("download done: \(backend.rawValue, privacy: .public)")
        } catch is CancellationError {
            // User-initiated cancel is not a failure: report the honest
            // post-cancel state with no error message so the picker shows
            // "not downloaded" rather than an error toast.
            await publishSettled(backend, state: .available)
            logger.info("download cancelled: \(backend.rawValue, privacy: .public)")
        } catch {
            await publishSettled(
                backend, state: .available, message: error.localizedDescription
            )
            logger.error(
                "download failed: \(backend.rawValue, privacy: .public) — \(error.localizedDescription, privacy: .public)"
            )
        }
    }

    /// Drive the Qwen3-ASR fetcher's progress stream. It reports real byte
    /// counts per file, so the fraction is genuine rather than per-file
    /// stepped.
    private func runQwen3Download() async throws {
        let fetcher = Qwen3ASRModelFetcher()
        let modelDir = Qwen3ASRModelFetcher.defaultModelDir
        var totalBytes: Int64 = 0
        var completedBytes: Int64 = 0
        var currentFileDone: Int64 = 0

        for try await progress in await fetcher.download(into: modelDir) {
            try Task.checkCancellation()
            switch progress {
            case let .manifestResolved(total, _):
                totalBytes = total
            case .fileStarted:
                currentFileDone = 0
            case let .fileBytes(_, completed, _):
                currentFileDone = completed
            case let .fileFinished(_, bytes):
                completedBytes += bytes
                currentFileDone = 0
            case .tokenizerPrepStarted, .tokenizerPrepFinished, .done:
                continue
            }
            guard totalBytes > 0 else { continue }
            let fraction = Double(completedBytes + currentFileDone) / Double(totalBytes)
            await setFraction(min(fraction, 0.999), for: .qwen3ASRPreview)
        }
    }
}

extension STTBackendID {
    /// Whether selecting this backend implies fetching weights. Apple STT is
    /// OS-provided, so it never downloads.
    var requiresDownload: Bool {
        self != .appleSTT
    }
}
