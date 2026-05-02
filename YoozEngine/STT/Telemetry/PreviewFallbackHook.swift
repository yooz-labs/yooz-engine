// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

// MARK: - Collaborator protocols (testable)

/// Indirection over `Qwen3ASRModelFetcher` so the fallback hook can
/// be exercised without hitting the live HF Hub. Production wiring
/// adapts the real fetcher's `download(into:)` stream.
public protocol PreviewModelFetcherAdapter: Sendable {
    /// Ensure the preview model directory is materialized on disk.
    /// Throw on any unrecoverable fetch / validation failure.
    func ensureModelOnDisk() async throws
}

/// Indirection over `Qwen3ASRBackend` so tests can inject a fake
/// that throws on load or transcribe.
public protocol PreviewBackendAdapter: Sendable {
    /// Lazy-load the preview pipeline. Throw on any load failure.
    func ensureLoaded() async throws

    /// Run a transcription. Throw on any transcribe failure.
    func transcribe(
        samples: [Float],
        language: STTLanguage
    ) async throws -> ParakeetResult
}

/// Indirection over the Parakeet path so the hook can hand off when
/// preview cold-start fails.
public protocol FallbackBackendAdapter: Sendable {
    /// Run a Parakeet (or other stable backend) transcription. Must
    /// not throw — the fallback path treats Parakeet as the floor,
    /// matching the existing `batchTranscribe` contract.
    func transcribe(
        samples: [Float],
        language: STTLanguage
    ) async -> ParakeetResult
}

/// Pluggable monotonic clock so latency measurements are testable.
public protocol PreviewFallbackClock: Sendable {
    /// Milliseconds since some fixed reference point. Only deltas
    /// are meaningful.
    func nowMs() -> UInt64
}

/// Production clock backed by `DispatchTime`.
public struct DispatchPreviewFallbackClock: PreviewFallbackClock {
    public init() {}
    public func nowMs() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds / 1_000_000
    }
}

// MARK: - Hook

/// Auto-fallback orchestrator for the `qwen3_asr_preview` backend.
///
/// On the first transcription request after a preview backend has
/// been selected, the hook tries:
///
/// 1. fetcher.ensureModelOnDisk()
/// 2. backend.ensureLoaded()
/// 3. backend.transcribe(...)
///
/// If any of those three steps throws, the hook hands the request
/// to the fallback backend (Parakeet) and emits a metrics record
/// with `fellBackFromPreview = true`.
///
/// **Cold-start scope.** Once the cold-start path resolves — either
/// step 3 succeeded *or* we fell back to Parakeet for this process —
/// the hook flips `coldStartCompleted = true` and from then on a
/// preview transcribe failure propagates instead of triggering
/// fallback. The user explicitly chose the preview backend; we only
/// protect them from the cold-start cliff, not from every subsequent
/// runtime hiccup. Flipping on fallback (rather than only on
/// success) prevents the hook from re-attempting the cold-start
/// machinery on every request when the underlying issue is sticky
/// (no network, disk full, repo missing).
public actor PreviewFallbackHook {

    private let logger = Logger(
        subsystem: "live.yooz.engine",
        category: "PreviewFallbackHook"
    )

    private let fetcher: PreviewModelFetcherAdapter
    private let preview: PreviewBackendAdapter
    private let fallback: FallbackBackendAdapter
    private let sink: any STTMetricsSink
    private let resolver: HardwareClassResolver
    private let clock: PreviewFallbackClock

    /// Flips `true` after the cold-start path resolves — either the
    /// first preview transcribe succeeded, or fallback ran. State
    /// machine: `false` → fallback-eligible, `true` → propagate.
    /// Renamed from `hasSucceededOnce` for accuracy; tests may still
    /// query the flag via `hasSucceededOnceForTesting()`.
    private var coldStartCompleted: Bool = false

    public init(
        fetcher: PreviewModelFetcherAdapter,
        preview: PreviewBackendAdapter,
        fallback: FallbackBackendAdapter,
        sink: any STTMetricsSink,
        resolver: HardwareClassResolver = SystemHardwareClassResolver(),
        clock: PreviewFallbackClock = DispatchPreviewFallbackClock()
    ) {
        self.fetcher = fetcher
        self.preview = preview
        self.fallback = fallback
        self.sink = sink
        self.resolver = resolver
        self.clock = clock
    }

    /// Outcome of a hook-mediated transcription request. The caller
    /// returns `result.text` to the user; `fellBack` is exposed so
    /// the calling layer can update its UI state (e.g. flip the
    /// active-backend indicator back to Parakeet).
    public struct Outcome: Sendable, Equatable {
        public let result: ParakeetResult
        public let fellBack: Bool
        public let backendUsed: STTBackendID

        public init(
            result: ParakeetResult,
            fellBack: Bool,
            backendUsed: STTBackendID
        ) {
            self.result = result
            self.fellBack = fellBack
            self.backendUsed = backendUsed
        }
    }

    /// Test-only accessor — true once the cold-start path has
    /// resolved (success or fallback).
    func hasSucceededOnceForTesting() -> Bool { coldStartCompleted }

    /// Alias — `true` once the cold-start path has resolved (either
    /// preview succeeded or fallback ran).
    func coldStartCompletedForTesting() -> Bool { coldStartCompleted }

    /// Run a transcription through the preview backend, with
    /// auto-fallback on cold-start failure.
    public func attemptPreviewWithFallback(
        samples: [Float],
        language: STTLanguage,
        sampleRate: Int = 16_000
    ) async -> Outcome {
        let started = clock.nowMs()
        let audioMs = audioDurationMs(
            sampleCount: samples.count,
            sampleRate: sampleRate
        )

        // Cold-start fallback path: protect the first transcribe call
        // from fetch / load / transcribe failures.
        if !coldStartCompleted {
            do {
                try await fetcher.ensureModelOnDisk()
                try await preview.ensureLoaded()
                let result = try await preview.transcribe(
                    samples: samples, language: language
                )
                coldStartCompleted = true
                await emitMetric(
                    backend: .qwen3ASRPreview,
                    audioMs: audioMs,
                    started: started,
                    fellBack: false
                )
                return Outcome(
                    result: result,
                    fellBack: false,
                    backendUsed: .qwen3ASRPreview
                )
            } catch {
                logger.warning(
                    "Preview cold-start failed (\(String(describing: error), privacy: .public)); falling back to Parakeet"
                )
                let result = await fallback.transcribe(
                    samples: samples, language: language
                )
                // Flip even on fallback so the hook stops thrashing
                // the cold-start path when the underlying issue is
                // sticky. Subsequent requests in this process go
                // directly to the post-warmup branch (preview
                // failures propagate, no further fallback) — the
                // user's preview selection is honored at next
                // process start.
                coldStartCompleted = true
                await emitMetric(
                    backend: .parakeet,
                    audioMs: audioMs,
                    started: started,
                    fellBack: true
                )
                return Outcome(
                    result: result,
                    fellBack: true,
                    backendUsed: .parakeet
                )
            }
        }

        // Post-warmup path: failures propagate. We deliberately do
        // NOT call `fallback.transcribe` here — the user has already
        // had one successful preview run and chose this backend.
        do {
            let result = try await preview.transcribe(
                samples: samples, language: language
            )
            await emitMetric(
                backend: .qwen3ASRPreview,
                audioMs: audioMs,
                started: started,
                fellBack: false
            )
            return Outcome(
                result: result,
                fellBack: false,
                backendUsed: .qwen3ASRPreview
            )
        } catch {
            logger.error(
                "Preview transcribe failed post-warmup (\(String(describing: error), privacy: .public)); not falling back"
            )
            // Surface an empty result; the metric still fires so
            // dashboards see the failure rate. We do NOT mark this as
            // a fallback because the user's selection is honored.
            await emitMetric(
                backend: .qwen3ASRPreview,
                audioMs: audioMs,
                started: started,
                fellBack: false
            )
            return Outcome(
                result: .empty,
                fellBack: false,
                backendUsed: .qwen3ASRPreview
            )
        }
    }

    // MARK: - Internals

    private func audioDurationMs(
        sampleCount: Int, sampleRate: Int
    ) -> UInt32 {
        guard sampleRate > 0 else { return 0 }
        let ms = (UInt64(sampleCount) * 1_000) / UInt64(sampleRate)
        return UInt32(min(ms, UInt64(UInt32.max)))
    }

    private func emitMetric(
        backend: STTBackendID,
        audioMs: UInt32,
        started: UInt64,
        fellBack: Bool
    ) async {
        let elapsed = clock.nowMs() &- started
        let metric = STTBackendMetrics(
            backend: backend,
            modelVariant: STTBackendMetrics.defaultModelVariant(
                for: backend
            ),
            audioDurationMs: audioMs,
            timeToFirstTokenMs: nil,
            endToEndLatencyMs: UInt32(min(elapsed, UInt64(UInt32.max))),
            hardwareClass: resolver.resolve(),
            fellBackFromPreview: fellBack,
            timestampUTC: Date()
        )
        await sink.record(metric)
    }
}

// MARK: - Production adapters

/// Adapter wrapping `Qwen3ASRModelFetcher.download(into:)`. Drains
/// the progress stream and translates "stream finished without
/// throwing" into "fetch succeeded".
public struct Qwen3ASRPreviewFetcherAdapter: PreviewModelFetcherAdapter {
    private let fetcher: Qwen3ASRModelFetcher
    private let modelDir: URL

    public init(
        fetcher: Qwen3ASRModelFetcher = .shared,
        modelDir: URL = Qwen3ASRModelFetcher.defaultModelDir
    ) {
        self.fetcher = fetcher
        self.modelDir = modelDir
    }

    public func ensureModelOnDisk() async throws {
        if await fetcher.isModelDirReady(modelDir) {
            return
        }
        let stream = await fetcher.download(into: modelDir)
        for try await _ in stream {
            // Drain. Progress consumption lives in the HTTP /
            // client layers (issue #59 sample snippet).
        }
    }
}

/// Adapter wrapping the `Qwen3ASRBackend` actor singleton.
public struct Qwen3ASRPreviewBackendAdapter: PreviewBackendAdapter {
    private let backend: Qwen3ASRBackend
    private let modelDir: URL

    public init(
        backend: Qwen3ASRBackend = .shared,
        modelDir: URL = Qwen3ASRModelFetcher.defaultModelDir
    ) {
        self.backend = backend
        self.modelDir = modelDir
    }

    public func ensureLoaded() async throws {
        try await backend.ensureLoaded(modelDir: modelDir)
    }

    public func transcribe(
        samples: [Float],
        language: STTLanguage
    ) async throws -> ParakeetResult {
        // Shared mapping with `YoozSTTEngine.batchTranscribeQwen3`.
        let result = try await backend.transcribe(
            pcm: samples, language: language.qwen3LanguageHint
        )
        return ParakeetResult(
            text: result.text, finalized: result.text, draft: ""
        )
    }
}

/// Adapter wrapping `YoozSTTEngine.batchTranscribe` for the Parakeet
/// fallback path.
public struct ParakeetFallbackAdapter: FallbackBackendAdapter {
    private static let logger = Logger(
        subsystem: "live.yooz.engine",
        category: "ParakeetFallbackAdapter"
    )

    private let engine: YoozSTTEngine

    public init(engine: YoozSTTEngine = .shared) {
        self.engine = engine
    }

    public func transcribe(
        samples: [Float],
        language: STTLanguage
    ) async -> ParakeetResult {
        // The Parakeet path expects the model to be loaded for the
        // requested language. `loadParakeetModel` is idempotent for
        // the same language and cheap once loaded.
        //
        // We do NOT call `setBackend(.parakeet)` here. The fallback
        // is a per-request escape hatch; flipping `currentBackend`
        // would silently strip the user of their preview selection
        // for every subsequent request without going through
        // `POST /v1/stt/engine`. The hook's `coldStartCompleted`
        // flag (in `PreviewFallbackHook`) is the one piece of
        // state that survives across requests and is enough to
        // prevent thrashing.
        do {
            try await engine.loadParakeetModel(language: language)
        } catch {
            // The fallback itself failed to start. Log loudly — the
            // hook can't surface this any other way (`FallbackBackendAdapter`
            // returns non-throwing), and a silent empty result here
            // would leave the user staring at no transcript with no
            // diagnostic.
            Self.logger.error(
                "Parakeet fallback failed to start (\(String(describing: error), privacy: .public)); returning empty result"
            )
            return .empty
        }
        return await engine.batchTranscribe(samples: samples)
    }
}
