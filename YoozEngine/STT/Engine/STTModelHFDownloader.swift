// STTModelHFDownloader.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import HuggingFace
import os.log

private let logger = Logger(
    subsystem: "live.yooz.engine",
    category: "STTModelHFDownloader"
)

/// Downloads STT model weights from Hugging Face on demand and reports
/// whether a snapshot is already on disk.
///
/// Uses `HuggingFace.HubClient.downloadSnapshot` (the same path the LLM
/// backend hits via the `#hubDownloader()` macro). Files land under the
/// Python-compatible `HubCache` root (default `~/.cache/huggingface/hub/`,
/// honors `HF_HUB_CACHE` / `HF_HOME`, falls back to the sandbox container's
/// caches dir for bundled helpers). The cache is shared with the LLM
/// path so a user who already pulled a Yooz model has it available for
/// every variant of the engine.
///
/// Notes for callers:
/// - Errors from `HubClient.downloadSnapshot` (`HTTPClientError`,
///   `HubCacheError`, `URLError`, `CancellationError`) propagate
///   unchanged so `APIServer.mapSTTLoadError` can demux them into
///   structured wire codes (matches the Qwen3 path's `mapFetchFailure`).
/// - Pass `localFilesOnly: true` to enforce offline mode — the call
///   throws `HubCacheError.cachedPathResolutionFailed` if no cached
///   snapshot exists rather than touching the network. This is what
///   `/v1/stt/load`'s `allow_fetch=false` branch relies on.
/// - The progress closure is hopped to the main actor by `HubClient`
///   itself; callers must not assume any other queue.
enum STTModelHFDownloader {

    /// Files we actually need to load a Parakeet/FastConformer model.
    /// Restricting the glob list saves bandwidth on repos that ship
    /// extra ONNX exports, training logs, or PyTorch shadows alongside
    /// the MLX weights. `mlx-community/parakeet-tdt-0.6b-v3` is ~2.5GB
    /// for `model.safetensors` alone, so trimming auxiliary files
    /// matters on metered connections.
    ///
    /// `isCached(for:)` uses the same set as a coupling point — adding
    /// a new required file means updating both this constant and the
    /// `requiredCachedFileExtensions` set below.
    static let modelGlobs: [String] = [
        "config.json",
        "*.safetensors",
        "tokenizer.*",
        "vocab.txt",
        "preprocessor_config.json"
    ]

    /// Download the HF snapshot for `language` if not already cached;
    /// return the local snapshot URL ready to pass to
    /// `ParakeetModel.fromDirectory(_:)` /
    /// `FastConformerModel.fromDirectory(_:)`.
    ///
    /// If every file the glob list matches is already present in the
    /// `HubCache`, `HubClient.downloadSnapshot` short-circuits and
    /// returns the existing snapshot URL without HEADing the upstream.
    ///
    /// - Parameters:
    ///   - language: STT language whose model family to fetch.
    ///   - localFilesOnly: When `true`, throw `HubCacheError
    ///     .cachedPathResolutionFailed` instead of fetching. Used by
    ///     `/v1/stt/load`'s `allow_fetch=false` path.
    ///   - progress: Optional handler called on the main actor with
    ///     fraction-completed in `[0.0, 1.0]` as files stream in.
    /// - Throws: `STTHFDownloadError.unsupportedLanguage` if no HF id is
    ///   wired for the language family, or any error
    ///   `HubClient.downloadSnapshot` propagates (`HTTPClientError`,
    ///   `HubCacheError`, `URLError`, `CancellationError`).
    static func snapshot(
        for language: STTLanguage,
        localFilesOnly: Bool = false,
        progress: (@MainActor @Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        guard let repo = repoID(for: language) else {
            throw STTHFDownloadError.unsupportedLanguage(language)
        }
        logger.info(
            "Resolving HF snapshot for \(language.rawValue, privacy: .public) → \(repo.namespace, privacy: .public)/\(repo.name, privacy: .public) (localFilesOnly=\(localFilesOnly, privacy: .public))"
        )
        let progressHandler: (@MainActor @Sendable (Progress) -> Void)?
        if let handler = progress {
            progressHandler = { @MainActor @Sendable (foundationProgress: Progress) -> Void in
                handler(foundationProgress.fractionCompleted)
            }
        } else {
            progressHandler = nil
        }
        return try await HubClient.default.downloadSnapshot(
            of: repo,
            kind: .model,
            matching: modelGlobs,
            localFilesOnly: localFilesOnly,
            progressHandler: progressHandler
        )
    }

    /// Whether the HF snapshot for `language` is fully on disk.
    ///
    /// "Fully on disk" means at least one snapshot directory under the
    /// repo's cache root contains both `config.json` and at least one
    /// `*.safetensors` blob (live or via symlink). An empty or partial
    /// snapshot dir reports `false` so the picker UX in yooz-whisper
    /// doesn't claim "ready" for an interrupted download.
    ///
    /// Returns `false` for unsupported languages (no `huggingFaceID`).
    /// `FileManager` errors are logged at `.warning` and treated as
    /// uncached — better to re-probe via a real `downloadSnapshot` call
    /// than to silently skip a `.cachedPathResolutionFailed` redirect
    /// the user could fix (permissions, disk full).
    static func isCached(for language: STTLanguage) -> Bool {
        guard let repo = repoID(for: language) else { return false }
        let snapshotsRoot = HubCache().snapshotsDirectory(repo: repo, kind: .model)

        let snapshots: [URL]
        do {
            snapshots = try FileManager.default.contentsOfDirectory(
                at: snapshotsRoot,
                includingPropertiesForKeys: nil
            )
        } catch let error as NSError where error.code == NSFileReadNoSuchFileError {
            return false
        } catch {
            logger.warning(
                "isCached probe failed for \(repo.namespace, privacy: .public)/\(repo.name, privacy: .public): \(error.localizedDescription, privacy: .public) — assuming uncached"
            )
            return false
        }

        for snapshot in snapshots {
            guard FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("config.json").path
            ) else { continue }
            let entries: [URL]
            do {
                entries = try FileManager.default.contentsOfDirectory(
                    at: snapshot, includingPropertiesForKeys: nil
                )
            } catch {
                logger.warning(
                    "isCached snapshot read failed at \(snapshot.path, privacy: .public): \(error.localizedDescription, privacy: .public)"
                )
                continue
            }
            if entries.contains(where: { $0.pathExtension == "safetensors" }) {
                return true
            }
        }
        return false
    }

    /// Build a `HuggingFace.Repo.ID` from the language's `huggingFaceID`
    /// string. Returns `nil` when the language has no mirror wired or
    /// when the id does not split cleanly into `namespace/name` (the
    /// latter is a typo guard — `STTLanguageHuggingFaceIDTests` pins
    /// the contract so this should never fire in practice).
    static func repoID(for language: STTLanguage) -> Repo.ID? {
        guard let id = language.huggingFaceID else { return nil }
        let parts = id.split(
            separator: "/", maxSplits: 1, omittingEmptySubsequences: true
        )
        guard parts.count == 2,
              !parts[0].isEmpty,
              !parts[1].isEmpty else {
            logger.error("Malformed HF id \(id, privacy: .public) for \(language.rawValue, privacy: .public)")
            return nil
        }
        return Repo.ID(
            namespace: String(parts[0]),
            name: String(parts[1])
        )
    }
}

/// Errors specific to the HF auto-download path.
///
/// Library errors from `HubClient.downloadSnapshot` (`HTTPClientError`,
/// `HubCacheError`, `URLError`, `CancellationError`) propagate as-is so
/// the route handler can map them to existing wire codes (see
/// `APIServer.mapSTTLoadError`). Only the truly STT-specific cases live
/// in this enum.
public enum STTHFDownloadError: Error, LocalizedError, Sendable, Equatable {
    /// The requested language has no HF repo wired in `STTLanguage`.
    /// Maps to `language_unmirrored` (501 Not Implemented) on the wire.
    case unsupportedLanguage(STTLanguage)

    public var errorDescription: String? {
        switch self {
        case .unsupportedLanguage(let lang):
            return "STT language \(lang.rawValue) (\(lang.modelFamily.rawValue))"
                + " does not yet have a Hugging Face mirror configured."
        }
    }
}
