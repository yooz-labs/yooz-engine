// STTModelHFDownloader.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

#if canImport(Hub)
import Hub
#endif

#if canImport(HuggingFace)
import HuggingFace
#endif

private let logger = Logger(
    subsystem: "live.yooz.engine",
    category: "STTModelHFDownloader"
)

/// Downloads STT model weights from Hugging Face on demand and reports
/// whether a snapshot is already on disk.
///
/// Mirrors the pattern in `MLXLLMBackend.isModelCached` so consumer apps
/// (yooz-whisper picker, /v1/stt/status) see consistent cache-resolution
/// semantics across LLM and STT.
///
/// Lifecycle:
/// - The downloader is stateless. A single static `snapshot(for:progress:)`
///   covers every call site today (`/v1/stt/load`).
/// - The progress closure is invoked on whatever queue swift-transformers'
///   `HubApi` calls it on; callers must marshal to their own actor as
///   needed (see `YoozSTTEngine.loadParakeetModel`).
///
/// On-disk layout:
/// - First-run downloads land under the swift-huggingface `HubCache`
///   default (`~/.cache/huggingface/hub/` for non-sandboxed processes,
///   `<container>/Library/Caches/huggingface/hub/` for sandboxed bundled
///   helpers like `yooz-whisper`'s embedded engine). Honors `HF_HUB_CACHE`
///   / `HF_HOME` overrides.
/// - The returned URL points at a snapshot directory that is directly
///   compatible with `ParakeetModel.fromDirectory(_:)` — `config.json` and
///   `model.safetensors` are surfaced via symlinks into the blob store.
enum STTModelHFDownloader {

    /// Files we actually need to load a Parakeet/FastConformer model.
    /// Restricting the glob list saves bandwidth on repos that ship
    /// extra ONNX exports, training logs, or PyTorch shadows alongside
    /// the MLX weights. `mlx-community/parakeet-tdt-0.6b-v3` is ~2.5GB
    /// for `model.safetensors` alone, so trimming auxiliary files
    /// matters on metered connections.
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
    /// cache, swift-transformers' `HubApi.snapshot` short-circuits the
    /// network entirely and returns the existing snapshot URL — so
    /// callers do not need a separate `isCached` branch for performance,
    /// only for UX (e.g. "this will download X MB" picker hint).
    ///
    /// - Parameters:
    ///   - language: STT language whose model family to fetch.
    ///   - progress: Optional handler called with fraction-completed
    ///     in `[0.0, 1.0]` as files stream in.
    /// - Throws: `STTHFDownloadError.unsupportedLanguage` if no HF id is
    ///   wired for the language family yet (FastConformer/CJK today),
    ///   `STTHFDownloadError.unavailable` if the engine binary was built
    ///   without `Hub` linkage, or any error swift-transformers'
    ///   `HubApi.snapshot` propagates.
    static func snapshot(
        for language: STTLanguage,
        progress: @escaping (Double) -> Void = { _ in }
    ) async throws -> URL {
        guard let hfID = language.huggingFaceID else {
            throw STTHFDownloadError.unsupportedLanguage(language)
        }
        #if canImport(Hub)
        logger.info(
            "Resolving HF snapshot for \(language.rawValue, privacy: .public) → \(hfID, privacy: .public)"
        )
        return try await HubApi.shared.snapshot(
            from: hfID,
            matching: modelGlobs,
            progressHandler: { p in progress(p.fractionCompleted) }
        )
        #else
        throw STTHFDownloadError.unavailable
        #endif
    }

    /// Whether the HF snapshot for `language` is fully on disk.
    ///
    /// "Fully on disk" means at least one snapshot directory under the
    /// repo's cache root contains both `config.json` and at least one
    /// `*.safetensors` blob. An empty or partial snapshot dir reports
    /// `false` so the picker doesn't claim "ready" for an interrupted
    /// download. Repo IDs without an owner segment fall back to `false`
    /// (the engine never wires such IDs today).
    ///
    /// Returns `false` for unsupported languages (no `huggingFaceID`).
    static func isCached(for language: STTLanguage) -> Bool {
        #if canImport(HuggingFace)
        guard let hfID = language.huggingFaceID else { return false }
        let parts = hfID.split(
            separator: "/",
            maxSplits: 1,
            omittingEmptySubsequences: true
        )
        guard parts.count == 2 else { return false }
        let repo = HuggingFace.Repo.ID(
            namespace: String(parts[0]),
            name: String(parts[1])
        )
        let snapshotsRoot = HubCache().snapshotsDirectory(repo: repo, kind: .model)
        guard let snapshots = try? FileManager.default.contentsOfDirectory(
            at: snapshotsRoot, includingPropertiesForKeys: nil
        ) else { return false }
        for snapshot in snapshots {
            let config = snapshot.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: config.path) else {
                continue
            }
            let entries = (try? FileManager.default.contentsOfDirectory(
                at: snapshot, includingPropertiesForKeys: nil
            )) ?? []
            let hasWeights = entries.contains { $0.pathExtension == "safetensors" }
            if hasWeights { return true }
        }
        return false
        #else
        return false
        #endif
    }
}

/// Errors specific to the HF auto-download path. Library errors from
/// `HubApi.snapshot` (network, 404, integrity) propagate as-is so call
/// sites can map them to existing wire codes (see
/// `APIServer.mapFetchFailure` for the LLM/Qwen3 mapping).
enum STTHFDownloadError: Error, LocalizedError, Sendable, Equatable {
    /// The requested language has no HF repo wired in `STTLanguage`.
    case unsupportedLanguage(STTLanguage)

    /// `Hub` could not be linked at compile time. Should never fire
    /// in shipped builds; guarded for completeness.
    case unavailable

    var errorDescription: String? {
        switch self {
        case .unsupportedLanguage(let lang):
            return "STT language \(lang.rawValue) (\(lang.modelFamily.rawValue))"
                + " does not yet have a Hugging Face mirror configured."
        case .unavailable:
            return "STT HF downloader is not linked into this build."
        }
    }
}
