// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Identifier for the active STT backend.
///
/// Backends are HTTP-switchable via `POST /v1/stt/engine` and config-
/// selectable via `EngineConfig.sttBackend`. Default is `parakeet`;
/// other backends are opt-in.
public enum STTBackendID: String, Codable, Sendable, CaseIterable {
    /// Parakeet TDT (default for Latin / European languages).
    case parakeet
    /// FastConformer Hybrid (Arabic / Persian / Hebrew).
    case fastConformer = "fast_conformer"
    /// Apple SFSpeechRecognizer (on-device).
    case appleSTT = "apple_stt"
    /// Qwen3-ASR 1.7B 8-bit (preview, batch-only in Phase 5).
    case qwen3ASRPreview = "qwen3_asr_preview"

    /// Whether this backend currently exposes a streaming endpoint.
    /// `qwen3_asr_preview` returns false for Phase 5; streaming lands
    /// in Phase 7 (issue #58).
    public var supportsStreaming: Bool {
        switch self {
        case .parakeet, .fastConformer:
            true
        case .appleSTT:
            true
        case .qwen3ASRPreview:
            false
        }
    }

    /// Whether this backend currently exposes a batch endpoint.
    public var supportsBatch: Bool {
        // All four backends support batch; the streaming-only flag is
        // the differentiator.
        true
    }

    // MARK: - Phase 6 — Display helpers (engine-side source of truth)

    /// User-facing label for the backend. Used by Whisper / Notes
    /// UIs (which live in their own repos) so they don't drift from
    /// the engine's canonical naming.
    public var displayName: String {
        switch self {
        case .parakeet:        return "Parakeet (Recommended)"
        case .fastConformer:   return "FastConformer (Arabic / Persian / Hebrew)"
        case .appleSTT:        return "Apple Speech (On-device)"
        case .qwen3ASRPreview: return "Multilingual (Preview)"
        }
    }

    /// `true` for backends explicitly tagged as preview (the user
    /// should be told the engine may auto-fallback). `false` for the
    /// stable, built-in backends.
    public var isPreview: Bool {
        switch self {
        case .qwen3ASRPreview: return true
        case .parakeet, .fastConformer, .appleSTT: return false
        }
    }

    /// Estimated first-run download size in megabytes. Returns `nil`
    /// for built-in backends (Parakeet, Apple) that ship with the
    /// engine and don't need a runtime fetch.
    ///
    /// Numbers are approximate (rounded to MB) and intended for
    /// "Download ~3500 MB?" confirmation dialogs, not exact
    /// progress display — `Qwen3ASRModelFetcher` streams precise
    /// byte counts via `DownloadProgress`.
    public var estimatedDownloadMB: Int? {
        switch self {
        case .parakeet:        return nil
        case .fastConformer:   return nil
        case .appleSTT:        return nil
        case .qwen3ASRPreview: return 3_500
        }
    }

    // MARK: - Languages

    /// Languages this backend can transcribe via the engine. The list
    /// reflects what the engine knows how to route — for `qwen3_asr_preview`
    /// the underlying model supports many more languages but the engine
    /// only exposes the ones it has fixtures for in Phase 5.
    public var supportedLanguages: [STTLanguage] {
        switch self {
        case .parakeet:
            return STTLanguage.allCases.filter {
                $0.modelFamily == .parakeetTDT
            }
        case .fastConformer:
            return [.arabic, .persian, .hebrew]
        case .appleSTT:
            // Apple covers everything available on the OS; we list the
            // engine's known set.
            return STTLanguage.allCases
        case .qwen3ASRPreview:
            // Qwen3-ASR is multilingual; the engine routes the canonical
            // set used in Phase 4 parity work.
            return [.english, .arabic, .persian]
        }
    }
}
