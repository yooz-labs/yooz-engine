// STTWireTypes.swift
// YoozEngineWire
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - STT Backend Picker (canonical module-picker pattern, second
// adopter; AGENTS.md "Module model picker pattern"). Single definition —
// previously separate copies in `YoozEngine/Server/APITypes.swift` (server)
// and `YoozEngineClient/Types/STTTypes.swift` (SDK).

/// One STT backend in the picker. Canonical fields mirror
/// `TouchUpModelInfo`; STT-specific capability flags ride along
/// as optional extensions per AGENTS.md "Module-specific picker
/// extensions". The extensions are `Optional` on the wire so a
/// future engine that drops a capability (e.g. once every backend
/// streams, `supportsStreaming` becomes meaningless) does not
/// brick older SDK consumers.
public struct STTBackendInfo: Codable, Sendable, Equatable {
    /// Stable wire id (e.g. `parakeet`, `fast_conformer`,
    /// `apple_stt`, `qwen3_asr_preview`). Matches
    /// `STTBackendID.rawValue`.
    public let id: String
    /// Picker-visible name.
    public let displayName: String
    /// One-line subtitle for picker UX.
    public let description: String
    /// Coarse tier (`light` / `quality` / `premium` / `unknown`).
    /// MLX backends report `.quality`; Apple STT reports `.premium`
    /// (OS-provided); preview backends report `.unknown` so the UI
    /// can render a "preview" hint without inventing a new tier.
    public let tier: ModelTier
    /// Approximate first-run download size. `nil` for backends
    /// that do not require a download (Apple, bundled MLX).
    public let sizeBytes: Int64?
    /// Lifecycle state. Same total ordering as TouchUp picker.
    public let loadState: ModelLoadState
    /// Whether `/v1/stt/batch` + `/v1/stt/stream` currently route
    /// through this backend. Exactly one row per response has
    /// `isActive == true`.
    public let isActive: Bool
    public let supportsBatch: Bool?
    public let supportsStreaming: Bool?
    public let supportedLanguages: [String]?

    public init(
        id: String,
        displayName: String,
        description: String,
        tier: ModelTier,
        sizeBytes: Int64?,
        loadState: ModelLoadState,
        isActive: Bool,
        supportsBatch: Bool? = nil,
        supportsStreaming: Bool? = nil,
        supportedLanguages: [String]? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.tier = tier
        self.sizeBytes = sizeBytes
        self.loadState = loadState
        self.isActive = isActive
        self.supportsBatch = supportsBatch
        self.supportsStreaming = supportsStreaming
        self.supportedLanguages = supportedLanguages
    }
}

/// Response for `GET /v1/stt/engine`. `activeId` is the id of the
/// entry where `isActive == true`.
public struct STTBackendsResponse: Codable, Sendable, Equatable {
    public let backends: [STTBackendInfo]
    public let activeId: String

    public init(backends: [STTBackendInfo], activeId: String) {
        self.backends = backends
        self.activeId = activeId
    }
}

/// Request body for `POST /v1/stt/engine`. The legacy `engine` fallback
/// field pre-#99 clients post is a server-only decode concern — see
/// `LegacySTTSetBackendRequest` in `YoozEngine/Server/APITypes.swift` — not
/// part of this canonical shared shape.
public struct STTSetBackendRequest: Codable, Sendable {
    public let id: String
    public let preload: Bool?

    public init(id: String, preload: Bool? = nil) {
        self.id = id
        self.preload = preload
    }
}

// MARK: - Status / languages

/// Response body for `GET /v1/stt/status`. Shape parity with `LLMStatus` so
/// consumer apps can template a single progress-banner view-model over both
/// endpoints.
public struct STTStatus: Codable, Sendable, Equatable {
    public let loaded: Bool
    public let language: String?
    public let streaming: Bool
    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. `nil` when no download is in flight (idle, loaded,
    /// or Apple STT — which has no HF pull).
    public let progress: Double?
    /// Lifecycle state for the active STT backend (engine#125). `nil`
    /// on builds that predate the fire-and-forget rollout.
    public let state: LoadState?
    /// Human-readable error message when `state == .failed`. `nil`
    /// in every other state.
    public let lastError: String?

    public init(
        loaded: Bool,
        language: String?,
        streaming: Bool,
        progress: Double? = nil,
        state: LoadState? = nil,
        lastError: String? = nil
    ) {
        self.loaded = loaded
        self.language = language
        self.streaming = streaming
        self.progress = progress
        self.state = state
        self.lastError = lastError
    }
}

public struct STTLanguageInfo: Codable, Sendable, Equatable {
    public let code: String
    public let name: String
    public let implemented: Bool
    public let family: String

    public init(code: String, name: String, implemented: Bool, family: String) {
        self.code = code
        self.name = name
        self.implemented = implemented
        self.family = family
    }
}

public struct STTLanguagesResponse: Codable, Sendable, Equatable {
    public let languages: [STTLanguageInfo]

    public init(languages: [STTLanguageInfo]) {
        self.languages = languages
    }
}

// MARK: - Load / batch requests

/// Request body for `POST /v1/stt/load`. `language`/`allowFetch` are
/// optional on this canonical shape because the server tolerates an absent
/// `language` (falls back to the current backend's default) — SDK callers
/// always supply one.
public struct STTLoadRequest: Codable, Sendable, Equatable {
    public let language: String?
    /// When true (or unset), the engine fetches the model from Hugging Face
    /// if no local snapshot is staged. When false, the load fails with
    /// `model_not_cached` rather than touching the network.
    public let allowFetch: Bool?

    public init(language: String? = nil, allowFetch: Bool? = nil) {
        self.language = language
        self.allowFetch = allowFetch
    }
}

/// Request body for `POST /v1/stt/batch`. `language`/`mode` are optional on
/// this canonical shape for the same reason as `STTLoadRequest`; SDK callers
/// always supply both.
public struct BatchSTTRequest: Codable, Sendable, Equatable {
    public let samples: [Float]
    public let language: String?
    public let mode: String?
    /// Opt-in flag for per-token alignment in the response. Omitted on the
    /// wire when `nil` so old clients/servers stay byte-identical.
    public let aligned: Bool?

    public init(samples: [Float], language: String? = nil, mode: String? = nil, aligned: Bool? = nil) {
        self.samples = samples
        self.language = language
        self.mode = mode
        self.aligned = aligned
    }
}

// MARK: - Batch transcription

/// A single token from an aligned transcription with start/end timestamps.
///
/// Used by callers that need word- or sub-word-level timing. Timestamps are
/// in seconds from the start of the submitted audio buffer. The shape
/// `text/start/end` is intentionally backend-agnostic: Parakeet +
/// FastConformer surface native TDT token alignments, Apple STT derives them
/// from `SFTranscriptionSegment` (`timestamp` + `duration`).
public struct AlignedToken: Codable, Sendable, Equatable {
    public let text: String
    /// Seconds from start of the audio buffer.
    public let start: Float
    /// Seconds from start of the audio buffer. `end >= start`.
    public let end: Float

    public init(text: String, start: Float, end: Float) {
        self.text = text
        self.start = start
        self.end = end
    }
}

/// Response for `POST /v1/stt/batch`.
public struct TranscriptionResult: Codable, Sendable, Equatable {
    public let text: String
    public let finalized: String
    public let draft: String
    public let language: String?
    /// Aligned tokens with timestamps. Non-nil (possibly empty) on every
    /// response from the `aligned=true` path — the server ships
    /// `"tokens": []` on silent audio rather than omitting the key. `nil`
    /// only on responses from the non-aligned path.
    public let tokens: [AlignedToken]?

    public init(
        text: String,
        finalized: String,
        draft: String,
        language: String? = nil,
        tokens: [AlignedToken]? = nil
    ) {
        self.text = text
        self.finalized = finalized
        self.draft = draft
        self.language = language
        self.tokens = tokens
    }
}
