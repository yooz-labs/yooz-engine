import Foundation

/// A single token from an aligned transcription with start/end timestamps.
///
/// Used by callers that need word- or sub-word-level timing (e.g. chunk-
/// boundary deduplication, subtitle rendering, word highlighting). Timestamps
/// are in seconds from the start of the submitted audio buffer.
///
/// Only populated on responses from aligned transcription paths (see
/// `STTClient.batchTranscribeAligned`). The shape `text/start/end` is
/// intentionally backend-agnostic: Parakeet + FastConformer surface native
/// TDT token alignments, Apple STT derives them from `SFTranscriptionSegment`
/// (`timestamp` + `duration`).
public struct AlignedToken: Codable, Sendable, Equatable {
    /// Token text. May be a word, sub-word, or punctuation fragment depending
    /// on backend tokenization; do not assume one-token-per-word.
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

public struct TranscriptionResult: Codable, Sendable {
    public let text: String
    public let finalized: String
    public let draft: String
    public let language: String?
    /// Aligned tokens with timestamps. Non-nil (possibly empty) on every
    /// response from `STTClient.batchTranscribeAligned` for the MLX + Apple
    /// backends — the server ships `"tokens": []` on silent audio rather than
    /// omitting the key. `nil` only on responses from `STTClient.transcribe`,
    /// which never opts into the aligned path.
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

public enum STTLanguage: String, Codable, Sendable, CaseIterable {
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case dutch = "nl"
    case polish = "pl"
    case russian = "ru"
    case ukrainian = "uk"
    case arabic = "ar"
    case persian = "fa"
    case hebrew = "he"
    case chinese = "zh"
    case japanese = "ja"
    case korean = "ko"
    case cantonese = "yue"
}

// MARK: - STT Backend Picker (canonical pattern, second adopter)
//
// Mirror of the engine's `STTBackendInfo` / `STTBackendsResponse`
// wire shapes (#99). Same tier + loadState typed enums as TouchUp
// (`ModelTier`, `ModelLoadState`) so consumer apps can template a
// single ModelPickerStore<T> across modules.

/// Stable wire id for an STT backend. Mirrors engine-side
/// `STTBackendID`. Renaming a case is a major SDK bump.
///
/// Yooz Engine exposes four speech backends today:
/// - `.parakeet` — MLX Parakeet TDT (Latin/European languages, high accuracy,
///    ~600 MB runtime)
/// - `.fastConformer` — MLX FastConformer (Arabic, Persian; RTL scripts)
/// - `.appleSTT` — Apple's built-in STT (`SFSpeechRecognizer` on macOS 14-25,
///    `SpeechAnalyzer` on macOS 26+). Zero MLX footprint; the only backend
///    linked into `YoozEngineLite`.
/// - `.qwen3ASRPreview` — preview Qwen3-ASR backend (variant-gated).
///
/// Picker lives server-side; clients consult `STTClient.availableEngines()`
/// to discover which are linked into the running build variant. Requesting an
/// unbundled engine returns HTTP 501 `module_not_bundled`.
public enum STTBackendID: String, Codable, Sendable, CaseIterable {
    case parakeet
    case fastConformer = "fast_conformer"
    case appleSTT = "apple_stt"
    case qwen3ASRPreview = "qwen3_asr_preview"
}

/// One STT backend in the picker. Canonical fields mirror
/// `TouchUpModelInfo`; STT-specific capability flags ride along
/// as optional extensions per AGENTS.md "Module-specific picker
/// extensions". The extensions are `Optional` on the wire so a
/// future engine that drops a capability (e.g. once every backend
/// streams, `supportsStreaming` becomes meaningless) does not
/// brick older SDK consumers.
public struct STTBackendInfo: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let description: String
    public let tier: ModelTier
    public let sizeBytes: Int64?
    public let loadState: ModelLoadState
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

/// Response for `availableEngines()`. `activeId` is the id of the
/// entry where `isActive == true`.
public struct STTBackendsResponse: Codable, Sendable {
    public let backends: [STTBackendInfo]
    public let activeId: String

    public init(backends: [STTBackendInfo], activeId: String) {
        self.backends = backends
        self.activeId = activeId
    }
}

/// Request body for `setEngine(id:preload:)`.
public struct STTSetBackendRequest: Codable, Sendable {
    public let id: String
    public let preload: Bool?

    public init(id: String, preload: Bool? = nil) {
        self.id = id
        self.preload = preload
    }
}
