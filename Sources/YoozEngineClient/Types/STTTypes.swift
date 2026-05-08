import Foundation

/// A single token from an aligned transcription with start/end
/// timestamps. Used by callers that need word- or sub-word-level
/// timing (chunk-boundary deduplication, subtitle rendering,
/// hallucination filters). Timestamps are seconds from the start
/// of the submitted audio buffer.
///
/// Only populated on responses from aligned transcription paths
/// (`STTClient.batchTranscribeAligned`). Restored on the SDK in
/// #99 after the simplification dropped it.
public struct AlignedToken: Codable, Sendable, Equatable {
    public let text: String
    public let start: Float
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
    /// Aligned tokens with timestamps. `nil` when the request did
    /// not opt into alignment or when the backend for the active
    /// engine did not return alignment information. Present on
    /// every response from `STTClient.batchTranscribeAligned`.
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
// (`TouchUpModelTier`, `TouchUpModelLoadState`) so consumer apps
// can template a single ModelPickerStore<T> across modules.

/// Stable wire id for an STT backend. Mirrors engine-side
/// `STTBackendID`. Renaming a case is a major SDK bump.
public enum STTBackendID: String, Codable, Sendable, CaseIterable {
    case parakeet
    case fastConformer = "fast_conformer"
    case appleSTT = "apple_stt"
    case qwen3ASRPreview = "qwen3_asr_preview"
}

/// One STT backend in the picker. Canonical fields mirror
/// `TouchUpModelInfo`; STT-specific capability flags ride along
/// as optional extensions per AGENTS.md "Module-specific picker
/// extensions".
public struct STTBackendInfo: Codable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let description: String
    public let tier: TouchUpModelTier
    public let sizeBytes: Int64?
    public let loadState: TouchUpModelLoadState
    public let isActive: Bool
    public let supportsBatch: Bool
    public let supportsStreaming: Bool
    public let supportedLanguages: [String]

    public init(
        id: String,
        displayName: String,
        description: String,
        tier: TouchUpModelTier,
        sizeBytes: Int64?,
        loadState: TouchUpModelLoadState,
        isActive: Bool,
        supportsBatch: Bool,
        supportsStreaming: Bool,
        supportedLanguages: [String]
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
