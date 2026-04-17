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

// MARK: - Engine-picker wire types

/// Identifies which STT backend the engine should route to.
///
/// Yooz Engine exposes three speech backends:
/// - `.parakeet` — MLX Parakeet TDT (Latin/European languages, high accuracy,
///    ~600 MB runtime)
/// - `.fastConformer` — MLX FastConformer (Arabic, Persian; RTL scripts)
/// - `.appleSTT` — Apple's built-in STT (`SFSpeechRecognizer` on macOS 14-25,
///    `SpeechAnalyzer` on macOS 26+). Zero MLX footprint; the only backend
///    linked into `YoozEngineLite`.
///
/// Picker lives server-side; clients consult `STTClient.availableEngineTypes()`
/// to discover which are linked into the running build variant. Requesting an
/// unbundled engine returns HTTP 501 `module_not_bundled`.
public enum STTEngineType: String, Codable, Sendable, CaseIterable {
    case parakeet
    case fastConformer = "fast_conformer"
    case appleSTT = "apple_stt"
}

/// Response body for `GET /v1/stt/engine`.
///
/// Advertises the currently active backend, every backend linked into the
/// build variant, and capability flags clients need for dispatch (e.g.
/// `hasBuiltInVAD` lets a client skip its own VAD pipeline when Apple STT
/// is selected).
public struct STTEngineResponse: Codable, Sendable, Equatable {
    public let current: STTEngineType
    public let available: [STTEngineType]
    public let hasBuiltInVAD: Bool

    public init(current: STTEngineType, available: [STTEngineType], hasBuiltInVAD: Bool) {
        self.current = current
        self.available = available
        self.hasBuiltInVAD = hasBuiltInVAD
    }
}

/// Request body for `POST /v1/stt/engine`.
///
/// Switching cancels any in-flight WebSocket stream for the session but does
/// not reset the chosen language — the new engine will load the same language
/// on its next transcribe call.
public struct STTEngineSwitchRequest: Codable, Sendable, Equatable {
    public let engine: STTEngineType

    public init(engine: STTEngineType) {
        self.engine = engine
    }
}
