import Hummingbird

struct HealthResponse: ResponseCodable {
    let status: String
    let version: String
    let modules: EngineModules
}

struct EngineModules: Codable {
    let stt: Bool
    let llm: Bool
    let touchup: Bool
    let grammar: Bool
    let vad: Bool
    let tts: Bool
}

struct ModelsResponse: ResponseCodable {
    let models: [ModelInfo]
}

struct ModelInfo: Codable {
    let name: String
    let module: String
    let loaded: Bool
    let sizeBytes: Int64?
}

struct ErrorResponse: ResponseCodable {
    let error: String
    let code: String
}

// MARK: - STT Types

struct BatchSTTRequest: Decodable {
    let samples: [Float]
    let language: String?
    let mode: String?
}

struct BatchSTTResponse: ResponseCodable {
    let text: String
    let finalized: String
    let draft: String
    let language: String
}

struct STTLanguagesResponse: ResponseCodable {
    let languages: [STTLanguageInfo]
}

struct STTLanguageInfo: Codable {
    let code: String
    let name: String
    let implemented: Bool
    let family: String
}

struct STTLoadRequest: Decodable {
    let language: String?
    /// When true (or unset), the engine will fetch the model from the
    /// remote source if it is not already on disk. When false, the
    /// load fails with `model_not_found` if the directory is empty.
    /// Only consulted by backends that own a first-run fetch path
    /// (`qwen3_asr_preview` in Phase 5); ignored otherwise.
    let allowFetch: Bool?
}

struct STTStatusResponse: ResponseCodable {
    let loaded: Bool
    let language: String?
    let streaming: Bool
}

// MARK: - Backend selection

struct STTEngineGetResponse: ResponseCodable {
    let current: String
    let available: [STTEngineCapabilities]
}

struct STTEngineCapabilities: Codable {
    let id: String
    let supportsBatch: Bool
    let supportsStreaming: Bool
    let supportedLanguages: [String]
}

struct STTEnginePostRequest: Codable {
    /// New backend identifier. Accepts the `STTBackendID` raw values:
    /// `parakeet`, `fast_conformer`, `apple_stt`, `qwen3_asr_preview`.
    let engine: String
}

struct STTEnginePostResponse: ResponseCodable {
    let current: String
}

// MARK: - WebSocket STT Messages

struct WSSTTConfig: Decodable {
    let type: String  // "config"
    let language: String?
    let mode: String?
}

struct WSSTTResult: Encodable {
    let type: String  // "partial" or "final"
    let text: String
    let finalized: String
    let draft: String
}

struct WSSTTError: Encodable {
    let type: String  // "error"
    let message: String
}

struct WSSTTReady: Encodable {
    let type: String  // "ready"
    let language: String
}

// MARK: - LLM Types

/// Server-side touch-up mode (mirrors client's TouchUpMode)
enum ServerTouchUpMode: String, Codable, Sendable {
    case off
    case light
    case standard
    case full
}

struct LLMGenerateServerRequest: Decodable {
    let prompt: String
    let model: String?
    let systemPrompt: String?
}

struct LLMGenerateServerResponse: ResponseCodable {
    let text: String
    let model: String
    let tokensGenerated: Int?
    let processingTimeMs: Int?
}

// MARK: - TouchUp Types

struct TouchUpServerRequest: Decodable {
    let text: String
    let mode: ServerTouchUpMode
    let language: String?
}

struct TouchUpServerResponse: ResponseCodable {
    let result: String
    let mode: ServerTouchUpMode
    let processingTimeMs: Int?
    let modelUsed: String?
    let warnings: [String]?
}

// MARK: - Grammar Types

struct GrammarCheckServerRequest: Decodable {
    let text: String
    let categories: [String]?
    /// Use NLTagger POS tagging for more accurate correction. Defaults to true.
    let usePOS: Bool?
}

struct GrammarCheckServerResponse: ResponseCodable {
    let result: String
    let correctionsApplied: Int
    let ruleCount: Int?
}

// MARK: - VAD Types

struct VADDetectServerRequest: Decodable {
    let samples: [Float]

    /// When true, resets the RNN hidden/cell state before detection.
    /// Defaults to true. Set to false when sending consecutive chunks from
    /// the same recording to preserve inter-frame state continuity.
    let reset: Bool?
}

struct VADDetectServerResponse: ResponseCodable {
    let segments: [VADSegment]
}

struct VADSegment: Codable {
    let startMs: Int
    let endMs: Int
    let probability: Float
}
