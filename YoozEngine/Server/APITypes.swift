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
}

struct STTStatusResponse: ResponseCodable {
    let loaded: Bool
    let language: String?
    let streaming: Bool
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
}

struct VADDetectServerResponse: ResponseCodable {
    let segments: [VADSegment]
}

struct VADSegment: Codable {
    let startMs: Int
    let endMs: Int
    let probability: Float
}
