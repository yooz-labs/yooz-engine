import Hummingbird
#if canImport(LLMModule)
import LLMModule
#endif

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
    /// Request per-token timestamps in the response.
    ///
    /// When `true`, the handler routes to each backend's alignment-aware
    /// entry point (`YoozSTTEngine.batchTranscribeAligned` for MLX,
    /// `AppleSTTEngine.batchTranscribeAligned` for Apple STT) and returns
    /// `BatchSTTResponse.tokens`. Absent / `false` keeps today's
    /// token-less behaviour byte-identical with v0.5.x clients.
    let aligned: Bool?
}

struct BatchSTTResponse: ResponseCodable {
    let text: String
    let finalized: String
    let draft: String
    let language: String
    /// Non-nil iff the request set `aligned = true`. Timestamps are in
    /// seconds from the start of the submitted audio buffer. `encodeIfPresent`
    /// keeps the field off the wire for non-aligned responses so v0.5.x
    /// clients see byte-identical traffic.
    let tokens: [AlignedTokenWire]?

    init(
        text: String,
        finalized: String,
        draft: String,
        language: String,
        tokens: [AlignedTokenWire]? = nil
    ) {
        self.text = text
        self.finalized = finalized
        self.draft = draft
        self.language = language
        self.tokens = tokens
    }

    enum CodingKeys: String, CodingKey {
        case text, finalized, draft, language, tokens
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(text, forKey: .text)
        try container.encode(finalized, forKey: .finalized)
        try container.encode(draft, forKey: .draft)
        try container.encode(language, forKey: .language)
        try container.encodeIfPresent(tokens, forKey: .tokens)
    }
}

/// Wire-level aligned-token shape for `/v1/stt/batch` with `aligned=true`.
///
/// Mirrors `YoozEngineClient.AlignedToken` exactly (text + start + end as
/// seconds). Engine-side `AlignedToken` types from STTModule
/// (`start + duration`) and AppleSTTModule (derived from
/// `SFTranscriptionSegment`) are both mapped into this shape at the route
/// boundary so the SDK surface stays backend-agnostic.
struct AlignedTokenWire: Codable, Sendable {
    let text: String
    let start: Float
    let end: Float
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

    #if canImport(LLMModule)
    /// Map the wire enum to the LLMModule domain enum. The two enums are
    /// intentionally distinct so the module stays free of server DTOs;
    /// mapping lives here, at the wire boundary.
    var asDomain: TouchUpMode {
        switch self {
        case .off: return .off
        case .light: return .light
        case .standard: return .standard
        case .full: return .full
        }
    }
    #endif
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

// MARK: - LLM Model Management Types

/// Wire body for `GET /v1/llm/models` response. Field set mirrors the
/// SDK's `LLMModelInfo` exactly so JSON round-trips through the thin
/// client without custom CodingKeys.
struct LLMModelInfoServer: ResponseCodable {
    let id: String
    let displayName: String
    let sizeBytes: Int64?
    let loaded: Bool
    let latencyHintMs: Int?
}

struct LLMModelsServerResponse: ResponseCodable {
    let current: String
    let available: [LLMModelInfoServer]
}

/// Shared body for `POST /v1/llm/model`, `POST /v1/llm/preload`, and
/// `POST /v1/llm/unload`. Single field so new options can be added
/// without breaking existing clients.
struct LLMModelSelectionRequest: Decodable {
    let model: String
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
