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
    /// (`qwen3_asr_preview`); ignored otherwise.
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
    /// Stable error code so clients can branch without parsing
    /// `message`. Optional to keep the wire format backward-
    /// compatible with older consumers.
    let code: String?

    init(type: String, message: String, code: String? = nil) {
        self.type = type
        self.message = message
        self.code = code
    }
}

struct WSSTTReady: Encodable {
    let type: String  // "ready"
    let language: String
}

/// Stable wire-level warning codes the engine emits over WS. The
/// `rawValue` is what crosses the wire; clients branch on it. Kept
/// as a typed enum so the compiler catches typos at the engine
/// emit-site (a free-form `String` like
/// `"buffer_cap_reachd"` would silently break client branching).
enum WSSTTWarningCode: String, Encodable, Sendable, CaseIterable {
    /// The streaming session's audio buffer hit its soft cap;
    /// additional audio is being discarded. The transcript on
    /// `final` reflects the buffered audio only.
    case bufferCapReached = "buffer_cap_reached"
}

/// One-shot non-fatal warning frame. Used when the engine wants to
/// keep the stream alive but signal a soft-capacity event (e.g. the
/// session buffer cap was reached and additional audio is being
/// dropped).
struct WSSTTWarning: Encodable {
    let type: String  // "warning"
    let code: WSSTTWarningCode
    let message: String
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
    /// Per-request KV cache compression override. When `nil` (or the field
    /// is omitted), the engine-wide default (`EngineConfig.kvCompression`,
    /// currently `.off`) is used. Wire-format strings are `"off"` and
    /// `"turbo3"`; any other value fails decode. Both `kvCompression` and
    /// `kv_compression` keys are accepted on the wire. See
    /// `KVCompressionMode` in `EngineConfig.swift`.
    let kvCompression: KVCompressionMode?

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: LLMGenerateRequestKey.self)
        self.prompt = try container.decode(String.self, forKey: LLMGenerateRequestKey("prompt"))
        self.model = try container.decodeIfPresent(String.self, forKey: LLMGenerateRequestKey("model"))
        self.systemPrompt = try container.decodeIfPresent(String.self, forKey: LLMGenerateRequestKey("systemPrompt"))
            ?? container.decodeIfPresent(String.self, forKey: LLMGenerateRequestKey("system_prompt"))
        self.kvCompression = try container.decodeIfPresent(KVCompressionMode.self, forKey: LLMGenerateRequestKey("kvCompression"))
            ?? container.decodeIfPresent(KVCompressionMode.self, forKey: LLMGenerateRequestKey("kv_compression"))
    }
}

/// Coding key shim for accepting both camelCase and snake_case wire keys
/// without committing to a global strategy on the Hummingbird decoder.
/// Used by `LLMGenerateServerRequest` to backward-compatibly accept
/// `systemPrompt` / `system_prompt` and `kvCompression` / `kv_compression`.
private struct LLMGenerateRequestKey: CodingKey {
    var stringValue: String
    init(_ stringValue: String) { self.stringValue = stringValue }
    init?(stringValue: String) { self.stringValue = stringValue }
    var intValue: Int? { nil }
    init?(intValue: Int) { return nil }
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
