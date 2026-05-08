import Hummingbird

struct HealthResponse: ResponseCodable {
    let status: String
    let version: String
    let modules: EngineModules
}

/// Per-module readiness reported by `/v1/health`.
///
/// The legacy `Bool` fields stay for SDK back-compat: a module reads
/// `true` once it transitions to `.ready`, `false` otherwise. The
/// new `detail` map carries the richer state (`loading` / `error` /
/// `unavailable`) so clients can render a spinner or a neutral tag
/// instead of a red dot. See `ModuleReadiness` for the wire-level
/// rawValues clients should branch on.
struct EngineModules: Codable {
    let stt: Bool
    let llm: Bool
    let touchup: Bool
    let grammar: Bool
    let vad: Bool
    let tts: Bool
    let detail: ModuleDetailMap
}

/// Response for `GET /v1/modules`. Reports the active build variant
/// plus the same per-module readiness map exposed under
/// `/v1/health.modules.detail`. Useful for clients that want a
/// purpose-built endpoint for their "Engine status" UI without
/// having to filter the health-check fields.
struct ModulesResponseV1: ResponseCodable {
    let variant: String
    let version: String
    let modules: ModuleDetailMap
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
    /// When true (or unset), the engine fetches the model from
    /// Hugging Face if no local snapshot is staged. When false, the
    /// load fails with `model_not_cached` rather than touching the
    /// network. Honored by every backend that owns a first-run fetch
    /// path (Parakeet via the shared HF cache, plus the
    /// `qwen3_asr_preview` URLSession fetcher). Apple STT ignores
    /// the flag — its model is supplied by the OS.
    let allowFetch: Bool?
}

struct STTStatusResponse: ResponseCodable {
    let loaded: Bool
    let language: String?
    let streaming: Bool
    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. Reset to 0 at the start of every `/v1/stt/load`
    /// call; ticks up to 1.0 as files stream in. Optional in the
    /// wire shape so older clients continue to decode the response.
    let progress: Double?
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

/// Stable wire-level error codes the engine emits over WS. The
/// `rawValue` is what crosses the wire; clients branch on it. Kept
/// as a typed enum (mirroring `WSSTTWarningCode`) so the compiler
/// catches typos at the emit-site — a free-form `String` like
/// `"session_error"` would silently break client branching.
enum WSSTTErrorCode: String, Encodable, Sendable, CaseIterable {
    /// Inbound text frame failed JSON decode (`WSSTTConfig`).
    case invalidMessageFormat = "invalid_message_format"
    /// `config.language` does not map to a known `STTLanguage`.
    case unknownLanguage = "unknown_language"
    /// Language is known but not implemented for any backend yet.
    case languageNotImplemented = "language_not_implemented"
    /// The active backend (`qwen3_asr_preview`) does not support
    /// the requested language.
    case languageNotSupportedByBackend = "language_not_supported_by_backend"
    /// `sttEngine.start(language:)` threw — the per-backend load
    /// path failed (model fetch, weight load, tokenizer prep).
    case modelLoadFailed = "model_load_failed"
    /// Inbound binary frame is not a whole-`Float32` multiple.
    case invalidAudioFrame = "invalid_audio_frame"
    /// Qwen3 streaming session raised a typed `SessionError`
    /// during `push`.
    case sessionError = "session_error"
    /// The WS message loop tore down via an exception path
    /// (oversized frame, abrupt disconnect, framer decode).
    case streamAborted = "stream_aborted"
    /// `finalize()` threw — the model produced no transcript or
    /// the post-processing path failed mid-stream.
    case finalizeFailed = "finalize_failed"
}

struct WSSTTError: Encodable {
    let type: String  // "error"
    let message: String
    /// Typed error code so clients can branch without parsing
    /// `message`. Optional to keep the wire format backward-
    /// compatible with older consumers; the wire still carries the
    /// snake_case rawValue.
    let code: WSSTTErrorCode?

    init(type: String, message: String, code: WSSTTErrorCode? = nil) {
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

// MARK: - TouchUp Picker (canonical module-picker pattern)
//
// Wire shape used by `GET /v1/touchup/models` and
// `POST /v1/touchup/model`. The same shape (`models`, `activeId`,
// `id/displayName/description/tier/sizeBytes/isAvailable/
// isCached/isLoaded/isActive`) is the documented canon for every
// future module picker (STT engine selection, TTS voice, etc.) so
// SDK + UI code can be templated. See AGENTS.md "Module model
// picker pattern" for the recipe.

/// One model in the TouchUp picker. All fields are server-authoritative
/// — the client treats this as a snapshot and re-fetches after a
/// `setModel(...)` to learn the new active id and any cache/load
/// changes the preload triggered.
public struct TouchUpModelInfo: Codable, Sendable, Equatable, ResponseEncodable {
    /// Stable wire id (e.g. `yooz-light-v3`). Matches
    /// `TouchUpModelSelection.rawValue`.
    public let id: String
    /// Picker-visible name (e.g. "Yooz-Light").
    public let displayName: String
    /// One-line subtitle for picker UX (latency hint etc.).
    public let description: String
    /// Coarse tier label (`light` / `quality` / `premium`). UI
    /// renders Pro badges or sort hints off this.
    public let tier: String
    /// Approximate on-disk size after first-run download. `nil` for
    /// OS-provided backends (Apple Intelligence).
    public let sizeBytes: Int64?
    /// Whether this option is selectable on this system. False for
    /// Apple Intelligence on pre-26 macOS or non-opted-in users.
    public let isAvailable: Bool
    /// Whether the weights are already on disk (no download needed).
    public let isCached: Bool
    /// Whether the model is currently resident in memory.
    public let isLoaded: Bool
    /// Whether `/v1/touchup` currently routes through this model.
    public let isActive: Bool

    public init(
        id: String,
        displayName: String,
        description: String,
        tier: String,
        sizeBytes: Int64?,
        isAvailable: Bool,
        isCached: Bool,
        isLoaded: Bool,
        isActive: Bool
    ) {
        self.id = id
        self.displayName = displayName
        self.description = description
        self.tier = tier
        self.sizeBytes = sizeBytes
        self.isAvailable = isAvailable
        self.isCached = isCached
        self.isLoaded = isLoaded
        self.isActive = isActive
    }
}

/// Response for `GET /v1/touchup/models`. `activeId` is the id of
/// the entry where `isActive == true` — surfaced separately so a
/// client that only cares about the current selection does not
/// have to scan the array.
public struct TouchUpModelsResponse: Codable, Sendable, ResponseCodable {
    public let models: [TouchUpModelInfo]
    public let activeId: String

    public init(models: [TouchUpModelInfo], activeId: String) {
        self.models = models
        self.activeId = activeId
    }
}

/// Request body for `POST /v1/touchup/model`. `preload` defaults
/// to `true` server-side so a one-shot picker change is enough to
/// avoid a cold-start on the next `/v1/touchup` call.
struct TouchUpSetModelRequest: Decodable {
    let id: String
    let preload: Bool?
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
