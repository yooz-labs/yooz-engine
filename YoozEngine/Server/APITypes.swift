import EngineCore
import Hummingbird
#if canImport(InfiniteModule)
import InfiniteModule
#endif
#if canImport(LLMModule)
import LLMModule
#endif

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

// MARK: - Session Types

/// Response for `POST /v1/session/begin` (engine issue #114).
///
/// `sessionId` is a fresh UUID per call. Engine state itself doesn't pin to
/// the value — `begin` is idempotent and unconditionally fans out
/// `resetForNewSession()` to every `SessionResettable` module — but
/// returning a UUID lets consumer apps tag their own logs / metrics so a
/// recording can be correlated end-to-end across engine + client traces.
/// `ts` is an ISO-8601 UTC timestamp captured server-side at fan-out start.
struct SessionBeginResponse: ResponseCodable {
    let sessionId: String
    let ts: String
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
    /// When true (or unset), the engine fetches the model from
    /// Hugging Face if no local snapshot is staged. When false, the
    /// load fails with `model_not_cached` rather than touching the
    /// network. Honored by every backend that owns a first-run fetch
    /// path (Parakeet via the shared HF cache, plus the
    /// `qwen3_asr_preview` URLSession fetcher). Apple STT ignores
    /// the flag — its model is supplied by the OS.
    let allowFetch: Bool?
}

struct LLMStatusResponse: ResponseCodable {
    /// True if the active LLM tier (or any tier) has finished loading.
    let loaded: Bool
    /// Wire id of the preferred LLM model
    /// (e.g. `"yooz-light-v2"`).
    let modelId: String?
    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download for the preferred LLM tier. `nil` when no download
    /// is in flight (idle or already loaded). Mirrors
    /// `STTStatusResponse.progress` shape.
    let progress: Double?
    /// Lifecycle state for the active LLM tier (engine#125). `nil`
    /// on builds that predate the fire-and-forget rollout — consumers
    /// MAY infer state from `loaded` + `progress` when nil.
    let state: LoadState?
    /// Human-readable error message when `state == .failed`. `nil`
    /// in every other state.
    let lastError: String?

    init(
        loaded: Bool,
        modelId: String?,
        progress: Double?,
        state: LoadState? = nil,
        lastError: String? = nil
    ) {
        self.loaded = loaded
        self.modelId = modelId
        self.progress = progress
        self.state = state
        self.lastError = lastError
    }
}

struct STTStatusResponse: ResponseCodable {
    let loaded: Bool
    let language: String?
    let streaming: Bool
    /// Fraction-completed [0.0, 1.0] for an in-progress HF model
    /// download. `nil` when no download is in flight (idle, loaded,
    /// or Apple STT — which has no HF pull). Non-nil only while the
    /// snapshot is actively streaming in. Mirrors
    /// `LLMStatusResponse.progress` shape; same filter applied
    /// server-side by `/v1/stt/status` (engine#145).
    let progress: Double?
    /// Lifecycle state for the active STT backend (engine#125). `nil`
    /// on builds that predate the fire-and-forget rollout — consumers
    /// MAY infer state from `loaded` + `progress` when nil.
    let state: LoadState?
    /// Human-readable error message when `state == .failed`. `nil`
    /// in every other state.
    let lastError: String?

    init(
        loaded: Bool,
        language: String?,
        streaming: Bool,
        progress: Double?,
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

// MARK: - STT Backend Picker (canonical module-picker pattern, second adopter)
//
// Mirrors the TouchUp picker shape from #97 / AGENTS.md "Module
// model picker pattern". Same `id / displayName / description /
// tier / sizeBytes / loadState / isActive` fields; STT-specific
// capability flags (`supportsBatch`, `supportsStreaming`,
// `supportedLanguages`) are added as optional extensions per the
// AGENTS.md "Module-specific picker extensions" guidance.
//
// The legacy `STTEngineGetResponse` / `STTEnginePostRequest`
// shapes are kept available on the wire (POST accepts both `id`
// and the legacy `engine` field) so an in-flight whisper build
// against a one-week-old SDK still works through one release.

/// One STT backend in the picker. Canonical fields match
/// `TouchUpModelInfo`; STT-specific fields are optional so the
/// generic `ModelPickerStore<T>` template still works.
struct STTBackendInfo: Codable, Sendable, Equatable, ResponseEncodable {
    /// Stable wire id (e.g. `parakeet`, `fast_conformer`,
    /// `apple_stt`, `qwen3_asr_preview`). Matches
    /// `STTBackendID.rawValue`.
    let id: String
    /// Picker-visible name.
    let displayName: String
    /// One-line subtitle for picker UX.
    let description: String
    /// Coarse tier (`light` / `quality` / `premium` / `unknown`).
    /// MLX backends report `.quality`; Apple STT reports `.premium`
    /// (OS-provided); preview backends report `.unknown` so the UI
    /// can render a "preview" hint without inventing a new tier.
    let tier: ModelTier
    /// Approximate first-run download size. `nil` for backends
    /// that do not require a download (Apple, bundled MLX).
    let sizeBytes: Int64?
    /// Lifecycle state. Same total ordering as TouchUp picker.
    let loadState: ModelLoadState
    /// Whether `/v1/stt/batch` + `/v1/stt/stream` currently route
    /// through this backend. Exactly one row per response has
    /// `isActive == true`.
    let isActive: Bool
    // MARK: - STT-specific extensions (optional per AGENTS.md
    // "Module-specific picker extensions"). Optional on the wire
    // so a future engine that drops a capability (e.g. once every
    // backend streams, `supportsStreaming` becomes meaningless)
    // does not brick older SDK consumers.
    let supportsBatch: Bool?
    let supportsStreaming: Bool?
    let supportedLanguages: [String]?
}

/// Response for `GET /v1/stt/engine` (canonical shape).
struct STTBackendsResponse: Codable, Sendable, ResponseCodable {
    let backends: [STTBackendInfo]
    let activeId: String
}

/// Request body for `POST /v1/stt/engine` (canonical shape).
/// The legacy `engine` field is accepted as a fallback for
/// pre-#99 clients; the route handler reads `id` first, then
/// `engine` if `id` is absent.
struct STTSetBackendRequest: Codable {
    let id: String?
    let preload: Bool?
    /// Legacy field accepted as a fallback. Old SDK clients post
    /// `{ "engine": "parakeet" }`; new clients post
    /// `{ "id": "parakeet", "preload": true }`. Removed once
    /// every shipped SDK is on the new shape.
    let engine: String?
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

// MARK: - TouchUp Picker (canonical module-picker pattern)
//
// Wire shape used by `GET /v1/touchup/models` and
// `POST /v1/touchup/model`. The same shape — `models`, `activeId`,
// `id / displayName / description / tier / sizeBytes / loadState /
// isActive` — is the documented canon for every future module
// picker (STT engine selection, TTS voice, etc.) so SDK + UI code
// can be templated. See AGENTS.md "Module model picker pattern".
//
// Visibility note: the picker types now live in LLMModule
// (`YoozEngine/TouchUp/TouchUpPickerTypes.swift`) so the producer
// (TouchUpEngine) can construct them without depending on Hummingbird.
// `ModelTier` and `ModelLoadState` live in EngineCore (shared with
// STTModule). The Hummingbird `ResponseEncodable` / `ResponseCodable`
// conformances are added as extensions in this file because the wire
// concern belongs here, not in LLMModule.
extension TouchUpModelInfo: ResponseEncodable {}
extension TouchUpModelsResponse: ResponseEncodable {}

// MARK: - Infinite Picker (engine-owned long-context module)

#if canImport(InfiniteModule)
extension InfiniteModelInfo: ResponseEncodable {}
extension InfiniteModelsResponse: ResponseEncodable {}
extension InfiniteStatus: ResponseEncodable {}
#endif

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
