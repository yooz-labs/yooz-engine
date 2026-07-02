import EngineCore
import Hummingbird
#if canImport(InfiniteModule)
import InfiniteModule
#endif
#if canImport(LLMModule)
import LLMModule
#endif
// Explicit import (redundant with `EngineCore`'s re-export for every other
// wire type) needed only to spell the qualified `YoozEngineWire.LLMModelInfo`
// below: `LLMModule` has its own internal `LLMModelInfo` domain type
// (`TouchUp/TouchUpEngine.swift`, fields `type` / `isLoaded` / `isCached`),
// so the bare name is ambiguous wherever both modules are imported together.
import YoozEngineWire

// Every request/response DTO that also crosses the SDK or the in-process
// transport now lives once in `YoozEngineWire` (#225), imported here
// transitively via `EngineCore`'s re-export
// (`Sources/EngineCore/WireReexport.swift`). This file holds:
//   - Hummingbird `ResponseEncodable`/`ResponseCodable` conformances for
//     those shared types (a wire-transport concern, not a `YoozEngineWire`
//     concern — that target stays dependency-free, so it can't import
//     Hummingbird itself).
//   - Types that are genuinely server-only: the `/v1/health` envelope, the
//     structured error envelope, the STT WebSocket message frames, and the
//     legacy `POST /v1/stt/engine` decode shim.

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
///
/// Not moved to `YoozEngineWire` (#225): `detail`'s value type
/// (`ModuleDetail`) lives in the app target's `ModuleEagerLoader.swift`,
/// which is eager-load diagnostics machinery, not a plain DTO — pulling it
/// into the dependency-free wire target would need moving that machinery
/// too, which is out of scope for the picker / models-status-modules /
/// STT-LLM-TouchUp-Grammar-VAD-bodies families #225 covers. The SDK's
/// `HealthStatus`/`ModuleStatus` (a strict subset — no `detail`) stays a
/// separate, intentionally different shape; it decodes this response fine
/// since Codable ignores the extra key.
struct EngineModules: Codable {
    let stt: Bool
    let llm: Bool
    let touchup: Bool
    let grammar: Bool
    let vad: Bool
    let tts: Bool
    /// True once the Infinite long-context module is bundled and loaded.
    /// Present on every variant; `false` where InfiniteModule isn't bundled
    /// (Lite/Whisper) or no model is loaded yet.
    let infinite: Bool
    let detail: ModuleDetailMap
}

struct ErrorResponse: ResponseCodable {
    let error: String
    let code: String
}

// MARK: - Wire-shared response conformances (#225)
//
// `ResponseEncodable`/`ResponseCodable` extensions for `YoozEngineWire`
// types this server returns directly from a route handler (bypassing the
// `jsonResponse(...)` helper) or that carried the conformance historically;
// added uniformly for parity. `YoozEngineWire` itself stays Hummingbird-free.

extension ManagedModelsResponse: ResponseEncodable {}
extension DeleteModelResult: ResponseEncodable {}
extension ModelCleanupResult: ResponseEncodable {}
extension SessionBeginResponse: ResponseEncodable {}
extension TranscriptionResult: ResponseEncodable {}
extension STTLanguagesResponse: ResponseEncodable {}
extension LLMStatus: ResponseEncodable {}
extension STTStatus: ResponseEncodable {}
extension STTBackendInfo: ResponseEncodable {}
extension STTBackendsResponse: ResponseEncodable {}
extension LLMGenerateResponse: ResponseEncodable {}
extension YoozEngineWire.LLMModelInfo: ResponseEncodable {}
extension LLMModelsResponse: ResponseEncodable {}
extension TouchUpResponse: ResponseEncodable {}
extension TouchUpModelInfo: ResponseEncodable {}
extension TouchUpModelsResponse: ResponseEncodable {}
extension GrammarCheckResponse: ResponseEncodable {}
extension VADResponse: ResponseEncodable {}

// MARK: - Legacy STT engine-selection decode shim
//
// `POST /v1/stt/engine` decode-only concern: pre-#99 clients post
// `{"engine": "parakeet"}`; current clients post `{"id": "parakeet",
// "preload": true}`. The canonical shared `STTSetBackendRequest`
// (`YoozEngineWire`) only has the modern shape — this shim is what the
// route handler actually decodes the raw body into, then resolves `id ??
// engine` itself. Kept local (not promoted to `YoozEngineWire`) because no
// other transport needs the legacy fallback: the SDK only ever encodes the
// modern shape, and the in-process transport has no equivalent legacy
// callers to support. `Codable` (not just `Decodable`) so tests can encode
// a legacy-shaped payload directly rather than hand-building JSON strings.
struct LegacySTTSetBackendRequest: Codable {
    let id: String?
    let preload: Bool?
    let engine: String?
}

/// `POST /v1/llm/generate` decode-only concern: some pre-SDK callers post
/// the snake_case `system_prompt` spelling. The canonical shared
/// `LLMGenerateRequest` (`YoozEngineWire`) carries only the camelCase key —
/// this shim is what the route handler actually decodes the raw body into,
/// accepting both spellings. Same pattern (and same rationale) as
/// `LegacySTTSetBackendRequest` above: legacy tolerance stays a
/// loopback-server concern, keeping the shared DTO free of
/// transport-specific compat baggage. `workloadClass` (engine#228) is
/// camelCase only — new with #228, so no second spelling is grandfathered
/// in; an unknown value fails the typed decode → 400 `invalid_request` (a
/// declared scheduling class is explicit caller intent; a silent downgrade
/// would hide a client-side typo or version skew).
struct LegacyLLMGenerateRequest: Decodable {
    let prompt: String
    let model: String?
    let systemPrompt: String?
    let workloadClass: MLXWorkloadClass?

    private enum CodingKeys: String, CodingKey {
        case prompt, model, systemPrompt, system_prompt, workloadClass
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.prompt = try container.decode(String.self, forKey: .prompt)
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
        self.systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt)
            ?? container.decodeIfPresent(String.self, forKey: .system_prompt)
        self.workloadClass = try container.decodeIfPresent(
            MLXWorkloadClass.self, forKey: .workloadClass
        )
    }
}

// MARK: - WebSocket STT Messages
//
// Not moved to `YoozEngineWire` (#225): these are the `/v1/stt/stream`
// WebSocket frame shapes, a loopback-only wire protocol the in-process
// transport has no socket to serve (see `InProcessTransport`'s
// `openSTTStream`). Out of scope for the REST DTO families #225 names.

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

// MARK: - Infinite Picker (engine-owned long-context module)
//
// Not moved to `YoozEngineWire` (#225): Infinite is loopback-only by design
// (its only consumer is the super-yooz host; `YoozEngineInProcess` doesn't
// even depend on `InfiniteModule`, per `RouteParityAllowlist` in
// `Sources/EngineCore/RouteManifest.swift`)
// and isn't one of the families the issue names. `InfiniteModelInfo` etc.
// already live once, in `InfiniteModule` — this is only the Hummingbird
// conformance, same pattern as the TouchUp picker below.

#if canImport(InfiniteModule)
extension InfiniteModelInfo: ResponseEncodable {}
extension InfiniteModelsResponse: ResponseEncodable {}
extension InfiniteStatus: ResponseEncodable {}
extension InfiniteSessionInfo: ResponseEncodable {}
extension InfiniteSessionsResponse: ResponseEncodable {}
extension InfiniteAppendSessionResponse: ResponseEncodable {}
extension InfiniteGenerateSessionResponse: ResponseEncodable {}
extension InfiniteCheckpointSessionResponse: ResponseEncodable {}
extension InfiniteDeleteSessionResponse: ResponseEncodable {}
#endif
