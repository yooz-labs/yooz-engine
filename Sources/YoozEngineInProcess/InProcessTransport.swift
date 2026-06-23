import AppleSTTModule
import EngineCore
import Foundation
import GrammarModule
import LLMModule
import STTModule
import VADModule
import YoozEngineClient

/// `EngineTransport` that serves the `YoozEngineClient` SDK surface by calling
/// the engine module actors directly — no loopback socket (epic #192 Phase 2).
///
/// A standalone App Store app links `YoozEngineInProcess`, builds its client as
/// `YoozEngineClient(transport: InProcessTransport())`, and gets the identical
/// SDK API in-sandbox.
///
/// ## Scope (Phase 2a)
///
/// Implemented (non-streaming app surface): `GET /v1/health`, `GET /v1/modules`,
/// `POST /v1/grammar/check`, `POST /v1/vad/detect`, `POST /v1/stt/batch`,
/// `POST /v1/llm/generate`.
///
/// Reported as `unsupportedInProcess` until a later cut:
///   - **Streaming STT** (`webSocketURL`) — Phase 2b.
///   - **Pickers / status / load** (`/v1/stt/engine`, `/v1/stt/status`,
///     `/v1/stt/load`, `/v1/stt/languages`, `/v1/llm/*` model management,
///     `/v1/touchup*`) — a Phase 2a follow-up.
///   - **Infinite** (`/v1/infinite/*`) — its consumer is the loopback host.
///
/// Each handler decodes the same wire body the SDK sub-client encoded and emits
/// the same wire shape the sub-client decodes, so the SDK is byte-for-byte
/// agnostic to which transport served it.
public final class InProcessTransport: EngineTransport {
    /// Non-routable placeholder — there is no socket. Present only because
    /// `baseURL` is part of the long-standing public SDK surface.
    public let baseURL = URL(string: "inprocess://engine")!
    public let port = 0

    private let host: EngineInProcessHost

    public init(host: EngineInProcessHost = .shared) {
        self.host = host
    }

    public func connect() async throws {
        await host.bootstrap()
    }

    public func isReachable() async throws -> Bool {
        // Modules are linked in-process; once bootstrapped the engine is always
        // reachable. Bootstrap is idempotent.
        await host.bootstrap()
        return true
    }

    public func get(_ path: String) async throws -> Data {
        try await connect()
        switch route(path) {
        case "/v1/health":
            return try await handleHealth()
        case "/v1/modules":
            return try await handleModules()
        default:
            throw YoozEngineError.unsupportedInProcess(operation: "GET \(route(path))")
        }
    }

    public func post(_ path: String, body: Data) async throws -> Data {
        try await connect()
        switch route(path) {
        case "/v1/grammar/check":
            return try await handleGrammar(body)
        case "/v1/vad/detect":
            return try await handleVAD(body)
        case "/v1/stt/batch":
            return try await handleBatch(body)
        case "/v1/llm/generate":
            return try await handleLLM(body)
        default:
            throw YoozEngineError.unsupportedInProcess(operation: "POST \(route(path))")
        }
    }

    public func delete(_ path: String) async throws -> Data {
        throw YoozEngineError.unsupportedInProcess(operation: "DELETE \(route(path))")
    }

    @available(macOS 14.0, iOS 17.0, *)
    public func webSocketURL(path: String) throws -> URL {
        // In-process streaming STT lands in Phase 2b.
        throw YoozEngineError.unsupportedInProcess(operation: "WS \(path)")
    }

    // MARK: - Routing

    /// Strip a `?query` suffix so routing matches on the path only.
    private func route(_ path: String) -> String {
        if let q = path.firstIndex(of: "?") {
            return String(path[..<q])
        }
        return path
    }

    // MARK: - Handlers

    private func handleHealth() async throws -> Data {
        let grammarReady = GrammarEngine.shared.isAvailable
        let llmReady = await TouchUpEngine.shared.isPreloaded
        let vadReady = await VADEngine.shared.isLoaded
        let sttReady = YoozSTTEngine.shared.isRunning

        let status = SDKHealthStatus(
            status: "ok",
            version: EngineConfig.version,
            modules: SDKModuleStatus(
                stt: sttReady,
                llm: llmReady,
                touchup: llmReady,
                grammar: grammarReady,
                vad: vadReady,
                tts: false,
                infinite: nil
            )
        )
        return try JSONEncoder().encode(status)
    }

    private func handleModules() async throws -> Data {
        let modules = await ModuleRegistry.shared.all()
        var manifests: [SDKModuleManifest] = []
        for module in modules {
            let health = await module.healthCheck()
            manifests.append(
                SDKModuleManifest(
                    name: type(of: module).name,
                    version: EngineConfig.version,
                    loaded: health.loaded,
                    error: health.error,
                    detail: health.detail
                )
            )
        }
        manifests.sort { $0.name < $1.name }

        let response = SDKModulesResponse(
            engineVersion: EngineConfig.version,
            buildVariant: BuildVariant.current.rawValue,
            modules: manifests
        )
        return try JSONEncoder().encode(response)
    }

    private func handleGrammar(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(GrammarBody.self, from: body)
        let outcome = await GrammarEngine.shared.check(
            text: request.text,
            categories: request.categories,
            usePOS: request.usePOS ?? true
        )
        let response = SDKGrammarCheckResponse(
            result: outcome.result,
            correctionsApplied: outcome.correctionsApplied,
            ruleCount: GrammarEngine.shared.ruleCount
        )
        return try JSONEncoder().encode(response)
    }

    private func handleVAD(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(VADBody.self, from: body)
        if await !VADEngine.shared.isLoaded {
            try await VADEngine.shared.load()
        }
        let segments = try await VADEngine.shared.detect(
            samples: request.samples,
            resetState: request.reset ?? true
        )
        let response = SDKVADResponse(
            segments: segments.map {
                SDKSpeechSegment(
                    startMs: $0.startMs,
                    endMs: $0.endMs,
                    probability: $0.probability
                )
            }
        )
        return try JSONEncoder().encode(response)
    }

    private func handleBatch(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(BatchBody.self, from: body)
        guard let language = STTModule.STTLanguage.fromCode(request.language) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_language",
                message: "Unknown STT language '\(request.language)'"
            )
        }
        // Unknown mode falls back to `.normal` for parity with the loopback
        // server (APIServer `/v1/stt/batch`), which also coerces rather than 400s.
        let mode = STTModule.AudioMode(rawValue: request.mode) ?? .normal
        try await YoozSTTEngine.shared.start(language: language)

        // `batchTranscribe` is non-throwing and returns `ParakeetResult.empty`
        // when no model is loaded — indistinguishable from genuine silence. The
        // throwing `start()` above normally guarantees a loaded model, but guard
        // explicitly so a load-state inconsistency surfaces as an error instead
        // of a misleading empty transcript.
        //
        // (When qwen3 becomes selectable in-process, this path must add the
        // qwen3 dispatch the loopback server uses; qwen3 is not reachable
        // in-process in Phase 2a because the backend picker is deferred.)
        guard YoozSTTEngine.shared.isRunning else {
            throw YoozEngineError.serverError(
                statusCode: 503,
                code: "stt_not_loaded",
                message: "STT model failed to load for language '\(request.language)'"
            )
        }

        if request.aligned == true {
            let result = try await YoozSTTEngine.shared.batchTranscribeAligned(
                samples: request.samples,
                mode: mode
            )
            let tokens = result.tokens.map {
                SDKAlignedToken(text: $0.text, start: $0.start, end: $0.end)
            }
            let response = SDKTranscriptionResult(
                text: result.text,
                finalized: result.text,
                draft: "",
                language: request.language,
                tokens: tokens
            )
            return try JSONEncoder().encode(response)
        }

        let result = await YoozSTTEngine.shared.batchTranscribe(
            samples: request.samples,
            mode: mode
        )
        let response = SDKTranscriptionResult(
            text: result.text,
            finalized: result.finalized,
            draft: result.draft,
            language: request.language,
            tokens: nil
        )
        return try JSONEncoder().encode(response)
    }

    private func handleLLM(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMBody.self, from: body)
        // An unrecognized model name is a hard error (parity with the loopback
        // server's `invalid_model` 400) — never a silent downgrade to Light,
        // which would return a response whose `model` field lies about what ran.
        let modelType: LLMModelType
        if let name = request.model, !name.isEmpty {
            guard let resolved = LLMModelType(rawValue: name) else {
                throw YoozEngineError.serverError(
                    statusCode: 400,
                    code: "invalid_model",
                    message: "Unknown LLM model '\(name)'"
                )
            }
            modelType = resolved
        } else {
            modelType = .yoozLight
        }
        let text = try await TouchUpEngine.shared.generate(
            prompt: request.prompt,
            systemPrompt: request.systemPrompt ?? "",
            modelType: modelType
        )
        let response = SDKLLMGenerateResponse(
            text: text,
            model: modelType.rawValue,
            tokensGenerated: nil,
            processingTimeMs: nil
        )
        return try JSONEncoder().encode(response)
    }
}

// MARK: - Wire request mirrors
//
// Decode structs matching the keys each SDK sub-client encodes. Mirrors (rather
// than the SDK request types) because several SDK request structs are internal
// to the SDK module; the field names/keys here are the wire contract.

private struct GrammarBody: Decodable {
    let text: String
    let categories: [String]?
    let usePOS: Bool?
}

private struct VADBody: Decodable {
    let samples: [Float]
    let reset: Bool?
}

private struct BatchBody: Decodable {
    let samples: [Float]
    let language: String
    let mode: String
    let aligned: Bool?
}

private struct LLMBody: Decodable {
    let prompt: String
    let model: String?
    let systemPrompt: String?
}
