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
/// Reported as `unsupportedOperation` until a later cut:
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
        case "/v1/stt/status":
            return try await handleSTTStatus()
        case "/v1/stt/languages":
            return try await handleSTTLanguages()
        case "/v1/stt/engine":
            return try await handleSTTEngine()
        case "/v1/llm/status":
            return try await handleLLMStatus()
        case "/v1/llm/models":
            return try await handleLLMModels()
        case "/v1/touchup/models":
            return try await handleTouchUpModels()
        default:
            throw YoozEngineError.unsupportedOperation(operation: "GET \(route(path))")
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
        case "/v1/stt/engine":
            return try await handleSetSTTEngine(body)
        case "/v1/llm/generate":
            return try await handleLLM(body)
        case "/v1/llm/model":
            return try await handleSetLLMModel(body)
        case "/v1/llm/preload":
            return try await handleLLMPreload(body)
        case "/v1/llm/unload":
            return try await handleLLMUnload(body)
        case "/v1/touchup":
            return try await handleTouchUp(body)
        case "/v1/touchup/model":
            return try await handleSetTouchUpModel(body)
        default:
            throw YoozEngineError.unsupportedOperation(operation: "POST \(route(path))")
        }
    }

    public func delete(_ path: String) async throws -> Data {
        throw YoozEngineError.unsupportedOperation(operation: "DELETE \(route(path))")
    }

    @available(macOS 14.0, iOS 17.0, *)
    public func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        try await connect()
        guard let lang = STTModule.STTLanguage.fromCode(language) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_language",
                message: "Unknown STT language '\(language)'"
            )
        }
        // Unknown mode falls back to .normal for parity with the loopback WS
        // handler (which also coerces). normal/whispered are the only cases and
        // share rawValues across the SDK and engine enums.
        let audioMode = STTModule.AudioMode(rawValue: mode) ?? .normal

        switch YoozSTTEngine.shared.currentBackend {
        case .appleSTT:
            guard let appleLang = AppleSTTLanguage.from(rawCode: language) else {
                throw YoozEngineError.serverError(
                    statusCode: 400, code: "invalid_language",
                    message: "Language '\(language)' is not supported by Apple STT"
                )
            }
            try await AppleSTTEngine.shared.start(language: appleLang)
            return InProcessSTTStreamSession(backend: .apple(AppleSTTEngine.shared))

        case .qwen3ASRPreview:
            // The preview backend is loopback/dev only (unstable; engine#154).
            throw YoozEngineError.unsupportedOperation(operation: "streaming qwen3 preview")

        case .parakeet, .fastConformer:
            try await YoozSTTEngine.shared.start(language: lang)
            guard let transcriber = YoozSTTEngine.shared.createBatchTranscriber(mode: audioMode) else {
                throw YoozEngineError.serverError(
                    statusCode: 503,
                    code: "stt_not_loaded",
                    message: "STT model failed to load for language '\(language)'"
                )
            }
            return InProcessSTTStreamSession(backend: .parakeet(transcriber))
        }
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

    // MARK: - Status

    private func handleSTTStatus() async throws -> Data {
        let status: SDKSTTStatus
        if YoozSTTEngine.shared.currentBackend == .appleSTT {
            let loaded = await AppleSTTEngine.shared.isLoaded
            let language = await AppleSTTEngine.shared.currentLanguage.rawValue
            let streaming = await AppleSTTEngine.shared.isStreaming
            status = SDKSTTStatus(
                loaded: loaded, language: language, streaming: streaming,
                progress: nil, state: nil, lastError: nil
            )
        } else {
            let progress = YoozSTTEngine.shared.downloadProgress
            status = SDKSTTStatus(
                loaded: YoozSTTEngine.shared.isRunning,
                language: YoozSTTEngine.shared.currentLanguage.rawValue,
                streaming: YoozSTTEngine.shared.isStreaming,
                progress: progress > 0 ? progress : nil,
                state: nil,
                lastError: nil
            )
        }
        return try JSONEncoder().encode(status)
    }

    private func handleLLMStatus() async throws -> Data {
        let active = await TouchUpEngine.shared.activeModel
        let loaded: Bool
        switch active {
        case .yoozLight:
            loaded = await TouchUpEngine.shared.isLightModelLoaded
        case .yoozQuality:
            loaded = await TouchUpEngine.shared.isQualityModelLoaded
        case .foundationModels:
            loaded = await TouchUpEngine.shared.isFoundationModelsLoaded
        }
        let status = SDKLLMStatus(
            loaded: loaded, modelId: active.rawValue, progress: nil,
            state: nil, lastError: nil
        )
        return try JSONEncoder().encode(status)
    }

    // MARK: - STT picker

    private func handleSTTLanguages() async throws -> Data {
        let infos = YoozSTTEngine.shared.availableLanguages.map {
            SDKSTTLanguageInfo(
                code: $0.rawValue,
                name: $0.displayName,
                implemented: $0.isImplemented,
                family: $0.modelFamily.rawValue
            )
        }
        return try JSONEncoder().encode(SDKSTTLanguagesResponse(languages: infos))
    }

    private func handleSTTEngine() async throws -> Data {
        let active = YoozSTTEngine.shared.currentBackend
        let activeLoaded = await YoozSTTEngine.shared.isCurrentBackendLoaded()
        let backends = STTModule.STTBackendID.allCases.map {
            sttBackendInfo($0, active: active, activeLoaded: activeLoaded)
        }
        return try JSONEncoder().encode(
            SDKSTTBackendsResponse(backends: backends, activeId: active.rawValue)
        )
    }

    private func handleSetSTTEngine(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(SetBackendBody.self, from: body)
        guard let backend = STTModule.STTBackendID(rawValue: request.id) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_backend",
                message: "Unknown STT backend '\(request.id)'"
            )
        }
        await YoozSTTEngine.shared.setBackend(backend)
        let loaded = await YoozSTTEngine.shared.isCurrentBackendLoaded()
        let info = sttBackendInfo(backend, active: backend, activeLoaded: loaded)
        return try JSONEncoder().encode(info)
    }

    private func sttBackendInfo(
        _ backend: STTModule.STTBackendID,
        active: STTModule.STTBackendID,
        activeLoaded: Bool
    ) -> SDKSTTBackendInfo {
        let isActive = backend == active
        return SDKSTTBackendInfo(
            id: backend.rawValue,
            displayName: backend.displayName,
            description: backend.pickerDescription,
            tier: SDKModelTier(rawValue: backend.pickerTier.rawValue) ?? .unknown,
            sizeBytes: backend.estimatedDownloadMB.map { Int64($0) * 1_000_000 },
            loadState: (isActive && activeLoaded) ? .loaded : .available,
            isActive: isActive,
            supportsBatch: backend.supportsBatch,
            supportsStreaming: backend.supportsStreaming,
            supportedLanguages: backend.supportedLanguages.map(\.rawValue)
        )
    }

    // MARK: - LLM / TouchUp model management

    private func handleLLMModels() async throws -> Data {
        let models = await TouchUpEngine.shared.availableModels()
        let active = await TouchUpEngine.shared.activeModel
        // `LLMModelInfo.loaded` is a Bool, so the engine's four-state lifecycle
        // collapses here: `.cached` (on disk, not resident) reports `loaded:false`.
        // This matches the SDK type; the full lifecycle is on the TouchUp picker
        // (`/v1/touchup/models` -> `TouchUpModelInfo.loadState`).
        let available = models.map {
            SDKLLMModelInfo(
                id: $0.id, displayName: $0.displayName,
                sizeBytes: $0.sizeBytes, loaded: $0.loadState == .loaded,
                latencyHintMs: nil
            )
        }
        return try JSONEncoder().encode(
            SDKLLMModelsResponse(current: active.rawValue, available: available)
        )
    }

    private func handleSetLLMModel(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(ModelSelectionBody.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        await TouchUpEngine.shared.setPreferredModel(modelType)
        return try await handleLLMModels()
    }

    private func handleLLMPreload(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(ModelSelectionBody.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        try await TouchUpEngine.shared.preloadModel(modelType)
        return try JSONEncoder().encode(await llmModelInfo(modelType))
    }

    private func handleLLMUnload(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(ModelSelectionBody.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        await TouchUpEngine.shared.unload(modelType)
        return try JSONEncoder().encode(await llmModelInfo(modelType))
    }

    private func handleTouchUp(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(TouchUpBody.self, from: body)
        // The requested mode is explicit caller intent (off/light/standard/full),
        // not a forward-compat wire value — an unknown mode is a hard error, never
        // a silent run at a different cleanup level. (Both enums share rawValues.)
        guard let engineMode = LLMModule.TouchUpMode(rawValue: request.mode),
              let sdkMode = SDKTouchUpMode(rawValue: request.mode)
        else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_mode",
                message: "Unknown TouchUp mode '\(request.mode)'"
            )
        }
        let result = await TouchUpEngine.shared.process(
            text: request.text, mode: engineMode, replacements: []
        )
        let response = SDKTouchUpResponse(
            result: result.text,
            mode: sdkMode,
            processingTimeMs: Int(result.latencyMs),
            modelUsed: result.modelUsed.rawValue,
            warnings: result.fallbackReason.map { [$0] }
        )
        return try JSONEncoder().encode(response)
    }

    private func handleTouchUpModels() async throws -> Data {
        let models = await TouchUpEngine.shared.availableModels()
        let active = await TouchUpEngine.shared.activeModel
        let mapped = models.map(touchUpModelInfo)
        let activeId = mapped.first(where: \.isActive)?.id ?? active.rawValue
        return try JSONEncoder().encode(
            SDKTouchUpModelsResponse(models: mapped, activeId: activeId)
        )
    }

    private func handleSetTouchUpModel(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(SetModelBody.self, from: body)
        guard let selection = TouchUpModelSelection(rawValue: request.id) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_model",
                message: "Unknown TouchUp model '\(request.id)'"
            )
        }
        let info = try await TouchUpEngine.shared.setActiveModel(
            selection, preload: request.preload ?? true
        )
        return try JSONEncoder().encode(touchUpModelInfo(info))
    }

    // MARK: - Mapping helpers

    private func resolveLLMModel(_ id: String) throws -> LLMModelType {
        guard let modelType = LLMModelType(rawValue: id) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_model",
                message: "Unknown LLM model '\(id)'"
            )
        }
        return modelType
    }

    /// Real post-operation model info for the preload/unload responses: re-query
    /// `availableModels()` so `displayName` / `sizeBytes` / `loaded` reflect the
    /// actual state rather than a fabricated row.
    private func llmModelInfo(_ modelType: LLMModelType) async -> SDKLLMModelInfo {
        let models = await TouchUpEngine.shared.availableModels()
        if let match = models.first(where: { $0.id == modelType.rawValue }) {
            return SDKLLMModelInfo(
                id: match.id, displayName: match.displayName,
                sizeBytes: match.sizeBytes, loaded: match.loadState == .loaded,
                latencyHintMs: nil
            )
        }
        return SDKLLMModelInfo(
            id: modelType.rawValue, displayName: modelType.rawValue,
            sizeBytes: nil, loaded: false, latencyHintMs: nil
        )
    }

    private func touchUpModelInfo(_ info: LLMModule.TouchUpModelInfo) -> SDKTouchUpModelInfo {
        SDKTouchUpModelInfo(
            id: info.id,
            displayName: info.displayName,
            description: info.description,
            tier: SDKModelTier(rawValue: info.tier.rawValue) ?? .unknown,
            sizeBytes: info.sizeBytes,
            loadState: SDKModelLoadState(rawValue: info.loadState.rawValue) ?? .unavailable,
            isActive: info.isActive
        )
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

private struct SetBackendBody: Decodable {
    let id: String
    let preload: Bool?
}

private struct ModelSelectionBody: Decodable {
    let model: String
}

private struct TouchUpBody: Decodable {
    let text: String
    let mode: String
    let language: String?
}

private struct SetModelBody: Decodable {
    let id: String
    let preload: Bool?
}
