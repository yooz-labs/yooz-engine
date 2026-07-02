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
/// ## Scope
///
/// Implemented (the full standalone-app surface): `GET /v1/health`,
/// `GET /v1/modules`; STT `GET /v1/stt/{status,languages,engine}`,
/// `POST /v1/stt/{batch,engine,load}`, and streaming via `openSTTStream`;
/// `POST /v1/grammar/check`; `POST /v1/vad/detect`; LLM
/// `GET /v1/llm/{status,models}`, `POST /v1/llm/{generate,model,preload,unload}`;
/// TouchUp `GET /v1/touchup/models`, `POST /v1/touchup{,/model}`;
/// `POST /v1/session/{begin,end}` — the per-recording session-reset boundary
/// (engine issue #114 / #222), fanned out via the shared
/// `EngineCore.SessionCoordinator` so this transport and the loopback server
/// behave identically.
///
/// Reported as `unsupportedOperation`:
///   - **Streaming qwen3 preview** — loopback/dev only (unstable; engine#154).
///   - **Infinite** (`/v1/infinite/*`) — its consumer is the loopback host.
///
/// `RouteParityAllowlist.loopbackOnly` (EngineCore, `RouteManifest.swift`) is
/// the reviewable, tested source of truth for this list — this comment is a
/// human-readable summary of it, not a second authority; update the allowlist
/// first and let this comment follow (#223).
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
        case "/v1/models":
            return try await handleModelsInventory()
        default:
            throw YoozEngineError.unsupportedOperation(operation: "GET \(route(path))")
        }
    }

    // Handler contract (#223): every POST handler dispatched below MUST
    // validate its request (JSON-decode the body, or an equivalent cheap
    // guard) BEFORE doing any disk/network/model work. RouteParityTests
    // drives each route with a minimal invalid request and relies on that
    // fail-fast gate to prove dispatch reachability without side effects.
    // Handlers with no request body to gate on (`/v1/models/cleanup`,
    // `/v1/session/*`) DO run their real work on every test sweep, so they
    // must stay cheap and side-effect-safe in a bare test process — the test
    // additionally redirects the HF cache env vars so cleanup can never
    // touch the machine's real model cache.
    public func post(_ path: String, body: Data) async throws -> Data {
        try await connect()
        switch route(path) {
        case "/v1/grammar/check":
            return try await handleGrammar(body)
        case "/v1/vad/detect":
            return try await handleVAD(body)
        case "/v1/stt/batch":
            return try await handleBatch(body)
        case "/v1/stt/load":
            return try await handleSTTLoad(body, wait: Self.parseWaitQuery(path))
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
        case "/v1/models/cleanup":
            return try await handleModelsCleanup()
        case "/v1/session/begin":
            return try await handleSessionBegin()
        case "/v1/session/end":
            return try await handleSessionEnd()
        default:
            throw YoozEngineError.unsupportedOperation(operation: "POST \(route(path))")
        }
    }

    public func delete(_ path: String) async throws -> Data {
        try await connect()
        let routed = route(path)
        let prefix = "/v1/models/"
        if routed.hasPrefix(prefix) {
            let raw = String(routed.dropFirst(prefix.count))
            let id = raw.removingPercentEncoding ?? raw
            return try await handleDeleteModel(id)
        }
        throw YoozEngineError.unsupportedOperation(operation: "DELETE \(routed)")
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
            // Bound the (possibly first-run) load so a stream open can't hang
            // indefinitely; routes through the same cancellable `enqueueLoad`
            // primitive the load endpoint uses.
            let task = await YoozSTTEngine.shared.enqueueLoad(language: lang) {
                try await YoozSTTEngine.shared.start(language: lang)
            }
            try await awaitLoadTask(
                task, deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
            )
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

    /// Parse the `?wait=...` flag a load caller may append. Mirrors the loopback
    /// `APIServer.parseWaitQuery`: `loadModel` posts `/v1/stt/load?wait=true`
    /// (blocking), `loadModelAsync` posts `/v1/stt/load` (fire-and-forget). The
    /// in-process `route()` strips the query for matching, so the original path
    /// is parsed here to recover the flag.
    static func parseWaitQuery(_ path: String) -> Bool {
        guard let q = path.firstIndex(of: "?") else { return false }
        let query = path[path.index(after: q)...]
        for pair in query.split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1)
            guard kv.first == "wait" else { continue }
            if kv.count == 1 { return true }
            return kv[1] == "true" || kv[1] == "1"
        }
        return false
    }

    // MARK: - Handlers

    private func handleHealth() async throws -> Data {
        let grammarReady = GrammarEngine.shared.isAvailable
        let llmReady = await TouchUpEngine.shared.isPreloaded
        let vadReady = await VADEngine.shared.isLoaded
        let sttReady = YoozSTTEngine.shared.isRunning

        let status = HealthStatus(
            status: "ok",
            version: EngineConfig.version,
            modules: ModuleStatus(
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
        var manifests: [ModuleManifest] = []
        for module in modules {
            let health = await module.healthCheck()
            manifests.append(
                ModuleManifest(
                    name: type(of: module).name,
                    version: EngineConfig.version,
                    loaded: health.loaded,
                    error: health.error,
                    detail: health.detail
                )
            )
        }
        manifests.sort { $0.name < $1.name }

        let response = ModulesResponse(
            engineVersion: EngineConfig.version,
            buildVariant: BuildVariant.current.rawValue,
            modules: manifests
        )
        return try JSONEncoder().encode(response)
    }

    // MARK: - Session boundary

    /// `POST /v1/session/begin` (engine issue #114 / #222). Fans out
    /// `resetForNewSession()` to every registered `SessionResettable` module
    /// via the shared `EngineCore.SessionCoordinator` — the same component
    /// the loopback `APIServer` route calls — so the response carries the
    /// same wire fields (`{sessionId, ts}`) regardless of transport.
    private func handleSessionBegin() async throws -> Data {
        let result = await SessionCoordinator.begin()
        NSLog(
            "InProcessTransport: session begin id=%@ fanout=%d",
            result.sessionId, result.fanoutCount
        )
        return try JSONEncoder().encode(
            SessionBeginResponse(sessionId: result.sessionId, ts: result.ts)
        )
    }

    /// `POST /v1/session/end` (engine issue #114 / #222). Same fan-out as
    /// `begin`; the loopback route returns 204 No Content, so this returns an
    /// empty body — `EngineTransport.post` has no separate "no content"
    /// signal, and an empty `Data` is what `HTTPTransport` also produces for
    /// a 204 response.
    private func handleSessionEnd() async throws -> Data {
        let fanoutCount = await SessionCoordinator.end()
        NSLog("InProcessTransport: session end fanout=%d", fanoutCount)
        return Data()
    }

    private func handleGrammar(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(GrammarCheckRequest.self, from: body)
        let outcome = await GrammarEngine.shared.check(
            text: request.text,
            categories: request.categories,
            usePOS: request.usePOS ?? true
        )
        let response = GrammarCheckResponse(
            result: outcome.result,
            correctionsApplied: outcome.correctionsApplied,
            ruleCount: GrammarEngine.shared.ruleCount
        )
        return try JSONEncoder().encode(response)
    }

    private func handleVAD(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(VADRequest.self, from: body)
        if await !VADEngine.shared.isLoaded {
            try await VADEngine.shared.load()
        }
        let segments = try await VADEngine.shared.detect(
            samples: request.samples,
            resetState: request.reset ?? true
        )
        let response = VADResponse(
            segments: segments.map {
                SpeechSegment(
                    startMs: $0.startMs,
                    endMs: $0.endMs,
                    probability: $0.probability
                )
            }
        )
        return try JSONEncoder().encode(response)
    }

    private func handleBatch(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(BatchSTTRequest.self, from: body)
        // A missing `language`/`mode` key decodes as `"en"`/`"normal"` — the
        // defaults live on the canonical `BatchSTTRequest` itself, so this
        // handler and the loopback route agree by construction (#225 review).
        // An unknown language string is still a hard 400; an unknown mode
        // string coerces to `.normal` for parity with the loopback server,
        // which also coerces rather than 400s.
        guard let language = STTModule.STTLanguage.fromCode(request.language) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_language",
                message: "Unknown STT language '\(request.language)'"
            )
        }
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

    /// Pre-load the active backend's STT model for a language and return the
    /// resulting status. Backs both `loadModel` (`/v1/stt/load?wait=true`) and
    /// `loadModelAsync` (`/v1/stt/load`, fire-and-forget).
    ///
    /// For the MLX backends the load is routed through the engine's cancellable
    /// `enqueueLoad` state machine — the same primitive the loopback
    /// `/v1/stt/load` route uses — so the load is observable via `loadState`
    /// and bounded. `wait == true` awaits completion under
    /// `EngineConfig.modelLoadDeadlineSeconds`; `wait == false` returns the
    /// current (`.loading`) status immediately and the consumer polls
    /// `/v1/stt/status` for the `.ready` / `.failed` transition. This replaces
    /// the prior synchronous, unbounded `start()` that pinned status at
    /// "Downloading 100%" until weights finished materializing.
    ///
    /// The same lazy `start()` also runs on the first `batch`/stream call, so
    /// this endpoint is a pre-warm, not a prerequisite, for transcription.
    private func handleSTTLoad(_ body: Data, wait: Bool) async throws -> Data {
        let request = try JSONDecoder().decode(STTLoadRequest.self, from: body)
        // A missing `language` key decodes as `"en"` — the default lives on
        // the canonical `STTLoadRequest` itself, so this handler and the
        // loopback route agree by construction (#225 review). An unknown
        // language string is still a hard 400.
        guard let language = STTModule.STTLanguage.fromCode(request.language) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_language",
                message: "Unknown STT language '\(request.language)'"
            )
        }
        // Honored by the MLX branch below; parity with the loopback route's
        // `runSTTLoad(language:allowFetch:)`. Apple STT ignores the flag —
        // its model is supplied by the OS.
        let allowFetch = request.allowFetch ?? true
        switch YoozSTTEngine.shared.currentBackend {
        case .appleSTT:
            guard let appleLang = AppleSTTLanguage.from(rawCode: request.language) else {
                throw YoozEngineError.serverError(
                    statusCode: 400, code: "invalid_language",
                    message: "Language '\(request.language)' is not supported by Apple STT"
                )
            }
            // Apple STT has no HF download/materialize phase; load synchronously.
            try await AppleSTTEngine.shared.start(language: appleLang)
        case .qwen3ASRPreview:
            // The preview backend is loopback/dev only (unstable; engine#154).
            throw YoozEngineError.unsupportedOperation(operation: "load qwen3 preview")
        case .parakeet, .fastConformer:
            let task = await YoozSTTEngine.shared.enqueueLoad(language: language) {
                try await YoozSTTEngine.shared.start(
                    language: language, allowFetch: allowFetch
                )
            }
            if wait {
                try await awaitLoadTask(
                    task, deadlineSeconds: EngineConfig.modelLoadDeadlineSeconds
                )
            } else {
                // Fire-and-forget: the task settles loadState/lastLoadError in the
                // background and a failure surfaces on the next /v1/stt/status
                // poll. Log at the dispatch site so that failure is correlatable
                // here, not only via the NSLog deep inside start().
                NSLog(
                    "InProcessTransport: STT load dispatched fire-and-forget for %@; poll /v1/stt/status for completion",
                    language.rawValue
                )
            }
        }
        return try await handleSTTStatus()
    }

    private func handleLLM(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMGenerateRequest.self, from: body)
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
            modelType: modelType,
            workloadClass: request.workloadClass ?? .background
        )
        let response = LLMGenerateResponse(
            text: text,
            model: modelType.rawValue,
            tokensGenerated: nil,
            processingTimeMs: nil
        )
        return try JSONEncoder().encode(response)
    }

    // MARK: - Status

    private func handleSTTStatus() async throws -> Data {
        let status: STTStatus
        if YoozSTTEngine.shared.currentBackend == .appleSTT {
            let loaded = await AppleSTTEngine.shared.isLoaded
            let language = await AppleSTTEngine.shared.currentLanguage.rawValue
            let streaming = await AppleSTTEngine.shared.isStreaming
            // Apple STT has no fetcher/materialize lifecycle; map loaded → ready.
            status = STTStatus(
                loaded: loaded, language: language, streaming: streaming,
                progress: nil, state: loaded ? .ready : .idle, lastError: nil
            )
        } else {
            // Surface the load lifecycle + last error (mirrors the loopback
            // /v1/stt/status, engine#125) so the consumer can distinguish a
            // download ("Downloading X%") from the synchronous materialization
            // window ("Loading model…": progress nil + state .loading) and from a
            // failed load (state .failed). The in-process path constructs the SDK
            // type directly, so the engine LoadState is bridged by rawValue.
            let engine = YoozSTTEngine.shared
            let loaded = engine.isRunning
            // Read the @Published, MainActor-written fields in a single hop so a
            // poll can't observe a torn snapshot (e.g. the stale ~1.0 progress
            // mid-reset alongside an already-advanced state).
            let (rawProgress, engineState, engineError) = await MainActor.run {
                (engine.downloadProgress, engine.loadState, engine.lastLoadError)
            }
            // When the model is resident, force progress nil + state .ready: a
            // just-finished load leaves downloadProgress at 1.0, and a redundant
            // enqueueLoad on an already-loaded engine briefly flips loadState to
            // .loading — neither should surface as "Downloading 100%" / "loading"
            // for a ready model. While not loaded, a fraction of exactly 1.0 means
            // "download done, materializing" -> nil (the STT analog of the LLM
            // `< 1` filter).
            let resolvedState: EngineCore.LoadState = loaded ? .ready : engineState
            status = STTStatus(
                loaded: loaded,
                language: engine.currentLanguage.rawValue,
                streaming: engine.isStreaming,
                progress: loaded ? nil : (rawProgress > 0 && rawProgress < 1 ? rawProgress : nil),
                state: LoadState(rawValue: resolvedState.rawValue),
                lastError: loaded ? nil : engineError
            )
        }
        return try JSONEncoder().encode(status)
    }

    private func handleLLMStatus() async throws -> Data {
        let engine = TouchUpEngine.shared
        let active = await engine.activeModel
        let loaded: Bool
        // Mirror the loopback `/v1/llm/status` progress contract so the
        // consumer-side touch-up download banner works in-process too (#214):
        // nil when the active tier is already loaded (no download to track),
        // nil for Apple Intelligence (OS-provided, no HF fetch), otherwise the
        // live fraction (>0) the `loadModelContainer` callback last reported.
        // Was hardcoded `nil`, so the in-process Light/Quality download bar
        // never moved on a first switch to a not-yet-cached tier.
        // A fraction of 1.0 means the download finished and the model is now
        // materializing (loadModelContainer hasn't returned yet): report nil so
        // the consumer renders "Loading model…" (state .loading) rather than a
        // frozen "Downloading 100%" — the LLM analog of the STT progress reset.
        let progress: Double?
        // engine#125: also surface the per-tier lifecycle state + last error so
        // consumers distinguish downloading / loading / failed. Bridged to the
        // SDK enum by rawValue (the in-process path builds the SDK type directly).
        let engineState: EngineCore.LoadState
        let lastError: String?
        switch active {
        case .yoozLight:
            loaded = await engine.isLightModelLoaded
            if loaded {
                progress = nil
            } else {
                let fraction = await engine.downloadProgress(for: .yoozLight) ?? 0
                progress = (fraction > 0 && fraction < 1) ? fraction : nil
            }
            engineState = await engine.loadState(for: .yoozLight)
            lastError = await engine.lastLoadError(for: .yoozLight)
        case .yoozQuality:
            loaded = await engine.isQualityModelLoaded
            if loaded {
                progress = nil
            } else {
                let fraction = await engine.downloadProgress(for: .yoozQuality) ?? 0
                progress = (fraction > 0 && fraction < 1) ? fraction : nil
            }
            engineState = await engine.loadState(for: .yoozQuality)
            lastError = await engine.lastLoadError(for: .yoozQuality)
        case .foundationModels:
            loaded = await engine.isFoundationModelsLoaded
            progress = nil
            engineState = loaded ? .ready : .idle
            lastError = nil
        }
        let status = LLMStatus(
            loaded: loaded, modelId: active.rawValue, progress: progress,
            state: LoadState(rawValue: engineState.rawValue),
            lastError: lastError
        )
        return try JSONEncoder().encode(status)
    }

    // MARK: - STT picker

    private func handleSTTLanguages() async throws -> Data {
        let infos = YoozSTTEngine.shared.availableLanguages.map {
            STTLanguageInfo(
                code: $0.rawValue,
                name: $0.displayName,
                implemented: $0.isImplemented,
                family: $0.modelFamily.rawValue
            )
        }
        return try JSONEncoder().encode(STTLanguagesResponse(languages: infos))
    }

    private func handleSTTEngine() async throws -> Data {
        let active = YoozSTTEngine.shared.currentBackend
        let activeLoaded = await YoozSTTEngine.shared.isCurrentBackendLoaded()
        let backends = STTModule.STTBackendID.allCases.map {
            sttBackendInfo($0, active: active, activeLoaded: activeLoaded)
        }
        return try JSONEncoder().encode(
            STTBackendsResponse(backends: backends, activeId: active.rawValue)
        )
    }

    private func handleSetSTTEngine(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(STTSetBackendRequest.self, from: body)
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
    ) -> STTBackendInfo {
        let isActive = backend == active
        return STTBackendInfo(
            id: backend.rawValue,
            displayName: backend.displayName,
            description: backend.pickerDescription,
            tier: ModelTier(rawValue: backend.pickerTier.rawValue) ?? .unknown,
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
            LLMModelsResponse(current: active.rawValue, available: available)
        )
    }

    private func handleSetLLMModel(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMModelSelection.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        await TouchUpEngine.shared.setPreferredModel(modelType)
        return try await handleLLMModels()
    }

    private func handleLLMPreload(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMModelSelection.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        try await TouchUpEngine.shared.preloadModel(modelType)
        return try JSONEncoder().encode(await llmModelInfo(modelType))
    }

    private func handleLLMUnload(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(LLMModelSelection.self, from: body)
        let modelType = try resolveLLMModel(request.model)
        await TouchUpEngine.shared.unload(modelType)
        return try JSONEncoder().encode(await llmModelInfo(modelType))
    }

    /// Resolve the optional GPU-admission class from its raw wire string
    /// (engine#228) for the `TouchUpBody` decode shim below. Nil/omitted means
    /// today's default (`.background`); an unrecognized value is a hard 400 —
    /// parity with the loopback server (whose typed `Decodable` enum rejects
    /// unknown values as `invalid_request`) and with this transport's own mode
    /// handling ("an unknown mode is a hard error"): a declared scheduling
    /// class is explicit caller intent, and silently downgrading a mistyped
    /// `.interactive` to `.background` would make the request queue behind
    /// other work with no trace of why. (`handleLLM` needs no equivalent: it
    /// decodes the canonical `LLMGenerateRequest`, whose typed
    /// `workloadClass` field rejects unknown values at decode.)
    private static func resolveWorkloadClass(
        _ raw: String?
    ) throws -> MLXWorkloadClass {
        guard let raw, !raw.isEmpty else { return .background }
        guard let resolved = MLXWorkloadClass(rawValue: raw) else {
            throw YoozEngineError.serverError(
                statusCode: 400,
                code: "invalid_request",
                message: "Unknown workloadClass '\(raw)'"
            )
        }
        return resolved
    }

    private func handleTouchUp(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(TouchUpBody.self, from: body)
        // The requested mode is explicit caller intent (off/light/standard/full),
        // not a forward-compat wire value — an unknown mode is a hard error, never
        // a silent run at a different cleanup level.
        guard let mode = TouchUpMode(rawValue: request.mode) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_mode",
                message: "Unknown TouchUp mode '\(request.mode)'"
            )
        }
        // Route through the active-model picker (not the legacy `process()`),
        // so the user's selection is honored in-process: Apple Intelligence
        // (FoundationModels), Yooz Light, or Yooz Quality. `process()` ignored
        // the selection (always MLX-light) and never lazy-loaded it, so
        // in-process cleanup silently passed text through. Each backend now
        // lazy-loads on first use (mirrors the STT lazy-load).
        let result = await TouchUpEngine.shared.processWithActiveModel(
            text: request.text,
            mode: mode,
            workloadClass: try Self.resolveWorkloadClass(request.workloadClass)
        )
        let response = TouchUpResponse(
            result: result.text,
            mode: mode,
            processingTimeMs: Int(result.latencyMs),
            modelUsed: result.modelUsed.rawValue,
            warnings: result.fallbackReason.map { [$0] }
        )
        return try JSONEncoder().encode(response)
    }

    private func handleTouchUpModels() async throws -> Data {
        let models = await TouchUpEngine.shared.availableModels()
        let active = await TouchUpEngine.shared.activeModel
        let activeId = models.first(where: \.isActive)?.id ?? active.rawValue
        return try JSONEncoder().encode(
            TouchUpModelsResponse(models: models, activeId: activeId)
        )
    }

    private func handleSetTouchUpModel(_ body: Data) async throws -> Data {
        let request = try JSONDecoder().decode(TouchUpSetModelRequest.self, from: body)
        guard let selection = TouchUpModelSelection(rawValue: request.id) else {
            throw YoozEngineError.serverError(
                statusCode: 400, code: "invalid_model",
                message: "Unknown TouchUp model '\(request.id)'"
            )
        }
        let info = try await TouchUpEngine.shared.setActiveModel(
            selection, preload: request.preload ?? true
        )
        return try JSONEncoder().encode(info)
    }

    // MARK: - Model management (disk hygiene)

    /// `GET /v1/models` — the cross-module inventory with real on-disk sizes.
    /// Mirrors the loopback `APIServer` `/v1/models` handler so the SDK is
    /// transport-agnostic.
    private func handleModelsInventory() async throws -> Data {
        let store = ModelStore()
        let rows = await store.inventory(
            llm: await llmInventoryInputs(),
            activeSTTRepoDirName: activeSTTRepoDirName()
        )
        let models = rows.map {
            ManagedModelInfo(
                id: $0.id, module: $0.module, displayName: $0.displayName,
                sizeBytes: $0.sizeBytes, cached: $0.cached, loaded: $0.loaded,
                isActive: $0.isActive, deletable: $0.deletable
            )
        }
        return try JSONEncoder().encode(ManagedModelsResponse(models: models))
    }

    /// `DELETE /v1/models/:id` — unload then remove a model's reclaimable copies.
    /// Refuses to delete the active model (409).
    private func handleDeleteModel(_ id: String) async throws -> Data {
        let store = ModelStore()

        if let modelType = LLMModelType(rawValue: id) {
            let active = await TouchUpEngine.shared.activeModel
            if active.rawValue == id {
                throw YoozEngineError.serverError(
                    statusCode: 409, code: "model_active",
                    message: "Cannot delete the active model '\(id)'"
                )
            }
            let descriptor = LLMModelCatalog.cacheDescriptors().first { $0.id == id }
            do {
                // Disk first; free resident weights only on success so a failed
                // delete leaves the model usable rather than unloaded-but-present.
                let reclaimed = try await store.deleteModel(
                    hfRepoDirName: descriptor?.hfRepoDirName,
                    modelsDirSubdir: descriptor?.modelsDirSubdir
                )
                await TouchUpEngine.shared.unload(modelType)
                return try JSONEncoder().encode(DeleteModelResult(id: id, reclaimedBytes: reclaimed))
            } catch {
                NSLog("InProcessTransport: DELETE model '%@' failed: %@", id, error.localizedDescription)
                throw error
            }
        }

        if id.hasPrefix("models--") {
            if id == activeSTTRepoDirName() {
                throw YoozEngineError.serverError(
                    statusCode: 409, code: "model_active",
                    message: "Cannot delete the active model '\(id)'"
                )
            }
            do {
                let reclaimed = try await store.deleteModel(hfRepoDirName: id, modelsDirSubdir: nil)
                return try JSONEncoder().encode(DeleteModelResult(id: id, reclaimedBytes: reclaimed))
            } catch {
                NSLog("InProcessTransport: DELETE model '%@' failed: %@", id, error.localizedDescription)
                throw error
            }
        }

        throw YoozEngineError.serverError(
            statusCode: 404, code: "unknown_model",
            message: "Unknown model '\(id)'"
        )
    }

    /// `POST /v1/models/cleanup` — the one-shot disk-hygiene migration.
    private func handleModelsCleanup() async throws -> Data {
        let store = ModelStore()
        do {
            let report = try await store.cleanupAll(
                descriptors: LLMModelCatalog.cacheDescriptors()
            )
            return try JSONEncoder().encode(ModelCleanupResult(
                totalReclaimedBytes: report.totalReclaimedBytes,
                perRepo: report.perRepo
            ))
        } catch {
            NSLog("InProcessTransport: model cleanup failed: %@", error.localizedDescription)
            throw error
        }
    }

    /// LLM rows for the inventory: cache descriptors + live picker state.
    private func llmInventoryInputs() async -> [ModelStore.LLMInventoryInput] {
        let picker = await TouchUpEngine.shared.availableModels()
        return LLMModelCatalog.cacheDescriptors().map { descriptor in
            let row = picker.first { $0.id == descriptor.id }
            return ModelStore.LLMInventoryInput(
                descriptor: descriptor,
                displayName: row?.displayName ?? descriptor.id,
                loaded: row?.loadState == .loaded,
                isActive: row?.isActive ?? false
            )
        }
    }

    /// Hub dir name of the active STT model, or `nil` for Apple Speech (no HF
    /// footprint). Parakeet TDT is the only HF-backed STT family.
    private func activeSTTRepoDirName() -> String? {
        let engine = YoozSTTEngine.shared
        guard engine.currentBackend != .appleSTT,
              let hfID = engine.currentLanguage.huggingFaceID
        else { return nil }
        return ModelCacheDescriptor.hubRepoDirName(forHuggingFaceID: hfID)
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

}

// MARK: - TouchUp mode decode shim
//
// `TouchUpBody` stays a local, minimal decode struct rather than the
// canonical `TouchUpRequest` (#225): `TouchUpRequest.mode` is the strict
// `TouchUpMode` enum, whose decode failure on an unrecognized string would
// surface as an opaque `DecodingError` instead of the `invalid_mode`
// `YoozEngineError.serverError` this handler has always returned. Decoding
// the raw string here and resolving it against `TouchUpMode(rawValue:)`
// explicitly preserves that error contract byte-for-byte. Every other
// former mirror struct in this section decoded a shape with no such
// custom-error behavior riding on it, so those were deleted outright in
// favor of the shared `YoozEngineWire` request types.

private struct TouchUpBody: Decodable {
    let text: String
    let mode: String
    let language: String?
    /// Raw wire value of `EngineCore.MLXWorkloadClass` (engine#228). Kept as
    /// a plain `String?` at the decode layer; `resolveWorkloadClass` maps
    /// nil/empty to `.background` and rejects unknown values with a 400.
    let workloadClass: String?
}
