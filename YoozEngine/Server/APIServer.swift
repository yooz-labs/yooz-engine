import Foundation
import HuggingFace
import Hummingbird
import HummingbirdWebSocket
import Logging
import NIOCore

@MainActor
final class APIServer: ObservableObject {
    enum State: Equatable {
        case stopped
        case starting
        case running
        case stopping
    }

    @Published var state: State = .stopped
    @Published var lastError: String?

    var isRunning: Bool { state == .running }

    private var app: (any ApplicationProtocol)?
    private var serverTask: Task<Void, any Error>?
    let logger = Logger(label: "live.yooz.engine.server")

    func start() async throws {
        guard state == .stopped else { return }
        state = .starting
        lastError = nil

        let httpRouter = buildRouter()
        let wsRouter = buildWebSocketRouter()

        let app = Application(
            router: httpRouter,
            server: .http1WebSocketUpgrade(webSocketRouter: wsRouter),
            configuration: .init(
                address: .hostname(EngineConfig.host, port: EngineConfig.port),
                serverName: "YoozEngine/\(EngineConfig.version)"
            ),
            logger: logger
        )

        self.app = app

        serverTask = Task { [weak self] in
            do {
                try await app.run()
            } catch is CancellationError {
                // Expected during stop()
            } catch {
                await MainActor.run {
                    self?.state = .stopped
                    self?.lastError = error.localizedDescription
                }
                self?.logger.error("Server terminated: \(error)")
            }
        }

        // Verify the server is actually listening before reporting ready
        try await Task.sleep(for: .milliseconds(200))
        let url = URL(string: "http://\(EngineConfig.host):\(EngineConfig.port)/v1/health")!
        do {
            let (_, response) = try await URLSession.shared.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                throw ServerStartError.healthCheckFailed
            }
        } catch is ServerStartError {
            serverTask?.cancel()
            state = .stopped
            throw ServerStartError.healthCheckFailed
        } catch {
            serverTask?.cancel()
            state = .stopped
            throw ServerStartError.failedToBind(error.localizedDescription)
        }

        state = .running
        logger.info("Yooz Engine started on \(EngineConfig.host):\(EngineConfig.port)")

        // Always apply variant gating so the snapshot reflects the
        // active build variant (e.g. VAD `unavailable` on whisper)
        // even when eager-load is disabled in tests. This is cheap
        // (touches the readiness map only); no module loads run.
        await ModuleEagerLoader.shared.markVariantUnavailableModules(
            variant: EngineConfig.variant
        )

        // Kick off the variant-aware module eager-load so modules
        // are primed by the time thin clients hit them. The kickoff
        // itself is non-blocking — it spawns a background TaskGroup
        // and returns immediately, so the server is fully responsive
        // while loads run.
        //
        // Tying this to `start()` (rather than only to the app's
        // `applicationDidFinishLaunching` hook) makes the menu Stop
        // -> Start path symmetric: `stop()` resets the loader,
        // `start()` re-kicks. Without this, a second `start()`
        // would leave every module in the previous run's terminal
        // state while the underlying engines are unloaded.
        //
        // Disabled in test runs (`EngineConfig.eagerLoadOnLaunch`
        // checks for XCTest env vars) so route tests boot fast and
        // don't pay the model-fetch cost.
        if EngineConfig.eagerLoadOnLaunch {
            await ModuleEagerLoader.shared.kickoff(
                variant: EngineConfig.variant
            )
        }
    }

    func stop() async {
        guard state == .running else { return }
        state = .stopping

        // Stop the STT engine to free GPU memory
        YoozSTTEngine.shared.stop()

        // Unload LLM models to free GPU memory
        await TouchUpEngine.shared.unload()

        // Reset VAD hidden/cell state to free memory
        do {
            try await VADEngine.shared.reset()
        } catch {
            logger.error("Failed to reset VAD state: \(error)")
        }

        // Reset the eager loader so a subsequent `start()` (e.g. the
        // user toggled Stop -> Start from the menu bar) re-runs the
        // load policy against the now-unloaded modules. Without this
        // reset, the second start would leave the loader thinking
        // every module is `ready` (its last terminal state) while
        // the underlying modules are actually unloaded — health
        // would lie until the user hit each route.
        await ModuleEagerLoader.shared.reset()

        serverTask?.cancel()
        _ = await serverTask?.result
        serverTask = nil
        app = nil

        state = .stopped
        logger.info("Yooz Engine stopped")
    }

    // MARK: - Helpers

    private nonisolated func errorResponse(
        status: HTTPResponse.Status,
        message: String,
        code: String
    ) -> Response {
        let body = (try? JSONEncoder().encode(ErrorResponse(error: message, code: code))) ?? Data()
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: body))
        )
    }

    private nonisolated func jsonResponse<T: Encodable>(
        _ value: T,
        status: HTTPResponse.Status = .ok
    ) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    /// Map a Qwen3-ASR backend error to the right HTTP error
    /// response. Mirrors the parakeet/fast_conformer error mapping
    /// so clients get a consistent shape across backends.
    ///
    /// `FetchFailure` payloads are demuxed into distinct HTTP codes
    /// so clients can branch on root cause: 401 for auth, 404 for
    /// repo missing, 502 for transient transport, 422 for local
    /// integrity violations (size/checksum mismatch), 500 for
    /// programmer error (`.other`, manifest decode bugs).
    nonisolated func mapQwen3BatchError(
        _ error: Qwen3ASRError
    ) -> Response {
        switch error {
        case .invalidInput(let detail):
            return errorResponse(
                status: .badRequest,
                message: detail,
                code: "invalid_input"
            )
        case .invalidConfig(let detail):
            return errorResponse(
                status: .internalServerError,
                message: "Qwen3 config invalid: \(detail)",
                code: "model_config_invalid"
            )
        case .pipelineNotLoaded:
            return errorResponse(
                status: .serviceUnavailable,
                message:
                    "Qwen3 pipeline is not loaded; call /v1/stt/load "
                    + "before /v1/stt/batch.",
                code: "pipeline_not_loaded"
            )
        case .fileNotFound, .noAudioTowerWeights:
            return errorResponse(
                status: .serviceUnavailable,
                message: error.description,
                code: "model_not_loaded"
            )
        case .malformedSafetensors,
             .missingTensor,
             .shapeMismatch,
             .dtypeMismatch,
             .unexpectedTensor:
            return errorResponse(
                status: .internalServerError,
                message: error.description,
                code: "model_load_failed"
            )
        case .fetchFailed(let failure):
            return mapFetchFailure(failure, message: error.description)
        case .tokenizerValidationFailed(let detail):
            return errorResponse(
                status: .internalServerError,
                message: detail,
                code: "tokenizer_validation_failed"
            )
        }
    }

    /// Demux a structured `FetchFailure` payload into its HTTP code.
    /// Shared between `/v1/stt/batch` and `/v1/stt/load` so clients
    /// see a single mapping regardless of which endpoint surfaced
    /// the failure.
    nonisolated func mapFetchFailure(
        _ failure: FetchFailure,
        message: String? = nil
    ) -> Response {
        let body = message ?? failure.description
        switch failure {
        case .httpStatus(let code, _):
            switch code {
            case 401:
                return errorResponse(
                    status: .unauthorized,
                    message: body,
                    code: "model_fetch_unauthorized"
                )
            case 404:
                return errorResponse(
                    status: .notFound,
                    message: body,
                    code: "model_fetch_not_found"
                )
            case 500..<600:
                return errorResponse(
                    status: .badGateway,
                    message: body,
                    code: "model_fetch_upstream_5xx"
                )
            default:
                return errorResponse(
                    status: .badGateway,
                    message: body,
                    code: "model_fetch_failed"
                )
            }
        case .checksumMismatch, .sizeMismatch:
            // Local integrity violation. Not a transient gateway
            // problem; retrying the same gateway will likely
            // surface the same bytes. Surface as 422 with a
            // dedicated code so retry policy doesn't spin.
            return errorResponse(
                status: .unprocessableContent,
                message: body,
                code: "model_corruption_detected"
            )
        case .rangeIgnored:
            return errorResponse(
                status: .conflict,
                message: body,
                code: "model_fetch_resume_failed"
            )
        case .manifestDecode:
            return errorResponse(
                status: .badGateway,
                message: body,
                code: "model_manifest_invalid"
            )
        case .other:
            // Internal contract violation (typo'd repo URL, etc.) —
            // not a gateway problem. 500 so it surfaces as a server
            // bug rather than an upstream blame.
            return errorResponse(
                status: .internalServerError,
                message: body,
                code: "model_fetch_internal_error"
            )
        case .transport:
            return errorResponse(
                status: .badGateway,
                message: body,
                code: "model_fetch_failed"
            )
        }
    }

    /// Demux a `/v1/stt/load` failure into a structured wire response.
    /// Mirrors `mapFetchFailure`'s code surface so a Parakeet/HF-cache
    /// failure looks identical on the wire to a Qwen3 fetch failure
    /// of the same root cause. Codes are stable contract — adding a
    /// new one is fine, renaming an existing one is a breaking change.
    nonisolated func mapSTTLoadError(_ error: any Error) -> Response {
        // Cooperative cancellation. 499 (client-closed-request) is a
        // de-facto convention for "the awaiter went away"; falling
        // back to 503 would conflate it with server unavailability.
        if error is CancellationError {
            return errorResponse(
                status: .init(code: 499, reasonPhrase: "Client Closed Request"),
                message: "Load cancelled",
                code: "cancelled"
            )
        }

        // No HF mirror wired for this language family. 501 is the
        // right code: the request was well-formed and the language
        // is recognised, but the server has no implementation path
        // for it on this build.
        if let download = error as? STTHFDownloadError {
            switch download {
            case .unsupportedLanguage(let language):
                return errorResponse(
                    status: .notImplemented,
                    message: download.errorDescription ?? "Language unmirrored: \(language.rawValue)",
                    code: "language_unmirrored"
                )
            }
        }

        // Offline-mode cache miss (`HubClient.downloadSnapshot` with
        // `localFilesOnly: true` and no cached snapshot). Distinct
        // from a generic "not found" — the client retries with
        // `allow_fetch=true` to recover.
        if let cacheError = error as? HubCacheError {
            switch cacheError {
            case .cachedPathResolutionFailed:
                return errorResponse(
                    status: .notFound,
                    message: cacheError.errorDescription ?? "STT model not in HF cache; retry with allow_fetch=true.",
                    code: "model_not_cached"
                )
            default:
                return errorResponse(
                    status: .internalServerError,
                    message: cacheError.errorDescription ?? "HF cache error",
                    code: "model_cache_error"
                )
            }
        }

        // HTTP-layer failures from the HF download. Demux on status
        // code so callers can distinguish auth-required from gone
        // from upstream-5xx; the codes match `mapFetchFailure` so
        // the Qwen3 and Parakeet paths look the same on the wire.
        if let httpError = error as? HTTPClientError {
            switch httpError {
            case .responseError(let response, let detail):
                let body = "\(detail) (HTTP \(response.statusCode) for \(response.url?.absoluteString ?? "unknown"))"
                switch response.statusCode {
                case 401:
                    return errorResponse(
                        status: .unauthorized,
                        message: body,
                        code: "model_fetch_unauthorized"
                    )
                case 404:
                    return errorResponse(
                        status: .notFound,
                        message: body,
                        code: "model_fetch_not_found"
                    )
                case 500..<600:
                    return errorResponse(
                        status: .badGateway,
                        message: body,
                        code: "model_fetch_upstream_5xx"
                    )
                default:
                    return errorResponse(
                        status: .badGateway,
                        message: body,
                        code: "model_fetch_failed"
                    )
                }
            case .requestError(let detail), .unexpectedError(let detail):
                return errorResponse(
                    status: .badGateway,
                    message: detail,
                    code: "model_fetch_failed"
                )
            case .decodingError(let response, let detail):
                return errorResponse(
                    status: .badGateway,
                    message: "\(detail) (HTTP \(response.statusCode))",
                    code: "model_fetch_failed"
                )
            }
        }

        // Network-layer failures (offline, DNS, timeout) — caller
        // should retry; not a server bug.
        if let urlError = error as? URLError {
            return errorResponse(
                status: .badGateway,
                message: urlError.localizedDescription,
                code: "model_fetch_network_error"
            )
        }

        // Engine-side validation. `languageNotSupported` is a 400
        // (the client sent a code we don't implement); any other
        // YoozSTTError surfaces as a generic load failure.
        if let sttError = error as? YoozSTTError {
            switch sttError {
            case .languageNotSupported:
                return errorResponse(
                    status: .badRequest,
                    message: sttError.errorDescription ?? "Language not supported",
                    code: "language_not_supported"
                )
            case .modelNotFound(let message):
                return errorResponse(
                    status: .notFound,
                    message: message,
                    code: "model_not_found"
                )
            default:
                return errorResponse(
                    status: .internalServerError,
                    message: sttError.errorDescription ?? "Load failed",
                    code: "load_failed"
                )
            }
        }

        return errorResponse(
            status: .internalServerError,
            message: error.localizedDescription,
            code: "load_failed"
        )
    }

    private nonisolated func parseLanguage(_ code: String?) throws -> STTLanguage {
        let languageCode = code ?? "en"
        guard let language = STTLanguage.fromCode(languageCode) else {
            throw STTRequestError.invalidLanguage(languageCode)
        }
        guard language.isImplemented else {
            throw STTRequestError.notImplemented(language.displayName)
        }
        return language
    }

    // MARK: - HTTP Router

    /// Build a router for testing without spinning up the full server
    /// task. Used by HTTP integration tests that exercise the route
    /// handlers via `Application.test(.router) { ... }`.
    func makeTestRouter() -> Router<BasicRequestContext> {
        buildRouter()
    }

    /// Build a WebSocket router for testing.
    func makeTestWebSocketRouter() -> Router<BasicWebSocketRequestContext> {
        buildWebSocketRouter()
    }

    private func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        let sttEngine = YoozSTTEngine.shared
        // Hoist the metrics sink + preview fallback hook so they
        // share lifecycle with the router (one engine `start()` call
        // = one sink + one hook, surviving across requests). The
        // hook's cold-start state machine relies on this — a
        // per-request hook would treat every request as a new
        // cold-start.
        let metricsSink: any STTMetricsSink = makeSTTMetricsSink(
            optedIn: EngineConfig.telemetryOptedIn,
            fileURL: EngineConfig.sttMetricsFileURL
        )
        let previewFallbackHook = PreviewFallbackHook(
            fetcher: Qwen3ASRPreviewFetcherAdapter(),
            preview: Qwen3ASRPreviewBackendAdapter(),
            fallback: ParakeetFallbackAdapter(),
            sink: metricsSink
        )

        // Health
        router.get("/v1/health") { _, _ in
            // Pull the variant-aware readiness map from the eager
            // loader. The map is the source of truth for the new
            // `loading` / `error` / `unavailable` states; the legacy
            // bool fields stay for SDK back-compat (true iff the
            // module reports `.ready`).
            //
            // We also OR in the engines' current "is loaded" flags
            // so a thin client that called `/v1/stt/load` or
            // `/v1/llm/generate` and bypassed the eager loader still
            // sees `true` here. The eager loader only writes its
            // own state; it does not poll the engines after kickoff.
            let detail = await ModuleEagerLoader.shared.snapshot()
            let llmLoaded = await TouchUpEngine.shared.isLightModelLoaded
            let touchupReady = await TouchUpEngine.shared.isPreloaded
            let vadLoaded = await VADEngine.shared.isLoaded

            func isReady(_ id: ModuleID, fallback: Bool) -> Bool {
                if detail[id.rawValue]?.state == .ready { return true }
                return fallback
            }

            return HealthResponse(
                status: "ok",
                version: EngineConfig.version,
                modules: EngineModules(
                    stt: isReady(.stt, fallback: sttEngine.isRunning),
                    llm: isReady(.llm, fallback: llmLoaded),
                    touchup: isReady(.touchup, fallback: touchupReady),
                    grammar: isReady(
                        .grammar,
                        fallback: GrammarEngine.shared.isAvailable
                    ),
                    vad: isReady(.vad, fallback: vadLoaded),
                    tts: isReady(.tts, fallback: false),
                    detail: detail
                )
            )
        }

        // Modules — purpose-built status endpoint for the engine
        // status UI in thin clients. Same `detail` map as
        // `/v1/health.modules.detail`, plus the active variant so
        // clients can special-case `unavailable` modules (whisper's
        // VAD row, lite's STT row).
        router.get("/v1/modules") { _, _ in
            let detail = await ModuleEagerLoader.shared.snapshot()
            return ModulesResponseV1(
                variant: EngineConfig.variant.rawValue,
                version: EngineConfig.version,
                modules: detail
            )
        }

        // Models
        router.get("/v1/models") { _, _ in
            var models: [ModelInfo] = []
            if sttEngine.isRunning {
                models.append(ModelInfo(
                    name: sttEngine.currentLanguage.modelIdentifier,
                    module: "stt",
                    loaded: true,
                    sizeBytes: nil
                ))
            }
            let llmInfo = await TouchUpEngine.shared.getModelInfo()
            models.append(ModelInfo(
                name: llmInfo.light.type.rawValue,
                module: "llm",
                loaded: llmInfo.light.isLoaded,
                sizeBytes: llmInfo.light.type.estimatedSize
            ))
            models.append(ModelInfo(
                name: llmInfo.quality.type.rawValue,
                module: "llm",
                loaded: llmInfo.quality.isLoaded,
                sizeBytes: llmInfo.quality.type.estimatedSize
            ))
            let fmLoaded = await TouchUpEngine.shared.isFoundationModelsLoaded
            if fmLoaded {
                models.append(ModelInfo(
                    name: "foundation-models",
                    module: "llm",
                    loaded: true,
                    sizeBytes: nil
                ))
            }
            return ModelsResponse(models: models)
        }

        // LLM: Generate
        router.post("/v1/llm/generate") { [self] request, context in
            let body: LLMGenerateServerRequest
            do {
                body = try await request.decode(as: LLMGenerateServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            // Resolve model type from name (default to light)
            let modelType: LLMModelType
            if let modelName = body.model {
                guard let resolved = LLMModelType(rawValue: modelName) else {
                    return errorResponse(
                        status: .badRequest,
                        message: "Unknown model: \(modelName). Available: \(LLMModelType.allCases.map(\.rawValue).joined(separator: ", "))",
                        code: "invalid_model"
                    )
                }
                modelType = resolved
            } else {
                modelType = .yoozLight
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            do {
                let result = try await TouchUpEngine.shared.generate(
                    prompt: body.prompt,
                    systemPrompt: body.systemPrompt ?? "",
                    modelType: modelType,
                    kvCompression: body.kvCompression
                )
                let timeMs = Int((CFAbsoluteTimeGetCurrent() - startTime) * 1000)
                return try jsonResponse(LLMGenerateServerResponse(
                    text: result,
                    model: modelType.rawValue,
                    tokensGenerated: nil,
                    processingTimeMs: timeMs
                ))
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "generation_failed"
                )
            }
        }

        // TouchUp: Process text
        router.post("/v1/touchup") { [self] request, context in
            let body: TouchUpServerRequest
            do {
                body = try await request.decode(as: TouchUpServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            if body.mode == .off {
                // Off mode: regex-only processing (no LLM)
                let result = await TouchUpEngine.shared.processRegexOnly(text: body.text)
                return try jsonResponse(TouchUpServerResponse(
                    result: result.text,
                    mode: .off,
                    processingTimeMs: Int(result.latencyMs),
                    modelUsed: result.modelUsed.rawValue,
                    warnings: nil
                ))
            }

            // Route through the picker-aware path so the model the
            // user picked via `POST /v1/touchup/model` is honored.
            // For callers that never set a model, this is identical
            // to the legacy `process(...)` path (default
            // `activeModel == .yoozLight`).
            let result = await TouchUpEngine.shared.processWithActiveModel(
                text: body.text,
                mode: body.mode
            )

            var warnings: [String]? = nil
            if let reason = result.fallbackReason {
                warnings = [reason]
            }

            return try jsonResponse(TouchUpServerResponse(
                result: result.text,
                mode: body.mode,
                processingTimeMs: Int(result.latencyMs),
                modelUsed: result.modelUsed.rawValue,
                warnings: warnings
            ))
        }

        // TouchUp: List available models for the picker UI.
        // See AGENTS.md "Module model picker pattern" — this is the
        // canonical shape every module's picker endpoint should
        // follow (id / displayName / description / tier / sizeBytes
        // / isAvailable / isCached / isLoaded / isActive).
        router.get("/v1/touchup/models") { _, _ in
            let models = await TouchUpEngine.shared.availableModels()
            let activeId = await TouchUpEngine.shared.activeModel.rawValue
            return TouchUpModelsResponse(models: models, activeId: activeId)
        }

        // TouchUp: Set active model. `preload` defaults to true so a
        // one-shot picker change is enough to warm the model before
        // the next `/v1/touchup` call. Returns the new active row so
        // clients don't need a follow-up GET.
        router.post("/v1/touchup/model") { [self] request, context in
            let body: TouchUpSetModelRequest
            do {
                body = try await request.decode(
                    as: TouchUpSetModelRequest.self, context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            guard let selection = TouchUpModelSelection(rawValue: body.id) else {
                let known = TouchUpModelSelection.allCases.map(\.rawValue).joined(separator: ", ")
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown TouchUp model id '\(body.id)'. Known: \(known).",
                    code: "invalid_model"
                )
            }

            do {
                let active = try await TouchUpEngine.shared.setActiveModel(
                    selection,
                    preload: body.preload ?? true
                )
                return try jsonResponse(active)
            } catch let error as LLMError {
                // `notAvailable` is the FoundationModels-on-pre-26
                // case — surface as 501 so the picker UI can render
                // "not supported on this Mac" cleanly. Any other
                // load failure (network, OOM) is 500.
                switch error {
                case .notAvailable(let detail):
                    return errorResponse(
                        status: .notImplemented,
                        message: detail,
                        code: "model_unavailable"
                    )
                default:
                    return errorResponse(
                        status: .internalServerError,
                        message: error.localizedDescription,
                        code: "model_set_failed"
                    )
                }
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "model_set_failed"
                )
            }
        }

        // Grammar: Check text
        router.post("/v1/grammar/check") { [self] request, context in
            let body: GrammarCheckServerRequest
            do {
                body = try await request.decode(as: GrammarCheckServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            guard GrammarEngine.shared.isAvailable else {
                return errorResponse(
                    status: .serviceUnavailable,
                    message: "Grammar rules not loaded",
                    code: "grammar_not_available"
                )
            }

            let usePOS = body.usePOS ?? true
            let result = await GrammarEngine.shared.check(
                text: body.text,
                categories: body.categories,
                usePOS: usePOS
            )
            return try jsonResponse(GrammarCheckServerResponse(
                result: result.result,
                correctionsApplied: result.correctionsApplied,
                ruleCount: GrammarEngine.shared.ruleCount
            ))
        }

        // VAD: Detect speech segments
        router.post("/v1/vad/detect") { [self] request, context in
            let body: VADDetectServerRequest
            do {
                body = try await request.decode(as: VADDetectServerRequest.self, context: context)
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            guard await VADEngine.shared.isLoaded else {
                return errorResponse(
                    status: .serviceUnavailable,
                    message: "VAD model not loaded",
                    code: "vad_not_loaded"
                )
            }

            guard body.samples.count >= VADEngine.windowSize else {
                return errorResponse(
                    status: .badRequest,
                    message: "Samples too short; need at least \(VADEngine.windowSize) (32ms at 16kHz)",
                    code: "samples_too_short"
                )
            }

            do {
                let shouldReset = body.reset ?? true
                let segments = try await VADEngine.shared.detect(
                    samples: body.samples,
                    resetState: shouldReset
                )
                let responseSegments = segments.map { seg in
                    VADSegment(startMs: seg.startMs, endMs: seg.endMs, probability: seg.probability)
                }
                return try jsonResponse(VADDetectServerResponse(segments: responseSegments))
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "vad_detection_failed"
                )
            }
        }

        // STT: Backend selection
        router.get("/v1/stt/engine") { _, _ in
            let current = sttEngine.currentBackend.rawValue
            let available = STTBackendID.allCases.map { backend in
                STTEngineCapabilities(
                    id: backend.rawValue,
                    supportsBatch: backend.supportsBatch,
                    supportsStreaming: backend.supportsStreaming,
                    supportedLanguages: backend.supportedLanguages
                        .map(\.rawValue)
                )
            }
            return STTEngineGetResponse(current: current, available: available)
        }

        router.post("/v1/stt/engine") { [self] request, context in
            let body: STTEnginePostRequest
            do {
                body = try await request.decode(
                    as: STTEnginePostRequest.self, context: context
                )
            } catch {
                return errorResponse(
                    status: .badRequest,
                    message: "Invalid request body: \(error.localizedDescription)",
                    code: "invalid_request"
                )
            }

            guard let backend = STTBackendID(rawValue: body.engine) else {
                return errorResponse(
                    status: .badRequest,
                    message: "Unknown engine '\(body.engine)'. Available: "
                        + STTBackendID.allCases.map(\.rawValue).joined(separator: ", "),
                    code: "invalid_engine"
                )
            }

            await sttEngine.setBackend(backend)
            return try jsonResponse(STTEnginePostResponse(
                current: sttEngine.currentBackend.rawValue
            ))
        }

        // STT: Available languages
        router.get("/v1/stt/languages") { _, _ in
            STTLanguagesResponse(languages: STTLanguage.allCases.map { lang in
                STTLanguageInfo(
                    code: lang.rawValue,
                    name: lang.displayName,
                    implemented: lang.isImplemented,
                    family: lang.modelFamily.rawValue
                )
            })
        }

        // STT: Status
        router.get("/v1/stt/status") { _, _ in
            // `isCurrentBackendLoaded()` is backend-aware: for
            // Parakeet/FastConformer/AppleSTT it reads `model != nil`
            // (same as `isRunning`); for `qwen3_asr_preview` it
            // reads the actor's `isLoaded` flag. Without this, after
            // a fallback the engine would report `loaded: true` for
            // qwen3_asr_preview while only the Parakeet model is in
            // memory — see round-2 silent-failure #B4.
            let backendLoaded = await sttEngine.isCurrentBackendLoaded()
            // Surface the language the *currently-selected* backend
            // is loaded for. For qwen3 the backend actor doesn't
            // expose a per-language state today, so we report the
            // engine's `currentLanguage` (which the WS / batch
            // handlers use for the Qwen3 hint) when the backend is
            // loaded; nil otherwise.
            let language: String? = backendLoaded
                ? sttEngine.currentLanguage.rawValue
                : nil
            return STTStatusResponse(
                loaded: backendLoaded,
                language: language,
                streaming: sttEngine.isStreaming,
                progress: sttEngine.downloadProgress
            )
        }

        // STT: Load model
        router.post("/v1/stt/load") { [self] request, context in
            let body = try await request.decode(as: STTLoadRequest.self, context: context)

            let language: STTLanguage
            do {
                language = try parseLanguage(body.language)
            } catch let error as STTRequestError {
                return errorResponse(
                    status: .badRequest,
                    message: error.message,
                    code: error.code
                )
            }

            // When the active backend is qwen3, ensure the model
            // directory is materialized on disk before calling
            // start(). The fetch is opt-in via `allowFetch` (default
            // true for ergonomics; CI/tests can disable it).
            if sttEngine.currentBackend == .qwen3ASRPreview {
                let modelDir = Qwen3ASRModelFetcher.defaultModelDir
                let fetcher = Qwen3ASRModelFetcher.shared
                let ready = await fetcher.isModelDirReady(modelDir)
                if !ready {
                    let allow = body.allowFetch ?? true
                    if !allow {
                        return errorResponse(
                            status: .notFound,
                            message: "Qwen3-ASR model not on disk and "
                                + "allow_fetch=false; bring the model "
                                + "directory into place first.",
                            code: "model_not_found"
                        )
                    }
                    do {
                        let stream = await fetcher.download(into: modelDir)
                        for try await _ in stream {
                            // Drain the progress stream; HTTP clients
                            // get a single response after the fetch
                            // completes. Streaming progress to clients
                            // is a future enhancement (issue #59).
                        }
                    } catch let asrError as Qwen3ASRError {
                        // Route through the same demux as
                        // /v1/stt/batch so a fetch failure surfaces
                        // the same HTTP code regardless of which
                        // endpoint the client hit.
                        return mapQwen3BatchError(asrError)
                    } catch let fetchFailure as FetchFailure {
                        return mapFetchFailure(fetchFailure)
                    } catch {
                        return errorResponse(
                            status: .internalServerError,
                            message: "Model fetch failed: \(error.localizedDescription)",
                            code: "model_fetch_failed"
                        )
                    }
                } else {
                    // Even when the directory is ready, run the
                    // tokenizer prep step in case a previous run
                    // missed it (e.g. user pre-staged the files
                    // manually). Idempotent.
                    do {
                        try await Qwen3ASRTokenizerPrep.prepare(
                            modelDir: modelDir
                        )
                    } catch let asrError as Qwen3ASRError {
                        // Same mapping as the batch path so the wire
                        // shape is consistent
                        // (`tokenizer_validation_failed` /
                        // `model_fetch_failed` / etc).
                        return mapQwen3BatchError(asrError)
                    } catch {
                        return errorResponse(
                            status: .internalServerError,
                            message: "Tokenizer prep failed: \(error.localizedDescription)",
                            code: "tokenizer_prep_failed"
                        )
                    }
                }
            }

            // See `YoozSTTEngine.getModelDirectory` for resolution
            // order. `allowFetch` defaults to true; clients that want
            // to fail fast on a missing model pass `allow_fetch=false`.
            let allowFetch = body.allowFetch ?? true
            do {
                try await sttEngine.start(
                    language: language,
                    allowFetch: allowFetch
                )
                return try jsonResponse(STTStatusResponse(
                    loaded: true,
                    language: language.rawValue,
                    streaming: false,
                    progress: sttEngine.downloadProgress
                ))
            } catch {
                return mapSTTLoadError(error)
            }
        }

        // STT: Batch transcribe
        router.post("/v1/stt/batch") { [self] request, context in
            let body = try await request.decode(as: BatchSTTRequest.self, context: context)

            let language: STTLanguage
            do {
                language = try parseLanguage(body.language)
            } catch let error as STTRequestError {
                return errorResponse(
                    status: .badRequest,
                    message: error.message,
                    code: error.code
                )
            }

            let mode = AudioMode(rawValue: body.mode ?? "normal") ?? .normal

            // Route through the Qwen3 backend when active. Typed
            // errors from the backend are mapped to HTTP error
            // responses below; we do NOT swallow Qwen3 failures as
            // 200 OK with empty text — the user deserves to see
            // pipeline-not-loaded / invalid-input as a structured
            // response, not silent emptiness.
            let result: ParakeetResult
            if sttEngine.currentBackend == .qwen3ASRPreview {
                // Route through the auto-fallback hook so a
                // first-run cold-start failure (network blip,
                // missing model) hands off to Parakeet instead of
                // returning a 500. Once the cold-start path
                // resolves, subsequent failures propagate with
                // typed HTTP codes (the hook only surfaces
                // `Outcome` for happy-path & fallback; raw
                // post-warmup throws still hit the catch below).
                let outcome = await previewFallbackHook
                    .attemptPreviewWithFallback(
                        samples: body.samples,
                        language: language
                    )
                result = outcome.result
            } else {
                // Ensure model is loaded for the requested language
                do {
                    try await sttEngine.start(language: language)
                } catch {
                    return errorResponse(
                        status: .internalServerError,
                        message: error.localizedDescription,
                        code: "model_load_failed"
                    )
                }
                result = await sttEngine.batchTranscribe(
                    samples: body.samples, mode: mode
                )
            }

            return try jsonResponse(BatchSTTResponse(
                text: result.text,
                finalized: result.finalized,
                draft: result.draft,
                language: language.rawValue
            ))
        }

        return router
    }

    // MARK: - WebSocket Router

    private func buildWebSocketRouter() -> Router<BasicWebSocketRequestContext> {
        let wsRouter = Router(context: BasicWebSocketRequestContext.self)
        let sttLogger = Logger(label: "live.yooz.engine.stt.stream")
        // One sink instance per server lifetime — file handle stays
        // owned by the actor. Per-request sinks would race on the
        // file URL and confuse lifecycle ("is this sink alive?").
        let metricsSink: any STTMetricsSink = makeSTTMetricsSink(
            optedIn: EngineConfig.telemetryOptedIn,
            fileURL: EngineConfig.sttMetricsFileURL
        )

        wsRouter.ws("/v1/stt/stream") { inbound, outbound, context in
            let sttEngine = YoozSTTEngine.shared
            var transcriber: StreamingTranscriber?
            // Per-connection streaming session for the qwen3
            // backend. Owns its own audio buffer; the underlying
            // backend actor serializes concurrent transcribe calls.
            var qwen3Session: Qwen3ASRStreamingSession?
            // Wall clock at session start for the telemetry record
            // emitted on `final`. Set when the qwen3 session is
            // configured (matches "stream start" semantics).
            var qwen3StreamStartedMs: UInt64?
            // True once we've sent the buffer-cap warning frame.
            // The frame is one-shot per connection so a long
            // dictation past the cap doesn't flood the client with
            // identical warnings.
            var qwen3BufferCapWarned = false
            let encoder = JSONEncoder()

            // Encode a WSSTTResult to JSON and send via WebSocket
            func sendResult(_ result: WSSTTResult) async {
                do {
                    let json = try encoder.encode(result)
                    guard let jsonStr = String(data: json, encoding: .utf8) else {
                        sttLogger.error("Failed to convert encoded result to UTF-8 string")
                        return
                    }
                    try await outbound.write(.text(jsonStr))
                } catch {
                    sttLogger.error("Failed to send STT result: \(error)")
                }
            }

            // Send a JSON-encoded error message to the client.
            // `code` is a stable identifier the client can branch on
            // (e.g. `protocol`, `stream_aborted`) without parsing
            // `message`. Best-effort: if the connection has already
            // gone, this is a no-op.
            func sendError(
                _ message: String, code: WSSTTErrorCode? = nil
            ) async {
                do {
                    let json = try encoder.encode(
                        WSSTTError(
                            type: "error", message: message, code: code
                        )
                    )
                    guard let jsonStr = String(data: json, encoding: .utf8) else { return }
                    try await outbound.write(.text(jsonStr))
                } catch {
                    sttLogger.error("Failed to send error message: \(error)")
                }
            }

            // Send a one-shot non-fatal warning frame. `code` is a
            // typed enum; the wire still carries the snake_case
            // rawValue ("buffer_cap_reached", etc.) so clients
            // don't see a wire-shape change.
            func sendWarning(
                code: WSSTTWarningCode, message: String
            ) async {
                do {
                    let json = try encoder.encode(
                        WSSTTWarning(
                            type: "warning",
                            code: code,
                            message: message
                        )
                    )
                    guard let jsonStr = String(data: json, encoding: .utf8) else { return }
                    try await outbound.write(.text(jsonStr))
                } catch {
                    sttLogger.error("Failed to send warning: \(error)")
                }
            }

            // Build a Qwen3 telemetry record. `audioMs` is the
            // amount of audio actually consumed by the session
            // (post-cap if truncation happened). `errored` flips
            // when the stream tore down via an exception path so
            // dashboards can distinguish clean closes from drops.
            func buildQwen3Metric(
                audioMs: UInt32, fellBack: Bool, errored: Bool
            ) -> STTBackendMetrics {
                let endMs = DispatchTime.now().uptimeNanoseconds / 1_000_000
                let elapsed: UInt32
                if let started = qwen3StreamStartedMs {
                    elapsed = UInt32(min(endMs &- started, UInt64(UInt32.max)))
                } else {
                    elapsed = 0
                }
                let variant = STTBackendMetrics.defaultModelVariant(
                    for: .qwen3ASRPreview
                )
                return STTBackendMetrics(
                    backend: .qwen3ASRPreview,
                    modelVariant: variant,
                    audioDurationMs: audioMs,
                    timeToFirstTokenMs: nil,
                    endToEndLatencyMs: elapsed,
                    hardwareClass: SystemHardwareClassResolver().resolve(),
                    fellBackFromPreview: fellBack,
                    streamAborted: errored,
                    timestampUTC: Date()
                )
            }

            sttLogger.info("STT WebSocket client connected")

            // Use message-level API to handle fragmented frames automatically.
            // Max message size: 10MB (enough for ~2.7 min of 16kHz Float32 audio).
            //
            // We wrap the loop in do/catch so a thrown error inside
            // `inbound.messages(...)` (oversized frame, abrupt
            // disconnect mid-frame, framer decode error) does NOT
            // bypass the cleanup block below — without this, the
            // qwen3 session never gets `discard()`-ed, no telemetry
            // record fires, and the audio buffer (up to ~19 MB at
            // the 5-min cap) lingers on the actor's executor.
            do {
            for try await message in inbound.messages(maxSize: 10 * 1024 * 1024) {
                switch message {
                case .text(let text):
                    // Config message: {"type":"config","language":"en","mode":"normal"}
                    let config: WSSTTConfig
                    do {
                        guard let data = text.data(using: .utf8) else {
                            sttLogger.warning("Received non-UTF8 text message")
                            continue
                        }
                        config = try JSONDecoder().decode(WSSTTConfig.self, from: data)
                    } catch {
                        sttLogger.warning("Invalid text message: \(error)")
                        await sendError(
                            "Invalid message format",
                            code: .invalidMessageFormat
                        )
                        continue
                    }

                    if config.type == "config" {
                        let languageCode = config.language ?? "en"
                        guard let language = STTLanguage.fromCode(languageCode) else {
                            await sendError(
                                "Unknown language: \(languageCode)",
                                code: .unknownLanguage
                            )
                            continue
                        }

                        guard language.isImplemented else {
                            await sendError(
                                "Language not implemented: \(language.displayName)",
                                code: .languageNotImplemented
                            )
                            continue
                        }

                        // Branch on the active backend at config
                        // time. The qwen3 path opens a streaming
                        // session against the (already-loaded)
                        // backend actor; the Parakeet/FastConformer
                        // path creates a per-connection
                        // `StreamingTranscriber` exactly as before.
                        if sttEngine.currentBackend == .qwen3ASRPreview {
                            // Allow only languages the backend
                            // declares; the engine's `start` would
                            // otherwise refuse later anyway, but
                            // surfacing the error here keeps the
                            // protocol response immediate.
                            guard
                                STTBackendID.qwen3ASRPreview.supportedLanguages
                                    .contains(language)
                            else {
                                await sendError(
                                    "Language not supported by qwen3_asr_preview: "
                                        + language.displayName,
                                    code: .languageNotSupportedByBackend
                                )
                                continue
                            }

                            do {
                                try await sttEngine.start(language: language)
                            } catch {
                                await sendError(
                                    "Failed to load model: \(error.localizedDescription)",
                                    code: .modelLoadFailed
                                )
                                continue
                            }

                            // Discard any prior session — clients
                            // sending a second `config` reset the
                            // transcript. Mirrors how the legacy
                            // path overwrites `transcriber`.
                            if let existing = qwen3Session {
                                await existing.discard()
                            }
                            qwen3Session = Qwen3ASRStreamingSession(
                                languageHint: language.qwen3LanguageHint
                            )
                            qwen3StreamStartedMs =
                                DispatchTime.now().uptimeNanoseconds / 1_000_000
                            // Each `config` message starts a brand-new
                            // session with an empty buffer; the cap
                            // warning is one-shot per session, not per
                            // WS connection, so the prior session's
                            // flag must not silence the next session's
                            // truncation.
                            qwen3BufferCapWarned = false

                            do {
                                let ready = try encoder.encode(
                                    WSSTTReady(type: "ready", language: languageCode)
                                )
                                if let readyStr = String(data: ready, encoding: .utf8) {
                                    try await outbound.write(.text(readyStr))
                                }
                            } catch {
                                sttLogger.error("Failed to send ready message: \(error)")
                            }
                            sttLogger.info(
                                "STT stream configured: language=\(languageCode), backend=qwen3_asr_preview"
                            )
                            continue
                        }

                        do {
                            try await sttEngine.start(language: language)
                        } catch {
                            await sendError(
                                "Failed to load model: \(error.localizedDescription)",
                                code: .modelLoadFailed
                            )
                            continue
                        }

                        let mode = AudioMode(rawValue: config.mode ?? "normal") ?? .normal
                        transcriber = sttEngine.createBatchTranscriber(mode: mode)

                        do {
                            let ready = try encoder.encode(WSSTTReady(type: "ready", language: languageCode))
                            if let readyStr = String(data: ready, encoding: .utf8) {
                                try await outbound.write(.text(readyStr))
                            }
                        } catch {
                            sttLogger.error("Failed to send ready message: \(error)")
                        }
                        sttLogger.info("STT stream configured: language=\(languageCode), mode=\(mode.rawValue)")
                    }

                case .binary(var buffer):
                    let byteCount = buffer.readableBytes
                    // Reject obviously-malformed frames early. The
                    // protocol expects a multiple of `Float32`-sized
                    // bytes; partial-sample frames are dropped with a
                    // typed error so the client can correct rather
                    // than silently corrupting the buffer.
                    if byteCount % MemoryLayout<Float>.size != 0 {
                        await sendError(
                            "Audio frame byte count (\(byteCount)) is not a "
                                + "multiple of Float32 size; expected raw "
                                + "Float32 PCM at 16 kHz mono.",
                            code: .invalidAudioFrame
                        )
                        continue
                    }
                    let sampleCount = byteCount / MemoryLayout<Float>.size
                    guard sampleCount > 0 else { continue }

                    // Safe copy into aligned Float array via raw byte copy
                    let samples: [Float] = buffer.withUnsafeReadableBytes { ptr in
                        [Float](unsafeUninitializedCapacity: sampleCount) { dest, initializedCount in
                            _ = UnsafeMutableRawBufferPointer(dest).copyBytes(
                                from: UnsafeRawBufferPointer(ptr).prefix(sampleCount * MemoryLayout<Float>.size)
                            )
                            initializedCount = sampleCount
                        }
                    }

                    if let session = qwen3Session {
                        let outcome: Qwen3ASRStreamingSession.PushOutcome
                        do {
                            outcome = try await session.push(samples: samples)
                        } catch let sessionError as Qwen3ASRStreamingSession.SessionError {
                            await sendError(
                                "Streaming session error: \(sessionError.description)",
                                code: .sessionError
                            )
                            continue
                        } catch {
                            await sendError(
                                "Streaming session error: \(error.localizedDescription)",
                                code: .sessionError
                            )
                            continue
                        }
                        // Detect a buffer-cap hit: push() silently
                        // truncates past the soft cap. Fire a
                        // one-shot warning frame the first time
                        // truncation happens — including the very
                        // first chunk that partially crosses the cap,
                        // not just subsequent fully-rejected chunks.
                        // The flag is reset when the client sends a
                        // second `config` (transcript reset), so a
                        // long dictation that re-configs mid-stream
                        // still gets a warning per session.
                        if outcome.truncated && !qwen3BufferCapWarned {
                            qwen3BufferCapWarned = true
                            await sendWarning(
                                code: .bufferCapReached,
                                message:
                                    "Audio buffer reached the streaming "
                                    + "cap. Additional audio is being "
                                    + "discarded. The transcript on "
                                    + "`final` reflects the buffered "
                                    + "audio only."
                            )
                        }
                        // Heartbeat partial — qwen3's streaming model
                        // doesn't expose intermediate text without a
                        // re-prefill on every chunk, so the partial
                        // text fields stay empty until `final`.
                        await sendResult(WSSTTResult(
                            type: "partial",
                            text: "",
                            finalized: "",
                            draft: ""
                        ))
                        continue
                    }

                    guard let transcriber = transcriber else {
                        sttLogger.warning("Audio received before config message")
                        continue
                    }

                    let result = transcriber.addAudio(samples: samples)

                    await sendResult(WSSTTResult(
                        type: "partial",
                        text: result.text,
                        finalized: result.finalized,
                        draft: result.draft
                    ))
                }
            }
            } catch is CancellationError {
                // Outer Task was cancelled (server stop, peer
                // dropped, etc.). Run the same cleanup as a clean
                // disconnect so the qwen3 session is `discard()`-ed
                // and a metric record fires with the audio actually
                // consumed before the cancel.
                if let session = qwen3Session {
                    let audioMs = await session.audioDurationMs
                    await session.discard()
                    qwen3Session = nil
                    let metric = buildQwen3Metric(
                        audioMs: audioMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                }
                sttLogger.info("STT WebSocket cancelled")
                throw CancellationError()
            } catch {
                // The message loop threw — oversized frame past the
                // 10 MB cap, abrupt connection drop with a partial
                // frame in flight, framer decode error. Run the
                // qwen3 session cleanup we would otherwise miss,
                // emit one error frame to the client (best-effort),
                // and emit a metric record so dashboards see the
                // drop rate.
                sttLogger.error(
                    "STT WebSocket loop threw: \(String(describing: error))"
                )
                await sendError(
                    "Stream aborted: \(error.localizedDescription)",
                    code: .streamAborted
                )
                if let session = qwen3Session {
                    let audioMs = await session.audioDurationMs
                    await session.discard()
                    qwen3Session = nil
                    let metric = buildQwen3Metric(
                        audioMs: audioMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                }
                sttLogger.info("STT WebSocket client disconnected (error)")
                return
            }

            // Client disconnected cleanly; finalize whichever
            // session is active.
            if let session = qwen3Session {
                do {
                    let final = try await session.finalize()
                    await sendResult(WSSTTResult(
                        type: "final",
                        text: final.text,
                        finalized: final.text,
                        draft: ""
                    ))
                    // Telemetry — opt-in via EngineConfig. The
                    // hoisted sink lives for the server's lifetime;
                    // a `DiscardingMetricsSink` swallows the record
                    // when the user hasn't enabled local telemetry.
                    let metric = buildQwen3Metric(
                        audioMs: final.audioDurationMs,
                        fellBack: false,
                        errored: false
                    )
                    await metricsSink.record(metric)
                } catch Qwen3ASRStreamingSession.SessionError.empty {
                    // Client disconnected without sending audio. Emit
                    // an empty `final` for protocol symmetry.
                    await sendResult(WSSTTResult(
                        type: "final",
                        text: "",
                        finalized: "",
                        draft: ""
                    ))
                } catch let sessionError as Qwen3ASRStreamingSession.SessionError {
                    sttLogger.error(
                        "Qwen3 streaming finalize failed: \(sessionError.description)"
                    )
                    await sendError(
                        "Streaming finalize failed: \(sessionError.description)",
                        code: .finalizeFailed
                    )
                    let metric = buildQwen3Metric(
                        audioMs: await session.audioDurationMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                } catch {
                    sttLogger.error(
                        "Qwen3 streaming finalize failed: \(error.localizedDescription)"
                    )
                    await sendError(
                        "Streaming finalize failed: \(error.localizedDescription)",
                        code: .finalizeFailed
                    )
                    let metric = buildQwen3Metric(
                        audioMs: await session.audioDurationMs,
                        fellBack: false,
                        errored: true
                    )
                    await metricsSink.record(metric)
                }
            } else if let transcriber = transcriber {
                let finalResult = transcriber.finalize()
                await sendResult(WSSTTResult(
                    type: "final",
                    text: finalResult.text,
                    finalized: finalResult.finalized,
                    draft: finalResult.draft
                ))
            }

            sttLogger.info("STT WebSocket client disconnected")
        }

        return wsRouter
    }
}

enum ServerStartError: LocalizedError {
    case failedToBind(String)
    case healthCheckFailed

    var errorDescription: String? {
        switch self {
        case .failedToBind(let reason):
            return "Failed to bind server: \(reason)"
        case .healthCheckFailed:
            return "Server started but health check failed"
        }
    }
}

enum STTRequestError: Error {
    case invalidLanguage(String)
    case notImplemented(String)

    var message: String {
        switch self {
        case .invalidLanguage(let code): return "Unknown language: \(code)"
        case .notImplemented(let name): return "Language not implemented: \(name)"
        }
    }

    var code: String {
        switch self {
        case .invalidLanguage: return "invalid_language"
        case .notImplemented: return "not_implemented"
        }
    }
}
