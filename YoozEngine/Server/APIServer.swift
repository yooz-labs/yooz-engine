import Foundation
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

    private func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        let sttEngine = YoozSTTEngine.shared

        // Health
        router.get("/v1/health") { _, _ in
            let llmLoaded = await TouchUpEngine.shared.isLightModelLoaded
            let touchupReady = await TouchUpEngine.shared.isPreloaded
            let vadLoaded = await VADEngine.shared.isLoaded
            return HealthResponse(
                status: "ok",
                version: EngineConfig.version,
                modules: EngineModules(
                    stt: sttEngine.isRunning,
                    llm: llmLoaded,
                    touchup: touchupReady,
                    grammar: GrammarEngine.shared.isAvailable,
                    vad: vadLoaded,
                    tts: false
                )
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
                    modelType: modelType
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

            let result = await TouchUpEngine.shared.process(
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

            do {
                let result = await GrammarEngine.shared.check(
                    text: body.text,
                    categories: body.categories
                )
                return try jsonResponse(GrammarCheckServerResponse(
                    result: result.result,
                    correctionsApplied: result.correctionsApplied
                ))
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "grammar_check_failed"
                )
            }
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
                let segments = try await VADEngine.shared.detect(samples: body.samples)
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
            let running = sttEngine.isRunning
            return STTStatusResponse(
                loaded: running,
                language: running ? sttEngine.currentLanguage.rawValue : nil,
                streaming: sttEngine.isStreaming
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

            do {
                try await sttEngine.start(language: language)
                return try jsonResponse(STTStatusResponse(
                    loaded: true,
                    language: language.rawValue,
                    streaming: false
                ))
            } catch {
                return errorResponse(
                    status: .internalServerError,
                    message: error.localizedDescription,
                    code: "load_failed"
                )
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

            let mode = AudioMode(rawValue: body.mode ?? "normal") ?? .normal
            let result = await sttEngine.batchTranscribe(samples: body.samples, mode: mode)

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

        wsRouter.ws("/v1/stt/stream") { inbound, outbound, context in
            let sttEngine = YoozSTTEngine.shared
            var transcriber: StreamingTranscriber?
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

            // Send a JSON-encoded error message to the client
            func sendError(_ message: String) async {
                do {
                    let json = try encoder.encode(WSSTTError(type: "error", message: message))
                    guard let jsonStr = String(data: json, encoding: .utf8) else { return }
                    try await outbound.write(.text(jsonStr))
                } catch {
                    sttLogger.error("Failed to send error message: \(error)")
                }
            }

            sttLogger.info("STT WebSocket client connected")

            // Use message-level API to handle fragmented frames automatically
            // Max message size: 10MB (enough for ~2.7 min of 16kHz Float32 audio)
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
                        await sendError("Invalid message format")
                        continue
                    }

                    if config.type == "config" {
                        let languageCode = config.language ?? "en"
                        guard let language = STTLanguage.fromCode(languageCode) else {
                            await sendError("Unknown language: \(languageCode)")
                            continue
                        }

                        guard language.isImplemented else {
                            await sendError("Language not implemented: \(language.displayName)")
                            continue
                        }

                        do {
                            try await sttEngine.start(language: language)
                        } catch {
                            await sendError("Failed to load model: \(error.localizedDescription)")
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
                    // Audio data: raw Float32 samples at 16kHz
                    guard let transcriber = transcriber else {
                        sttLogger.warning("Audio received before config message")
                        continue
                    }

                    let byteCount = buffer.readableBytes
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

                    let result = transcriber.addAudio(samples: samples)

                    await sendResult(WSSTTResult(
                        type: "partial",
                        text: result.text,
                        finalized: result.finalized,
                        draft: result.draft
                    ))
                }
            }

            // Client disconnected; finalize if a transcriber is active
            if let transcriber = transcriber {
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
