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

        serverTask?.cancel()
        _ = await serverTask?.result
        serverTask = nil
        app = nil

        state = .stopped
        logger.info("Yooz Engine stopped")
    }

    // MARK: - HTTP Router

    private func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        let sttEngine = YoozSTTEngine.shared

        // Health
        router.get("/v1/health") { _, _ in
            HealthResponse(
                status: "ok",
                version: EngineConfig.version,
                modules: EngineModules(
                    stt: sttEngine.isRunning,
                    llm: false,
                    touchup: false,
                    grammar: false,
                    vad: false,
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
            return ModelsResponse(models: models)
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
            STTStatusResponse(
                loaded: sttEngine.isRunning,
                language: sttEngine.isRunning ? sttEngine.currentLanguage.rawValue : nil,
                streaming: sttEngine.isStreaming
            )
        }

        // STT: Load model
        router.post("/v1/stt/load") { request, context in
            let body = try await request.decode(as: STTLoadRequest.self, context: context)
            let languageCode = body.language ?? "en"

            guard let language = STTLanguage.fromCode(languageCode) else {
                return Response(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: ByteBuffer(string: "{\"error\":\"Unknown language: \(languageCode)\",\"code\":\"invalid_language\"}"))
                )
            }

            guard language.isImplemented else {
                return Response(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: ByteBuffer(string: "{\"error\":\"Language not implemented: \(language.displayName)\",\"code\":\"not_implemented\"}"))
                )
            }

            do {
                try await sttEngine.start(language: language)
                let responseData = try JSONEncoder().encode(STTStatusResponse(
                    loaded: true,
                    language: language.rawValue,
                    streaming: false
                ))
                return Response(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: ByteBuffer(data: responseData))
                )
            } catch {
                return Response(
                    status: .internalServerError,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: ByteBuffer(string: "{\"error\":\"\(error.localizedDescription)\",\"code\":\"load_failed\"}"))
                )
            }
        }

        // STT: Batch transcribe
        router.post("/v1/stt/batch") { request, context in
            let body = try await request.decode(as: BatchSTTRequest.self, context: context)

            let languageCode = body.language ?? "en"
            guard let language = STTLanguage.fromCode(languageCode) else {
                return Response(
                    status: .badRequest,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: ByteBuffer(string: "{\"error\":\"Unknown language: \(languageCode)\",\"code\":\"invalid_language\"}"))
                )
            }

            // Ensure model is loaded for the requested language
            do {
                try await sttEngine.start(language: language)
            } catch {
                return Response(
                    status: .internalServerError,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: ByteBuffer(string: "{\"error\":\"\(error.localizedDescription)\",\"code\":\"model_load_failed\"}"))
                )
            }

            let mode: AudioMode = (body.mode == "whispered") ? .whispered : .normal
            let result = await sttEngine.batchTranscribe(samples: body.samples, mode: mode)

            let responseData = try JSONEncoder().encode(BatchSTTResponse(
                text: result.text,
                finalized: result.finalized,
                draft: result.draft,
                language: language.rawValue
            ))
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: ByteBuffer(data: responseData))
            )
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

            sttLogger.info("STT WebSocket client connected")

            // Use message-level API to handle fragmented frames automatically
            // Max message size: 10MB (enough for ~5 min of 16kHz Float32 audio)
            for try await message in inbound.messages(maxSize: 10 * 1024 * 1024) {
                switch message {
                case .text(let text):
                    // Config message: {"type":"config","language":"en","mode":"normal"}
                    guard let data = text.data(using: .utf8),
                          let config = try? JSONDecoder().decode(WSSTTConfig.self, from: data) else {
                        sttLogger.warning("Invalid text message received")
                        continue
                    }

                    if config.type == "config" {
                        let languageCode = config.language ?? "en"
                        guard let language = STTLanguage.fromCode(languageCode) else {
                            try await outbound.write(.text("{\"type\":\"error\",\"message\":\"Unknown language: \(languageCode)\"}"))
                            continue
                        }

                        do {
                            try await sttEngine.start(language: language)
                        } catch {
                            try await outbound.write(.text("{\"type\":\"error\",\"message\":\"\(error.localizedDescription)\"}"))
                            continue
                        }

                        let mode: AudioMode = (config.mode == "whispered") ? .whispered : .normal
                        transcriber = sttEngine.createBatchTranscriber(mode: mode)

                        try await outbound.write(.text("{\"type\":\"ready\",\"language\":\"\(languageCode)\"}"))
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

                    let samples: [Float] = buffer.withUnsafeReadableBytes { ptr in
                        let floatPtr = ptr.baseAddress!.assumingMemoryBound(to: Float.self)
                        return Array(UnsafeBufferPointer(start: floatPtr, count: sampleCount))
                    }

                    let result = transcriber.addAudio(samples: samples)

                    let wsResult = WSSTTResult(
                        type: "partial",
                        text: result.text,
                        finalized: result.finalized,
                        draft: result.draft
                    )
                    if let json = try? encoder.encode(wsResult),
                       let jsonStr = String(data: json, encoding: .utf8) {
                        try await outbound.write(.text(jsonStr))
                    }
                }
            }

            // Client disconnected; finalize if a transcriber is active
            if let transcriber = transcriber {
                let finalResult = transcriber.finalize()
                let wsResult = WSSTTResult(
                    type: "final",
                    text: finalResult.text,
                    finalized: finalResult.finalized,
                    draft: finalResult.draft
                )
                if let json = try? encoder.encode(wsResult),
                   let jsonStr = String(data: json, encoding: .utf8) {
                    try await outbound.write(.text(jsonStr))
                }
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
