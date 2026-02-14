import Foundation
import Hummingbird
import HummingbirdWebSocket
import Logging

@MainActor
final class APIServer: ObservableObject {
    @Published var isRunning = false

    private var app: (any ApplicationProtocol)?
    private var serverTask: Task<Void, any Error>?
    private let logger = Logger(label: "live.yooz.engine.server")

    func start() async throws {
        guard !isRunning else { return }

        let router = buildRouter()

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname(EngineConfig.host, port: EngineConfig.port),
                serverName: "YoozEngine/\(EngineConfig.version)"
            ),
            logger: logger
        )

        serverTask = Task {
            try await app.run()
        }

        self.app = app
        isRunning = true
        logger.info("Yooz Engine started on \(EngineConfig.host):\(EngineConfig.port)")
    }

    func stop() async {
        serverTask?.cancel()
        serverTask = nil
        app = nil
        isRunning = false
        logger.info("Yooz Engine stopped")
    }

    private func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()

        router.get("/v1/health") { _, _ in
            HealthResponse(
                status: "ok",
                version: EngineConfig.version,
                modules: EngineModules(
                    stt: false,
                    llm: false,
                    touchup: false,
                    grammar: false,
                    vad: false,
                    tts: false
                )
            )
        }

        router.get("/v1/models") { _, _ in
            ModelsResponse(models: [])
        }

        return router
    }
}
