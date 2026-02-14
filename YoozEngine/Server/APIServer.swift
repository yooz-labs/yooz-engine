import Foundation
import Hummingbird
import HummingbirdWebSocket
import Logging

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

        let router = buildRouter()

        let app = Application(
            router: router,
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

        serverTask?.cancel()
        // Wait for the task to finish
        _ = await serverTask?.result
        serverTask = nil
        app = nil

        state = .stopped
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
