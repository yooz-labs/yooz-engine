import Foundation
#if canImport(AppKit)
import AppKit
#endif

/// Thin client SDK for communicating with the Yooz Engine service.
///
/// Usage:
/// ```swift
/// let client = YoozEngineClient()
/// try await client.connect()
/// let health = try await client.health()
/// ```
public final class YoozEngineClient: @unchecked Sendable {
    public let baseURL: URL
    private let session: URLSession
    private let engineBundleID = "live.yooz.engine"

    public init(
        host: String = "127.0.0.1",
        port: Int = 19920
    ) {
        self.baseURL = URL(string: "http://\(host):\(port)")!

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        self.session = URLSession(configuration: config)
    }

    /// Check if the engine is reachable and launch it if not.
    public func connect() async throws {
        if await isReachable() { return }

        try launchEngine()

        // Wait for engine to become ready (up to 15 seconds)
        for _ in 0..<30 {
            try await Task.sleep(for: .milliseconds(500))
            if await isReachable() { return }
        }

        throw YoozEngineError.engineNotReachable
    }

    /// Check if the engine is currently reachable.
    public func isReachable() async -> Bool {
        do {
            let _ = try await health()
            return true
        } catch {
            return false
        }
    }

    /// Get engine health status.
    public func health() async throws -> HealthStatus {
        let data = try await get("/v1/health")
        return try JSONDecoder().decode(HealthStatus.self, from: data)
    }

    // MARK: - HTTP helpers

    func get(_ path: String) async throws -> Data {
        let url = baseURL.appendingPathComponent(path)
        let (data, response) = try await session.data(from: url)
        try validateResponse(response)
        return data
    }

    func post(_ path: String, body: Data) async throws -> Data {
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return data
    }

    private func validateResponse(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else {
            throw YoozEngineError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            throw YoozEngineError.httpError(statusCode: http.statusCode)
        }
    }

    // MARK: - Service clients

    /// STT (speech-to-text) service client.
    public var stt: STTClient { STTClient(engine: self) }

    /// LLM generation service client.
    public var llm: LLMClient { LLMClient(engine: self) }

    /// Touch-up (text cleanup) service client.
    public var touchUp: TouchUpClient { TouchUpClient(engine: self) }

    /// Grammar check service client.
    public var grammar: GrammarClient { GrammarClient(engine: self) }

    /// VAD (voice activity detection) service client.
    public var vad: VADClient { VADClient(engine: self) }

    // MARK: - Engine lifecycle

    private func launchEngine() throws {
        #if canImport(AppKit)
        let config = NSWorkspace.OpenConfiguration()
        config.activates = false

        guard let engineURL = NSWorkspace.shared.urlForApplication(
            withBundleIdentifier: engineBundleID
        ) else {
            throw YoozEngineError.engineNotInstalled
        }

        NSWorkspace.shared.openApplication(
            at: engineURL,
            configuration: config
        ) { _, error in
            if let error {
                print("[YoozEngineClient] Failed to launch engine: \(error)")
            }
        }
        #else
        throw YoozEngineError.engineNotInstalled
        #endif
    }
}
