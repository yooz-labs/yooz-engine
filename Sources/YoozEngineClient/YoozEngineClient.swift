import Foundation

/// Thin client SDK for communicating with the Yooz Engine service.
///
/// The same surface regardless of transport (epic #192). By default the client
/// talks to a helper over an HTTP/WebSocket loopback socket; a host that links
/// the engine modules in-process constructs the client with an in-process
/// transport (the `YoozEngineInProcess` product) so app code is identical:
///
/// ```swift
/// // Loopback (default) — launches/relays to a helper on 127.0.0.1:<port>.
/// let client = YoozEngineClient()
/// try await client.connect()
/// let health = try await client.health()
///
/// // In-process — calls the engine actors directly, no socket.
/// let client = YoozEngineClient(transport: InProcessTransport())
/// ```
public final class YoozEngineClient: Sendable {
    /// The active transport. `HTTPTransport` by default; an in-process / XPC
    /// transport when constructed via `init(transport:)`.
    public let transport: EngineTransport

    /// Loopback base URL (HTTP transport). A non-routable placeholder for
    /// transports without a socket.
    public var baseURL: URL { transport.baseURL }

    /// Loopback port (HTTP transport). `0` for transports without a socket.
    public var port: Int { transport.port }

    /// Construct a loopback (HTTP/WebSocket) client. Unchanged behavior for
    /// existing consumers: builds an `HTTPTransport` on `host:port`.
    public init(
        host: String = "127.0.0.1",
        port: Int = 19920
    ) {
        self.transport = HTTPTransport(host: host, port: port)
    }

    /// Construct a client over an explicit transport (in-process / XPC).
    public init(transport: EngineTransport) {
        self.transport = transport
    }

    /// Bring the transport to a ready state.
    ///
    /// - Loopback: probe `/v1/health`, launching the bundled helper if needed.
    /// - In-process: register + bootstrap the linked engine modules.
    public func connect() async throws {
        try await transport.connect()
    }

    /// Whether the engine is currently reachable/ready.
    public func isReachable() async throws -> Bool {
        try await transport.isReachable()
    }

    /// Get engine health status.
    public func health() async throws -> HealthStatus {
        let data = try await transport.get("/v1/health")
        return try JSONDecoder().decode(HealthStatus.self, from: data)
    }

    /// Get the full module manifest: build variant, engine version, and every
    /// registered module with its current health. Thin clients read this to
    /// render "About" panels and to decide which endpoints are served by the
    /// running build variant. See `Types/ModulesResponse.swift`.
    public func modules() async throws -> ModulesResponse {
        let data = try await transport.get("/v1/modules")
        return try JSONDecoder().decode(ModulesResponse.self, from: data)
    }

    // MARK: - Service clients

    /// STT (speech-to-text) service client.
    public var stt: STTClient { STTClient(engine: self) }

    /// LLM generation service client.
    public var llm: LLMClient { LLMClient(engine: self) }

    /// Touch-up (text cleanup) service client.
    public var touchUp: TouchUpClient { TouchUpClient(engine: self) }

    /// Infinite long-context service client.
    public var infinite: InfiniteClient { InfiniteClient(engine: self) }

    /// Grammar check service client.
    public var grammar: GrammarClient { GrammarClient(engine: self) }

    /// VAD (voice activity detection) service client.
    public var vad: VADClient { VADClient(engine: self) }

    // MARK: - Transport delegation (used by the service clients)

    func get(_ path: String) async throws -> Data {
        try await transport.get(path)
    }

    func post(_ path: String, body: Data) async throws -> Data {
        try await transport.post(path, body: body)
    }

    func delete(_ path: String) async throws -> Data {
        try await transport.delete(path)
    }

    @available(macOS 14.0, iOS 17.0, *)
    func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        try await transport.openSTTStream(language: language, mode: mode)
    }
}
