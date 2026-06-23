import YoozEngineClient

extension YoozEngineClient {
    /// Construct a client that runs the engine **in-process** — the linked
    /// module actors are called directly, with no loopback socket (epic #192).
    ///
    /// ```swift
    /// let client = YoozEngineClient.inProcess()
    /// try await client.connect()              // registers the modules
    /// let corrected = try await client.grammar.correct(text: "...")
    /// ```
    public static func inProcess(host: EngineInProcessHost = .shared) -> YoozEngineClient {
        YoozEngineClient(transport: InProcessTransport(host: host))
    }
}
