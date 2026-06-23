import Foundation

/// Transport seam behind the `YoozEngineClient` SDK surface.
///
/// The SDK is identical regardless of how it reaches the engine. Three
/// transports are envisaged (epic #192):
///
///   - **HTTP/WebSocket loopback** (`HTTPTransport`, the default): talks to a
///     helper process over `127.0.0.1:<port>`. This is the super-yooz host +
///     local-dev packaging.
///   - **In-process** (`InProcessTransport`, the `YoozEngineInProcess` product):
///     calls the engine module actors (`*Engine.shared`) directly, with no
///     socket. This is the App Store standalone packaging.
///   - **XPC** (Phase 3): a sandboxed XPC service.
///
/// The sub-clients (`STTClient`, `LLMClient`, …) issue requests as
/// `(method, path, JSON body) -> JSON Data`, decoding the same client-local
/// wire DTOs in `Types/`. A transport therefore only has to move bytes for a
/// path; it does not need to know the per-endpoint shapes. This byte-level
/// seam maps cleanly onto all three transports (HTTP request, in-process
/// route dispatch, XPC message) and keeps the sub-client surface unchanged.
public protocol EngineTransport: Sendable {
    /// Loopback base URL. Real for `HTTPTransport`; a non-routable placeholder
    /// for transports without a socket (in-process / XPC). Exposed because it
    /// is part of the long-standing public `YoozEngineClient` surface.
    var baseURL: URL { get }

    /// Loopback port. `0` for transports without a socket.
    var port: Int { get }

    /// Bring the transport to a ready state.
    ///
    /// - HTTP: probe `/v1/health`, launching the bundled helper if needed.
    /// - In-process: register + bootstrap the linked engine modules.
    func connect() async throws

    /// Whether the engine is currently reachable/ready.
    func isReachable() async throws -> Bool

    /// `GET <path>` returning the raw response body.
    func get(_ path: String) async throws -> Data

    /// `POST <path>` with a JSON body, returning the raw response body.
    func post(_ path: String, body: Data) async throws -> Data

    /// `DELETE <path>` returning the raw response body.
    func delete(_ path: String) async throws -> Data

    /// WebSocket URL for a streaming endpoint (e.g. `v1/stt/stream`).
    ///
    /// `HTTPTransport` returns a `ws://` URL. Transports without a loopback
    /// socket throw `YoozEngineError.unsupportedInProcess` until the
    /// in-process streaming path lands (epic #192 Phase 2b).
    @available(macOS 14.0, iOS 17.0, *)
    func webSocketURL(path: String) throws -> URL
}
