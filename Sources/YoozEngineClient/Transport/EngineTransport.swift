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

    /// Open a streaming STT session for `language` / `mode`.
    ///
    /// `HTTPTransport` performs the WebSocket config/ready handshake and returns
    /// a WebSocket-backed session. `InProcessTransport` sets up an engine
    /// `StreamingTranscriber` / Qwen3 session / Apple buffer and returns an
    /// in-process session. `language` / `mode` are raw wire values
    /// (`STTLanguage.rawValue` / `AudioMode.rawValue`).
    @available(macOS 14.0, iOS 17.0, *)
    func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession

    /// Open the `/v1/events` push channel (engine#226): a live feed of
    /// `EngineEvent`s (model-selection changes, load-state transitions,
    /// download progress, residency changes) across every module the
    /// engine hosts. One-directional (server → client) — unlike
    /// `openSTTStream` there is nothing for the caller to send back, so the
    /// return type is a plain `AsyncStream` rather than a session protocol.
    ///
    /// - HTTP: opens a WebSocket at `/v1/events` and decodes each frame.
    /// - In-process: subscribes directly to the shared `EngineEventBus`.
    /// - XPC: bridges the push channel over the connection's callback proxy
    ///   (engine#244) — see `XPCTransport.openEvents()` for the
    ///   finishes-on-connection-death contract specific to that transport.
    @available(macOS 14.0, iOS 17.0, *)
    func openEvents() async throws -> AsyncStream<EngineEvent>
}
