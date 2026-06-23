import Foundation

/// `EngineTransport` over `NSXPCConnection` — the App Store standalone packaging
/// (epic #192 Phase 3). The engine runs in a sandboxed XPC service under
/// `Contents/XPCServices/`; the app addresses it by service name (no ports, no
/// loopback, Apple's sanctioned in-bundle IPC).
///
/// Byte-level: `get/post/delete` funnel through the single `request` XPC method.
/// Streaming STT is Phase 3b (it needs a bidirectional callback proxy, not a
/// once-fired reply block) — `openSTTStream` reports `unsupportedOperation` until then.
public final class XPCTransport: EngineTransport, @unchecked Sendable {
    /// Non-routable placeholder — XPC addresses by service name, not URL/port.
    public let baseURL = URL(string: "xpc://engine")!
    public let port = 0

    private let connection: NSXPCConnection

    /// Connect to a sandboxed XPC service by name
    /// (`Contents/XPCServices/<serviceName>.xpc`). `codeSigningRequirement`, when
    /// set, pins the peer's code signature (macOS 13+) — the first-class trust
    /// check loopback has no equivalent for.
    public convenience init(serviceName: String, codeSigningRequirement: String? = nil) {
        let connection = NSXPCConnection(serviceName: serviceName)
        if let codeSigningRequirement, #available(macOS 13.0, *) {
            connection.setCodeSigningRequirement(codeSigningRequirement)
        }
        self.init(connection: connection)
    }

    /// Wrap an already-configured connection. Used by the anonymous-listener
    /// tests and by callers that build the connection themselves.
    public init(connection: NSXPCConnection) {
        connection.remoteObjectInterface = NSXPCInterface(with: YoozEngineXPCProtocol.self)
        self.connection = connection
        connection.resume()
    }

    public func connect() async throws {
        // Confirm the service is reachable (and decodes our health contract).
        _ = try await get("/v1/health")
    }

    public func isReachable() async throws -> Bool {
        do {
            _ = try await get("/v1/health")
            return true
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return false
        }
    }

    public func get(_ path: String) async throws -> Data { try await send("GET", path, nil) }

    public func post(_ path: String, body: Data) async throws -> Data { try await send("POST", path, body) }

    public func delete(_ path: String) async throws -> Data { try await send("DELETE", path, nil) }

    @available(macOS 14.0, iOS 17.0, *)
    public func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        throw YoozEngineError.unsupportedOperation(operation: "XPC streaming STT (epic #192 Phase 3b)")
    }

    private func send(_ method: String, _ path: String, _ body: Data?) async throws -> Data {
        // Guard against the (rare) case where both the proxy error handler and the
        // reply fire — a CheckedContinuation traps on double resume.
        let resumed = ResumeOnce()
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data, Error>) in
            let proxy = connection.remoteObjectProxyWithErrorHandler { error in
                if resumed.claim() {
                    continuation.resume(throwing: XPCErrorBridge.toYoozEngineError(error))
                }
            }
            guard let service = proxy as? YoozEngineXPCProtocol else {
                if resumed.claim() {
                    continuation.resume(throwing: YoozEngineError.engineNotReachable)
                }
                return
            }
            service.request(method: method, path: path, body: body) { data, error in
                guard resumed.claim() else { return }
                if let error {
                    continuation.resume(throwing: XPCErrorBridge.toYoozEngineError(error))
                } else {
                    continuation.resume(returning: data ?? Data())
                }
            }
        }
    }
}

/// One-shot latch so a continuation is resumed exactly once.
private final class ResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if done { return false }
        done = true
        return true
    }
}
