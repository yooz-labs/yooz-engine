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
    /// Client-exported callback router the service pushes streaming results to.
    private let streamClient = XPCStreamClient()

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
        // Bidirectional: export the streaming callback so the service can push
        // results back (harmless for the request-only path). Fail any in-flight
        // streams if the connection drops.
        connection.exportedInterface = NSXPCInterface(with: YoozEngineXPCStreamClientProtocol.self)
        let streamClient = self.streamClient
        connection.exportedObject = streamClient
        connection.interruptionHandler = { streamClient.finishAll(error: YoozEngineError.engineNotReachable) }
        connection.invalidationHandler = { streamClient.finishAll(error: YoozEngineError.engineNotReachable) }
        self.connection = connection
        connection.resume()
    }

    deinit {
        // Release the Mach port pair (and let a system-launched service idle out)
        // when the transport is torn down.
        connection.invalidate()
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
        // Generate the id and register the receiving session BEFORE telling the
        // service to open, so an immediate partial from a fast backend routes
        // even if it arrives before this call returns.
        let streamID = UUID().uuidString
        let session = XPCSTTStreamSession(streamID: streamID, connection: connection)
        streamClient.register(session, for: streamID)
        do {
            try await openStream(streamID: streamID, language: language, mode: mode)
        } catch {
            streamClient.unregister(streamID)
            throw error
        }
        return session
    }

    private func openStream(streamID: String, language: String, mode: String) async throws {
        let state = SendState<Void>()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                guard state.register(continuation) else { return }
                let proxy = connection.remoteObjectProxyWithErrorHandler { error in
                    state.resume(.failure(XPCErrorBridge.toYoozEngineError(error)))
                }
                guard let service = proxy as? YoozEngineXPCProtocol else {
                    state.resume(.failure(YoozEngineError.engineNotReachable))
                    return
                }
                service.openSTTStream(streamID: streamID, language: language, mode: mode) { error in
                    if let error {
                        state.resume(.failure(XPCErrorBridge.toYoozEngineError(error)))
                    } else {
                        state.resume(.success(()))
                    }
                }
            }
        } onCancel: {
            state.cancel()
        }
    }

    private func send(_ method: String, _ path: String, _ body: Data?) async throws -> Data {
        // `SendState` resumes the continuation exactly once across all paths:
        // the proxy error handler, the reply block, AND task cancellation (which
        // NSXPCConnection does not surface on its own — without this a cancelled
        // caller would leak the continuation, trapping on dealloc).
        let state = SendState<Data>()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data, Error>) in
                guard state.register(continuation) else {
                    return  // already cancelled before we registered — resolved.
                }
                let proxy = connection.remoteObjectProxyWithErrorHandler { error in
                    state.resume(.failure(XPCErrorBridge.toYoozEngineError(error)))
                }
                guard let service = proxy as? YoozEngineXPCProtocol else {
                    state.resume(.failure(YoozEngineError.engineNotReachable))
                    return
                }
                service.request(method: method, path: path, body: body) { data, error in
                    if let error {
                        state.resume(.failure(XPCErrorBridge.toYoozEngineError(error)))
                    } else if let data {
                        state.resume(.success(data))
                    } else {
                        // Protocol violation: reply(nil, nil). Surface it, never a
                        // plausible-but-wrong empty success.
                        state.resume(.failure(YoozEngineError.invalidResponse))
                    }
                }
            }
        } onCancel: {
            state.cancel()
        }
    }
}

/// Resolves a single XPC request's continuation exactly once, whether the
/// outcome arrives from XPC (reply / error handler) or from task cancellation.
private final class SendState<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Value, Error>?
    private var resumed = false
    private var cancelledEarly = false

    /// Register the continuation. Returns false (and resumes with
    /// `CancellationError`) if cancellation already fired before registration.
    func register(_ continuation: CheckedContinuation<Value, Error>) -> Bool {
        lock.lock()
        if cancelledEarly {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return false
        }
        self.continuation = continuation
        lock.unlock()
        return true
    }

    func resume(_ result: Result<Value, Error>) {
        lock.lock()
        guard !resumed, let continuation else { lock.unlock(); return }
        resumed = true
        self.continuation = nil
        lock.unlock()
        continuation.resume(with: result)
    }

    func cancel() {
        lock.lock()
        if resumed { lock.unlock(); return }
        if let continuation {
            resumed = true
            self.continuation = nil
            lock.unlock()
            continuation.resume(throwing: CancellationError())
        } else {
            // Cancellation raced ahead of registration; register() will resolve.
            cancelledEarly = true
            lock.unlock()
        }
    }
}
