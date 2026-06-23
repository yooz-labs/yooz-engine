import Foundation
import OSLog

/// Service-side XPC handler (epic #192 Phase 3): implements
/// `YoozEngineXPCProtocol` by delegating to an injected `EngineTransport` — the
/// consumer wires it with `InProcessTransport()` so the XPC service runs the
/// engine modules in its own sandboxed process.
///
/// Reusable across the real `.xpc` bundle and the anonymous-listener tests.
public final class XPCServiceHandler: NSObject, YoozEngineXPCProtocol, @unchecked Sendable {
    private let transport: any EngineTransport
    private let logger = Logger(subsystem: "live.yooz.engine.client", category: "xpc-service")

    // Active streaming sessions keyed by id. Each holds the session + the
    // connection to push results back on + the drain task feeding the callback.
    private struct StreamEntry {
        let session: any STTStreamSession
        let connection: NSXPCConnection
        let drainTask: Task<Void, Never>
    }
    private let streamLock = NSLock()
    private var streams: [String: StreamEntry] = [:]

    public init(transport: any EngineTransport) {
        self.transport = transport
        super.init()
    }

    public func request(
        method: String,
        path: String,
        body: Data?,
        withReply reply: @escaping (Data?, Error?) -> Void
    ) {
        let transport = self.transport
        Task {
            do {
                let data: Data
                switch method {
                case "GET":
                    data = try await transport.get(path)
                case "POST":
                    data = try await transport.post(path, body: body ?? Data())
                case "DELETE":
                    data = try await transport.delete(path)
                default:
                    throw YoozEngineError.serverError(
                        statusCode: 405, code: "method_not_allowed",
                        message: "Unsupported method '\(method)'"
                    )
                }
                reply(data, nil)
            } catch {
                reply(nil, XPCErrorBridge.toNSError(error))
            }
        }
    }

    // MARK: - Streaming (epic #192 Phase 3b)

    public func openSTTStream(
        streamID: String,
        language: String,
        mode: String,
        withReply reply: @escaping (Error?) -> Void
    ) {
        // Capture the connection now — `NSXPCConnection.current()` is only valid
        // inside an incoming call, not later from the drain task.
        guard let connection = NSXPCConnection.current() else {
            reply(XPCErrorBridge.toNSError(YoozEngineError.engineNotReachable))
            return
        }
        let transport = self.transport
        Task { [weak self] in
            do {
                let session = try await transport.openSTTStream(language: language, mode: mode)
                guard let self else {
                    session.close()
                    reply(XPCErrorBridge.toNSError(YoozEngineError.engineNotReachable))
                    return
                }
                let drainTask = Task { [weak self] in
                    guard let self else { return }
                    await self.drain(streamID: streamID, session: session, connection: connection)
                }
                self.addStream(
                    StreamEntry(session: session, connection: connection, drainTask: drainTask),
                    id: streamID
                )
                reply(nil)
            } catch {
                reply(XPCErrorBridge.toNSError(error))
            }
        }
    }

    public func sendAudio(streamID: String, data: Data) {
        streamLock.lock()
        let session = streams[streamID]?.session
        streamLock.unlock()
        guard let session else { return }
        let samples = Self.floats(from: data)
        Task { [logger] in
            do {
                try await session.sendAudio(samples)
            } catch {
                // A feed failure must end the stream rather than silently drop
                // audio: close the session so the drain delivers a terminal
                // result instead of leaving a ghost stream.
                logger.error("xpc: sendAudio(\(streamID, privacy: .public)) failed — closing stream: \(error.localizedDescription, privacy: .public)")
                session.close()
            }
        }
    }

    public func closeStream(streamID: String) {
        streamLock.lock()
        let session = streams[streamID]?.session
        streamLock.unlock()
        // close() makes the session finalize; the drain loop then sees the final
        // result + nil, pushes streamDidFinish, and removes the entry.
        session?.close()
    }

    /// Cancel + close every active stream — called when the client connection
    /// drops, so orphaned drain tasks and engine sessions don't leak.
    public func cancelAllStreams() {
        streamLock.lock()
        let entries = streams
        streams.removeAll()
        streamLock.unlock()
        for entry in entries.values {
            entry.drainTask.cancel()
            entry.session.close()
        }
    }

    /// Drains a session's results to the client callback over `connection`.
    private func drain(streamID: String, session: any STTStreamSession, connection: NSXPCConnection) async {
        let encoder = JSONEncoder()
        // Error handler: if the callback connection fails, stop draining and
        // release the session (otherwise the drain runs to completion pushing
        // results into a dead connection).
        let proxy = connection.remoteObjectProxyWithErrorHandler { [weak self] error in
            self?.logger.error("xpc: stream \(streamID, privacy: .public) callback failed: \(error.localizedDescription, privacy: .public)")
            self?.cancelAndRemoveStream(streamID)
        }
        let callback = proxy as? YoozEngineXPCStreamClientProtocol
        do {
            while let result = try await session.receive() {
                // A `StreamingSTTResult` is plain Codable strings; let an encode
                // failure surface as a stream error rather than pushing empty Data
                // (which the client would misreport as a decode error).
                let data = try encoder.encode(result)
                callback?.streamDidProduce(streamID: streamID, resultData: data)
            }
            callback?.streamDidFinish(streamID: streamID, error: nil)
        } catch {
            logger.error("xpc: drain \(streamID, privacy: .public) error: \(error.localizedDescription, privacy: .public)")
            callback?.streamDidFinish(streamID: streamID, error: XPCErrorBridge.toNSError(error))
        }
        removeStream(streamID)
    }

    private func addStream(_ entry: StreamEntry, id: String) {
        streamLock.lock()
        streams[id] = entry
        streamLock.unlock()
    }

    private func removeStream(_ streamID: String) {
        streamLock.lock()
        streams[streamID] = nil
        streamLock.unlock()
    }

    private func cancelAndRemoveStream(_ streamID: String) {
        streamLock.lock()
        let entry = streams[streamID]
        streams[streamID] = nil
        streamLock.unlock()
        entry?.drainTask.cancel()
        entry?.session.close()
    }

    /// Decode Float32 little-endian PCM bytes into `[Float]` (alignment-safe).
    private static func floats(from data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Float>.size
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            [Float](unsafeUninitializedCapacity: count) { dest, initialized in
                UnsafeMutableRawBufferPointer(dest).copyBytes(
                    from: UnsafeRawBufferPointer(raw).prefix(count * MemoryLayout<Float>.size)
                )
                initialized = count
            }
        }
    }
}

/// `NSXPCListenerDelegate` that exports a fresh `XPCServiceHandler` per
/// connection. The `.xpc` bundle's `main` is a few lines around it:
///
/// ```swift
/// import YoozEngineClient
/// import YoozEngineInProcess
///
/// let delegate = XPCServiceListenerDelegate {
///     XPCServiceHandler(transport: InProcessTransport())
/// }
/// let listener = NSXPCListener.service()   // system-launched service
/// listener.delegate = delegate
/// listener.resume()
/// ```
///
/// (Building the `.xpc` bundle itself — Info.plist, entitlements, the
/// team-ID-prefixed app group for the shared model cache — is app/xcodegen work;
/// see `docs/engine-app-packaging.md`.)
public final class XPCServiceListenerDelegate: NSObject, NSXPCListenerDelegate, @unchecked Sendable {
    private let makeHandler: @Sendable () -> XPCServiceHandler
    private let logger = Logger(subsystem: "live.yooz.engine.client", category: "xpc-service")

    public init(makeHandler: @escaping @Sendable () -> XPCServiceHandler) {
        self.makeHandler = makeHandler
        super.init()
    }

    public func listener(
        _ listener: NSXPCListener,
        shouldAcceptNewConnection newConnection: NSXPCConnection
    ) -> Bool {
        let handler = makeHandler()
        newConnection.exportedInterface = NSXPCInterface(with: YoozEngineXPCProtocol.self)
        newConnection.exportedObject = handler
        // The service pushes streaming results back to the client's exported
        // callback object, so the connection's remote interface must be the
        // client-callback protocol. Without this, `connection.remoteObjectProxy`
        // on the service side resolves to nothing and streaming callbacks are
        // dropped (the client's receive() would hang).
        newConnection.remoteObjectInterface = NSXPCInterface(with: YoozEngineXPCStreamClientProtocol.self)
        // On peer drop, cancel + close the handler's active streams so orphaned
        // drain tasks and engine sessions don't leak (and log the event).
        let logger = self.logger
        newConnection.interruptionHandler = {
            logger.debug("xpc: connection interrupted (peer crashed or restarted)")
            handler.cancelAllStreams()
        }
        newConnection.invalidationHandler = {
            logger.debug("xpc: connection invalidated")
            handler.cancelAllStreams()
        }
        newConnection.resume()
        return true
    }
}
