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
        language: String,
        mode: String,
        withReply reply: @escaping (String?, Error?) -> Void
    ) {
        // Capture the connection now — `NSXPCConnection.current()` is only valid
        // inside an incoming call, not later from the drain task.
        guard let connection = NSXPCConnection.current() else {
            reply(nil, XPCErrorBridge.toNSError(YoozEngineError.engineNotReachable))
            return
        }
        let transport = self.transport
        Task { [weak self] in
            do {
                let session = try await transport.openSTTStream(language: language, mode: mode)
                let streamID = UUID().uuidString
                guard let self else {
                    session.close()
                    reply(nil, XPCErrorBridge.toNSError(YoozEngineError.engineNotReachable))
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
                reply(streamID, nil)
            } catch {
                reply(nil, XPCErrorBridge.toNSError(error))
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
                // A send error surfaces to the client through the drain's
                // receive() path when the session finalizes; log it here so a
                // mid-stream feed failure is observable.
                logger.debug("xpc: sendAudio(\(streamID, privacy: .public)) failed: \(error.localizedDescription, privacy: .public)")
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

    /// Drains a session's results to the client callback over `connection`.
    private func drain(streamID: String, session: any STTStreamSession, connection: NSXPCConnection) async {
        let callback = connection.remoteObjectProxy as? YoozEngineXPCStreamClientProtocol
        do {
            while let result = try await session.receive() {
                let data = (try? JSONEncoder().encode(result)) ?? Data()
                callback?.streamDidProduce(streamID: streamID, resultData: data)
            }
            callback?.streamDidFinish(streamID: streamID, error: nil)
        } catch {
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
        newConnection.exportedInterface = NSXPCInterface(with: YoozEngineXPCProtocol.self)
        newConnection.exportedObject = makeHandler()
        // The service pushes streaming results back to the client's exported
        // callback object, so the connection's remote interface must be the
        // client-callback protocol. Without this, `connection.remoteObjectProxy`
        // on the service side resolves to nothing and streaming callbacks are
        // dropped (the client's receive() would hang).
        newConnection.remoteObjectInterface = NSXPCInterface(with: YoozEngineXPCStreamClientProtocol.self)
        // Surface lifecycle events so a mid-request peer drop isn't silent — any
        // engine work already in flight on the service side runs to completion
        // but with no one waiting; logging makes that observable.
        let logger = self.logger
        newConnection.interruptionHandler = {
            logger.debug("xpc: connection interrupted (peer crashed or restarted)")
        }
        newConnection.invalidationHandler = {
            logger.debug("xpc: connection invalidated")
        }
        newConnection.resume()
        return true
    }
}
