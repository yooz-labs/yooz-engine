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

    /// Request-forensics logger (yooz-labs/yooz-whisper#280): every unary
    /// `request(method:path:body:)` call is logged at entry + exit with byte
    /// counts and elapsed time. Separate from `logger` above (a different
    /// subsystem/category on purpose) so `log stream --predicate
    /// 'subsystem == "live.yooz.engine"'` isolates this forensic stream from
    /// the connection-lifecycle logging `logger` already carries. This must
    /// work in Release builds — no `#if DEBUG` gating — because the service's
    /// stdout `print()` output is unrecoverable once launchd-managed (the
    /// blindness that made #280 nearly undebuggable in the first place).
    private let requestLogger = Logger(subsystem: "live.yooz.engine", category: "xpc")
    private let signposter = OSSignposter(subsystem: "live.yooz.engine", category: "xpc")

    // Active streaming sessions keyed by id. Each holds the session + the
    // connection to push results back on + the drain task feeding the callback.
    private struct StreamEntry {
        let session: any STTStreamSession
        let connection: NSXPCConnection
        let drainTask: Task<Void, Never>
    }
    private let streamLock = NSLock()
    private var streams: [String: StreamEntry] = [:]

    // Active `/v1/events` subscriptions keyed by client-generated
    // subscriptionID (engine#244). Separate dict + lock from `streams` above
    // — distinct concern (one bus subscription, not an `STTStreamSession`),
    // no reason to couple their locking.
    private struct EventEntry {
        let connection: NSXPCConnection
        let drainTask: Task<Void, Never>
    }
    private let eventLock = NSLock()
    private var eventSubscriptions: [String: EventEntry] = [:]

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
        let bodyByteCount = body?.count ?? 0
        let requestLogger = self.requestLogger
        let start = ContinuousClock.now
        let signpostID = signposter.makeSignpostID()
        let state = signposter.beginInterval(
            "xpc.request", id: signpostID, "\(method, privacy: .public) \(path, privacy: .public)"
        )
        requestLogger.log(
            "xpc.request.enter method=\(method, privacy: .public) path=\(path, privacy: .public) bodyBytes=\(bodyByteCount, privacy: .public)"
        )
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
                let elapsedMs = start.duration(to: .now).milliseconds
                requestLogger.log(
                    "xpc.request.exit method=\(method, privacy: .public) path=\(path, privacy: .public) responseBytes=\(data.count, privacy: .public) elapsedMs=\(elapsedMs, privacy: .public)"
                )
                signposter.endInterval("xpc.request", state, "ok")
                reply(data, nil)
            } catch {
                let elapsedMs = start.duration(to: .now).milliseconds
                requestLogger.error(
                    "xpc.request.error method=\(method, privacy: .public) path=\(path, privacy: .public) elapsedMs=\(elapsedMs, privacy: .public) error=\(String(describing: error), privacy: .public)"
                )
                signposter.endInterval("xpc.request", state, "error")
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

    // MARK: - Events (engine#244)

    public func openEvents(subscriptionID: String, withReply reply: @escaping (Error?) -> Void) {
        // Capture the connection now — `NSXPCConnection.current()` is only valid
        // inside an incoming call, not later from the drain task.
        guard let connection = NSXPCConnection.current() else {
            reply(XPCErrorBridge.toNSError(YoozEngineError.engineNotReachable))
            return
        }
        let transport = self.transport
        Task { [weak self] in
            do {
                let stream = try await transport.openEvents()
                guard let self else {
                    // Handler deallocated between the request arriving and the
                    // bus subscribing. Dropping `stream` here IS the cleanup:
                    // a plain `AsyncStream` has no close handle (unlike STT's
                    // `session.close()` in the analogous branch above), and
                    // deallocating an un-iterated stream fires its
                    // `onTermination`, which removes the `EngineEventBus`
                    // subscriber (see `EngineEventBus.subscribe()`'s doc).
                    reply(XPCErrorBridge.toNSError(YoozEngineError.engineNotReachable))
                    return
                }
                let drainTask = Task { [weak self] in
                    guard let self else { return }
                    await self.drainEvents(subscriptionID: subscriptionID, stream: stream, connection: connection)
                }
                self.addEventSubscription(
                    EventEntry(connection: connection, drainTask: drainTask), id: subscriptionID
                )
                reply(nil)
            } catch {
                reply(XPCErrorBridge.toNSError(error))
            }
        }
    }

    public func closeEvents(subscriptionID: String) {
        eventLock.lock()
        let entry = eventSubscriptions[subscriptionID]
        eventSubscriptions[subscriptionID] = nil
        eventLock.unlock()
        // Cancelling the drain task ends its `for await` over the bus
        // stream — `AsyncStream`'s cancellation-aware `next()` returns nil
        // as soon as the task is cancelled, which fires `EngineEventBus`'s
        // `onTermination` and releases the subscription. No separate
        // "unsubscribe" call to make (see `EngineEventBus.subscribe()`'s doc).
        entry?.drainTask.cancel()
    }

    /// Cancel + release every active event subscription — called when the
    /// client connection drops, so an orphaned drain task (and its
    /// `EngineEventBus` subscription) don't leak.
    public func cancelAllEventSubscriptions() {
        eventLock.lock()
        let entries = eventSubscriptions
        eventSubscriptions.removeAll()
        eventLock.unlock()
        for entry in entries.values {
            entry.drainTask.cancel()
        }
    }

    /// Drains an `/v1/events` subscription to the client callback over
    /// `connection`, keyed by `subscriptionID`. Mirrors
    /// `drain(streamID:session:connection:)` above, but one-directional and
    /// over a plain `AsyncStream` (no `STTStreamSession` to close) — see
    /// `EngineTransport.openEvents()`'s doc.
    private func drainEvents(subscriptionID: String, stream: AsyncStream<EngineEvent>, connection: NSXPCConnection) async {
        let encoder = JSONEncoder()
        let proxy = connection.remoteObjectProxyWithErrorHandler { [weak self] error in
            self?.logger.error("xpc: events \(subscriptionID, privacy: .public) callback failed: \(error.localizedDescription, privacy: .public)")
            self?.cancelAndRemoveEventSubscription(subscriptionID)
        }
        let callback = proxy as? YoozEngineXPCStreamClientProtocol
        // Carried into `eventsDidFinish` so the service's diagnosis of an
        // abnormal end (the encode failure below) crosses the process
        // boundary into the host app's log stream — client-side it is
        // log-only, never surfaced through the `AsyncStream` (PR #245
        // review; see the protocol doc on `eventsDidFinish`).
        var finishError: Error?
        for await event in stream {
            do {
                let data = try encoder.encode(event)
                callback?.eventDidOccur(subscriptionID: subscriptionID, eventData: data)
            } catch {
                // An encode failure ends the subscription rather than
                // silently dropping the frame — matches `drain`'s STT
                // decode-failure handling. Wrapped in a typed
                // `YoozEngineError` (NOT the raw `EncodingError`) because
                // `XPCErrorBridge.toNSError` only guarantees an
                // NSSecureCoding-safe userInfo (String/Int) for the typed
                // cases; a bridged Swift error crashes the XPC encoder
                // with `__SwiftValue` (caught by
                // `testServiceEncodeFailureFinishesClientStream`).
                logger.error("xpc: events \(subscriptionID, privacy: .public) encode failed: \(error.localizedDescription, privacy: .public)")
                finishError = YoozEngineError.decodingError(
                    "Event frame encode failed: \(error.localizedDescription)"
                )
                break
            }
        }
        // The loop above exits via cancellation (closeEvents / connection
        // death — both already tore down the local entry) or the encode
        // failure just above (which has not) — `EngineEventBus` never
        // finishes a subscriber's stream on its own (see its doc comment).
        // Tell the client either way so a still-live connection's
        // `AsyncStream` ends deterministically rather than going quiet.
        // (The callback-proxy error handler above covers the remaining
        // failure mode — the push itself failing on a dying connection —
        // which, like its STT `drain` counterpart, has no headless test:
        // `NSXPCConnection` offers no public API to fail only the callback
        // direction of an in-process anonymous-listener pair.)
        callback?.eventsDidFinish(
            subscriptionID: subscriptionID,
            error: finishError.map { XPCErrorBridge.toNSError($0) }
        )
        removeEventSubscription(subscriptionID)
    }

    private func addEventSubscription(_ entry: EventEntry, id: String) {
        eventLock.lock()
        eventSubscriptions[id] = entry
        eventLock.unlock()
    }

    private func removeEventSubscription(_ subscriptionID: String) {
        eventLock.lock()
        eventSubscriptions[subscriptionID] = nil
        eventLock.unlock()
    }

    private func cancelAndRemoveEventSubscription(_ subscriptionID: String) {
        eventLock.lock()
        let entry = eventSubscriptions[subscriptionID]
        eventSubscriptions[subscriptionID] = nil
        eventLock.unlock()
        entry?.drainTask.cancel()
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
            handler.cancelAllEventSubscriptions()
        }
        newConnection.invalidationHandler = {
            logger.debug("xpc: connection invalidated")
            handler.cancelAllStreams()
            handler.cancelAllEventSubscriptions()
        }
        newConnection.resume()
        return true
    }
}

/// Millisecond projection used by the request-forensics logging above
/// (#280) — `Duration` has no built-in `Double` millisecond accessor.
/// `public` so `YoozEngineInProcess`'s matching STT-batch forensic logging
/// (`InProcessTransport.handleBatch`) can share it rather than duplicate it.
public extension Duration {
    var milliseconds: Double {
        let (seconds, attoseconds) = components
        return Double(seconds) * 1_000 + Double(attoseconds) / 1_000_000_000_000_000
    }
}
