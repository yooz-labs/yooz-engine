import Foundation

/// Client-side streaming session over XPC (epic #192 Phase 3b).
///
/// Audio is sent to the service as chunked `Data` (`sendAudio`); results arrive
/// asynchronously through the connection's client callback
/// (`YoozEngineXPCStreamClientProtocol`), are routed here by `XPCStreamClient`,
/// and surface through `receive()` via an `STTResultChannel`. `close()` asks the
/// service to finalize; the service then delivers the `final` result and a
/// finish callback (so `close()` must NOT finish the channel itself).
@available(macOS 14.0, iOS 17.0, *)
final class XPCSTTStreamSession: STTStreamSession, @unchecked Sendable {
    private let streamID: String
    private let connection: NSXPCConnection
    private let channel = STTResultChannel()

    init(streamID: String, connection: NSXPCConnection) {
        self.streamID = streamID
        self.connection = connection
    }

    func sendAudio(_ samples: [Float]) async throws {
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        let channel = self.channel
        let proxy = connection.remoteObjectProxyWithErrorHandler { error in
            channel.finish(throwing: XPCErrorBridge.toYoozEngineError(error))
        }
        (proxy as? YoozEngineXPCProtocol)?.sendAudio(streamID: streamID, data: data)
    }

    func receive() async throws -> StreamingSTTResult? {
        try await channel.receive()
    }

    func close() {
        // Ask the service to finalize; it delivers the final result and then a
        // finish callback. The router routes both here and unregisters this
        // session on finish — so DON'T unregister or finish the channel here, or
        // the final result would be dropped (the consumer's receive() would hang).
        (connection.remoteObjectProxy as? YoozEngineXPCProtocol)?.closeStream(streamID: streamID)
    }

    // MARK: - Called by XPCStreamClient (the callback router)

    func deliver(_ result: StreamingSTTResult) { channel.yield(result) }

    func finish(error: Error?) { channel.finish(throwing: error) }
}

/// Client-exported callback object: receives the service's streaming pushes and
/// routes them to the right `XPCSTTStreamSession` by `streamID`. One per
/// connection (set as `exportedObject`). Also routes the `/v1/events`
/// (engine#244) push callbacks to the right `XPCEventSubscription` by
/// `subscriptionID` — folded into this one exported object rather than a
/// second `exportedObject`, since `NSXPCConnection.exportedInterface` /
/// `exportedObject` are each singular per connection.
@available(macOS 14.0, iOS 17.0, *)
final class XPCStreamClient: NSObject, YoozEngineXPCStreamClientProtocol, @unchecked Sendable {
    private let lock = NSLock()
    private var sessions: [String: XPCSTTStreamSession] = [:]
    private var eventSubscriptions: [String: XPCEventSubscription] = [:]

    func register(_ session: XPCSTTStreamSession, for streamID: String) {
        lock.lock()
        sessions[streamID] = session
        lock.unlock()
    }

    func unregister(_ streamID: String) {
        lock.lock()
        sessions[streamID] = nil
        lock.unlock()
    }

    func registerEvents(_ subscription: XPCEventSubscription, for subscriptionID: String) {
        lock.lock()
        eventSubscriptions[subscriptionID] = subscription
        lock.unlock()
    }

    func unregisterEvents(_ subscriptionID: String) {
        lock.lock()
        eventSubscriptions[subscriptionID] = nil
        lock.unlock()
    }

    /// Finish every active STT stream AND event subscription (e.g. on
    /// connection invalidation/interruption). This is the client-side half
    /// of the "the stream finishes rather than silently going quiet"
    /// contract `XPCTransport.openEvents()` documents (engine#244) — STT
    /// streams already had it, events now share the same wiring.
    func finishAll(error: Error) {
        lock.lock()
        let activeSessions = sessions
        let activeEvents = eventSubscriptions
        sessions.removeAll()
        eventSubscriptions.removeAll()
        lock.unlock()
        for session in activeSessions.values {
            session.finish(error: error)
        }
        for subscription in activeEvents.values {
            subscription.finish()
        }
    }

    private func session(_ streamID: String) -> XPCSTTStreamSession? {
        lock.lock()
        defer { lock.unlock() }
        return sessions[streamID]
    }

    private func eventSubscription(_ subscriptionID: String) -> XPCEventSubscription? {
        lock.lock()
        defer { lock.unlock() }
        return eventSubscriptions[subscriptionID]
    }

    func streamDidProduce(streamID: String, resultData: Data) {
        // Single lookup so a concurrent streamDidFinish/finishAll can't change the
        // session between the decode-failure branch and delivery.
        guard let session = session(streamID) else { return }
        guard let result = try? JSONDecoder().decode(StreamingSTTResult.self, from: resultData) else {
            // A malformed result is a real failure, not something to drop silently.
            session.finish(error: YoozEngineError.decodingError("Malformed streaming result over XPC"))
            return
        }
        session.deliver(result)
    }

    func streamDidFinish(streamID: String, error: Error?) {
        let session = session(streamID)
        unregister(streamID)
        session?.finish(error: error.map { XPCErrorBridge.toYoozEngineError($0) })
    }

    // MARK: - Events (engine#244)

    func eventDidOccur(subscriptionID: String, eventData: Data) {
        // Single lookup so a concurrent eventsDidFinish/finishAll can't change
        // the subscription between the decode-failure branch and delivery —
        // mirrors `streamDidProduce` above.
        guard let subscription = eventSubscription(subscriptionID) else { return }
        guard let event = try? JSONDecoder().decode(EngineEvent.self, from: eventData) else {
            // A malformed frame ends the subscription rather than silently
            // dropping it, matching `streamDidProduce`'s decode-failure handling.
            unregisterEvents(subscriptionID)
            subscription.finish()
            return
        }
        subscription.deliver(event)
    }

    func eventsDidFinish(subscriptionID: String) {
        let subscription = eventSubscription(subscriptionID)
        unregisterEvents(subscriptionID)
        subscription?.finish()
    }
}
