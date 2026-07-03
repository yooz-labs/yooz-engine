import Foundation

/// Client-side `/v1/events` subscription over XPC (engine#244).
///
/// Mirrors `XPCSTTStreamSession`'s callback-proxy shape, but one-directional:
/// the service only ever pushes to `deliver(_:)` — there is no
/// `sendAudio`/request-response counterpart, since `/v1/events` has nothing
/// for the client to send. `finish()` ends the `AsyncStream` the SDK handed
/// the caller; whichever of these fires first drives it:
///
///   - the consumer stops iterating (a cancelled `Task`) — `onTermination`
///     below fires, telling the service to release its `EngineEventBus`
///     subscription via `closeEvents`;
///   - the connection interrupts/invalidates — `XPCStreamClient.finishAll`
///     calls `finish()` directly (the "stream finishes, does not silently go
///     quiet" contract `XPCTransport.openEvents()` documents); or
///   - the service pushes `eventsDidFinish` (a service-side teardown — see
///     `XPCServiceHandler.drainEvents`'s doc for when that happens).
@available(macOS 14.0, iOS 17.0, *)
final class XPCEventSubscription: @unchecked Sendable {
    private let subscriptionID: String
    private let connection: NSXPCConnection
    private let continuation: AsyncStream<EngineEvent>.Continuation

    init(subscriptionID: String, connection: NSXPCConnection, continuation: AsyncStream<EngineEvent>.Continuation) {
        self.subscriptionID = subscriptionID
        self.connection = connection
        self.continuation = continuation
        // Best-effort notice to the service when the CONSUMER stops
        // iterating. Captures `self` (not `connection` directly) so the
        // `@Sendable` closure only crosses a type this class already
        // promises is safe (`@unchecked Sendable`) rather than the
        // non-Sendable `NSXPCConnection` itself.
        continuation.onTermination = { [weak self] _ in
            self?.notifyServiceOfClose()
        }
    }

    private func notifyServiceOfClose() {
        // Plain `remoteObjectProxy` (no error handler), matching
        // `XPCSTTStreamSession.close()` — a send on a dead connection is
        // silently dropped, never a crash.
        (connection.remoteObjectProxy as? YoozEngineXPCProtocol)?.closeEvents(subscriptionID: subscriptionID)
    }

    func deliver(_ event: EngineEvent) { continuation.yield(event) }

    func finish() { continuation.finish() }
}
