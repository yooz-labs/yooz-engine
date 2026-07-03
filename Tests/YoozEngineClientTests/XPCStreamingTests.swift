import Foundation
import XCTest
@testable import YoozEngineClient

/// Unit tests for the cancellation-safe streaming channel.
final class STTResultChannelTests: XCTestCase {
    private func partial(_ text: String) -> StreamingSTTResult {
        StreamingSTTResult(type: "partial", text: text, finalized: "", draft: text)
    }

    func testYieldThenReceiveReturnsInOrder() async throws {
        let channel = STTResultChannel()
        channel.yield(partial("a"))
        channel.yield(partial("b"))
        let first = try await channel.receive()
        let second = try await channel.receive()
        XCTAssertEqual(first?.text, "a")
        XCTAssertEqual(second?.text, "b")
    }

    func testFinishEndsStreamWithNil() async throws {
        let channel = STTResultChannel()
        channel.finish()
        let result = try await channel.receive()
        XCTAssertNil(result)
    }

    func testFinishWithErrorThrows() async throws {
        let channel = STTResultChannel()
        channel.finish(throwing: YoozEngineError.invalidResponse)
        do {
            _ = try await channel.receive()
            XCTFail("expected the finish error")
        } catch let error as YoozEngineError {
            XCTAssertEqual(error, .invalidResponse)
        }
    }

    func testReceiveSuspendsThenYieldDelivers() async throws {
        let channel = STTResultChannel()
        async let received = channel.receive()
        try await Task.sleep(for: .milliseconds(20))  // let receive() suspend
        channel.yield(StreamingSTTResult(type: "final", text: "x", finalized: "x", draft: ""))
        let result = try await received
        XCTAssertEqual(result?.text, "x")
    }

    func testCancelledReceiveThrowsCancellation() async throws {
        let channel = STTResultChannel()
        let task = Task { try await channel.receive() }
        try await Task.sleep(for: .milliseconds(20))  // let it suspend
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected CancellationError")
        } catch is CancellationError {
            // expected
        }
    }
}

/// XPC streaming round trip (epic #192 Phase 3b) over a real `NSXPCConnection`.
///
/// The service side is `XPCServiceHandler` backed by a canned `EngineTransport`
/// whose stream yields deterministic partials/final — so the full bidirectional
/// callback proxy is exercised (open -> sendAudio -> service push -> client
/// receive -> close -> final) with no model weights / GPU, under plain `swift test`.
final class XPCStreamingTests: XCTestCase {
    private final class Service {
        let listener = NSXPCListener.anonymous()
        let delegate: XPCServiceListenerDelegate
        init() {
            delegate = XPCServiceListenerDelegate { XPCServiceHandler(transport: CannedStreamTransport()) }
            listener.delegate = delegate
            listener.resume()
        }
        func makeClient() -> YoozEngineClient {
            YoozEngineClient(transport: XPCTransport(connection: NSXPCConnection(listenerEndpoint: listener.endpoint)))
        }
        func invalidate() { listener.invalidate() }
    }

    func testOpenStreamReturnsOverXPC() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()
        let stream = try await client.stt.startStream(language: .english)
        stream.close()
    }

    func testStreamingRoundTripsOverXPC() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()

        let stream = try await client.stt.startStream(language: .english)

        // Each sendAudio yields one partial on the service side; it must arrive
        // back over the callback proxy.
        try await stream.sendAudio([0.1, 0.2, 0.3])
        let partial = try await stream.receive()
        XCTAssertEqual(partial?.type, "partial")
        XCTAssertFalse(partial?.isFinal ?? true)

        // close() makes the service finalize; the `final` result must arrive,
        // then the stream ends (nil).
        stream.close()
        var sawFinal = false
        while let result = try await stream.receive() {
            if result.isFinal {
                sawFinal = true
                XCTAssertEqual(result.text, "final text")
                break
            }
        }
        XCTAssertTrue(sawFinal, "streaming over XPC must deliver a final result")
    }
}

/// `/v1/events` plumbing over XPC (engine#244), independent of any real
/// engine module — `CannedEventsTransport` stands in for `InProcessTransport`
/// exactly like `CannedStreamTransport` does for STT above, so these tests
/// exercise the encode/decode + callback-proxy wiring under plain `swift
/// test` with no model weights. `XPCRoundTripTests`
/// (`Tests/YoozEngineInProcessTests/`) covers the real-engine round trip
/// (a genuine `EngineEventBus` publish reaching a subscribed XPC client).
final class XPCEventsTests: XCTestCase {
    private final class Service {
        let listener = NSXPCListener.anonymous()
        let delegate: XPCServiceListenerDelegate
        init() {
            delegate = XPCServiceListenerDelegate { XPCServiceHandler(transport: CannedEventsTransport()) }
            listener.delegate = delegate
            listener.resume()
        }
        func makeClient() -> YoozEngineClient {
            YoozEngineClient(transport: XPCTransport(connection: NSXPCConnection(listenerEndpoint: listener.endpoint)))
        }
        func invalidate() { listener.invalidate() }
    }

    /// The canned transport's `openEvents()` yields one deterministic event
    /// on open, then holds the stream open (like the real `EngineEventBus`,
    /// which never finishes a subscriber's stream on its own — see its doc
    /// comment) until the connection tears it down.
    func testOpenEventsDeliversFrameOverXPC() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()

        let stream = try await client.openEvents()
        let iterator = IteratorBox(stream.makeAsyncIterator())
        let event = try await Self.withTimeout(seconds: 5) { await iterator.next() }

        XCTAssertEqual(event?.kind, .modelChanged)
        XCTAssertEqual(event?.module, "canned")
        XCTAssertEqual(event?.modelId, "m1")
    }

    /// The contract documented on `XPCTransport.openEvents()`: when the
    /// connection dies, the stream must FINISH (the `for await`/iterator
    /// loop ends) rather than leave the caller awaiting a frame that will
    /// never arrive. `NSXPCConnection` has no public API to simulate the
    /// OTHER trigger (peer-process interruption) from an in-process
    /// anonymous listener — `XPCTransport.init` wires `interruptionHandler`
    /// and `invalidationHandler` to the exact same closure
    /// (`streamClient.finishAll`), so exercising invalidation covers both.
    func testEventsStreamFinishesOnConnectionInvalidation() async throws {
        let service = Service()
        let client = service.makeClient()

        let stream = try await client.openEvents()
        let iterator = IteratorBox(stream.makeAsyncIterator())
        _ = try await Self.withTimeout(seconds: 5) { await iterator.next() }  // the canned open-frame

        service.invalidate()

        let finished = try await Self.withTimeout(seconds: 5) {
            while await iterator.next() != nil {}
            return true
        }
        XCTAssertEqual(finished, true, "the events stream must finish once the connection invalidates")
    }

    /// A service-side frame ENCODE failure (here: `progress: .infinity`,
    /// which `JSONEncoder` rejects by default) must end the subscription
    /// deterministically — `XPCServiceHandler.drainEvents` breaks its drain
    /// loop and pushes `eventsDidFinish(subscriptionID:error:)`, so the
    /// client stream finishes instead of going quiet with the service still
    /// nominally subscribed. Exercises the full error-threading path the
    /// PR #245 review asked for (service diagnosis crossing the wire).
    func testServiceEncodeFailureFinishesClientStream() async throws {
        let listener = NSXPCListener.anonymous()
        let delegate = XPCServiceListenerDelegate {
            XPCServiceHandler(transport: UnencodableEventsTransport())
        }
        listener.delegate = delegate
        listener.resume()
        defer { listener.invalidate() }

        let transport = XPCTransport(connection: NSXPCConnection(listenerEndpoint: listener.endpoint))
        let stream = try await transport.openEvents()

        let iterator = IteratorBox(stream.makeAsyncIterator())
        let finished = try await Self.withTimeout(seconds: 5) {
            while await iterator.next() != nil {}
            return true
        }
        XCTAssertEqual(finished, true, "an unencodable frame must finish the client stream via eventsDidFinish")
    }

    /// A malformed frame reaching the client callback must END the
    /// subscription (finish the stream), never silently drop the frame and
    /// leave the stream nominally live — `eventDidOccur`'s decode-failure
    /// branch, unit-tested directly: `XPCStreamClient` is a plain exported
    /// object, so no XPC connection round trip is needed to drive it.
    func testMalformedFrameEndsSubscription() async throws {
        let streamClient = XPCStreamClient()
        let (stream, continuation) = AsyncStream<EngineEvent>.makeStream()
        // Never resumed; exists only because XPCEventSubscription's close
        // notification needs a connection to (harmlessly) address.
        let connection = NSXPCConnection(serviceName: "live.yooz.engine.test.unused")
        defer { connection.invalidate() }
        let subscription = XPCEventSubscription(
            subscriptionID: "sub-1", connection: connection, continuation: continuation
        )
        streamClient.registerEvents(subscription, for: "sub-1")

        streamClient.eventDidOccur(subscriptionID: "sub-1", eventData: Data("not json".utf8))

        let iterator = IteratorBox(stream.makeAsyncIterator())
        let next = try await Self.withTimeout(seconds: 5) { await iterator.next() }
        XCTAssertNil(next, "a malformed frame must finish the stream, not drop silently")
    }

    /// Bounds an async operation so a genuine regression (a stream that
    /// never finishes) fails the test instead of hanging it forever.
    private static func withTimeout<Result: Sendable>(
        seconds: Double, operation: @escaping @Sendable () async -> Result
    ) async throws -> Result {
        try await withThrowingTaskGroup(of: Result.self) { group in
            group.addTask { await operation() }
            group.addTask {
                try await Task.sleep(for: .seconds(seconds))
                throw YoozEngineError.invalidResponse  // timeout marker
            }
            defer { group.cancelAll() }
            let result = try await group.next()!
            return result
        }
    }
}

/// Reference-type wrapper so a value-type `AsyncStream<EngineEvent>.AsyncIterator`
/// can be shared across suspension points inside an escaping `@Sendable`
/// closure (e.g. the `withTimeout` task-group closures above) without an
/// `inout` capture — mirrors `EngineStateAndEventsTests.IteratorBox`
/// (`Tests/YoozEngineInProcessTests/`), duplicated here rather than shared
/// since the two test targets don't otherwise depend on each other.
private actor IteratorBox {
    private var iterator: AsyncStream<EngineEvent>.AsyncIterator
    init(_ iterator: AsyncStream<EngineEvent>.AsyncIterator) { self.iterator = iterator }
    func next() async -> EngineEvent? {
        var local = iterator
        let result = await local.next()
        iterator = local
        return result
    }
}

/// `EngineTransport` whose `openEvents()` yields one deterministic event and
/// then holds the stream open indefinitely — mirroring `EngineEventBus`'s
/// real behavior (never finishes a subscriber on its own; only cancellation
/// of the iterating task ends it), so tests against this double genuinely
/// exercise the XPC teardown path rather than a stream that would have ended
/// on its own regardless.
private final class CannedEventsTransport: EngineTransport, @unchecked Sendable {
    let baseURL = URL(string: "canned://test")!
    let port = 0
    func connect() async throws {}
    func isReachable() async throws -> Bool { true }
    func get(_ path: String) async throws -> Data { Data() }
    func post(_ path: String, body: Data) async throws -> Data { Data() }
    func delete(_ path: String) async throws -> Data { Data() }

    @available(macOS 14.0, iOS 17.0, *)
    func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        CannedStreamSession()
    }

    @available(macOS 14.0, iOS 17.0, *)
    func openEvents() async throws -> AsyncStream<EngineEvent> {
        AsyncStream<EngineEvent> { continuation in
            continuation.yield(EngineEvent(kind: .modelChanged, module: "canned", modelId: "m1"))
            // No further yields, no continuation.finish() — see this type's
            // doc comment for why that matters.
        }
    }
}

/// `EngineTransport` whose `openEvents()` yields a frame `JSONEncoder`
/// cannot encode (`progress: .infinity` — rejected by the default
/// `nonConformingFloatEncodingStrategy`), driving the service-side
/// encode-failure branch of `XPCServiceHandler.drainEvents`. Same held-open
/// semantics as `CannedEventsTransport`.
private final class UnencodableEventsTransport: EngineTransport, @unchecked Sendable {
    let baseURL = URL(string: "canned://test")!
    let port = 0
    func connect() async throws {}
    func isReachable() async throws -> Bool { true }
    func get(_ path: String) async throws -> Data { Data() }
    func post(_ path: String, body: Data) async throws -> Data { Data() }
    func delete(_ path: String) async throws -> Data { Data() }

    @available(macOS 14.0, iOS 17.0, *)
    func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        CannedStreamSession()
    }

    @available(macOS 14.0, iOS 17.0, *)
    func openEvents() async throws -> AsyncStream<EngineEvent> {
        AsyncStream<EngineEvent> { continuation in
            continuation.yield(
                EngineEvent(kind: .downloadProgress, module: "canned", modelId: "m1", progress: .infinity)
            )
        }
    }
}

// MARK: - Canned non-GPU stream backend

/// `EngineTransport` whose `openSTTStream` returns a deterministic stream — a
/// stand-in for the real engine so the XPC streaming PLUMBING can be tested
/// without model weights. (Not a mock of the code under test; it's a backend
/// double, like `SpyTransport`.)
private final class CannedStreamTransport: EngineTransport, @unchecked Sendable {
    let baseURL = URL(string: "canned://test")!
    let port = 0
    func connect() async throws {}
    func isReachable() async throws -> Bool { true }
    func get(_ path: String) async throws -> Data { Data() }
    func post(_ path: String, body: Data) async throws -> Data { Data() }
    func delete(_ path: String) async throws -> Data { Data() }

    @available(macOS 14.0, iOS 17.0, *)
    func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        CannedStreamSession()
    }

    @available(macOS 14.0, iOS 17.0, *)
    func openEvents() async throws -> AsyncStream<EngineEvent> {
        throw YoozEngineError.unsupportedOperation(operation: "canned events")
    }
}

@available(macOS 14.0, iOS 17.0, *)
private final class CannedStreamSession: STTStreamSession, @unchecked Sendable {
    private let channel = STTResultChannel()
    private let lock = NSLock()
    private var count = 0

    func sendAudio(_ samples: [Float]) async throws {
        lock.lock()
        count += 1
        let n = count
        lock.unlock()
        channel.yield(StreamingSTTResult(type: "partial", text: "p\(n)", finalized: "", draft: "p\(n)"))
    }

    func receive() async throws -> StreamingSTTResult? {
        try await channel.receive()
    }

    func close() {
        channel.yield(StreamingSTTResult(type: "final", text: "final text", finalized: "final text", draft: ""))
        channel.finish()
    }
}
