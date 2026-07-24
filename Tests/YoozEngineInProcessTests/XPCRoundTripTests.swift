import EngineCore
import Foundation
import XCTest
import YoozEngineClient
@testable import YoozEngineInProcess

/// End-to-end XPC tests (epic #192 Phase 3a) over a REAL `NSXPCConnection`.
///
/// An `NSXPCListener.anonymous()` runs the service side in-process, exporting an
/// `XPCServiceHandler` backed by `InProcessTransport()`. The client talks to it
/// through `XPCTransport`, so the whole stack is exercised: SDK request encode ->
/// XPC -> handler -> InProcessTransport -> real engine actor -> Data -> XPC ->
/// SDK decode. Grammar runs entirely on the Rust text-cleanup FFI, so this needs
/// no model weights and no MLX/metallib — it runs under plain `swift test`.
final class XPCRoundTripTests: XCTestCase {
    /// Keeps the listener + delegate alive for the duration of a test.
    private final class Service {
        let listener: NSXPCListener
        let delegate: XPCServiceListenerDelegate
        init() {
            listener = NSXPCListener.anonymous()
            delegate = XPCServiceListenerDelegate {
                XPCServiceHandler(transport: InProcessTransport())
            }
            listener.delegate = delegate
            listener.resume()
        }
        func makeClient() -> YoozEngineClient {
            let connection = NSXPCConnection(listenerEndpoint: listener.endpoint)
            return YoozEngineClient(transport: XPCTransport(connection: connection))
        }
        func invalidate() { listener.invalidate() }
    }

    func testGrammarRoundTripsOverRealXPCConnection() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()

        try await client.connect()  // GET /v1/health across XPC

        let response = try await client.grammar.check(GrammarCheckRequest(text: "i have a apple"))
        // The Rust FFI ran on the service side and the result crossed XPC back.
        XCTAssertNotNil(response.ruleCount)
        XCTAssertGreaterThan(response.ruleCount ?? 0, 0)
        XCTAssertFalse(response.result.isEmpty)
    }

    func testHealthAndModulesRoundTripOverXPC() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()

        let health = try await client.health()
        XCTAssertTrue(health.isHealthy)
        XCTAssertTrue(health.modules.grammar)

        let modules = try await client.modules()
        XCTAssertTrue(modules.modules.contains { $0.name == "grammar" })
    }

    /// `serverError` keeps its status + machine-readable code across XPC (not
    /// flattened to a generic error). Setting an unknown LLM model id is a
    /// `400 invalid_model` on the service side and must arrive intact.
    func testServerErrorPreservesCodeAndStatusOverXPC() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()
        try await client.connect()

        do {
            try await client.touchUp.setModel("definitely-not-a-real-model")  // POST /v1/touchup/model
            XCTFail("expected a serverError for an unknown model id")
        } catch let error as YoozEngineError {
            guard case .serverError(let statusCode, let code, _) = error else {
                XCTFail("expected serverError, got \(error)")
                return
            }
            XCTAssertEqual(statusCode, 400)
            XCTAssertEqual(code, "invalid_model")
        }
    }

    /// The per-recording session boundary (engine issue #114 / #222) rides the
    /// XPC path for free because `XPCServiceHandler` forwards to
    /// `InProcessTransport`: `begin` returns `{sessionId, ts}` and `end`
    /// returns an empty body, both intact across a real `NSXPCConnection`.
    func testSessionBeginAndEndRoundTripOverXPC() async throws {
        let service = Service()
        defer { service.invalidate() }
        let connection = NSXPCConnection(listenerEndpoint: service.listener.endpoint)
        let transport = XPCTransport(connection: connection)
        try await transport.connect()

        struct BeginResponse: Decodable {
            let sessionId: String
            let ts: String
        }

        let beginData = try await transport.post("/v1/session/begin", body: Data())
        let begin = try JSONDecoder().decode(BeginResponse.self, from: beginData)
        XCTAssertFalse(begin.sessionId.isEmpty)
        XCTAssertFalse(begin.ts.isEmpty)

        let endData = try await transport.post("/v1/session/end", body: Data())
        XCTAssertTrue(endData.isEmpty, "end should cross XPC as an empty body")
    }

    /// Typed errors survive the XPC boundary: an unsupported endpoint comes back
    /// as `unsupportedOperation`, not a generic connection error.
    func testUnsupportedEndpointPropagatesTypedErrorOverXPC() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()
        try await client.connect()

        do {
            _ = try await client.infinite.status()
            XCTFail("expected unsupportedOperation across XPC")
        } catch let error as YoozEngineError {
            guard case .unsupportedOperation = error else {
                XCTFail("expected unsupportedOperation, got \(error)")
                return
            }
        }
    }

    /// `/v1/events` (engine#226) round-trips over a REAL XPC connection
    /// (engine#244): a state change made through the SAME packaged-style
    /// transport publishes to the shared `EngineEventBus`, and
    /// `XPCTransport.openEvents()` — the callback-proxy bridge added in
    /// #244 — must deliver the resulting frame back across the connection.
    /// This is the acceptance criterion from the engine#244 issue body:
    /// "an event fired engine-side reaches a subscribed `XPCTransport`
    /// client."
    func testEventsRoundTripOverRealXPCConnection() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()
        try await client.connect()

        let stream = try await client.openEvents()

        let body = try JSONEncoder().encode(
            TouchUpSetModelRequest(id: "yooz-quality-v3", preload: false)
        )
        _ = try await client.transport.post("/v1/touchup/model", body: body)

        let matched = try await Self.firstMatchingEvent(stream, timeoutSeconds: 5) {
            $0.kind == .modelChanged && $0.module == "touchup" && $0.modelId == "yooz-quality-v3"
        }
        XCTAssertNotNil(matched, "expected a modelChanged event for yooz-quality-v3 delivered over XPC")

        // Leave the shared engine in a known state for any other in-process
        // test relying on the .yoozLight default (mirrors
        // EngineStateAndEventsTests' own cleanup for the same reason).
        _ = try await client.transport.post(
            "/v1/touchup/model",
            body: try JSONEncoder().encode(TouchUpSetModelRequest(id: "yooz-light-v3", preload: false))
        )
    }

    /// Teardown: invalidating the connection must (a) finish the client's
    /// `AsyncStream` promptly — the "stream finishes rather than silently
    /// going quiet" contract `XPCTransport.openEvents()` documents — and
    /// (b) release the service-side `EngineEventBus` subscription, so a
    /// crashed/killed peer never leaks a subscriber. `EngineEventBus.shared`
    /// is a process-wide singleton other tests may also be subscribing to
    /// concurrently, so this asserts the count returns to its OWN baseline
    /// (before minus after), not an absolute value.
    func testEventsSubscriptionReleasedWhenConnectionInvalidates() async throws {
        let service = Service()
        let connection = NSXPCConnection(listenerEndpoint: service.listener.endpoint)
        let transport = XPCTransport(connection: connection)
        try await transport.connect()

        let baseline = await EngineEventBus.shared.subscriberCount
        let stream = try await transport.openEvents()

        let subscribedCount = await Self.pollUntil(timeoutSeconds: 5) {
            await EngineEventBus.shared.subscriberCount > baseline
        }
        XCTAssertTrue(subscribedCount, "opening the XPC events channel must add a bus subscriber")

        service.invalidate()

        // The stream must FINISH (loop exits, no hang) rather than go quiet
        // forever — this is the client-side half of the contract.
        let iterator = IteratorBox(stream.makeAsyncIterator())
        let finished = await Self.raceAgainstTimeout(timeoutSeconds: 5) {
            while await iterator.next() != nil {}
            return true
        }
        XCTAssertEqual(finished, true, "the events stream must finish after the connection invalidates")

        // The service-side bus subscription must be released, not leaked —
        // poll briefly since teardown (connection -> drain-task cancel ->
        // AsyncStream onTermination -> bus removal) is asynchronous.
        let released = await Self.pollUntil(timeoutSeconds: 5) {
            await EngineEventBus.shared.subscriberCount <= baseline
        }
        XCTAssertTrue(released, "the service-side EngineEventBus subscription must be released on connection death")
    }

    /// The STEADY-STATE unsubscribe (PR #245 review): the consumer simply
    /// stops iterating — e.g. `EngineStateStore.stop()` at picker teardown —
    /// while the connection and both processes stay alive. Cancelling the
    /// consuming task ends the client `AsyncStream`'s iteration, which fires
    /// its `onTermination` -> `closeEvents` over the wire -> the service
    /// cancels the drain task -> the `EngineEventBus` subscription is
    /// released. A regression here would leak one bus subscription per
    /// picker open/close cycle in a long-running app, invisibly. Also pins
    /// that the connection itself remains healthy for unrelated calls
    /// afterward (close is a subscription-level teardown, not connection-level).
    func testCancellingConsumerReleasesServiceSubscriptionAndKeepsConnectionAlive() async throws {
        let service = Service()
        defer { service.invalidate() }
        let connection = NSXPCConnection(listenerEndpoint: service.listener.endpoint)
        let transport = XPCTransport(connection: connection)
        try await transport.connect()

        let baseline = await EngineEventBus.shared.subscriberCount
        let stream = try await transport.openEvents()
        let consumer = Task {
            for await _ in stream {}
        }

        let subscribed = await Self.pollUntil(timeoutSeconds: 5) {
            await EngineEventBus.shared.subscriberCount > baseline
        }
        XCTAssertTrue(subscribed, "opening the XPC events channel must add a bus subscriber")

        consumer.cancel()

        let released = await Self.pollUntil(timeoutSeconds: 5) {
            await EngineEventBus.shared.subscriberCount <= baseline
        }
        XCTAssertTrue(released, "cancelling the consuming task must release the service-side subscription via closeEvents")

        // The connection survives a subscription-level close.
        let health = try await transport.get("/v1/health")
        XCTAssertFalse(health.isEmpty, "the connection must remain usable after an events unsubscribe")
    }

    /// The wire protocol supports N concurrent subscriptions per connection
    /// (keyed by client-generated `subscriptionID`) — prove two live
    /// subscriptions on ONE transport each independently receive the same
    /// engine-side publish, with no cross-talk (PR #245 review: nothing
    /// structurally prevents a diagnostics overlay and `EngineStateStore`
    /// both calling `openEvents()` on the same client).
    func testTwoConcurrentSubscriptionsBothReceiveTheSameEvent() async throws {
        let service = Service()
        defer { service.invalidate() }
        let client = service.makeClient()
        try await client.connect()

        let streamA = try await client.openEvents()
        let streamB = try await client.openEvents()

        let body = try JSONEncoder().encode(
            TouchUpSetModelRequest(id: "yooz-quality-v3", preload: false)
        )
        _ = try await client.transport.post("/v1/touchup/model", body: body)

        // Sequential scans are safe: each client-side AsyncStream buffers
        // (unbounded default), so B's frame waits while A is drained.
        let matchedA = try await Self.firstMatchingEvent(streamA, timeoutSeconds: 5) {
            $0.kind == .modelChanged && $0.module == "touchup" && $0.modelId == "yooz-quality-v3"
        }
        let matchedB = try await Self.firstMatchingEvent(streamB, timeoutSeconds: 5) {
            $0.kind == .modelChanged && $0.module == "touchup" && $0.modelId == "yooz-quality-v3"
        }
        XCTAssertNotNil(matchedA, "the first concurrent subscription must receive the event")
        XCTAssertNotNil(matchedB, "the second concurrent subscription must receive the event")

        // Restore the shared engine's default (same cleanup as the
        // round-trip test above).
        _ = try await client.transport.post(
            "/v1/touchup/model",
            body: try JSONEncoder().encode(TouchUpSetModelRequest(id: "yooz-light-v3", preload: false))
        )
    }

    // MARK: - Helpers

    private static func firstMatchingEvent(
        _ stream: AsyncStream<EngineEvent>,
        timeoutSeconds: Double,
        where predicate: @escaping @Sendable (EngineEvent) -> Bool
    ) async throws -> EngineEvent? {
        let iterator = IteratorBox(stream.makeAsyncIterator())
        let deadline = ContinuousClock.now.advanced(by: .seconds(timeoutSeconds))
        while ContinuousClock.now < deadline {
            let remaining = deadline - ContinuousClock.now
            guard let event = await raceAgainstTimeout(timeout: remaining, operation: { await iterator.next() }) else {
                return nil  // timed out waiting for the next frame
            }
            guard let event else { return nil }  // stream ended
            if predicate(event) { return event }
        }
        return nil
    }

    /// Poll `condition` until it returns true or `timeoutSeconds` elapses.
    private static func pollUntil(timeoutSeconds: Double, condition: @escaping () async -> Bool) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: .seconds(timeoutSeconds))
        while ContinuousClock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(50))
        }
        return await condition()
    }

    /// Run `operation`, returning its result, or `nil` if `timeoutSeconds`
    /// elapses first — used to bound "drain the stream to completion" so a
    /// genuine regression (stream that never finishes) fails the test
    /// instead of hanging it.
    private static func raceAgainstTimeout<Result: Sendable>(
        timeoutSeconds: Double, operation: @escaping @Sendable () async -> Result
    ) async -> Result? {
        await withTaskGroup(of: Optional<Result>.self) { group in
            group.addTask { Optional(await operation()) }
            group.addTask {
                try? await Task.sleep(for: .seconds(timeoutSeconds))
                return nil
            }
            defer { group.cancelAll() }
            return await group.next() ?? nil
        }
    }

    private static func raceAgainstTimeout<Result: Sendable>(
        timeout: Duration, operation: @escaping @Sendable () async -> Result
    ) async -> Result? {
        await withTaskGroup(of: Optional<Result>.self) { group in
            group.addTask { Optional(await operation()) }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }
            defer { group.cancelAll() }
            return await group.next() ?? nil
        }
    }
}

/// Reference-type wrapper so a value-type `AsyncStream<EngineEvent>.AsyncIterator`
/// can be shared across suspension points inside an escaping `@Sendable`
/// closure (the `raceAgainstTimeout` task-group closures above) without an
/// `inout` capture, which those closures can't take. Mirrors
/// `EngineStateAndEventsTests.IteratorBox` in this same test target;
/// duplicated (not shared) since that one is file-private to its own class.
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
