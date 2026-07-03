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
