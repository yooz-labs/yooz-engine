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
}
