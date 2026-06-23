import XCTest
import YoozEngineClient
@testable import YoozEngineInProcess

/// End-to-end in-process facade tests (epic #192 Phase 2a).
///
/// These exercise the full seam — `YoozEngineClient` SDK surface ->
/// `InProcessTransport` router -> real engine actors -> SDK DTOs — with NO
/// loopback socket and NO mocks. Grammar is the anchor because it runs entirely
/// from the Rust text-cleanup FFI (no model download, deterministic).
final class InProcessTransportTests: XCTestCase {
    private func makeClient() -> YoozEngineClient {
        YoozEngineClient.inProcess()
    }

    func testHealthReportsGrammarLoadedInProcess() async throws {
        let client = makeClient()
        try await client.connect()
        let health = try await client.health()
        XCTAssertTrue(health.isHealthy)
        // Grammar loads its rules in `GrammarEngine.init` (Rust FFI), so it is
        // ready immediately after bootstrap.
        XCTAssertTrue(health.modules.grammar)
    }

    func testModulesManifestIncludesAllRegisteredModulesInProcess() async throws {
        let client = makeClient()
        try await client.connect()
        let modules = try await client.modules()
        // Every module `EngineInProcessHost.bootstrap()` registers must appear.
        let names = Set(modules.modules.map(\.name))
        XCTAssertTrue(
            names.isSuperset(of: ["grammar", "llm", "stt", "apple_stt", "vad"]),
            "expected all registered modules, got \(names.sorted())"
        )
        let grammar = modules.modules.first { $0.name == "grammar" }
        XCTAssertNotNil(grammar, "grammar module should be registered")
        XCTAssertTrue(grammar?.loaded ?? false, "grammar should be loaded (FFI rules present)")
    }

    func testInProcessFactoryWiresInProcessTransport() {
        let client = YoozEngineClient.inProcess()
        XCTAssertTrue(client.transport is InProcessTransport)
        XCTAssertEqual(client.port, 0)
    }

    func testDeleteAndWebSocketThrowUnsupportedInProcess() async throws {
        let transport = InProcessTransport()

        do {
            _ = try await transport.delete("/v1/infinite/sessions/x")
            XCTFail("delete should throw unsupportedInProcess")
        } catch let error as YoozEngineError {
            guard case .unsupportedInProcess = error else {
                XCTFail("expected unsupportedInProcess, got \(error)")
                return
            }
        }

        XCTAssertThrowsError(try transport.webSocketURL(path: "v1/stt/stream")) { error in
            guard case YoozEngineError.unsupportedInProcess = error else {
                XCTFail("expected unsupportedInProcess, got \(error)")
                return
            }
        }
    }

    func testGrammarCheckRoundTripsThroughTheSeam() async throws {
        let client = makeClient()
        try await client.connect()

        let response = try await client.grammar.check(
            GrammarCheckRequest(text: "i have a apple")
        )
        // Rule counts come straight from the Rust FFI — non-nil and positive
        // proves the binary loaded and the whole transport path round-tripped.
        XCTAssertNotNil(response.ruleCount)
        XCTAssertGreaterThan(response.ruleCount ?? 0, 0)
        XCTAssertFalse(response.result.isEmpty)

        // The `correct(text:)` convenience returns the corrected string back
        // through the same in-process seam.
        let corrected = try await client.grammar.correct(text: "i have a apple")
        XCTAssertFalse(corrected.isEmpty)
    }

    func testUnsupportedEndpointThrowsInProcess() async throws {
        let client = makeClient()
        try await client.connect()
        do {
            _ = try await client.stt.status()
            XCTFail("expected unsupportedInProcess for an unimplemented endpoint")
        } catch let error as YoozEngineError {
            guard case .unsupportedInProcess = error else {
                XCTFail("expected unsupportedInProcess, got \(error)")
                return
            }
        }
    }
}
