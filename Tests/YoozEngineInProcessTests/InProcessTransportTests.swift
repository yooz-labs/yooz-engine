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

    func testDeleteThrowsUnsupportedInProcess() async throws {
        let transport = InProcessTransport()
        do {
            _ = try await transport.delete("/v1/infinite/sessions/x")
            XCTFail("delete should throw unsupportedOperation")
        } catch let error as YoozEngineError {
            guard case .unsupportedOperation = error else {
                XCTFail("expected unsupportedOperation, got \(error)")
                return
            }
        }
    }

    // MARK: - Picker / status (Phase 2b, no models required)

    func testInProcessSTTLanguages() async throws {
        let client = makeClient()
        try await client.connect()
        let languages = try await client.stt.languages()
        XCTAssertFalse(languages.isEmpty)
        XCTAssertTrue(languages.contains { $0.code == "en" })
    }

    func testInProcessSTTEnginePickerListsBackends() async throws {
        let client = makeClient()
        try await client.connect()
        let response = try await client.stt.availableEngines()
        let ids = Set(response.backends.map(\.id))
        XCTAssertTrue(ids.contains("parakeet"))
        XCTAssertTrue(ids.contains("apple_stt"))
    }

    func testInProcessSTTStatusReadsCleanly() async throws {
        let client = makeClient()
        try await client.connect()
        // No model is started by this test path, so status must return without
        // throwing; streaming is false.
        let status = try await client.stt.status()
        XCTAssertFalse(status.streaming)
    }

    func testInProcessLLMStatusReadsCleanly() async throws {
        let client = makeClient()
        try await client.connect()
        let status = try await client.llm.status()
        XCTAssertNotNil(status.modelId)
    }

    /// Streaming dispatch is wired (no longer `unsupportedOperation` wholesale):
    /// switching to the qwen3 preview backend — which the in-process streaming
    /// path intentionally does NOT support — and opening a stream must throw
    /// `unsupportedOperation` for THAT backend specifically. This exercises the
    /// streaming dispatch + the STT engine picker without needing model weights.
    func testInProcessQwen3StreamingIsUnsupported() async throws {
        let client = makeClient()
        try await client.connect()
        _ = try await client.stt.setEngine(id: "qwen3_asr_preview")
        do {
            _ = try await client.stt.startStream()
            XCTFail("qwen3 streaming should be unsupported in-process")
        } catch let error as YoozEngineError {
            guard case .unsupportedOperation = error else {
                XCTFail("expected unsupportedOperation, got \(error)")
                return
            }
        }
        // Restore the default backend so test ordering can't strand the shared
        // singleton on qwen3.
        _ = try await client.stt.setEngine(id: "parakeet")
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
        // Infinite is intentionally not served in-process (its consumer is the
        // loopback super-yooz host), so it remains a clean unsupported throw.
        let client = makeClient()
        try await client.connect()
        do {
            _ = try await client.infinite.status()
            XCTFail("expected unsupportedOperation for an unimplemented endpoint")
        } catch let error as YoozEngineError {
            guard case .unsupportedOperation = error else {
                XCTFail("expected unsupportedOperation, got \(error)")
                return
            }
        }
    }
}
