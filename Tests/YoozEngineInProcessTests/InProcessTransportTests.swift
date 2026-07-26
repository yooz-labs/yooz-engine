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

    // MARK: - LLM cache clearing (engine#299, no models required)

    /// Omitting `model` when nothing is loaded must be a success no-op, not
    /// an error — an idle-policy caller needs to invoke this
    /// unconditionally without first checking per-tier load state.
    func testInProcessClearCacheOmittedModelIsANoOpWhenNothingLoaded() async throws {
        let client = makeClient()
        try await client.connect()
        let cleared = try await client.touchUp.clearCache()
        XCTAssertTrue(cleared.isEmpty, "clearing with nothing loaded must report no cleared tiers")
    }

    /// Same no-op contract, scoped to a specific (valid, but not resident)
    /// tier: clearing an already-empty cache never throws.
    func testInProcessClearCacheNamedModelIsANoOpWhenNotLoaded() async throws {
        let client = makeClient()
        try await client.connect()
        let cleared = try await client.touchUp.clearCache("yooz-light-v3")
        XCTAssertTrue(cleared.isEmpty, "clearing a not-loaded tier must report no cleared tiers")
    }

    /// Unknown model id -> 400 `invalid_model`, mirroring
    /// `/v1/llm/preload` and `/v1/llm/unload`.
    func testInProcessClearCacheUnknownModelThrows400() async throws {
        let client = makeClient()
        try await client.connect()
        do {
            _ = try await client.touchUp.clearCache("totally-unknown-model")
            XCTFail("expected a 400 for an unknown model")
        } catch let error as YoozEngineError {
            guard case .serverError(let status, let code, _) = error,
                  status == 400, code == "invalid_model" else {
                return XCTFail("expected serverError 400 invalid_model, got \(error)")
            }
        }
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

    // MARK: - Load lifecycle (Phase 1: model load reliability)

    /// The in-process STT load now routes `loadModelAsync` through the engine's
    /// `enqueueLoad` state machine, so `/v1/stt/status` reports a non-nil
    /// lifecycle `state` (was hardcoded `nil`). A cold engine with no load in
    /// flight reports `.idle` — this is what lets the consumer distinguish a
    /// download ("Downloading X%") from the materialization window
    /// ("Loading model…": state `.loading` + progress nil) and a failure
    /// (`.failed`) without weights or GPU.
    func testInProcessSTTStatusSurfacesLoadStateWhenCold() async throws {
        let client = makeClient()
        try await client.connect()
        _ = try await client.stt.setEngine(id: "parakeet")
        let status = try await client.stt.status()
        XCTAssertEqual(status.state, .idle, "cold MLX engine should report .idle, not nil")
    }

    /// The in-process LLM status handler now surfaces the per-tier `loadState`
    /// (was hardcoded `nil`), so the touch-up picker can render a distinct
    /// "Loading model…" phase during materialization.
    func testInProcessLLMStatusSurfacesLoadStateWhenCold() async throws {
        let client = makeClient()
        try await client.connect()
        let status = try await client.llm.status()
        // A cold engine should report .idle specifically (was hardcoded nil); a
        // bare != nil would also pass for an erroneous .failed/.loading.
        XCTAssertEqual(status.state, .idle)
    }

    /// The Apple STT status branch now maps loaded -> .ready / not-loaded -> .idle
    /// (was hardcoded nil), parity with the loopback route. Cold reports .idle.
    func testInProcessAppleSTTStatusSurfacesIdleStateWhenCold() async throws {
        let client = makeClient()
        try await client.connect()
        _ = try await client.stt.setEngine(id: "apple_stt")
        let status = try await client.stt.status()
        XCTAssertEqual(status.state, .idle, "cold Apple STT should report .idle, not nil")
        // Restore the default backend so test ordering can't strand the singleton.
        _ = try await client.stt.setEngine(id: "parakeet")
    }

    /// `parseWaitQuery` recovers the `?wait` flag the in-process `route()`
    /// strips, so `loadModel` (`?wait=true`, blocking) and `loadModelAsync`
    /// (no query, fire-and-forget) keep their distinct contracts in-process.
    func testParseWaitQuery() {
        XCTAssertFalse(InProcessTransport.parseWaitQuery("/v1/stt/load"))
        XCTAssertTrue(InProcessTransport.parseWaitQuery("/v1/stt/load?wait=true"))
        XCTAssertTrue(InProcessTransport.parseWaitQuery("/v1/stt/load?wait=1"))
        XCTAssertTrue(InProcessTransport.parseWaitQuery("/v1/stt/load?wait"))
        XCTAssertFalse(InProcessTransport.parseWaitQuery("/v1/stt/load?wait=false"))
        XCTAssertFalse(InProcessTransport.parseWaitQuery("/v1/stt/load?wait=0"))
        XCTAssertTrue(InProcessTransport.parseWaitQuery("/v1/stt/load?lang=en&wait=true"))
    }

    // MARK: - Model management (disk hygiene)

    /// `GET /v1/models` routes in-process and returns an internally consistent
    /// inventory. Read-only — never mutates the real cache. State-agnostic: the
    /// machine may have zero or many installed models, so it pins the invariants
    /// rather than a specific list.
    func testInProcessModelsInventoryIsConsistent() async throws {
        let client = makeClient()
        try await client.connect()
        let response = try await client.models.list()
        for model in response.models {
            // Not-cached implies no reclaimable footprint.
            if !model.cached { XCTAssertEqual(model.sizeBytes, 0, "\(model.id)") }
            // The active model is never offered for deletion.
            if model.isActive { XCTAssertFalse(model.deletable, "\(model.id)") }
            // Deletable implies a real footprint and not active.
            if model.deletable {
                XCTAssertGreaterThan(model.sizeBytes, 0, "\(model.id)")
                XCTAssertFalse(model.isActive, "\(model.id)")
            }
        }
    }

    /// Deleting an id that is neither an LLM tier nor a hub repo dir is a 404 —
    /// and the rejection happens before any disk touch.
    func testInProcessDeleteUnknownModelThrows404() async throws {
        let client = makeClient()
        try await client.connect()
        do {
            _ = try await client.models.delete(id: "totally-unknown-model")
            XCTFail("expected a 404 for an unknown model")
        } catch let error as YoozEngineError {
            guard case .serverError(let status, _, _) = error, status == 404 else {
                return XCTFail("expected serverError 404, got \(error)")
            }
        }
    }

    /// The active model can never be deleted (409). Targets whichever LLM tier is
    /// active; the rejection short-circuits before any unload or disk removal, so
    /// this is safe against the real cache.
    func testInProcessDeleteActiveModelIsRejected() async throws {
        let client = makeClient()
        try await client.connect()
        let picker: TouchUpModelsResponse = try await client.touchUp.availableModels()
        guard let activeId = picker.models.first(where: \.isActive)?.id else {
            return XCTFail("expected exactly one active TouchUp model")
        }
        // Apple Intelligence (foundation-models) has no LLMModelType mapping, so
        // it 404s rather than 409s; the active-rejection contract applies to the
        // deletable LLM tiers.
        guard activeId == "yooz-light-v3" || activeId == "yooz-quality-v3" else {
            throw XCTSkip("active model \(activeId) is not a deletable LLM tier")
        }
        do {
            _ = try await client.models.delete(id: activeId)
            XCTFail("deleting the active model should be rejected")
        } catch let error as YoozEngineError {
            guard case .serverError(let status, let code, _) = error,
                  status == 409, code == "model_active" else {
                return XCTFail("expected serverError 409 model_active, got \(error)")
            }
        }
    }

    /// `POST /v1/models/cleanup` routes end-to-end and actually collapses a
    /// stacked cache. `HF_HOME` is redirected to a temp hub holding a real
    /// symlinked fixture, so the assertions are deterministic and the machine's
    /// real cache is never touched.
    func testInProcessCleanupCollapsesRedirectedCache() async throws {
        let fm = FileManager.default
        let home = fm.temporaryDirectory
            .appendingPathComponent("ip-cleanup-\(UUID().uuidString)")
        let repo = home.appendingPathComponent("hub/models--Foo--Bar")
        let blobs = repo.appendingPathComponent("blobs")
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        try Data(count: 4_096).write(to: blobs.appendingPathComponent("cfg"))
        try Data(count: 16_384).write(to: blobs.appendingPathComponent("old"))
        try Data(count: 16_384).write(to: blobs.appendingPathComponent("new"))
        // Each snapshot is complete (config.json + model.safetensors) so the live
        // one counts as a materialized survivor.
        for (commit, weights) in [("old111", "old"), ("new222", "new")] {
            let dir = repo.appendingPathComponent("snapshots/\(commit)")
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try fm.createSymbolicLink(
                atPath: dir.appendingPathComponent("config.json").path,
                withDestinationPath: "../../blobs/cfg"
            )
            try fm.createSymbolicLink(
                atPath: dir.appendingPathComponent("model.safetensors").path,
                withDestinationPath: "../../blobs/\(weights)"
            )
        }
        let refs = repo.appendingPathComponent("refs")
        try fm.createDirectory(at: refs, withIntermediateDirectories: true)
        try "new222".write(
            to: refs.appendingPathComponent("main"), atomically: true, encoding: .utf8
        )

        // HF_HUB_CACHE takes precedence over HF_HOME in
        // `EngineConfig.huggingFaceCacheDirectory`; clear it too or an ambient
        // HF_HUB_CACHE would defeat the redirect and point cleanup at the
        // machine's real cache.
        let savedHome = ProcessInfo.processInfo.environment["HF_HOME"]
        let savedHubCache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
        setenv("HF_HOME", home.path, 1)
        unsetenv("HF_HUB_CACHE")
        defer {
            if let savedHome { setenv("HF_HOME", savedHome, 1) } else { unsetenv("HF_HOME") }
            if let savedHubCache {
                setenv("HF_HUB_CACHE", savedHubCache, 1)
            } else {
                unsetenv("HF_HUB_CACHE")
            }
            try? fm.removeItem(at: home)
        }

        let client = makeClient()
        try await client.connect()
        let result = try await client.models.cleanup()

        XCTAssertGreaterThan(result.totalReclaimedBytes, 0)
        XCTAssertFalse(fm.fileExists(atPath: repo.appendingPathComponent("snapshots/old111").path))
        XCTAssertTrue(fm.fileExists(atPath: repo.appendingPathComponent("snapshots/new222").path))
        XCTAssertFalse(fm.fileExists(atPath: blobs.appendingPathComponent("old").path))
        XCTAssertTrue(fm.fileExists(atPath: blobs.appendingPathComponent("new").path))
    }
}
