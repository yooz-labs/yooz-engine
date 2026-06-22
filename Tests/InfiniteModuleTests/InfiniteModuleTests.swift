// InfiniteModuleTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import XCTest
@testable import InfiniteModule

final class InfiniteModuleTests: XCTestCase {

    override func setUp() async throws {
        await InfiniteEngine.shared.reset()
    }

    override func tearDown() async throws {
        await InfiniteEngine.shared.reset()
    }

    /// Resets `InfiniteEngine.shared` to a clean baseline (clears the leaked
    /// `preparedBackend`/sessions that order-dependent flakes came from) and
    /// skips on unsupported tiers. setUp/tearDown also reset for belt-and-braces.
    private func resetEngine() async throws {
        try requireSupportedTier()
        await InfiniteEngine.shared.reset()
    }

    func testAIModuleName() {
        XCTAssertEqual(InfiniteEngine.name, "infinite")
    }

    func testIsReadyDefaultStateReflectsNoBackendLoaded() async throws {
        try await resetEngine()
        let ready = await InfiniteEngine.shared.isReady
        XCTAssertFalse(ready)
    }

    func testHealthCheckReportsActiveModelAndLoadedFalse() async throws {
        try await resetEngine()
        let health = await InfiniteEngine.shared.healthCheck()
        XCTAssertFalse(health.loaded)
        XCTAssertEqual(health.detail["active_model"], "gemma4-e4b-1m")
        XCTAssertEqual(health.detail["backend_kind"], "paged-kv")
        XCTAssertEqual(health.detail["max_context_tokens"], "1000000")
        XCTAssertEqual(health.detail["ram_tier"], "reduced")
    }

    func testAvailableModelsHasExactlyOneActiveRow() async throws {
        try await resetEngine()
        let models = await InfiniteEngine.shared.availableModels()
        XCTAssertEqual(models.count, InfiniteModelSelection.allCases.count)
        XCTAssertEqual(models.filter(\.isActive).count, 1)
        XCTAssertEqual(models.first(where: \.isActive)?.id, "gemma4-e4b-1m")
    }

    func testAvailableModelsReflectRAMTierGating() async throws {
        try await resetEngine()
        let models = await InfiniteEngine.shared.availableModels()
        let byID = Dictionary(uniqueKeysWithValues: models.map { ($0.id, $0) })
        let selectable: Set<ModelLoadState> = [.available, .cached]

        XCTAssertTrue(selectable.contains(try XCTUnwrap(byID["gemma4-e4b-1m"]?.loadState)))
        if InfiniteRAMTier.current == .full {
            XCTAssertTrue(selectable.contains(try XCTUnwrap(byID["qwen3-35b-1m"]?.loadState)))
            XCTAssertTrue(selectable.contains(try XCTUnwrap(byID["s3-retrieval"]?.loadState)))
        } else {
            XCTAssertEqual(byID["qwen3-35b-1m"]?.loadState, .unavailable)
            XCTAssertEqual(byID["s3-retrieval"]?.loadState, .unavailable)
        }
    }

    func testDescriptorsPinRepositoriesAndAdapterKinds() {
        XCTAssertEqual(
            InfiniteModelSelection.gemma4E4B1M.huggingFaceID,
            "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit"
        )
        XCTAssertEqual(
            InfiniteModelSelection.gemma4E4B1M.revision,
            "b4966f32e71f9f4976a78f74bc8944b1d064bcbf"
        )
        XCTAssertEqual(
            InfiniteModelSelection.gemma4_26B_A4B1M.huggingFaceID,
            "mlx-community/gemma-4-26b-a4b-it-4bit"
        )
        XCTAssertEqual(
            InfiniteModelSelection.qwen35B1M.huggingFaceID,
            "mlx-community/Qwen3.6-35B-A3B-4bit"
        )
        XCTAssertEqual(InfiniteModelSelection.s3Retrieval.huggingFaceID, nil)
        XCTAssertEqual(InfiniteModelSelection.gemma4E4B1M.adapterKind, "infinite-paged-kv-mlx-v1")
        XCTAssertEqual(InfiniteModelSelection.s3Retrieval.adapterKind, "infinite-retrieval-index-v1")
    }

    /// Only architectures the Swift mlx-swift-lm fork can load + generate.
    /// Qwen3.6 (`qwen3_5_moe`) and Gemma4 26B-A4B (`gemma4`) run, verified vs
    /// Python mlx-lm (#184). Gemma4 E4B stays gated on the OptiQ-quant fork-fix
    /// (#186); retrieval has no MLX backend wired. load/generate gate on this.
    func testSwiftRuntimeSupportReflectsMLXBackendCoverage() {
        XCTAssertTrue(InfiniteModelSelection.qwen35B1M.swiftRuntimeSupported)
        XCTAssertTrue(InfiniteModelSelection.gemma4_26B_A4B1M.swiftRuntimeSupported)
        XCTAssertFalse(InfiniteModelSelection.gemma4E4B1M.swiftRuntimeSupported)
        XCTAssertFalse(InfiniteModelSelection.s3Retrieval.swiftRuntimeSupported)
    }

    /// Real end-to-end MLX generation on a small supported model (`qwen3_5`),
    /// proving the native-context backend produces actual tokens. Gated behind
    /// YOOZ_INFINITE_LIVE=1 because it downloads ~0.6 GB on first run; not a
    /// mock — real weights, real decode. Run:
    ///   YOOZ_INFINITE_LIVE=1 xcodebuild ... -only-testing:InfiniteModuleTests/InfiniteModuleTests/testRealNativeContextGenerationProducesTokens test
    func testRealNativeContextGenerationProducesTokens() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_INFINITE_LIVE"] == "1",
            "set YOOZ_INFINITE_LIVE=1 to run the live MLX generation test (downloads a model)"
        )
        // A small, definitely-supported model (qwen3_5). Reuses the real load
        // path via a descriptor. The selection label (.qwen35B1M) is cosmetic:
        // this backend is used directly here and never stored in
        // InfiniteEngine.loadedBackend, so the label/weights mismatch is inert.
        let descriptor = InfiniteBackendDescriptor(
            selection: .qwen35B1M,
            repository: InfiniteModelRepository(
                id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
                revision: "5d894f8cc4ef3e6c88537bf3746ed262f549da6a"
            ),
            backendKind: .pagedKV,
            adapterKind: .pagedKVMLX,
            nativeContextTokens: 32_768,
            targetContextTokens: 32_768,
            requiredRAMTier: .reduced
        )
        let backend = try await MLXInfiniteBackend.load(descriptor)
        let result = try await backend.generate(
            context: "",
            prompt: "Reply with exactly one word: hello.",
            maxTokens: 8,
            nativeContextTokens: 32_768
        )
        XCTAssertFalse(result.text.isEmpty, "generation should produce real tokens")
        XCTAssertGreaterThan(result.tokenCount, 0)
        XCTAssertGreaterThan(result.decodeTokensPerSecond, 0)
    }

    func testSetActiveModelWithoutPreloadReturnsActiveRow() async throws {
        try await resetEngine()
        let selection: InfiniteModelSelection =
            InfiniteRAMTier.current == .full ? .gemma4_26B_A4B1M : .gemma4E4B1M
        let active = try await InfiniteEngine.shared.setActiveModel(
            selection,
            preload: false
        )
        XCTAssertEqual(active.id, selection.rawValue)
        XCTAssertTrue(active.isActive)

        let status = await InfiniteEngine.shared.status()
        XCTAssertEqual(status.modelId, selection.rawValue)
        XCTAssertFalse(status.loaded)

        try await resetEngine()
    }

    func testPreloadPreparesAdapterWithoutMarkingGenerationReady() async throws {
        try await resetEngine()
        let active = try await InfiniteEngine.shared.setActiveModel(.gemma4E4B1M, preload: true)
        XCTAssertEqual(active.id, "gemma4-e4b-1m")

        let status = await InfiniteEngine.shared.status()
        XCTAssertFalse(status.loaded)
        XCTAssertEqual(status.state, "adapter_ready")

        let ready = await InfiniteEngine.shared.isReady
        XCTAssertFalse(ready)

        try await resetEngine()
    }

    func testSessionLifecycleTracksContextCheckpointsAndCleanup() async throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
        let engine = InfiniteEngine()
        _ = try await engine.setActiveModel(.gemma4E4B1M, preload: false)

        let created = try await engine.createSession(
            request: InfiniteCreateSessionRequest(label: "fixture-real-session")
        )
        XCTAssertEqual(created.modelId, "gemma4-e4b-1m")
        XCTAssertEqual(created.label, "fixture-real-session")
        XCTAssertEqual(created.contextWindowTokens, 1_000_000)
        XCTAssertEqual(created.inputCharacters, 0)
        XCTAssertEqual(created.estimatedInputTokens, 0)
        XCTAssertEqual(created.checkpointCount, 0)
        XCTAssertEqual(created.cleanupPolicy, InfiniteEngine.cleanupPolicy)
        XCTAssertGreaterThan(created.resources.physicalMemoryBytes, 0)
        XCTAssertEqual(
            created.resources.wiredMemoryLimitBytes,
            InfiniteRAMTier.reduced.minimumPhysicalMemoryBytes
        )

        let appended = try await engine.append(
            sessionID: created.id,
            request: InfiniteAppendSessionRequest(text: "real context")
        )
        XCTAssertEqual(appended.appendedCharacters, 12)
        XCTAssertEqual(appended.estimatedAppendedTokens, 3)
        XCTAssertEqual(appended.session.inputCharacters, 12)
        XCTAssertEqual(appended.session.estimatedInputTokens, 3)

        let checkpoint = try await engine.checkpoint(
            sessionID: created.id,
            request: InfiniteCheckpointSessionRequest(label: "after-append")
        )
        XCTAssertEqual(checkpoint.checkpoint.label, "after-append")
        XCTAssertEqual(checkpoint.checkpoint.inputCharacters, 12)
        XCTAssertEqual(checkpoint.session.checkpointCount, 1)

        let listed = await engine.listSessions()
        XCTAssertEqual(listed.map(\.id), [created.id])

        let deleted = try await engine.deleteSession(id: created.id)
        XCTAssertTrue(deleted.deleted)
        XCTAssertEqual(deleted.sessionId, created.id)
        let status = await engine.status()
        XCTAssertEqual(status.activeSessions, 0)
        XCTAssertEqual(status.cleanupPolicy, InfiniteEngine.cleanupPolicy)
    }

    func testRecordingBoundaryDoesNotDeleteInfiniteSessions() async throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
        let engine = InfiniteEngine()
        _ = try await engine.setActiveModel(.gemma4E4B1M, preload: false)
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest())

        await engine.resetForRecordingBoundary()

        let status = await engine.status()
        XCTAssertEqual(status.activeSessions, 1)
        let fetched = try await engine.session(id: created.id)
        XCTAssertEqual(fetched.id, created.id)
    }

    func testGenerateRefusesModelNotSupportedBySwiftRuntime() async throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
        let engine = InfiniteEngine()
        // Gemma4 E4B passes the RAM/tier gate but its OptiQ-4bit build does not
        // load in the Swift fork yet (#186), so generate must refuse cleanly and
        // point at that fork-fix issue.
        _ = try await engine.setActiveModel(.gemma4E4B1M, preload: false)
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest())

        do {
            _ = try await engine.generate(
                sessionID: created.id,
                request: InfiniteGenerateSessionRequest(prompt: "summarize", maxTokens: 16)
            )
            XCTFail("generate should refuse a model the Swift runtime can't run")
        } catch InfiniteError.generationUnavailable(let reason) {
            XCTAssertTrue(reason.contains("186"), "should cite the Gemma4 E4B OptiQ-quant fork-fix issue #186")
        }

        let status = await engine.status()
        XCTAssertEqual(status.activeSessions, 1)
    }

    // MARK: - RAM-tier gating (runs on any machine — no skip)

    func testRAMTierSupportsPredicate() {
        XCTAssertTrue(InfiniteRAMTier.full.supports(required: .full))
        XCTAssertTrue(InfiniteRAMTier.full.supports(required: .reduced))
        XCTAssertTrue(InfiniteRAMTier.reduced.supports(required: .reduced))
        XCTAssertFalse(InfiniteRAMTier.reduced.supports(required: .full))
        XCTAssertFalse(InfiniteRAMTier.belowMinimum.supports(required: .reduced))
        XCTAssertFalse(InfiniteRAMTier.belowMinimum.supports(required: .full))
    }

    func testRAMTierMinimumPhysicalMemoryThresholds() {
        XCTAssertEqual(InfiniteRAMTier.belowMinimum.minimumPhysicalMemoryBytes, 0)
        XCTAssertEqual(InfiniteRAMTier.reduced.minimumPhysicalMemoryBytes, 32 * 1024 * 1024 * 1024)
        XCTAssertEqual(InfiniteRAMTier.full.minimumPhysicalMemoryBytes, 64 * 1024 * 1024 * 1024)
    }

    // MARK: - Error branches (fresh engine instances for isolation)

    func testUnknownSessionIdThrowsSessionNotFound() async throws {
        let engine = InfiniteEngine()
        let missing = "does-not-exist"
        await XCTAssertThrowsInfiniteError(.sessionNotFound(missing)) {
            _ = try await engine.session(id: missing)
        }
        await XCTAssertThrowsInfiniteError(.sessionNotFound(missing)) {
            _ = try await engine.append(
                sessionID: missing,
                request: InfiniteAppendSessionRequest(text: "x")
            )
        }
        // Even with otherwise-invalid input, an unknown session is a 404, not a
        // 400 — session existence is checked before input validation.
        await XCTAssertThrowsInfiniteError(.sessionNotFound(missing)) {
            _ = try await engine.append(
                sessionID: missing,
                request: InfiniteAppendSessionRequest(text: "")
            )
        }
        await XCTAssertThrowsInfiniteError(.sessionNotFound(missing)) {
            _ = try await engine.checkpoint(
                sessionID: missing,
                request: InfiniteCheckpointSessionRequest(label: nil)
            )
        }
        await XCTAssertThrowsInfiniteError(.sessionNotFound(missing)) {
            _ = try await engine.generate(
                sessionID: missing,
                request: InfiniteGenerateSessionRequest(prompt: "x", maxTokens: 4)
            )
        }
        await XCTAssertThrowsInfiniteError(.sessionNotFound(missing)) {
            _ = try await engine.deleteSession(id: missing)
        }
    }

    func testSessionLimitRejectsBeyondMax() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        _ = try await engine.setActiveModel(.gemma4E4B1M, preload: false)
        for _ in 0..<InfiniteEngine.maxActiveSessions {
            _ = try await engine.createSession(request: InfiniteCreateSessionRequest())
        }
        await XCTAssertThrowsInfiniteError(.sessionLimitExceeded(InfiniteEngine.maxActiveSessions)) {
            _ = try await engine.createSession(request: InfiniteCreateSessionRequest())
        }
    }

    func testInvalidSessionInputs() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        _ = try await engine.setActiveModel(.gemma4E4B1M, preload: false)
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest())

        await XCTAssertThrowsInfiniteError(.invalidSessionInput("append text must not be empty")) {
            _ = try await engine.append(
                sessionID: created.id,
                request: InfiniteAppendSessionRequest(text: "")
            )
        }
        // maxTokens <= 0 is validated before the generation-unavailable boundary.
        for badMax in [0, -1] {
            await XCTAssertThrowsInfiniteError(
                .invalidSessionInput("maxTokens must be greater than zero")
            ) {
                _ = try await engine.generate(
                    sessionID: created.id,
                    request: InfiniteGenerateSessionRequest(prompt: "x", maxTokens: badMax)
                )
            }
        }
    }

    func testModelSetFailedSurfacesAdapterError() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine(backendAdapter: FailingInfiniteBackendAdapter())
        do {
            _ = try await engine.setActiveModel(.gemma4E4B1M, preload: true)
            XCTFail("preload with a failing adapter should throw modelSetFailed")
        } catch InfiniteError.modelSetFailed {
            // expected: adapter.prepare error is surfaced as modelSetFailed (HTTP 500)
        }
    }

    func testFullTierModelRefusedOnReducedTier() async throws {
        try XCTSkipUnless(
            InfiniteRAMTier.current == .reduced,
            "Exercises the reduced-tier refusal of a full-tier model"
        )
        let engine = InfiniteEngine()
        await XCTAssertThrowsInfiniteError(.modelUnavailable(InfiniteModelSelection.qwen35B1M.rawValue)) {
            _ = try await engine.setActiveModel(.qwen35B1M, preload: false)
        }
    }

    // MARK: - Wire contract (engine half of the drift guard)

    /// Asserts the engine encodes the contract keys the SDK decodes
    /// (`Tests/YoozEngineClientTests/InfiniteTypesTests.swift`). The two DTO
    /// definitions are hand-maintained copies; if the engine drops or renames a
    /// field the SDK requires, this superset check fails. (Adding a field is
    /// non-breaking — the SDK ignores unknown keys.)
    func testEngineWireShapeContainsContractKeys() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        _ = try await engine.setActiveModel(.gemma4E4B1M, preload: false)

        let encoder = JSONEncoder()
        func object<T: Encodable>(_ value: T) throws -> [String: Any] {
            let json = try JSONSerialization.jsonObject(with: encoder.encode(value))
            return (json as? [String: Any]) ?? [:]
        }
        func keys<T: Encodable>(_ value: T) throws -> Set<String> {
            Set(try object(value).keys)
        }
        // Required (non-optional) keys of the nested InfiniteResourceMetrics;
        // checked because the top-level `keys` helper only sees `resources`
        // itself, not its sub-fields.
        let resourceKeys: Set<String> = [
            "physicalMemoryBytes", "wiredMemoryLimitBytes", "requiredRAMTier",
        ]

        let models = await engine.availableModels()
        let modelInfo = try XCTUnwrap(models.first(where: \.isActive))
        XCTAssertTrue(try keys(modelInfo).isSuperset(of: [
            "id", "displayName", "description", "tier", "loadState", "isActive",
            "maxContextTokens", "nativeContextTokens", "ramTier",
            "requiresAppleSilicon", "evidenceRef",
        ]))

        let status = await engine.status()
        let statusObject = try object(status)
        XCTAssertTrue(Set(statusObject.keys).isSuperset(of: [
            "loaded", "modelId", "state", "activeSessions", "maxContextTokens",
            "ramTier", "backendKind", "cleanupPolicy", "resources",
        ]))
        XCTAssertTrue(
            Set((statusObject["resources"] as? [String: Any] ?? [:]).keys)
                .isSuperset(of: resourceKeys)
        )

        let session = try await engine.createSession(
            request: InfiniteCreateSessionRequest(label: "contract")
        )
        let sessionObject = try object(session)
        XCTAssertTrue(Set(sessionObject.keys).isSuperset(of: [
            "id", "modelId", "state", "createdAt", "updatedAt", "contextWindowTokens",
            "inputCharacters", "estimatedInputTokens", "checkpointCount",
            "cleanupPolicy", "resources",
        ]))
        XCTAssertTrue(
            Set((sessionObject["resources"] as? [String: Any] ?? [:]).keys)
                .isSuperset(of: resourceKeys)
        )
    }

    // MARK: - Helpers

    private func requireSupportedTier() throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
    }

    private func XCTAssertThrowsInfiniteError(
        _ expected: InfiniteError,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () async throws -> Void
    ) async {
        do {
            try await body()
            XCTFail("expected \(expected) but no error was thrown", file: file, line: line)
        } catch let error as InfiniteError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("expected InfiniteError.\(expected) but got \(error)", file: file, line: line)
        }
    }
}

/// Real adapter whose `prepare` always throws — exercises the
/// `modelSetFailed` path (no fabricated data; a genuine failing code path).
private struct FailingInfiniteBackendAdapter: InfiniteBackendAdapter {
    struct PrepareFailure: Error {}
    func prepare(_ descriptor: InfiniteBackendDescriptor) async throws -> InfiniteBackendHandle {
        throw PrepareFailure()
    }
}
