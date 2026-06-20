// InfiniteModuleTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import XCTest
@testable import InfiniteModule

final class InfiniteModuleTests: XCTestCase {

    private func resetEngine() async throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
        for session in await InfiniteEngine.shared.listSessions() {
            _ = try await InfiniteEngine.shared.deleteSession(id: session.id)
        }
        _ = try await InfiniteEngine.shared.setActiveModel(.gemma4E4B1M, preload: false)
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
            InfiniteModelSelection.gemma4_26BA4B1M.huggingFaceID,
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

    func testSetActiveModelWithoutPreloadReturnsActiveRow() async throws {
        try await resetEngine()
        let selection: InfiniteModelSelection =
            InfiniteRAMTier.current == .full ? .gemma4_26BA4B1M : .gemma4E4B1M
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

    func testGenerateOperationIsExplicitlyUnavailableUntilBackendInferenceLands() async throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
        let engine = InfiniteEngine()
        _ = try await engine.setActiveModel(.gemma4E4B1M, preload: false)
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest())

        do {
            _ = try await engine.generate(
                sessionID: created.id,
                request: InfiniteGenerateSessionRequest(prompt: "summarize", maxTokens: 16)
            )
            XCTFail("generate should not report synthetic text before inference is wired")
        } catch InfiniteError.generationUnavailable(let reason) {
            XCTAssertTrue(reason.contains("Phase 3"))
        }

        let status = await engine.status()
        XCTAssertEqual(status.activeSessions, 1)
    }
}
