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
        let reducedExpected: ModelLoadState =
            InfiniteRAMTier.current == .belowMinimum ? .unavailable : .available
        let fullExpected: ModelLoadState =
            InfiniteRAMTier.current == .full ? .available : .unavailable

        XCTAssertEqual(byID["gemma4-e4b-1m"]?.loadState, reducedExpected)
        XCTAssertEqual(byID["qwen3-35b-1m"]?.loadState, fullExpected)
        XCTAssertEqual(byID["s3-retrieval"]?.loadState, fullExpected)
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

    func testPreloadFailsUntilBackendWiringLands() async throws {
        try await resetEngine()
        do {
            _ = try await InfiniteEngine.shared.setActiveModel(.gemma4E4B1M, preload: true)
            XCTFail("Phase 1 scaffold must not pretend backend preload works")
        } catch let error as InfiniteError {
            XCTAssertEqual(
                error,
                .modelSetFailed("Infinite backend loading is not implemented in Phase 1")
            )
        }
    }
}
