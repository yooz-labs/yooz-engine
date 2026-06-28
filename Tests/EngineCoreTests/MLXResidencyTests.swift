// MLXResidencyTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Pure-logic coverage for `MLXResidency`, the per-category MLX cache-budget
/// coordinator. No Metal: every test drives an isolated instance and asserts
/// the returned `MLXResidencyDirective`, so the suite runs under a plain
/// `swift test` where `default.metallib` is absent.
final class MLXResidencyTests: XCTestCase {
    private var perCategory: Int { EngineConfig.mlxCacheBudgetPerCategoryBytes }

    func testBudgetScalesWithResidentCategoryCount() {
        XCTAssertEqual(
            MLXResidency.budgetBytes(forResidentCount: 0), perCategory,
            "empty set floors at one category's budget"
        )
        XCTAssertEqual(MLXResidency.budgetBytes(forResidentCount: 1), perCategory)
        XCTAssertEqual(MLXResidency.budgetBytes(forResidentCount: 2), 2 * perCategory)
        XCTAssertEqual(MLXResidency.budgetBytes(forResidentCount: 3), 3 * perCategory)
    }

    func testRegisterGrowsBudgetPerCategoryAndNeverFlushes() {
        let residency = MLXResidency()

        let stt = residency.register(.stt)
        XCTAssertEqual(stt.cacheLimitBytes, perCategory)
        XCTAssertFalse(stt.flush)

        let touchUp = residency.register(.touchUp)
        XCTAssertEqual(
            touchUp.cacheLimitBytes, 2 * perCategory,
            "two coexisting categories sum their per-category budgets"
        )
        XCTAssertFalse(touchUp.flush)
    }

    func testSecondModelInSameCategoryIsBudgetIdempotent() {
        let residency = MLXResidency()
        _ = residency.register(.stt)
        let again = residency.register(.stt)
        XCTAssertEqual(
            again.cacheLimitBytes, perCategory,
            "a second model in an already-resident category does not grow the budget"
        )
        XCTAssertTrue(residency.isResident(.stt))
    }

    func testSttTeardownDoesNotFlushWhileTouchUpResident() {
        let residency = MLXResidency()
        _ = residency.register(.stt)
        _ = residency.register(.touchUp)

        let afterStt = residency.unregister(.stt)
        XCTAssertEqual(
            afterStt.cacheLimitBytes, perCategory,
            "budget trims to the one remaining category"
        )
        XCTAssertFalse(
            afterStt.flush,
            "must NOT flush while the LLM is still resident (the cross-category stomp)"
        )
        XCTAssertFalse(residency.isResident(.stt))
        XCTAssertTrue(residency.isResident(.touchUp))
    }

    func testTouchUpTierSwitchOverlapKeepsCategoryResident() {
        // Mirrors the load-new-then-evict-old TouchUp tier switch: two
        // `.touchUp` registrations, then the old tier unloads. The category
        // must stay resident (refcount), so the surviving tier's buffers are
        // NOT flushed — the bug a plain set would introduce.
        let residency = MLXResidency()
        _ = residency.register(.touchUp) // tier A loaded
        _ = residency.register(.touchUp) // tier B loaded during the switch

        let evictA = residency.unregister(.touchUp) // tier A evicted
        XCTAssertFalse(
            evictA.flush,
            "evicting one of two LLM tiers must not flush the survivor"
        )
        XCTAssertTrue(residency.isResident(.touchUp))
        XCTAssertEqual(evictA.cacheLimitBytes, perCategory)

        let evictB = residency.unregister(.touchUp) // tier B evicted -> empty
        XCTAssertTrue(evictB.flush, "the last model out flushes")
        XCTAssertFalse(residency.isResident(.touchUp))
    }

    func testFlushOnlyWhenLastCategoryRemoved() {
        let residency = MLXResidency()
        _ = residency.register(.touchUp)

        let last = residency.unregister(.touchUp)
        XCTAssertTrue(last.flush, "flush when the resident set becomes empty")
        XCTAssertEqual(last.cacheLimitBytes, perCategory, "limit floors at one budget")
        XCTAssertFalse(residency.isResident(.touchUp))
    }

    func testOverBalancedUnregisterStaysEmpty() {
        let residency = MLXResidency()
        let directive = residency.unregister(.stt) // never registered
        XCTAssertTrue(directive.flush, "an already-empty set reports the empty/flush signal")
        XCTAssertEqual(directive.cacheLimitBytes, perCategory)
        XCTAssertFalse(residency.isResident(.stt))
    }

    func testCurrentCacheLimitTracksResidentSet() {
        let residency = MLXResidency()
        XCTAssertEqual(residency.currentCacheLimitBytes(), perCategory)
        _ = residency.register(.stt)
        XCTAssertEqual(residency.currentCacheLimitBytes(), perCategory)
        _ = residency.register(.touchUp)
        XCTAssertEqual(residency.currentCacheLimitBytes(), 2 * perCategory)
    }
}
