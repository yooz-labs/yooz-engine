// GrammarModuleTests.swift
// GrammarModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// These tests hit the real Rust FFI (YoozTextCleanup.xcframework) and the
// real NLTagger. No mocks, per yooz project policy.

import XCTest
import EngineCore
@testable import GrammarModule

final class GrammarModuleTests: XCTestCase {

    // MARK: - Rule availability

    func testRulesLoadedFromFFI() async {
        let engine = GrammarEngine.shared
        XCTAssertTrue(engine.isAvailable, "Rust FFI failed to load any rules")
        XCTAssertGreaterThan(engine.ruleCount, 1_500, "Expected 1,560+ rules; got \(engine.ruleCount)")
        XCTAssertGreaterThan(engine.simpleRuleCount, 0)
        XCTAssertGreaterThan(engine.posRuleCount, 0)
        XCTAssertFalse(engine.version.isEmpty)
        XCTAssertNotEqual(engine.version, "unavailable")
    }

    // MARK: - Correction behavior

    func testSubjectVerbAgreement() async {
        let engine = GrammarEngine.shared
        let result = await engine.check(
            text: "I are happy and they is sad",
            categories: nil,
            usePOS: true
        )
        XCTAssertNotEqual(result.result, "I are happy and they is sad",
                          "SVA rule did not fire")
        XCTAssertGreaterThan(result.correctionsApplied, 0)
    }

    func testRepeatedWordsRemoved() async {
        let engine = GrammarEngine.shared
        let result = await engine.check(
            text: "we need to to understand this",
            categories: nil,
            usePOS: false
        )
        XCTAssertFalse(result.result.contains("to to"),
                       "programmatic WORD_REPEAT rule should collapse duplicate 'to'")
    }

    func testEmptyTextPassesThrough() async {
        let engine = GrammarEngine.shared
        let result = await engine.check(text: "", categories: nil)
        XCTAssertEqual(result.result, "")
        XCTAssertEqual(result.correctionsApplied, 0)
    }

    func testUnknownCategoryFallsBackToAll() async {
        let engine = GrammarEngine.shared
        let resultNil = await engine.check(
            text: "I are happy",
            categories: nil,
            usePOS: true
        )
        let resultBogus = await engine.check(
            text: "I are happy",
            categories: ["does_not_exist_xyz"],
            usePOS: true
        )
        XCTAssertEqual(resultBogus.result, resultNil.result,
                       "bogus category should degrade to 'all categories'")
    }

    // MARK: - Tier routing

    func testFreeTierHasFewerCategoriesThanPro() async {
        let engine = GrammarEngine.shared
        let freeCats = await engine.categoriesForTier(.free)
        let proCats = await engine.categoriesForTier(.pro)
        XCTAssertLessThan(freeCats.count, proCats.count,
                          "free tier should be a subset")
    }

    func testPremiumTierMatchesPro() async {
        let engine = GrammarEngine.shared
        let pro = await engine.categoriesForTier(.pro)
        let premium = await engine.categoriesForTier(.premium)
        XCTAssertEqual(pro.count, premium.count,
                       "premium == pro at engine layer; LLM fallback lives in TouchUp")
    }

    // MARK: - AIModule conformance

    func testAIModuleName() {
        XCTAssertEqual(GrammarEngine.name, "grammar")
    }

    func testAIModuleIsReadyReflectsAvailability() {
        XCTAssertEqual(GrammarEngine.shared.isReady, GrammarEngine.shared.isAvailable)
    }

    func testHealthCheckReportsRuleCounts() async {
        let health = await GrammarEngine.shared.healthCheck()
        XCTAssertTrue(health.loaded)
        XCTAssertNil(health.error)
        XCTAssertEqual(health.detail["rules_total"], String(GrammarEngine.shared.totalRuleCount))
        XCTAssertEqual(health.detail["library_version"], GrammarEngine.shared.version)
    }

    func testHealthCheckDetailKeysPresent() async {
        let health = await GrammarEngine.shared.healthCheck()
        let expected: Set<String> = ["rules_total", "rules_simple", "rules_pos",
                                     "rules_programmatic", "library_version"]
        XCTAssertEqual(Set(health.detail.keys), expected)
    }
}
