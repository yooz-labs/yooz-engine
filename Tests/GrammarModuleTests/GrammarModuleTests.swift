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

    // MARK: - Structured matches: extractor (deterministic, no FFI)
    //
    // GrammarMatchExtractor is pure (no FFI / NLTagger), so these assert exact
    // UTF-16 offsets and lengths independent of the rule set.

    func testExtractorNoMatchOnIdenticalText() {
        let matches = GrammarMatchExtractor.matches(
            original: "the cat sat",
            corrected: "the cat sat"
        )
        XCTAssertTrue(matches.isEmpty, "identical text must yield no matches")
    }

    func testExtractorSingleReplacement() {
        let original = "I has a cat"
        let corrected = "I have a cat"
        let matches = GrammarMatchExtractor.matches(original: original, corrected: corrected)

        XCTAssertEqual(matches.count, 1)
        let m = matches[0]
        // "has" begins at UTF-16 offset 2, length 3.
        XCTAssertEqual(m.offset, 2)
        XCTAssertEqual(m.length, 3)
        XCTAssertEqual(m.original, "has")
        XCTAssertEqual(m.replacement, "have")
        XCTAssertEqual(m.ruleId, "GRAMMAR_DIFF_REPLACE")
        XCTAssertEqual(m.category, "grammar")

        // Offset/length must address the original substring under NSString.
        let ns = original as NSString
        XCTAssertEqual(ns.substring(with: NSRange(location: m.offset, length: m.length)), "has")
    }

    func testExtractorMultipleMatches() {
        // Two independent edits: "has" -> "have" and "a" -> "an".
        let original = "I has a apple"
        let corrected = "I have an apple"
        let matches = GrammarMatchExtractor.matches(original: original, corrected: corrected)

        XCTAssertEqual(matches.count, 2, "two distinct edits should yield two matches")

        let first = matches[0]
        XCTAssertEqual(first.original, "has")
        XCTAssertEqual(first.replacement, "have")
        XCTAssertEqual(first.offset, 2)
        XCTAssertEqual(first.length, 3)

        let second = matches[1]
        XCTAssertEqual(second.original, "a")
        XCTAssertEqual(second.replacement, "an")
        // "a" sits at UTF-16 offset 6, length 1, in the original.
        XCTAssertEqual(second.offset, 6)
        XCTAssertEqual(second.length, 1)

        let ns = original as NSString
        XCTAssertEqual(ns.substring(with: NSRange(location: second.offset, length: second.length)), "a")
    }

    func testExtractorDeletionHasEmptyReplacement() {
        // Collapsing a repeated word deletes the second "to".
        let original = "we need to to go"
        let corrected = "we need to go"
        let matches = GrammarMatchExtractor.matches(original: original, corrected: corrected)

        XCTAssertEqual(matches.count, 1)
        let m = matches[0]
        XCTAssertEqual(m.replacement, "", "deletion must report an empty replacement")
        XCTAssertEqual(m.original, "to")
        XCTAssertEqual(m.ruleId, "GRAMMAR_DIFF_DELETE")

        // The reported range must cover a "to" token in the original.
        let ns = original as NSString
        XCTAssertEqual(ns.substring(with: NSRange(location: m.offset, length: m.length)), "to")
    }

    func testExtractorUnicodeEmojiOffsetsAreUTF16() {
        // The cat emoji is a single Unicode scalar that occupies TWO UTF-16
        // code units (a surrogate pair). The edit after it must therefore be
        // reported at a UTF-16 offset that accounts for both units.
        let original = "🐱 I has a cat"
        let corrected = "🐱 I have a cat"
        let matches = GrammarMatchExtractor.matches(original: original, corrected: corrected)

        XCTAssertEqual(matches.count, 1)
        let m = matches[0]
        XCTAssertEqual(m.original, "has")
        XCTAssertEqual(m.replacement, "have")

        // UTF-16 layout: 🐱 = units 0-1, space = 2, "I" = 3, space = 4,
        // "has" starts at unit 5.
        XCTAssertEqual(m.offset, 5)
        XCTAssertEqual(m.length, 3)

        let ns = original as NSString
        XCTAssertEqual(ns.length, 14, "emoji should contribute 2 UTF-16 units")
        XCTAssertEqual(ns.substring(with: NSRange(location: m.offset, length: m.length)), "has")
    }

    func testExtractorOffsetsRemainInOriginalCoordinatesAfterEarlierEdit() {
        // An earlier edit changes downstream lengths; later matches must still
        // be expressed in ORIGINAL-text coordinates, not corrected-text ones.
        let original = "a apple is a orange"
        let corrected = "an apple is an orange"
        let matches = GrammarMatchExtractor.matches(original: original, corrected: corrected)

        XCTAssertEqual(matches.count, 2)
        let ns = original as NSString
        for m in matches {
            XCTAssertEqual(m.original, "a")
            XCTAssertEqual(m.replacement, "an")
            XCTAssertEqual(
                ns.substring(with: NSRange(location: m.offset, length: m.length)),
                "a",
                "offset must address the original text even after an earlier edit"
            )
        }
        // Second "a" is at UTF-16 offset 11 in the ORIGINAL (not shifted by the
        // first "a"->"an" expansion).
        XCTAssertEqual(matches[1].offset, 11)
    }

    // MARK: - Structured matches: checkDetailed (real FFI)

    func testCheckDetailedNoMatchWhenClean() async {
        let engine = GrammarEngine.shared
        let detailed = await engine.checkDetailed(
            text: "The cat sat on the mat.",
            categories: nil,
            usePOS: true
        )
        // Clean input: corrected == original, so no matches.
        if detailed.result == "The cat sat on the mat." {
            XCTAssertTrue(detailed.matches.isEmpty)
        }
    }

    func testCheckDetailedParityWithCheck() async {
        let engine = GrammarEngine.shared
        let text = "I are happy and they is sad"
        let plain = await engine.check(text: text, categories: nil, usePOS: true)
        let detailed = await engine.checkDetailed(text: text, categories: nil, usePOS: true)

        XCTAssertEqual(detailed.result, plain.result,
                       "checkDetailed must not change correction behavior")
        XCTAssertEqual(detailed.correctionsApplied, plain.correctionsApplied)
    }

    func testCheckDetailedProducesAnchoredMatches() async {
        let engine = GrammarEngine.shared
        let text = "I has a apple"
        let detailed = await engine.checkDetailed(text: text, categories: nil, usePOS: true)

        // Only assert structure when the rule set actually corrected something;
        // keeps the test robust to rule-set evolution while still verifying the
        // offset contract whenever matches exist.
        guard detailed.result != text else {
            XCTAssertTrue(detailed.matches.isEmpty)
            return
        }
        XCTAssertFalse(detailed.matches.isEmpty, "a corrected result must surface matches")

        let ns = text as NSString
        for m in detailed.matches {
            XCTAssertGreaterThanOrEqual(m.offset, 0)
            XCTAssertLessThanOrEqual(m.offset + m.length, ns.length)
            // The reported original substring must match the original-text slice.
            let slice = ns.substring(with: NSRange(location: m.offset, length: m.length))
            XCTAssertEqual(slice, m.original)
            XCTAssertFalse(m.message.isEmpty)
            XCTAssertFalse(m.ruleId.isEmpty)
        }
    }

    func testCheckDetailedFreeVsProCategoryFiltering() async {
        let engine = GrammarEngine.shared
        // A POS-dependent correction. Free tier excludes the verb category and
        // uses simple (non-POS) rules; pro applies the full set with POS.
        let text = "I has a apple"

        let free = await engine.checkDetailed(
            text: text,
            categories: ["basic", "grammar", "articles", "informal"],
            usePOS: false
        )
        let pro = await engine.checkDetailed(
            text: text,
            categories: nil,
            usePOS: true
        )

        // Category filtering must flow through to matches: a narrower category
        // set cannot produce MORE matches than the full set on the same input.
        XCTAssertLessThanOrEqual(free.matches.count, pro.matches.count,
                                 "free-tier category subset must not exceed pro matches")

        // Whatever matches each tier reports, every reported range must be a
        // valid original-text slice.
        let ns = text as NSString
        for m in free.matches + pro.matches {
            let slice = ns.substring(with: NSRange(location: m.offset, length: m.length))
            XCTAssertEqual(slice, m.original)
        }
    }
}
