// TouchUpFidelityHelperTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Ungated, table-driven known-answer tests for TouchUpFidelityEvalTests's
// pure helper functions and fixture table, plus a pin on the production
// sampling constants/maxTokens sizing those helpers exist to validate
// (engine#279 review items 5/7/8/11/12/13 — "all seven detector helpers
// execute only behind the model gate: zero CI coverage"). No mocks, no
// model load: every test here is a pure function over literal strings and
// numbers, so it runs on every CI invocation.

import XCTest
@testable import LLMModule

final class TouchUpFidelityHelperTests: XCTestCase {

    // MARK: - (a) introducesDuplicateClause

    func testIntroducesDuplicateClauseDetectsARealClone() {
        let input = "I need to check my credit card statement before I decide, or something like that."
        let output = "I need to check my credit card statement before I decide, "
            + "check my credit card statement, or something like that."
        XCTAssertTrue(TouchUpFidelityEvalTests.introducesDuplicateClause(input: input, output: output))
    }

    func testIntroducesDuplicateClauseAllowsSameRepeatCountAsInput() {
        // The input already repeats "I don't wanna" twice; an output
        // keeping the SAME count must not fire (no NEW clone introduced).
        let input = "I don't wanna, I don't wanna move beyond that."
        let output = "I don't wanna, I don't wanna move beyond that."
        XCTAssertFalse(TouchUpFidelityEvalTests.introducesDuplicateClause(input: input, output: output))
    }

    func testIntroducesDuplicateClauseAllowsLegitimateDedup() {
        // Model deduplicating a repeated clause DECREASES the count — fine.
        let input = "I don't wanna, I don't wanna move beyond that."
        let output = "I don't want to move beyond that."
        XCTAssertFalse(TouchUpFidelityEvalTests.introducesDuplicateClause(input: input, output: output))
    }

    // MARK: - (b) wordCountRatio

    func testWordCountRatioStripsFillersFromInputSide() {
        let input = "So um I was uh thinking about it."
        let output = "I was thinking about it."
        // Input words minus "um"/"uh": "so i was thinking about it" = 6.
        // Output: "i was thinking about it" = 5. Ratio 5/6.
        XCTAssertEqual(
            TouchUpFidelityEvalTests.wordCountRatio(input: input, output: output), 5.0 / 6.0, accuracy: 0.0001
        )
    }

    func testWordCountRatioEmptyInputReturnsOne() {
        XCTAssertEqual(TouchUpFidelityEvalTests.wordCountRatio(input: "", output: "anything"), 1.0)
    }

    // MARK: - (c) introducesDigitizedSingleDigit

    func testIntroducesDigitizedSingleDigitFiresOnNewDigit() {
        XCTAssertTrue(
            TouchUpFidelityEvalTests.introducesDigitizedSingleDigit(input: "One is correct.", output: "1 is correct.")
        )
    }

    func testIntroducesDigitizedSingleDigitAllowsPreExistingDigit() {
        XCTAssertFalse(
            TouchUpFidelityEvalTests.introducesDigitizedSingleDigit(input: "Port 5 is open.", output: "Port 5 is open.")
        )
    }

    func testIntroducesDigitizedSingleDigitDoesNotFireOnMultiDigitNumeral() {
        // "10" is not a standalone single-digit numeral (0-9); the check
        // must not misfire on it (its tokens are "1" and "0" only if
        // split, but tokenize() keeps "10" as one alphanumeric run).
        XCTAssertFalse(
            TouchUpFidelityEvalTests.introducesDigitizedSingleDigit(input: "ten units", output: "10 units")
        )
    }

    // MARK: - (f) introducesNegation

    func testIntroducesNegationFiresOnInsertedNot() {
        XCTAssertTrue(
            TouchUpFidelityEvalTests.introducesNegation(input: "One is correct.", output: "One is not correct.")
        )
    }

    func testIntroducesNegationFiresOnInsertedContraction() {
        XCTAssertTrue(
            TouchUpFidelityEvalTests.introducesNegation(
                input: "I think that's ready.", output: "I don't think that's ready."
            )
        )
    }

    func testIntroducesNegationAllowsLegitimateDrop() {
        // Self-correction dropping a retracted negation is fine — a
        // DECREASE in negation count is never flagged.
        XCTAssertFalse(
            TouchUpFidelityEvalTests.introducesNegation(
                input: "It's not, I mean it's fine now.", output: "It's fine now."
            )
        )
    }

    func testIntroducesNegationDoesNotFireOnUnchangedCount() {
        XCTAssertFalse(
            TouchUpFidelityEvalTests.introducesNegation(
                input: "The data is not normalized, but it is usable.",
                output: "It is usable, though it is not fully normalized."
            )
        )
    }

    // MARK: - (g) similarity

    func testSimilarityIdenticalTextIsOne() {
        XCTAssertEqual(TouchUpFidelityEvalTests.similarity("This is fine as is.", "This is fine as is."), 1.0)
    }

    func testSimilarityCompletelyDifferentTextIsZero() {
        XCTAssertEqual(TouchUpFidelityEvalTests.similarity("apple banana cherry", "xylophone zebra yak"), 0.0)
    }

    func testSimilarityBothEmptyIsOne() {
        XCTAssertEqual(TouchUpFidelityEvalTests.similarity("", ""), 1.0)
    }

    func testSimilarityOneEmptyIsZero() {
        XCTAssertEqual(TouchUpFidelityEvalTests.similarity("", "something"), 0.0)
    }

    // MARK: - (e) missingRequiredContent

    func testMissingRequiredContentNormalizesCurlyApostrophe() {
        // Output has a curly U+2019 apostrophe; the fixture requirement
        // uses a straight one. Must not false-fail.
        let missing = TouchUpFidelityEvalTests.missingRequiredContent(
            required: ["people's"],
            output: "Don\u{2019}t change other people\u{2019}s code."
        )
        XCTAssertTrue(missing.isEmpty, "expected no missing terms, got \(missing)")
    }

    func testMissingRequiredContentIsCaseInsensitive() {
        let missing = TouchUpFidelityEvalTests.missingRequiredContent(
            required: ["intel"],
            output: "I use a Linux INTEL system."
        )
        XCTAssertTrue(missing.isEmpty, "expected no missing terms, got \(missing)")
    }

    func testMissingRequiredContentReportsWhatIsActuallyMissing() {
        let missing = TouchUpFidelityEvalTests.missingRequiredContent(
            required: ["credit card statement", "unrelated phrase"],
            output: "I need to check my credit card statement before deciding."
        )
        XCTAssertEqual(missing, ["unrelated phrase"])
    }

    // MARK: - tokenize (pinned contraction-splitting behavior)

    func testTokenizeSplitsContractionsOnApostrophe() {
        // Pinned, not a bug: tokenize() treats the apostrophe as a
        // separator (CharacterSet.alphanumerics.inverted), so "people's"
        // becomes two tokens. Assertions that need the apostrophe kept
        // (negation's "n't" detection) use a dedicated, separate
        // tokenizer internally rather than this one.
        XCTAssertEqual(TouchUpFidelityEvalTests.tokenize("people's"), ["people", "s"])
    }

    func testTokenizeLowercasesAndDropsPunctuation() {
        XCTAssertEqual(TouchUpFidelityEvalTests.tokenize("Hello, World!"), ["hello", "world"])
    }

    // MARK: - Fixture-table sanity (review item 12)

    private static var allFixtures: [TouchUpFidelityEvalTests.Fixture] {
        TouchUpFidelityEvalTests.qualityFixtures + TouchUpFidelityEvalTests.lightFixtures
    }

    func testAllFixtureIDsAreUniqueWithinEachList() {
        // Checked per-list, not on the concatenation: `lightFixtures`
        // intentionally reuses three `qualityFixtures` ids (same fixture,
        // `lightFull` swapped in for `qualityFull`) — that reuse is by
        // design, not a collision.
        for fixtures in [TouchUpFidelityEvalTests.qualityFixtures, TouchUpFidelityEvalTests.lightFixtures] {
            let ids = fixtures.map(\.id)
            XCTAssertEqual(ids.count, Set(ids).count, "duplicate fixture id(s) found: \(ids)")
        }
    }

    func testAllFixtureInputsAreNonEmpty() {
        for fixture in Self.allFixtures {
            XCTAssertFalse(
                fixture.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "fixture '\(fixture.id)' has an empty input"
            )
        }
    }

    func testEveryMustContainTermIsPresentInItsOwnFixtureInput() {
        // A `mustContain` term that isn't even present in the INPUT would
        // make the fixture untestable (nothing there to preserve) — this
        // enforces every requirement traces back to real fixture content,
        // under the same normalization `missingRequiredContent` itself uses.
        for fixture in Self.allFixtures {
            let missing = TouchUpFidelityEvalTests.missingRequiredContent(
                required: fixture.mustContain, output: fixture.input
            )
            XCTAssertTrue(
                missing.isEmpty,
                "fixture '\(fixture.id)' requires \(missing) but its own input doesn't contain them"
            )
        }
    }

    // MARK: - Production constant/boundary pinning (review item 13)

    func testComputeMaxTokensBoundaries() {
        XCTAssertEqual(MLXLLMBackend.computeMaxTokens(userTokenCount: 0), 160)
        XCTAssertEqual(MLXLLMBackend.computeMaxTokens(userTokenCount: 96), 160)
        XCTAssertEqual(MLXLLMBackend.computeMaxTokens(userTokenCount: 97), 161)
    }

    func testSamplingConstantsArePinned() {
        XCTAssertEqual(MLXLLMBackend.touchUpTemperature, 0)
        XCTAssertEqual(MLXLLMBackend.touchUpRepetitionPenalty, 1.1)
        XCTAssertEqual(MLXLLMBackend.touchUpRepetitionContextSize, 64)
    }
}
