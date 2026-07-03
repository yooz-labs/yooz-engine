// HarnessArgumentsTests.swift
// XPCHarnessSupportTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest

@testable import XPCHarnessSupport

final class HarnessArgumentsTests: XCTestCase {

    // MARK: - intArgument

    func testIntArgumentParsesValidValue() {
        XCTAssertEqual(HarnessArguments.intArgument(["--warm-runs", "3"], flag: "--warm-runs"), 3)
    }

    func testIntArgumentReturnsNilWhenFlagMissing() {
        XCTAssertNil(HarnessArguments.intArgument(["--batch-wav", "x.wav"], flag: "--warm-runs"))
    }

    func testIntArgumentReturnsNilWhenFlagIsLastArgument() {
        XCTAssertNil(HarnessArguments.intArgument(["--warm-runs"], flag: "--warm-runs"))
    }

    func testIntArgumentParsesNegativeValue() {
        // intArgument itself does not reject negatives — that is a
        // call-site policy decision (see doc comment). It must still
        // parse correctly so the caller can validate and report it.
        XCTAssertEqual(HarnessArguments.intArgument(["--warm-runs", "-1"], flag: "--warm-runs"), -1)
    }

    func testIntArgumentReturnsNilForGarbageValue() {
        XCTAssertNil(HarnessArguments.intArgument(["--warm-runs", "not-a-number"], flag: "--warm-runs"))
    }

    // MARK: - sentenceCoverage

    func testSentenceCoverageZeroExpectedReturnsZeroWithoutTrapping() {
        XCTAssertEqual(HarnessArguments.sentenceCoverage("anything at all", upTo: 0), 0)
    }

    func testSentenceCoverageNegativeExpectedReturnsZeroWithoutTrapping() {
        XCTAssertEqual(HarnessArguments.sentenceCoverage("anything at all", upTo: -5), 0)
    }

    func testSentenceCoveragePartialMatch() {
        let text = "This is sentence number one. This is sentence number two."
        XCTAssertEqual(HarnessArguments.sentenceCoverage(text, upTo: 3), 2)
    }

    func testSentenceCoverageFullMatchDigitForm() {
        let text = "number 1. number 2. number 3."
        XCTAssertEqual(HarnessArguments.sentenceCoverage(text, upTo: 3), 3)
    }

    func testSentenceCoverageFullMatchOrdinalWordForm() {
        let text = "This is sentence number one. This is sentence number two."
        XCTAssertEqual(HarnessArguments.sentenceCoverage(text, upTo: 2), 2)
    }
}
