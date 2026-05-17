// JSONParsingTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Unit tests for `parseProofreadResponse` and `parseValidateResponse`,
// including the placeholder-echo guard added for #113. These are pure
// string-in / tuple-out functions with no model dependencies, so they
// run unconditionally on every CI invocation.

import XCTest
@testable import LLMModule

final class JSONParsingTests: XCTestCase {

    // MARK: - parseProofreadResponse: happy paths

    func testParseProofreadDirectJSON() {
        let response = "{\"result\": \"Hello world.\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello world.")
        XCTAssertTrue(success)
    }

    func testParseProofreadJSONWithSurroundingText() {
        // Models sometimes emit the JSON object alongside prose.
        let response = "Here you go: {\"result\": \"It's ready.\"} done."
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "It's ready.")
        XCTAssertTrue(success)
    }

    func testParseProofreadTrimsLeadingTrailingWhitespace() {
        let response = "\n\n  {\"result\": \"Trimmed.\"}  \n"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Trimmed.")
        XCTAssertTrue(success)
    }

    // MARK: - parseProofreadResponse: placeholder echo (#113)

    func testParseProofreadRejectsPlaceholderEcho() {
        // The Light and Quality prompts end with
        // `Always respond with {"result": "corrected text"}.`
        // Small models occasionally echo this shape example verbatim;
        // the guard must reject it and return the fallback.
        let response = "{\"result\": \"corrected text\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "original input")
        XCTAssertEqual(text, "original input",
                       "placeholder echo must fall through to fallback, not surface to the user")
        XCTAssertFalse(success)
    }

    func testParseProofreadRejectsPlaceholderEchoCaseInsensitive() {
        // Models may emit any casing of the echoed placeholder; guard is
        // intentionally case-insensitive.
        let response = "{\"result\": \"Corrected Text\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "original input")
        XCTAssertEqual(text, "original input")
        XCTAssertFalse(success)
    }

    func testParseProofreadRejectsPlaceholderEchoWithWhitespace() {
        let response = "{\"result\": \"  corrected text  \"}"
        let (text, success) = parseProofreadResponse(response, fallback: "original input")
        XCTAssertEqual(text, "original input")
        XCTAssertFalse(success)
    }

    func testParseProofreadRejectsPlaceholderEchoFromExtractedJSON() {
        // Echo via the extractJSON path (extra prose around the object).
        let response = "Sure: {\"result\": \"corrected text\"} ok."
        let (text, success) = parseProofreadResponse(response, fallback: "original input")
        XCTAssertEqual(text, "original input")
        XCTAssertFalse(success)
    }

    func testParseProofreadAcceptsResultContainingPlaceholderSubstring() {
        // The user legitimately dictated a sentence containing the placeholder
        // text. Substring match would create a false positive; exact-match
        // guard must allow this through.
        let response = "{\"result\": \"Here is the corrected text you asked for.\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Here is the corrected text you asked for.")
        XCTAssertTrue(success)
    }

    // MARK: - parseProofreadResponse: failure paths

    func testParseProofreadReturnsFallbackOnInvalidJSON() {
        let response = "this is not json at all"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "fallback")
        XCTAssertFalse(success)
    }

    func testParseProofreadReturnsFallbackOnWrongShape() {
        // Valid JSON but missing `result` key.
        let response = "{\"text\": \"wrong shape\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "fallback")
        XCTAssertFalse(success)
    }

    func testParseProofreadReturnsFallbackOnEmptyInput() {
        let (text, success) = parseProofreadResponse("", fallback: "fallback")
        XCTAssertEqual(text, "fallback")
        XCTAssertFalse(success)
    }

    // MARK: - parseValidateResponse: happy paths

    func testParseValidateDirectJSON() {
        let response = "{\"result\": \"Hello world.\", \"keep\": [true, false]}"
        let (text, keep, success) = parseValidateResponse(
            response,
            fallback: "fallback",
            numReplacements: 2
        )
        XCTAssertEqual(text, "Hello world.")
        XCTAssertEqual(keep, [true, false])
        XCTAssertTrue(success)
    }

    func testParseValidateNormalizesKeepCount() {
        // Returned keep array is shorter than expected; padding fills with
        // `true` (keep by default if model omits the decision).
        let response = "{\"result\": \"text\", \"keep\": [false]}"
        let (_, keep, success) = parseValidateResponse(
            response,
            fallback: "fallback",
            numReplacements: 3
        )
        XCTAssertEqual(keep, [false, true, true])
        XCTAssertTrue(success)
    }

    // MARK: - parseValidateResponse: placeholder echo (#113)

    func testParseValidateRejectsPlaceholderEcho() {
        // Validate-path echo: the guard must trip even when the model
        // produces a structurally-valid ValidateResponse whose `result`
        // is the placeholder.
        let response = "{\"result\": \"corrected text\", \"keep\": [true]}"
        let (text, keep, success) = parseValidateResponse(
            response,
            fallback: "original input",
            numReplacements: 1
        )
        XCTAssertEqual(text, "original input",
                       "placeholder echo must fall through to fallback for the validate path too")
        // Default keep on parse failure is all-false (revert all replacements).
        XCTAssertEqual(keep, [false])
        XCTAssertFalse(success)
    }

    func testParseValidateRejectsPlaceholderEchoCaseInsensitive() {
        let response = "{\"result\": \"CORRECTED TEXT\", \"keep\": [true, true]}"
        let (text, _, success) = parseValidateResponse(
            response,
            fallback: "original",
            numReplacements: 2
        )
        XCTAssertEqual(text, "original")
        XCTAssertFalse(success)
    }

    // MARK: - parseValidateResponse: failure paths

    func testParseValidateReturnsFallbackOnInvalidJSON() {
        let response = "not json"
        let (text, keep, success) = parseValidateResponse(
            response,
            fallback: "fallback",
            numReplacements: 2
        )
        XCTAssertEqual(text, "fallback")
        XCTAssertEqual(keep, [false, false],
                       "parse failure must revert all replacements (all-false keep)")
        XCTAssertFalse(success)
    }
}
