// JSONParsingDefensiveTests.swift
// LLMModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Defensive parsing coverage for `parseProofreadResponse` and
// `parseValidateResponse` against the malformed LLM output shapes observed
// on long inputs from Yooz-Light (engine #134):
//
//   - markdown-fenced JSON (```json ... ```)
//   - preamble text before the JSON object
//   - multi-object responses ({"result":"A"}{"result":"B"})
//   - prose surrounding a single JSON object
//
// As with `JSONParsingTests`, these are pure string-in / tuple-out functions
// with no model dependency, so they run on every CI invocation without any
// model bundle present.

import XCTest
@testable import LLMModule

final class JSONParsingDefensiveTests: XCTestCase {

    // MARK: - Markdown fence wrapping

    func testParseProofreadStripsJSONMarkdownFence() {
        // Model emits a ```json ... ``` fenced block. The fence must be
        // stripped before the JSON is decoded.
        let response = "```json\n{\"result\": \"Hello.\"}\n```"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello.")
        XCTAssertTrue(success)
    }

    func testParseProofreadStripsBareMarkdownFence() {
        // No language tag on the opening fence.
        let response = "```\n{\"result\": \"Hello.\"}\n```"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello.")
        XCTAssertTrue(success)
    }

    func testParseProofreadStripsFenceWithProseInside() {
        // Fence wraps both prose and the JSON object. Strip the fence,
        // then the balanced-object scan inside picks the JSON out.
        let response = "```json\nHere you go: {\"result\": \"Hello.\"}\n```"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello.")
        XCTAssertTrue(success)
    }

    // MARK: - Preamble text

    func testParseProofreadHandlesPreambleText() {
        // Model emits a single object preceded by an explanatory sentence.
        let response = "Here is the corrected text:\n{\"result\": \"Hello.\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello.")
        XCTAssertTrue(success)
    }

    // MARK: - Multi-object responses

    func testParseProofreadPicksFirstOfMultipleObjects() {
        // Model concatenates two `{"result": ...}` objects. We commit to the
        // first parseable one rather than letting the lastIndex(of:) heuristic
        // merge them into an undecodable blob.
        let response = "{\"result\": \"Hello.\"}{\"result\": \"World.\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello.",
                       "multi-object response must yield the first decodable result")
        XCTAssertTrue(success)
    }

    func testParseProofreadSkipsCorruptedFirstObjectIfRecoverable() {
        // The first balanced object has the wrong shape (missing `result`);
        // the scanner should fall through to the next candidate.
        let response = "{\"wrong\": \"shape\"}{\"result\": \"valid.\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "valid.")
        XCTAssertTrue(success)
    }

    // MARK: - Prose surrounding a single object

    func testParseProofreadHandlesCorruptedPrefixBeforeValidJSON() {
        // Prose with no braces precedes a single valid object.
        let response = "corrupted{\"result\": \"valid.\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "valid.")
        XCTAssertTrue(success)
    }

    // MARK: - Placeholder echo guard still fires through new code path

    func testParseProofreadStillRejectsPlaceholderEcho() {
        // Regression guard for engine #113: the placeholder-echo rejection
        // must continue to fire after defensive-parsing changes, including
        // when the placeholder arrives wrapped in a markdown fence.
        let response = "```json\n{\"result\": \"corrected text\"}\n```"
        let (text, success) = parseProofreadResponse(response, fallback: "original input")
        XCTAssertEqual(text, "original input",
                       "placeholder echo guard must still trip on fenced responses")
        XCTAssertFalse(success)
    }

    func testParseProofreadStillRejectsBarePlaceholderEcho() {
        // The non-fenced placeholder-echo case from JSONParsingTests, repeated
        // here so this file is self-contained as a defensive-parsing regression
        // suite.
        let response = "{\"result\": \"corrected text\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "original input")
        XCTAssertEqual(text, "original input")
        XCTAssertFalse(success)
    }

    // MARK: - Real failure paths

    func testParseProofreadReturnsFallbackOnBareStringWithNoJSON() {
        // No `{` at all: there is no balanced object to recover.
        let response = "   bare string with no JSON"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "fallback")
        XCTAssertFalse(success)
    }

    func testParseProofreadReturnsFallbackOnUnterminatedJSON() {
        // Truncated output (max_tokens hit mid-response): opening `{` but no
        // matching `}`. The scanner must not crash and must fall through.
        let response = "{\"result\": \"unterminated..."
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "fallback")
        XCTAssertFalse(success)
    }

    func testParseProofreadIgnoresBracesInsideStringLiterals() {
        // Braces that live inside a JSON string literal must not throw off
        // the balanced-object scan.
        let response = "{\"result\": \"contains a { and a } in prose\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "contains a { and a } in prose")
        XCTAssertTrue(success)
    }

    func testParseProofreadHandlesNestedJSONObjects() {
        // ProofreadResponse only decodes the top-level `result`, but the
        // candidate scanner must still walk past any nested object/array
        // braces (e.g. model occasionally adds a `meta` sub-object) without
        // closing the outer candidate early.
        let response = "{\"result\": \"Hello.\", \"meta\": {\"nested\": true}}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello.")
        XCTAssertTrue(success)
    }

    func testParseProofreadHandlesEscapedQuotesInResult() {
        // Backslash-escaped quotes inside the result string must not exit
        // the in-string state, otherwise the next `{` or `}` in the prose
        // would be counted toward brace depth.
        let response = "{\"result\": \"he said \\\"hi\\\" today.\"}"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "he said \"hi\" today.")
        XCTAssertTrue(success)
    }

    func testParseProofreadLeavesFenceUntouchedWhenOpenerHasNoNewline() {
        // Opening ``` with no newline before the body is treated as a
        // non-paired fence; the original input is returned to
        // `extractJSONCandidates`, which still recovers the object via
        // balanced-brace scanning.
        let response = "```{\"result\": \"Hello.\"}```"
        let (text, success) = parseProofreadResponse(response, fallback: "fallback")
        XCTAssertEqual(text, "Hello.")
        XCTAssertTrue(success)
    }

    // MARK: - Validate path coverage

    func testParseValidateStripsMarkdownFence() {
        // The validate path mirrors the proofread path's defensive parsing.
        let response = "```json\n{\"result\": \"Hello.\", \"keep\": [true]}\n```"
        let (text, keep, success) = parseValidateResponse(
            response,
            fallback: "fallback",
            numReplacements: 1
        )
        XCTAssertEqual(text, "Hello.")
        XCTAssertEqual(keep, [true])
        XCTAssertTrue(success)
    }

    func testParseValidatePicksFirstOfMultipleObjects() {
        let response = """
        {"result": "Hello.", "keep": [true]}{"result": "World.", "keep": [false]}
        """
        let (text, keep, success) = parseValidateResponse(
            response,
            fallback: "fallback",
            numReplacements: 1
        )
        XCTAssertEqual(text, "Hello.")
        XCTAssertEqual(keep, [true])
        XCTAssertTrue(success)
    }
}
