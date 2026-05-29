// GrammarMatchSDKTests.swift
// YoozEngineClientTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Codable-shape tests for the additive `matches` field on the grammar SDK
// types. These do not require a live server.

import XCTest
@testable import YoozEngineClient

final class GrammarMatchSDKTests: XCTestCase {

    // MARK: - Backward compatibility

    func testResponseDecodesLegacyJSONWithoutMatches() throws {
        // An older server that predates `matches`. The field must decode to
        // nil, not fail and not become an empty array.
        let json = """
        {"result": "I have a cat", "correctionsApplied": 1, "ruleCount": 1560}
        """.data(using: .utf8)!

        let response = try JSONDecoder().decode(GrammarCheckResponse.self, from: json)
        XCTAssertEqual(response.result, "I have a cat")
        XCTAssertEqual(response.correctionsApplied, 1)
        XCTAssertEqual(response.ruleCount, 1560)
        XCTAssertNil(response.matches, "absent matches key must decode to nil")
    }

    func testResponseDecodesJSONWithExplicitNullMatches() throws {
        let json = """
        {"result": "ok", "correctionsApplied": 0, "ruleCount": null, "matches": null}
        """.data(using: .utf8)!

        let response = try JSONDecoder().decode(GrammarCheckResponse.self, from: json)
        XCTAssertNil(response.ruleCount)
        XCTAssertNil(response.matches)
    }

    // MARK: - New field decoding

    func testResponseDecodesMatches() throws {
        let json = """
        {
          "result": "I have an apple",
          "correctionsApplied": 2,
          "ruleCount": 1560,
          "matches": [
            {
              "offset": 2, "length": 3,
              "original": "has", "replacement": "have",
              "ruleId": "GRAMMAR_DIFF_REPLACE", "category": "grammar",
              "message": "Replace \\"has\\" with \\"have\\"", "shortMessage": null
            },
            {
              "offset": 6, "length": 1,
              "original": "a", "replacement": "an",
              "ruleId": "GRAMMAR_DIFF_REPLACE", "category": "grammar",
              "message": "Replace \\"a\\" with \\"an\\""
            }
          ]
        }
        """.data(using: .utf8)!

        let response = try JSONDecoder().decode(GrammarCheckResponse.self, from: json)
        let matches = try XCTUnwrap(response.matches)
        XCTAssertEqual(matches.count, 2)

        XCTAssertEqual(matches[0].offset, 2)
        XCTAssertEqual(matches[0].length, 3)
        XCTAssertEqual(matches[0].original, "has")
        XCTAssertEqual(matches[0].replacement, "have")
        XCTAssertEqual(matches[0].ruleId, "GRAMMAR_DIFF_REPLACE")
        XCTAssertEqual(matches[0].category, "grammar")
        XCTAssertNil(matches[0].shortMessage)

        // Missing shortMessage key decodes to nil.
        XCTAssertEqual(matches[1].original, "a")
        XCTAssertNil(matches[1].shortMessage)
    }

    func testMatchRoundTrip() throws {
        let original = GrammarMatch(
            offset: 5,
            length: 4,
            original: "teh",
            replacement: "the",
            ruleId: "GRAMMAR_DIFF_REPLACE",
            category: "grammar",
            message: "Replace \"teh\" with \"the\"",
            shortMessage: "typo"
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(GrammarMatch.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testDeletionMatchHasEmptyReplacement() throws {
        let json = """
        {
          "offset": 8, "length": 3,
          "original": "to", "replacement": "",
          "ruleId": "GRAMMAR_DIFF_DELETE", "category": "grammar",
          "message": "Remove \\"to\\""
        }
        """.data(using: .utf8)!

        let match = try JSONDecoder().decode(GrammarMatch.self, from: json)
        XCTAssertEqual(match.replacement, "", "deletion replacement must be empty")
        XCTAssertEqual(match.ruleId, "GRAMMAR_DIFF_DELETE")
    }

    // MARK: - Constructed response

    func testResponseInitDefaultsMatchesToNil() {
        let response = GrammarCheckResponse(
            result: "ok",
            correctionsApplied: 0,
            ruleCount: nil
        )
        XCTAssertNil(response.matches)
    }
}
