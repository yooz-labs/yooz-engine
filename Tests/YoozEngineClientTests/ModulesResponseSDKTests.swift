// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import YoozEngineClient

/// Contract test for `/v1/modules`. Decodes a canonical engine-shape JSON
/// body — the exact shape `APIServer.swift` emits via
/// `EngineCore.ModulesResponse.build` — through the SDK import path
/// (`YoozEngineClient` re-exports `YoozEngineWire.ModulesResponse`, #225).
///
/// `EngineCore` and `YoozEngineClient` share a single `ModulesResponse`
/// declaration since #225, so this and `EngineCoreTests.ModulesResponseTests`
/// exercise the same type; kept as a separate suite because it pins the
/// exact JSON shape yooz-whisper's About panel reads (`engineVersion`,
/// `buildVariant`, `modules[].name`, `modules[].loaded`,
/// `modules[].detail[*]`) rather than round-tripping through the type's own
/// encoder. See yooz-engine#133 for the wire-shape alignment that motivated
/// this contract test.
final class ModulesResponseSDKTests: XCTestCase {

    /// Canonical wire body. Field order matches what the server emits
    /// under `.sortedKeys`: `buildVariant` < `engineVersion` < `modules`
    /// at the top level, and `detail` < `error` < `loaded` < `name` <
    /// `version` per manifest. Detail keys are also sorted.
    private let canonicalJSON: String = #"""
    {
      "buildVariant": "whisper",
      "engineVersion": "0.6.0",
      "modules": [
        {
          "detail": {
            "library_version": "0.10.0",
            "rules_total": "1560"
          },
          "loaded": true,
          "name": "grammar",
          "version": "0.6.0"
        },
        {
          "detail": {
            "language": "en",
            "model": "parakeet-tdt-0.6b-v2"
          },
          "loaded": true,
          "name": "stt",
          "version": "0.6.0"
        }
      ]
    }
    """#

    func testSDKDecodesCanonicalEngineResponse() throws {
        let data = Data(canonicalJSON.utf8)
        let decoded = try JSONDecoder().decode(
            ModulesResponse.self, from: data
        )

        XCTAssertEqual(decoded.engineVersion, "0.6.0")
        XCTAssertEqual(decoded.buildVariant, "whisper")
        XCTAssertEqual(decoded.modules.count, 2)

        let grammar = decoded.modules[0]
        XCTAssertEqual(grammar.name, "grammar")
        XCTAssertEqual(grammar.version, "0.6.0")
        XCTAssertTrue(grammar.loaded)
        XCTAssertNil(grammar.error)
        XCTAssertEqual(grammar.detail["rules_total"], "1560")
        XCTAssertEqual(grammar.detail["library_version"], "0.10.0")

        let stt = decoded.modules[1]
        XCTAssertEqual(stt.name, "stt")
        XCTAssertTrue(stt.loaded)
        XCTAssertEqual(stt.detail["language"], "en")
        XCTAssertEqual(stt.detail["model"], "parakeet-tdt-0.6b-v2")
    }

    /// Server emits `error: null` as key-absence (see
    /// `EngineCoreTests.testErrorFieldEncodedAsNullWhenAbsent`). The
    /// SDK type's `error: String?` must decode that as `nil` rather
    /// than fail with `keyNotFound`.
    func testSDKHandlesOmittedErrorField() throws {
        let data = Data(canonicalJSON.utf8)
        let decoded = try JSONDecoder().decode(
            ModulesResponse.self, from: data
        )
        XCTAssertNil(decoded.modules[0].error)
    }

    /// Round-trip the SDK type back to JSON and re-decode. Catches any
    /// asymmetric Codable customization (e.g. one side adding custom
    /// `CodingKeys` that the other doesn't mirror).
    func testSDKRoundTripsThroughItsOwnEncoder() throws {
        let data = Data(canonicalJSON.utf8)
        let first = try JSONDecoder().decode(
            ModulesResponse.self, from: data
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let reencoded = try encoder.encode(first)
        let second = try JSONDecoder().decode(
            ModulesResponse.self, from: reencoded
        )
        XCTAssertEqual(first, second)
    }
}
