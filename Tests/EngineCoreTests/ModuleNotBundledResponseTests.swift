// ModuleNotBundledResponseTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Unit coverage for the shared 501 response body (A4 / #28). The body is the
// contract thin clients rely on, so we pin: field names, field values, and
// deterministic Codable round-trip. Route-level coverage lives in
// `Tests/YoozEngineTests/ModuleNotBundledTests.swift`.

import XCTest
@testable import EngineCore

final class ModuleNotBundledResponseTests: XCTestCase {

    // MARK: - Message composition

    func testConvenienceInitProducesSpecMessage() {
        let response = ModuleNotBundledResponse(module: "vad", buildVariant: "whisper")
        XCTAssertEqual(response.module, "vad")
        XCTAssertEqual(response.code, "module_not_bundled")
        XCTAssertEqual(
            response.error,
            "Module 'vad' not bundled in this build variant (whisper)"
        )
    }

    func testStableCodeConstant() {
        // Clients switch on `code`; guarding the constant prevents a silent
        // rename from breaking every thin client simultaneously.
        XCTAssertEqual(ModuleNotBundledResponse.code, "module_not_bundled")
        let response = ModuleNotBundledResponse(module: "stt", buildVariant: "notes")
        XCTAssertEqual(response.code, ModuleNotBundledResponse.code)
    }

    func testVariantNameAppearsInMessage() {
        // Each variant shows up verbatim in parentheses. Covers the full
        // variant set we ship today plus a hypothetical future one to lock
        // the interpolation slot.
        for variant in ["full", "whisper", "notes"] {
            let response = ModuleNotBundledResponse(module: "llm", buildVariant: variant)
            XCTAssertTrue(
                response.error.contains("(\(variant))"),
                "variant '\(variant)' missing from: \(response.error)"
            )
        }
    }

    // MARK: - Codable

    func testEncodedJSONMatchesSpecShape() throws {
        let response = ModuleNotBundledResponse(module: "vad", buildVariant: "whisper")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(response)
        let json = String(data: data, encoding: .utf8) ?? ""

        // Exact body from the A4 spec. `.sortedKeys` orders fields
        // alphabetically: code, error, module.
        let expected = #"{"code":"module_not_bundled","error":"Module 'vad' not bundled in this build variant (whisper)","module":"vad"}"#
        XCTAssertEqual(json, expected)
    }

    func testCodableRoundTrip() throws {
        let original = ModuleNotBundledResponse(module: "stt", buildVariant: "notes")
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ModuleNotBundledResponse.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    func testDesignatedInitUsedByCodable() throws {
        // The designated (error, code, module) init powers Codable. Make sure
        // a hand-authored JSON payload with a non-default error string decodes
        // verbatim rather than being rebuilt from module + variant.
        let payload = #"{"error":"custom","code":"module_not_bundled","module":"vad"}"#
        let data = Data(payload.utf8)
        let decoded = try JSONDecoder().decode(ModuleNotBundledResponse.self, from: data)
        XCTAssertEqual(decoded.error, "custom")
        XCTAssertEqual(decoded.code, "module_not_bundled")
        XCTAssertEqual(decoded.module, "vad")
    }
}
