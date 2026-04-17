// BuildVariantTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

final class BuildVariantTests: XCTestCase {

    func testDefaultVariantIsFull() {
        #if VARIANT_WHISPER
        XCTAssertEqual(BuildVariant.current, .whisper)
        #elseif VARIANT_LITE
        XCTAssertEqual(BuildVariant.current, .lite)
        #else
        XCTAssertEqual(BuildVariant.current, .full)
        #endif
    }

    func testRawValueRoundTrip() {
        XCTAssertEqual(BuildVariant(rawValue: "full"), .full)
        XCTAssertEqual(BuildVariant(rawValue: "whisper"), .whisper)
        XCTAssertEqual(BuildVariant(rawValue: "lite"), .lite)
        XCTAssertNil(BuildVariant(rawValue: "made-up"))
    }

    func testAllCasesCodable() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        for variant in [BuildVariant.full, .whisper, .lite] {
            let data = try encoder.encode(variant)
            let decoded = try decoder.decode(BuildVariant.self, from: data)
            XCTAssertEqual(decoded, variant)
        }
    }

    /// Documentary manifest of which modules each build variant is expected
    /// to ship with. This is the contract that `project.yml` targets encode
    /// via their `dependencies:` lists; keeping it in code (rather than only
    /// in prose) gives us one place to update when a new variant is added
    /// (Notes, Voice) or a module moves (e.g. VAD is locally embedded in
    /// whisper per the A1 design, §4 exception). EngineCore can't import the
    /// module targets, so this stays a string manifest — no `canImport` here.
    ///
    /// Lite drops both MLX STT and VAD; it ships only Apple STT, Grammar,
    /// and LLM, per the Phase 5 epic (`Apple STT + Lite Variant` section).
    func testExpectedModulesPerVariant() {
        let expected: [BuildVariant: Set<String>] = [
            .full: ["stt", "apple_stt", "grammar", "llm", "vad"],
            .whisper: ["stt", "apple_stt", "grammar", "llm"],  // VAD embedded in whisper client
            .lite: ["apple_stt", "grammar", "llm"]  // no MLX, no VAD
        ]
        // Whisper's module set is a strict subset of full.
        XCTAssertTrue(expected[.whisper]!.isSubset(of: expected[.full]!))
        XCTAssertEqual(
            expected[.full]!.subtracting(expected[.whisper]!),
            ["vad"],
            "Whisper variant drops only VAD today; future slim variants will drop more."
        )
        // Lite is a strict subset of whisper (drops MLX STT on top of VAD).
        XCTAssertTrue(expected[.lite]!.isSubset(of: expected[.whisper]!))
        XCTAssertEqual(
            expected[.whisper]!.subtracting(expected[.lite]!),
            ["stt"],
            "Lite drops the MLX STT module — Apple STT is the only speech backend."
        )
    }
}
