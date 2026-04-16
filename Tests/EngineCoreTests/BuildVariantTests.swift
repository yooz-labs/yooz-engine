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
        #else
        XCTAssertEqual(BuildVariant.current, .full)
        #endif
    }

    func testRawValueRoundTrip() {
        XCTAssertEqual(BuildVariant(rawValue: "full"), .full)
        XCTAssertEqual(BuildVariant(rawValue: "whisper"), .whisper)
        XCTAssertNil(BuildVariant(rawValue: "made-up"))
    }

    func testAllCasesCodable() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        for variant in [BuildVariant.full, .whisper] {
            let data = try encoder.encode(variant)
            let decoded = try decoder.decode(BuildVariant.self, from: data)
            XCTAssertEqual(decoded, variant)
        }
    }
}
