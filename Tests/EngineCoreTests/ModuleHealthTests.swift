// ModuleHealthTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

final class ModuleHealthTests: XCTestCase {

    func testDefaultDetailIsEmpty() {
        let health = ModuleHealth(loaded: true)
        XCTAssertTrue(health.loaded)
        XCTAssertNil(health.error)
        XCTAssertTrue(health.detail.isEmpty)
    }

    func testCodableRoundTrip() throws {
        let original = ModuleHealth(
            loaded: false,
            error: "init failed",
            detail: ["model": "parakeet-tdt-0.6b", "language": "en"]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(ModuleHealth.self, from: data)
        XCTAssertEqual(decoded, original)
    }
}
