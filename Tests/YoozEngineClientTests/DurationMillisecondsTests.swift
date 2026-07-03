// DurationMillisecondsTests.swift
// YoozEngineClientTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pins `Duration.milliseconds` (whisper#280 PR #251 review) — the
// millisecond projection the XPC request/STT-batch forensics logging
// (`XPCServiceHandler.swift`, `InProcessTransport.handleBatch`) uses
// since `Duration` has no built-in `Double` millisecond accessor.

import XCTest

@testable import YoozEngineClient

final class DurationMillisecondsTests: XCTestCase {

    func testZeroDuration() {
        XCTAssertEqual(Duration.seconds(0).milliseconds, 0, accuracy: 0.001)
    }

    func testWholeSeconds() {
        XCTAssertEqual(Duration.seconds(2).milliseconds, 2_000, accuracy: 0.001)
    }

    func testSubSecondFraction() {
        XCTAssertEqual(Duration.milliseconds(123).milliseconds, 123, accuracy: 0.001)
    }

    func testMixedSecondsAndFraction() {
        // 1.5s == 1500ms.
        let duration = Duration.seconds(1) + Duration.milliseconds(500)
        XCTAssertEqual(duration.milliseconds, 1_500, accuracy: 0.001)
    }

    func testMicrosecondPrecision() {
        XCTAssertEqual(Duration.microseconds(1_500).milliseconds, 1.5, accuracy: 0.0001)
    }
}
