// EngineConfigGPUAdmissionAgingTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pin the `YOOZ_GPU_ADMISSION_AGING_SEC` env-var contract for
// `EngineConfig.gpuAdmissionAgingSeconds` (engine#228): default 2.0,
// clamp-down above the 30s ceiling, and default fallback on zero /
// negative / non-numeric / empty input. This knob is the starvation guard
// backing the "no deadlock" acceptance criterion, so its parse rules are
// contract, not convenience. Mirrors `EngineConfigStreamingCadenceTests`.

import XCTest
@testable import EngineCore

final class EngineConfigGPUAdmissionAgingTests: XCTestCase {

    /// `setenv`/`unsetenv` mutate process-global state. Snapshot + restore
    /// so test ordering can't pollute later runs. Same helper shape as
    /// `EngineConfigStreamingCadenceTests`.
    private func withEnvVar(
        _ name: String,
        value: String?,
        _ body: () throws -> Void
    ) rethrows {
        let prior = ProcessInfo.processInfo.environment[name]
        defer {
            if let prior {
                setenv(name, prior, 1)
            } else {
                unsetenv(name)
            }
        }
        if let value {
            setenv(name, value, 1)
        } else {
            unsetenv(name)
        }
        try body()
    }

    func testUnsetEnvVarReturnsDefault() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: nil) {
            XCTAssertEqual(EngineConfig.gpuAdmissionAgingSeconds, 2.0, accuracy: 0.001)
        }
    }

    func testValidOverrideRoundTrips() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: "0.05") {
            XCTAssertEqual(EngineConfig.gpuAdmissionAgingSeconds, 0.05, accuracy: 0.0001)
        }
    }

    /// Zero would silently defeat the starvation guard (every background
    /// checkpoint force-admits instantly, i.e. no yielding at all), so it
    /// falls back to the default rather than being honored.
    func testZeroReturnsDefault() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: "0") {
            XCTAssertEqual(EngineConfig.gpuAdmissionAgingSeconds, 2.0, accuracy: 0.001)
        }
    }

    func testNegativeValueReturnsDefault() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: "-3") {
            XCTAssertEqual(EngineConfig.gpuAdmissionAgingSeconds, 2.0, accuracy: 0.001)
        }
    }

    func testNonNumericValueReturnsDefault() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: "forever") {
            XCTAssertEqual(EngineConfig.gpuAdmissionAgingSeconds, 2.0, accuracy: 0.001)
        }
    }

    func testEmptyValueReturnsDefault() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: "") {
            XCTAssertEqual(EngineConfig.gpuAdmissionAgingSeconds, 2.0, accuracy: 0.001)
        }
    }

    /// An absurdly large override (typo, load-test env leaking into a real
    /// launch) is clamped to the ceiling instead of silently turning the
    /// starvation guard into an effective deadlock.
    func testOversizedValueClampsToCeiling() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: "999999") {
            XCTAssertEqual(
                EngineConfig.gpuAdmissionAgingSeconds,
                EngineConfig.gpuAdmissionAgingCeilingSeconds,
                accuracy: 0.001
            )
        }
    }

    /// A value exactly at the ceiling passes through unclamped.
    func testCeilingValueRoundTrips() {
        withEnvVar("YOOZ_GPU_ADMISSION_AGING_SEC", value: "30") {
            XCTAssertEqual(EngineConfig.gpuAdmissionAgingSeconds, 30.0, accuracy: 0.001)
        }
    }
}
