// EngineConfigStreamingCadenceTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Regression coverage for engine#135: pin the
// `YOOZ_STT_PARTIAL_INTERVAL_SEC` env-var contract and the resolution
// rules (default, clamp-up below 0.1, explicit `0` opt-out, non-numeric
// fallback). The default value is part of the public API surface — host
// apps may read `EngineConfig.defaultStreamingPartialIntervalSec`
// directly when building a config UI — so changing it should be a
// deliberate decision flagged by a failing test.

import XCTest
@testable import EngineCore

final class EngineConfigStreamingCadenceTests: XCTestCase {

    /// `setenv`/`unsetenv` mutate process-global state. Each test takes
    /// a snapshot of the current value, mutates it for the test body,
    /// and restores it on exit so test ordering can't pollute later
    /// runs (or other env-driven `EngineConfig` accessors).
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

    /// The compiled default is the experience host apps get out of the
    /// box. Pinning it as a test makes any deliberate change show up in
    /// review as a single-line diff rather than a silent regression.
    func testDefaultStreamingPartialIntervalSec() {
        XCTAssertEqual(EngineConfig.defaultStreamingPartialIntervalSec, 2.0, accuracy: 0.001)
    }

    /// Unset env var → compiled default. Confirms the "no override"
    /// fast path doesn't accidentally honor an empty string or some
    /// other sentinel.
    func testUnsetEnvVarReturnsDefault() {
        withEnvVar("YOOZ_STT_PARTIAL_INTERVAL_SEC", value: nil) {
            XCTAssertEqual(
                EngineConfig.streamingPartialIntervalSec,
                EngineConfig.defaultStreamingPartialIntervalSec,
                accuracy: 0.001
            )
        }
    }

    /// Numeric override that's well above the clamp floor: round-trips
    /// the parsed value verbatim.
    func testValidOverrideRoundTrips() {
        withEnvVar("YOOZ_STT_PARTIAL_INTERVAL_SEC", value: "1.5") {
            XCTAssertEqual(EngineConfig.streamingPartialIntervalSec, 1.5, accuracy: 0.001)
        }
    }

    /// Literal `0` is honored as a "disable throttle" opt-out — the
    /// only way for an integrator to ask the engine to re-encode on
    /// every WS frame.
    func testZeroDisablesThrottle() {
        withEnvVar("YOOZ_STT_PARTIAL_INTERVAL_SEC", value: "0") {
            XCTAssertEqual(EngineConfig.streamingPartialIntervalSec, 0, accuracy: 0.001)
        }
    }

    /// Tiny positive values are clamped up to `0.1`. Without this, a
    /// typo (`0.001`) would effectively re-enable the legacy per-frame
    /// re-encode behaviour and burn CPU silently.
    func testTinyPositiveIsClampedToFloor() {
        withEnvVar("YOOZ_STT_PARTIAL_INTERVAL_SEC", value: "0.001") {
            XCTAssertEqual(EngineConfig.streamingPartialIntervalSec, 0.1, accuracy: 0.001)
        }
    }

    /// Non-numeric value → default. The fallback prevents a misspelled
    /// env-var from silently disabling the throttle.
    func testNonNumericValueReturnsDefault() {
        withEnvVar("YOOZ_STT_PARTIAL_INTERVAL_SEC", value: "fast") {
            XCTAssertEqual(
                EngineConfig.streamingPartialIntervalSec,
                EngineConfig.defaultStreamingPartialIntervalSec,
                accuracy: 0.001
            )
        }
    }

    /// Negative values are rejected and fall back to the default —
    /// they have no defined semantic in the cadence-floor contract.
    func testNegativeValueReturnsDefault() {
        withEnvVar("YOOZ_STT_PARTIAL_INTERVAL_SEC", value: "-1") {
            XCTAssertEqual(
                EngineConfig.streamingPartialIntervalSec,
                EngineConfig.defaultStreamingPartialIntervalSec,
                accuracy: 0.001
            )
        }
    }

    /// Empty string is treated as non-numeric and falls back to the
    /// default. Pinned because the accessor's docstring explicitly
    /// promises "unset, empty, or non-numeric values" map to the
    /// default — a behaviour an integrator may rely on when shell
    /// scripts inadvertently export an empty value.
    func testEmptyValueReturnsDefault() {
        withEnvVar("YOOZ_STT_PARTIAL_INTERVAL_SEC", value: "") {
            XCTAssertEqual(
                EngineConfig.streamingPartialIntervalSec,
                EngineConfig.defaultStreamingPartialIntervalSec,
                accuracy: 0.001
            )
        }
    }
}
