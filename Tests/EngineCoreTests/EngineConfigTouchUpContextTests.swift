// EngineConfigTouchUpContextTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pin the `YOOZ_TOUCHUP_CONTEXT_ENABLED` env-var contract for
// `EngineConfig.touchUpContextEnabled` (engine#280 Phase 4 / whisper#317):
// explicit "1"/"true" forces on, explicit "0"/"false" forces off, and
// unset/unrecognized falls back to the compiled `defaultTouchUpContextEnabled`
// (set by the Phase 4 eval-gate result — see that constant's doc). Mirrors
// `EngineConfigGPUAdmissionAgingTests`.

import XCTest
@testable import EngineCore

final class EngineConfigTouchUpContextTests: XCTestCase {

    /// `setenv`/`unsetenv` mutate process-global state. Snapshot + restore
    /// so test ordering can't pollute later runs.
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

    func testUnsetEnvVarFallsBackToCompiledDefault() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: nil) {
            XCTAssertEqual(EngineConfig.touchUpContextEnabled, EngineConfig.defaultTouchUpContextEnabled)
        }
    }

    func testExplicitOneForcesOn() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "1") {
            XCTAssertTrue(EngineConfig.touchUpContextEnabled)
        }
    }

    func testExplicitTrueForcesOn() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "true") {
            XCTAssertTrue(EngineConfig.touchUpContextEnabled)
        }
    }

    func testExplicitZeroForcesOff() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "0") {
            XCTAssertFalse(EngineConfig.touchUpContextEnabled)
        }
    }

    func testExplicitFalseForcesOff() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "false") {
            XCTAssertFalse(EngineConfig.touchUpContextEnabled)
        }
    }

    func testUnrecognizedValueFallsBackToCompiledDefault() {
        withEnvVar("YOOZ_TOUCHUP_CONTEXT_ENABLED", value: "maybe") {
            XCTAssertEqual(EngineConfig.touchUpContextEnabled, EngineConfig.defaultTouchUpContextEnabled)
        }
    }
}
