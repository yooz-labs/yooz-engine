// EngineConfigHelperModeTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Regression coverage for issue #42: the helper-mode contract is the
// single switch that determines whether the engine constructs SwiftUI
// (with `MenuBarExtra`) or runs as a pure AppKit background process
// with `.prohibited` activation policy. If `EngineConfig.isHelper`
// drifts (e.g. wrong env-var name, wrong sentinel value), `main.swift`
// silently falls into the SwiftUI branch and the brain icon comes back.
// These tests pin the contract.

import XCTest
@testable import YoozEngine

final class EngineConfigHelperModeTests: XCTestCase {

    /// The env-var name is part of the public contract between host apps
    /// (yooz-whisper, yooz-notes) and the engine. Renaming it without
    /// also bumping every host's launch code silently breaks helper
    /// suppression. Pinning the name in a test makes such a rename
    /// surface immediately as a failed test instead of a runtime UI bug.
    func testHeadlessEnvVarNameIsStable() {
        XCTAssertEqual(EngineConfig.headlessEnvVar, "YOOZ_ENGINE_HEADLESS")
    }

    /// `isHelper` reads the env var from `ProcessInfo` lazily on every
    /// call. Mutating the in-process environment table via `setenv` and
    /// `unsetenv` is observed by `ProcessInfo` on the next read, so we
    /// can drive the predicate through every state machine transition
    /// without spawning a subprocess. Avoids mocking — real env, real
    /// ProcessInfo, real predicate.
    func testIsHelperFollowsEnvVar() {
        // Save and restore so a flaky test doesn't poison sibling tests
        // running in the same xctest process.
        let prior = ProcessInfo.processInfo.environment[EngineConfig.headlessEnvVar]
        defer {
            if let prior {
                setenv(EngineConfig.headlessEnvVar, prior, 1)
            } else {
                unsetenv(EngineConfig.headlessEnvVar)
            }
        }

        // Unset → not a helper.
        unsetenv(EngineConfig.headlessEnvVar)
        XCTAssertFalse(EngineConfig.isHelper, "Unset env var should mean standalone mode")

        // Set to "1" → helper.
        setenv(EngineConfig.headlessEnvVar, "1", 1)
        XCTAssertTrue(EngineConfig.isHelper, "YOOZ_ENGINE_HEADLESS=1 should mean helper mode")

        // Anything else → not a helper. Strict-equality with "1" guards
        // against a stale "0" or "true" value flipping the engine into
        // the headless branch unintentionally.
        setenv(EngineConfig.headlessEnvVar, "0", 1)
        XCTAssertFalse(EngineConfig.isHelper, "value '0' should not enable helper mode")

        setenv(EngineConfig.headlessEnvVar, "true", 1)
        XCTAssertFalse(EngineConfig.isHelper, "value 'true' should not enable helper mode")

        setenv(EngineConfig.headlessEnvVar, "", 1)
        XCTAssertFalse(EngineConfig.isHelper, "empty value should not enable helper mode")
    }
}
