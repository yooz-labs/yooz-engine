// EngineConfigHelperModeTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Regression coverage for issues #42 and #117: the helper-mode contract
// is the single switch that determines whether the engine constructs
// SwiftUI (with `MenuBarExtra`) or runs as a pure AppKit background
// process with `.prohibited` activation policy. If `EngineConfig.isHelper`
// drifts (e.g. wrong env-var name, wrong sentinel value, missing argv
// path), `main.swift` silently falls into the SwiftUI branch and the
// brain icon comes back. These tests pin both the env-var channel
// (kept for backward compat with direct shell exec) and the argv
// channel (the reliable path for `NSWorkspace.openApplication` spawns,
// which silently drops `OpenConfiguration.environment` on macOS 26).

import XCTest
@testable import EngineCore

final class EngineConfigHelperModeTests: XCTestCase {

    /// The env-var name is part of the public contract between host apps
    /// (yooz-whisper, yooz-notes) and the engine. Renaming it without
    /// also bumping every host's launch code silently breaks helper
    /// suppression. Pinning the name in a test makes such a rename
    /// surface immediately as a failed test instead of a runtime UI bug.
    func testHeadlessEnvVarNameIsStable() {
        XCTAssertEqual(EngineConfig.headlessEnvVar, "YOOZ_ENGINE_HEADLESS")
    }

    /// Argv flag is the reliable channel for `NSWorkspace.openApplication`
    /// spawns (#117). Pin the spelling so the host-side launch code
    /// (yooz-whisper `EngineHelperController`, future hosts) and the
    /// engine-side detection cannot drift independently.
    func testHelperModeArgNameIsStable() {
        XCTAssertEqual(EngineConfig.helperModeArg, "--headless")
    }

    func testPortEnvVarNameIsStable() {
        XCTAssertEqual(EngineConfig.portEnvVar, "YOOZ_ENGINE_PORT")
    }

    /// `isHelper` reads the env var from `ProcessInfo` lazily on every
    /// call. Mutating the in-process environment table via `setenv` and
    /// `unsetenv` is observed by `ProcessInfo` on the next read, so we
    /// can drive the predicate through every state machine transition
    /// without spawning a subprocess. Avoids mocking — real env, real
    /// ProcessInfo, real predicate.
    func testIsHelperFollowsEnvVar() throws {
        // Skip the env-channel-only assertions if the test process was
        // launched with `--headless` in its argv (the argv channel would
        // unconditionally force helper mode true and mask env-channel
        // transitions). In practice xctest is never invoked with
        // `--headless`, so the guard simply documents the precondition.
        try XCTSkipIf(
            CommandLine.arguments.contains(EngineConfig.helperModeArg),
            "Test process argv contains --headless; argv channel masks env-channel transitions."
        )

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

        // Anything else → not a helper via the env channel. Strict-equality
        // with "1" guards against a stale "0" or "true" value flipping the
        // engine into the headless branch unintentionally.
        setenv(EngineConfig.headlessEnvVar, "0", 1)
        XCTAssertFalse(EngineConfig.isHelper, "value '0' should not enable helper mode")

        setenv(EngineConfig.headlessEnvVar, "true", 1)
        XCTAssertFalse(EngineConfig.isHelper, "value 'true' should not enable helper mode")

        setenv(EngineConfig.headlessEnvVar, "", 1)
        XCTAssertFalse(EngineConfig.isHelper, "empty value should not enable helper mode")
    }

    func testPortFollowsValidatedEnvVar() {
        let prior = ProcessInfo.processInfo.environment[EngineConfig.portEnvVar]
        defer {
            if let prior {
                setenv(EngineConfig.portEnvVar, prior, 1)
            } else {
                unsetenv(EngineConfig.portEnvVar)
            }
        }

        unsetenv(EngineConfig.portEnvVar)
        XCTAssertEqual(EngineConfig.port, EngineConfig.defaultPort)

        setenv(EngineConfig.portEnvVar, "19921", 1)
        XCTAssertEqual(EngineConfig.port, 19921)

        setenv(EngineConfig.portEnvVar, "0", 1)
        XCTAssertEqual(EngineConfig.port, EngineConfig.defaultPort)

        setenv(EngineConfig.portEnvVar, "65536", 1)
        XCTAssertEqual(EngineConfig.port, EngineConfig.defaultPort)

        setenv(EngineConfig.portEnvVar, "not-a-port", 1)
        XCTAssertEqual(EngineConfig.port, EngineConfig.defaultPort)
    }

    // MARK: - Pure predicate (`isHelperMode(environment:arguments:)`)
    //
    // `CommandLine.arguments` is process-global and read-only at the
    // language level, so we can't drive the live `EngineConfig.isHelper`
    // through every argv combination without spawning a subprocess. The
    // pure predicate `isHelperMode(environment:arguments:)` exists for
    // exactly this reason — the live accessor is a one-line binding of
    // the pure predicate to `ProcessInfo` + `CommandLine`. Testing the
    // pure form covers the same logic with no global state mutation
    // and no mocks.

    /// Neither channel set → standalone mode. This is the standalone
    /// launch path (user double-clicks `Yooz Engine.app`); MUST construct
    /// SwiftUI + MenuBarExtra.
    func testIsHelperModeFalseWhenNeitherChannelSet() {
        XCTAssertFalse(
            EngineConfig.isHelperMode(environment: [:], arguments: ["/path/to/Yooz Engine"])
        )
    }

    /// Env-var channel alone is sufficient (backward compat: direct shell
    /// exec, scripts, integration test harnesses).
    func testIsHelperModeTrueViaEnvVarAlone() {
        XCTAssertTrue(
            EngineConfig.isHelperMode(
                environment: ["YOOZ_ENGINE_HEADLESS": "1"],
                arguments: ["/path/to/Yooz Engine (Whisper)"]
            )
        )
    }

    /// Argv channel alone is sufficient — this is the production path
    /// for `NSWorkspace.openApplication` spawns (#117), where the env
    /// dict is silently dropped by LaunchServices.
    func testIsHelperModeTrueViaArgvAlone() {
        XCTAssertTrue(
            EngineConfig.isHelperMode(
                environment: [:],
                arguments: ["/path/to/Yooz Engine (Whisper)", "--headless"]
            )
        )
    }

    /// Both channels set → helper mode (this is the recommended host
    /// configuration: belt-and-suspenders).
    func testIsHelperModeTrueWhenBothChannelsSet() {
        XCTAssertTrue(
            EngineConfig.isHelperMode(
                environment: ["YOOZ_ENGINE_HEADLESS": "1"],
                arguments: ["/path/to/Yooz Engine (Whisper)", "--headless"]
            )
        )
    }

    /// The env-var sentinel comparison stays strict-equality with "1".
    /// A stale "0" / "true" / empty value in the host's environment
    /// must not flip helper mode on through the env channel (the argv
    /// channel can still flip it on, which is desired).
    func testIsHelperModeIgnoresNonOneEnvValues() {
        XCTAssertFalse(
            EngineConfig.isHelperMode(
                environment: ["YOOZ_ENGINE_HEADLESS": "0"],
                arguments: []
            )
        )
        XCTAssertFalse(
            EngineConfig.isHelperMode(
                environment: ["YOOZ_ENGINE_HEADLESS": "true"],
                arguments: []
            )
        )
        XCTAssertFalse(
            EngineConfig.isHelperMode(
                environment: ["YOOZ_ENGINE_HEADLESS": ""],
                arguments: []
            )
        )
    }

    /// The argv match is exact (`contains("--headless")`), not a prefix
    /// match. A look-alike like `--headless-mode` must NOT flip helper
    /// mode on, otherwise a future host adding a verbose-named flag
    /// would silently inherit headless behaviour.
    func testIsHelperModeArgvRequiresExactMatch() {
        XCTAssertFalse(
            EngineConfig.isHelperMode(
                environment: [:],
                arguments: ["/path/to/Yooz Engine", "--headless-mode"]
            )
        )
        XCTAssertFalse(
            EngineConfig.isHelperMode(
                environment: [:],
                arguments: ["/path/to/Yooz Engine", "headless"]
            )
        )
    }

    /// `isHelper` is a thin live binding of the pure predicate to
    /// `ProcessInfo.processInfo.environment` and `CommandLine.arguments`.
    /// We can't flip argv at runtime, but we can verify the env channel
    /// still flips the live accessor on regardless of argv state
    /// (the OR semantics).
    func testIsHelperLiveAccessorFlipsOnWhenEnvVarSet() {
        let prior = ProcessInfo.processInfo.environment[EngineConfig.headlessEnvVar]
        defer {
            if let prior {
                setenv(EngineConfig.headlessEnvVar, prior, 1)
            } else {
                unsetenv(EngineConfig.headlessEnvVar)
            }
        }

        setenv(EngineConfig.headlessEnvVar, "1", 1)
        XCTAssertTrue(EngineConfig.isHelper)
    }
}
