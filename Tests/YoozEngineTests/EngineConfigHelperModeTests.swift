// EngineConfigHelperModeTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Regression coverage for issues #42, #117, and #128: the helper-mode
// contract is the single switch that determines whether the engine
// constructs SwiftUI (with `MenuBarExtra`) or runs as a pure AppKit
// background process with `.prohibited` activation policy.
//
// Two layers of coverage:
//
// 1. **Runtime channel** — `EngineConfig.isHelper` reads the env var and
//    argv flag. Used by the full standalone `YoozEngine` build. If the
//    env-var name or sentinel value drifts, `main.swift` silently falls
//    into the SwiftUI branch and the brain icon comes back. Pinned below.
//
// 2. **Compile-time invariant** — embedded variants (`VARIANT_WHISPER`,
//    `VARIANT_LITE`) bake `isHelperMode = true` at compile time via
//    `HelperVariantInvariant.isCompileTimeHelperOnly`. The runtime
//    channels became unreliable on macOS 26 (LaunchServices strips both
//    env AND argv from `openApplication`-spawned helpers, verified via
//    `ps -E` and `lsappinfo`). #128 pins the variant invariant at the
//    compile flag layer instead. The test target builds against the full
//    standalone target so the invariant resolves to `false` here; a flip
//    to `true` would mean someone accidentally set `VARIANT_WHISPER` or
//    `VARIANT_LITE` on the full target.

import XCTest
import EngineCore
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

    // MARK: - Compile-time variant invariant (#128)
    //
    // Embedded helper variants (`VARIANT_WHISPER`, `VARIANT_LITE`) bake the
    // helper invariant at compile time because LaunchServices strips both
    // env and argv from `openApplication`-spawned helpers on macOS 26 (#128).
    // This test bundle builds against the full standalone `YoozEngine`
    // target, so the invariant must resolve to `false` here — a flip to
    // `true` means the full target accidentally picked up the variant
    // compile flag and would no longer present a menu-bar UI in dev builds.

    /// Pins the compile-time invariant for the full standalone build.
    /// The full `YoozEngine` target must NOT carry the `VARIANT_WHISPER`
    /// or `VARIANT_LITE` compile condition; if it does, double-clicking
    /// `Yooz Engine.app` would silently skip SwiftUI and the dev workflow
    /// (Settings scene, menu-bar icon, picker UI) regresses.
    func testFullVariantIsNotCompileTimeHelperOnly() {
        XCTAssertFalse(
            HelperVariantInvariant.isCompileTimeHelperOnly,
            """
            The full YoozEngine build flipped `HelperVariantInvariant` on. \
            That means VARIANT_WHISPER or VARIANT_LITE leaked into the \
            standalone target's SWIFT_ACTIVE_COMPILATION_CONDITIONS. Check \
            project.yml — the flag belongs only on YoozEngineWhisper / \
            YoozEngineLite.
            """
        )
    }

    /// Documents the inverse expectation under the helper variants. The
    /// YoozEngineTests bundle hosts on the full `Yooz Engine.app`, so it
    /// can never execute under `VARIANT_WHISPER` or `VARIANT_LITE`. The
    /// `#if` block is dead code in this target by design — its presence
    /// makes the invariant contract explicit and would surface if someone
    /// tried to retarget this test bundle onto a helper variant host
    /// without thinking through the implications.
    #if VARIANT_WHISPER || VARIANT_LITE
    func testHelperVariantIsCompileTimeHelperOnly() {
        XCTAssertTrue(
            HelperVariantInvariant.isCompileTimeHelperOnly,
            "Helper variant build must compile-time pin headless mode (#128)."
        )
    }
    #endif
}
