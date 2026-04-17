// ModuleNotBundledTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Route-level coverage for the 501 `module_not_bundled` helper (A4 / #28).
//
// ## Integration gap
//
// `ModulesEndpointTests` in this target already documents the constraint:
// we cannot easily boot the live Hummingbird server from inside XCTest (the
// port is hard-coded at 19920 in `EngineConfig` and Hummingbird's in-process
// test client is not wired into this project's dependency set). There is also
// a known test-host codesign problem for the YoozEngineTests bundle: the
// host app's binary on disk is `Yooz Engine.app/Contents/MacOS/Yooz Engine`,
// while the test harness looks for `YoozEngine.app/Contents/MacOS/YoozEngine`.
// That blocks `xcodebuild test` from launching this target in the current
// project shape.
//
// Given both constraints, the exhaustive body/status/content-type contract
// for the 501 response is locked at the `EngineCore` layer in
// `Tests/EngineCoreTests/ModuleNotBundledResponseTests.swift`. That is the
// same `ModuleNotBundledResponse` struct the `APIServer.moduleNotBundled(_:)`
// helper encodes into its Hummingbird `Response` body, so the wire shape is
// deterministic and fully covered there.
//
// This file carries the registry-level integration assertions that the route
// closures rely on — specifically that the registry reports `isBundled` false
// for anything it has not seen. That guard is what every module-specific
// route in `APIServer.swift` checks before returning 501 or dispatching to
// the real module, and it is testable from this target without needing
// Hummingbird's types or the host app running.
//
// No mocks. Real registry.

import XCTest
import EngineCore

final class ModuleNotBundledTests: XCTestCase {

    /// A module name that this process will never register. Documents that
    /// the registry says "no" for unknown names — exactly the condition the
    /// route guards use to trigger `moduleNotBundled(_:)`.
    func testUnregisteredModuleNameReportsNotBundled() async {
        let bundled = await ModuleRegistry.shared
            .isBundled("__never_registered_module_a4__")
        XCTAssertFalse(bundled)
    }

    /// A2 gated VAD with `#if canImport(VADModule)`; A4 (this change) makes
    /// every module-specific route follow the same shape. The route-level
    /// guard is `ModuleRegistry.shared.isBundled("<name>")`. This test just
    /// verifies the names the routes check against are valid registry keys
    /// (registry is case-sensitive, string-compared). The actual "bundled"
    /// answer depends on which modules this test binary links — we do NOT
    /// assert that value here to avoid coupling to test-target deps.
    func testRouteGuardNamesAreStrings() async {
        // Names pulled from the route guards in `APIServer.buildRouter()`.
        // Each call is just confirming the API accepts the string;
        // ModuleRegistry returns `Bool` for any input without throwing.
        for name in ["stt", "llm", "grammar", "vad", "tts"] {
            _ = await ModuleRegistry.shared.isBundled(name)
        }
    }

    /// The 501 response body is a `ModuleNotBundledResponse`. The helper on
    /// `APIServer` constructs it with `BuildVariant.current.rawValue`; this
    /// test pins that the same construction path, invoked directly here,
    /// matches the A4 spec shape. Combined with
    /// `EngineCoreTests/ModuleNotBundledResponseTests` this keeps the contract
    /// observable even while the host-app launch gap blocks full integration.
    func testResponseStructMatchesA4Spec() throws {
        let response = ModuleNotBundledResponse(
            module: "vad",
            buildVariant: BuildVariant.current.rawValue
        )
        XCTAssertEqual(response.code, "module_not_bundled")
        XCTAssertEqual(response.module, "vad")
        XCTAssertEqual(
            response.error,
            "Module 'vad' not bundled in this build variant (\(BuildVariant.current.rawValue))"
        )

        // Wire-shape contract: `.sortedKeys` emits code, error, module in
        // alphabetical order — exactly what the helper produces.
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(response)
        let json = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(json.contains("\"code\":\"module_not_bundled\""))
        XCTAssertTrue(json.contains("\"module\":\"vad\""))
        XCTAssertTrue(
            json.contains("Module 'vad' not bundled in this build variant"),
            "expected A4 spec message in: \(json)"
        )
    }
}
