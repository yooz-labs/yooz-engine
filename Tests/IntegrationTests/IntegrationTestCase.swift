// IntegrationTestCase.swift
// IntegrationTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest
import YoozEngineClient

/// Base class for end-to-end tests that drive a running engine process
/// through `YoozEngineClient`. Subclasses get:
///
/// - A gated lifecycle: unless `YOOZ_INTEGRATION=1`, every test is
///   skipped at `setUp` so default CI runs remain green.
/// - A shared `YoozEngineClient` pointed at the engine under test.
/// - Per-test timing capture via `measureEndpoint(_:)`.
///
/// Two execution modes, chosen at runtime in `launchOrAttach`:
///
/// - **External engine** (`YOOZ_TEST_ENGINE_URL` set): skip subprocess
///   management, assume the URL is already reachable. Test authors who run
///   against `localhost:19920` directly get fast feedback without rebuilding.
/// - **Subprocess** (default): locate `Yooz Engine.app` via
///   `EngineProcessLauncher.locateAppBundle()`, start it, wait for health.
///
/// Teardown reverses the process: terminates the subprocess if we owned it,
/// leaves the external engine alone.
class IntegrationTestCase: XCTestCase {

    /// Shared client for the engine under test. Non-nil inside test methods
    /// when `YOOZ_INTEGRATION=1`.
    var client: YoozEngineClient!

    /// Tracks whether THIS test case started the subprocess. External
    /// engines (via `YOOZ_TEST_ENGINE_URL`) are left untouched on tearDown.
    private var launcher: EngineProcessLauncher?

    /// Endpoint timings collected via `measureEndpoint(_:)`. Printed by
    /// `IntegrationTestCase.tearDown` so the runner script can grep for
    /// `[timing]` and produce a summary.
    private var timings: [(String, Duration)] = []

    /// Shared health-check URL derived from the client's baseURL.
    private var healthURL: URL {
        client.baseURL.appendingPathComponent("v1/health")
    }

    // MARK: - Lifecycle

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_INTEGRATION"] == "1",
            "Set YOOZ_INTEGRATION=1 to run end-to-end tests."
        )
        try await launchOrAttach()
    }

    override func tearDown() async throws {
        // Always print whatever timings were captured before teardown work
        // so a failing test still surfaces its before-failure measurements.
        for (label, duration) in timings {
            print("[timing] \(label) = \(duration)")
        }
        timings.removeAll()
        launcher?.terminate()
        launcher = nil
        client = nil
        try await super.tearDown()
    }

    // MARK: - Test helpers

    /// Measures `block`'s wall-clock duration and records it under `label`.
    /// Duration surfaces via `print("[timing] ...")` in tearDown.
    @discardableResult
    func measureEndpoint<T>(
        _ label: String,
        _ block: () async throws -> T
    ) async throws -> T {
        let start = ContinuousClock.now
        let result = try await block()
        let elapsed = ContinuousClock.now - start
        timings.append((label, elapsed))
        return result
    }

    // MARK: - Engine connection

    private func launchOrAttach() async throws {
        let env = ProcessInfo.processInfo.environment
        if let override = env["YOOZ_TEST_ENGINE_URL"], !override.isEmpty {
            guard let (host, port) = Self.parseHostPort(override) else {
                XCTFail("YOOZ_TEST_ENGINE_URL must be http://host:port, got \(override)")
                return
            }
            self.client = YoozEngineClient(host: host, port: port)
            let probeURL = healthURL
            try await TimeoutGuard.waitUntil(
                "external engine health at \(override)",
                timeout: .seconds(10)
            ) {
                await Self.probeHealth(at: probeURL)
            }
            return
        }

        // No override: make sure we aren't about to step on a dev instance.
        if try PortAvailability.isInUse(19920) {
            XCTFail("""
                Port 19920 is already bound. Either kill the dev engine or \
                set YOOZ_TEST_ENGINE_URL=http://127.0.0.1:19920 to reuse it.
                """)
            return
        }

        // Locate + launch.
        let appURL = try EngineProcessLauncher.locateAppBundle()
        let launcher = try EngineProcessLauncher(appBundleURL: appURL)
        try launcher.launch()
        self.launcher = launcher
        self.client = YoozEngineClient()

        let probeURL = healthURL
        try await TimeoutGuard.waitUntil(
            "subprocess /v1/health ready",
            timeout: .seconds(10)
        ) {
            await Self.probeHealth(at: probeURL)
        }
    }

    private static func probeHealth(at url: URL) async -> Bool {
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        do {
            let (_, response) = try await URLSession.shared.data(for: req)
            guard let http = response as? HTTPURLResponse else { return false }
            return http.statusCode == 200
        } catch {
            return false
        }
    }

    /// Parses `http://host:port` into components. `nil` if malformed.
    static func parseHostPort(_ raw: String) -> (String, Int)? {
        guard let url = URL(string: raw),
              let host = url.host,
              let port = url.port
        else { return nil }
        return (host, port)
    }
}
