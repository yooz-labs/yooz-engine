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
/// Two execution modes, chosen at runtime in `ensureEngineRunning`:
///
/// - **External engine** (`YOOZ_TEST_ENGINE_URL` set): skip subprocess
///   management, assume the URL is already reachable. Test authors who run
///   against `localhost:19920` directly get fast feedback without rebuilding.
/// - **Subprocess** (default): locate `Yooz Engine.app` via
///   `EngineProcessLauncher.locateAppBundle()`, start it, wait for health.
///
/// Lifecycle is **class-scoped**, not per-test: the subprocess is launched
/// once in the first instance's `setUp`, shared by every test method in the
/// class, and terminated in `class tearDown`. Launching per test method
/// races on port 19920 because macOS doesn't release the socket fast enough
/// between back-to-back launches (#37).
class IntegrationTestCase: XCTestCase {

    /// Shared client for the engine under test. Non-nil inside test methods
    /// when `YOOZ_INTEGRATION=1` and the engine is healthy.
    var client: YoozEngineClient!

    /// Endpoint timings collected via `measureEndpoint(_:)`. Printed by
    /// `IntegrationTestCase.tearDown` so the runner script can grep for
    /// `[timing]` and produce a summary.
    private var timings: [(String, Duration)] = []

    // MARK: - Class-scoped shared engine state

    /// Holds the process we launched (if any) and the base URL every test
    /// method should point its client at. One instance per XCTestCase
    /// subclass, torn down in `class tearDown`.
    private struct SharedEngine: @unchecked Sendable {
        let launcher: EngineProcessLauncher?   // nil when attaching to an external engine
        let baseURL: URL
    }

    /// The single bring-up task, created lazily by the first `setUp` and
    /// awaited by every subsequent one. Using a memoised `Task` means every
    /// caller observes exactly one launch attempt and the same success or
    /// failure result, without holding a lock across `await`. Mutation is
    /// guarded by `bootstrapLock` so this stays correct even when XCTest
    /// parallelises test discovery across classes.
    private static var bootstrapTask: Task<SharedEngine, Error>?

    /// Guards `bootstrapTask` mutation. Only held for synchronous reads and
    /// writes (never across `await`), so `NSLock` is sufficient and
    /// cross-thread unlock is never attempted.
    private static let bootstrapLock = NSLock()

    // MARK: - Lifecycle

    override func setUp() async throws {
        try await super.setUp()
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_INTEGRATION"] == "1",
            "Set YOOZ_INTEGRATION=1 to run end-to-end tests."
        )
        let baseURL = try await Self.ensureEngineRunning()
        guard let (host, port) = Self.hostAndPort(from: baseURL) else {
            XCTFail("Shared engine baseURL malformed: \(baseURL)")
            return
        }
        self.client = YoozEngineClient(host: host, port: port)
    }

    override func tearDown() async throws {
        // Always print whatever timings were captured before teardown work
        // so a failing test still surfaces its before-failure measurements.
        for (label, duration) in timings {
            print("[timing] \(label) = \(duration)")
        }
        timings.removeAll()
        client = nil
        try await super.tearDown()
    }

    override class func tearDown() {
        bootstrapLock.lock()
        let task = bootstrapTask
        bootstrapTask = nil
        bootstrapLock.unlock()

        if let task {
            // The bootstrap may still be in flight (rare, but possible if
            // a class run aborts before any test body ran). Wait
            // synchronously so we never leak the subprocess.
            let engine = Self.awaitSynchronously(task)
            engine?.launcher?.terminate()
        }
        super.tearDown()
    }

    /// Blocks the current thread until `task` completes, returning its
    /// value or nil on failure. Used only from `class tearDown` which runs
    /// synchronously after all tests in the class finish.
    ///
    /// The captured value is carried through a `DispatchQueue`-confined
    /// box so we never share a plain `var` across the Task boundary.
    private static func awaitSynchronously(_ task: Task<SharedEngine, Error>) -> SharedEngine? {
        final class Box: @unchecked Sendable {
            var value: SharedEngine?
        }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached {
            box.value = try? await task.value
            semaphore.signal()
        }
        semaphore.wait()
        return box.value
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

    // MARK: - Shared engine bring-up

    /// Ensures the class-scoped engine is running and healthy. The first
    /// caller creates a bootstrap `Task`; every subsequent caller awaits
    /// the same task so there is exactly one launch per class run,
    /// regardless of how many test methods contend on `setUp`.
    ///
    /// Returns the base URL that tests should point their clients at.
    private static func ensureEngineRunning() async throws -> URL {
        bootstrapLock.lock()
        let task: Task<SharedEngine, Error>
        if let existing = bootstrapTask {
            task = existing
        } else {
            task = Task { try await bootstrapEngine() }
            bootstrapTask = task
        }
        bootstrapLock.unlock()

        let engine = try await task.value
        return engine.baseURL
    }

    /// Performs the actual launch-or-attach. Extracted from `setUp` so the
    /// class-level bring-up can run once per class instead of once per test.
    private static func bootstrapEngine() async throws -> SharedEngine {
        let env = ProcessInfo.processInfo.environment

        // External-engine mode: no subprocess, just validate the URL.
        if let override = resolvedEnvironmentValue(env["YOOZ_TEST_ENGINE_URL"]) {
            guard let url = URL(string: override),
                  hostAndPort(from: url) != nil else {
                throw SetUpFailure(
                    "YOOZ_TEST_ENGINE_URL must be http://host:port, got \(override)"
                )
            }
            let probe = url.appendingPathComponent("v1/health")
            try await TimeoutGuard.waitUntil(
                "external engine health at \(override)",
                timeout: .seconds(10)
            ) {
                await probeHealth(at: probe)
            }
            return SharedEngine(launcher: nil, baseURL: url)
        }

        // No override: refuse to step on a dev instance holding the port.
        if try PortAvailability.isInUse(19920) {
            throw SetUpFailure("""
                Port 19920 is already bound. Either kill the dev engine or \
                set YOOZ_TEST_ENGINE_URL=http://127.0.0.1:19920 to reuse it.
                """)
        }

        let appURL = try EngineProcessLauncher.locateAppBundle()
        let launcher = try EngineProcessLauncher(appBundleURL: appURL)
        try launcher.launch()

        let baseURL = URL(string: "http://127.0.0.1:19920")!
        let probe = baseURL.appendingPathComponent("v1/health")
        do {
            try await TimeoutGuard.waitUntil(
                "subprocess /v1/health ready",
                timeout: .seconds(10)
            ) {
                await probeHealth(at: probe)
            }
        } catch {
            // Health never came up: tear the orphan child down so the next
            // class run isn't racing against a zombie still holding 19920.
            launcher.terminate()
            throw error
        }
        return SharedEngine(launcher: launcher, baseURL: baseURL)
    }

    /// Setup-time failure distinct from `XCTFail` so it can be thrown out of
    /// the shared bootstrap and re-surfaced on every test in the class.
    struct SetUpFailure: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
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

    /// Splits a `http://host:port` URL into `(host, port)`. Returns `nil`
    /// if either component is missing.
    static func hostAndPort(from url: URL) -> (String, Int)? {
        guard let host = url.host, let port = url.port else { return nil }
        return (host, port)
    }

    /// Parses `http://host:port` into components. `nil` if malformed.
    /// Preserved as a static helper for `IntegrationHelperTests`.
    static func parseHostPort(_ raw: String) -> (String, Int)? {
        guard let url = URL(string: raw) else { return nil }
        return hostAndPort(from: url)
    }

    static func resolvedEnvironmentValue(_ raw: String?) -> String? {
        guard let raw, !raw.isEmpty, !raw.hasPrefix("$(") else { return nil }
        return raw
    }
}
