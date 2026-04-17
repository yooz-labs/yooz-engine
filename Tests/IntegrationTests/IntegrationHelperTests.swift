// IntegrationHelperTests.swift
// IntegrationTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pure-Swift unit tests for the harness support types. These DO NOT need
// `YOOZ_INTEGRATION=1` — they run in the default CI sweep so regressions
// in the harness itself surface immediately.

import Darwin
import Foundation
import XCTest

final class IntegrationHelperTests: XCTestCase {

    // MARK: - PortAvailability

    /// A fresh high-numbered port should report as free. 0-range ephemeral
    /// ports are deliberately avoided; we pick a high static value unlikely
    /// to be held by a typical dev setup.
    func testPortAvailabilityFreePort() throws {
        let probe = 49_321
        XCTAssertFalse(try PortAvailability.isInUse(probe))
    }

    /// Bind to a port ourselves, then re-probe and expect `true`. The bound
    /// listener is closed by the `defer` so the test leaves no side effects.
    func testPortAvailabilityDetectsInUse() throws {
        let probe = 49_322

        // Hold the port open with a listening socket.
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0, "failed to open test socket")
        defer { close(fd) }

        var reuse: Int32 = 1
        _ = setsockopt(
            fd, SOL_SOCKET, SO_REUSEADDR,
            &reuse, socklen_t(MemoryLayout<Int32>.size)
        )

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(probe).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { rawPtr in
                Darwin.bind(fd, rawPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        XCTAssertEqual(bindResult, 0, "bind() failed, errno=\(errno)")
        XCTAssertEqual(listen(fd, 1), 0)

        XCTAssertTrue(try PortAvailability.isInUse(probe))
    }

    // MARK: - EngineProcessLauncher

    /// Non-.app path must throw `.notAppBundle` before touching the
    /// filesystem contents. Guarantees the unit test never tries to run a
    /// phantom executable.
    func testLauncherRejectsNonAppBundle() {
        let tmp = URL(fileURLWithPath: "/tmp/definitely-not-a-bundle")
        XCTAssertThrowsError(try EngineProcessLauncher(appBundleURL: tmp)) { err in
            guard case EngineProcessLauncher.LaunchError.notAppBundle = err else {
                XCTFail("expected .notAppBundle, got \(err)")
                return
            }
        }
    }

    /// An .app path with no inner Contents/MacOS/<exe> throws
    /// `.executableMissing`. Uses a tmp directory so the test is hermetic.
    func testLauncherRejectsMissingExecutable() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "Fake-\(UUID().uuidString).app",
                isDirectory: true
            )
        try FileManager.default.createDirectory(
            at: tmp.appendingPathComponent("Contents/MacOS"),
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tmp) }

        XCTAssertThrowsError(try EngineProcessLauncher(appBundleURL: tmp)) { err in
            guard case EngineProcessLauncher.LaunchError.executableMissing = err else {
                XCTFail("expected .executableMissing, got \(err)")
                return
            }
        }
    }

    /// `locateAppBundle` honors `YOOZ_ENGINE_APP_PATH` when the path
    /// exists. We point it at a fabricated directory so we don't depend on
    /// DerivedData state.
    func testLocateAppBundleHonorsEnvOverride() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "Fake-\(UUID().uuidString).app",
                isDirectory: true
            )
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        setenv("YOOZ_ENGINE_APP_PATH", tmp.path, 1)
        defer { unsetenv("YOOZ_ENGINE_APP_PATH") }

        let resolved = try EngineProcessLauncher.locateAppBundle()
        XCTAssertEqual(resolved.path, tmp.path)
    }

    /// Missing override path must throw `.bundleNotFound`.
    func testLocateAppBundleMissingOverrideThrows() {
        setenv("YOOZ_ENGINE_APP_PATH", "/tmp/does-not-exist-\(UUID().uuidString).app", 1)
        defer { unsetenv("YOOZ_ENGINE_APP_PATH") }

        XCTAssertThrowsError(try EngineProcessLauncher.locateAppBundle()) { err in
            guard case EngineProcessLauncher.LaunchError.bundleNotFound = err else {
                XCTFail("expected .bundleNotFound, got \(err)")
                return
            }
        }
    }

    // MARK: - TimeoutGuard

    /// Predicate already true -> returns without sleeping.
    func testTimeoutGuardReturnsImmediatelyWhenReady() async throws {
        let start = ContinuousClock.now
        try await TimeoutGuard.waitUntil(
            "instantly ready",
            timeout: .seconds(5),
            poll: .milliseconds(50)
        ) { true }
        let elapsed = ContinuousClock.now - start
        XCTAssertLessThan(elapsed, .milliseconds(250))
    }

    /// Predicate never true -> throws `TimedOut` within the deadline.
    func testTimeoutGuardThrowsAfterDeadline() async {
        do {
            try await TimeoutGuard.waitUntil(
                "never ready",
                timeout: .milliseconds(200),
                poll: .milliseconds(50)
            ) { false }
            XCTFail("expected TimedOut error, got success")
        } catch is TimeoutGuard.TimedOut {
            // expected
        } catch {
            XCTFail("expected TimeoutGuard.TimedOut, got \(error)")
        }
    }

    /// Transient errors in the predicate are swallowed until the deadline,
    /// then a timeout surfaces. Guarantees the harness doesn't abort on
    /// benign "server not up yet" URL errors.
    func testTimeoutGuardSwallowsTransientErrors() async {
        struct Transient: Error {}
        do {
            try await TimeoutGuard.waitUntil(
                "always-throws",
                timeout: .milliseconds(150),
                poll: .milliseconds(50)
            ) { throw Transient() }
            XCTFail("expected TimedOut error")
        } catch is TimeoutGuard.TimedOut {
            // expected
        } catch {
            XCTFail("expected TimeoutGuard.TimedOut, got \(error)")
        }
    }

    // MARK: - IntegrationTestCase.parseHostPort

    func testParseHostPortValid() {
        let parsed = IntegrationTestCase.parseHostPort("http://127.0.0.1:19920")
        XCTAssertEqual(parsed?.0, "127.0.0.1")
        XCTAssertEqual(parsed?.1, 19920)
    }

    func testParseHostPortMissingPort() {
        XCTAssertNil(IntegrationTestCase.parseHostPort("http://127.0.0.1"))
    }

    func testParseHostPortMalformed() {
        XCTAssertNil(IntegrationTestCase.parseHostPort("not a url"))
    }
}
