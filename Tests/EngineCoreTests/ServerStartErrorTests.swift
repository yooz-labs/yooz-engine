// ServerStartErrorTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest
@testable import EngineCore
#if canImport(Darwin)
import Darwin
#endif

final class ServerStartErrorTests: XCTestCase {

    // MARK: - portInUse

    func testPortInUseWithPIDDescription() {
        let error = ServerStartError.portInUse(port: 19920, pid: 12345)
        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("19920"))
        XCTAssertTrue(description.contains("12345"))
        XCTAssertTrue(description.contains("YOOZ_ENGINE_AUTO_RECOVER"))
    }

    func testPortInUseWithoutPIDDescription() {
        let error = ServerStartError.portInUse(port: 19920, pid: nil)
        let description = error.errorDescription ?? ""
        XCTAssertTrue(description.contains("19920"))
        XCTAssertFalse(description.contains("process \(0)"))
        XCTAssertTrue(description.contains("in use"))
    }

    func testPortInUseCode() {
        XCTAssertEqual(ServerStartError.portInUse(port: 19920, pid: nil).code, "port_in_use")
    }

    // MARK: - failedToBind, healthCheckFailed, crashed

    func testFailedToBindPropagatesReason() {
        let error = ServerStartError.failedToBind("connection reset")
        XCTAssertTrue((error.errorDescription ?? "").contains("connection reset"))
        XCTAssertEqual(error.code, "failed_to_bind")
    }

    func testHealthCheckFailedDescription() {
        let error = ServerStartError.healthCheckFailed
        XCTAssertEqual(error.code, "health_check_failed")
        XCTAssertTrue((error.errorDescription ?? "").contains("health check"))
    }

    func testCrashedCarriesReason() {
        let error = ServerStartError.crashed("IOError kqueue")
        XCTAssertTrue((error.errorDescription ?? "").contains("IOError kqueue"))
        XCTAssertEqual(error.code, "crashed")
    }

    // MARK: - Equatable

    func testPortInUseEquatable() {
        XCTAssertEqual(
            ServerStartError.portInUse(port: 19920, pid: 42),
            ServerStartError.portInUse(port: 19920, pid: 42)
        )
        XCTAssertNotEqual(
            ServerStartError.portInUse(port: 19920, pid: 42),
            ServerStartError.portInUse(port: 19920, pid: nil)
        )
        XCTAssertNotEqual(
            ServerStartError.portInUse(port: 19920, pid: nil),
            ServerStartError.healthCheckFailed
        )
    }

    // MARK: - PortDiagnostics.isAddressInUse

    func testIsAddressInUseRecognisesErrnoEADDRINUSE() {
        let posix = NSError(domain: NSPOSIXErrorDomain, code: Int(EADDRINUSE), userInfo: nil)
        XCTAssertTrue(PortDiagnostics.isAddressInUse(posix))
    }

    func testIsAddressInUseRecognisesDescriptionMatch() {
        struct MockError: Error, CustomStringConvertible {
            var description: String { "IOError { errnoCode: 48, reason: Address already in use }" }
        }
        XCTAssertTrue(PortDiagnostics.isAddressInUse(MockError()))
    }

    func testIsAddressInUseRecognisesEaddrinuseTokenMatch() {
        struct MockError: Error, CustomStringConvertible {
            var description: String { "bind() failed with EADDRINUSE" }
        }
        XCTAssertTrue(PortDiagnostics.isAddressInUse(MockError()))
    }

    func testIsAddressInUseRejectsUnrelatedErrors() {
        let unrelated = NSError(domain: "Test", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "connection refused"
        ])
        XCTAssertFalse(PortDiagnostics.isAddressInUse(unrelated))
    }

    // MARK: - PortDiagnostics.pidHoldingPort

    func testPidHoldingPortReturnsNilForUnusedPort() {
        // Port 1 is a privileged port that nothing in test environment
        // will hold. Expect nil rather than an error.
        let pid = PortDiagnostics.pidHoldingPort(1)
        XCTAssertNil(pid, "pidHoldingPort should return nil for an unheld port")
    }

    // MARK: - PortDiagnostics.isPortInUse

    func testIsPortInUseReturnsFalseForUnusedPort() {
        // Pick a high port unlikely to be bound by anything in a test
        // environment. If something is bound here the test will fail
        // loudly rather than silently pass — intentional; the real
        // risk is false positives, not false negatives.
        let port = 59421
        XCTAssertFalse(PortDiagnostics.isPortInUse(port))
    }

    func testIsPortInUseReturnsTrueWhenPortBound() throws {
        // Bind a real socket on a high port, then assert the probe
        // sees it. No mocks — we exercise the real POSIX sockets path.
        #if canImport(Darwin)
        let port: Int = 59422
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            XCTFail("Could not open test socket: errno=\(errno)")
            return
        }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let rc = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { rawPtr in
                Darwin.bind(fd, rawPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard rc == 0 else {
            // Port already in use by someone else; skip rather than
            // flake. This is rare in CI but not impossible.
            throw XCTSkip("Could not bind test socket at :\(port), errno=\(errno); port likely already bound")
        }
        // Don't need to listen() — a successful bind is enough for
        // our probe's second bind() to fail with EADDRINUSE.
        XCTAssertTrue(PortDiagnostics.isPortInUse(port))
        #else
        throw XCTSkip("Test only runs on Darwin")
        #endif
    }
}
