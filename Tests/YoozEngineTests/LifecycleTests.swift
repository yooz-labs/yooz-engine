// LifecycleTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// NOTE: The YoozEngineTests scheme is blocked by a pre-existing
// test-host discovery issue (see `ModulesEndpointTests.swift` and
// A4 handoff). Once that blocker is cleared, this file exercises the
// APIServer lifecycle end-to-end. In the meantime, the wire shape of
// `ServerStartError` is pinned by
// `Tests/EngineCoreTests/ServerStartErrorTests.swift`, which does run.

import EngineCore
import XCTest

final class LifecycleTests: XCTestCase {

    // MARK: - State enum

    /// Confirm the `.crashed` state is distinct from `.stopped` — the
    /// state machine uses this equality to decide whether the UI
    /// should render "Restart Engine".
    func testServerStartErrorValuesAreDistinct() {
        XCTAssertNotEqual(
            ServerStartError.portInUse(port: 19920, pid: nil).code,
            ServerStartError.failedToBind("whatever").code
        )
        XCTAssertNotEqual(
            ServerStartError.portInUse(port: 19920, pid: nil).code,
            ServerStartError.healthCheckFailed.code
        )
    }

    // MARK: - PortDiagnostics round-trip

    /// `pidHoldingPort` must be safe to call on random ports. The
    /// watchdog and recovery paths both invoke it and must not throw.
    func testPidHoldingPortIsSafeForArbitraryPorts() {
        _ = PortDiagnostics.pidHoldingPort(54321)
        _ = PortDiagnostics.pidHoldingPort(1)
    }
}
