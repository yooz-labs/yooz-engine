// XPCErrorBridgeDeadlineTests.swift
// YoozEngineClientTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pins the `XPCErrorBridge` round trip for the `load_deadline_exceeded`
// `.serverError` case (engine#252, PR #255 review finding I4).
// `InProcessTransport.awaitLoadOrTypedDeadline` (YoozEngineInProcess)
// converts the engine-side `LoadDeadlineExceeded` into this typed error
// before it ever reaches the XPC boundary — this file pins that once it IS
// a `YoozEngineError.serverError`, the bridge carries it correctly, exactly
// like every other `.serverError` case, instead of collapsing it to
// `.engineNotReachable`.

import Foundation
import XCTest

@testable import YoozEngineClient

final class XPCErrorBridgeDeadlineTests: XCTestCase {

    func testLoadDeadlineExceededServerErrorRoundTripsThroughBridge() {
        let original = YoozEngineError.serverError(
            statusCode: 504, code: "load_deadline_exceeded",
            message: "STT model load did not complete within 600s"
        )
        let nsError = XPCErrorBridge.toNSError(original)
        let roundTripped = XPCErrorBridge.toYoozEngineError(nsError)
        XCTAssertEqual(roundTripped, original)
    }

    /// Explicitly pins the failure mode the fix closes: a NON-bridged
    /// error (the shape `LoadDeadlineExceeded` would have taken if it
    /// crossed the boundary unconverted) collapses to `.engineNotReachable`
    /// — this is what made a 600s load timeout look like a dead service
    /// before the conversion existed, and is why the conversion must
    /// happen before `XPCErrorBridge.toNSError` ever sees the error.
    func testUnconvertedNonYoozEngineErrorCollapsesToEngineNotReachable() {
        struct PlainError: Error {}
        let nsError = XPCErrorBridge.toNSError(PlainError())
        let roundTripped = XPCErrorBridge.toYoozEngineError(nsError)
        XCTAssertEqual(roundTripped, .engineNotReachable)
    }
}
