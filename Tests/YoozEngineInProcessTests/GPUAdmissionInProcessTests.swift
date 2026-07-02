// GPUAdmissionInProcessTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// In-process coverage for engine#228's GPU admission wiring:
//
// 1. The interactive-marker lifecycle of `InProcessSTTStreamSession` — the
//    begin signal fired from `init` must be provably paired (and ordered
//    before) the end signal from `close()` / `deinit`, on the process-wide
//    `MLXAdmissionGate.shared`. The rapid create-close loop is a statistical
//    regression net for the review-found ordering race where a
//    fire-and-forget end task could land before the begin task and leak a
//    permanently elevated interactive count.
//
// 2. The wire contract of the optional `workloadClass` request field on the
//    in-process TouchUp path: omitted/known values pass, unknown values are
//    a hard 400 (parity with the loopback server's typed enum decode and
//    with this transport's own "unknown mode is a hard error" precedent).
//
// No model weights: the streaming session is constructed directly against
// the Apple backend (buffer-then-finalize; finalizing an empty buffer may
// error, which still must clear the marker), and the TouchUp requests use
// mode "off" (regex-only, no LLM load).

import EngineCore
import XCTest
import YoozEngineClient

@testable import AppleSTTModule
@testable import YoozEngineInProcess

final class GPUAdmissionInProcessTests: XCTestCase {

    /// Poll the shared gate until `interactiveActive` returns to `baseline`
    /// or the deadline passes. Bounded so a leak fails with a clear message
    /// instead of hanging the suite.
    private func waitForInteractiveActive(
        toReturnTo baseline: Int,
        within: Duration = .seconds(5)
    ) async throws -> Bool {
        let deadline = ContinuousClock.now.advanced(by: within)
        while ContinuousClock.now < deadline {
            let state = await MLXAdmissionGate.shared.queueState
            if state.interactiveActive == baseline { return true }
            try await Task.sleep(for: .milliseconds(10))
        }
        return false
    }

    // MARK: - Interactive-marker lifecycle

    func testStreamSessionCloseClearsInteractiveMarker() async throws {
        let baseline = await MLXAdmissionGate.shared.queueState.interactiveActive

        let session = InProcessSTTStreamSession(
            backend: .apple(AppleSTTEngine.shared)
        )
        session.close()

        let cleared = try await waitForInteractiveActive(toReturnTo: baseline)
        XCTAssertTrue(
            cleared,
            "interactiveActive did not return to \(baseline) after close(); the session leaked its marker"
        )
    }

    func testRapidCreateCloseLoopDoesNotLeakInteractiveMarker() async throws {
        // Statistical net for the begin/end ordering race: close()
        // immediately after init is the trigger window. 25 iterations keeps
        // the test fast while giving the scheduler plenty of chances to
        // reorder unordered tasks — under the pre-fix code this leaked
        // within a handful of iterations.
        let baseline = await MLXAdmissionGate.shared.queueState.interactiveActive

        for _ in 0..<25 {
            let session = InProcessSTTStreamSession(
                backend: .apple(AppleSTTEngine.shared)
            )
            session.close()
        }

        let cleared = try await waitForInteractiveActive(toReturnTo: baseline)
        XCTAssertTrue(
            cleared,
            "interactiveActive did not return to \(baseline) after 25 create-close cycles; the begin/end pairing is leaking"
        )
    }

    func testDroppedSessionWithoutCloseClearsMarkerViaDeinit() async throws {
        let baseline = await MLXAdmissionGate.shared.queueState.interactiveActive

        // Create and immediately drop without calling close() — the deinit
        // backstop must clear the marker so an abandoning consumer cannot
        // throttle every future background generation for the process
        // lifetime.
        _ = InProcessSTTStreamSession(backend: .apple(AppleSTTEngine.shared))

        let cleared = try await waitForInteractiveActive(toReturnTo: baseline)
        XCTAssertTrue(
            cleared,
            "interactiveActive did not return to \(baseline) after dropping a session without close(); the deinit backstop is not firing"
        )
    }

    // MARK: - workloadClass wire contract (in-process TouchUp)

    private func touchUpBody(workloadClassJSON: String?) -> Data {
        var json = #"{"text":"hello world","mode":"off""#
        if let workloadClassJSON {
            json += #","workloadClass":\#(workloadClassJSON)"#
        }
        json += "}"
        return Data(json.utf8)
    }

    func testTouchUpOmittedWorkloadClassSucceeds() async throws {
        let transport = InProcessTransport()
        try await transport.connect()
        // mode "off" is regex-only — no LLM load, deterministic.
        let data = try await transport.post(
            "/v1/touchup", body: touchUpBody(workloadClassJSON: nil)
        )
        XCTAssertFalse(data.isEmpty)
    }

    func testTouchUpKnownWorkloadClassSucceeds() async throws {
        let transport = InProcessTransport()
        try await transport.connect()
        let data = try await transport.post(
            "/v1/touchup", body: touchUpBody(workloadClassJSON: "\"interactive\"")
        )
        XCTAssertFalse(data.isEmpty)
    }

    func testTouchUpUnknownWorkloadClassIsHard400() async throws {
        let transport = InProcessTransport()
        try await transport.connect()
        do {
            _ = try await transport.post(
                "/v1/touchup", body: touchUpBody(workloadClassJSON: "\"turbo\"")
            )
            XCTFail("expected a 400 for an unknown workloadClass")
        } catch let error as YoozEngineError {
            guard case let .serverError(statusCode, code, _) = error else {
                return XCTFail("expected serverError, got \(error)")
            }
            XCTAssertEqual(statusCode, 400)
            XCTAssertEqual(code, "invalid_request")
        }
    }
}
