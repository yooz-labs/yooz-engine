// EngineStateRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP + WebSocket route coverage for engine#226's engine-owned model
// selection over the REAL loopback server. Boots a real `APIServer` and
// dials it via `URLSession` — same harness as `TouchUpPickerRouteTests`.
//
// Pins the two surfaces the SPM suites cannot reach (PR #239 review):
//   1. `GET /v1/state` served by Hummingbird — the loopback half of the
//      shared `EngineStateEndpoints` handler (the in-process half is
//      covered by `EngineStateAndEventsTests` under `swift test`).
//   2. The hand-registered `WS /v1/events` route — connect, receive a
//      frame end-to-end, clean close. That handler is WS-specific code
//      with no table-shared implementation, so without this test it had
//      zero live coverage.
//
// The WS side dials a raw `URLSessionWebSocketTask` rather than
// `HTTPTransport.openEvents()`: this app-hosted bundle links the engine
// framework targets (`EngineCore` re-exports `YoozEngineWire`), and adding
// the `YoozEngineClient` SPM product would compile a SECOND copy of
// `YoozEngineWire` into the same bundle (duplicate-module hazard). The SDK
// loop is byte-compatible — it decodes the same `EngineEvent` DTO from the
// same frames this test consumes.

import Foundation
import XCTest
@testable import YoozEngine
@testable import EngineCore
@testable import LLMModule

final class EngineStateRouteTests: XCTestCase {

    @MainActor
    private func withServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
        // Reserve a fresh port for this boot — see yooz-engine#122.
        UniqueEnginePort.assignFreshPort()
        let server = APIServer()
        try await server.start()
        let result: T
        do {
            result = try await body(server)
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
        return result
    }

    private func baseURL() -> URL {
        URL(string: "http://\(EngineConfig.host):\(EngineConfig.port)")!
    }

    private func get(_ path: String) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "GET"
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    private func post(_ path: String, body: Data) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    /// Reset the singleton's active model between tests.
    @MainActor
    private func resetEngineState() async throws {
        _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
    }

    // MARK: - GET /v1/state

    @MainActor
    func testGetStateReturnsTouchUpSnapshotOverLoopback() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (http, body) = try await get("/v1/state")
            XCTAssertEqual(http.statusCode, 200)
            let snapshot = try JSONDecoder().decode(EngineStateSnapshot.self, from: body)
            let touchUp = try XCTUnwrap(
                snapshot.modules.first(where: { $0.module == "touchup" })
            )
            XCTAssertEqual(touchUp.models.count, TouchUpModelSelection.allCases.count)
            XCTAssertEqual(touchUp.activeId, TouchUpModelSelection.yoozLight.rawValue)
            XCTAssertEqual(touchUp.models.filter(\.isActive).count, 1)
            XCTAssertEqual(
                touchUp.models.first(where: { $0.isActive })?.id,
                touchUp.activeId
            )
        }
    }

    /// Loopback/in-process shape parity, asserted over live wire bytes:
    /// the `GET /v1/state` body served by Hummingbird must decode to the
    /// same value the shared handler produces directly (both compile the
    /// same `EngineStateEndpoints` closure — this turns that structural
    /// guarantee into an observed one).
    @MainActor
    func testGetStateMatchesSharedHandlerOutput() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (_, loopbackBody) = try await get("/v1/state")
            let loopback = try JSONDecoder().decode(EngineStateSnapshot.self, from: loopbackBody)

            let direct = await EngineStateEndpoints.touchUpSnapshot()
            let loopbackTouchUp = try XCTUnwrap(
                loopback.modules.first(where: { $0.module == "touchup" })
            )
            XCTAssertEqual(loopbackTouchUp, direct)
        }
    }

    // MARK: - WS /v1/events

    /// End-to-end over the live socket: connect to `WS /v1/events`,
    /// trigger a `modelChanged` publish through the real
    /// `POST /v1/touchup/model` route, and receive + decode the frame.
    /// Covers the Hummingbird WS registration, the handler's task-group
    /// send loop, and frame encoding in one pass.
    ///
    /// Hardened after a 4259s (71 min) wall-clock PASS on a desktop run of
    /// the original shape (PR #239 merge review). Diagnosis: delivery
    /// through this exact stack measures ~35 ms against the live engine at
    /// the same commit (raw `URLSessionWebSocketTask`; empty redirected HF
    /// cache, so no hidden download either — `preload: false` throughout).
    /// The implementation is prompt; the anomaly points at an environmental
    /// suspension of the test process mid-wait (machine sleep / App Nap),
    /// which the original shape permitted: `wsTask.receive()` had no
    /// per-receive bound, so the 10s "deadline" only applied BETWEEN
    /// frames. This version is structurally incapable of a silent
    /// multi-minute stall:
    ///
    ///  - every wait is bounded (a reader task appends frames to a buffer;
    ///    the test polls the buffer with short timeouts and never awaits a
    ///    raw receive), so worst-case runtime is seconds and a stall fails
    ///    loudly instead of hanging;
    ///  - the blind 300 ms post-upgrade sleep is replaced with a
    ///    deterministic handshake (re-selecting the already-active model
    ///    publishes `modelChanged` unconditionally; poll until the first
    ///    frame proves the subscription is live);
    ///  - prompt delivery is ASSERTED (< 10 s on the suspending clock) —
    ///    it is the PR's feature claim, not a nice-to-have;
    ///  - an `NSProcessInfo` activity assertion opts out of App Nap for
    ///    the duration; and
    ///  - elapsed time is measured on BOTH `ContinuousClock` (advances
    ///    through process suspension) and `SuspendingClock` (does not), so
    ///    a recurrence self-diagnoses: continuous >> suspending in the
    ///    failure message means the PROCESS was suspended, not the engine
    ///    slow.
    @MainActor
    func testEventsWebSocketDeliversModelChangedFrame() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let activity = ProcessInfo.processInfo.beginActivity(
                options: [.userInitiated, .idleSystemSleepDisabled],
                reason: "EngineStateRouteTests WS latency assertion"
            )
            defer { ProcessInfo.processInfo.endActivity(activity) }

            let wsURL = URL(string: "ws://\(EngineConfig.host):\(EngineConfig.port)/v1/events")!
            let session = URLSession(configuration: .ephemeral)
            let wsTask = session.webSocketTask(with: wsURL)
            wsTask.resume()
            defer {
                wsTask.cancel(with: .normalClosure, reason: nil)
                session.invalidateAndCancel()
            }

            // Single long-lived reader owns every raw receive() and appends
            // decoded frames to a buffer; consumers poll the buffer with
            // short bounded waits. No consumer ever awaits a raw receive(),
            // so no wait can outlive its timeout, and no cancellation race
            // can drop a frame (the buffer is append-only until popped).
            let buffer = FrameBuffer()
            let reader = Task {
                while !Task.isCancelled {
                    guard let message = try? await wsTask.receive() else { break }
                    guard case .string(let text) = message,
                          let data = text.data(using: .utf8),
                          let event = try? JSONDecoder().decode(EngineEvent.self, from: data)
                    else { continue }
                    await buffer.append(event)
                }
            }
            defer { reader.cancel() }

            // Handshake: prove the subscription is live before the timed
            // assertion. Re-selecting the already-active model publishes
            // `modelChanged` unconditionally. Bounded: 20 x (POST + 250ms).
            var subscriptionLive = false
            for _ in 0..<20 {
                let (http, _) = try await post(
                    "/v1/touchup/model",
                    body: try JSONEncoder().encode(
                        TouchUpSetModelRequest(id: "yooz-light-v3", preload: false)
                    )
                )
                XCTAssertEqual(http.statusCode, 200)
                if await buffer.popFirst(waitingUpTo: .milliseconds(250)) != nil {
                    subscriptionLive = true
                    break
                }
            }
            guard subscriptionLive else {
                XCTFail("WS subscription never went live: no frame within ~5s across 20 publishes")
                return
            }

            // Timed assertion: switch to quality; the frame must arrive
            // promptly.
            let continuousStart = ContinuousClock.now
            let suspendingStart = SuspendingClock.now
            let (http, _) = try await post(
                "/v1/touchup/model",
                body: try JSONEncoder().encode(
                    TouchUpSetModelRequest(id: "yooz-quality-v3", preload: false)
                )
            )
            XCTAssertEqual(http.statusCode, 200)

            // Bounded scan: at most 40 frame-waits of 250ms each (~10s
            // worst case). Unrelated events from the shared bus (leftover
            // handshake frames included) may interleave.
            let expected = EngineEventPayload.modelChanged(modelId: "yooz-quality-v3")
            var matched: EngineEvent?
            for _ in 0..<40 {
                guard let event = await buffer.popFirst(waitingUpTo: .milliseconds(250)) else { continue }
                if event.module == "touchup", event.payload == expected {
                    matched = event
                    break
                }
            }
            let continuousElapsed = continuousStart.duration(to: .now)
            let suspendingElapsed = suspendingStart.duration(to: .now)

            XCTAssertNotNil(
                matched,
                """
                modelChanged frame not delivered within the bounded scan. \
                continuous=\(continuousElapsed) suspending=\(suspendingElapsed) \
                (continuous >> suspending means the test process was suspended \
                mid-wait — machine sleep / App Nap — not that the engine was slow).
                """
            )
            // The feature claim: events arrive promptly. Measured ~35ms via
            // this stack; 10s is three orders of magnitude of headroom. The
            // suspending clock excludes process-suspension time so an
            // environmental sleep cannot fail this assertion spuriously.
            XCTAssertLessThan(
                suspendingElapsed, .seconds(10),
                "event delivery must be prompt"
            )

            // Restore the default for subsequent tests.
            _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
        }
    }
}

/// Append-only frame buffer shared between the WS reader task and the test
/// body. `popFirst(waitingUpTo:)` polls with short sleeps rather than
/// suspending on the socket, so no wait can outlive its timeout and no
/// cancellation race can drop a frame (a timed-out poll leaves the buffer
/// untouched; the frame is picked up by the next poll).
private actor FrameBuffer {
    private var frames: [EngineEvent] = []

    func append(_ event: EngineEvent) {
        frames.append(event)
    }

    func popFirst(waitingUpTo timeout: Duration) async -> EngineEvent? {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while true {
            if !frames.isEmpty {
                return frames.removeFirst()
            }
            guard ContinuousClock.now < deadline else { return nil }
            try? await Task.sleep(for: .milliseconds(25))
        }
    }
}
