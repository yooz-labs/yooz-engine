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
    @MainActor
    func testEventsWebSocketDeliversModelChangedFrame() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let wsURL = URL(string: "ws://\(EngineConfig.host):\(EngineConfig.port)/v1/events")!
            let session = URLSession(configuration: .ephemeral)
            let wsTask = session.webSocketTask(with: wsURL)
            wsTask.resume()
            defer {
                wsTask.cancel(with: .normalClosure, reason: nil)
                session.invalidateAndCancel()
            }

            // Give the WS upgrade a moment to complete before publishing —
            // the bus has no replay, so a frame published before the
            // subscription lands is legitimately dropped.
            try await Task.sleep(for: .milliseconds(300))

            let (http, _) = try await post(
                "/v1/touchup/model",
                body: try JSONEncoder().encode(
                    TouchUpSetModelRequest(id: "yooz-quality-v2", preload: false)
                )
            )
            XCTAssertEqual(http.statusCode, 200)

            // Scan (bounded) for the frame this test caused; unrelated
            // events from the shared bus may interleave.
            let expected = EngineEventPayload.modelChanged(modelId: "yooz-quality-v2")
            var matched: EngineEvent?
            let deadline = ContinuousClock.now.advanced(by: .seconds(10))
            while ContinuousClock.now < deadline {
                let message = try await wsTask.receive()
                guard case .string(let text) = message,
                      let data = text.data(using: .utf8),
                      let event = try? JSONDecoder().decode(EngineEvent.self, from: data)
                else { continue }
                if event.module == "touchup", event.payload == expected {
                    matched = event
                    break
                }
            }
            XCTAssertNotNil(
                matched,
                "expected the modelChanged frame over the live /v1/events WebSocket"
            )

            // Restore the default for subsequent tests.
            _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
        }
    }
}
