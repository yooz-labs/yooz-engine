// LLMStatusRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for `/v1/llm/status` (engine#124). Boots a
// real `APIServer` and dials it via `URLSession`. Same harness as
// `TouchUpPickerRouteTests` — see that file for the rationale on
// why in-process Hummingbird tests aren't used.
//
// Pins three things the unit tests cannot reach:
//   1. JSON wire shape of `GET /v1/llm/status` (loaded, modelId,
//      progress fields).
//   2. The route follows `activeModel` (what the picker mutates),
//      not `preferredModel`. Switching via the touch-up picker
//      changes which backend's download is reported.
//   3. The progress filter: 0.0 fraction collapses to nil; loaded
//      tier reports nil; foundation-models always reports nil.

import Foundation
import XCTest
@testable import YoozEngine
@testable import EngineCore
@testable import LLMModule
@testable import YoozEngineClient

final class LLMStatusRouteTests: XCTestCase {

    @MainActor
    private func withServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
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

    /// Reset the singleton's active model between tests so a prior
    /// run's picker selection doesn't leak into this one.
    @MainActor
    private func resetEngineState() async throws {
        _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
    }

    // MARK: - Cold engine: nothing loaded, no download in flight

    @MainActor
    func testGetStatusOnColdEngineReportsNotLoadedNilProgress() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (http, body) = try await get("/v1/llm/status")
            XCTAssertEqual(http.statusCode, 200)
            let status = try JSONDecoder().decode(LLMStatus.self, from: body)
            XCTAssertFalse(status.loaded,
                           "Cold engine: light backend not instantiated, can't be loaded")
            XCTAssertEqual(status.modelId, "yooz-light-v2",
                           "Active model defaults to yoozLight after reset")
            XCTAssertNil(status.progress,
                         "0.0 fraction must collapse to nil (idle, banner hides)")
        }
    }

    /// Picker swap: switch the active model to Apple Intelligence
    /// (which has no HF download). The status route must follow the
    /// picker's selection — not the LLM-only `preferredModel` field.
    @MainActor
    func testStatusFollowsActivePickerSelectionFoundationModels() async throws {
        try await resetEngineState()
        try await withServer { _ in
            _ = try await TouchUpEngine.shared.setActiveModel(
                .foundationModels,
                preload: false
            )
            let (_, body) = try await get("/v1/llm/status")
            let status = try JSONDecoder().decode(LLMStatus.self, from: body)
            XCTAssertEqual(status.modelId, "foundation-models",
                           "Status route must report the picker's active id, not preferredModel")
            XCTAssertNil(status.progress,
                         "Foundation Models has no HF download to report")
            // Reset for the next test.
            _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
        }
    }

    /// Picker swap to Quality: status route must read the quality
    /// backend's progress rather than the (stale) preferredModel
    /// pointing at light.
    @MainActor
    func testStatusFollowsActivePickerSelectionQuality() async throws {
        try await resetEngineState()
        try await withServer { _ in
            _ = try await TouchUpEngine.shared.setActiveModel(
                .yoozQuality,
                preload: false
            )
            let (_, body) = try await get("/v1/llm/status")
            let status = try JSONDecoder().decode(LLMStatus.self, from: body)
            XCTAssertEqual(status.modelId, "yooz-quality-v2",
                           "Status route must report the picker's active id")
            XCTAssertFalse(status.loaded,
                           "Quality not preloaded yet")
            XCTAssertNil(status.progress,
                         "Backend not instantiated yet -> nil progress")
            _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
        }
    }
}
