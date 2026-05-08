// TouchUpPickerRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for the canonical picker (issue #97). Boots
// a real `APIServer` and dials it via `URLSession` — same harness
// as `Qwen3ASREngineRouteTests` (the in-process Hummingbird test
// path triggers a Logging-resolution linker bug per that file's
// header).
//
// Pins three things the unit tests cannot reach:
//   1. JSON wire shape of `GET /v1/touchup/models`.
//   2. Status-code mapping in `POST /v1/touchup/model` —
//      400 invalid_model / 501 model_unavailable / 200 success.
//   3. The `code` field per error path. Picker UIs branch on
//      `code` to render contextual messages; renaming a code is a
//      user-visible regression.

import Foundation
import XCTest
@testable import YoozEngine
@testable import EngineCore
@testable import LLMModule

final class TouchUpPickerRouteTests: XCTestCase {

    @MainActor
    private func withServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
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

    /// Reset the singleton's active model between tests — `setActiveModel`
    /// from a prior test would otherwise leak into the next one.
    @MainActor
    private func resetEngineState() async throws {
        _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
    }

    // MARK: - GET /v1/touchup/models

    @MainActor
    func testGetModelsReturnsCanonicalShape() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (http, body) = try await get("/v1/touchup/models")
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(TouchUpModelsResponse.self, from: body)
            XCTAssertEqual(decoded.models.count, TouchUpModelSelection.allCases.count)
            XCTAssertEqual(decoded.activeId, TouchUpModelSelection.yoozLight.rawValue)
            // Exactly one active row — pinned both engine-side
            // (precondition) and over the wire.
            XCTAssertEqual(decoded.models.filter(\.isActive).count, 1)
            XCTAssertEqual(
                decoded.models.first(where: { $0.isActive })?.id,
                decoded.activeId
            )
        }
    }

    /// Picker UIs key off the per-row `tier` to render Pro badges
    /// and sort hints. Pin the mapping over the wire so a backend
    /// rename can't silently flip a tier.
    @MainActor
    func testGetModelsTierMapping() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (_, body) = try await get("/v1/touchup/models")
            let decoded = try JSONDecoder().decode(TouchUpModelsResponse.self, from: body)
            let byID = Dictionary(uniqueKeysWithValues: decoded.models.map { ($0.id, $0) })
            XCTAssertEqual(byID["yooz-light-v3"]?.tier, .light)
            XCTAssertEqual(byID["yooz-quality-v3"]?.tier, .quality)
            XCTAssertEqual(byID["foundation-models"]?.tier, .premium)
        }
    }

    // MARK: - POST /v1/touchup/model

    @MainActor
    func testPostModelWithUnknownIdReturns400InvalidModel() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let body = try JSONEncoder().encode(
                TouchUpSetModelRequest(id: "totally-fake-model-id", preload: false)
            )
            let (http, payload) = try await post("/v1/touchup/model", body: body)
            XCTAssertEqual(http.statusCode, 400)

            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            // Per `errorResponse` shape — `code` field is the
            // contract picker UIs branch on.
            XCTAssertEqual(json["code"] as? String, "invalid_model")
        }
    }

    @MainActor
    func testPostModelWithMalformedBodyReturns400InvalidRequest() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let body = Data("not json at all".utf8)
            let (http, payload) = try await post("/v1/touchup/model", body: body)
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "invalid_request")
        }
    }

    @MainActor
    func testPostModelWithFoundationModelsOn26MinusReturns501() async throws {
        // Self-skip when Apple Intelligence really is available —
        // the unavailable-branch is the one we need to pin.
        guard !FoundationModelsBackend().isAvailable() else { return }

        try await resetEngineState()
        try await withServer { _ in
            let body = try JSONEncoder().encode(
                TouchUpSetModelRequest(id: "foundation-models", preload: true)
            )
            let (http, payload) = try await post("/v1/touchup/model", body: body)
            XCTAssertEqual(http.statusCode, 501)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "model_unavailable")
        }
    }

    /// Happy path. Picking `.yoozLight` with `preload: false` is
    /// the only branch we can exercise without weights on disk on
    /// CI; it's enough to pin the success status code + response
    /// shape (a TouchUpModelInfo for the new active row).
    @MainActor
    func testPostModelYoozLightWithoutPreloadReturns200AndActiveRow() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let body = try JSONEncoder().encode(
                TouchUpSetModelRequest(id: "yooz-light-v3", preload: false)
            )
            let (http, payload) = try await post("/v1/touchup/model", body: body)
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(TouchUpModelInfo.self, from: payload)
            XCTAssertEqual(decoded.id, "yooz-light-v3")
            XCTAssertTrue(decoded.isActive)
            XCTAssertEqual(decoded.tier, .light)
        }
    }
}
