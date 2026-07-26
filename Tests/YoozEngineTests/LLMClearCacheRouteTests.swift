// LLMClearCacheRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for `POST /v1/llm/clear-cache` (engine#299). Boots a
// real `APIServer` and dials it via `URLSession`, same harness as
// `LLMStatusRouteTests` / `TouchUpPickerRouteTests`.
//
// This route is the ONLY one in `APIServer` that hand-collects its request
// body instead of using `request.decode(as:context:)`, because its `model`
// field is optional and an empty body has to mean the same as `{}` ("clear
// every loaded tier") — `request.decode` rejects zero bytes as invalid JSON.
// `InProcessTransport` implements the route separately and therefore never
// exercises that parsing at all, so without these tests the bespoke code path
// has no automated coverage on the transport that actually uses it.
//
// Deliberately cold-engine only: no weights are loaded, so `cleared` is empty
// throughout. What is pinned here is the PARSING and STATUS contract that
// consumers depend on (remi's client distinguishes 400 from 404/501 to decide
// whether the engine simply predates the route); the "cache actually dropped,
// weights stay resident" behavior needs real weights and lives in
// `InProcessLiveModelTests` behind `YOOZ_LLM_LOAD_MODELS=1`.

import Foundation
import XCTest
@testable import EngineCore
@testable import LLMModule
@testable import YoozEngine

final class LLMClearCacheRouteTests: XCTestCase {

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

    /// POST with a RAW body, so a genuinely empty body can be sent — the case
    /// this route exists to special-case.
    private func post(_ path: String, raw: Data?) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = raw
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    private func decodeCleared(_ data: Data) throws -> [String] {
        try JSONDecoder().decode(LLMClearCacheResponse.self, from: data).cleared
    }

    // MARK: - Body parsing: the three spellings of "clear everything"

    @MainActor
    func testEmptyBodyIsAcceptedAndMeansClearEverything() async throws {
        // `request.decode` would reject zero bytes; the handler special-cases
        // it. If that special case regresses, this returns 400.
        try await withServer { _ in
            let (http, body) = try await post("/v1/llm/clear-cache", raw: nil)
            XCTAssertEqual(http.statusCode, 200)
            XCTAssertEqual(try decodeCleared(body), [])
        }
    }

    @MainActor
    func testEmptyJSONObjectMeansClearEverything() async throws {
        // What remi actually sends when it omits the model.
        try await withServer { _ in
            let (http, body) = try await post("/v1/llm/clear-cache", raw: Data("{}".utf8))
            XCTAssertEqual(http.statusCode, 200)
            XCTAssertEqual(try decodeCleared(body), [])
        }
    }

    @MainActor
    func testExplicitNullModelMeansClearEverything() async throws {
        try await withServer { _ in
            let (http, body) = try await post(
                "/v1/llm/clear-cache",
                raw: Data("{\"model\":null}".utf8)
            )
            XCTAssertEqual(http.statusCode, 200)
            XCTAssertEqual(try decodeCleared(body), [])
        }
    }

    // MARK: - Named model

    @MainActor
    func testNamedButNotResidentModelIsASuccessfulNoOp() async throws {
        // Not an error: a consumer reclaiming memory should not have to know
        // what is resident first.
        try await withServer { _ in
            let (http, body) = try await post(
                "/v1/llm/clear-cache",
                raw: Data("{\"model\":\"yooz-quality-v3\"}".utf8)
            )
            XCTAssertEqual(http.statusCode, 200)
            XCTAssertEqual(try decodeCleared(body), [])
        }
    }

    @MainActor
    func testUnknownModelIsRejectedWithInvalidModel() async throws {
        // 400 and NOT 404: remi's client treats 404/501 as "this engine
        // predates the route" and silently disables its cache-drop stage, so
        // a bad id arriving as 404 would quietly turn the feature off instead
        // of reporting a mistake.
        try await withServer { _ in
            let (http, body) = try await post(
                "/v1/llm/clear-cache",
                raw: Data("{\"model\":\"not-a-model\"}".utf8)
            )
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: body) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "invalid_model")
        }
    }

    @MainActor
    func testMalformedBodyIsRejectedAsInvalidRequest() async throws {
        try await withServer { _ in
            let (http, body) = try await post(
                "/v1/llm/clear-cache",
                raw: Data("{not json".utf8)
            )
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: body) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "invalid_request")
        }
    }
}
