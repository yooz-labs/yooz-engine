// GPUAdmissionRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for the optional `workloadClass` GPU-admission field
// (engine#228) on `POST /v1/touchup` and `POST /v1/llm/generate`. Boots a
// real `APIServer` and dials it via `URLSession` — same harness as
// `TouchUpPickerRouteTests`.
//
// Pins the wire contract of acceptance criterion #1 ("workload class
// visible on the wire and defaulted sensibly"):
//   1. Omitting `workloadClass` preserves today's behavior (request
//      succeeds; the engine defaults to `.background`).
//   2. A known value ("interactive"/"background") round-trips.
//   3. An unknown value fails the body decode -> 400 `invalid_request`
//      on both routes — a declared scheduling class is explicit caller
//      intent, never silently downgraded.
//
// TouchUp requests use mode "off" (regex-only) so no test triggers an LLM
// load. The `/v1/llm/generate` case only exercises the 400 decode path for
// the same reason — a successful generate needs real weights.

import Foundation
import XCTest
@testable import YoozEngine
@testable import EngineCore
@testable import LLMModule

final class GPUAdmissionRouteTests: XCTestCase {

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

    private func post(_ path: String, json: String) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(json.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    private func errorCode(in body: Data) throws -> String {
        struct ErrorBody: Decodable { let code: String? }
        return try XCTUnwrap(
            JSONDecoder().decode(ErrorBody.self, from: body).code,
            "error response should carry a code field"
        )
    }

    // MARK: - POST /v1/touchup

    @MainActor
    func testTouchUpOmittedWorkloadClassSucceeds() async throws {
        try await withServer { _ in
            let (http, _) = try await post(
                "/v1/touchup",
                json: #"{"text":"hello world","mode":"off"}"#
            )
            XCTAssertEqual(http.statusCode, 200)
        }
    }

    @MainActor
    func testTouchUpKnownWorkloadClassSucceeds() async throws {
        try await withServer { _ in
            let (http, _) = try await post(
                "/v1/touchup",
                json: #"{"text":"hello world","mode":"off","workloadClass":"interactive"}"#
            )
            XCTAssertEqual(http.statusCode, 200)
        }
    }

    @MainActor
    func testTouchUpUnknownWorkloadClassIs400InvalidRequest() async throws {
        try await withServer { _ in
            let (http, body) = try await post(
                "/v1/touchup",
                json: #"{"text":"hello world","mode":"off","workloadClass":"turbo"}"#
            )
            XCTAssertEqual(http.statusCode, 400)
            XCTAssertEqual(try errorCode(in: body), "invalid_request")
        }
    }

    // MARK: - POST /v1/llm/generate

    @MainActor
    func testGenerateUnknownWorkloadClassIs400InvalidRequest() async throws {
        try await withServer { _ in
            let (http, body) = try await post(
                "/v1/llm/generate",
                json: #"{"prompt":"hi","workloadClass":"turbo"}"#
            )
            XCTAssertEqual(http.statusCode, 400)
            XCTAssertEqual(try errorCode(in: body), "invalid_request")
        }
    }
}
