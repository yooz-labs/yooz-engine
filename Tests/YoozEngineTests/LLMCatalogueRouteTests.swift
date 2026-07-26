// LLMCatalogueRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for the engine#303 catalogue-backed `LLMModelType`:
// `GET /v1/llm/models` must enumerate every catalogued model (not a
// hardcoded light/quality pair), and `POST /v1/llm/generate`'s
// `invalid_model` gate must still 400 an unknown id while listing every
// catalogued id as "Available:". Boots a real `APIServer` and dials it via
// `URLSession`, same harness as `LLMClearCacheRouteTests` / `LLMStatusRouteTests`.
//
// Deliberately does NOT exercise a real `/v1/llm/generate` call against the
// new catalogue entry — that would trigger a multi-GB HF download on first
// run. The resolution contract itself (`LLMModelType(rawValue:)`, including
// HF-repo-id alias resolution) is unit-tested in `LLMModuleTests`; this file
// only proves the route wiring: the gate accepts the new id (indirectly, via
// the 400 message listing it) and the picker-surface route reports it.

import Foundation
import XCTest
@testable import EngineCore
@testable import LLMModule
@testable import YoozEngine

final class LLMCatalogueRouteTests: XCTestCase {

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

    private func post(_ path: String, body: Data) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    // MARK: - GET /v1/llm/models

    @MainActor
    func testGetModelsEnumeratesFullCatalogue() async throws {
        try await withServer { _ in
            let (data, response) = try await URLSession.shared.data(
                from: baseURL().appendingPathComponent("/v1/llm/models")
            )
            let http = try XCTUnwrap(response as? HTTPURLResponse)
            XCTAssertEqual(http.statusCode, 200)

            let decoded = try JSONDecoder().decode(LLMModelsResponse.self, from: data)
            let ids = Set(decoded.available.map(\.id))
            XCTAssertEqual(
                ids, Set(LLMModelType.allCases.map(\.rawValue)),
                "GET /v1/llm/models must list every catalogued model, not a fixed pair"
            )
            XCTAssertTrue(ids.contains("yooz-instruct-4b"))

            let instruct = try XCTUnwrap(decoded.available.first { $0.id == "yooz-instruct-4b" })
            XCTAssertEqual(instruct.purpose, .general,
                           "the general/classify base must be distinguishable from TouchUp tiers")
            let light = try XCTUnwrap(decoded.available.first { $0.id == "yooz-light-v3" })
            XCTAssertEqual(light.purpose, .proofread)
            let quality = try XCTUnwrap(decoded.available.first { $0.id == "yooz-quality-v3" })
            XCTAssertEqual(quality.purpose, .proofread)
        }
    }

    // MARK: - POST /v1/llm/generate gate

    @MainActor
    func testGenerateInvalidModelListsEveryCatalogueEntry() async throws {
        try await withServer { _ in
            let body = try JSONEncoder().encode(
                LLMGenerateRequest(prompt: "hi", model: "not-a-real-model")
            )
            let (http, data) = try await post("/v1/llm/generate", body: body)
            XCTAssertEqual(http.statusCode, 400)

            let json = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: data) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "invalid_model")
            // Wire shape is `ErrorResponse { error, code }` (APIServer.swift
            // `errorResponse(status:message:code:)`) — the human-readable
            // text lands under the `error` key, not `message`.
            let message = try XCTUnwrap(json["error"] as? String)
            XCTAssertTrue(
                message.contains("yooz-instruct-4b"),
                "the gate's error message must list the new catalogue entry; got: \(message)"
            )
        }
    }

    @MainActor
    func testGenerateUnrelatedYoozLabsRepoIsStillRejected() async throws {
        // Curated, not a bare `YoozLabs/` prefix rule: an org repo that is
        // not a servable MLX causal LM must still 400, not silently trigger
        // a multi-GB download that fails at load.
        try await withServer { _ in
            let body = try JSONEncoder().encode(
                LLMGenerateRequest(prompt: "hi", model: "YoozLabs/Qwen3-ASR-1.7B-8bit")
            )
            let (http, data) = try await post("/v1/llm/generate", body: body)
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: data) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "invalid_model")
        }
    }
}
