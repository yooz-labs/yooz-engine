// InfiniteStatusRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest
@testable import EngineCore
@testable import InfiniteModule
@testable import YoozEngine

final class InfiniteStatusRouteTests: XCTestCase {

    @MainActor
    private func withServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
        UniqueEnginePort.assignFreshPort()
        await ModuleRegistry.shared.register(InfiniteEngine.shared)
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

    private func post(_ path: String, body: Data = Data()) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    private func delete(_ path: String) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "DELETE"
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    override func setUp() async throws {
        await InfiniteEngine.shared.reset()
    }

    override func tearDown() async throws {
        await InfiniteEngine.shared.reset()
    }

    private func requireSupportedTier() throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
    }

    @MainActor
    func testGetStatusOnColdEngineReportsActiveModel() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let (http, body) = try await get("/v1/infinite/status")
            XCTAssertEqual(http.statusCode, 200)
            let status = try JSONDecoder().decode(InfiniteStatus.self, from: body)
            XCTAssertFalse(status.loaded)
            XCTAssertEqual(status.modelId, "gemma4-e4b-1m")
            XCTAssertNil(status.progress)
            XCTAssertEqual(status.state, "idle")
            // The default active model (gemma4-e4b-1m) is a `reduced`-tier model
            // regardless of the host machine's tier, so this is host-independent.
            XCTAssertEqual(status.activeSessions, 0)
            XCTAssertEqual(status.maxContextTokens, 1_000_000)
            XCTAssertEqual(status.ramTier, "reduced")
            XCTAssertEqual(status.backendKind, "paged-kv")
        }
    }

    @MainActor
    func testModulesManifestIncludesInfinite() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let (http, body) = try await get("/v1/modules")
            XCTAssertEqual(http.statusCode, 200)
            let manifest = try JSONDecoder().decode(ModulesResponse.self, from: body)
            let infinite = try XCTUnwrap(
                manifest.modules.first(where: { $0.name == "infinite" })
            )
            XCTAssertFalse(infinite.loaded)
            XCTAssertEqual(infinite.detail["active_model"], "gemma4-e4b-1m")
            XCTAssertEqual(infinite.detail["adapter_kind"], "infinite-paged-kv-mlx-v1")
            XCTAssertEqual(
                infinite.detail["hf_repo"],
                "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit"
            )
        }
    }

    @MainActor
    func testInfiniteSessionRoutesSurviveRecordingReset() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let createBody = try JSONEncoder().encode(
                InfiniteCreateSessionRequest(label: "route-session")
            )
            let (createHTTP, createPayload) = try await post(
                "/v1/infinite/sessions",
                body: createBody
            )
            XCTAssertEqual(createHTTP.statusCode, 200)
            let created = try JSONDecoder().decode(InfiniteSessionInfo.self, from: createPayload)
            XCTAssertEqual(created.modelId, "gemma4-e4b-1m")
            XCTAssertEqual(created.label, "route-session")

            let appendBody = try JSONEncoder().encode(
                InfiniteAppendSessionRequest(text: "real route context")
            )
            let (appendHTTP, appendPayload) = try await post(
                "/v1/infinite/sessions/\(created.id)/append",
                body: appendBody
            )
            XCTAssertEqual(appendHTTP.statusCode, 200)
            let appended = try JSONDecoder().decode(
                InfiniteAppendSessionResponse.self,
                from: appendPayload
            )
            XCTAssertEqual(appended.session.inputCharacters, 18)

            let (beginHTTP, _) = try await post("/v1/session/begin")
            XCTAssertEqual(beginHTTP.statusCode, 200)

            let (getHTTP, getPayload) = try await get("/v1/infinite/sessions/\(created.id)")
            XCTAssertEqual(getHTTP.statusCode, 200)
            let fetched = try JSONDecoder().decode(InfiniteSessionInfo.self, from: getPayload)
            XCTAssertEqual(fetched.id, created.id)
            XCTAssertEqual(fetched.inputCharacters, 18)

            let (deleteHTTP, deletePayload) = try await delete(
                "/v1/infinite/sessions/\(created.id)"
            )
            XCTAssertEqual(deleteHTTP.statusCode, 200)
            let deleted = try JSONDecoder().decode(
                InfiniteDeleteSessionResponse.self,
                from: deletePayload
            )
            XCTAssertTrue(deleted.deleted)
        }
    }

    /// Retrieval is the only row with no MLX backend, so generate on it returns
    /// 501 generation_unavailable. The MLX rows (Qwen, Gemma4 26B/E4B) all run
    /// now, so this is the remaining route-level refusal case. Full-tier only.
    @MainActor
    func testGenerateOnRetrievalModelReturns501() async throws {
        guard InfiniteRAMTier.current == .full else {
            throw XCTSkip("retrieval mode (the only non-MLX row) is full-tier; needs 64 GB")
        }
        try await withServer { _ in
            let setBody = try JSONEncoder().encode(
                InfiniteSetModelRequest(id: "s3-retrieval", preload: false)
            )
            let (setHTTP, _) = try await post("/v1/infinite/model", body: setBody)
            XCTAssertEqual(setHTTP.statusCode, 200)

            let createBody = try JSONEncoder().encode(InfiniteCreateSessionRequest())
            let (createHTTP, createPayload) = try await post(
                "/v1/infinite/sessions", body: createBody
            )
            XCTAssertEqual(createHTTP.statusCode, 200)
            let created = try JSONDecoder().decode(InfiniteSessionInfo.self, from: createPayload)
            XCTAssertEqual(created.modelId, "s3-retrieval")

            let generateBody = try JSONEncoder().encode(
                InfiniteGenerateSessionRequest(prompt: "summarize", maxTokens: 16)
            )
            let (generateHTTP, generatePayload) = try await post(
                "/v1/infinite/sessions/\(created.id)/generate", body: generateBody
            )
            XCTAssertEqual(generateHTTP.statusCode, 501)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: generatePayload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "generation_unavailable")
        }
    }

    @MainActor
    func testSessionLimitReturns409() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let body = try JSONEncoder().encode(InfiniteCreateSessionRequest(label: "limit"))
            for _ in 0..<InfiniteEngine.maxActiveSessions {
                let (http, _) = try await post("/v1/infinite/sessions", body: body)
                XCTAssertEqual(http.statusCode, 200)
            }
            let (overflow, payload) = try await post("/v1/infinite/sessions", body: body)
            XCTAssertEqual(overflow.statusCode, 409)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "session_limit_exceeded")
        }
    }

    @MainActor
    func testGetUnknownSessionReturns404() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let (http, payload) = try await get("/v1/infinite/sessions/nonexistent-id")
            XCTAssertEqual(http.statusCode, 404)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "session_not_found")
        }
    }
}
