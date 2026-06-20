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

    private func resetEngineState() async throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
        _ = try await InfiniteEngine.shared.setActiveModel(.gemma4E4B1M, preload: false)
    }

    @MainActor
    func testGetStatusOnColdEngineReportsActiveModel() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (http, body) = try await get("/v1/infinite/status")
            XCTAssertEqual(http.statusCode, 200)
            let status = try JSONDecoder().decode(InfiniteStatus.self, from: body)
            XCTAssertFalse(status.loaded)
            XCTAssertEqual(status.modelId, "gemma4-e4b-1m")
            XCTAssertNil(status.progress)
            XCTAssertEqual(status.state, "idle")
            XCTAssertEqual(status.activeSessions, 0)
            XCTAssertEqual(status.maxContextTokens, 1_000_000)
            XCTAssertEqual(status.ramTier, "reduced")
            XCTAssertEqual(status.backendKind, "paged-kv")
        }
    }

    @MainActor
    func testModulesManifestIncludesInfinite() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (http, body) = try await get("/v1/modules")
            XCTAssertEqual(http.statusCode, 200)
            let manifest = try JSONDecoder().decode(ModulesResponse.self, from: body)
            let infinite = try XCTUnwrap(
                manifest.modules.first(where: { $0.name == "infinite" })
            )
            XCTAssertFalse(infinite.loaded)
            XCTAssertEqual(infinite.detail["active_model"], "gemma4-e4b-1m")
        }
    }
}
