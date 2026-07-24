// ModelManagementRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for the disk-hygiene endpoints (issue #256):
// `GET /v1/models`, `DELETE /v1/models/:id`, `POST /v1/models/cleanup`. Boots a
// real `APIServer` and dials it via `URLSession`, same harness as
// `TouchUpPickerRouteTests`.
//
// Safety: every assertion is non-destructive to the machine's real cache. GET is
// read-only; the DELETE cases hit the active-model (409) and unknown-id (404)
// guards that short-circuit before any disk removal; cleanup runs against an
// `HF_HOME`-redirected empty temp hub.

import Foundation
import XCTest
@testable import YoozEngine
@testable import EngineCore
@testable import LLMModule

final class ModelManagementRouteTests: XCTestCase {

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

    private func send(_ method: String, _ path: String) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = method
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    @MainActor
    private func resetEngineState() async throws {
        _ = try await TouchUpEngine.shared.setActiveModel(.yoozLight, preload: false)
    }

    // MARK: - GET /v1/models

    @MainActor
    func testGetModelsReturns200AndConsistentInventory() async throws {
        try await resetEngineState()
        try await withServer { _ in
            let (http, body) = try await send("GET", "/v1/models")
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(ManagedModelsResponse.self, from: body)
            for model in decoded.models {
                if !model.cached { XCTAssertEqual(model.sizeBytes, 0, "\(model.id)") }
                if model.isActive { XCTAssertFalse(model.deletable, "\(model.id)") }
                if model.deletable {
                    XCTAssertGreaterThan(model.sizeBytes, 0, "\(model.id)")
                    XCTAssertFalse(model.isActive, "\(model.id)")
                }
            }
        }
    }

    // MARK: - DELETE /v1/models/:id

    @MainActor
    func testDeleteUnknownModelReturns404() async throws {
        try await withServer { _ in
            let (http, payload) = try await send("DELETE", "/v1/models/totally-unknown-model")
            XCTAssertEqual(http.statusCode, 404)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "unknown_model")
        }
    }

    @MainActor
    func testDeleteActiveModelReturns409() async throws {
        try await resetEngineState()  // active == yooz-light-v3
        try await withServer { _ in
            let (http, payload) = try await send("DELETE", "/v1/models/yooz-light-v3")
            XCTAssertEqual(http.statusCode, 409)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "model_active")
        }
    }

    // MARK: - POST /v1/models/cleanup

    @MainActor
    func testCleanupReturns200OnRedirectedEmptyCache() async throws {
        let fm = FileManager.default
        let home = fm.temporaryDirectory
            .appendingPathComponent("route-cleanup-\(UUID().uuidString)")
        try fm.createDirectory(
            at: home.appendingPathComponent("hub"), withIntermediateDirectories: true
        )
        let saved = ProcessInfo.processInfo.environment["HF_HOME"]
        setenv("HF_HOME", home.path, 1)
        defer {
            if let saved { setenv("HF_HOME", saved, 1) } else { unsetenv("HF_HOME") }
            try? fm.removeItem(at: home)
        }

        try await withServer { _ in
            let (http, body) = try await send("POST", "/v1/models/cleanup")
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(ModelCleanupResult.self, from: body)
            // Empty hub -> nothing to reclaim.
            XCTAssertEqual(decoded.totalReclaimedBytes, 0)
            XCTAssertTrue(decoded.perRepo.isEmpty)
        }
    }
}
