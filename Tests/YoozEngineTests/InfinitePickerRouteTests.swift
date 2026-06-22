// InfinitePickerRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest
@testable import EngineCore
@testable import InfiniteModule
@testable import YoozEngine

final class InfinitePickerRouteTests: XCTestCase {

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

    private func post(_ path: String, body: Data) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
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
    func testGetModelsReturnsCanonicalShape() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let (http, body) = try await get("/v1/infinite/models")
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(InfiniteModelsResponse.self, from: body)
            XCTAssertEqual(decoded.models.count, InfiniteModelSelection.allCases.count)
            XCTAssertEqual(decoded.activeId, "gemma4-e4b-1m")
            XCTAssertEqual(decoded.models.filter(\.isActive).count, 1)
            XCTAssertEqual(decoded.models.first(where: \.isActive)?.id, decoded.activeId)
        }
    }

    @MainActor
    func testGetModelsTierAndMetadataMapping() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let (_, body) = try await get("/v1/infinite/models")
            let decoded = try JSONDecoder().decode(InfiniteModelsResponse.self, from: body)
            let byID = Dictionary(uniqueKeysWithValues: decoded.models.map { ($0.id, $0) })
            XCTAssertEqual(byID["gemma4-e4b-1m"]?.tier, .light)
            XCTAssertEqual(byID["gemma4-e4b-1m"]?.ramTier, "reduced")
            XCTAssertEqual(byID["gemma4-e4b-1m"]?.maxContextTokens, 1_000_000)
            XCTAssertEqual(byID["gemma4-e4b-1m"]?.nativeContextTokens, 131_072)
            XCTAssertEqual(
                byID["gemma4-e4b-1m"]?.huggingFaceID,
                "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit"
            )
            XCTAssertEqual(byID["gemma4-e4b-1m"]?.adapterKind, "infinite-paged-kv-mlx-v1")
            XCTAssertTrue(
                [.available, .cached].contains(
                    try XCTUnwrap(byID["gemma4-e4b-1m"]?.loadState)
                )
            )
            XCTAssertEqual(byID["gemma4-26b-a4b-1m"]?.tier, .quality)
            XCTAssertEqual(byID["qwen3-35b-1m"]?.tier, .premium)
            if InfiniteRAMTier.current == .full {
                XCTAssertTrue(
                    [.available, .cached].contains(
                        try XCTUnwrap(byID["qwen3-35b-1m"]?.loadState)
                    )
                )
            } else {
                XCTAssertEqual(byID["qwen3-35b-1m"]?.loadState, .unavailable)
            }
            XCTAssertEqual(byID["s3-retrieval"]?.backendKind, "retrieval")
            XCTAssertEqual(byID["s3-retrieval"]?.adapterKind, "infinite-retrieval-index-v1")
        }
    }

    @MainActor
    func testPostModelWithUnknownIdReturns400InvalidModel() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let body = try JSONEncoder().encode(
                InfiniteSetModelRequest(id: "missing-model", preload: false)
            )
            let (http, payload) = try await post("/v1/infinite/model", body: body)
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "invalid_model")
        }
    }

    @MainActor
    func testPostModelWithMalformedBodyReturns400InvalidRequest() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let body = Data("not json".utf8)
            let (http, payload) = try await post("/v1/infinite/model", body: body)
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "invalid_request")
        }
    }

    @MainActor
    func testPostModelWithoutPreloadReturns200AndActiveRow() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let selection: InfiniteModelSelection =
                InfiniteRAMTier.current == .full ? .gemma4_26B_A4B1M : .gemma4E4B1M
            let body = try JSONEncoder().encode(
                InfiniteSetModelRequest(id: selection.rawValue, preload: false)
            )
            let (http, payload) = try await post("/v1/infinite/model", body: body)
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(InfiniteModelInfo.self, from: payload)
            XCTAssertEqual(decoded.id, selection.rawValue)
            XCTAssertTrue(decoded.isActive)
            XCTAssertTrue([.available, .cached].contains(decoded.loadState))
        }
    }

    @MainActor
    func testPostModelWithPreloadReturns200AndAdapterReadyStatus() async throws {
        try requireSupportedTier()
        try await withServer { _ in
            let body = try JSONEncoder().encode(
                InfiniteSetModelRequest(id: "gemma4-e4b-1m", preload: true)
            )
            let (http, payload) = try await post("/v1/infinite/model", body: body)
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(InfiniteModelInfo.self, from: payload)
            XCTAssertEqual(decoded.id, "gemma4-e4b-1m")

            let (_, statusPayload) = try await get("/v1/infinite/status")
            let status = try JSONDecoder().decode(InfiniteStatus.self, from: statusPayload)
            XCTAssertFalse(status.loaded)
            XCTAssertEqual(status.state, "adapter_ready")
        }
    }

    /// When the module isn't bundled, every `/v1/infinite/*` route returns
    /// 501 `module_not_bundled` (the Lite/Whisper-variant behaviour). Runs on
    /// any tier — the not-bundled guard precedes any RAM-tier logic.
    @MainActor
    func testInfiniteRoutesReturn501WhenModuleNotBundled() async throws {
        UniqueEnginePort.assignFreshPort()
        await ModuleRegistry.shared.reset()
        let server = APIServer()
        try await server.start()
        do {
            let (http, payload) = try await get("/v1/infinite/models")
            XCTAssertEqual(http.statusCode, 501)
            let json = try XCTUnwrap(
                JSONSerialization.jsonObject(with: payload) as? [String: Any]
            )
            XCTAssertEqual(json["code"] as? String, "module_not_bundled")
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
    }
}
