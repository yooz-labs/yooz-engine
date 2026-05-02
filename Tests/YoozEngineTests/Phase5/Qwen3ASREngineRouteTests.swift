// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngine

/// Phase 5 — `/v1/stt/engine` GET + POST and qwen3-aware routing on
/// `/v1/stt/load`. We boot a real `APIServer` (it binds to
/// `localhost:19920` per `EngineConfig`) and dial it with
/// `URLSession`. Each test class instance gets its own server
/// lifecycle so XCTest's serial execution stays correct.
///
/// Why not Hummingbird's in-process `.router` test framework?
/// Pulling `HummingbirdTesting` into the test target triggers a
/// Logging-resolution linker bug in `swift-websocket`'s `WSCore` and
/// in `HummingbirdCore` itself when the test target re-links them.
/// The live-server path avoids that linker tarpit and exercises the
/// same code path the engine ships.
final class Qwen3ASREngineRouteTests: XCTestCase {

    // MARK: - Server lifecycle helpers

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

    private func get(
        _ path: String, on server: APIServer
    ) async throws -> (HTTPURLResponse, Data) {
        let url = baseURL().appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "test", code: 0)
        }
        return (http, data)
    }

    private func post(
        _ path: String,
        body: Data,
        on server: APIServer
    ) async throws -> (HTTPURLResponse, Data) {
        let url = baseURL().appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "test", code: 0)
        }
        return (http, data)
    }

    @MainActor
    private func resetEngineState() async {
        await YoozSTTEngine.shared.setBackend(.parakeet)
    }

    // MARK: - GET /v1/stt/engine

    @MainActor
    func testGetEngineListsAllBackends() async throws {
        await resetEngineState()
        try await withServer { server in
            let (http, body) = try await get("/v1/stt/engine", on: server)
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(
                STTEngineGetResponse.self, from: body
            )
            XCTAssertEqual(decoded.current, "parakeet")
            let ids = Set(decoded.available.map(\.id))
            XCTAssertEqual(
                ids, Set(STTBackendID.allCases.map(\.rawValue))
            )
            let qwen3 = decoded.available.first {
                $0.id == "qwen3_asr_preview"
            }
            XCTAssertNotNil(qwen3)
            XCTAssertTrue(qwen3?.supportsBatch ?? false)
            // Phase 7 (issue #61) flipped streaming support on for
            // the qwen3 backend. The capability is now exposed via
            // the `/v1/stt/engine` GET response.
            XCTAssertTrue(qwen3?.supportsStreaming ?? false)
        }
    }

    // MARK: - POST /v1/stt/engine

    @MainActor
    func testPostEngineSwitchesToQwen3() async throws {
        await resetEngineState()
        try await withServer { server in
            let payload = try JSONEncoder().encode(
                STTEnginePostRequest(engine: "qwen3_asr_preview")
            )
            let (http, body) = try await post(
                "/v1/stt/engine", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(
                STTEnginePostResponse.self, from: body
            )
            XCTAssertEqual(decoded.current, "qwen3_asr_preview")

            // GET should now report the new state.
            let (_, getBody) = try await get("/v1/stt/engine", on: server)
            let getDecoded = try JSONDecoder().decode(
                STTEngineGetResponse.self, from: getBody
            )
            XCTAssertEqual(getDecoded.current, "qwen3_asr_preview")
        }
        // Cleanup global state.
        await resetEngineState()
    }

    @MainActor
    func testPostEngineRejectsUnknownValue() async throws {
        await resetEngineState()
        try await withServer { server in
            let payload = try JSONEncoder().encode(
                STTEnginePostRequest(engine: "no-such-backend")
            )
            let (http, body) = try await post(
                "/v1/stt/engine", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 400)
            let decoded = try JSONDecoder().decode(
                ErrorResponse.self, from: body
            )
            XCTAssertEqual(decoded.code, "invalid_engine")
        }
    }

    // MARK: - POST /v1/stt/load

    /// With qwen3 selected and `allow_fetch: false`, the load route
    /// must surface `model_not_found` rather than silently falling
    /// back. CI-safe: no network, no model on disk.
    @MainActor
    func testLoadRefusesFetchWhenDisabled() async throws {
        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer { Task { @MainActor in await self.resetEngineState() } }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-load-test-\(UUID().uuidString)")
        setenv("YOOZ_QWEN3_ASR_DIR", tmp.path, 1)
        defer { unsetenv("YOOZ_QWEN3_ASR_DIR") }

        try await withServer { server in
            struct Body: Encodable {
                let language: String
                let allowFetch: Bool
            }
            let payload = try JSONEncoder().encode(
                Body(language: "en", allowFetch: false)
            )
            let (http, body) = try await post(
                "/v1/stt/load", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 404)
            let decoded = try JSONDecoder().decode(
                ErrorResponse.self, from: body
            )
            XCTAssertEqual(decoded.code, "model_not_found")
        }
    }

    // MARK: - Sanity: existing routes still work

    @MainActor
    func testLanguagesRouteStillWorks() async throws {
        await resetEngineState()
        try await withServer { server in
            let (http, _) = try await get("/v1/stt/languages", on: server)
            XCTAssertEqual(http.statusCode, 200)
        }
    }

    @MainActor
    func testHealthRouteStillWorks() async throws {
        await resetEngineState()
        try await withServer { server in
            let (http, body) = try await get("/v1/health", on: server)
            XCTAssertEqual(http.statusCode, 200)
            // Health body should decode as a HealthResponse.
            let decoded = try JSONDecoder().decode(
                HealthResponse.self, from: body
            )
            XCTAssertEqual(decoded.status, "ok")
        }
    }
}
