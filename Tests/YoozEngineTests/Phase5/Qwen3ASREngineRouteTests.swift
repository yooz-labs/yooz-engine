// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import STTModule
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

    // MARK: - GET /v1/stt/engine — canonical picker shape (#99)

    @MainActor
    func testGetEngineListsAllBackends() async throws {
        await resetEngineState()
        try await withServer { server in
            let (http, body) = try await get("/v1/stt/engine", on: server)
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(
                STTBackendsResponse.self, from: body
            )
            XCTAssertEqual(decoded.activeId, "parakeet")
            let ids = Set(decoded.backends.map(\.id))
            XCTAssertEqual(
                ids, Set(STTBackendID.allCases.map(\.rawValue))
            )
            let qwen3 = decoded.backends.first {
                $0.id == "qwen3_asr_preview"
            }
            XCTAssertNotNil(qwen3)
            XCTAssertTrue(qwen3?.supportsBatch ?? false)
            // Phase 7 (issue #61) flipped streaming support on for
            // the qwen3 backend. The capability is now exposed via
            // the canonical picker's `supportsStreaming` extension.
            XCTAssertTrue(qwen3?.supportsStreaming ?? false)
            // Canonical picker invariant: exactly one row active.
            XCTAssertEqual(decoded.backends.filter(\.isActive).count, 1)
        }
    }

    // MARK: - POST /v1/stt/engine — canonical picker shape

    @MainActor
    func testPostEngineSwitchesToQwen3() async throws {
        await resetEngineState()
        try await withServer { server in
            let payload = try JSONEncoder().encode(
                STTSetBackendRequest(
                    id: "qwen3_asr_preview",
                    preload: nil,
                    engine: nil
                )
            )
            let (http, body) = try await post(
                "/v1/stt/engine", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(
                STTBackendInfo.self, from: body
            )
            XCTAssertEqual(decoded.id, "qwen3_asr_preview")
            XCTAssertTrue(decoded.isActive)

            // GET should now report the new state.
            let (_, getBody) = try await get("/v1/stt/engine", on: server)
            let getDecoded = try JSONDecoder().decode(
                STTBackendsResponse.self, from: getBody
            )
            XCTAssertEqual(getDecoded.activeId, "qwen3_asr_preview")
        }
        // Cleanup global state.
        await resetEngineState()
    }

    @MainActor
    func testPostEngineRejectsUnknownValue() async throws {
        await resetEngineState()
        try await withServer { server in
            let payload = try JSONEncoder().encode(
                STTSetBackendRequest(
                    id: "no-such-backend",
                    preload: nil,
                    engine: nil
                )
            )
            let (http, body) = try await post(
                "/v1/stt/engine", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 400)
            let decoded = try JSONDecoder().decode(
                ErrorResponse.self, from: body
            )
            // Canonical picker code (matches /v1/touchup/model so
            // picker UIs branch on the same key across modules).
            XCTAssertEqual(decoded.code, "invalid_model")
        }
    }

    /// Backward compat: a one-week-old SDK posts the legacy
    /// `{ "engine": "..." }` shape. The route accepts it as a
    /// fallback so an in-flight whisper build keeps working
    /// through one release. New clients post `{ "id": "...",
    /// "preload": true }`.
    @MainActor
    func testPostEngineAcceptsLegacyEngineField() async throws {
        await resetEngineState()
        try await withServer { server in
            let payload = try JSONEncoder().encode(
                STTSetBackendRequest(id: nil, preload: nil, engine: "qwen3_asr_preview")
            )
            let (http, body) = try await post(
                "/v1/stt/engine", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(
                STTBackendInfo.self, from: body
            )
            XCTAssertEqual(decoded.id, "qwen3_asr_preview")
        }
        await resetEngineState()
    }

    /// Conflict guard (#99 review I2): when both `id` and legacy
    /// `engine` are sent with different values, the route must
    /// reject the request rather than silently picking one. The
    /// failure mode without this guard was "the picker just picks
    /// something other than what I sent" — an invisible bug
    /// during a build-script migration.
    @MainActor
    func testPostEngineRejectsConflictingIdAndLegacyEngine() async throws {
        await resetEngineState()
        try await withServer { server in
            let payload = try JSONEncoder().encode(
                STTSetBackendRequest(
                    id: "parakeet",
                    preload: nil,
                    engine: "apple_stt"
                )
            )
            let (http, body) = try await post(
                "/v1/stt/engine", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 400)
            let decoded = try JSONDecoder().decode(
                ErrorResponse.self, from: body
            )
            XCTAssertEqual(decoded.code, "invalid_request")
            // Active backend must NOT have changed — the request
            // was rejected up front.
            let current = await YoozSTTEngine.shared.currentBackend
            XCTAssertEqual(current, .parakeet)
        }
    }

    /// Identical `id` and legacy `engine` is benign — the legacy
    /// fallback path still works when both fields agree (e.g. a
    /// build script that emits both during migration).
    @MainActor
    func testPostEngineAcceptsIdAndLegacyEngineWhenIdentical() async throws {
        await resetEngineState()
        try await withServer { server in
            let payload = try JSONEncoder().encode(
                STTSetBackendRequest(
                    id: "qwen3_asr_preview",
                    preload: nil,
                    engine: "qwen3_asr_preview"
                )
            )
            let (http, body) = try await post(
                "/v1/stt/engine", body: payload, on: server
            )
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(
                STTBackendInfo.self, from: body
            )
            XCTAssertEqual(decoded.id, "qwen3_asr_preview")
        }
        await resetEngineState()
    }

    /// C1 regression: the active row's `loadState` must reflect
    /// `isCurrentBackendLoaded()` rather than always reporting
    /// `.available`. Without this, the whisper picker would show
    /// "Downloads on first use" for the model that's already in
    /// memory and serving requests. Fresh test instance has no
    /// backend loaded, so we assert `.available`; flipping the
    /// flag end-to-end requires real model weights and lives in
    /// the integration suite.
    @MainActor
    func testGetEngineActiveLoadStateReflectsLoadedFlag() async throws {
        await resetEngineState()
        try await withServer { server in
            let (_, body) = try await get("/v1/stt/engine", on: server)
            let decoded = try JSONDecoder().decode(
                STTBackendsResponse.self, from: body
            )
            let active = try XCTUnwrap(
                decoded.backends.first(where: { $0.isActive })
            )
            // Test host hasn't called start() — backend is not
            // loaded, so the wire reports .available, not the
            // dead `.available` from the broken ternary.
            XCTAssertEqual(active.loadState, .available)
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
