// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import Hummingbird
import HummingbirdTesting
import XCTest

@testable import YoozEngine

/// Phase 5 — `/v1/stt/engine` GET + POST and qwen3-aware routing on
/// `/v1/stt/load` and `/v1/stt/batch`. Uses Hummingbird's `.router`
/// test framework which sends requests directly into the route
/// dispatcher without spinning up a real socket.
final class Qwen3ASREngineRouteTests: XCTestCase {

    /// Stand up a fresh `APIServer` for each test and yank the test
    /// router off of it. `APIServer` is `@MainActor`-bound; we thread
    /// the construction through `MainActor.run`.
    @MainActor
    private func makeApp() -> Application<RouterResponder<BasicRequestContext>> {
        let server = APIServer()
        let router = server.makeTestRouter()
        return Application(router: router)
    }

    /// Reset global engine state between tests so backend / language
    /// flips don't leak across cases. `setBackend(.parakeet)` is the
    /// post-condition.
    @MainActor
    private func resetEngineState() async {
        await YoozSTTEngine.shared.setBackend(.parakeet)
    }

    // MARK: - GET /v1/stt/engine

    @MainActor
    func testGetEngineListsAllBackends() async throws {
        await resetEngineState()
        let app = makeApp()
        try await app.test(.router) { client in
            try await client.execute(uri: "/v1/stt/engine", method: .get) { res in
                XCTAssertEqual(res.status, .ok)
                let body = try JSONDecoder().decode(
                    STTEngineGetResponse.self,
                    from: Data(buffer: res.body)
                )
                XCTAssertEqual(body.current, "parakeet")
                let ids = Set(body.available.map(\.id))
                XCTAssertEqual(
                    ids,
                    Set(STTBackendID.allCases.map(\.rawValue))
                )
                let qwen3 = body.available.first {
                    $0.id == "qwen3_asr_preview"
                }
                XCTAssertNotNil(qwen3)
                XCTAssertTrue(qwen3?.supportsBatch ?? false)
                XCTAssertFalse(qwen3?.supportsStreaming ?? true)
            }
        }
    }

    // MARK: - POST /v1/stt/engine

    @MainActor
    func testPostEngineSwitchesToQwen3() async throws {
        await resetEngineState()
        let app = makeApp()
        try await app.test(.router) { client in
            let body = try JSONEncoder().encode(
                STTEnginePostRequest(engine: "qwen3_asr_preview")
            )
            try await client.execute(
                uri: "/v1/stt/engine",
                method: .post,
                body: ByteBuffer(data: body)
            ) { res in
                XCTAssertEqual(res.status, .ok)
                let decoded = try JSONDecoder().decode(
                    STTEnginePostResponse.self,
                    from: Data(buffer: res.body)
                )
                XCTAssertEqual(decoded.current, "qwen3_asr_preview")
            }

            // Confirm the GET endpoint reports the new state.
            try await client.execute(uri: "/v1/stt/engine", method: .get) { res in
                let decoded = try JSONDecoder().decode(
                    STTEngineGetResponse.self,
                    from: Data(buffer: res.body)
                )
                XCTAssertEqual(decoded.current, "qwen3_asr_preview")
            }
        }

        // Cleanup.
        await resetEngineState()
    }

    @MainActor
    func testPostEngineRejectsUnknownValue() async throws {
        await resetEngineState()
        let app = makeApp()
        try await app.test(.router) { client in
            let body = try JSONEncoder().encode(
                STTEnginePostRequest(engine: "no-such-backend")
            )
            try await client.execute(
                uri: "/v1/stt/engine",
                method: .post,
                body: ByteBuffer(data: body)
            ) { res in
                XCTAssertEqual(res.status, .badRequest)
                let decoded = try JSONDecoder().decode(
                    ErrorResponse.self, from: Data(buffer: res.body)
                )
                XCTAssertEqual(decoded.code, "invalid_engine")
            }
        }
    }

    // MARK: - POST /v1/stt/load

    /// With qwen3 selected and `allow_fetch: false`, the load route
    /// must surface `model_not_found` rather than silently falling
    /// back. This is the CI-safe path: no network calls, no model on
    /// disk.
    @MainActor
    func testLoadRefusesFetchWhenDisabled() async throws {
        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer { Task { @MainActor in await self.resetEngineState() } }

        // Point the fetcher at a guaranteed-empty directory so we
        // can't accidentally pick up real artifacts on disk.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-load-test-\(UUID().uuidString)")
        setenv("YOOZ_QWEN3_ASR_DIR", tmp.path, 1)
        defer { unsetenv("YOOZ_QWEN3_ASR_DIR") }

        let app = makeApp()
        try await app.test(.router) { client in
            struct Body: Encodable {
                let language: String
                let allowFetch: Bool
                enum CodingKeys: String, CodingKey {
                    case language
                    case allowFetch = "allow_fetch"
                }
            }
            // The server-side decoder uses the field name `allowFetch`
            // (CodingKeys mapping is implicit), so encode with that
            // key directly.
            struct DirectBody: Encodable {
                let language: String
                let allowFetch: Bool
            }
            let payload = try JSONEncoder().encode(
                DirectBody(language: "en", allowFetch: false)
            )
            try await client.execute(
                uri: "/v1/stt/load",
                method: .post,
                body: ByteBuffer(data: payload)
            ) { res in
                XCTAssertEqual(res.status, .notFound)
                let decoded = try JSONDecoder().decode(
                    ErrorResponse.self, from: Data(buffer: res.body)
                )
                XCTAssertEqual(decoded.code, "model_not_found")
            }
        }
    }

    // MARK: - GET /v1/stt/languages (sanity that other routes still work)

    @MainActor
    func testLanguagesRouteStillWorks() async throws {
        await resetEngineState()
        let app = makeApp()
        try await app.test(.router) { client in
            try await client.execute(
                uri: "/v1/stt/languages", method: .get
            ) { res in
                XCTAssertEqual(res.status, .ok)
            }
        }
    }
}
