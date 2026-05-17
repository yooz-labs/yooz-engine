// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import YoozEngine

/// Phase 8 — `/v1/modules` endpoint + the `detail` field on
/// `/v1/health`. Boots a real `APIServer` (binds to
/// `localhost:19920`) and dials it with `URLSession`, mirroring the
/// pattern in `Qwen3ASREngineRouteTests`. Eager-load runs only when
/// `EngineAppDelegate` calls `kickoff` — these tests exercise the
/// route handlers directly so the loader's state is whatever a fresh
/// process inherited (the `shared` actor from prior tests in the same
/// xctest invocation may have run kickoff). The assertions therefore
/// only check wire shape, not specific module states.
final class ModulesEndpointTests: XCTestCase {

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

    private func get(
        _ path: String
    ) async throws -> (HTTPURLResponse, Data) {
        let url = URL(
            string: "http://\(EngineConfig.host):\(EngineConfig.port)\(path)"
        )!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "test", code: 0)
        }
        return (http, data)
    }

    @MainActor
    func testModulesEndpointShape() async throws {
        try await withServer { _ in
            let (http, body) = try await get("/v1/modules")
            XCTAssertEqual(http.statusCode, 200)

            let decoded = try JSONDecoder().decode(
                ModulesResponseV1.self, from: body
            )

            // Variant must be one of the known rawValues. We can't
            // pin to "full" because tests may run under any variant
            // flag, but the wire format is fixed.
            XCTAssertNotNil(
                BuildVariant(rawValue: decoded.variant),
                "variant `\(decoded.variant)` should decode as a BuildVariant"
            )

            // Version matches the engine config.
            XCTAssertEqual(decoded.version, EngineConfig.version)

            // Every ModuleID should appear in the map (the loader
            // seeds them all in `init`). TTS is always present, even
            // though it's `unavailable` until Phase 7.
            for id in ModuleID.allCases {
                XCTAssertNotNil(
                    decoded.modules[id.rawValue],
                    "module `\(id.rawValue)` should appear in /v1/modules response"
                )
            }
        }
    }

    @MainActor
    func testHealthDetailFieldShape() async throws {
        try await withServer { _ in
            let (http, body) = try await get("/v1/health")
            XCTAssertEqual(http.statusCode, 200)

            let decoded = try JSONDecoder().decode(
                HealthResponse.self, from: body
            )
            XCTAssertEqual(decoded.status, "ok")

            // `detail` should be present and contain every module.
            for id in ModuleID.allCases {
                XCTAssertNotNil(
                    decoded.modules.detail[id.rawValue],
                    "module `\(id.rawValue)` should appear in /v1/health.modules.detail"
                )
            }

            // The legacy bool fields should agree with the detail
            // map: bool=true iff state==ready.
            let detail = decoded.modules.detail
            let pairs: [(Bool, ModuleID)] = [
                (decoded.modules.stt, .stt),
                (decoded.modules.llm, .llm),
                (decoded.modules.touchup, .touchup),
                (decoded.modules.grammar, .grammar),
                (decoded.modules.vad, .vad),
                (decoded.modules.tts, .tts)
            ]
            for (legacyBool, id) in pairs {
                if detail[id.rawValue]?.state == .ready {
                    XCTAssertTrue(
                        legacyBool,
                        "\(id.rawValue) is `ready` in detail but bool field is false"
                    )
                }
                // Note: bool may be `true` while detail is non-`ready`
                // when a thin client called the per-module load route
                // and bypassed the eager loader (the OR fallback in
                // the handler). That's intentional; we don't assert
                // the converse direction.
            }
        }
    }

    @MainActor
    func testTTSAlwaysUnavailable() async throws {
        // TTS isn't shipped on any variant today (Phase 7 work). The
        // wire shape MUST consistently report `unavailable` so thin
        // clients don't render TTS as red on any variant.
        try await withServer { _ in
            let (_, body) = try await get("/v1/modules")
            let decoded = try JSONDecoder().decode(
                ModulesResponseV1.self, from: body
            )
            XCTAssertEqual(
                decoded.modules["tts"]?.state, .unavailable,
                "TTS should always be `unavailable` until Phase 7 ships"
            )
        }
    }
}
