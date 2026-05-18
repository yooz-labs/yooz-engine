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
///
/// Wire shape is the canonical `EngineCore.ModulesResponse`
/// (`engineVersion` / `buildVariant` / `modules: [ModuleManifest]`),
/// which yooz-whisper's About panel already consumes
/// (`AboutEngineInfoModel.transform`). See yooz-engine#133.
final class ModulesEndpointTests: XCTestCase {

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
                ModulesResponse.self, from: body
            )

            // buildVariant must decode as a known BuildVariant. We can't
            // pin to "full" because tests may run under any variant
            // flag, but the wire format is fixed.
            XCTAssertNotNil(
                BuildVariant(rawValue: decoded.buildVariant),
                "buildVariant `\(decoded.buildVariant)` should decode as a BuildVariant"
            )

            // engineVersion matches the engine config under the unified-
            // versioning scheme (each manifest's `version` is identical).
            XCTAssertEqual(decoded.engineVersion, EngineConfig.version)
            for manifest in decoded.modules {
                XCTAssertEqual(
                    manifest.version, EngineConfig.version,
                    "module `\(manifest.name)` version must match engineVersion"
                )
            }

            // The registry populates modules variant-aware; we don't
            // assert a specific set here (variant gating is covered by
            // ModuleEagerLoaderTests + ModuleNotBundledTests). Just
            // verify the wire shape: names are non-empty and sorted.
            //
            // Sort comes from `ModuleRegistry.all()` (which sorts by
            // `type(of:).name`); the encoder's `.sortedKeys` is a
            // separate concern that sorts JSON object keys, not array
            // elements. Both invariants must hold for the response to
            // be byte-deterministic, but this assertion only covers
            // the array-order half.
            let names = decoded.modules.map(\.name)
            XCTAssertEqual(
                names, names.sorted(),
                "ModuleRegistry.all() must return modules sorted by name"
            )
            for name in names {
                XCTAssertFalse(name.isEmpty, "module name should not be empty")
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

            // TTS isn't shipped on any variant today (Phase 7 work) but
            // `ModuleEagerLoader` unconditionally seeds it as
            // `.unavailable` so thin clients render a neutral tag rather
            // than a red dot. Asserting the exact state here (not just
            // presence) is the contract the whisper About panel reads
            // through /v1/health; /v1/modules drops TTS entirely since
            // it isn't registered with `ModuleRegistry`. See
            // yooz-engine#133.
            XCTAssertEqual(
                decoded.modules.detail["tts"]?.state, .unavailable,
                "TTS must report `.unavailable` under /v1/health until Phase 7 ships"
            )

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
    func testTTSNotSurfacedUntilPhase7() async throws {
        // TTS isn't shipped on any variant today (Phase 7 work). It is
        // absent from `ModuleRegistry` (`EngineAppDelegate.registerModules`
        // skips it) but `ModuleEagerLoader` still seeds it as
        // `.unavailable` so `/v1/health.modules.detail` carries a row.
        // The two endpoints therefore disagree on TTS by design:
        // `/v1/modules` drops it (this test); `/v1/health.modules.detail`
        // shows it as `.unavailable` (asserted in
        // `testHealthDetailFieldShape`). Thin clients render absent
        // modules as "not available" rather than a red dot.
        try await withServer { _ in
            let (_, body) = try await get("/v1/modules")
            let decoded = try JSONDecoder().decode(
                ModulesResponse.self, from: body
            )
            XCTAssertFalse(
                decoded.modules.contains(where: { $0.name == "tts" }),
                "TTS must not appear in /v1/modules until Phase 7 ships"
            )
        }
    }
}
