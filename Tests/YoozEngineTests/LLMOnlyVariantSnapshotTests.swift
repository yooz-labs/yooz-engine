// LLMOnlyVariantSnapshotTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Coverage for the `YoozEngineLLM` variant (engine#297) — the first variant
// to drop AppleSTTModule and GrammarModule on top of Lite's existing MLX-STT
// and VAD exclusions. Boots a real `APIServer` and dials it via `URLSession`,
// same harness as `LLMClearCacheRouteTests` / `LLMCatalogueRouteTests`.
//
// ## What this simulates, and what it cannot
//
// This target (`YoozEngineTests`) is hosted by the FULL `YoozEngine` app,
// which links every module — `#if canImport(GrammarModule)` etc. are all
// TRUE in this binary. So these tests cannot exercise the `#else` branches
// that only compile into the real `YoozEngineLLM` binary (where those
// modules aren't linked at all); that half of the contract was verified by
// booting the actual `dist/YoozEngineLLM.app` and curling it directly (see
// the engine#297 PR description).
//
// What IS both real and variant-agnostic: `EngineAppDelegate.registerModules()`
// gates each module's `ModuleRegistry` registration behind the SAME
// `#if canImport(...)`, so a build that omits a module never registers it,
// and `/v1/grammar/check` + `/v1/vad/detect` + `/v1/modules` decide their
// shape from `ModuleRegistry.shared.isBundled(_:)`, not from which modules
// this test binary happens to link. Registering only `TouchUpEngine` (name
// "llm") here reproduces exactly what `.llm`'s `registerModules()` would
// produce, so those registry-driven response shapes are genuine coverage of
// the variant contract, not a simulation of it. `/v1/stt/engine` is the one
// route in this family WITHOUT a registry guard on its module-present branch
// (see the note above that test below) — it cannot be exercised this way and
// is covered by the live binary check instead.
//
// Decoding into the strongly-typed wire structs (rather than probing raw
// JSON) is deliberate: `EngineModules`' fields are non-optional `Bool`, so an
// absent key — the exact "empty-array-vs-absent-key" failure mode engine#297
// flagged as unverified — would throw at decode time, not silently read as
// `false`.
//
// No mocks. Real registry, real server, real HTTP.

import Foundation
import XCTest
@testable import EngineCore
@testable import LLMModule
@testable import YoozEngine

final class LLMOnlyVariantSnapshotTests: XCTestCase {

    @MainActor
    private func withServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
        UniqueEnginePort.assignFreshPort()

        // Reproduce `.llm`'s `registerModules()`: only LLM/TouchUp is
        // registered. Defensively unregister the modules `.llm` never
        // links first, in case an earlier test in this process left one
        // registered without cleaning up — the registry is a process-wide
        // singleton shared across every file in this target.
        for name in ["grammar", "apple_stt", "stt", "vad", "infinite"] {
            await ModuleRegistry.shared.unregister(name)
        }
        await ModuleRegistry.shared.register(TouchUpEngine.shared)

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

    override func tearDown() async throws {
        // Restore ambient state (nothing registered) so later test files in
        // this process see the same pre-test registry they always have.
        await ModuleRegistry.shared.unregister("llm")
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

    // MARK: - GET /v1/health

    @MainActor
    func testHealthShapeIsCompleteWithNoAbsentKeys() async throws {
        // `EngineModules`' legacy Bool fields (`llm`/`touchup`/`grammar`)
        // fall back to the real engine singletons, not `ModuleRegistry`
        // (see `APIServer`'s `/v1/health` handler) — and this test target is
        // hosted by the FULL `YoozEngine` app, which links `GrammarModule`
        // unconditionally. `GrammarEngine.shared.isAvailable` auto-loads its
        // FFI on first touch (doc comment on `ModuleEagerLoader.loadGrammar`),
        // so it reads `true` here regardless of what this file unregisters —
        // that reflects THIS HOST's linkage, not the `.llm` variant's. Only
        // `stt`/`vad`/`infinite` need an explicit `.start()`/`.load()` this
        // test never calls, so their fallbacks are deterministically `false`
        // here and DO reflect the `.llm` variant faithfully. The `grammar`
        // boolean's actual `.llm` behavior (`false`) was verified by booting
        // the real `dist/YoozEngineLLM.app` and curling `/v1/health` directly
        // (see the engine#297 PR description).
        try await withServer { _ in
            let (http, data) = try await get("/v1/health")
            XCTAssertEqual(http.statusCode, 200)

            // Strict decode: every field on `EngineModules` is non-optional,
            // so a response missing any of them (the absent-key regression
            // engine#297 asked to rule out) throws here instead of silently
            // decoding to `false`.
            let decoded = try JSONDecoder().decode(HealthResponse.self, from: data)

            XCTAssertFalse(decoded.modules.stt)
            XCTAssertFalse(decoded.modules.vad)
            XCTAssertFalse(decoded.modules.infinite)

            // The eager loader still seeds every `ModuleID` key regardless
            // of variant (it is not kicked off under XCTest at all, see
            // `EngineConfig.eagerLoadOnLaunch`), so `detail` carries all six
            // — this is the field the empty-array-vs-absent-key concern was
            // actually about, and it decodes and matches exactly.
            XCTAssertEqual(Set(decoded.modules.detail.keys), Set(ModuleID.allCases.map(\.rawValue)))
        }
    }

    // MARK: - GET /v1/modules

    @MainActor
    func testModulesListsExactlyLLM() async throws {
        try await withServer { _ in
            let (http, data) = try await get("/v1/modules")
            XCTAssertEqual(http.statusCode, 200)

            let decoded = try JSONDecoder().decode(ModulesResponse.self, from: data)
            XCTAssertEqual(
                decoded.modules.map(\.name), ["llm"],
                "only the registered module should appear — no placeholder rows for absent ones"
            )
        }
    }

    // MARK: - GET /v1/state

    @MainActor
    func testStateReturnsTouchUpSnapshotRegardlessOfRegistry() async throws {
        // `/v1/state` (EngineStateEndpoints) is LLM-module-owned and calls
        // `TouchUpEngine` directly rather than consulting `ModuleRegistry` —
        // it is not gated by the registry the way `/v1/health` is. Pinning
        // that it still returns the canonical one-row shape confirms `.llm`
        // gets a working `/v1/state` for the one module it actually ships,
        // not an empty `modules: []`.
        try await withServer { _ in
            let (http, data) = try await get("/v1/state")
            XCTAssertEqual(http.statusCode, 200)

            let decoded = try JSONDecoder().decode(EngineStateSnapshot.self, from: data)
            XCTAssertEqual(decoded.modules.map(\.module), ["touchup"])
            XCTAssertFalse(decoded.modules[0].models.isEmpty)
            XCTAssertFalse(decoded.modules[0].activeId.isEmpty)
        }
    }

    // MARK: - GET /v1/models (disk-hygiene inventory)

    @MainActor
    func testModelsInventoryDoesNotErrorWithoutGrammarOrSTT() async throws {
        // The cross-module inventory sweeps disk directly (see
        // `ModelManagementEndpoints`) and is not gated by which modules are
        // linked — it must return 200 with a decodable body even though
        // `activeSTTRepoDirName` has nothing to report and no Grammar-owned
        // rows exist.
        try await withServer { _ in
            let (http, data) = try await get("/v1/models")
            XCTAssertEqual(http.statusCode, 200)
            let decoded = try JSONDecoder().decode(ManagedModelsResponse.self, from: data)
            // At minimum, the LLM catalogue's cache descriptors are present.
            XCTAssertTrue(decoded.models.contains { $0.module == "llm" })
        }
    }

    // MARK: - Absent-module routes still 501, not a bare 404

    @MainActor
    func testGrammarCheckReportsModuleNotBundled() async throws {
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/grammar/check", body: Data(#"{"text":"a"}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 501)
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(json["code"] as? String, "module_not_bundled")
            XCTAssertEqual(json["module"] as? String, "grammar")
        }
    }

    @MainActor
    func testVADDetectReportsModuleNotBundled() async throws {
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/vad/detect", body: Data(#"{"samples":[]}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 501)
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(json["code"] as? String, "module_not_bundled")
            XCTAssertEqual(json["module"] as? String, "vad")
        }
    }

    // `/v1/stt/engine` is deliberately NOT covered here: unlike
    // grammar/vad, its `#if canImport(STTModule)` branch has no
    // `ModuleRegistry.isBundled("stt")` guard at all (see `APIServer`) — it
    // is gated purely by compilation, so in this GrammarModule/STTModule-
    // linked host it always serves 200 regardless of what this file
    // unregisters, no matter the registry state. That branch only compiles
    // out of the real `YoozEngineLLM` binary, where it correctly falls to
    // the `#else moduleNotBundled("stt")` arm — verified live (see the
    // engine#297 PR description: `501 {"code":"module_not_bundled",
    // "module":"stt",...}`).

    // MARK: - LLM generate routing (catalogue alias, engine#297)
    //
    // `/v1/llm/generate` is never gated on the module set this file
    // manipulates — LLMModule ships in every variant, so there is nothing
    // variant-specific left to prove about it in this suite once the
    // registry-driven routes above are covered. Deliberately does NOT POST
    // to it here: doing so would trigger a real generation (or a multi-GB
    // download on an uncached machine), the exact cost
    // `LLMCatalogueRouteTests` already opts out of for the same reason.
    // `YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx` resolving via
    // `LLMModelType(rawValue:)` is unit-tested in `LLMModuleTests`, and its
    // presence in the picker catalogue is pinned by
    // `LLMCatalogueRouteTests.testGetModelsEnumeratesFullCatalogue`. The
    // end-to-end proof that the route actually reaches generation — not a
    // `module_not_bundled` 501 — on a real `YoozEngineLLM` binary is a
    // documented manual run (see the engine#297 PR description): a cached
    // engine returned `200 {"model":"yooz-instruct-4b","text":"ping",...}`.
}
