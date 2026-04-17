// ModulesEndpointTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Integration-level coverage for the /v1/modules endpoint. We cannot easily
// boot the live HTTP server from inside XCTest (port 19920 is hard-coded in
// EngineConfig and collides with a running dev instance, and Hummingbird's
// in-process test client is not wired into this project's dependency set).
// Instead, we exercise the exact construction path the route uses:
// `ModuleRegistry.shared.all()` -> `ModulesResponse.build(...)` with real
// module singletons from the app. This is the same code path that runs
// inside the route closure in `APIServer.swift`; the only thing not
// exercised is Hummingbird's routing + encoder bridge (both already
// covered by Hummingbird's own tests).
//
// No mocks. Real `GrammarEngine.shared` / `VADEngine.shared` instances.

import XCTest
import EngineCore
import GrammarModule
#if canImport(VADModule)
import VADModule
#endif

final class ModulesEndpointTests: XCTestCase {

    /// A private ephemeral registry would be ideal, but `ModuleRegistry` is
    /// deliberately a singleton actor (one-and-only registry per process).
    /// We register the modules we care about up-front; other tests in this
    /// target must tolerate a populated registry, and the app's own launch
    /// registers the same instances anyway.
    override func setUp() async throws {
        try await super.setUp()
        await ModuleRegistry.shared.register(GrammarEngine.shared)
        #if canImport(VADModule)
        await ModuleRegistry.shared.register(VADEngine.shared)
        #endif
    }

    func testModulesResponseContainsRegisteredModules() async {
        let registered = await ModuleRegistry.shared.all()
        let response = await ModulesResponse.build(
            from: registered,
            engineVersion: EngineConfig.version,
            buildVariant: BuildVariant.current.rawValue
        )

        XCTAssertEqual(response.engineVersion, EngineConfig.version)
        XCTAssertEqual(response.buildVariant, BuildVariant.current.rawValue)
        XCTAssertFalse(response.modules.isEmpty, "registry should return at least one module")

        let names = response.modules.map(\.name)
        XCTAssertTrue(names.contains("grammar"), "grammar module missing: \(names)")

        // Sorted-by-name invariant from ModuleRegistry.all()
        XCTAssertEqual(names, names.sorted(), "modules must be sorted by name")

        // Every manifest uses the engine version (unified scheme).
        for manifest in response.modules {
            XCTAssertEqual(manifest.version, EngineConfig.version,
                           "module \(manifest.name) should report engine version")
        }
    }

    func testGrammarManifestReportsRealHealth() async {
        let registered = await ModuleRegistry.shared.all()
        let response = await ModulesResponse.build(
            from: registered,
            engineVersion: EngineConfig.version,
            buildVariant: BuildVariant.current.rawValue
        )

        guard let grammar = response.modules.first(where: { $0.name == "grammar" }) else {
            XCTFail("grammar manifest missing"); return
        }
        XCTAssertTrue(grammar.loaded,
                      "GrammarEngine.shared should load Rust rules on init")
        XCTAssertNil(grammar.error)
        XCTAssertNotNil(grammar.detail["rules_total"],
                        "expected rules_total key in detail")
        XCTAssertNotNil(grammar.detail["library_version"],
                        "expected library_version key in detail")
    }

    func testResponseSerializesDeterministically() async throws {
        let registered = await ModuleRegistry.shared.all()
        let response = await ModulesResponse.build(
            from: registered,
            engineVersion: EngineConfig.version,
            buildVariant: BuildVariant.current.rawValue
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let first = try encoder.encode(response)
        let second = try encoder.encode(response)
        XCTAssertEqual(first, second,
                       "encoded response must be byte-identical across encodes")

        // Round-trip: encoded -> decoded should equal the original.
        let decoded = try JSONDecoder().decode(ModulesResponse.self, from: first)
        XCTAssertEqual(decoded, response)
    }
}
