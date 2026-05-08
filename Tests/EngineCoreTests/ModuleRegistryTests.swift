// ModuleRegistryTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

final class ModuleRegistryTests: XCTestCase {

    /// A minimal module stub so the registry has something to register. The
    /// registry contract is exercised here; module-specific behavior lives
    /// in each module's own test target.
    struct StubModule: AIModule {
        static let name = "stub"
        let isReady: Bool
        func healthCheck() async -> ModuleHealth {
            ModuleHealth(loaded: isReady, error: nil, detail: ["kind": "stub"])
        }
    }

    func testRegisterAndLookup() async {
        let registry = ModuleRegistry.shared
        await registry.register(StubModule(isReady: true))

        let isBundled = await registry.isBundled("stub")
        XCTAssertTrue(isBundled)

        let module = await registry.module("stub")
        XCTAssertNotNil(module)
        XCTAssertEqual(type(of: module!).name, "stub")
    }

    func testUnknownModuleNotBundled() async {
        let isBundled = await ModuleRegistry.shared.isBundled("does-not-exist-xyz")
        XCTAssertFalse(isBundled)
    }

    func testReregisterReplacesInstance() async {
        let registry = ModuleRegistry.shared
        await registry.register(StubModule(isReady: false))
        await registry.register(StubModule(isReady: true))

        guard let module = await registry.module("stub") else {
            XCTFail("stub module missing after re-register")
            return
        }
        let ready = await module.isReady
        XCTAssertTrue(ready, "second register should replace first")
    }

    func testAllSortedByName() async {
        struct ModuleA: AIModule {
            static let name = "a-test"
            var isReady: Bool { true }
            func healthCheck() async -> ModuleHealth { ModuleHealth(loaded: true) }
        }
        struct ModuleZ: AIModule {
            static let name = "z-test"
            var isReady: Bool { true }
            func healthCheck() async -> ModuleHealth { ModuleHealth(loaded: true) }
        }

        await ModuleRegistry.shared.register(ModuleZ())
        await ModuleRegistry.shared.register(ModuleA())

        let all = await ModuleRegistry.shared.all()
        let names = all.map { type(of: $0).name }
        let filtered = names.filter { $0 == "a-test" || $0 == "z-test" }
        XCTAssertEqual(filtered, ["a-test", "z-test"])
    }
}
