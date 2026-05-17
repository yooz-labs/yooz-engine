// SessionResettableTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Exercises the real `SessionResettable` + `ModuleRegistry.allResettable()`
/// plumbing that backs the `/v1/session/begin` and `/v1/session/end` fan-out
/// in `APIServer` (engine issue #114).
///
/// Per project convention, these tests use real types (no mocks) — a small
/// stub `AIModule` that also conforms to `SessionResettable` is a real
/// conformer, not a mock, exercising the same dispatch path production
/// modules use.
final class SessionResettableTests: XCTestCase {

    /// Counts `resetForNewSession()` calls so tests can assert fan-out
    /// reached this conformer. `Atomic`-style increment via an actor so
    /// concurrent fan-out iterations are race-free.
    actor ResetCounter {
        private(set) var count: Int = 0
        func bump() { count += 1 }
    }

    /// Module that participates in both the registry and the session-reset
    /// boundary. Used to verify `allResettable()` returns it.
    struct ResettableStub: AIModule, SessionResettable {
        static let name = "resettable-stub"
        let counter: ResetCounter
        var isReady: Bool { true }
        func healthCheck() async -> ModuleHealth {
            ModuleHealth(loaded: true, error: nil, detail: ["kind": "stub"])
        }
        func resetForNewSession() async {
            await counter.bump()
        }
    }

    /// Module that is registered but does NOT conform to `SessionResettable`.
    /// Used to verify `allResettable()` filters it out.
    struct NonResettableStub: AIModule {
        static let name = "non-resettable-stub"
        var isReady: Bool { true }
        func healthCheck() async -> ModuleHealth {
            ModuleHealth(loaded: true, error: nil, detail: ["kind": "stub"])
        }
    }

    func testAllResettableIncludesConformers() async {
        let counter = ResetCounter()
        let registry = ModuleRegistry.shared
        await registry.register(ResettableStub(counter: counter))
        await registry.register(NonResettableStub())

        let resettables = await registry.allResettable()
        let names = resettables.compactMap { ($0 as? ResettableStub).map { type(of: $0).name } }
        XCTAssertTrue(
            names.contains("resettable-stub"),
            "allResettable() should include modules conforming to SessionResettable"
        )

        // NonResettableStub must not surface as a SessionResettable.
        let nonResettableSurfaced = resettables.contains { module in
            // Avoid `as?` to NonResettableStub directly because it's not Sendable
            // through `any SessionResettable`; rely on a marker.
            String(describing: type(of: module)) == "NonResettableStub"
        }
        XCTAssertFalse(
            nonResettableSurfaced,
            "allResettable() must filter out modules without SessionResettable"
        )
    }

    func testResetForNewSessionInvokedOnFanOut() async {
        let counter = ResetCounter()
        let registry = ModuleRegistry.shared
        await registry.register(ResettableStub(counter: counter))

        // Simulate what `/v1/session/begin` and `/v1/session/end` do.
        for module in await registry.allResettable() {
            await module.resetForNewSession()
        }
        for module in await registry.allResettable() {
            await module.resetForNewSession()
        }

        let calls = await counter.count
        XCTAssertGreaterThanOrEqual(
            calls, 2,
            "Each fan-out pass should hit the stub once; two passes -> >= 2 calls"
        )
    }
}
