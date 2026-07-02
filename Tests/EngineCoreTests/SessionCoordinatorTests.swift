// SessionCoordinatorTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Exercises `SessionCoordinator` directly — the shared fan-out component
/// `APIServer` (loopback HTTP) and `InProcessTransport` (in-process) both
/// call so `/v1/session/begin` and `/v1/session/end` behave identically
/// regardless of transport (engine issue #222).
///
/// Per project convention, no mocks: a small stub `AIModule` that also
/// conforms to `SessionResettable` is a real conformer, registered into the
/// same `ModuleRegistry.shared` singleton `SessionCoordinator` reads.
final class SessionCoordinatorTests: XCTestCase {

    actor ResetCounter {
        private(set) var count = 0
        func bump() { count += 1 }
    }

    struct CoordinatorTestConformer: AIModule, SessionResettable {
        static let name = "session-coordinator-test-conformer"
        let counter: ResetCounter
        var isReady: Bool { true }
        func healthCheck() async -> ModuleHealth {
            ModuleHealth(loaded: true, error: nil, detail: ["kind": "session-coordinator-test"])
        }
        func resetForNewSession() async {
            await counter.bump()
        }
    }

    func testBeginFansOutAndReturnsFreshSessionId() async {
        let counter = ResetCounter()
        await ModuleRegistry.shared.register(CoordinatorTestConformer(counter: counter))
        defer {
            Task { await ModuleRegistry.shared.unregister(CoordinatorTestConformer.name) }
        }

        let first = await SessionCoordinator.begin()
        let second = await SessionCoordinator.begin()

        XCTAssertFalse(first.sessionId.isEmpty)
        XCTAssertFalse(first.ts.isEmpty)
        XCTAssertNotEqual(first.sessionId, second.sessionId, "each begin() mints a fresh session id")

        let calls = await counter.count
        XCTAssertEqual(calls, 2, "each begin() call fans out to the registered conformer once")
    }

    func testEndFansOutWithNoWirePayload() async {
        let counter = ResetCounter()
        await ModuleRegistry.shared.register(CoordinatorTestConformer(counter: counter))
        defer {
            Task { await ModuleRegistry.shared.unregister(CoordinatorTestConformer.name) }
        }

        let fanoutCount = await SessionCoordinator.end()
        XCTAssertGreaterThanOrEqual(
            fanoutCount, 1,
            "end() should report at least the registered conformer in its fan-out count"
        )

        let calls = await counter.count
        XCTAssertEqual(calls, 1, "end() fans out to the registered conformer once")
    }

    func testBeginTimestampIsISO8601UTC() async {
        let result = await SessionCoordinator.begin()
        // Default ISO8601DateFormatter options produce a UTC `Z`-suffixed
        // `yyyy-MM-ddTHH:mm:ssZ` string; this pins the wire contract
        // `APIServer` and `InProcessTransport` both hand back verbatim.
        XCTAssertTrue(result.ts.hasSuffix("Z"), "ts should be UTC Z-suffixed, got \(result.ts)")
        XCTAssertNotNil(
            ISO8601DateFormatter().date(from: result.ts),
            "ts should parse back as a valid ISO8601 date, got \(result.ts)"
        )
    }
}
