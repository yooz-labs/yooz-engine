// InProcessSessionTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import XCTest
@testable import YoozEngineInProcess

/// Proves `POST /v1/session/{begin,end}` routes through `InProcessTransport`
/// (engine issue #222) — the gap where the loopback `APIServer` served the
/// per-recording session-reset boundary (engine issue #114) but the
/// in-process facade did not.
///
/// Per project convention, no mocks: a small stub `AIModule` that also
/// conforms to `SessionResettable` is a real conformer, registered into the
/// same `ModuleRegistry.shared` singleton the transport reads, so the fan-out
/// dispatch under test is the exact production path.
final class InProcessSessionTests: XCTestCase {

    /// Counts `resetForNewSession()` calls. An actor so concurrent fan-out
    /// iterations (across the shared `ModuleRegistry` and other tests running
    /// in the same process) stay race-free.
    actor ResetCounter {
        private(set) var count = 0
        func bump() { count += 1 }
    }

    /// Registered into `ModuleRegistry.shared` for the duration of a test so
    /// the session fan-out has a conformer to reach besides the real engine
    /// modules `EngineInProcessHost.bootstrap()` also registers.
    struct SessionTestConformer: AIModule, SessionResettable {
        static let name = "session-inprocess-test-conformer"
        let counter: ResetCounter
        var isReady: Bool { true }
        func healthCheck() async -> ModuleHealth {
            ModuleHealth(loaded: true, error: nil, detail: ["kind": "session-test"])
        }
        func resetForNewSession() async {
            await counter.bump()
        }
    }

    func testSessionBeginAndEndRouteInProcessAndReachRegisteredConformer() async throws {
        let counter = ResetCounter()
        await ModuleRegistry.shared.register(SessionTestConformer(counter: counter))
        // Awaited teardown (not a fire-and-forget Task) so the shared-singleton
        // registry is guaranteed clean before any later test observes it.
        addTeardownBlock {
            await ModuleRegistry.shared.unregister(SessionTestConformer.name)
        }

        let transport = InProcessTransport()
        try await transport.connect()

        struct BeginResponse: Decodable {
            let sessionId: String
            let ts: String
        }

        let beginData = try await transport.post("/v1/session/begin", body: Data())
        let begin = try JSONDecoder().decode(BeginResponse.self, from: beginData)
        XCTAssertFalse(begin.sessionId.isEmpty, "begin should mint a non-empty session id")
        XCTAssertFalse(begin.ts.isEmpty, "begin should stamp a non-empty timestamp")

        let countAfterBegin = await counter.count
        XCTAssertEqual(
            countAfterBegin, 1,
            "begin should fan out resetForNewSession() to the registered conformer exactly once"
        )

        // Wire parity with the loopback route: end returns an empty body
        // (the in-process analog of HTTP 204 No Content).
        let endData = try await transport.post("/v1/session/end", body: Data())
        XCTAssertTrue(endData.isEmpty, "end should return an empty body, matching HTTP 204")

        let countAfterEnd = await counter.count
        XCTAssertEqual(
            countAfterEnd, 2,
            "end should fan out resetForNewSession() to the registered conformer exactly once more"
        )
    }

    /// Two consecutive `begin` calls mint different session ids — the wire
    /// contract callers correlate their own logs against.
    func testSessionBeginMintsFreshSessionIdEachCall() async throws {
        let transport = InProcessTransport()
        try await transport.connect()

        struct BeginResponse: Decodable {
            let sessionId: String
            let ts: String
        }

        let first = try JSONDecoder().decode(
            BeginResponse.self, from: try await transport.post("/v1/session/begin", body: Data())
        )
        let second = try JSONDecoder().decode(
            BeginResponse.self, from: try await transport.post("/v1/session/begin", body: Data())
        )
        XCTAssertNotEqual(first.sessionId, second.sessionId)
    }
}
