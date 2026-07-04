// InfiniteCheckpointLifecycleTests.swift
// InfiniteModuleTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest
@testable import InfiniteModule

/// Unit coverage (no model) for the engine#266 checkpoint/park/resume/fork
/// state machine: the busy guard, the resume no-op shortcut, the fork
/// session-cap enforcement, the unknown-checkpoint 404, and delete cleaning
/// the on-disk store. All of these are pure engine wiring + a real
/// `InfiniteSessionStore` against a temp directory — no MLX/model needed,
/// unlike the checkpoint round-trip itself (see `InfiniteSessionRuntimeTests`'s
/// `YOOZ_INFINITE_LIVE=1`-gated live equivalence tests for that).
final class InfiniteCheckpointLifecycleTests: XCTestCase {

    private var tempRoot: URL!

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("InfiniteCheckpointLifecycleTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempRoot = dir
        setenv("YOOZ_INFINITE_SESSIONS_DIR", dir.path, 1)
    }

    override func tearDownWithError() throws {
        unsetenv("YOOZ_INFINITE_SESSIONS_DIR")
        if let tempRoot, FileManager.default.fileExists(atPath: tempRoot.path) {
            try FileManager.default.removeItem(at: tempRoot)
        }
        tempRoot = nil
    }

    private func requireSupportedTier() throws {
        guard InfiniteRAMTier.current != .belowMinimum else {
            throw XCTSkip("Infinite requires at least 32 GB unified memory")
        }
    }

    // MARK: - Busy guard

    /// Every session-scoped op throws `sessionBusy` (409) while a session is
    /// `.generating` — modeled directly via the test-only
    /// `setGeneratingForTesting` seam (no real in-flight generate needed;
    /// that would require a live model, see `InfiniteSessionRuntimeTests`).
    func testBusyGuardRejectsEveryOpOnAGeneratingSession() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        let session = try await engine.createSession(request: InfiniteCreateSessionRequest())
        await engine.setGeneratingForTesting(id: session.id)

        await assertThrowsSessionBusy(session.id) {
            _ = try await engine.append(
                sessionID: session.id, request: InfiniteAppendSessionRequest(text: "x")
            )
        }
        await assertThrowsSessionBusy(session.id) {
            _ = try await engine.generate(
                sessionID: session.id, request: InfiniteGenerateSessionRequest(prompt: "x")
            )
        }
        await assertThrowsSessionBusy(session.id) {
            _ = try await engine.checkpoint(
                sessionID: session.id, request: InfiniteCheckpointSessionRequest()
            )
        }
        await assertThrowsSessionBusy(session.id) {
            _ = try await engine.resume(
                sessionID: session.id, request: InfiniteResumeSessionRequest()
            )
        }
        await assertThrowsSessionBusy(session.id) {
            _ = try await engine.fork(
                sessionID: session.id, request: InfiniteForkSessionRequest()
            )
        }
        await assertThrowsSessionBusy(session.id) {
            _ = try await engine.deleteSession(id: session.id)
        }
    }

    private func assertThrowsSessionBusy(
        _ id: String,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () async throws -> Void
    ) async {
        do {
            try await body()
            XCTFail("expected sessionBusy but no error was thrown", file: file, line: line)
        } catch InfiniteError.sessionBusy(let busyId) {
            XCTAssertEqual(busyId, id, file: file, line: line)
        } catch {
            XCTFail("expected InfiniteError.sessionBusy, got \(error)", file: file, line: line)
        }
    }

    /// engine#266 review: `setActiveModel` must refuse a switch away from
    /// the resident model while one of its sessions is `.generating` —
    /// evicting that backend would silently discard the session's live KV
    /// cache the next time something (a different session's append, or a
    /// resume) actually triggers the swap. No real backend is needed here:
    /// the refusal only reads `sessions`/`loadedModel` bookkeeping, wired up
    /// via the test-only `setLoadedModelForTesting` seam (mirrors the real
    /// path a live `loadBackend` call would hit — see
    /// `InfiniteCheckpointResumeLiveTests.testModelSwitchRefusedWhileGenerating`
    /// for the full live proof, including the backend-eviction-parking
    /// half once the busy session clears).
    func testSetActiveModelRefusesWhileOutgoingSessionGenerating() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        let residentSelection = InfiniteModelSelection.gemma4E4B1M
        let otherSelection = InfiniteModelSelection.gemma4_12B1M

        let busySession = try await engine.createSession(
            request: InfiniteCreateSessionRequest(modelId: residentSelection.rawValue, label: "resident")
        )
        await engine.setLoadedModelForTesting(residentSelection)
        await engine.setGeneratingForTesting(id: busySession.id)

        let activeBefore = await engine.status().modelId
        await assertThrowsSessionBusy(busySession.id) {
            _ = try await engine.setActiveModel(otherSelection, preload: false)
        }
        let statusWhileBusy = await engine.status()
        XCTAssertEqual(statusWhileBusy.modelId, activeBefore, "a refused switch must not mutate activeModel")
        XCTAssertNil(statusWhileBusy.lastError, "a busy refusal is backpressure, not a load failure")

        // Clear busy; the identical switch must now succeed.
        await engine.setOpenForTesting(id: busySession.id)
        let active = try await engine.setActiveModel(otherSelection, preload: false)
        XCTAssertEqual(active.id, otherSelection.rawValue)
        XCTAssertEqual(await engine.status().modelId, otherSelection.rawValue)
    }

    // MARK: - Resume no-op

    /// Resuming an already-`.open` session with no explicit `checkpointId`
    /// is a no-op success: it must not touch the backend (no model needed
    /// for this test to pass) and must return the session unchanged.
    func testResumeOfOpenSessionWithNoCheckpointIdIsNoOp() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest(label: "no-op"))
        XCTAssertEqual(created.state, "open")

        let resumed = try await engine.resume(
            sessionID: created.id, request: InfiniteResumeSessionRequest(checkpointId: nil)
        )
        XCTAssertEqual(resumed.id, created.id)
        XCTAssertEqual(resumed.state, "open")
        XCTAssertEqual(resumed.updatedAt, created.updatedAt)
    }

    /// Resuming an unknown session (never created, no on-disk checkpoint
    /// either) is a 404 `session_not_found`, not a `checkpoint_not_found`.
    func testResumeOfUnknownSessionThrowsSessionNotFound() async throws {
        let engine = InfiniteEngine()
        do {
            _ = try await engine.resume(
                sessionID: "does-not-exist", request: InfiniteResumeSessionRequest()
            )
            XCTFail("expected sessionNotFound")
        } catch InfiniteError.sessionNotFound(let id) {
            XCTAssertEqual(id, "does-not-exist")
        }
    }

    // MARK: - Unknown checkpoint id -> 404

    /// An explicit but nonexistent `checkpointId` 404s as
    /// `checkpoint_not_found` before ever touching the backend (no model
    /// needed — the checkpoint id is resolved and its manifest read from
    /// disk before any `loadBackend` call).
    func testResumeUnknownCheckpointIdThrowsCheckpointNotFound() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest())

        do {
            _ = try await engine.resume(
                sessionID: created.id,
                request: InfiniteResumeSessionRequest(checkpointId: "no-such-checkpoint")
            )
            XCTFail("expected checkpointNotFound")
        } catch InfiniteError.checkpointNotFound(let session, let checkpoint) {
            XCTAssertEqual(session, created.id)
            XCTAssertEqual(checkpoint, "no-such-checkpoint")
        }

        // The session must not be left stuck `.generating` after the failure.
        let status = try await engine.resume(
            sessionID: created.id, request: InfiniteResumeSessionRequest(checkpointId: nil)
        )
        XCTAssertEqual(status.state, "open")
    }

    // MARK: - Fork cap enforcement

    /// Forking once the 16-session cap is already full throws
    /// `sessionLimitExceeded` (409) before any checkpoint resolution — no
    /// model needed, since the cap check runs before the (potentially
    /// backend-touching) implicit-checkpoint path.
    func testForkRespectsSessionCap() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        var ids: [String] = []
        for _ in 0..<InfiniteEngine.maxActiveSessions {
            let session = try await engine.createSession(request: InfiniteCreateSessionRequest())
            ids.append(session.id)
        }

        await XCTAssertThrowsInfiniteErrorForkCap(InfiniteEngine.maxActiveSessions) {
            _ = try await engine.fork(
                sessionID: try XCTUnwrap(ids.first), request: InfiniteForkSessionRequest()
            )
        }
    }

    private func XCTAssertThrowsInfiniteErrorForkCap(
        _ limit: Int,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () async throws -> Void
    ) async {
        do {
            try await body()
            XCTFail("expected sessionLimitExceeded", file: file, line: line)
        } catch InfiniteError.sessionLimitExceeded(let actualLimit) {
            XCTAssertEqual(actualLimit, limit, file: file, line: line)
        } catch {
            XCTFail("expected InfiniteError.sessionLimitExceeded, got \(error)", file: file, line: line)
        }
    }

    /// Forking an unknown source session (no tracked record, no on-disk
    /// checkpoint) is `session_not_found`.
    func testForkOfUnknownSessionThrowsSessionNotFound() async throws {
        let engine = InfiniteEngine()
        do {
            _ = try await engine.fork(sessionID: "does-not-exist", request: InfiniteForkSessionRequest())
            XCTFail("expected sessionNotFound")
        } catch InfiniteError.sessionNotFound(let id) {
            XCTAssertEqual(id, "does-not-exist")
        }
    }

    /// A `.parked` source with genuinely no checkpoint (a corrupted/synthetic
    /// state that should be unreachable in practice, since parking always
    /// checkpoints first) 404s as `checkpoint_not_found` rather than
    /// crashing or fabricating a checkpoint.
    func testForkOfParkedSourceWithNoCheckpointThrowsCheckpointNotFound() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest())
        await engine.setParkedForTesting(id: created.id)

        do {
            _ = try await engine.fork(sessionID: created.id, request: InfiniteForkSessionRequest())
            XCTFail("expected checkpointNotFound")
        } catch InfiniteError.checkpointNotFound(let session, _) {
            XCTAssertEqual(session, created.id)
        }
    }

    // MARK: - Delete cleans disk

    /// `deleteSession` removes the on-disk checkpoint tree too, not just the
    /// in-memory record (engine#266) — verified against a real
    /// `InfiniteSessionStore` pointed at the same temp root via
    /// `YOOZ_INFINITE_SESSIONS_DIR`.
    func testDeleteSessionCleansDisk() async throws {
        try requireSupportedTier()
        let engine = InfiniteEngine()
        let created = try await engine.createSession(request: InfiniteCreateSessionRequest())

        // Write a checkpoint directly against the store (same env-overridden
        // root the engine's own `store` property reads) rather than through
        // a real backend — this test only needs a checkpoint directory to
        // exist on disk under this session's id.
        let store = InfiniteSessionStore()
        let tokenIds = [1, 2, 3]
        let tokensData = try InfiniteSessionStore.encodeTokens(tokenIds)
        try store.writeTokens(tokenIds, session: created.id, checkpoint: "c1")
        try store.writeManifest(
            InfiniteSessionManifest(
                model: ModelIdentity(selectionId: "gemma4-e4b-1m", repoId: "r", revision: "v"),
                tokenizerHash: "hash",
                tokenCount: tokenIds.count,
                tokenIdsSHA256: InfiniteSessionStore.sha256Hex(tokensData),
                cacheConfig: SessionKnobs()
            ),
            session: created.id,
            checkpoint: "c1"
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.sessionDirectory(created.id).path))

        let deleted = try await engine.deleteSession(id: created.id)
        XCTAssertTrue(deleted.deleted)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.sessionDirectory(created.id).path))

        do {
            _ = try await engine.session(id: created.id)
            XCTFail("expected sessionNotFound after delete")
        } catch InfiniteError.sessionNotFound {
            // expected
        }
    }
}
