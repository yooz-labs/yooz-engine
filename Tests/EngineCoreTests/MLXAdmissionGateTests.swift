// MLXAdmissionGateTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Pure-logic coverage for `MLXAdmissionGate` using fake workloads (plain
/// `beginInteractive()` / `endInteractive()` / `checkpoint()` calls — no
/// model weights, no Metal). Every test builds an isolated gate instance
/// with a small `agingInterval`/`pollInterval` so the starvation-guard path
/// exercises in milliseconds under `swift test`, matching the pattern in
/// `MLXResidencyTests`.
final class MLXAdmissionGateTests: XCTestCase {
    /// Short-aging gate for tests that want the starvation guard to fire
    /// quickly. `pollInterval` is small relative to `agingInterval` so the
    /// aging deadline is observed closely.
    private func fastAgingGate(
        agingMs: Int = 40, pollMs: Int = 5
    ) -> MLXAdmissionGate {
        MLXAdmissionGate(
            agingInterval: .milliseconds(agingMs),
            pollInterval: .milliseconds(pollMs)
        )
    }

    // MARK: - Admit (no contention)

    func testBackgroundAdmittedImmediatelyWithNoInteractiveLoad() async throws {
        let gate = fastAgingGate()
        try await gate.checkpoint()
        let state = await gate.queueState
        XCTAssertEqual(state.backgroundAdmittedImmediately, 1)
        XCTAssertEqual(state.backgroundWaiting, 0)
        XCTAssertEqual(state.backgroundAdmittedAfterWait, 0)
        XCTAssertEqual(state.backgroundForcedByAging, 0)
    }

    func testInteractiveNeverQueues() async {
        // beginInteractive/endInteractive never suspend regardless of
        // concurrent background activity — there is no gate-side wait path
        // for interactive submissions at all, only a counter.
        let gate = fastAgingGate()
        await gate.beginInteractive()
        await gate.beginInteractive()
        var state = await gate.queueState
        XCTAssertEqual(state.interactiveActive, 2)

        await gate.endInteractive()
        state = await gate.queueState
        XCTAssertEqual(state.interactiveActive, 1)

        await gate.endInteractive()
        state = await gate.queueState
        XCTAssertEqual(state.interactiveActive, 0)
    }

    func testEndInteractiveIgnoresOverBalancedCall() async {
        let gate = fastAgingGate()
        // No matching beginInteractive() — must not underflow.
        await gate.endInteractive()
        let state = await gate.queueState
        XCTAssertEqual(state.interactiveActive, 0)
    }

    // MARK: - Queue

    func testBackgroundQueuesWhileInteractiveActiveAndAdmitsWhenCleared() async throws {
        // Long aging interval so this test exercises the "cleared before
        // aging" path, not the starvation guard.
        let gate = MLXAdmissionGate(agingInterval: .seconds(5), pollInterval: .milliseconds(5))
        await gate.beginInteractive()

        let checkpointTask = Task { try await gate.checkpoint() }

        // Poll until the checkpoint call has registered itself as queued.
        // Bounded loop (not a fixed sleep) so this isn't flaky under CI
        // scheduling jitter.
        var observedQueued = false
        for _ in 0..<200 {
            let state = await gate.queueState
            if state.backgroundWaiting == 1 {
                observedQueued = true
                break
            }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertTrue(observedQueued, "checkpoint() should register as queued while interactive is active")

        await gate.endInteractive()
        try await checkpointTask.value

        let finalState = await gate.queueState
        XCTAssertEqual(finalState.backgroundWaiting, 0)
        XCTAssertEqual(finalState.backgroundAdmittedAfterWait, 1)
        XCTAssertEqual(finalState.backgroundForcedByAging, 0)
    }

    // MARK: - Starvation guard (aging)

    func testStarvationGuardForceAdmitsAfterAgingWhileInteractiveStaysActive() async throws {
        let gate = fastAgingGate(agingMs: 30, pollMs: 5)
        await gate.beginInteractive()
        // Deliberately never call endInteractive() — the starvation guard
        // must force admission anyway. Bound the wait so a regression that
        // removes the guard fails the test instead of hanging it forever.
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        try await withTimeout(until: deadline) {
            try await gate.checkpoint()
        }

        let state = await gate.queueState
        XCTAssertEqual(state.backgroundForcedByAging, 1)
        XCTAssertEqual(state.backgroundAdmittedAfterWait, 0)
        // Interactive load is still reported active — the guard admits the
        // background call without clearing the interactive signal.
        XCTAssertEqual(state.interactiveActive, 1)
    }

    func testMultipleConcurrentBackgroundWaitersEachAgeIndependently() async throws {
        let gate = fastAgingGate(agingMs: 30, pollMs: 5)
        await gate.beginInteractive()

        let tasks = (0..<3).map { _ in Task { try await gate.checkpoint() } }

        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        try await withTimeout(until: deadline) {
            for task in tasks {
                try await task.value
            }
        }

        let state = await gate.queueState
        XCTAssertEqual(state.backgroundForcedByAging, 3)
        XCTAssertEqual(state.backgroundWaiting, 0)
    }

    func testSharedAgingBudgetForceAdmitsImmediatelyOnceSpent() async throws {
        // Models the per-generation shared budget (checkpoint(workStartedAt:)):
        // once a generation's aging budget is spent, every later checkpoint
        // of that generation force-admits without re-paying the ceiling —
        // total deferral per generation is bounded by one agingInterval,
        // not agingInterval x chunkCount.
        let gate = MLXAdmissionGate(
            agingInterval: .milliseconds(30), pollInterval: .milliseconds(5)
        )
        await gate.beginInteractive()

        // A work-start far enough in the past that the budget is spent.
        let start = ContinuousClock.now.advanced(by: .milliseconds(-100))
        let clock = ContinuousClock.now
        try await gate.checkpoint(workStartedAt: start)
        try await gate.checkpoint(workStartedAt: start)
        try await gate.checkpoint(workStartedAt: start)
        let elapsed = ContinuousClock.now - clock

        let state = await gate.queueState
        XCTAssertEqual(
            state.backgroundForcedByAging, 3,
            "spent-budget checkpoints must force-admit, not queue afresh"
        )
        XCTAssertLessThan(
            elapsed, .milliseconds(30 * 3),
            "spent-budget checkpoints must not each re-pay the aging ceiling"
        )
    }

    func testBackgroundStaysQueuedUntilLastInteractiveSessionEnds() async throws {
        // Overlapping interactive sessions (e.g. two concurrent streaming
        // STT connections): a queued background checkpoint is admitted only
        // when the LAST interactive session ends, not the first.
        let gate = MLXAdmissionGate(agingInterval: .seconds(5), pollInterval: .milliseconds(5))
        await gate.beginInteractive()
        await gate.beginInteractive()

        let checkpointTask = Task { try await gate.checkpoint() }
        var observedQueued = false
        for _ in 0..<200 {
            let state = await gate.queueState
            if state.backgroundWaiting == 1 {
                observedQueued = true
                break
            }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertTrue(observedQueued)

        // First session ends — background must STAY queued.
        await gate.endInteractive()
        try await Task.sleep(for: .milliseconds(30))
        var state = await gate.queueState
        XCTAssertEqual(
            state.backgroundWaiting, 1,
            "background must stay queued while any interactive session remains"
        )

        // Last session ends — background admits.
        await gate.endInteractive()
        try await checkpointTask.value
        state = await gate.queueState
        XCTAssertEqual(state.backgroundAdmittedAfterWait, 1)
        XCTAssertEqual(state.backgroundForcedByAging, 0)
    }

    // MARK: - Cancellation

    func testCheckpointPropagatesCancellationWhileQueued() async throws {
        // Aging interval far longer than the test needs, so cancellation —
        // not aging — is what unblocks the queued call.
        let gate = MLXAdmissionGate(agingInterval: .seconds(30), pollInterval: .milliseconds(5))
        await gate.beginInteractive()

        let task = Task { try await gate.checkpoint() }

        var observedQueued = false
        for _ in 0..<200 {
            let state = await gate.queueState
            if state.backgroundWaiting == 1 {
                observedQueued = true
                break
            }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertTrue(observedQueued)

        task.cancel()
        do {
            try await task.value
            XCTFail("expected CancellationError")
        } catch is CancellationError {
            // expected
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }
    }

    // MARK: - Chunk-level yielding shape

    func testRepeatedCheckpointCallsMirrorPerChunkYielding() async throws {
        // Simulates the LLM generation loop's per-chunk gate check: many
        // `checkpoint()` calls in a row with no interactive load should all
        // take the immediate-admit fast path.
        let gate = fastAgingGate()
        for _ in 0..<25 {
            try await gate.checkpoint()
        }
        let state = await gate.queueState
        XCTAssertEqual(state.backgroundAdmittedImmediately, 25)
    }

    func testInteractiveBeginningMidGenerationCausesSubsequentCheckpointsToQueue() async throws {
        // Models the "pause between chunks, not only at submission" case:
        // the first few checkpoints (no interactive load) are immediate;
        // once beginInteractive() fires mid-loop, later checkpoints queue.
        let gate = MLXAdmissionGate(agingInterval: .seconds(5), pollInterval: .milliseconds(5))

        try await gate.checkpoint()
        try await gate.checkpoint()

        await gate.beginInteractive()

        let queuedCheckpoint = Task { try await gate.checkpoint() }
        var observedQueued = false
        for _ in 0..<200 {
            let state = await gate.queueState
            if state.backgroundWaiting == 1 {
                observedQueued = true
                break
            }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertTrue(observedQueued, "checkpoint mid-generation should queue once interactive begins")

        await gate.endInteractive()
        try await queuedCheckpoint.value

        let state = await gate.queueState
        XCTAssertEqual(state.backgroundAdmittedImmediately, 2)
        XCTAssertEqual(state.backgroundAdmittedAfterWait, 1)
    }
}

/// Runs `operation`, failing the test (via a thrown timeout error) if it
/// has not completed by `deadline`. Used to bound the starvation-guard
/// tests so a regression that removes the aging force-admit hangs the test
/// with a clear failure instead of the suite timing out opaquely.
private func withTimeout(
    until deadline: ContinuousClock.Instant,
    operation: @escaping () async throws -> Void
) async throws {
    try await withThrowingTaskGroup(of: Void.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            let remaining = deadline - ContinuousClock.now
            if remaining > .zero {
                try await Task.sleep(for: remaining)
            }
            throw TimeoutError()
        }
        try await group.next()
        group.cancelAll()
    }
}

private struct TimeoutError: Error, CustomStringConvertible {
    var description: String { "operation did not complete before the test deadline" }
}
