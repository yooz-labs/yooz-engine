// LoadDeadlineTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Coverage for `awaitLoadTask(_:deadlineSeconds:)` — the bound the in-process
// transport wraps every blocking model load in so a wedged load can never hang
// a caller indefinitely. GPU-free and deterministic: uses plain Tasks, no MLX.

import XCTest
@testable import EngineCore

final class LoadDeadlineTests: XCTestCase {

    /// Happy path: a load that finishes before the deadline returns normally and
    /// the deadline timer is torn down (no spurious throw).
    func testReturnsWhenTaskCompletesBeforeDeadline() async throws {
        let task = Task<Void, Error> { /* completes immediately */ }
        try await awaitLoadTask(task, deadlineSeconds: 5)
    }

    /// Deadline path: a load that never settles is given up on after the
    /// deadline — `awaitLoadTask` throws `LoadDeadlineExceeded` AND cancels the
    /// underlying load Task so cooperatively-cancellable work unwinds.
    func testThrowsDeadlineExceededAndCancelsTask() async {
        let task = Task<Void, Error> {
            // Sleeps "forever"; Task.sleep throws CancellationError on cancel,
            // exactly as real load bodies propagate cancellation.
            try await Task.sleep(nanoseconds: 60_000_000_000)
        }
        do {
            try await awaitLoadTask(task, deadlineSeconds: 0.2)
            XCTFail("expected LoadDeadlineExceeded")
        } catch is LoadDeadlineExceeded {
            // expected
        } catch {
            XCTFail("expected LoadDeadlineExceeded, got \(error)")
        }
        XCTAssertTrue(task.isCancelled, "the load task must be cancelled on deadline")
    }

    /// A genuine load error that races the deadline must propagate verbatim, not
    /// be masked as `LoadDeadlineExceeded` (silent-failure review Finding 1). The
    /// task swallows cancellation and throws its own error, simulating a failure
    /// that coincides with the deadline; only `CancellationError` is re-interpreted
    /// as a timeout, so this surfaces as `Boom`.
    func testRealErrorAfterDeadlineIsNotMaskedAsTimeout() async {
        struct Boom: Error {}
        let task = Task<Void, Error> {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
            throw Boom()
        }
        do {
            try await awaitLoadTask(task, deadlineSeconds: 0.2)
            XCTFail("expected Boom")
        } catch is Boom {
            // correct — the real error propagated, not masked as a timeout
        } catch {
            XCTFail("expected Boom, got \(error)")
        }
    }

    /// A real load error before the deadline propagates verbatim — the deadline
    /// must never mask a genuine failure as a timeout.
    func testRethrowsTaskErrorBeforeDeadline() async {
        struct Boom: Error {}
        let task = Task<Void, Error> { throw Boom() }
        do {
            try await awaitLoadTask(task, deadlineSeconds: 5)
            XCTFail("expected the task's own error")
        } catch is Boom {
            // expected
        } catch {
            XCTFail("expected Boom, got \(error)")
        }
    }
}
