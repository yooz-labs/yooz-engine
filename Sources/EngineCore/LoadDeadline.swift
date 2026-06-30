// LoadDeadline.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Thrown when a model load `Task` does not finish within its deadline.
///
/// The in-process transport has no HTTP-client timeout to fall back on, so a
/// blocking caller that awaits a load wraps it in `awaitLoadTask(_:deadlineSeconds:)`.
/// On expiry the load Task is cancelled and this error is thrown so the caller
/// surfaces a real failure instead of blocking forever.
public struct LoadDeadlineExceeded: Error, LocalizedError, Sendable {
    public let seconds: Double

    public init(seconds: Double) {
        self.seconds = seconds
    }

    public var errorDescription: String? {
        "Model load did not complete within \(Int(seconds))s."
    }
}

/// Thread-safe one-shot flag shared between the watchdog Task and the awaiting
/// caller. NSLock is used synchronously (no `await` between lock/unlock).
private final class DeadlineFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var didFire = false

    func markFired() {
        lock.lock()
        didFire = true
        lock.unlock()
    }

    var fired: Bool {
        lock.lock()
        defer { lock.unlock() }
        return didFire
    }
}

/// Await a load `Task`, but give up after `deadlineSeconds` — cancelling the
/// task and throwing `LoadDeadlineExceeded`. If the task finishes first (success
/// or its own error), that result is returned/rethrown and the watchdog is
/// cancelled. Bounds the in-process blocking load path so a wedged load can never
/// hang a caller indefinitely; mirrors, for the blocking callers that still await
/// completion, the fire-and-forget + poll contract the loopback `APIServer` uses.
///
/// Error semantics (so a real failure is never masked as a timeout): the deadline
/// is surfaced ONLY when the load task throws `CancellationError` after the
/// watchdog fired. A genuine load error (network, missing/corrupt weights, …)
/// that races the deadline propagates verbatim, and a task that completes — even
/// a hair after the deadline cancel — reports success, because the load did finish.
///
/// Caveat: cancellation only unwinds *cooperatively cancellable* work — the
/// Hugging Face download + materialization inside `loadModelContainer` (the LLM
/// switch path) IS cancellable, so the deadline frees the caller promptly there.
/// A fully synchronous `ParakeetModel.fromDirectory` cannot be interrupted
/// mid-call, so the `await` below still blocks until it completes even if the
/// deadline fires. The blocking in-process callers affected are
/// `loadModel(wait:true)` and stream opens (`openSTTStream`) for the
/// parakeet/fastConformer backends; `loadModelAsync` (whisper's pre-warm path)
/// takes the `wait == false` route and never calls `awaitLoadTask`.
public func awaitLoadTask(
    _ task: Task<Void, Error>,
    deadlineSeconds: Double
) async throws {
    let flag = DeadlineFlag()
    // Clamp to a finite, positive sleep so a non-finite caller value (e.g. a
    // `.infinity` "wait forever" sentinel) can't trap the watchdog's UInt64
    // conversion. The shipping caller passes a fixed 600s.
    let nanos: UInt64
    let product = deadlineSeconds * 1_000_000_000
    if deadlineSeconds.isFinite, product > 0, product < Double(UInt64.max) {
        nanos = UInt64(product)
    } else {
        nanos = .max
    }
    let watchdog = Task {
        try? await Task.sleep(nanoseconds: nanos)
        if !Task.isCancelled {
            flag.markFired()
            task.cancel()
        }
    }
    defer { watchdog.cancel() }

    do {
        try await task.value
        // The load finished — even if a late watchdog cancel raced in, the work
        // completed, so report success. Only an actual `CancellationError` below
        // is re-interpreted as a deadline.
    } catch is CancellationError {
        // Either our watchdog cancelled the load (deadline) or the caller did.
        if flag.fired { throw LoadDeadlineExceeded(seconds: deadlineSeconds) }
        throw CancellationError()
    }
    // A genuine (non-cancellation) load error is NOT caught above, so it
    // propagates verbatim and is never masked as a timeout — even if it raced
    // the deadline.
}
