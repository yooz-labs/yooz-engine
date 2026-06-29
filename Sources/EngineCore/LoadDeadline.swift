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
/// Caveat: cancellation only unwinds *cooperatively cancellable* work — the
/// Hugging Face download + materialization inside `loadModelContainer` (the LLM
/// switch path) IS cancellable, so the deadline frees the caller promptly there.
/// A fully synchronous `ParakeetModel.fromDirectory` cannot be interrupted
/// mid-call, so the `await` below still waits for it; the only blocking in-process
/// caller of a synchronous STT load is the legacy `loadModel(wait:true)` —
/// whisper's STT path is fire-and-forget and never reaches here.
public func awaitLoadTask(
    _ task: Task<Void, Error>,
    deadlineSeconds: Double
) async throws {
    let flag = DeadlineFlag()
    let watchdog = Task {
        try? await Task.sleep(nanoseconds: UInt64(deadlineSeconds * 1_000_000_000))
        if !Task.isCancelled {
            flag.markFired()
            task.cancel()
        }
    }
    defer { watchdog.cancel() }

    do {
        try await task.value
    } catch {
        // A cancellation/error that lands after the watchdog fired is the
        // deadline surfacing; otherwise propagate the load's own error verbatim
        // so a real failure is never masked as a timeout.
        if flag.fired { throw LoadDeadlineExceeded(seconds: deadlineSeconds) }
        throw error
    }
    // The load completed, but if the watchdog had already fired (a
    // cancellation-swallowing body that still ran to completion), treat it as a
    // deadline so the contract is consistent.
    if flag.fired { throw LoadDeadlineExceeded(seconds: deadlineSeconds) }
}
