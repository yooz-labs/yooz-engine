import Foundation

/// Single-consumer async channel for streaming STT results.
///
/// `yield` / `finish` are called from the producer (the XPC callback, or the
/// in-process transcriber drain); `receive()` is the consumer. It is
/// **cancellation-safe**: a consumer whose task is cancelled while suspended is
/// resumed with `CancellationError` rather than leaking the continuation (which
/// would trap on deallocation). Buffers when the consumer isn't waiting.
final class STTResultChannel: @unchecked Sendable {
    private let lock = NSLock()
    private var queue: [StreamingSTTResult] = []
    private var finished = false
    private var failure: Error?
    private var waiter: CheckedContinuation<StreamingSTTResult?, Error>?

    /// Hand a result to a waiting `receive()`, or buffer it. No-op after finish.
    func yield(_ result: StreamingSTTResult) {
        lock.lock()
        if finished {
            lock.unlock()
            return
        }
        if let waiter {
            self.waiter = nil
            lock.unlock()
            waiter.resume(returning: result)
            return
        }
        queue.append(result)
        lock.unlock()
    }

    /// End the stream — `nil` is a clean close, otherwise the error surfaces from
    /// `receive()`. Idempotent.
    func finish(throwing error: Error? = nil) {
        lock.lock()
        if finished {
            lock.unlock()
            return
        }
        finished = true
        failure = error
        if let waiter {
            self.waiter = nil
            lock.unlock()
            if let error {
                waiter.resume(throwing: error)
            } else {
                waiter.resume(returning: nil)
            }
            return
        }
        lock.unlock()
    }

    /// Next buffered result, or suspend until one arrives / the stream ends.
    /// Returns nil at a clean end; throws the finish error (or `CancellationError`).
    ///
    /// The lock is only touched from the synchronous helpers below — never
    /// directly in this `async` body (`NSLock` is `noasync`).
    func receive() async throws -> StreamingSTTResult? {
        switch dequeueOrSuspend() {
        case .value(let result):
            return result
        case .finished(let error):
            if let error { throw error }
            return nil
        case .suspend:
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    register(continuation)
                }
            } onCancel: {
                resumeWaiterWithCancellation()
            }
        }
    }

    private enum FastPath {
        case value(StreamingSTTResult)
        case finished(Error?)
        case suspend
    }

    private func dequeueOrSuspend() -> FastPath {
        lock.lock()
        defer { lock.unlock() }
        if !queue.isEmpty { return .value(queue.removeFirst()) }
        if finished { return .finished(failure) }
        return .suspend
    }

    /// Register a waiting continuation, re-checking state (yield/finish/cancel
    /// may have raced since `dequeueOrSuspend`).
    private func register(_ continuation: CheckedContinuation<StreamingSTTResult?, Error>) {
        lock.lock()
        if !queue.isEmpty {
            let result = queue.removeFirst()
            lock.unlock()
            continuation.resume(returning: result)
            return
        }
        if finished {
            let pendingError = failure
            lock.unlock()
            if let pendingError {
                continuation.resume(throwing: pendingError)
            } else {
                continuation.resume(returning: nil)
            }
            return
        }
        if Task.isCancelled {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
        }
        waiter = continuation
        lock.unlock()
    }

    private func resumeWaiterWithCancellation() {
        lock.lock()
        let pending = waiter
        waiter = nil
        lock.unlock()
        pending?.resume(throwing: CancellationError())
    }
}
