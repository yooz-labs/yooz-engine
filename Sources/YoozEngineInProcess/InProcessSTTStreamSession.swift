import AppleSTTModule
import Foundation
import STTModule
import YoozEngineClient

/// In-process `STTStreamSession`: drives the engine's streaming transcriber (or
/// Apple buffer) directly — no WebSocket (epic #192 Phase 2b).
///
/// Partials are produced on `sendAudio` and delivered through `receive()` via a
/// single-consumer async channel. `close()` triggers `finalize()` asynchronously,
/// delivers the final result, then ends the stream (a later `receive()` returns
/// nil). A finalize error surfaces by throwing from `receive()`.
///
/// Backends: Parakeet / FastConformer (incremental `StreamingTranscriber`) and
/// Apple STT (buffer-then-finalize). The qwen3 preview backend is loopback/dev
/// only — `InProcessTransport.openSTTStream` reports `unsupportedInProcess` for it.
@available(macOS 14.0, iOS 17.0, *)
final class InProcessSTTStreamSession: STTStreamSession, @unchecked Sendable {
    enum Backend {
        /// Parakeet / FastConformer: synchronous incremental transcriber.
        case parakeet(StreamingTranscriber)
        /// Apple STT: buffer audio, transcribe on finalize.
        case apple(AppleSTTEngine)
    }

    private let backend: Backend

    // Single-consumer async channel. All fields guarded by `lock`.
    private let lock = NSLock()
    private var queue: [StreamingSTTResult] = []
    private var finished = false
    private var failure: Error?
    private var waiter: CheckedContinuation<StreamingSTTResult?, Error>?
    private var appleBuffer: [Float] = []
    private var closeTriggered = false

    init(backend: Backend) {
        self.backend = backend
    }

    func sendAudio(_ samples: [Float]) async throws {
        switch backend {
        case .parakeet(let transcriber):
            // addAudio is synchronous and returns the running partial.
            let result = transcriber.addAudio(samples: samples)
            deliver(
                StreamingSTTResult(
                    type: "partial",
                    text: result.text,
                    finalized: result.finalized,
                    draft: result.draft
                )
            )
        case .apple:
            lock.lock()
            appleBuffer.append(contentsOf: samples)
            lock.unlock()
            // Apple has no intermediate inference — emit an empty heartbeat,
            // matching the loopback WS handler.
            deliver(StreamingSTTResult(type: "partial", text: "", finalized: "", draft: ""))
        }
    }

    func receive() async throws -> StreamingSTTResult? {
        lock.lock()
        if !queue.isEmpty {
            let result = queue.removeFirst()
            lock.unlock()
            return result
        }
        if finished {
            let pendingError = failure
            lock.unlock()
            if let pendingError { throw pendingError }
            return nil
        }
        lock.unlock()

        // No data yet and not finished: suspend until the next deliver/finish.
        // Single consumer, so at most one waiter exists at a time.
        // `withTaskCancellationHandler` (plus the in-closure re-checks) makes
        // this cancellation-safe: a consumer whose task is cancelled while
        // suspended is resumed with `CancellationError` rather than leaking the
        // continuation (which would trap on deallocation).
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                // Re-check under the lock: deliver/finish may have run between
                // the unlock above and here.
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
        } onCancel: {
            lock.lock()
            let pending = waiter
            waiter = nil
            lock.unlock()
            pending?.resume(throwing: CancellationError())
        }
    }

    func close() {
        lock.lock()
        if closeTriggered {
            lock.unlock()
            return
        }
        closeTriggered = true
        lock.unlock()
        // finalize() may be async (Apple), so run it off the synchronous close().
        Task { await self.finalizeAndFinish() }
    }

    // MARK: - Channel internals

    /// Hand a result to a waiting `receive()`, or buffer it.
    private func deliver(_ result: StreamingSTTResult) {
        lock.lock()
        if finished {
            lock.unlock()
            return
        }
        if let continuation = waiter {
            waiter = nil
            lock.unlock()
            continuation.resume(returning: result)
            return
        }
        queue.append(result)
        lock.unlock()
    }

    /// Close the channel, optionally with an error. Resumes any waiter.
    private func finish(throwing error: Error?) {
        lock.lock()
        if finished {
            lock.unlock()
            return
        }
        finished = true
        failure = error
        if let continuation = waiter {
            waiter = nil
            lock.unlock()
            if let error {
                continuation.resume(throwing: error)
            } else {
                continuation.resume(returning: nil)
            }
            return
        }
        lock.unlock()
    }

    private func finalizeAndFinish() async {
        do {
            let result: StreamingSTTResult
            switch backend {
            case .parakeet(let transcriber):
                let final = transcriber.finalize()
                result = StreamingSTTResult(
                    type: "final",
                    text: final.text,
                    finalized: final.finalized,
                    draft: ""
                )
            case .apple(let engine):
                lock.lock()
                let buffered = appleBuffer
                lock.unlock()
                let text = try await engine.batchTranscribe(samples: buffered)
                result = StreamingSTTResult(type: "final", text: text, finalized: text, draft: "")
            }
            deliver(result)
            finish(throwing: nil)
        } catch {
            finish(throwing: error)
        }
    }
}
