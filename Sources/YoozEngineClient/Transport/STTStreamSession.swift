import Foundation

/// Backing for a streaming STT session, provided by the active transport
/// (epic #192 Phase 2b). `STTStream` is a thin public wrapper over this so its
/// API is identical regardless of transport:
///
///   - `WebSocketSTTStreamSession` (loopback): frames over a WebSocket.
///   - `InProcessSTTStreamSession` (`YoozEngineInProcess`): drives the engine
///     `StreamingTranscriber` / Qwen3 session / Apple buffer directly.
@available(macOS 14.0, iOS 17.0, *)
public protocol STTStreamSession: Sendable {
    /// Feed audio samples (Float32 at 16 kHz) into the session.
    func sendAudio(_ samples: [Float]) async throws
    /// Receive the next transcription result, or `nil` once the session is
    /// closed/exhausted.
    func receive() async throws -> StreamingSTTResult?
    /// Close the session and release its resources.
    func close()
}

/// WebSocket-backed `STTStreamSession` — the loopback transport's streaming
/// implementation, lifted out of `STTStream` behind the session seam. Behavior
/// is unchanged from the pre-seam `STTStream`.
@available(macOS 14.0, iOS 17.0, *)
final class WebSocketSTTStreamSession: STTStreamSession, @unchecked Sendable {
    private let task: URLSessionWebSocketTask
    private let session: URLSession

    init(task: URLSessionWebSocketTask, session: URLSession) {
        self.task = task
        self.session = session
    }

    deinit {
        task.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
    }

    func sendAudio(_ samples: [Float]) async throws {
        let data = samples.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
        try await task.send(.data(data))
    }

    func receive() async throws -> StreamingSTTResult? {
        let message: URLSessionWebSocketTask.Message
        do {
            message = try await task.receive()
        } catch is CancellationError {
            return nil
        } catch let urlError as URLError
            where urlError.code == .cancelled || urlError.code == .networkConnectionLost {
            return nil
        } catch let nsError as NSError
            where nsError.domain == NSPOSIXErrorDomain && nsError.code == 57 {
            // POSIX error 57: socket is not connected (graceful close)
            return nil
        }

        switch message {
        case .string(let text):
            guard let data = text.data(using: .utf8) else {
                throw YoozEngineError.decodingError("Received non-UTF8 text from WebSocket")
            }
            return try JSONDecoder().decode(StreamingSTTResult.self, from: data)
        case .data(let data):
            return try JSONDecoder().decode(StreamingSTTResult.self, from: data)
        @unknown default:
            throw YoozEngineError.invalidResponse
        }
    }

    func close() {
        task.cancel(with: .normalClosure, reason: nil)
    }
}
