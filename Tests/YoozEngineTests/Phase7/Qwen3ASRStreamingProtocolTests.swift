// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import STTModule
@testable import YoozEngine

/// Phase 7 — protocol-conformance tests for the qwen3 WS streaming
/// path that do NOT require the live Qwen3-ASR model checkpoint.
///
/// The model is only loaded on `start(language:)`. By selecting an
/// unsupported language code (or an out-of-range frame) the server
/// emits a typed error frame before any model load happens. These
/// tests exercise the WS handler's error/path branches without
/// needing /Volumes/S1.
///
/// Heavy end-to-end tests (smoke against the canonical Phase 4
/// reference, WER-delta, latency) live in their own files and skip
/// gracefully when the model isn't on disk.
final class Qwen3ASRStreamingProtocolTests: XCTestCase {

    @MainActor
    func testMalformedConfigReturnsTypedError() async throws {
        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer {
            Task { @MainActor in
                await YoozSTTEngine.shared.setBackend(.parakeet)
            }
        }

        // Reserve a fresh port for this boot — see engine#122.
        UniqueEnginePort.assignFreshPort()
        let server = APIServer()
        try await server.start()
        defer { Task { @MainActor in await server.stop() } }

        let task = try Self.openWS()
        defer { task.cancel(with: .normalClosure, reason: nil) }

        // Send a non-JSON text frame; the handler must reply with a
        // typed `error` JSON message containing
        // "Invalid message format" rather than crashing the server.
        try await task.send(.string("definitely not json"))
        let frame = try await task.receive()
        let text = Self.extractText(from: frame)
        XCTAssertTrue(
            text.contains("Invalid message format"),
            "Expected typed error for bad text frame, got: \(text)"
        )
    }

    @MainActor
    func testUnsupportedLanguageReturnsTypedError() async throws {
        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer {
            Task { @MainActor in
                await YoozSTTEngine.shared.setBackend(.parakeet)
            }
        }

        // Reserve a fresh port for this boot — see engine#122.
        UniqueEnginePort.assignFreshPort()
        let server = APIServer()
        try await server.start()
        defer { Task { @MainActor in await server.stop() } }

        let task = try Self.openWS()
        defer { task.cancel(with: .normalClosure, reason: nil) }

        // qwen3 only routes EN/AR/FA in this engine. Russian is a
        // valid `STTLanguage` (so we hit the per-backend list filter,
        // not the unknown-code branch).
        try await task.send(.string(#"{"type":"config","language":"ru"}"#))
        let frame = try await task.receive()
        let text = Self.extractText(from: frame)
        XCTAssertTrue(
            text.contains("not supported") || text.contains("Russian"),
            "Expected typed error for unsupported language, got: \(text)"
        )
    }

    @MainActor
    func testOddByteCountAudioFrameReturnsTypedError() async throws {
        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer {
            Task { @MainActor in
                await YoozSTTEngine.shared.setBackend(.parakeet)
            }
        }

        // Reserve a fresh port for this boot — see engine#122.
        UniqueEnginePort.assignFreshPort()
        let server = APIServer()
        try await server.start()
        defer { Task { @MainActor in await server.stop() } }

        let task = try Self.openWS()
        defer { task.cancel(with: .normalClosure, reason: nil) }

        // Send 7 bytes — not a multiple of MemoryLayout<Float>.size.
        // The handler must emit a typed error and stay alive.
        let bogus = Data([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
        try await task.send(.data(bogus))
        let frame = try await task.receive()
        let text = Self.extractText(from: frame)
        XCTAssertTrue(
            text.contains("not a multiple of Float32"),
            "Expected typed error for odd-byte frame, got: \(text)"
        )

        // Connection must remain valid: send a valid (empty)
        // text-frame and confirm we still get a typed error rather
        // than a crash / hang.
        try await task.send(.string("not json"))
        let nextFrame = try await task.receive()
        let nextText = Self.extractText(from: nextFrame)
        XCTAssertTrue(
            nextText.contains("Invalid message format"),
            "Connection state should survive a malformed audio frame; got: \(nextText)"
        )
    }

    // MARK: - Session unit tests (no server, no WS)

    /// Pure-Swift unit test for the streaming session's buffer math.
    /// Doesn't load the model — only exercises the audio-buffering
    /// arithmetic and finalization-once invariant.
    func testSessionBuffersAndFinalizesOnce() async throws {
        let session = Qwen3ASRStreamingSession(
            languageHint: "English",
            backend: Qwen3ASRBackend.shared,
            sampleRate: 16_000,
            maxBufferedSeconds: 1
        )

        // Buffer 0.5 s of silence in two pushes.
        let half = [Float](repeating: 0.0, count: 8_000)
        try await session.push(samples: half)
        try await session.push(samples: half)
        let bufferedMs = await session.audioDurationMs
        XCTAssertEqual(bufferedMs, 1_000)

        // Cap at 1 second — extra samples drop silently.
        let extra = [Float](repeating: 0.0, count: 4_000)
        try await session.push(samples: extra)
        let cappedMs = await session.audioDurationMs
        XCTAssertEqual(cappedMs, 1_000, "Buffer must not grow past the cap")

        // Discard avoids hitting the model.
        await session.discard()
        let isFinalized = await session.isFinalized
        XCTAssertTrue(isFinalized)

        // A second `push` after finalize throws.
        do {
            _ = try await session.push(samples: half)
            XCTFail("push() after finalize must throw")
        } catch Qwen3ASRStreamingSession.SessionError.finalized {
            // expected
        }
    }

    // MARK: - Helpers

    private static func openWS() throws -> URLSessionWebSocketTask {
        guard
            let url = URL(
                string: "ws://\(EngineConfig.host):\(EngineConfig.port)/v1/stt/stream"
            )
        else {
            throw NSError(
                domain: "test",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to compose WS URL"]
            )
        }
        let task = URLSession(configuration: .default).webSocketTask(with: url)
        task.resume()
        return task
    }

    private static func extractText(
        from message: URLSessionWebSocketTask.Message
    ) -> String {
        switch message {
        case .string(let s):
            return s
        case .data(let d):
            return String(data: d, encoding: .utf8) ?? ""
        @unknown default:
            return ""
        }
    }
}
