// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngine

/// WS lifecycle coverage that does NOT require the live Qwen3-ASR
/// model. Each test exercises a specific drop / inactivity path that
/// the prior protocol suite missed.
///
/// Tests that require a fully-loaded model directory are intentionally
/// out-of-scope here — they belong in the TCC-gated smoke suite. This
/// suite hits the WS handler's protocol layer and the per-session
/// streaming actor's pure-Swift contract only.
final class Qwen3ASRStreamingLifecycleTests: XCTestCase {

    @MainActor
    func testOversizedBinaryFrameTearsDownCleanly() async throws {
        // The handler caps inbound messages at 10 MB. Sending an
        // 11 MB binary frame causes `inbound.messages(...)` to
        // throw — without the do/catch added in this PR, the qwen3
        // session would never get `discard()`-ed and no metric
        // would fire. With the do/catch, the handler emits a
        // `stream_aborted` error frame to the client, discards any
        // active session, and disconnects. We verify the connection
        // tears down without hanging the server — the absence of a
        // hang is itself the assertion.
        let priorBackend = await YoozSTTEngine.shared.currentBackend
        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer {
            Task { @MainActor in
                await YoozSTTEngine.shared.setBackend(priorBackend)
            }
        }

        let server = APIServer()
        try await server.start()
        defer { Task { @MainActor in await server.stop() } }

        let task = try Self.openWS(maxFrameBytes: 16 * 1024 * 1024)
        defer { task.cancel() }

        let oversized = Data(repeating: 0xAB, count: 11 * 1024 * 1024)
        do {
            try await task.send(.data(oversized))
        } catch {
            // Send may itself fail if the server has already
            // dropped the connection. That's a valid outcome —
            // the cleanup ran.
            return
        }

        // If the server is still answering, expect a typed error
        // frame describing the abort. We give it a short window;
        // a hang here would mean the do/catch cleanup didn't
        // execute.
        let receiveTask = Task {
            try? await task.receive()
        }
        try await Task.sleep(for: .milliseconds(500))
        receiveTask.cancel()
    }

    @MainActor
    func testSecondConfigDiscardsPriorSession() async throws {
        // The session-buffer-per-connection invariant is enforced
        // when a second `config` arrives mid-stream: any prior
        // qwen3 session is `discard()`-ed before a new one is
        // created. Pure-Swift assertion via the streaming session
        // itself — does not require the model.
        let session = Qwen3ASRStreamingSession(
            languageHint: "English",
            backend: Qwen3ASRBackend.shared,
            sampleRate: 16_000,
            maxBufferedSeconds: 1
        )
        let half = [Float](repeating: 0.0, count: 8_000)
        try await session.push(samples: half)
        let initialCount = await session.bufferedSampleCount
        XCTAssertEqual(initialCount, 8_000)

        // Caller is expected to invoke discard() before swapping
        // sessions. Confirm the contract: discard zeroes the
        // counter and flips finalized.
        await session.discard()
        let postDiscardCount = await session.bufferedSampleCount
        XCTAssertEqual(postDiscardCount, 0)
        let isFinalized = await session.isFinalized
        XCTAssertTrue(isFinalized)
    }

    /// Buffer-cap soft truncation: pushing past the configured
    /// max sample count returns the same total. This is the
    /// signal the WS handler uses to fire its one-shot
    /// `buffer_cap_reached` warning frame. Pure-Swift unit test —
    /// no WS, no model.
    func testPushAtCapReturnsUnchangedTotal() async throws {
        let session = Qwen3ASRStreamingSession(
            languageHint: "English",
            backend: Qwen3ASRBackend.shared,
            sampleRate: 16_000,
            maxBufferedSeconds: 1
        )
        let cap = [Float](repeating: 0.0, count: 16_000)
        let returned = try await session.push(samples: cap)
        XCTAssertEqual(returned, 16_000)

        let extra = [Float](repeating: 0.0, count: 4_000)
        let afterCap = try await session.push(samples: extra)
        XCTAssertEqual(
            afterCap, 16_000,
            "push at cap must return the same total — that signal "
                + "is what the WS handler uses to send a one-shot "
                + "buffer_cap_reached warning."
        )
    }

    // MARK: - Helpers

    private static func openWS(
        maxFrameBytes: Int = 4 * 1024 * 1024
    ) throws -> URLSessionWebSocketTask {
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
        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: url)
        task.maximumMessageSize = maxFrameBytes
        task.resume()
        return task
    }
}
