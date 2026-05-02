// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngine

/// Phase 5 — WebSocket smoke test for the qwen3 backend. Streaming
/// for `qwen3_asr_preview` lands in Phase 7 (issue #58); for now the
/// server must reject the connection cleanly with a JSON error frame
/// rather than crashing or pretending to stream.
///
/// We can't use HummingbirdWSTesting here because of a known
/// transitive-Logging link issue in `swift-websocket`'s WSCore target.
/// Instead we boot the real APIServer on its localhost port (port
/// 19920) and dial it via `URLSessionWebSocketTask`. The server reuse
/// of the canonical port means only one of these tests can run at a
/// time per process, which XCTest already serializes by default.
final class Qwen3ASRWebSocketTests: XCTestCase {

    @MainActor
    func testQwen3StreamRejectsWithDeferredMessage() async throws {
        // Switch the global engine to qwen3 BEFORE booting the server
        // so the upgrade handler sees the right state on connect.
        await YoozSTTEngine.shared.setBackend(.qwen3ASRPreview)
        defer {
            Task { @MainActor in
                await YoozSTTEngine.shared.setBackend(.parakeet)
            }
        }

        let server = APIServer()
        try await server.start()
        defer {
            Task { @MainActor in await server.stop() }
        }

        // Connect over WS, read the first frame, assert it carries the
        // deferred-streaming sentinel "Phase 7".
        guard
            let url = URL(
                string: "ws://\(EngineConfig.host):\(EngineConfig.port)/v1/stt/stream"
            )
        else {
            return XCTFail("Failed to compose WS URL")
        }
        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: url)
        task.resume()

        // First frame: should be a JSON error message containing
        // "Phase 7". The server then closes the connection on its own.
        let message = try await task.receive()
        let text: String
        switch message {
        case .string(let s):
            text = s
        case .data(let d):
            text = String(data: d, encoding: .utf8) ?? ""
        @unknown default:
            text = ""
        }

        XCTAssertTrue(
            text.contains("Phase 7"),
            "Expected deferred-streaming error frame, got: \(text)"
        )

        task.cancel(with: .normalClosure, reason: nil)
    }
}
