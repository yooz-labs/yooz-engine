// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import STTModule
@testable import YoozEngine

/// Phase 5 placeholder — under Phase 5 this file asserted that the WS
/// `/v1/stt/stream` endpoint *rejected* connections for the
/// `qwen3_asr_preview` backend with a JSON error frame containing the
/// "Phase 7" sentinel. Phase 7 (issue #61) lifts that rejection: the
/// streaming path is now wired up.
///
/// This test is kept in place (rather than deleted) so the symbol
/// referenced by the Phase 5 PR's CI history still exists; its
/// assertion has been inverted to guard against a regression that
/// puts the rejection back. The full streaming protocol is exercised
/// by `Qwen3ASRStreamingProtocolTests` and the heavy end-to-end
/// streaming smoke (which requires the live model) lives in
/// `Qwen3ASRStreamingSmokeTests`.
final class Qwen3ASRWebSocketTests: XCTestCase {

    @MainActor
    func testQwen3StreamNoLongerSendsDeferredError() async throws {
        // Switch the global engine to qwen3 BEFORE booting the server
        // so the upgrade handler sees the right state on connect.
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
        defer {
            Task { @MainActor in await server.stop() }
        }

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

        // The server should NOT proactively send a frame on connect
        // anymore — it waits for a config message. To assert "no
        // Phase-7 deferred frame" without blocking on receive forever
        // we push a config that asks for an unsupported language;
        // the response is a typed error that does not contain
        // "Phase 7".
        let badConfig = """
        {"type":"config","language":"zz"}
        """
        try await task.send(.string(badConfig))

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

        XCTAssertFalse(
            text.contains("Phase 7"),
            "Expected the qwen3 streaming path to NOT emit the legacy "
                + "Phase-7 deferred frame, got: \(text)"
        )
        XCTAssertTrue(
            text.contains("error") || text.contains("Unknown")
                || text.contains("not implemented"),
            "Expected a typed error frame for the bogus language code, got: \(text)"
        )

        task.cancel(with: .normalClosure, reason: nil)
    }
}
