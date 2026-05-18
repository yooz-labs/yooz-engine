// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import AppleSTTModule
import EngineCore
@testable import YoozEngine

/// Tests for the `/v1/stt/stream` WebSocket path when `apple_stt` is the
/// active backend. Engine #123 — Option A (batch-on-close): the WS handler
/// accepts the handshake, buffers Float32 frames, and emits a single final
/// `WSSTTResult` derived from `AppleSTTEngine.batchTranscribe(samples:)`
/// when the client closes the stream.
///
/// These are protocol-level tests; the speech-recognition-authorized
/// happy path is gated behind `YOOZ_STT_LOAD_APPLE=1` to match the
/// AppleSTTModule unit tests. The unauthorized + malformed-frame paths
/// run unconditionally so CI catches handler regressions without needing
/// an authorized test host.
final class AppleSTTStreamingWSTests: XCTestCase {

    /// Matches `AppleSTTModuleTests.shouldRunAuthedTests`. Gate any test
    /// that requires speech-recognition authorization on this flag so the
    /// "no mocks" policy doesn't force a real OS prompt during CI.
    private var shouldRunAuthedTests: Bool {
        ProcessInfo.processInfo.environment["YOOZ_STT_LOAD_APPLE"] == "1"
    }

    // MARK: - Server lifecycle helpers

    @MainActor
    private func withAppleSTTServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
        UniqueEnginePort.assignFreshPort()
        let server = APIServer()
        try await server.start()
        do {
            // Switch the server's active engine to apple_stt via the
            // canonical HTTP route. We can't poke `currentSTTEngine`
            // directly — it's `private` on `APIServer`. Tests on the
            // whisper variant default to `parakeet`; lite defaults to
            // `appleSTT` but the POST is idempotent.
            try await switchToAppleSTT(on: server)
            let result = try await body(server)
            await server.stop()
            return result
        } catch {
            await server.stop()
            throw error
        }
    }

    private func switchToAppleSTT(on server: APIServer) async throws {
        let url = URL(
            string: "http://\(EngineConfig.host):\(EngineConfig.port)/v1/stt/engine"
        )!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = #"{"id":"apple_stt"}"#.data(using: .utf8)
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw NSError(
                domain: "test", code: 0,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "POST /v1/stt/engine apple_stt did not return 200"
                ]
            )
        }
    }

    // MARK: - Protocol tests (no auth required)

    /// `apple_stt` config arriving on an unauthorized host returns a
    /// typed `model_load_failed` error frame — never a silent hang.
    /// Whisper#178 surfaces as "completely silent" today; this test pins
    /// the failure mode to an observable error frame instead.
    @MainActor
    func testConfigUnauthorizedReturnsModelLoadFailed() async throws {
        // If speech recognition IS authorized on this host the engine
        // will load successfully and this test won't exercise the
        // failure branch — skip rather than misreport.
        try XCTSkipIf(
            AppleSTTEngine.authorizationStatus == .authorized,
            "host is authorized for speech recognition; skipping unauth path"
        )

        try await withAppleSTTServer { _ in
            let task = try Self.openWS()
            defer { task.cancel(with: .normalClosure, reason: nil) }

            try await task.send(.string(#"{"type":"config","language":"en"}"#))
            let frame = try await task.receive()
            let text = Self.extractText(from: frame)
            XCTAssertTrue(
                text.contains("Failed to load Apple STT engine"),
                "Expected typed model_load_failed for unauthorized host, got: \(text)"
            )
            XCTAssertTrue(
                text.contains("\"code\":\"model_load_failed\""),
                "Expected error frame to carry typed code, got: \(text)"
            )
        }
    }

    /// A language with no Apple recognizer (Vietnamese is in
    /// `STTLanguage` but not in `AppleSTTLanguage`) must produce a
    /// typed `language_not_supported_by_backend` error frame.
    @MainActor
    func testConfigUnsupportedLanguageReturnsTypedError() async throws {
        try await withAppleSTTServer { _ in
            let task = try Self.openWS()
            defer { task.cancel(with: .normalClosure, reason: nil) }

            try await task.send(.string(#"{"type":"config","language":"vi"}"#))
            let frame = try await task.receive()
            let text = Self.extractText(from: frame)
            XCTAssertTrue(
                text.contains("not supported by apple_stt"),
                "Expected typed error for unsupported language, got: \(text)"
            )
            XCTAssertTrue(
                text.contains("\"code\":\"language_not_supported_by_backend\""),
                "Expected error frame to carry typed code, got: \(text)"
            )
        }
    }

    /// Binary frames whose byte count isn't a multiple of `Float32` size
    /// must return a typed `invalid_audio_frame` error and the connection
    /// must remain open for the next message.
    @MainActor
    func testOddByteCountAudioFrameReturnsTypedError() async throws {
        try await withAppleSTTServer { _ in
            let task = try Self.openWS()
            defer { task.cancel(with: .normalClosure, reason: nil) }

            let bogus = Data([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
            try await task.send(.data(bogus))
            let frame = try await task.receive()
            let text = Self.extractText(from: frame)
            XCTAssertTrue(
                text.contains("not a multiple of Float32"),
                "Expected typed error for odd-byte frame, got: \(text)"
            )

            // Connection must survive; next malformed text frame should
            // also produce a typed error rather than a crash.
            try await task.send(.string("definitely not json"))
            let next = try await task.receive()
            let nextText = Self.extractText(from: next)
            XCTAssertTrue(
                nextText.contains("Invalid message format"),
                "Connection must survive malformed audio; got: \(nextText)"
            )
        }
    }

    // MARK: - Authorized happy path (gated)

    /// With speech-recognition authorized, the config→ready handshake
    /// must complete and a `final` frame must arrive when the client
    /// closes the stream — even without any audio buffered. Mirrors the
    /// qwen3 path's empty-final emission for protocol symmetry.
    @MainActor
    func testConfigReadyAndEmptyFinalOnClose() async throws {
        try XCTSkipUnless(
            shouldRunAuthedTests,
            "YOOZ_STT_LOAD_APPLE=1 not set; skipping authorized handshake"
        )
        try XCTSkipUnless(
            AppleSTTEngine.authorizationStatus == .authorized,
            "speech recognition not authorized on host"
        )

        try await withAppleSTTServer { _ in
            let task = try Self.openWS()
            defer { task.cancel(with: .normalClosure, reason: nil) }

            try await task.send(.string(#"{"type":"config","language":"en"}"#))
            let ready = try await task.receive()
            let readyText = Self.extractText(from: ready)
            XCTAssertTrue(
                readyText.contains("\"ready\""),
                "Expected ready frame, got: \(readyText)"
            )
            XCTAssertTrue(
                readyText.contains("\"language\":\"en\""),
                "Ready frame should echo language code, got: \(readyText)"
            )

            // Close without sending audio. Server-side close handling
            // should emit a single `final` frame with empty text.
            //
            // URLSessionWebSocketTask doesn't expose a clean half-close
            // — `cancel(with:reason:)` sends the close frame but the
            // server may already have written the `final` frame before
            // we drain. Receive with a short timeout to catch it.
            let finalReceived = expectation(description: "final frame received")
            Task {
                while true {
                    do {
                        let msg = try await task.receive()
                        let txt = Self.extractText(from: msg)
                        if txt.contains("\"final\"") {
                            XCTAssertTrue(
                                txt.contains("\"text\":\"\""),
                                "Empty audio should yield empty final text, got: \(txt)"
                            )
                            finalReceived.fulfill()
                            break
                        }
                    } catch {
                        break
                    }
                }
            }
            task.cancel(with: .normalClosure, reason: nil)
            await fulfillment(of: [finalReceived], timeout: 5.0)
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
                domain: "test", code: 1,
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
        case .string(let s): return s
        case .data(let d): return String(data: d, encoding: .utf8) ?? ""
        @unknown default: return ""
        }
    }
}
