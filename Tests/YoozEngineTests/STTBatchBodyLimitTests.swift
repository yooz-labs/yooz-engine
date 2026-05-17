// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import EngineCore
@testable import YoozEngine

/// Regression coverage for engine #111 / yooz-whisper #176:
/// `/v1/stt/batch` rejected medium-long audio chunks (~17 s+) with
/// HTTP 400 / `NIOTooManyBytesError` because Hummingbird 2's default
/// `BasicRequestContext.maxUploadSize` is 2 MiB. The fix uses a
/// custom `YoozEngineRequestContext` that raises the ceiling to
/// 64 MiB so JSON-encoded Float samples for ~5 minutes of audio
/// fit comfortably.
///
/// We exercise the bounds directly:
///
/// 1. The custom context exposes the new ceiling as a constant.
/// 2. A POST body larger than the old 2 MiB limit but smaller than
///    the new 64 MiB limit must reach the handler — i.e. the
///    framework no longer rejects it up front with a body-size
///    error. We probe this with an intentionally malformed JSON
///    body of ~3 MiB; the handler answers 400 `invalid_request`
///    with a JSON-decode message (NOT a "too many bytes" framework
///    message). Both buggy and fixed code return 400; the buggy
///    code's message contained the NIO error signature, while the
///    fix's message comes from `JSONDecoder`.
///
/// Boots a real `APIServer` (same harness as
/// `Qwen3ASREngineRouteTests` — `HummingbirdTesting` triggers a
/// linker bug in this target).
final class STTBatchBodyLimitTests: XCTestCase {

    // MARK: - Server lifecycle helpers

    @MainActor
    private func withServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
        let server = APIServer()
        try await server.start()
        let result: T
        do {
            result = try await body(server)
        } catch {
            await server.stop()
            throw error
        }
        await server.stop()
        return result
    }

    private func baseURL() -> URL {
        URL(string: "http://\(EngineConfig.host):\(EngineConfig.port)")!
    }

    private func post(
        _ path: String,
        body: Data
    ) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(
            url: baseURL().appendingPathComponent(path)
        )
        request.httpMethod = "POST"
        request.setValue(
            "application/json",
            forHTTPHeaderField: "Content-Type"
        )
        request.httpBody = body
        // Default timeout is 60 s; 3 MiB upload over loopback is
        // sub-second but we leave the default in place.
        let (data, response) = try await URLSession.shared.data(
            for: request
        )
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "test", code: 0)
        }
        return (http, data)
    }

    // MARK: - Tests

    /// The body-size constant is the contract. If a future refactor
    /// shrinks it below the JSON cost of a ~5-minute recording the
    /// regression for #111 is back.
    func testRequestContextExposesUploadCeiling() {
        // 64 MiB exactly. Bumping this is a deliberate decision; if
        // someone bumps it down the test forces them to think about it.
        XCTAssertEqual(
            YoozEngineRequestContext.maxUploadBytes,
            64 * 1024 * 1024
        )
    }

    /// A 3 MiB body — larger than Hummingbird's 2 MiB default but
    /// well under the 64 MiB ceiling — must reach the route handler
    /// rather than be rejected by the framework. We send invalid
    /// JSON on purpose so the handler returns 400 from its own
    /// catch block; the assertion is on the response message
    /// shape, which differs between "framework rejected the body"
    /// and "handler decoded the body and rejected the payload".
    @MainActor
    func testBatchAcceptsBodyLargerThanLegacyLimit() async throws {
        // Build a ~3 MiB payload of non-JSON bytes. The size is
        // chosen above the 2 MiB legacy ceiling and well under the
        // 64 MiB new ceiling.
        let payloadSize = 3 * 1024 * 1024
        let largeBody = Data(repeating: 0x41, count: payloadSize) // 'A'

        try await withServer { _ in
            let (http, body) = try await post(
                "/v1/stt/batch",
                body: largeBody
            )

            // The handler always returns 400 for a non-JSON body —
            // that is fine. What we are pinning is that the body
            // reached the handler at all (i.e. the framework did
            // not trip on the legacy 2 MiB ceiling).
            XCTAssertEqual(http.statusCode, 400)

            let decoded = try JSONDecoder().decode(
                ErrorResponse.self, from: body
            )
            XCTAssertEqual(decoded.code, "invalid_request")

            // The handler's message prefix proves we hit the
            // route's own catch block (it always starts with
            // "Invalid request body:"). Anything else means the
            // framework rejected the request before the handler
            // ran — which is exactly the regression we are
            // guarding against.
            XCTAssertTrue(
                decoded.error.hasPrefix("Invalid request body:"),
                "handler-side decode error expected; got: \(decoded.error)"
            )

            // Body must NOT carry the NIO "too many bytes" error
            // signature. If we ever drop the ceiling below 3 MiB
            // this assertion fires loudly.
            let lowered = decoded.error.lowercased()
            XCTAssertFalse(
                lowered.contains("too many bytes"),
                "body-size rejection regressed: \(decoded.error)"
            )
            XCTAssertFalse(
                lowered.contains("niotoomanybyteserror"),
                "body-size rejection regressed: \(decoded.error)"
            )
        }
    }
}
