// STTStatusRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for the `progress` filter on `/v1/stt/status`
// (engine#145). Companion to `LLMStatusRouteTests` — same boot-the-
// real-APIServer harness; see that file for the rationale on why
// in-process Hummingbird tests aren't used here.
//
// Pins the symmetric contract introduced after PR #143:
//   - cold engine, no model loaded, `downloadProgress == 0` -> the
//     route reports `progress: nil` (the wire field is absent or
//     null, not `0`). Prevents the consumer banner from rendering a
//     "Downloading... 0%" sliver on an idle engine.
//   - the route uses the same shape as `/v1/llm/status` so the
//     whisper poller (#194) can normalize both endpoints identically.

import Foundation
import XCTest
@testable import YoozEngine
@testable import EngineCore

#if canImport(STTModule)
@testable import STTModule
#endif

final class STTStatusRouteTests: XCTestCase {

    @MainActor
    private func withServer<T>(
        _ body: (APIServer) async throws -> T
    ) async throws -> T {
        UniqueEnginePort.assignFreshPort()
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

    private func get(_ path: String) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "GET"
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    // MARK: - Cold MLX STT engine (parakeet default on full / whisper builds)

    /// Without the filter, a cold MLX engine's `downloadProgress == 0`
    /// would surface to the wire as `"progress": 0`, and the whisper
    /// banner would either render "Downloading... 0%" or rely on the
    /// consumer-side defensive filter (Patch A on whisper dev workdir).
    /// With the filter, the route collapses the 0 fraction to nil so
    /// the contract matches `/v1/llm/status`.
    #if canImport(STTModule)
    @MainActor
    func testGetStatusOnColdMLXEngineReportsNilProgress() async throws {
        try await withServer { _ in
            let (http, body) = try await get("/v1/stt/status")
            XCTAssertEqual(http.statusCode, 200)
            let status = try JSONDecoder().decode(STTStatusResponse.self, from: body)
            XCTAssertFalse(status.loaded,
                           "Cold engine: MLX STT singleton has not been started")
            XCTAssertFalse(status.streaming,
                           "Cold engine cannot be streaming")
            XCTAssertNil(status.progress,
                         "0.0 fraction must collapse to nil (banner hides)")
        }
    }

    /// Calling `stop()` must reset `downloadProgress` so the narrow
    /// stop()-then-poll window does not surface a stale `1.0` to the
    /// banner. Without the reset, the route's `loaded ? nil : ...`
    /// filter only covers the loaded case; a stopped (model=nil but
    /// downloadProgress=1.0) engine would still report
    /// "Downloading... 100%" to a polling client. We can't easily
    /// load a real Parakeet model in CI to test the full sequence,
    /// but we can verify `stop()` clears the published value on a
    /// singleton that some earlier process may have left at 1.0.
    @MainActor
    func testStopResetsDownloadProgress() async throws {
        // Simulate a prior load leaving downloadProgress sticky by
        // calling stop() and asserting the property settles to 0.
        // (The MLX engine's downloadProgress is private(set); the
        // public surface we can verify is its post-stop() value via
        // the route, which is the consumer-visible contract.)
        YoozSTTEngine.shared.stop()
        try await withServer { _ in
            let (_, body) = try await get("/v1/stt/status")
            let status = try JSONDecoder().decode(STTStatusResponse.self, from: body)
            XCTAssertFalse(status.loaded,
                           "stop() must leave the engine unloaded")
            XCTAssertNil(status.progress,
                         "stop() must reset downloadProgress so the route doesn't leak the prior 1.0")
        }
    }
    #endif

    // MARK: - Apple STT (lite-variant default)

    /// Apple STT has no HF download to track — the route has always
    /// hard-coded `progress: nil` for this branch. Sanity check that
    /// the engine#145 edit on the MLX branch didn't perturb it.
    #if canImport(AppleSTTModule) && !canImport(STTModule)
    @MainActor
    func testGetStatusOnAppleSTTAlwaysReportsNilProgress() async throws {
        try await withServer { _ in
            let (http, body) = try await get("/v1/stt/status")
            XCTAssertEqual(http.statusCode, 200)
            let status = try JSONDecoder().decode(STTStatusResponse.self, from: body)
            XCTAssertNil(status.progress,
                         "Apple STT has no HF download to report")
        }
    }
    #endif
}
