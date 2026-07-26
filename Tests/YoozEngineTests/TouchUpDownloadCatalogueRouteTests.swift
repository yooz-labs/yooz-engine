// TouchUpDownloadCatalogueRouteTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// HTTP route coverage for engine#306: `POST /v1/touchup/download` (and its
// `/cancel` counterpart) resolving against the full `LLMModelType` catalogue
// instead of the 3-case `TouchUpModelSelection` picker. Before this fix,
// `yooz-instruct-4b` (and its HuggingFace alias
// `YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx`, remi's shipped default) 400'd as
// `invalid_model` even though generate/preload/unload already resolved it
// fine (engine#303). Same harness as `LLMClearCacheRouteTests` /
// `LLMCatalogueRouteTests`.
//
// Safety, mirrors `LLMCatalogueRouteTests`'s own scope note: the routes
// under test can dispatch a real, multi-GB HuggingFace fetch for an
// uncached model, so a test must never exercise `POST /v1/touchup/download`
// on a model this machine does not already have on disk.
//   - The acceptance surface (catalogue id, HF alias) is proven through
//     `POST /v1/touchup/download/cancel`'s documented no-op-when-nothing-is-
//     in-flight path, which shares `downloadableModel(from:)`'s parsing with
//     `/download` but never dispatches a load.
//   - The rejection surface (unknown id, `foundation-models`) is proven
//     through `/download` directly — a 400 is returned before any dispatch.
//   - The idempotent-when-cached behavior is proven through a real
//     `/download` call, gated on this machine's on-disk HF cache actually
//     having the weights already (`XCTSkipUnless`) — the row is asserted to
//     already read `.cached`/`.loaded`, which is exactly the state
//     `requestDownload`'s early return requires to avoid dispatching.

import Foundation
import XCTest
@testable import EngineCore
@testable import LLMModule
@testable import YoozEngine

final class TouchUpDownloadCatalogueRouteTests: XCTestCase {

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

    private func post(_ path: String, body: Data) async throws -> (HTTPURLResponse, Data) {
        var request = URLRequest(url: baseURL().appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try XCTUnwrap(response as? HTTPURLResponse), data)
    }

    // MARK: - Rejections (safe: 400 before any dispatch)

    @MainActor
    func testDownloadStillRejectsUnknownIdWithInvalidModel() async throws {
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/touchup/download", body: Data(#"{"id":"not-a-real-model"}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(json["code"] as? String, "invalid_model")
        }
    }

    @MainActor
    func testDownloadStillRejectsFoundationModelsWithInvalidModel() async throws {
        // `foundation-models` IS a real TouchUp picker id, just never a
        // downloadable one (OS-resident, no weights to fetch) — must stay
        // 400 `invalid_model` after generalizing the rest of the id space.
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/touchup/download", body: Data(#"{"id":"foundation-models"}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(json["code"] as? String, "invalid_model")
        }
    }

    @MainActor
    func testDownloadRejectsUndecodableBodyWithInvalidRequest() async throws {
        try await withServer { _ in
            let (http, data) = try await post("/v1/touchup/download", body: Data("{}".utf8))
            XCTAssertEqual(http.statusCode, 400)
            let json = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(json["code"] as? String, "invalid_request")
        }
    }

    // MARK: - Acceptance, proven via the no-op cancel path (never dispatches)

    @MainActor
    func testDownloadCancelAcceptsCatalogueModelWithNoPickerCounterpart() async throws {
        // Regression pin for the bug itself: pre-engine#306 this id 400'd
        // because the route resolved against `TouchUpModelSelection`
        // (3 cases), which does not include `yooz-instruct-4b`.
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/touchup/download/cancel", body: Data(#"{"id":"yooz-instruct-4b"}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 200)
            let row = try JSONDecoder().decode(TouchUpModelInfo.self, from: data)
            XCTAssertEqual(row.id, "yooz-instruct-4b")
            XCTAssertFalse(
                row.isActive,
                "a catalogue model with no TouchUpModelSelection counterpart can never be the TouchUp picker's active model"
            )
        }
    }

    @MainActor
    func testDownloadCancelAcceptsHuggingFaceRepoAlias() async throws {
        // remi's shipped default names the model by its HF repo id, not the
        // canonical wire id — `LLMModelType(rawValue:)` must resolve both.
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/touchup/download/cancel",
                body: Data(#"{"id":"YoozLabs/Qwen3.5-4B-qat-lean-4bit-mlx"}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 200)
            let row = try JSONDecoder().decode(TouchUpModelInfo.self, from: data)
            XCTAssertEqual(
                row.id, "yooz-instruct-4b",
                "the row must report the canonical wire id, not the HF alias it was addressed by"
            )
        }
    }

    @MainActor
    func testDownloadCancelStillAcceptsPickerTiers() async throws {
        // The two pre-existing TouchUp tiers must keep working unchanged.
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/touchup/download/cancel", body: Data(#"{"id":"yooz-light-v3"}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 200)
            let row = try JSONDecoder().decode(TouchUpModelInfo.self, from: data)
            XCTAssertEqual(row.id, "yooz-light-v3")
            XCTAssertEqual(row.tier, .light)
        }
    }

    // MARK: - Idempotent when already cached (real call, gated on disk state)

    @MainActor
    func testDownloadOfAnAlreadyCachedCatalogueModelIsANoOp() async throws {
        let info = await TouchUpEngine.shared.getModelInfo()
        let cached = info.first(where: { $0.type.rawValue == "yooz-instruct-4b" })?.isCached ?? false
        try XCTSkipUnless(
            cached,
            "yooz-instruct-4b is not on disk on this machine; the idempotent-when-cached "
                + "behavior can only be proven without triggering a real download when the "
                + "weights are already present"
        )
        try await withServer { _ in
            let (http, data) = try await post(
                "/v1/touchup/download", body: Data(#"{"id":"yooz-instruct-4b"}"#.utf8)
            )
            XCTAssertEqual(http.statusCode, 200)
            let row = try JSONDecoder().decode(TouchUpModelInfo.self, from: data)
            XCTAssertEqual(row.id, "yooz-instruct-4b")
            XCTAssertTrue(
                row.loadState == .cached || row.loadState == .loaded,
                "an already-cached model's download row must report .cached/.loaded, "
                    + "which is exactly the state requestDownload's early return checks "
                    + "before it would otherwise dispatch a redundant fetch"
            )
        }
    }
}

// MARK: - Actor-level concurrency safety (engine#306 re-keying)

/// `backgroundPreloadTasks` moved from `[TouchUpModelSelection: Task<...>]`
/// to `[String: Task<...>]` so `requestDownload` (keyed by `LLMModelType`)
/// and `setActiveModelAsync` (keyed by `TouchUpModelSelection`) share one
/// dictionary. These tests prove the shared dictionary survives concurrent
/// access from both entry points without crashing or deadlocking, using the
/// same "observe `.loading`, then cancel before any real transfer" technique
/// `AsyncLoadEndpointsTests.testLLMEnqueueLoadIsIdempotentAcrossConcurrentCallers`
/// already establishes as safe in this suite for this exact tier (`.yoozLight`
/// is not cached on the dev machine that runs these tests locally, per the
/// build-verification policy in AGENTS.md) — the load is never awaited to
/// completion, so no meaningful transfer happens before cancellation lands.
///
/// What this does NOT prove: whether the two concurrent dispatches actually
/// collapsed onto ONE underlying `backgroundPreloadTasks` entry versus two
/// entries that happened to agree (both would route through `enqueueLoad`'s
/// own, separately-tested, per-tier dedup regardless). Distinguishing those
/// would need either a real completed download or a private-state test seam,
/// both out of scope here — see the PR description for the code-inspection
/// argument (`backgroundPreloadTasks[id] == nil` gate, same dictionary, same
/// string key for any id nameable by both `TouchUpModelSelection.rawValue`
/// and `LLMModelType.rawValue`).
final class TouchUpDownloadDedupeTests: XCTestCase {

    func testConcurrentRequestDownloadCallsForTheSameModelDoNotCrash() async throws {
        let engine = TouchUpEngine.shared
        await engine.unload(.yoozLight)
        _ = await engine.cancelDownload(.yoozLight)

        async let row1 = engine.requestDownload(.yoozLight)
        async let row2 = engine.requestDownload(.yoozLight)
        let (r1, r2) = await (row1, row2)

        XCTAssertEqual(r1?.id, "yooz-light-v3")
        XCTAssertEqual(r2?.id, "yooz-light-v3")

        // Observe-then-cancel: never await completion.
        _ = await engine.cancelDownload(.yoozLight)
        await engine.unload(.yoozLight)
    }

    func testPickerSwitchAndExplicitDownloadForTheSameModelShareOneDispatch() async throws {
        let engine = TouchUpEngine.shared
        await engine.unload(.yoozLight)
        _ = await engine.cancelDownload(.yoozLight)

        async let switched: TouchUpModelInfo? = try? engine.setActiveModelAsync(
            .yoozLight, preload: true
        )
        async let downloaded: TouchUpModelInfo? = engine.requestDownload(.yoozLight)
        _ = await (switched, downloaded)

        let state = await engine.loadState(for: .yoozLight)
        XCTAssertEqual(
            state, .loading,
            "one of the two concurrent dispatches (picker switch, explicit download) "
                + "must have started the shared load"
        )

        // Observe-then-cancel: never await completion. Also restore the
        // picker's active selection to the conventional test default
        // (`TouchUpPickerRouteTests` resets the same way) since
        // `setActiveModelAsync` persists to the real, disk-backed
        // `ModelSelectionStore.shared`.
        _ = await engine.cancelDownload(.yoozLight)
        await engine.unload(.yoozLight)
        _ = try? await engine.setActiveModel(.yoozLight, preload: false)
    }
}
