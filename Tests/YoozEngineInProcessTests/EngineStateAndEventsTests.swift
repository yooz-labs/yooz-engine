// EngineStateAndEventsTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import XCTest
import YoozEngineClient
@testable import LLMModule
@testable import YoozEngineInProcess

/// Engine-owned model selection (engine#226): `GET /v1/state`, `/v1/events`
/// (via `openEvents()`), and the non-blocking `setActiveModelAsync` +
/// persisted-selection contract, exercised through `InProcessTransport` —
/// the same endpoint-table handlers the loopback `APIServer` binds, so this
/// suite covers both transports' shared code path.
///
/// Every test here uses `preload: false` (or a fresh, ephemeral-store
/// `TouchUpEngine()` instance for the persistence-only assertions) so it
/// runs under a plain `swift test` with no MLX/metallib dependency —
/// matching `InProcessMemoryTests`' documented gating rationale. A real
/// preload's timing/eviction/progress-event behavior needs
/// `YOOZ_LLM_LOAD_MODELS=1` under the app-hosted xctest; see
/// `testSetActiveModelAsyncReturnsBeforePreloadCompletes` below.
final class EngineStateAndEventsTests: XCTestCase {
    private var llmLoadEnabled: Bool {
        ProcessInfo.processInfo.environment["YOOZ_LLM_LOAD_MODELS"] == "1"
    }

    // MARK: - GET /v1/state

    func testGetStateReturnsTouchUpModuleSnapshot() async throws {
        let transport = InProcessTransport()
        try await transport.connect()

        let data = try await transport.get("/v1/state")
        let snapshot = try JSONDecoder().decode(EngineStateSnapshot.self, from: data)

        let touchUp = try XCTUnwrap(
            snapshot.modules.first(where: { $0.module == "touchup" }),
            "GET /v1/state must include a touchup module snapshot"
        )
        XCTAssertEqual(
            touchUp.models.count, 3,
            "touchup snapshot must list every TouchUpModelSelection case (light/quality/foundation)"
        )
        XCTAssertTrue(
            touchUp.models.contains { $0.id == touchUp.activeId },
            "activeId must reference one of the listed rows"
        )
        XCTAssertEqual(
            touchUp.models.filter(\.isActive).count, 1,
            "exactly one row must be marked active, matching the picker invariant"
        )
    }

    // MARK: - /v1/events + async setModel

    /// The core engine#226 acceptance criterion: `POST /v1/touchup/model`
    /// publishes `modelChanged` to `/v1/events` subscribers, observable
    /// through `openEvents()` on the SAME transport the request went
    /// through. Uses `preload: false` so no MLX load is triggered — the
    /// event-publish half of `setActiveModelAsync` runs synchronously
    /// before the (here, skipped) background preload Task is even spawned.
    func testSetModelPublishesModelChangedEvent() async throws {
        let transport = InProcessTransport()
        try await transport.connect()

        let stream = try await transport.openEvents()

        let body = try JSONEncoder().encode(
            TouchUpSetModelRequest(id: "yooz-quality-v3", preload: false)
        )
        _ = try await transport.post("/v1/touchup/model", body: body)

        // The bus has no replay; a concurrently-running test in the same
        // process could interleave unrelated events on this shared
        // singleton, so scan (bounded) for the one this test caused rather
        // than asserting the very next event.
        let matched = try await firstMatchingEvent(stream, timeoutSeconds: 5) {
            $0.kind == .modelChanged && $0.module == "touchup" && $0.modelId == "yooz-quality-v3"
        }
        XCTAssertNotNil(matched, "expected a modelChanged event for yooz-quality-v3")

        // Leave the shared engine in a known state for any other in-process
        // test relying on the .yoozLight default.
        _ = try await transport.post(
            "/v1/touchup/model",
            body: try JSONEncoder().encode(TouchUpSetModelRequest(id: "yooz-light-v3", preload: false))
        )
    }

    /// `setActiveModelAsync` returns the picker row IMMEDIATELY — the
    /// non-blocking contract engine#226 requires (`POST
    /// /v1/<module>/model` never blocks on a download). With `preload:
    /// false` there is no background load to race against, so this pins
    /// the baseline: the call must not itself introduce any blocking
    /// behavior even before a real load is layered on.
    func testSetModelAsyncReturnsQuicklyWithoutPreload() async throws {
        let engine = TouchUpEngine()
        let start = Date()
        let info = try await engine.setActiveModelAsync(.yoozQuality, preload: false)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(info.id, TouchUpModelSelection.yoozQuality.rawValue)
        XCTAssertTrue(info.isActive)
        XCTAssertLessThan(elapsed, 1.0, "setActiveModelAsync(preload: false) must return promptly")
    }

    /// Real preload timing: gated behind `YOOZ_LLM_LOAD_MODELS=1` (app-hosted
    /// xctest, `default.metallib` present) — mirrors `InProcessMemoryTests`.
    /// Proves the call returns before the model finishes loading (the
    /// non-blocking contract with an ACTUAL download/load in flight), and
    /// that a terminal `loadStateChanged` event eventually arrives.
    func testSetActiveModelAsyncReturnsBeforePreloadCompletes() async throws {
        try XCTSkipUnless(
            llmLoadEnabled,
            "Set YOOZ_LLM_LOAD_MODELS=1 (app-hosted xctest) to exercise the real preload timing."
        )
        let engine = TouchUpEngine()
        let start = Date()
        let info = try await engine.setActiveModelAsync(.yoozLight, preload: true)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertEqual(info.id, TouchUpModelSelection.yoozLight.rawValue)
        // A synchronous preload of an uncached model takes multiple seconds
        // at minimum (network + weight materialization); returning under 1s
        // proves the call did not await it.
        XCTAssertLessThan(elapsed, 1.0, "setActiveModelAsync(preload: true) must not block on the load")

        // Poll for the eventual loaded state via the picker, bounded so a
        // genuine regression fails the test instead of hanging CI.
        var loaded = false
        for _ in 0..<200 {
            let models = await engine.availableModels()
            if models.first(where: { $0.id == TouchUpModelSelection.yoozLight.rawValue })?.loadState == .loaded {
                loaded = true
                break
            }
            try await Task.sleep(for: .milliseconds(300))
        }
        XCTAssertTrue(loaded, "background preload must eventually reach .loaded")
    }

    /// Rapid picker double-switch (engine#226; PR #239 review): with tier
    /// A's background preload still in flight, switching to tier B must
    /// end with B active and RESIDENT — A's late completion must not evict
    /// B (the `activeModel == selection` + `Task.isCancelled` guards in
    /// `preloadActiveSelectionInBackground`). Needs two real MLX loads to
    /// exercise authentically, so gated like the other live-load tests.
    func testRapidDoubleSwitchKeepsTheSecondTierResident() async throws {
        try XCTSkipUnless(
            llmLoadEnabled,
            "Set YOOZ_LLM_LOAD_MODELS=1 (app-hosted xctest) to exercise the double-switch eviction guard."
        )
        let engine = TouchUpEngine()

        // Kick A's background load and immediately switch to B while A is
        // (almost certainly) still loading.
        _ = try await engine.setActiveModelAsync(.yoozQuality, preload: true)
        _ = try await engine.setActiveModelAsync(.yoozLight, preload: true)

        // Wait for both dust clouds to settle: B must reach .loaded.
        var lightLoaded = false
        for _ in 0..<400 {
            let models = await engine.availableModels()
            if models.first(where: { $0.id == TouchUpModelSelection.yoozLight.rawValue })?.loadState == .loaded {
                lightLoaded = true
                break
            }
            try await Task.sleep(for: .milliseconds(300))
        }
        XCTAssertTrue(lightLoaded, "the second (current) tier must finish loading")

        // Give A's superseded dispatch time to complete + (incorrectly)
        // evict, if the guard were broken.
        try await Task.sleep(for: .seconds(2))

        let active = await engine.activeModel
        XCTAssertEqual(active, .yoozLight, "the second switch must win")
        let models = await engine.availableModels()
        XCTAssertEqual(
            models.first(where: { $0.id == TouchUpModelSelection.yoozLight.rawValue })?.loadState,
            .loaded,
            "A's late completion must not evict the tier the user switched to"
        )
    }

    // MARK: - Persisted selection (engine#226)

    /// The headline acceptance criterion: the active model survives a
    /// "restart" (a fresh actor instance) with no consumer involvement.
    /// Uses two `TouchUpEngine` instances sharing one injected
    /// `ModelSelectionStore` — the in-process stand-in for the real engine
    /// process restarting while `EngineConfig.stateDirectory` persists.
    func testActiveModelSurvivesAFreshEngineInstance() async throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("touchup-selection-survive-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let store = ModelSelectionStore(fileURL: fileURL)

        let firstRun = TouchUpEngine(selectionStore: store)
        _ = try await firstRun.setActiveModelAsync(.yoozQuality, preload: false)

        let secondRun = TouchUpEngine(selectionStore: store)
        let active = await secondRun.activeModel
        XCTAssertEqual(
            active, .yoozQuality,
            "a fresh TouchUpEngine sharing the same persisted-selection store must restore the prior selection"
        )

        let models = await secondRun.availableModels()
        XCTAssertEqual(
            models.first(where: { $0.isActive })?.id,
            TouchUpModelSelection.yoozQuality.rawValue
        )
    }

    /// A persisted id written by a pre-#282 build (`yooz-quality-v2`) no
    /// longer parses as a `TouchUpModelSelection`. The restore must fall
    /// back to the compiled-in default (logging the mismatch) rather than
    /// crash or silently adopt an unknown id (PR #283 review).
    func testLegacyPersistedIdFallsBackToCompiledDefault() async throws {
        let fileURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("touchup-selection-legacy-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let store = ModelSelectionStore(fileURL: fileURL)

        let firstRun = TouchUpEngine(selectionStore: store)
        _ = try await firstRun.setActiveModelAsync(.yoozQuality, preload: false)

        // Rewrite the persisted id to the retired v2 wire id, exactly as a
        // pre-upgrade engine build would have left it on disk.
        let json = try String(contentsOf: fileURL, encoding: .utf8)
            .replacingOccurrences(of: "yooz-quality-v3", with: "yooz-quality-v2")
        try json.write(to: fileURL, atomically: true, encoding: .utf8)

        // A fresh store over the same file: the first store instance holds
        // the pre-rewrite id in memory, and a real engine restart would
        // re-read from disk.
        let restartStore = ModelSelectionStore(fileURL: fileURL)
        let secondRun = TouchUpEngine(selectionStore: restartStore)
        let active = await secondRun.activeModel
        XCTAssertEqual(
            active, .yoozLight,
            "an unparsable persisted id must fall back to the compiled-in default"
        )
    }

    /// Without a persisted selection, a fresh instance still defaults to
    /// `.yoozLight` — restoring must never invent a selection out of thin
    /// air, only recall a genuinely persisted one.
    func testFreshEngineWithNoPersistedSelectionKeepsCompiledDefault() async {
        let engine = TouchUpEngine()
        let active = await engine.activeModel
        XCTAssertEqual(active, .yoozLight)
    }

    // MARK: - Helpers

    /// Scan an `AsyncStream` for the first event matching `predicate`,
    /// bounded by `timeoutSeconds` so a missing event fails the test
    /// instead of hanging it forever. Races stream consumption against a
    /// timer `Task` in one `TaskGroup`; whichever finishes first wins, and
    /// the loser is cancelled. Owns its iterator locally (rather than
    /// taking one `inout`) so it can be captured by the escaping task-group
    /// closure — an `inout` parameter cannot be.
    private func firstMatchingEvent(
        _ stream: AsyncStream<EngineEvent>,
        timeoutSeconds: Double,
        where predicate: @escaping @Sendable (EngineEvent) -> Bool
    ) async throws -> EngineEvent? {
        let box = IteratorBox(stream.makeAsyncIterator())
        let deadline = ContinuousClock.now.advanced(by: .seconds(timeoutSeconds))
        while ContinuousClock.now < deadline {
            let remaining = deadline - ContinuousClock.now
            guard let event = await raceAgainstTimeout(
                timeout: remaining, operation: { await box.next() }
            ) else {
                return nil  // timed out waiting for the next frame
            }
            guard let event else { continue }  // stream yielded nil (ended); keep looping until deadline
            if predicate(event) { return event }
        }
        return nil
    }

    /// Reference-type wrapper so a value-type `AsyncIterator` can be shared
    /// across suspension points inside an escaping closure without an
    /// `inout` capture.
    private actor IteratorBox {
        private var iterator: AsyncStream<EngineEvent>.AsyncIterator
        init(_ iterator: AsyncStream<EngineEvent>.AsyncIterator) { self.iterator = iterator }
        func next() async -> EngineEvent? {
            // Materialize a local copy to mutate across the `await` —
            // mutating the actor-isolated stored property directly across a
            // suspension point is disallowed.
            var local = iterator
            let result = await local.next()
            iterator = local
            return result
        }
    }

    /// Run `operation`, returning its result, or `nil` if `timeout` elapses
    /// first. `Result` is itself `EngineEvent?` here (the stream's element
    /// type), so the return type is the same optional-of-optional either
    /// helper naturally produces; callers unwrap once for "timed out" and
    /// once for "stream ended".
    private func raceAgainstTimeout<Result: Sendable>(
        timeout: Duration, operation: @escaping @Sendable () async -> Result
    ) async -> Result? {
        await withTaskGroup(of: Optional<Result>.self) { group in
            group.addTask { Optional(await operation()) }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }
            defer { group.cancelAll() }
            return await group.next() ?? nil
        }
    }
}
