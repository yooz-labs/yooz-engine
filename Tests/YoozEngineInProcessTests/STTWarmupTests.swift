// STTWarmupTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Coverage for engine#252 (PR #255 review): proactive STT warmup at XPC
// service startup. All tests run under plain `swift test` — none touch
// MLX/Metal (the cache-miss path in `runWarmup` returns before any model
// load; the coalescing test exercises `enqueueLoad`'s dedup with a cheap
// counting closure, not a real `ParakeetModel` load).

import EngineCore
import Foundation
import XCTest
import YoozEngineClient

@testable import STTModule
@testable import YoozEngineInProcess

final class STTWarmupTests: XCTestCase {

    // MARK: - Finding I3: warmup language selection order

    func testResolveWarmupLanguagePrefersPersistedSelection() async throws {
        let store = try makeIsolatedSelectionStore()
        await store.setActiveId("es", for: "stt")
        let language = await YoozSTTEngine.resolveWarmupLanguage(selectionStore: store)
        XCTAssertEqual(language, .spanish)
    }

    func testResolveWarmupLanguageFallsBackToDefaultWhenStoreEmpty() async throws {
        let store = try makeIsolatedSelectionStore()
        let language = await YoozSTTEngine.resolveWarmupLanguage(selectionStore: store)
        XCTAssertEqual(language, EngineConfig.defaultSTTLanguage)
    }

    func testResolveWarmupLanguageFallsBackWhenPersistedLanguageIsNotImplemented() async throws {
        let store = try makeIsolatedSelectionStore()
        // Chinese ("zh") parses via `STTLanguage(rawValue:)` but is not
        // `isImplemented` (CJK has no MLX mirror) — must fall through to
        // the default, not resolve to an unusable language.
        await store.setActiveId("zh", for: "stt")
        let language = await YoozSTTEngine.resolveWarmupLanguage(selectionStore: store)
        XCTAssertEqual(language, EngineConfig.defaultSTTLanguage)
    }

    func testResolveWarmupLanguageFallsBackOnUnparseableStoredValue() async throws {
        let store = try makeIsolatedSelectionStore()
        await store.setActiveId("not-a-language-code", for: "stt")
        let language = await YoozSTTEngine.resolveWarmupLanguage(selectionStore: store)
        XCTAssertEqual(language, EngineConfig.defaultSTTLanguage)
    }

    private func makeIsolatedSelectionStore() throws -> ModelSelectionStore {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("stt-warmup-selection-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? fm.removeItem(at: dir) }
        return ModelSelectionStore(fileURL: dir.appendingPathComponent("selection.json"))
    }

    // MARK: - Finding C1 + item 6: warmup never downloads; a skip leaves state pristine

    /// A cache miss must be a silent no-op: no download (an HF fetch with
    /// `HF_HOME` redirected to an empty, freshly-created directory would
    /// need real network access to succeed — this test asserts the load
    /// never even reaches that point by asserting the engine stays
    /// unloaded), and no `loadState`/`lastLoadError` mutation visible to a
    /// `/v1/stt/status` poll on an otherwise-untouched, freshly-spawned
    /// service (finding 6).
    @MainActor
    func testWarmupSkipsSilentlyWhenWeightsNotCached() async throws {
        let engine = YoozSTTEngine.shared
        // Reset any prior state so this test starts from a known slate —
        // the singleton may carry state from sibling tests (mirrors
        // AsyncLoadEndpointsTests' LLM equivalent).
        engine.stop()
        XCTAssertFalse(engine.isRunning, "precondition: engine must be unloaded before this test")
        XCTAssertEqual(engine.loadState, .idle, "precondition")

        let fm = FileManager.default
        let emptyHome = fm.temporaryDirectory.appendingPathComponent("stt-warmup-empty-hf-\(UUID().uuidString)")
        try fm.createDirectory(at: emptyHome.appendingPathComponent("hub"), withIntermediateDirectories: true)
        let savedHFHome = ProcessInfo.processInfo.environment["HF_HOME"]
        let savedHFHubCache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"]
        setenv("HF_HOME", emptyHome.path, 1)
        setenv("HF_HUB_CACHE", emptyHome.appendingPathComponent("hub").path, 1)
        addTeardownBlock {
            if let savedHFHome { setenv("HF_HOME", savedHFHome, 1) } else { unsetenv("HF_HOME") }
            if let savedHFHubCache { setenv("HF_HUB_CACHE", savedHFHubCache, 1) } else { unsetenv("HF_HUB_CACHE") }
            try? fm.removeItem(at: emptyHome)
        }

        engine.warmupIfNeeded(language: .english)
        // Bounded wait: the cache-miss path returns almost immediately (a
        // filesystem check, no network), so a short deadline is enough —
        // if it ever takes longer than this, something is wrong (e.g. it
        // is not actually skipping and is instead attempting a real fetch).
        await engine.awaitWarmupIfNeeded(deadlineSeconds: 10)

        XCTAssertFalse(engine.isRunning, "a cache miss must never load a model")
        XCTAssertEqual(engine.loadState, .idle, "a skipped warmup must leave loadState untouched")
        XCTAssertNil(engine.lastLoadError, "a skipped warmup must not surface as a failure")
    }

    // MARK: - Finding I5: enqueueLoad coalescing (the mechanism handleBatch/openSTTStream/warmup all share)

    /// Two concurrent `enqueueLoad` calls for the same language must
    /// return the same in-flight `Task` — the underlying `body` runs once
    /// and both callers observe the same result. Mirrors
    /// `AsyncLoadEndpointsTests.testLLMEnqueueLoadIsIdempotentAcrossConcurrentCallers`
    /// for STT. Uses a cheap counting closure instead of a real model load
    /// (no MLX/Metal needed), matching how that LLM test avoids a real
    /// weights fetch too — this tests the real, unmocked `enqueueLoad`
    /// dedup mechanism, not a fake stand-in for it.
    @MainActor
    func testSTTEnqueueLoadIsIdempotentAcrossConcurrentCallers() async throws {
        let engine = YoozSTTEngine.shared
        engine.stop()

        let counter = Counter()
        async let task1 = engine.enqueueLoad(language: .english) {
            await counter.increment()
            try? await Task.sleep(for: .milliseconds(50))
        }
        async let task2 = engine.enqueueLoad(language: .english) {
            await counter.increment()
        }
        let (t1, t2) = await (task1, task2)

        XCTAssertTrue(t1 == t2, "concurrent enqueueLoad calls for the same language must dedup to one Task")
        _ = try await t1.value
        let calls = await counter.value
        XCTAssertEqual(calls, 1, "the body closure must run exactly once across both callers")
    }

    private actor Counter {
        private(set) var value = 0
        func increment() { value += 1 }
    }
}

// MARK: - Finding I4: LoadDeadlineExceeded -> typed YoozEngineError mapping

final class LoadDeadlineTypedErrorTests: XCTestCase {

    /// `awaitLoadOrTypedDeadline` must convert a `LoadDeadlineExceeded`
    /// timeout into a `YoozEngineError.serverError` — the raw struct is
    /// not a `YoozEngineError`, so `XPCErrorBridge` would otherwise
    /// collapse it to `.engineNotReachable` client-side (PR #255 review,
    /// finding I4), making a slow load look like a dead service.
    func testDeadlineExceededMapsToTypedServerError() async throws {
        let transport = InProcessTransport()
        let neverFinishes = Task<Void, Error> {
            try await Task.sleep(for: .seconds(30))
        }
        do {
            try await transport.awaitLoadOrTypedDeadline(neverFinishes, deadlineSeconds: 0.05)
            XCTFail("expected a load_deadline_exceeded error")
        } catch let error as YoozEngineError {
            guard case .serverError(let statusCode, let code, _) = error else {
                XCTFail("expected .serverError, got \(error)")
                return
            }
            XCTAssertEqual(statusCode, 504)
            XCTAssertEqual(code, "load_deadline_exceeded")
        }
    }

}

// The `XPCErrorBridge` round trip itself is tested in
// `Tests/YoozEngineClientTests/XPCErrorBridgeDeadlineTests.swift` — that
// type lives in `YoozEngineClient`, a module this test target does not
// depend on. This file pins the conversion site
// (`InProcessTransport.awaitLoadOrTypedDeadline`); that file pins the
// bridge's handling of the resulting `.serverError` case.
