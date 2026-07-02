// ModelSelectionStoreTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Pins the persisted-selection contract engine#226 introduces:
/// `ModelSelectionStore` is the engine-owned source of truth for each
/// module's active model id, surviving a fresh actor instance (the
/// in-process stand-in for "surviving an engine restart") with no consumer
/// involvement.
final class ModelSelectionStoreTests: XCTestCase {
    /// A fresh temp-file path per test so runs never share state and never
    /// touch a developer's real `EngineConfig.stateDirectory`.
    private func makeTempFileURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("model-selection-tests-\(UUID().uuidString).json")
    }

    func testActiveIdIsNilBeforeAnyWrite() async {
        let store = ModelSelectionStore(fileURL: makeTempFileURL())
        let id = await store.activeId(for: "touchup")
        XCTAssertNil(id)
    }

    func testSetThenGetRoundTripsWithinOneInstance() async {
        let store = ModelSelectionStore(fileURL: makeTempFileURL())
        await store.setActiveId("yooz-quality-v2", for: "touchup")
        let id = await store.activeId(for: "touchup")
        XCTAssertEqual(id, "yooz-quality-v2")
    }

    /// The actual "survives a restart" contract: write via one actor
    /// instance, read back via a SECOND instance pointed at the same file —
    /// standing in for the engine process restarting while the disk file
    /// persists.
    func testSelectionSurvivesAFreshInstanceAtTheSameFileURL() async {
        let fileURL = makeTempFileURL()
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let firstRun = ModelSelectionStore(fileURL: fileURL)
        await firstRun.setActiveId("foundation-models", for: "touchup")

        let secondRun = ModelSelectionStore(fileURL: fileURL)
        let restored = await secondRun.activeId(for: "touchup")
        XCTAssertEqual(restored, "foundation-models")
    }

    func testModulesArePersistedIndependently() async {
        let fileURL = makeTempFileURL()
        let store = ModelSelectionStore(fileURL: fileURL)
        await store.setActiveId("yooz-light-v2", for: "touchup")
        await store.setActiveId("parakeet", for: "stt")

        let secondRun = ModelSelectionStore(fileURL: fileURL)
        let touchUp = await secondRun.activeId(for: "touchup")
        let stt = await secondRun.activeId(for: "stt")
        XCTAssertEqual(touchUp, "yooz-light-v2")
        XCTAssertEqual(stt, "parakeet")
    }

    /// Migration contract: a module with no stored entry (first run, or a
    /// module added after the file already existed) returns nil rather than
    /// throwing or crashing, so the caller's compiled-in default applies.
    func testUnknownModuleReturnsNilRatherThanThrowing() async {
        let fileURL = makeTempFileURL()
        let store = ModelSelectionStore(fileURL: fileURL)
        await store.setActiveId("yooz-light-v2", for: "touchup")

        let secondRun = ModelSelectionStore(fileURL: fileURL)
        let unknown = await secondRun.activeId(for: "some-future-module")
        XCTAssertNil(unknown)
    }

    /// A missing/corrupt file must degrade to "nothing persisted", not crash
    /// — the store is best-effort persistence, never a hard dependency for
    /// the engine to boot.
    func testCorruptFileDegradesToEmptyRatherThanThrowing() async throws {
        let fileURL = makeTempFileURL()
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try Data("not json".utf8).write(to: fileURL)
        defer { try? FileManager.default.removeItem(at: fileURL) }

        let store = ModelSelectionStore(fileURL: fileURL)
        let id = await store.activeId(for: "touchup")
        XCTAssertNil(id)

        // The store must still be writable after recovering from a corrupt
        // read — a bad file must not permanently wedge persistence.
        await store.setActiveId("yooz-light-v2", for: "touchup")
        let written = await store.activeId(for: "touchup")
        XCTAssertEqual(written, "yooz-light-v2")
    }

    func testOverwritingAnExistingSelectionReplacesIt() async {
        let store = ModelSelectionStore(fileURL: makeTempFileURL())
        await store.setActiveId("yooz-light-v2", for: "touchup")
        await store.setActiveId("yooz-quality-v2", for: "touchup")
        let id = await store.activeId(for: "touchup")
        XCTAssertEqual(id, "yooz-quality-v2")
    }
}
