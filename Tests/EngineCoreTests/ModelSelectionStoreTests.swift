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

    /// The on-disk artifact is the migration/debug contract: a flat
    /// `{module: activeId}` JSON object. Pin it against the RAW file bytes
    /// (not just the actor API round trip) so a future refactor that
    /// changes the serialization silently — stranding every user's
    /// persisted selection on upgrade — fails here (PR #239 review).
    func testOnDiskFileIsFlatModuleToIdJSONObject() async throws {
        let fileURL = makeTempFileURL()
        defer { try? FileManager.default.removeItem(at: fileURL) }
        let store = ModelSelectionStore(fileURL: fileURL)
        await store.setActiveId("yooz-quality-v2", for: "touchup")
        await store.setActiveId("parakeet", for: "stt")

        let raw = try Data(contentsOf: fileURL)
        let decoded = try XCTUnwrap(
            JSONSerialization.jsonObject(with: raw) as? [String: String],
            "on-disk shape must be a flat {module: activeId} string map"
        )
        XCTAssertEqual(decoded, ["touchup": "yooz-quality-v2", "stt": "parakeet"])
    }

    /// A persist failure is best-effort but must be OBSERVABLE: it sets
    /// `lastPersistError`, and a subsequent successful persist clears it
    /// (PR #239 review). An unwritable directory path (a FILE occupying
    /// the parent-directory path) forces the failure deterministically.
    func testPersistFailureSetsLastPersistErrorAndSuccessClearsIt() async throws {
        // Parent path is a FILE, so createDirectory/write must fail.
        let blockingFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("model-selection-block-\(UUID().uuidString)")
        try Data("block".utf8).write(to: blockingFile)
        defer { try? FileManager.default.removeItem(at: blockingFile) }

        let failingStore = ModelSelectionStore(
            fileURL: blockingFile.appendingPathComponent("nested.json")
        )
        await failingStore.setActiveId("yooz-light-v2", for: "touchup")
        let failure = await failingStore.lastPersistError
        XCTAssertNotNil(failure, "a failed persist must record lastPersistError")

        // In-memory selection still took effect despite the disk failure.
        let inMemory = await failingStore.activeId(for: "touchup")
        XCTAssertEqual(inMemory, "yooz-light-v2")

        // A store with a writable path clears the error on success.
        let workingStore = ModelSelectionStore(fileURL: makeTempFileURL())
        await workingStore.setActiveId("yooz-light-v2", for: "touchup")
        let cleared = await workingStore.lastPersistError
        XCTAssertNil(cleared)
    }
}
