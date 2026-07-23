// EngineStateWireFixtureExportTests.swift
// YoozEngineWireTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import YoozEngineWire

/// Captures wire JSON for the engine#226 DTOs (`EngineEvent`,
/// `EngineModelSnapshotRow`, `EngineModuleSnapshot`, `EngineStateSnapshot`).
///
/// Unlike `EngineCoreTests/WireFixtureExportTests` and
/// `YoozEngineClientTests/WireFixtureExportTests` (which captured PRE-#225
/// shapes from types that were moving house), these types are brand new —
/// there is no pre-refactor shape to preserve, so this generator just
/// freezes the current encoder output directly from the `YoozEngineWire`
/// definitions. Gated behind `EXPORT_WIRE_FIXTURES=1`, same convention;
/// does not run in CI. `WireCompatFixtureTests` reads the committed output
/// and round-trips it — a wire id, field name, or JSON shape change here
/// breaks that test, which is the point.
final class EngineStateWireFixtureExportTests: XCTestCase {
    func testExportEngineStateFixtures() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["EXPORT_WIRE_FIXTURES"] == "1",
            "set EXPORT_WIRE_FIXTURES=1 to (re)generate committed wire fixtures"
        )

        let dir = try fixturesDirectory()

        try write(
            EngineEvent(
                kind: .downloadProgress,
                module: "touchup",
                modelId: "yooz-quality-v3",
                loadState: nil,
                progress: 0.42,
                message: nil,
                ts: "2026-07-02T09:00:00Z"
            ),
            "EngineEvent", to: dir
        )

        let row = EngineModelSnapshotRow(
            id: "yooz-light-v3", displayName: "Yooz-Light",
            description: "Fast, on-device cleanup", tier: .light,
            sizeBytes: 276_000_000, loadState: .loaded, isActive: true
        )
        try write(row, "EngineModelSnapshotRow", to: dir)

        let moduleSnapshot = EngineModuleSnapshot(
            module: "touchup", models: [row], activeId: "yooz-light-v3"
        )
        try write(moduleSnapshot, "EngineModuleSnapshot", to: dir)

        try write(
            EngineStateSnapshot(modules: [moduleSnapshot]),
            "EngineStateSnapshot", to: dir
        )
    }

    private func fixturesDirectory() throws -> URL {
        let thisFile = URL(fileURLWithPath: #filePath)
        let dir = thisFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/wire-v0.7.5")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func write<T: Encodable>(_ value: T, _ name: String, to dir: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let data = try encoder.encode(value)
        try data.write(to: dir.appendingPathComponent("\(name).json"))
    }
}
