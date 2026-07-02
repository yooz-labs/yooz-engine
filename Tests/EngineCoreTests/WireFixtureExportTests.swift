// WireFixtureExportTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Captures v0.7.5-era wire JSON for the `EngineCore`-owned DTOs the #225
/// wire-type consolidation moves into `YoozEngineWire`.
///
/// Not a normal assertion test: gated behind `EXPORT_WIRE_FIXTURES=1` and
/// run once, by hand, against the pre-refactor tree to freeze
/// `Tests/Fixtures/wire-v0.7.5/*.json`. Every other test target's
/// decode-compat suite reads the committed output; this generator does not
/// run in CI. Kept in the repo (rather than deleted after one run) so a
/// future DTO family migration can regenerate fixtures the same way.
final class WireFixtureExportTests: XCTestCase {
    func testExportEngineCoreOwnedFixtures() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["EXPORT_WIRE_FIXTURES"] == "1",
            "set EXPORT_WIRE_FIXTURES=1 to (re)generate committed wire fixtures"
        )

        let dir = try fixturesDirectory()

        let modulesResponse = ModulesResponse(
            engineVersion: "0.7.5",
            buildVariant: "full",
            modules: [
                ModuleManifest(
                    name: "grammar",
                    version: "0.7.5",
                    loaded: true,
                    error: nil,
                    detail: ["rules_total": "1560"]
                ),
                ModuleManifest(
                    name: "stt",
                    version: "0.7.5",
                    loaded: false,
                    error: "not loaded",
                    detail: [:]
                ),
            ]
        )
        try write(modulesResponse, "ModulesResponse", to: dir)
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
