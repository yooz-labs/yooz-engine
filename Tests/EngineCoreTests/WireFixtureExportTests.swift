// WireFixtureExportTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

/// Captures v0.7.5-era wire JSON for the `EngineCore`-owned DTOs the #225
/// wire-type consolidation moves into `YoozEngineWire`.
///
/// Not a normal assertion test: gated behind `EXPORT_WIRE_FIXTURES=1`.
/// First run by hand against the pre-refactor tree (commit `a6614bc`) to
/// freeze `Tests/Fixtures/wire-v0.7.5/*.json`; `SessionBeginResponse` was
/// added post-move (the pre-refactor type was a server-private struct with
/// no reachable encoder) with the field layout verified against the
/// `a6614bc` definition. `Tests/YoozEngineWireTests/WireCompatFixtureTests`
/// reads the committed output and round-trips it; this generator does not
/// run in CI. Kept in the repo (rather than deleted after one run) so a
/// future DTO family migration can regenerate fixtures the same way — any
/// byte diff under regeneration is itself a wire-compat red flag.
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

        // `POST /v1/session/begin` body — produced from
        // `SessionCoordinator.begin()`'s result by both transports.
        try write(
            SessionBeginResponse(
                sessionId: "8f14e45f-ceea-467e-bd44-59c7f5c3c8f9",
                ts: "2026-07-02T09:00:00Z"
            ),
            "SessionBeginResponse", to: dir
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
