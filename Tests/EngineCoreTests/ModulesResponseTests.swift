// ModulesResponseTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

final class ModulesResponseTests: XCTestCase {

    // MARK: - Fixtures

    /// Small deterministic module stubs. Each concrete type fixes its own
    /// `static var name` (Swift protocol statics can't be per-instance), so
    /// tests use one type per logical module rather than a generic fixture.
    /// Real module-specific behavior lives in each module's own test target.
    struct GrammarFixture: AIModule {
        static var name: String { "grammar" }
        var isReady: Bool { true }
        func healthCheck() async -> ModuleHealth {
            ModuleHealth(loaded: true, error: nil, detail: ["rules_total": "1560"])
        }
    }

    struct SttFixture: AIModule {
        static var name: String { "stt" }
        var isReady: Bool { false }
        func healthCheck() async -> ModuleHealth {
            ModuleHealth(
                loaded: false,
                error: "not loaded",
                detail: ["language": "", "streaming": "false"]
            )
        }
    }

    // MARK: - Codable round-trip

    func testCodableRoundTrip() throws {
        let original = ModulesResponse(
            engineVersion: "0.6.0",
            buildVariant: "full",
            modules: [
                ModuleManifest(
                    name: "grammar",
                    version: "0.6.0",
                    loaded: true,
                    error: nil,
                    detail: ["rules_total": "1560", "library_version": "0.10.0"]
                ),
                ModuleManifest(
                    name: "stt",
                    version: "0.6.0",
                    loaded: false,
                    error: "not loaded",
                    detail: [:]
                )
            ]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(ModulesResponse.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    // MARK: - Empty module list

    func testEmptyModuleList() async {
        let response = await ModulesResponse.build(
            from: [],
            engineVersion: "0.6.0",
            buildVariant: "full"
        )
        XCTAssertEqual(response.engineVersion, "0.6.0")
        XCTAssertEqual(response.buildVariant, "full")
        XCTAssertTrue(response.modules.isEmpty)

        // Empty list still serializes cleanly.
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try? encoder.encode(response)
        XCTAssertNotNil(data)
        let json = data.flatMap { String(data: $0, encoding: .utf8) }
        XCTAssertEqual(
            json,
            #"{"buildVariant":"full","engineVersion":"0.6.0","modules":[]}"#
        )
    }

    // MARK: - Build from modules

    func testBuildFromModulesPopulatesManifests() async {
        // Registry returns modules sorted by name, so pass them that order
        // (grammar before stt) to match real server behavior.
        let modules: [any AIModule] = [GrammarFixture(), SttFixture()]
        let response = await ModulesResponse.build(
            from: modules,
            engineVersion: "0.6.0",
            buildVariant: "full"
        )
        XCTAssertEqual(response.modules.count, 2)
        XCTAssertEqual(response.modules[0].name, "grammar")
        XCTAssertEqual(response.modules[0].version, "0.6.0")
        XCTAssertTrue(response.modules[0].loaded)
        XCTAssertNil(response.modules[0].error)
        XCTAssertEqual(response.modules[0].detail["rules_total"], "1560")

        XCTAssertEqual(response.modules[1].name, "stt")
        XCTAssertFalse(response.modules[1].loaded)
        XCTAssertEqual(response.modules[1].error, "not loaded")
    }

    // MARK: - Deterministic key order

    func testSortedKeysEncodingStableAcrossRuns() throws {
        // Dictionary iteration order is unspecified; `.sortedKeys` forces a
        // deterministic byte-for-byte output. This is load-bearing for
        // clients that cache/compare the response body.
        let manifest = ModuleManifest(
            name: "grammar",
            version: "0.6.0",
            loaded: true,
            error: nil,
            detail: [
                "z_last": "z",
                "a_first": "a",
                "m_middle": "m"
            ]
        )
        let response = ModulesResponse(
            engineVersion: "0.6.0",
            buildVariant: "full",
            modules: [manifest]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys

        // Same struct encoded twice must produce byte-identical output.
        let first = try encoder.encode(response)
        let second = try encoder.encode(response)
        XCTAssertEqual(first, second)

        // Field names and detail keys must appear in sorted order.
        guard let json = String(data: first, encoding: .utf8) else {
            XCTFail("non-UTF8 encoded output"); return
        }
        let aIdx = json.range(of: "a_first")
        let mIdx = json.range(of: "m_middle")
        let zIdx = json.range(of: "z_last")
        XCTAssertNotNil(aIdx); XCTAssertNotNil(mIdx); XCTAssertNotNil(zIdx)
        XCTAssertLessThan(aIdx!.lowerBound, mIdx!.lowerBound)
        XCTAssertLessThan(mIdx!.lowerBound, zIdx!.lowerBound)

        // Top-level `buildVariant` sorts before `engineVersion` sorts before
        // `modules`. Verifies struct field names also got sorted.
        let bvIdx = json.range(of: "\"buildVariant\"")!
        let evIdx = json.range(of: "\"engineVersion\"")!
        let modIdx = json.range(of: "\"modules\"")!
        XCTAssertLessThan(bvIdx.lowerBound, evIdx.lowerBound)
        XCTAssertLessThan(evIdx.lowerBound, modIdx.lowerBound)
    }

    // MARK: - Null error omission

    func testErrorFieldEncodedAsNullWhenAbsent() throws {
        let response = ModulesResponse(
            engineVersion: "0.6.0",
            buildVariant: "full",
            modules: [ModuleManifest(
                name: "grammar",
                version: "0.6.0",
                loaded: true,
                error: nil,
                detail: [:]
            )]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = try encoder.encode(response)
        // Swift's default JSONEncoder omits nil optionals rather than emitting
        // `"error":null`. Clients treat key-absence as "no error"; keep the
        // wire shape aligned with that idiom.
        let json = String(data: data, encoding: .utf8) ?? ""
        XCTAssertFalse(json.contains("\"error\""),
                       "expected error key to be absent when nil; got: \(json)")
        // Round-trip still works: decoder fills nil back.
        let decoded = try JSONDecoder().decode(ModulesResponse.self, from: data)
        XCTAssertNil(decoded.modules.first?.error)
    }
}
