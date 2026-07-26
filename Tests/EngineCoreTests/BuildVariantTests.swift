// BuildVariantTests.swift
// EngineCoreTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import XCTest
@testable import EngineCore

final class BuildVariantTests: XCTestCase {

    /// Unit-test bundles run under xctest's host bundle, which never carries
    /// the `YoozBuildVariant` Info.plist key we set on the three app targets.
    /// Absent-key behaviour must therefore fall back to `.full` so client
    /// code has a sane default when running against an unannotated bundle.
    func testCurrentFallsBackToFullInTestBundle() {
        XCTAssertEqual(BuildVariant.current, .full)
    }

    /// `resolved(from:)` is the pure seam exercised directly. We construct
    /// a throwaway bundle wrapping a temporary directory with an Info.plist
    /// carrying each known variant value, plus an unknown value to verify
    /// the fallback path. No mocking — real Bundle, real disk.
    func testResolvedFromBundleReadsInfoPlistKey() throws {
        for raw in ["full", "whisper", "lite", "llm"] {
            let bundle = try makeBundle(withVariant: raw)
            XCTAssertEqual(
                BuildVariant.resolved(from: bundle).rawValue,
                raw,
                "bundle annotated with YoozBuildVariant=\(raw) should resolve"
            )
        }
    }

    func testResolvedFallsBackWhenKeyMissing() throws {
        let bundle = try makeBundle(withVariant: nil)
        XCTAssertEqual(BuildVariant.resolved(from: bundle), .full)
    }

    func testResolvedFallsBackWhenKeyUnknown() throws {
        let bundle = try makeBundle(withVariant: "made-up-variant")
        XCTAssertEqual(BuildVariant.resolved(from: bundle), .full)
    }

    func testRawValueRoundTrip() {
        XCTAssertEqual(BuildVariant(rawValue: "full"), .full)
        XCTAssertEqual(BuildVariant(rawValue: "whisper"), .whisper)
        XCTAssertEqual(BuildVariant(rawValue: "lite"), .lite)
        XCTAssertEqual(BuildVariant(rawValue: "llm"), .llm)
        XCTAssertNil(BuildVariant(rawValue: "made-up"))
    }

    func testAllCasesCodable() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        for variant in [BuildVariant.full, .whisper, .lite, .llm] {
            let data = try encoder.encode(variant)
            let decoded = try decoder.decode(BuildVariant.self, from: data)
            XCTAssertEqual(decoded, variant)
        }
    }

    /// Documentary manifest of which modules each build variant is expected
    /// to ship with. This is the contract that `project.yml` targets encode
    /// via their `dependencies:` lists; keeping it in code (rather than only
    /// in prose) gives us one place to update when a new variant is added
    /// (Notes, Voice) or a module moves (e.g. VAD is locally embedded in
    /// whisper per the A1 design, §4 exception). EngineCore can't import the
    /// module targets, so this stays a string manifest — no `canImport` here.
    ///
    /// Lite drops both MLX STT and VAD; it ships only Apple STT, Grammar,
    /// and LLM, per the Phase 5 epic (`Apple STT + Lite Variant` section).
    /// LLM (engine#297) drops Apple STT and Grammar on top of that — it
    /// ships generation/classification only, nothing speech- or
    /// grammar-related.
    func testExpectedModulesPerVariant() {
        let expected: [BuildVariant: Set<String>] = [
            .full: ["stt", "apple_stt", "grammar", "llm", "vad"],
            .whisper: ["stt", "apple_stt", "grammar", "llm"],  // VAD embedded in whisper client
            .lite: ["apple_stt", "grammar", "llm"],  // no MLX, no VAD
            .llm: ["llm"]  // no speech stack, no grammar
        ]
        // Whisper's module set is a strict subset of full.
        XCTAssertTrue(expected[.whisper]!.isSubset(of: expected[.full]!))
        XCTAssertEqual(
            expected[.full]!.subtracting(expected[.whisper]!),
            ["vad"],
            "Whisper variant drops only VAD today; future slim variants will drop more."
        )
        // Lite is a strict subset of whisper (drops MLX STT on top of VAD).
        XCTAssertTrue(expected[.lite]!.isSubset(of: expected[.whisper]!))
        XCTAssertEqual(
            expected[.whisper]!.subtracting(expected[.lite]!),
            ["stt"],
            "Lite drops the MLX STT module — Apple STT is the only speech backend."
        )
        // LLM is a strict subset of lite (drops Apple STT + Grammar on top).
        XCTAssertTrue(expected[.llm]!.isSubset(of: expected[.lite]!))
        XCTAssertEqual(
            expected[.lite]!.subtracting(expected[.llm]!),
            ["apple_stt", "grammar"],
            "LLM drops Apple STT and Grammar — generation/classification only."
        )
    }

    // MARK: - Helpers

    /// Build a real on-disk bundle whose Info.plist optionally carries the
    /// `YoozBuildVariant` key. The directory is created under the test's
    /// temporary directory so it survives for the test's lifetime; XCTest
    /// tears down `tempDir` automatically.
    private func makeBundle(withVariant variant: String?) throws -> Bundle {
        let tempDir = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: URL(fileURLWithPath: NSTemporaryDirectory()),
            create: true
        )
        let bundleURL = tempDir.appendingPathComponent("variant.bundle")
        try FileManager.default.createDirectory(at: bundleURL, withIntermediateDirectories: true)

        // `CFBundlePackageType = BNDL` + `CFBundleIdentifier` is the minimum
        // Bundle(url:) accepts across macOS versions — see advisor note.
        var plist: [String: Any] = [
            "CFBundleIdentifier": "live.yooz.engine.test.variant",
            "CFBundlePackageType": "BNDL",
        ]
        if let variant = variant {
            plist[BuildVariant.infoPlistKey] = variant
        }

        let plistData = try PropertyListSerialization.data(
            fromPropertyList: plist,
            format: .xml,
            options: 0
        )
        try plistData.write(to: bundleURL.appendingPathComponent("Info.plist"))

        guard let bundle = Bundle(url: bundleURL) else {
            throw NSError(
                domain: "BuildVariantTests",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "failed to load synthesised bundle at \(bundleURL)"]
            )
        }
        return bundle
    }
}
