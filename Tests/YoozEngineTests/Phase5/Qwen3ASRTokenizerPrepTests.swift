// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import STTModule
@testable import YoozEngine

/// Phase 5 — `Qwen3ASRTokenizerPrep` idempotency + missing-input
/// failure paths.
final class Qwen3ASRTokenizerPrepTests: XCTestCase {

    // Source the canonical checkpoint for the "prep against real
    // artifacts" test. When `/Volumes/S1` is unmounted (CI), the test
    // skips. The pure-failure tests run regardless because they only
    // need an empty temporary directory.
    private static var canonicalCheckpoint: URL {
        URL(
            fileURLWithPath:
                "/Volumes/S1/yooz/research/issue-12/models/hf_cache/hub/"
                + "models--mlx-community--Qwen3-ASR-1.7B-8bit/snapshots/"
                + "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
        )
    }

    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-prep-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true
        )
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        super.tearDown()
    }

    // MARK: - Failure paths

    /// Empty directory: no tokenizer.json, no fallback inputs ->
    /// fileNotFound on the first missing fallback artifact.
    func testRejectsEmptyDirectory() async {
        do {
            try await Qwen3ASRTokenizerPrep.prepare(modelDir: tempDir)
            XCTFail("expected fileNotFound, got success")
        } catch let error as Qwen3ASRError {
            if case .fileNotFound(let url) = error {
                XCTAssertTrue(
                    url.lastPathComponent == "tokenizer_config.json"
                        || url.lastPathComponent == "vocab.json"
                        || url.lastPathComponent == "merges.txt",
                    "fileNotFound carried unexpected URL: \(url.lastPathComponent)"
                )
            } else {
                XCTFail("expected fileNotFound, got \(error)")
            }
        } catch {
            XCTFail("expected Qwen3ASRError, got \(error)")
        }
    }

    /// Partial fallback inputs (vocab.json present, merges.txt and
    /// tokenizer_config.json absent) -> fileNotFound.
    func testRejectsPartialFallbackInputs() async throws {
        try Data("{}".utf8).write(
            to: tempDir.appendingPathComponent("vocab.json")
        )
        do {
            try await Qwen3ASRTokenizerPrep.prepare(modelDir: tempDir)
            XCTFail("expected fileNotFound, got success")
        } catch let error as Qwen3ASRError {
            if case .fileNotFound = error {
                // expected
            } else {
                XCTFail("expected fileNotFound, got \(error)")
            }
        }
    }

    // MARK: - Happy path against real artifacts

    /// Hardlink the canonical checkpoint into a fresh dir so we can
    /// safely add / remove / check sentinels without touching the
    /// shared cache.
    ///
    /// macOS TCC blocks the GUI xctest host from reading `/Volumes/S1`
    /// without an interactive prompt — and the prompt has nowhere to
    /// render under headless `xcodebuild test`, so reads HANG instead
    /// of failing. Phase 4 hit the same trap. We refuse to run the
    /// heavy tests under xcodebuild and require an explicit opt-in
    /// (`YOOZ_RUN_TCC_TESTS=1`); under `swift test` the suite's CLI
    /// binary is exempt from TCC and the tests run cleanly.
    private func cloneCanonicalCheckpoint() throws -> URL {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["YOOZ_RUN_TCC_TESTS"] == "1"
                || isLikelySwiftTestHost(),
            "Skipping /Volumes/S1-backed tokenizer prep test under "
                + "xcodebuild (macOS TCC). Run `swift test --filter "
                + "Qwen3ASRTokenizerPrepTests` to exercise this suite, or "
                + "set YOOZ_RUN_TCC_TESTS=1 after granting Full Disk Access "
                + "to the test host."
        )

        let source = Self.canonicalCheckpoint
        let configURL = source.appendingPathComponent("config.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: configURL.path),
            "Canonical Qwen3-ASR checkpoint not mounted at \(source.path)"
        )
        let dest = tempDir.appendingPathComponent("clone")
        try FileManager.default.createDirectory(
            at: dest, withIntermediateDirectories: true
        )
        let names = try FileManager.default.contentsOfDirectory(
            atPath: source.path
        )
        for name in names {
            let from = source.appendingPathComponent(name)
            let to = dest.appendingPathComponent(name)
            try FileManager.default.linkItem(at: from, to: to)
        }
        return dest
    }

    /// Heuristic: `swift test` invokes the test binary directly via
    /// the Swift toolchain. The test bundle path contains a recognizable
    /// SwiftPM build directory marker (`.build/`).
    private func isLikelySwiftTestHost() -> Bool {
        Bundle(for: Self.self).bundleURL.path.contains(".build/")
    }

    /// Run prep twice on the same directory; the second run must be a
    /// pure no-op (sentinel already present, no filesystem deltas).
    func testIdempotentSecondRun() async throws {
        let dir = try cloneCanonicalCheckpoint()

        try await Qwen3ASRTokenizerPrep.prepare(modelDir: dir)
        let sentinel = dir.appendingPathComponent(
            Qwen3ASRTokenizerPrep.sentinelFilename
        )
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: sentinel.path),
            "Sentinel must be created on first prep run"
        )

        // Snapshot mtime + names after first run.
        let attrs1 = try FileManager.default.attributesOfItem(
            atPath: sentinel.path
        )
        let names1 = Set(
            try FileManager.default.contentsOfDirectory(atPath: dir.path)
        )

        // Second run.
        try await Qwen3ASRTokenizerPrep.prepare(modelDir: dir)

        let attrs2 = try FileManager.default.attributesOfItem(
            atPath: sentinel.path
        )
        let names2 = Set(
            try FileManager.default.contentsOfDirectory(atPath: dir.path)
        )

        XCTAssertEqual(names1, names2, "Directory contents diverged")
        // mtime must be unchanged (no rewrite on second run).
        let m1 = (attrs1[.modificationDate] as? Date)
        let m2 = (attrs2[.modificationDate] as? Date)
        XCTAssertEqual(m1, m2, "Sentinel was rewritten on second run")
    }

    /// When `tokenizer.json` already exists, prep must validate it and
    /// drop the sentinel without touching the existing file.
    func testPreservesExistingTokenizerJSON() async throws {
        let dir = try cloneCanonicalCheckpoint()
        let tokenizer = dir.appendingPathComponent("tokenizer.json")
        let originalSize =
            (try FileManager.default.attributesOfItem(
                atPath: tokenizer.path
            )[.size] as? Int64) ?? 0

        try await Qwen3ASRTokenizerPrep.prepare(modelDir: dir)

        let afterSize =
            (try FileManager.default.attributesOfItem(
                atPath: tokenizer.path
            )[.size] as? Int64) ?? 0
        XCTAssertEqual(originalSize, afterSize, "tokenizer.json size changed")

        let sentinel = dir.appendingPathComponent(
            Qwen3ASRTokenizerPrep.sentinelFilename
        )
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: sentinel.path),
            "Sentinel must be present after prep"
        )
    }
}
