// LLMBundledModelResolverTests.swift
// YoozEngineInProcessTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Phase 2 (zero-download): the LLM load path resolves a locally-bundled snapshot
// before falling back to a Hugging Face fetch. The resolver lives in LLMModule;
// this test lives in the in-process target so it runs under the package scheme
// (PR CI). The end-to-end "bundled model loads with no network" is verified
// on-device. Pure file I/O — no MLX, no GPU.

import XCTest
@testable import LLMModule

final class LLMBundledModelResolverTests: XCTestCase {

    func testFirstModelDirectoryPicksFirstWithConfigSentinel() throws {
        let fileManager = FileManager.default
        let base = fileManager.temporaryDirectory
            .appendingPathComponent("llm-bundle-\(UUID().uuidString)")
        let missing = base.appendingPathComponent("no-config")
        let present = base.appendingPathComponent("has-config")
        try fileManager.createDirectory(at: missing, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: present, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: present.appendingPathComponent("config.json"))
        defer { try? fileManager.removeItem(at: base) }

        // A candidate without the config.json sentinel is skipped; the next wins.
        XCTAssertEqual(
            MLXLLMBackend.firstModelDirectory(containingConfigIn: [missing, present]),
            present
        )
        // No candidate carries the sentinel -> nil, so the caller falls back to HF.
        XCTAssertNil(
            MLXLLMBackend.firstModelDirectory(containingConfigIn: [missing])
        )
    }
}
