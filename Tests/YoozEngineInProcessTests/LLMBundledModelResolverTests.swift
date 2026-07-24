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

    func testFirstModelDirectoryRequiresConfigAndWeights() throws {
        let fileManager = FileManager.default
        let base = fileManager.temporaryDirectory
            .appendingPathComponent("llm-bundle-\(UUID().uuidString)")
        let missing = base.appendingPathComponent("empty")             // nothing
        let partial = base.appendingPathComponent("config-only")       // config, no weights
        let complete = base.appendingPathComponent("config-and-weights")
        for dir in [missing, partial, complete] {
            try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        try Data("{}".utf8).write(to: partial.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: complete.appendingPathComponent("config.json"))
        try Data("w".utf8).write(to: complete.appendingPathComponent("model.safetensors"))
        defer { try? fileManager.removeItem(at: base) }

        // Skips the empty dir AND the config-only (partial) dir; picks the
        // complete snapshot (config.json + .safetensors).
        XCTAssertEqual(
            MLXLLMBackend.firstModelDirectory(containingConfigIn: [missing, partial, complete]),
            complete
        )
        // A partial bundle (config.json but no weights) is NOT selected, so the
        // caller falls through to HF instead of hard-failing the load.
        XCTAssertNil(
            MLXLLMBackend.firstModelDirectory(containingConfigIn: [missing, partial])
        )
    }

    // MARK: - Host-app candidates for a nested XPC service (engine#284)

    func testNestedServiceBundleYieldsHostAppResourceCandidates() {
        let serviceBundle = URL(
            fileURLWithPath:
            "/Applications/Yooz Whisper.app/Contents/XPCServices/YoozWhisperXPC.xpc"
        )
        let candidates = MLXLLMBackend.hostAppResourceCandidates(
            forNestedServiceAt: serviceBundle, id: "yooz-light-v3"
        )
        XCTAssertEqual(candidates.map(\.path), [
            "/Applications/Yooz Whisper.app/Contents/Resources/Models/yooz-light-v3",
            "/Applications/Yooz Whisper.app/Contents/Resources/yooz-light-v3",
        ])
    }

    func testNonNestedBundleYieldsNoHostAppCandidates() {
        // In-process build: Bundle.main is the host app itself.
        let appBundle = URL(fileURLWithPath: "/Applications/Yooz Whisper.app")
        XCTAssertTrue(
            MLXLLMBackend.hostAppResourceCandidates(
                forNestedServiceAt: appBundle, id: "yooz-light-v3"
            ).isEmpty
        )
        // "XPCServices" appearing elsewhere in the path must not match:
        // only the exact <app>/Contents/XPCServices/<service> nesting counts.
        let lookalike = URL(
            fileURLWithPath: "/Users/dev/XPCServices/Contents/NotAService.xpc"
        )
        XCTAssertTrue(
            MLXLLMBackend.hostAppResourceCandidates(
                forNestedServiceAt: lookalike, id: "yooz-light-v3"
            ).isEmpty
        )
    }
}
