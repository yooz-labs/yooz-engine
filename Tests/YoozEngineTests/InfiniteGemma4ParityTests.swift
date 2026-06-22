// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import InfiniteModule

/// Live verification for issue #184: the `mlx-swift-lm` fork pinned in
/// `project.yml` (yooz-labs/mlx-swift-lm) loads and greedy-generates Gemma4 at
/// native context, including the E4B OptiQ build via the #186 fix. mlx-swift
/// and mlx-python share the same MLX C++/Metal core, so argmax decoding on
/// identical quantized weights is expected to match Python `mlx-lm==0.31.3` (the
/// version Infinite validated on).
///
/// This is the evidence that justifies flipping
/// `InfiniteModelSelection.swiftRuntimeSupported` to `true` for the Gemma4 rows:
///
/// - **26B-A4B** (`gemma4`, full tier) loads + generates a correct answer. It is
///   a reasoning model, so whether the `<|channel>thought` preamble is shown is a
///   chat-template difference from Python, not a model-numerics one; the
///   assertion is therefore a correct on-task answer, not exact token parity.
/// - **E4B** (`gemma4`, reduced tier) uses the stricter exact-greedy-parity
///   assertion: its OptiQ-4bit build loads since the #186 fork fix
///   (`per_layer_model_projection` made quantizable + KV-shared layers'
///   projections made `has_kv`-conditional), and its greedy output matches the
///   Python reference token-for-token.
///
/// Tiered like the other heavy suites (KVCompression / Qwen3ASR): gated by
/// `INFINITE_LIVE=1` AND the weights being present in the HF cache, so CI — which
/// has neither — skips cleanly instead of triggering a multi-GB download.
///
/// Run locally via the dedicated scheme (which sets `INFINITE_LIVE=1` through a
/// test plan — xcodebuild does not propagate CLI env to the hosted test runner):
///
///     xcodebuild -project YoozEngine.xcodeproj -scheme InfiniteLive \
///       -skipMacroValidation -derivedDataPath build \
///       -destination 'platform=macOS' test
///
/// Reference fixtures live in `YoozEngine/Infinite/results/` and are produced by
/// `scripts/gemma4_parity_reference.py`. See `results/PARITY.md`.
final class InfiniteGemma4ParityTests: XCTestCase {

    private enum ParityMode {
        /// Decoded text must equal the Python reference exactly (non-reasoning).
        case exactGreedy
        /// Decoded text must contain a correct on-task answer (reasoning model
        /// whose thinking-channel template differs from Python's).
        case containsAnswer(String)
    }

    private struct Reference: Decodable {
        let modelRepo: String
        let promptText: String
        let maxTokens: Int
        let generatedText: String

        enum CodingKeys: String, CodingKey {
            case modelRepo = "model_repo"
            case promptText = "prompt_text"
            case maxTokens = "max_tokens"
            case generatedText = "generated_text"
        }
    }

    /// Full-tier 26B-A4B MoE: loads + generates the correct answer.
    func testGemma4_26B_A4B_LoadsAndGeneratesNativeContext() async throws {
        try XCTSkipUnless(
            InfiniteRAMTier.current == .full,
            "gemma4-26b-a4b needs a full (64 GiB) RAM tier; current tier is "
                + "\(InfiniteRAMTier.current.rawValue)."
        )
        try await runGenerate(
            .gemma4_26B_A4B1M,
            referenceFile: "gemma4_26b-a4b_parity_reference.json",
            mode: .containsAnswer("2, 3, 5, 7, 11")
        )
    }

    /// Reduced-tier E4B: exact greedy parity vs Python. Non-reasoning, short
    /// completion, so token-exact greedy is the strong faithfulness signal.
    /// Loadable since #186 made the per-layer-input projection quantizable.
    func testGemma4E4BGreedyParityVsPython() async throws {
        try await runGenerate(
            .gemma4E4B1M,
            referenceFile: "gemma4_e4b_parity_reference.json",
            mode: .exactGreedy
        )
    }

    /// Dense 12B (`gemma4_unified`, #187): exact greedy parity vs the mlx-vlm
    /// reference. mlx-vlm and the Swift engine share the same HF tokenizer_config
    /// chat template (including the `<|channel>thought` preamble), so token-exact
    /// greedy is the right assertion — and it exercises the K-eq-V value path
    /// fixed in this issue (values from the raw k_proj output, pre-k_norm/RoPE).
    /// Reduced-tier model, so it runs on reduced (32 GiB) or full hardware.
    func testGemma4_12B_GreedyParityVsPython() async throws {
        try XCTSkipUnless(
            InfiniteRAMTier.current.supports(required: .reduced),
            "gemma4-12b needs at least a reduced (32 GiB) RAM tier; current tier "
                + "is \(InfiniteRAMTier.current.rawValue)."
        )
        try await runGenerate(
            .gemma4_12B1M,
            referenceFile: "gemma4_12b_parity_reference.json",
            mode: .exactGreedy
        )
    }

    private func runGenerate(
        _ selection: InfiniteModelSelection,
        referenceFile: String,
        mode: ParityMode
    ) async throws {
        try Self.skipUnlessLive()
        let reference = try Self.loadReference(referenceFile)
        try Self.skipUnlessModelCached(reference.modelRepo)

        // Exact parity must use the reference's token budget; the contains-answer
        // mode gives generous headroom so a reasoning-style chat template still
        // reaches the answer even though the Python reference is length-capped.
        let maxTokens: Int
        switch mode {
        case .exactGreedy: maxTokens = reference.maxTokens
        case .containsAnswer: maxTokens = max(reference.maxTokens, 256)
        }

        let backend = try await MLXInfiniteBackend.load(selection.descriptor)
        let result = try await backend.generate(
            context: "",
            prompt: reference.promptText,
            maxTokens: maxTokens,
            nativeContextTokens: selection.nativeContextTokens,
            temperature: 0.0  // greedy / argmax — deterministic
        )

        // The engine path actually ran: real tokens, measured decode rate,
        // and a typed stop reason (not a transport-error fallback).
        XCTAssertGreaterThan(result.tokenCount, 0, "no tokens generated")
        XCTAssertGreaterThan(
            result.decodeTokensPerSecond, 0, "decode throughput not measured"
        )
        XCTAssertTrue(
            ["stop", "length"].contains(result.finishReason),
            "unexpected finishReason \(result.finishReason)"
        )
        print(
            "[InfiniteGemma4ParityTests] \(selection.rawValue): "
                + String(format: "%.1f", result.decodeTokensPerSecond)
                + " tok/s decode, \(result.tokenCount) tokens, "
                + "finish=\(result.finishReason)"
        )

        let got = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch mode {
        case .exactGreedy:
            // `generatedText` from the fixture is only consumed here (exact
            // parity); the contains-answer mode ignores it by design.
            let want = reference.generatedText
                .trimmingCharacters(in: .whitespacesAndNewlines)
            XCTAssertEqual(
                got, want,
                """
                Swift greedy output diverged from the Python mlx-lm reference \
                for \(selection.rawValue).
                want: \(want)
                got:  \(got)
                """
            )
        case let .containsAnswer(answer):
            XCTAssertTrue(
                got.contains(answer),
                "expected \(selection.rawValue) output to contain \(answer); "
                    + "got: \(got)"
            )
        }
    }

    // MARK: - Gates

    private static func skipUnlessLive(
        file: StaticString = #file, line: UInt = #line
    ) throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["INFINITE_LIVE"] == "1",
            "Set INFINITE_LIVE=1 to run the live gemma4 test "
                + "(loads multi-GB weights).",
            file: file, line: line
        )
    }

    private static func skipUnlessModelCached(
        _ repo: String, file: StaticString = #file, line: UInt = #line
    ) throws {
        let dirName = "models--"
            + repo.replacingOccurrences(of: "/", with: "--")
        let cached = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/\(dirName)")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: cached.path),
            "Weights for \(repo) are not in the HF cache; skipping rather than "
                + "triggering a multi-GB download.",
            file: file, line: line
        )
    }

    private static func loadReference(
        _ fileName: String, file: StaticString = #file, line: UInt = #line
    ) throws -> Reference {
        let dir: URL
        if let override = ProcessInfo.processInfo
            .environment["INFINITE_PARITY_DIR"] {
            dir = URL(fileURLWithPath: override)
        } else {
            // <repo>/Tests/YoozEngineTests/<thisfile> -> <repo>
            dir = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()  // YoozEngineTests
                .deletingLastPathComponent()  // Tests
                .deletingLastPathComponent()  // repo root
                .appendingPathComponent("YoozEngine/Infinite/results")
        }
        let url = dir.appendingPathComponent(fileName)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: url.path),
            "Parity reference \(url.path) missing; regenerate via "
                + "scripts/gemma4_parity_reference.py.",
            file: file, line: line
        )
        return try JSONDecoder().decode(
            Reference.self, from: Data(contentsOf: url)
        )
    }
}
