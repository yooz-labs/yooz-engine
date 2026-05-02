// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN
import XCTest

#if canImport(YoozEngine)
@testable import YoozEngine
#elseif canImport(Qwen3ASR)
@testable import Qwen3ASR
#endif

/// Structural tests: weight-free, no S1 artifacts. These run on any
/// machine and protect the static math + parameter graph from
/// regressions.
final class Qwen3AudioEncoderStructuralTests: XCTestCase {

    func testFeatExtractOutputLengthMatchesReference() {
        // Ground-truth values pulled directly from the Python
        // reference's `_get_feat_extract_output_lengths`.
        XCTAssertEqual(Qwen3AudioEncoder.featExtractOutputLength(1), 1)
        XCTAssertEqual(Qwen3AudioEncoder.featExtractOutputLength(50), 7)
        XCTAssertEqual(Qwen3AudioEncoder.featExtractOutputLength(99), 13)
        XCTAssertEqual(
            Qwen3AudioEncoder.featExtractOutputLength(100), 13
        )
        XCTAssertEqual(
            Qwen3AudioEncoder.featExtractOutputLength(200), 26
        )
        XCTAssertEqual(
            Qwen3AudioEncoder.featExtractOutputLength(3000), 390
        )
    }

    func testFloorDivMatchesPythonSemantics() {
        XCTAssertEqual(Qwen3AudioEncoder.floorDiv(-1, 2), -1)
        XCTAssertEqual(Qwen3AudioEncoder.floorDiv(-3, 2), -2)
        XCTAssertEqual(Qwen3AudioEncoder.floorDiv(5, 2), 2)
        XCTAssertEqual(Qwen3AudioEncoder.floorDiv(-5, -2), 2)
        XCTAssertEqual(Qwen3AudioEncoder.floorDiv(5, -2), -3)
        XCTAssertEqual(Qwen3AudioEncoder.floorDiv(0, 1), 0)
    }

    func testEncoderShapesMatchCheckpoint() {
        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        let params = encoder.parameters().flattened()
        let lookup = Dictionary(uniqueKeysWithValues: params)

        let expectations: [(String, [Int])] = [
            ("conv2d1.weight", [480, 3, 3, 1]),
            ("conv2d1.bias", [480]),
            ("conv2d3.weight", [480, 3, 3, 480]),
            ("conv_out.weight", [1024, 7680]),
            ("ln_post.weight", [1024]),
            ("proj1.weight", [1024, 1024]),
            ("proj2.weight", [2048, 1024]),
            ("layers.0.self_attn.q_proj.weight", [1024, 1024]),
            ("layers.0.self_attn.q_proj.bias", [1024]),
            ("layers.0.fc1.weight", [4096, 1024]),
            ("layers.23.fc2.weight", [1024, 4096]),
            ("layers.23.final_layer_norm.weight", [1024]),
        ]
        for (name, shape) in expectations {
            guard let arr = lookup[name] else {
                XCTFail("missing parameter \(name)")
                continue
            }
            XCTAssertEqual(arr.shape, shape, "shape mismatch for \(name)")
        }
        XCTAssertEqual(encoder.layers.count, 24)
    }

    func testCheckedInitializerRejectsBadConfig() {
        var cfg = Qwen3ASRConfig()
        cfg.encoderAttentionHeads = 17
        XCTAssertThrowsError(try Qwen3AudioEncoder(checked: cfg)) {
            error in
            guard case Qwen3ASRError.invalidConfig = error else {
                XCTFail("expected invalidConfig, got \(error)")
                return
            }
        }
    }

    // MARK: - Forward error paths (no weights required)

    func testForwardRejectsWrongRank() {
        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        // Rank-2 input — encoder demands rank 3.
        let bad = MLXArray.zeros([128, 100], dtype: .float32)
        XCTAssertThrowsError(
            try encoder.forward(inputFeatures: bad)
        ) { error in
            guard case Qwen3ASRError.invalidInput(let msg) = error else {
                XCTFail("expected invalidInput, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("rank-2"))
        }
    }

    func testForwardRejectsZeroFrames() {
        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        let bad = MLXArray.zeros([1, 128, 0], dtype: .float32)
        XCTAssertThrowsError(
            try encoder.forward(inputFeatures: bad)
        ) { error in
            guard case Qwen3ASRError.invalidInput = error else {
                XCTFail("expected invalidInput, got \(error)")
                return
            }
        }
    }

    func testForwardRejectsWrongMelBinCount() {
        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        let bad = MLXArray.zeros([1, 64, 100], dtype: .float32)
        XCTAssertThrowsError(
            try encoder.forward(inputFeatures: bad)
        ) { error in
            guard case Qwen3ASRError.invalidInput(let msg) = error else {
                XCTFail("expected invalidInput, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("mel-bin"))
        }
    }

    func testForwardRejectsMismatchedAttentionMaskShape() {
        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        let feats = MLXArray.zeros([1, 128, 100], dtype: .float32)
        // Mask should be (1, 100); supply (1, 50) instead.
        let badMask = MLXArray(
            Array(repeating: Int32(1), count: 50)
        ).reshaped(1, 50)
        XCTAssertThrowsError(
            try encoder.forward(
                inputFeatures: feats,
                featureAttentionMask: badMask
            )
        ) { error in
            guard case Qwen3ASRError.invalidInput(let msg) = error else {
                XCTFail("expected invalidInput, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("featureAttentionMask"))
        }
    }

    func testForwardRejectsAllZeroAttentionMask() {
        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        let feats = MLXArray.zeros([1, 128, 100], dtype: .float32)
        let zeroMask = MLXArray.zeros([1, 100], dtype: .int32)
        XCTAssertThrowsError(
            try encoder.forward(
                inputFeatures: feats,
                featureAttentionMask: zeroMask
            )
        ) { error in
            guard case Qwen3ASRError.invalidInput(let msg) = error else {
                XCTFail("expected invalidInput, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("zero-length"))
        }
    }
}
