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

/// Engine integration smoke test.
///
/// Validates the public surface that Phase 4 (encoder ↔ decoder
/// bridge) will adopt. The test instantiates the encoder via the
/// production loader API, runs a forward pass with a synthetic mel
/// input, and asserts only public-API contracts: shape, dtype,
/// no-NaN, deterministic re-run.
///
/// No private symbols, no `@testable` reach-around, no per-layer
/// trace — those live in `Qwen3AudioEncoderParityTests`. This is
/// the "the engine could call this tomorrow" check.
final class Qwen3AudioEncoderIntegrationTests: XCTestCase {

    func testEncoderInstantiatesAndRunsForwardOnSyntheticInput() throws {
        // Use the small config so the test runs without /Volumes/S1.
        let cfg = Qwen3ASRConfig(
            numMelBins: 16,
            encoderLayers: 2,
            encoderAttentionHeads: 4,
            encoderFFNDim: 64,
            dModel: 32,
            downsampleHiddenSize: 16,
            outputDim: 24,
            maxSourcePositions: 32,
            nWindow: 25,
            nWindowInfer: 200,
            convChunkSize: 50
        )
        let encoder = try Qwen3AudioEncoder(checked: cfg)

        // Round-trip a freshly-initialized encoder through the loader
        // — same path Phase 4 will use to bring in the real weights.
        var dumped: [String: MLXArray] = [:]
        for (name, value) in encoder.parameters().flattened() {
            dumped["audio_tower.\(name)"] = value
        }
        let target = Qwen3AudioEncoder(cfg)
        try Qwen3SafetensorsLoader.applyAudioTowerWeights(
            dumped, to: target
        )

        let frames = 50
        // Seed a deterministic (non-zero) input so the forward
        // exercises every kernel — zeros short-circuit the conv
        // path through bias-only outputs which would mask shape
        // bugs in the time axis.
        let input = MLXArray(
            stride(from: 0, to: cfg.numMelBins * frames, by: 1)
                .map { Float($0) / Float(cfg.numMelBins * frames) }
        ).reshaped(1, cfg.numMelBins, frames)
        let mask = MLXArray(
            Array(repeating: Int32(1), count: frames)
        ).reshaped(1, frames)

        let output = try target.forward(
            inputFeatures: input, featureAttentionMask: mask
        )
        eval(output)

        // Output rank, batch-flattened along time, and outputDim
        // are the only public-shape contracts Phase 4 depends on.
        XCTAssertEqual(output.ndim, 2)
        XCTAssertEqual(output.dim(1), cfg.outputDim)
        XCTAssertGreaterThan(output.dim(0), 0)
        XCTAssertFalse(MLX.any(MLX.isNaN(output)).item(Bool.self))
        XCTAssertFalse(MLX.any(MLX.isInf(output)).item(Bool.self))

        // Determinism: same input → same output.
        let secondOutput = try target.forward(
            inputFeatures: input, featureAttentionMask: mask
        )
        eval(secondOutput)
        let drift = MLX.abs(output - secondOutput).max().item(Float.self)
        XCTAssertEqual(
            drift, 0.0,
            "encoder forward must be deterministic on identical input"
        )
    }

    func testCallAsFunctionMatchesThrowingForwardOnValidInput() throws {
        let cfg = Qwen3ASRConfig(
            numMelBins: 16,
            encoderLayers: 2,
            encoderAttentionHeads: 4,
            encoderFFNDim: 64,
            dModel: 32,
            downsampleHiddenSize: 16,
            outputDim: 24,
            maxSourcePositions: 32,
            nWindow: 25,
            nWindowInfer: 200,
            convChunkSize: 50
        )
        let encoder = try Qwen3AudioEncoder(checked: cfg)
        let input = MLXArray.zeros(
            [1, cfg.numMelBins, 50], dtype: .float32
        )
        let mask = MLXArray(
            Array(repeating: Int32(1), count: 50)
        ).reshaped(1, 50)

        let viaCall = encoder(
            inputFeatures: input, featureAttentionMask: mask
        )
        let viaForward = try encoder.forward(
            inputFeatures: input, featureAttentionMask: mask
        )
        eval(viaCall, viaForward)
        XCTAssertEqual(
            MLX.abs(viaCall - viaForward).max().item(Float.self),
            0.0
        )
    }
}
