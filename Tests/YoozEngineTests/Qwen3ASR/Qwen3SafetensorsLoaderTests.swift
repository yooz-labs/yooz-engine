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

/// Loader tests: weight round-trip + error-path coverage.
///
/// Round-trip is the strongest check that the loader's parameter
/// graph is wired correctly — synthesize random weights, save, load,
/// and confirm the forward output is identical.
///
/// Error-path tests confirm typed `Qwen3ASRError` cases are produced
/// for every realistic failure mode the engine may encounter at
/// runtime: missing file, missing tensor, shape mismatch, dtype
/// mismatch, malformed safetensors header, wrong-architecture
/// checkpoint.
final class Qwen3SafetensorsLoaderTests: XCTestCase {

    // MARK: - Helpers

    /// Small synthetic config that fits in memory in a few MB. Used
    /// throughout this suite so the tests run on any machine.
    private static let smallConfig = Qwen3ASRConfig(
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

    private func makeTempURL(prefix: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(
            "\(prefix)-\(UUID().uuidString).safetensors"
        )
    }

    private func dumpEncoderToURL(
        _ encoder: Qwen3AudioEncoder, url: URL
    ) throws {
        var dumped: [String: MLXArray] = [:]
        for (name, value) in encoder.parameters().flattened() {
            dumped["audio_tower.\(name)"] = value
        }
        try MLX.save(arrays: dumped, url: url)
    }

    // MARK: - Round-trip

    func testRoundTripSynthesizedWeightsProducesIdenticalForward()
        throws
    {
        let cfg = Self.smallConfig
        let original = Qwen3AudioEncoder(cfg)
        eval(original)  // materialize random init at fixed values

        let tmpURL = makeTempURL(prefix: "qwen3-roundtrip")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try dumpEncoderToURL(original, url: tmpURL)

        let restored = Qwen3AudioEncoder(cfg)
        try Qwen3SafetensorsLoader.loadAudioTower(
            from: tmpURL, into: restored
        )

        let testInput = MLXArray.zeros(
            [1, cfg.numMelBins, 50], dtype: .float32
        )
        let mask = MLXArray(
            Array(repeating: Int32(1), count: 50)
        ).reshaped(1, 50)
        let lhs = original(
            inputFeatures: testInput, featureAttentionMask: mask
        )
        let rhs = restored(
            inputFeatures: testInput, featureAttentionMask: mask
        )
        eval(lhs, rhs)
        let delta = MLX.abs(lhs - rhs).max().item(Float.self)
        XCTAssertLessThan(
            delta, 1e-6,
            "round-tripped weights should produce identical outputs"
        )
    }

    func testRoundTripSurvivesReSerialization() throws {
        // Load → save again → reload → compare. Catches any path
        // that mutates a tensor in place during loading.
        let cfg = Self.smallConfig
        let original = Qwen3AudioEncoder(cfg)
        eval(original)

        let firstURL = makeTempURL(prefix: "qwen3-roundtrip-1")
        let secondURL = makeTempURL(prefix: "qwen3-roundtrip-2")
        defer {
            try? FileManager.default.removeItem(at: firstURL)
            try? FileManager.default.removeItem(at: secondURL)
        }
        try dumpEncoderToURL(original, url: firstURL)

        let intermediate = Qwen3AudioEncoder(cfg)
        try Qwen3SafetensorsLoader.loadAudioTower(
            from: firstURL, into: intermediate
        )
        try dumpEncoderToURL(intermediate, url: secondURL)

        let final = Qwen3AudioEncoder(cfg)
        try Qwen3SafetensorsLoader.loadAudioTower(
            from: secondURL, into: final
        )

        let input = MLXArray.zeros(
            [1, cfg.numMelBins, 50], dtype: .float32
        )
        let mask = MLXArray(
            Array(repeating: Int32(1), count: 50)
        ).reshaped(1, 50)
        let lhs = original(
            inputFeatures: input, featureAttentionMask: mask
        )
        let rhs = final(
            inputFeatures: input, featureAttentionMask: mask
        )
        eval(lhs, rhs)
        XCTAssertLessThan(
            MLX.abs(lhs - rhs).max().item(Float.self), 1e-6
        )
    }

    // MARK: - Error paths

    func testLoaderRejectsMissingFile() {
        let bogus = URL(fileURLWithPath: "/tmp/does-not-exist.safetensors")
        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        XCTAssertThrowsError(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: bogus, into: encoder
            )
        ) { error in
            guard
                case Qwen3ASRError.fileNotFound(let url) = error
            else {
                XCTFail("expected fileNotFound, got \(error)")
                return
            }
            XCTAssertEqual(url, bogus)
        }
    }

    func testLoaderRejectsCheckpointWithoutAudioTower() throws {
        // Wrong-architecture safetensors: contains tensors but none
        // with the `audio_tower.` prefix (e.g. text-decoder slice).
        let tmpURL = makeTempURL(prefix: "qwen3-noaudio")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try MLX.save(
            arrays: [
                "thinker.layers.0.q_proj.weight": MLXArray.zeros(
                    [4, 4], dtype: .float32
                )
            ],
            url: tmpURL
        )

        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        XCTAssertThrowsError(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: tmpURL, into: encoder
            )
        ) { error in
            guard case Qwen3ASRError.noAudioTowerWeights = error else {
                XCTFail("expected noAudioTowerWeights, got \(error)")
                return
            }
        }
    }

    func testLoaderRejectsMissingTensor() throws {
        // Drop one required tensor and confirm the loader names the
        // missing key in the error rather than aborting somewhere
        // deep in MLXNN.
        let cfg = Self.smallConfig
        let donor = Qwen3AudioEncoder(cfg)
        eval(donor)
        var dumped: [String: MLXArray] = [:]
        for (name, value) in donor.parameters().flattened()
        where name != "conv2d1.bias" {
            dumped["audio_tower.\(name)"] = value
        }
        let tmpURL = makeTempURL(prefix: "qwen3-missing-key")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try MLX.save(arrays: dumped, url: tmpURL)

        let target = Qwen3AudioEncoder(cfg)
        XCTAssertThrowsError(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: tmpURL, into: target
            )
        ) { error in
            guard case Qwen3ASRError.missingTensor(let key) = error
            else {
                XCTFail("expected missingTensor, got \(error)")
                return
            }
            XCTAssertEqual(key, "conv2d1.bias")
        }
    }

    func testLoaderRejectsShapeMismatch() throws {
        let cfg = Self.smallConfig
        let donor = Qwen3AudioEncoder(cfg)
        eval(donor)

        var dumped: [String: MLXArray] = [:]
        for (name, value) in donor.parameters().flattened() {
            if name == "conv2d1.bias" {
                // Wrong shape: bias is `[downsampleHiddenSize=16]`.
                dumped["audio_tower.\(name)"] = MLXArray.zeros(
                    [32], dtype: value.dtype
                )
            } else {
                dumped["audio_tower.\(name)"] = value
            }
        }
        let tmpURL = makeTempURL(prefix: "qwen3-bad-shape")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try MLX.save(arrays: dumped, url: tmpURL)

        let target = Qwen3AudioEncoder(cfg)
        XCTAssertThrowsError(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: tmpURL, into: target
            )
        ) { error in
            guard
                case Qwen3ASRError.shapeMismatch(
                    let key, let expected, let actual
                ) = error
            else {
                XCTFail("expected shapeMismatch, got \(error)")
                return
            }
            XCTAssertEqual(key, "conv2d1.bias")
            XCTAssertEqual(expected, [16])
            XCTAssertEqual(actual, [32])
        }
    }

    func testLoaderRejectsDtypeMismatch() throws {
        // Force a non-float dtype (int32) on one tensor and confirm
        // the loader rejects it before MLXNN sees the mismatch.
        let cfg = Self.smallConfig
        let donor = Qwen3AudioEncoder(cfg)
        eval(donor)

        var dumped: [String: MLXArray] = [:]
        for (name, value) in donor.parameters().flattened() {
            if name == "conv2d1.bias" {
                dumped["audio_tower.\(name)"] = MLXArray.zeros(
                    value.shape, dtype: .int32
                )
            } else {
                dumped["audio_tower.\(name)"] = value
            }
        }
        let tmpURL = makeTempURL(prefix: "qwen3-bad-dtype")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try MLX.save(arrays: dumped, url: tmpURL)

        let target = Qwen3AudioEncoder(cfg)
        XCTAssertThrowsError(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: tmpURL, into: target
            )
        ) { error in
            guard
                case Qwen3ASRError.dtypeMismatch(let key, _, _) = error
            else {
                XCTFail("expected dtypeMismatch, got \(error)")
                return
            }
            XCTAssertEqual(key, "conv2d1.bias")
        }
    }

    func testLoaderRejectsUnexpectedTensor() throws {
        // Add an `audio_tower.bogus` key on top of a valid dump and
        // confirm the loader surfaces it as `unexpectedTensor`
        // rather than silently dropping it through to MLXNN's
        // generic verify error.
        let cfg = Self.smallConfig
        let donor = Qwen3AudioEncoder(cfg)
        eval(donor)

        var dumped: [String: MLXArray] = [:]
        for (name, value) in donor.parameters().flattened() {
            dumped["audio_tower.\(name)"] = value
        }
        dumped["audio_tower.bogus"] = MLXArray.zeros(
            [4], dtype: .float32
        )
        let tmpURL = makeTempURL(prefix: "qwen3-extra-key")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try MLX.save(arrays: dumped, url: tmpURL)

        let target = Qwen3AudioEncoder(cfg)
        XCTAssertThrowsError(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: tmpURL, into: target
            )
        ) { error in
            guard case Qwen3ASRError.unexpectedTensor(let key) = error
            else {
                XCTFail("expected unexpectedTensor, got \(error)")
                return
            }
            XCTAssertEqual(key, "bogus")
        }
    }

    func testLoaderRejectsMalformedSafetensorsHeader() throws {
        // Truncated safetensors file: write only the first 4 bytes
        // of the would-be header. `MLX.loadArrays` treats this as a
        // parse error; the loader must wrap that in
        // `.malformedSafetensors` so callers don't need to special-
        // case MLX's opaque error type.
        let tmpURL = makeTempURL(prefix: "qwen3-truncated")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try Data([0x00, 0x00, 0x00, 0x00]).write(to: tmpURL)

        let encoder = Qwen3AudioEncoder(Qwen3ASRConfig())
        XCTAssertThrowsError(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: tmpURL, into: encoder
            )
        ) { error in
            guard case Qwen3ASRError.malformedSafetensors = error else {
                XCTFail(
                    "expected malformedSafetensors, got \(error)"
                )
                return
            }
        }
    }

    func testLoaderAcceptsFloat16Weights() throws {
        // The 8-bit checkpoint stores audio_tower in bf16, but a
        // re-quantization pipeline could emit f16; the loader's
        // dtype check must accept both without forcing the caller
        // to cast.
        let cfg = Self.smallConfig
        let donor = Qwen3AudioEncoder(cfg)
        eval(donor)

        var dumped: [String: MLXArray] = [:]
        for (name, value) in donor.parameters().flattened() {
            dumped["audio_tower.\(name)"] = value.asType(.float16)
        }
        let tmpURL = makeTempURL(prefix: "qwen3-f16")
        defer { try? FileManager.default.removeItem(at: tmpURL) }
        try MLX.save(arrays: dumped, url: tmpURL)

        let target = Qwen3AudioEncoder(cfg)
        XCTAssertNoThrow(
            try Qwen3SafetensorsLoader.loadAudioTower(
                from: tmpURL, into: target
            )
        )
    }

    // MARK: - Real-weights smoke test (S1 required)

    func testEncoderLoadsRealWeights() throws {
        // Honor the same env-var override the parity tests use so a
        // local /tmp copy can sidestep macOS TCC restrictions on
        // /Volumes access from the xctest GUI test host. See
        // Qwen3AudioEncoderParityTests for details.
        let weightsURL: URL = {
            if let override = ProcessInfo.processInfo.environment[
                "YOOZ_PHASE3_ARTIFACTS"
            ] {
                return URL(fileURLWithPath: override)
                    .appendingPathComponent(
                        "phase1-spike/artifacts/"
                            + "audio_tower_bf16.safetensors"
                    )
            }
            return URL(
                fileURLWithPath:
                    "/Volumes/S1/yooz/research/issue-46/phase1-spike/"
                    + "artifacts/audio_tower_bf16.safetensors"
            )
        }()
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsURL.path),
            "Phase 1 weights not on /Volumes/S1; skipping"
        )
        let encoder = try Qwen3SafetensorsLoader.encoder(
            from: weightsURL, config: Qwen3ASRConfig()
        )
        XCTAssertEqual(encoder.layers.count, 24)

        // Tiny synthetic forward to confirm the weights graphed in.
        let synthetic = MLXArray.zeros([1, 128, 100], dtype: .float32)
        let mask = MLXArray(
            Array(repeating: Int32(1), count: 100)
        ).reshaped(1, 100)
        let out = try encoder.forward(
            inputFeatures: synthetic, featureAttentionMask: mask
        )
        eval(out)
        XCTAssertEqual(out.shape, [13, 2048])
        XCTAssertFalse(MLX.any(MLX.isNaN(out)).item(Bool.self))
    }
}
