import Foundation
import MLX
import MLXNN
import XCTest

@testable import SpikeASR

/// Phase 1 spike test bed.
///
/// The "structural" tests are weight-free: they construct an encoder
/// from a config and confirm shapes, parameter counts, deterministic
/// helper math, and error paths. They run on any machine.
///
/// The parity tests are gated by the presence of the artifacts
/// produced by `dump_parity.py`. When the artifacts are missing
/// (e.g. CI without /Volumes/S1 mounted), the parity tests
/// `XCTSkipUnless(...)` and the suite still passes — explicit
/// behavior, not a silent pass.
final class SpikeASRTests: XCTestCase {

    // MARK: - Structural tests (no weights, no S1)

    func testFreqAfterConvMatchesReference() {
        // Reference: ((((128 + 1) // 2) + 1) // 2 + 1) // 2 == 16
        XCTAssertEqual(AudioEncoderConfig().freqAfterConv, 16)
        XCTAssertEqual(
            AudioEncoderConfig(numMelBins: 80).freqAfterConv, 10
        )
    }

    func testFeatExtractOutputLengthMatchesReference() {
        // Ground-truth values pulled directly from running the Python
        // reference `_get_feat_extract_output_lengths` on the same
        // inputs (see /Volumes/S1/yooz/research/issue-46/reference).
        // L=99 / L=100 are both 13 because both yield a full first
        // window after the floor-div / +1 trick.
        XCTAssertEqual(AudioEncoder.featExtractOutputLength(1), 1)
        XCTAssertEqual(AudioEncoder.featExtractOutputLength(50), 7)
        XCTAssertEqual(AudioEncoder.featExtractOutputLength(99), 13)
        XCTAssertEqual(AudioEncoder.featExtractOutputLength(100), 13)
        XCTAssertEqual(AudioEncoder.featExtractOutputLength(200), 26)
        XCTAssertEqual(AudioEncoder.featExtractOutputLength(3000), 390)
    }

    func testFloorDivMatchesPythonSemantics() {
        // Python: -1 // 2 == -1, -3 // 2 == -2, 5 // 2 == 2,
        //        -5 // -2 == 2, 5 // -2 == -3, 0 // 1 == 0.
        XCTAssertEqual(AudioEncoder.floorDiv(-1, 2), -1)
        XCTAssertEqual(AudioEncoder.floorDiv(-3, 2), -2)
        XCTAssertEqual(AudioEncoder.floorDiv(5, 2), 2)
        XCTAssertEqual(AudioEncoder.floorDiv(-5, -2), 2)
        XCTAssertEqual(AudioEncoder.floorDiv(5, -2), -3)
        XCTAssertEqual(AudioEncoder.floorDiv(0, 1), 0)
    }

    func testEncoderShapesMatchCheckpoint() {
        // Build an encoder and confirm every parameter shape lines up
        // with the audio_tower.* slice of the 1.7B-8bit checkpoint.
        let encoder = AudioEncoder(AudioEncoderConfig())
        let params = encoder.parameters().flattened()
        let lookup = Dictionary(uniqueKeysWithValues: params)

        // Expected shapes for a sampling of representative weights;
        // the loader test below covers the rest exhaustively.
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
            (
                "layers.23.final_layer_norm.weight", [1024]
            ),
        ]
        for (name, shape) in expectations {
            guard let arr = lookup[name] else {
                XCTFail("missing parameter \(name)")
                continue
            }
            XCTAssertEqual(arr.shape, shape, "shape mismatch for \(name)")
        }

        // Confirm 24 layers materialized, each with the same key set.
        XCTAssertEqual(encoder.layers.count, 24)
    }

    func testLoaderRoundTripsRandomWeights() throws {
        // Synthesize a deterministic encoder, dump its parameters as
        // `audio_tower.*` keys, write them to a temp safetensors file,
        // load them back into a fresh encoder, and assert that a
        // forward pass on a fixed input matches.
        let cfgSmall = AudioEncoderConfig(
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

        let original = AudioEncoder(cfgSmall)
        // Force materialization of all params at fixed values.
        eval(original)

        let originalParams = original.parameters().flattened()
        var dumped: [String: MLXArray] = [:]
        for (name, value) in originalParams {
            dumped["audio_tower.\(name)"] = value
        }

        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "spike-asr-roundtrip-\(UUID().uuidString).safetensors"
            )
        try MLX.save(arrays: dumped, url: tmpURL)
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let restored = AudioEncoder(cfgSmall)
        try SpikeLoader.loadAudioTower(from: tmpURL, into: restored)

        let testInput = MLXArray.zeros([1, 16, 50], dtype: .float32)
        let mask = MLXArray(
            Array(repeating: Int32(1), count: 50)
        ).reshaped(1, 50)
        let lhs = original(inputFeatures: testInput, featureAttentionMask: mask)
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

    func testLoaderRejectsMissingFile() {
        let bogus = URL(fileURLWithPath: "/tmp/does-not-exist.safetensors")
        let encoder = AudioEncoder(AudioEncoderConfig())
        XCTAssertThrowsError(
            try SpikeLoader.loadAudioTower(from: bogus, into: encoder)
        ) { error in
            guard case SpikeLoaderError.fileNotFound(let url) = error else {
                XCTFail("expected fileNotFound, got \(error)")
                return
            }
            XCTAssertEqual(url, bogus)
        }
    }

    func testLoaderRejectsUnrelatedSafetensors() throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "spike-asr-noaudio-\(UUID().uuidString).safetensors"
            )
        try MLX.save(
            arrays: ["unrelated": MLXArray.zeros([2, 2], dtype: .float32)],
            url: tmpURL
        )
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let encoder = AudioEncoder(AudioEncoderConfig())
        XCTAssertThrowsError(
            try SpikeLoader.loadAudioTower(from: tmpURL, into: encoder)
        ) { error in
            guard case SpikeLoaderError.noAudioTowerWeights = error else {
                XCTFail("expected noAudioTowerWeights, got \(error)")
                return
            }
        }
    }

    // MARK: - Smoke test (weights required, runs on dev machines)

    private static let weightsURL = URL(
        fileURLWithPath:
            "/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/"
            + "audio_tower_bf16.safetensors")
    private static let parityInputsURL = URL(
        fileURLWithPath:
            "/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/"
            + "parity_inputs.safetensors")
    private static let parityOutputsURL = URL(
        fileURLWithPath:
            "/Volumes/S1/yooz/research/issue-46/phase1-spike/artifacts/"
            + "parity_outputs.safetensors")

    func testEncoderLoadsRealWeights() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.weightsURL.path),
            "Phase 1 weights not on /Volumes/S1; skipping"
        )
        let encoder = try SpikeLoader.encoder(
            from: Self.weightsURL, config: AudioEncoderConfig()
        )
        XCTAssertEqual(encoder.layers.count, 24)

        // Tiny synthetic forward (1 chunk of zeros) — verifies the
        // weights actually graphed in and the conv/attention path
        // doesn't NaN with zero input. Output shape for a 100-frame
        // synthetic chunk is `(13, 2048)` per `_get_feat_extract_output_lengths`.
        let synthetic = MLXArray.zeros([1, 128, 100], dtype: .float32)
        let mask = MLXArray(
            Array(repeating: Int32(1), count: 100)
        ).reshaped(1, 100)
        let out = encoder(
            inputFeatures: synthetic, featureAttentionMask: mask
        )
        eval(out)
        XCTAssertEqual(out.shape, [13, 2048])
        XCTAssertFalse(MLX.any(MLX.isNaN(out)).item(Bool.self))
    }

    // MARK: - Parity test (weights + reference outputs required)

    func testEncoderParityWithPython() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.weightsURL.path)
                && FileManager.default.fileExists(
                    atPath: Self.parityInputsURL.path
                )
                && FileManager.default.fileExists(
                    atPath: Self.parityOutputsURL.path
                ),
            "Phase 1 parity artifacts missing; skipping"
        )
        let encoder = try SpikeLoader.encoder(
            from: Self.weightsURL, config: AudioEncoderConfig()
        )

        let inputs = try MLX.loadArrays(url: Self.parityInputsURL)
        let referenceOutputs = try MLX.loadArrays(
            url: Self.parityOutputsURL
        )
        guard let features = inputs["input_features"],
            let referenceHidden = referenceOutputs["encoder_hidden_states"]
        else {
            XCTFail("parity artifacts missing expected keys")
            return
        }
        let mask = inputs["feature_attention_mask"]

        let actual = encoder(
            inputFeatures: features, featureAttentionMask: mask
        )
        eval(actual)

        XCTAssertEqual(actual.shape, referenceHidden.shape)

        let actualF32 = actual.asType(.float32)
        let refF32 = referenceHidden.asType(.float32)
        let diff = MLX.abs(actualF32 - refF32)
        let maxAbs = diff.max().item(Float.self)
        let meanAbs = diff.mean().item(Float.self)

        // Bf16-noise envelope; matches the pass criterion in the
        // spike spec.
        XCTAssertLessThanOrEqual(
            maxAbs, 1e-3,
            "max-abs-delta=\(maxAbs) exceeds 1e-3 parity bar"
        )
        XCTAssertLessThan(
            meanAbs, 5e-5,
            "mean-abs-delta=\(meanAbs) outside expected envelope"
        )

        // Persist the diagnostic numbers next to the artifacts so
        // PARITY.md can quote them without rerunning the test. This
        // is best-effort; a read-only artifacts directory must not
        // fail an otherwise-passing parity check.
        let lines: String = """
            max_abs_delta=\(maxAbs)
            mean_abs_delta=\(meanAbs)
            shape=\(actual.shape)
            """
        let dest = Self.parityOutputsURL
            .deletingLastPathComponent()
            .appendingPathComponent("parity_swift_metrics.txt")
        try? lines.write(to: dest, atomically: true, encoding: .utf8)
    }
}
