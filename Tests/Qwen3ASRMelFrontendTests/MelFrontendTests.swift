import Foundation
import XCTest

@testable import Qwen3ASRMelFrontend

/// Tests for the Qwen3-ASR Swift mel frontend.
///
/// The parity gate compares Swift output against reference dumps
/// produced by `dump_mel.py` (running on `transformers` +
/// `WhisperFeatureExtractor` with the canonical Qwen3-ASR
/// `preprocessor_config.json`). Reference fixtures live on the S1
/// scratch volume so the engine repo doesn't carry binary blobs.
///
/// All four reference clips share the parity gate: silence,
/// synthetic speech, synthetic music, and a real recorded speech
/// clip. The bar is `max-abs-delta <= 1e-4` per the issue gate.
final class MelFrontendTests: XCTestCase {
    /// Per-test parity tolerance. The Phase 2 issue mandates 1e-4;
    /// this leaves no slack for "just barely passing" — a regression
    /// on any clip will surface immediately.
    static let parityTolerance: Float = 1e-4

    static let parityRoot = URL(
        fileURLWithPath:
            "/Volumes/S1/yooz/research/issue-46/phase2-mel/parity"
    )

    static let parityClips: [String] = [
        "silence",
        "speech_synthetic",
        "music_synthetic",
        "real_speech",
    ]

    // MARK: - Fixture loading

    /// Load a reference parity fixture. Returns nil if the fixture is
    /// missing (e.g. CI without S1 mounted), so the test is gracefully
    /// skipped rather than failing in an environment that physically
    /// can't reach the dump volume.
    private func loadFixture(_ name: String) throws -> ParityFixture? {
        let url = Self.parityRoot.appendingPathComponent("\(name).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        let bundle = try SafetensorsReader.load(url: url)
        let (audio, _) = try bundle.float32("audio")
        let (features, featuresShape) = try bundle.float32(
            "input_features", expectedShape: [128, 3000]
        )
        let (mask, _) = try bundle.int32(
            "attention_mask", expectedShape: [3000]
        )
        return ParityFixture(
            name: name,
            audio: audio,
            inputFeatures: features,
            inputFeaturesShape: featuresShape,
            attentionMask: mask
        )
    }

    // MARK: - Mel filter bank parity

    func testMelFilterBankMatchesReference() throws {
        let url = Self.parityRoot.appendingPathComponent(
            "mel_filters.safetensors"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: url.path),
            "Reference mel_filters.safetensors not present at \(url.path);"
                + " run dump_mel.py first."
        )
        let bundle = try SafetensorsReader.load(url: url)
        // Reference shape is (num_frequency_bins, num_mel_filters) = (201, 128).
        // Our filter-bank stores the transposed layout (mel, freq) row-major,
        // so we transpose the reference before comparing.
        let (reference, _) = try bundle.float32(
            "mel_filters", expectedShape: [201, 128]
        )

        let bank = MelFilterBank.whisperDefault()
        XCTAssertEqual(bank.numMelFilters, 128)
        XCTAssertEqual(bank.numFrequencyBins, 201)

        // bank.weights[m * 201 + k]  vs  reference[k * 128 + m]
        var maxAbsDelta: Float = 0
        for m in 0..<128 {
            for k in 0..<201 {
                let swiftValue = bank.weights[m * 201 + k]
                let pythonValue = reference[k * 128 + m]
                let delta = abs(swiftValue - pythonValue)
                if delta > maxAbsDelta { maxAbsDelta = delta }
            }
        }
        // Filter-bank construction is closed-form arithmetic; tolerance
        // is the float32 round-trip noise (~1 ulp at unit magnitude).
        XCTAssertLessThan(
            maxAbsDelta, 1e-7,
            "Mel filter bank diverges from reference by \(maxAbsDelta);"
                + " check Slaney mel-scale arithmetic."
        )
    }

    // MARK: - Offline parity gate

    func testOfflineParityAllReferenceClips() throws {
        let frontend = MelFrontend()
        var ranAtLeastOne = false

        for name in Self.parityClips {
            guard let fixture = try loadFixture(name) else {
                continue
            }
            ranAtLeastOne = true
            let result = try frontend.computeFeatures(
                pcm: fixture.audio, sampleRate: 16_000
            )

            // Shape sanity
            XCTAssertEqual(fixture.inputFeaturesShape, [128, 3000])
            XCTAssertEqual(result.numMelFilters, 128)
            XCTAssertEqual(result.numTotalFrames, 3000)
            XCTAssertEqual(
                result.features.count, 128 * 3000,
                "feature buffer size mismatch on clip \(name)"
            )

            // Attention mask sanity. nValid = len(audio)//hop = 80000/160 = 500.
            XCTAssertEqual(
                result.numValidFrames, 500,
                "attention-mask valid count drift on clip \(name)"
            )
            for i in 0..<500 {
                XCTAssertEqual(
                    result.attentionMask[i], 1,
                    "valid-frame mask wrong at idx=\(i) clip=\(name)"
                )
            }
            for i in 500..<3000 {
                XCTAssertEqual(
                    result.attentionMask[i], 0,
                    "padding mask wrong at idx=\(i) clip=\(name)"
                )
            }

            // Numerical parity vs Python reference.
            let (maxAbs, meanAbs) = Self.maxAndMeanAbsDelta(
                result.features, fixture.inputFeatures
            )
            print(
                "[parity] clip=\(name)"
                    + " max_abs_delta=\(maxAbs)"
                    + " mean_abs_delta=\(meanAbs)"
            )
            XCTAssertLessThanOrEqual(
                maxAbs, Self.parityTolerance,
                "Parity gate failed on clip \(name):"
                    + " max_abs_delta=\(maxAbs) > \(Self.parityTolerance)"
            )
        }

        try XCTSkipUnless(
            ranAtLeastOne,
            "No parity fixtures available under \(Self.parityRoot.path);"
                + " run dump_mel.py to populate."
        )
    }

    // MARK: - Streaming parity

    func testStreamingMatchesOfflineForAllChunkSizes() throws {
        guard let fixture = try loadFixture("real_speech") else {
            throw XCTSkip(
                "real_speech fixture unavailable; streaming test needs"
                    + " a reference clip with non-trivial spectral content."
            )
        }
        let frontend = MelFrontend()
        let oneShot = try frontend.computeFeatures(
            pcm: fixture.audio, sampleRate: 16_000
        )

        // Probe several chunk sizes including:
        //   - hop boundary (160) — exact frame stride
        //   - mid-frame (137) — coprime with hop
        //   - sub-frame  (80)  — half a hop, every other chunk has new frames
        //   - large      (8000) — half-second blocks
        for chunkSize in [160, 137, 80, 8000] {
            let session = frontend.makeStreamingSession()
            var pos = 0
            while pos < fixture.audio.count {
                let end = Swift.min(pos + chunkSize, fixture.audio.count)
                let slice = Array(fixture.audio[pos..<end])
                _ = try session.push(pcm: slice, sampleRate: 16_000)
                pos = end
            }
            let streamed = try session.finish()

            // Bit-equivalent: we feed the same samples in the same order,
            // so with deterministic single-precision arithmetic the
            // outputs must be identical, not just within a tolerance.
            XCTAssertEqual(
                streamed.numTotalFrames, oneShot.numTotalFrames,
                "frame count drift at chunk_size=\(chunkSize)"
            )
            XCTAssertEqual(
                streamed.numValidFrames, oneShot.numValidFrames,
                "valid-frame drift at chunk_size=\(chunkSize)"
            )
            let (maxAbs, _) = Self.maxAndMeanAbsDelta(
                streamed.features, oneShot.features
            )
            XCTAssertEqual(
                maxAbs, 0,
                "streaming output drifted from one-shot output"
                    + " at chunk_size=\(chunkSize): max_abs=\(maxAbs)"
            )
            // And the mask must be byte-identical too.
            XCTAssertEqual(
                streamed.attentionMask, oneShot.attentionMask,
                "attention-mask drifted from one-shot at chunk=\(chunkSize)"
            )
        }
    }

    // MARK: - Edge cases

    func testRejectsWrongSampleRate() throws {
        let frontend = MelFrontend()
        let bogus = [Float](repeating: 0.0, count: 1_000)
        XCTAssertThrowsError(
            try frontend.computeFeatures(pcm: bogus, sampleRate: 44_100)
        ) { error in
            guard let melError = error as? MelFrontendError else {
                XCTFail("expected MelFrontendError, got \(error)")
                return
            }
            switch melError {
            case .unsupportedSampleRate(let actual, let expected):
                XCTAssertEqual(actual, 44_100)
                XCTAssertEqual(expected, 16_000)
            default:
                XCTFail("expected unsupportedSampleRate, got \(melError)")
            }
        }
    }

    func testRejectsEmptyInput() throws {
        let frontend = MelFrontend()
        XCTAssertThrowsError(
            try frontend.computeFeatures(pcm: [], sampleRate: 16_000)
        ) { error in
            XCTAssertEqual(error as? MelFrontendError, .emptyInput)
        }
    }

    func testStreamingSessionClosedAfterFinish() throws {
        // After `finish()` the session is single-shot: subsequent
        // `push` and a second `finish` must throw, not silently
        // produce a stale `MelFeatures` from the previous run.
        let frontend = MelFrontend()
        let session = frontend.makeStreamingSession()
        let chunk = [Float](repeating: 0.1, count: 16_000)
        _ = try session.push(pcm: chunk, sampleRate: 16_000)
        _ = try session.finish()

        XCTAssertThrowsError(
            try session.push(pcm: chunk, sampleRate: 16_000)
        ) { error in
            XCTAssertEqual(
                error as? MelFrontendError, .streamingSessionClosed
            )
        }
        XCTAssertThrowsError(try session.finish()) { error in
            XCTAssertEqual(
                error as? MelFrontendError, .streamingSessionClosed
            )
        }
    }

    func testStreamingSessionRejectsEmptyOnFinish() throws {
        // A session that never received any samples must surface
        // `emptyInput` on finish, not produce all-silence mel features
        // (which the encoder would silently accept).
        let frontend = MelFrontend()
        let session = frontend.makeStreamingSession()
        XCTAssertThrowsError(try session.finish()) { error in
            XCTAssertEqual(error as? MelFrontendError, .emptyInput)
        }
    }

    func testFrontendInstanceCanProcessMultipleUtterances() throws {
        // Frontend caches the Hann window + DFT bases at init; verify
        // those caches are reusable across calls and don't accumulate
        // any per-utterance state.
        let frontend = MelFrontend()
        var utterance1 = [Float](repeating: 0, count: 8_000)
        var utterance2 = [Float](repeating: 0, count: 8_000)
        for i in 0..<8_000 {
            let t = Float(i) / 16_000.0
            utterance1[i] = sin(2.0 * .pi * 440.0 * t)
            utterance2[i] = sin(2.0 * .pi * 880.0 * t)
        }
        let r1a = try frontend.computeFeatures(
            pcm: utterance1, sampleRate: 16_000
        )
        let r2 = try frontend.computeFeatures(
            pcm: utterance2, sampleRate: 16_000
        )
        let r1b = try frontend.computeFeatures(
            pcm: utterance1, sampleRate: 16_000
        )
        // Same input twice -> bit-identical output.
        XCTAssertEqual(r1a.features, r1b.features)
        XCTAssertEqual(r1a.attentionMask, r1b.attentionMask)
        // Different input -> different output (sanity).
        XCTAssertNotEqual(r1a.features, r2.features)
    }

    func testHandlesSubFrameInput() throws {
        // Audio shorter than a single frame: Whisper pads to 30 s with
        // zeros, so the mel frontend is just running on a 480k-sample
        // buffer with the first 100 samples set to a few small values.
        // The result must not NaN and must produce 3000 frames; the
        // first ~few frames will have "real" content and the rest will
        // be silence-equivalent.
        let frontend = MelFrontend()
        let tiny = (0..<100).map { Float($0) / 1000.0 }
        let result = try frontend.computeFeatures(
            pcm: tiny, sampleRate: 16_000
        )
        XCTAssertEqual(result.numTotalFrames, 3000)
        XCTAssertEqual(result.numValidFrames, 0)  // 100 / 160 == 0
        // No NaN/Inf anywhere, and bounded to the post-normalisation
        // window. After `(log_spec + 4)/4` with log_spec clipped to
        // `[log_max - 8, log_max]`, the output range is exactly
        // `[(log_max - 4)/4, (log_max + 4)/4]`. For tiny inputs
        // log_max is small (close to log10(mel_floor) = -10), so the
        // window collapses to roughly [-1.5, +1.0]; bound a touch
        // wider for safety.
        for v in result.features {
            XCTAssertFalse(v.isNaN, "mel value NaN on sub-frame input")
            XCTAssertFalse(v.isInfinite, "mel value inf on sub-frame input")
            XCTAssertGreaterThan(v, -2.0)
            XCTAssertLessThan(v, 2.0)
        }
    }

    func testHandlesAllZeroInput() throws {
        // Reference (silence.safetensors) gives the constant -1.5 since
        // (log10(1e-10) + 4) / 4 = -1.5. Verify against that constant
        // directly so this test holds even without the parity dump
        // available.
        let frontend = MelFrontend()
        let zeros = [Float](repeating: 0.0, count: 80_000)
        let result = try frontend.computeFeatures(
            pcm: zeros, sampleRate: 16_000
        )
        var maxAbs: Float = 0
        for v in result.features {
            let d = abs(v - (-1.5))
            if d > maxAbs { maxAbs = d }
        }
        XCTAssertLessThan(
            maxAbs, 1e-6,
            "all-zero input should give constant -1.5 mel; max_abs=\(maxAbs)"
        )
    }

    func testHandlesClippingHighAmplitude() throws {
        // Saturated sine. Not realistic but exercises the high end of
        // the dynamic range without producing NaN/Inf.
        let frontend = MelFrontend()
        let n = 80_000
        var clipped = [Float](repeating: 0.0, count: n)
        for i in 0..<n {
            let v = sin(2.0 * .pi * 440.0 * Float(i) / 16_000.0) * 10.0
            clipped[i] = Swift.max(Swift.min(v, 1.0), -1.0)
        }
        let result = try frontend.computeFeatures(
            pcm: clipped, sampleRate: 16_000
        )
        for v in result.features {
            XCTAssertFalse(v.isNaN, "mel value NaN on clipped input")
            XCTAssertFalse(v.isInfinite, "mel value inf on clipped input")
            // After (x+4)/4 with x clipped to [log_max - 8, log_max],
            // the output range is exactly [-1, +1] relative to log_max
            // shift. log10 of mel_floor=1e-10 floor never appears here.
            // Just sanity-bound to a wide window.
            XCTAssertGreaterThan(v, -10.0)
            XCTAssertLessThan(v, 10.0)
        }
    }

    // MARK: - Helpers

    private struct ParityFixture {
        let name: String
        let audio: [Float]
        let inputFeatures: [Float]
        let inputFeaturesShape: [Int]
        let attentionMask: [Int32]
    }

    private static func maxAndMeanAbsDelta(
        _ a: [Float], _ b: [Float]
    ) -> (Float, Double) {
        precondition(a.count == b.count)
        var maxAbs: Float = 0
        var sum: Double = 0
        for i in 0..<a.count {
            let d = abs(a[i] - b[i])
            if d > maxAbs { maxAbs = d }
            sum += Double(d)
        }
        return (maxAbs, sum / Double(a.count))
    }
}
