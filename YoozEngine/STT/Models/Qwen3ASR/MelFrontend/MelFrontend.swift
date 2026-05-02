import Accelerate
import Foundation

/// Errors surfaced by the Qwen3-ASR mel frontend.
public enum MelFrontendError: Error, CustomStringConvertible, Equatable {
    case unsupportedSampleRate(actual: Int, expected: Int)
    case emptyInput
    /// `push` or `finish` was called on a streaming session that has
    /// already produced its output. Sessions are single-shot; create a
    /// new one with `MelFrontend.makeStreamingSession()` to continue.
    case streamingSessionClosed

    public var description: String {
        switch self {
        case .unsupportedSampleRate(let actual, let expected):
            return
                "MelFrontend expects \(expected) Hz mono PCM; got \(actual) Hz."
                + " Resample upstream — the Qwen3-ASR audio tower was trained on 16 kHz."
        case .emptyInput:
            return "MelFrontend cannot process zero-sample audio."
        case .streamingSessionClosed:
            return
                "MelFrontend streaming session has been finished; create a new one to continue."
        }
    }
}

/// Result of running the offline mel frontend on a single utterance.
///
/// `features` is the `(numMelFilters, numTotalFrames)` log-mel matrix
/// after Whisper's `(log_spec + 4)/4` post-processing, padded to the
/// target chunk length (3000 frames at 30 s × 100 fps for the canonical
/// Qwen3-ASR config). `attentionMask[i] == 1` for valid frames and 0 for
/// frames that fall inside the right-padding region.
public struct MelFeatures: Sendable, Equatable {
    public let features: [Float]  // numMelFilters * numTotalFrames, row-major
    public let attentionMask: [Int32]  // numTotalFrames
    public let numMelFilters: Int
    public let numTotalFrames: Int
    public let numValidFrames: Int

    /// Zero-copy view of a single mel row. Useful for parity tests
    /// that need to slice a specific bin without rebuilding a 2D
    /// container.
    public func row(_ m: Int) -> ArraySlice<Float> {
        let start = m * numTotalFrames
        return features[start..<(start + numTotalFrames)]
    }

    /// Element accessor with bounds checks via Array's own subscript.
    public subscript(_ m: Int, _ t: Int) -> Float {
        features[m * numTotalFrames + t]
    }
}

/// Configuration mirroring the Whisper preprocessor used by Qwen3-ASR.
///
/// All defaults match `preprocessor_config.json` in
/// `mlx-community/Qwen3-ASR-1.7B-8bit`. Callers should not change these
/// values for the Qwen3-ASR audio path; they are exposed as `public`
/// only so the type can be reused for other Whisper-family encoders
/// in the future.
public struct MelFrontendConfig: Sendable, Equatable {
    public let samplingRate: Int       // 16_000
    public let nFFT: Int               // 400
    public let frameLength: Int        // = nFFT (no zero-padding inside the buffer)
    public let hopLength: Int          // 160
    public let chunkLength: Int        // 30 (seconds)
    public let numMelFilters: Int      // 128
    public let minFrequency: Double    // 0.0
    public let maxFrequency: Double    // 8000.0

    /// Total samples per padded chunk (`samplingRate * chunkLength`).
    public var nSamples: Int { samplingRate * chunkLength }

    /// Total frames per padded chunk.
    /// Reference: `n_samples // hop_length`. For 480 000 / 160 = 3000.
    public var numTotalFrames: Int { nSamples / hopLength }

    public static let qwen3ASR = MelFrontendConfig(
        samplingRate: 16_000,
        nFFT: 400,
        frameLength: 400,
        hopLength: 160,
        chunkLength: 30,
        numMelFilters: 128,
        minFrequency: 0.0,
        maxFrequency: 8000.0
    )

    public init(
        samplingRate: Int,
        nFFT: Int,
        frameLength: Int,
        hopLength: Int,
        chunkLength: Int,
        numMelFilters: Int,
        minFrequency: Double,
        maxFrequency: Double
    ) {
        precondition(samplingRate > 0, "samplingRate must be positive")
        precondition(nFFT >= frameLength, "nFFT must be >= frameLength")
        precondition(hopLength > 0, "hopLength must be positive")
        precondition(chunkLength > 0, "chunkLength must be positive")
        precondition(numMelFilters >= 1, "numMelFilters must be >= 1")
        self.samplingRate = samplingRate
        self.nFFT = nFFT
        self.frameLength = frameLength
        self.hopLength = hopLength
        self.chunkLength = chunkLength
        self.numMelFilters = numMelFilters
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
    }
}

/// Whisper-style 128-mel log-spectrogram frontend.
///
/// Bit-near-equivalent to the reference Python pipeline
/// (`WhisperFeatureExtractor` + `transformers.audio_utils.spectrogram`,
/// numpy path) on which the Qwen3-ASR audio tower was trained. The
/// `MelFrontendTests` parity gate enforces a max-abs-delta of 1e-4
/// against the reference dump on multiple clips (silence, synthetic
/// speech, synthetic music, real recorded speech); measured deltas
/// land at ~2.4e-7 (single-precision unit roundoff) once the float64
/// STFT lands in float32 mel power.
///
/// Implementation notes:
///   - The DFT is implemented as two precomputed real cosine / sine
///     bases multiplied against the windowed frames via `cblas_dgemm`
///     (double-precision GEMM). vDSP_DFT does not support length 400
///     (not a `f * 2^n` radix with f in {1, 3, 5, 15}), and an ad-hoc
///     Bluestein wrapper would add review surface for no measured
///     accuracy gain. The double-precision matmul mirrors the numpy
///     reference exactly: rfft + |X|^2 are float64 and only the final
///     mel power is narrowed to float32.
///   - The Hann window is the *periodic* variant
///     (`np.hanning(N+1)[:-1]`), not the symmetric one — Whisper's
///     `window_function("hann", periodic=True)` default.
///   - The waveform is centered with reflect-padding by
///     `frameLength / 2` samples on each side
///     (`spectrogram(center=True, pad_mode="reflect")`). This is what
///     produces 3001 STFT frames out of 480 000 padded samples; the
///     last frame is dropped to land on the encoder's expected 3000.
///   - `.max() - 8.0` clipping is computed per utterance over the
///     padded log-mel grid, exactly as the reference numpy path does.
public final class MelFrontend: Sendable {
    public let config: MelFrontendConfig
    public let filterBank: MelFilterBank

    /// Hann window, length `frameLength`, float64. Used to build the
    /// windowed frames in the float64 STFT path so that the windowing
    /// operation itself doesn't introduce a rounding step before the
    /// DFT. The reference numpy path also widens the window to float64.
    private let windowDouble: [Double]
    /// Real DFT basis, row-major `(nFFT, numFrequencyBins)` in float64.
    /// `cosBasis[n, k] = cos(2π * k * n / nFFT)`.
    /// We carry the basis in float64 (~1.3 MB for 400×201) because the
    /// numpy reference path runs the entire STFT in float64 internally.
    /// Casting to float32 only at the end of the STFT keeps the parity
    /// gate well below 1e-4 even for broadband clips that exercise
    /// cancellation in many bins simultaneously.
    private let cosBasis: [Double]
    /// Imaginary DFT basis, row-major `(nFFT, numFrequencyBins)` in float64.
    /// `sinBasis[n, k] = sin(2π * k * n / nFFT)`.
    /// The negative sign of `np.fft.rfft`'s exponent is folded in at
    /// magnitude time (we square the real and imag parts), so this
    /// stays positive `sin`.
    private let sinBasis: [Double]
    /// `numFrequencyBins == 1 + nFFT / 2`.
    private let numFrequencyBins: Int

    public convenience init() {
        self.init(config: .qwen3ASR)
    }

    public init(config: MelFrontendConfig) {
        self.config = config
        self.numFrequencyBins = 1 + config.nFFT / 2
        self.filterBank = MelFilterBank(
            numFrequencyBins: numFrequencyBins,
            numMelFilters: config.numMelFilters,
            minFrequency: config.minFrequency,
            maxFrequency: config.maxFrequency,
            samplingRate: config.samplingRate
        )

        // Periodic Hann: hann(N+1)[:-1]
        // np.hanning(L) = 0.5 - 0.5 * cos(2π * i / (L - 1)), i = 0..L-1
        // With L = N+1 and dropping the last point, this is
        //   w[i] = 0.5 - 0.5 * cos(2π * i / N), i = 0..N-1
        // which matches torch.hann_window(N, periodic=True).
        let N = config.frameLength
        var hannDouble = [Double](repeating: 0.0, count: N)
        let twoPiOverN = 2.0 * Double.pi / Double(N)
        for i in 0..<N {
            hannDouble[i] = 0.5 - 0.5 * cos(twoPiOverN * Double(i))
        }
        self.windowDouble = hannDouble

        // DFT basis in float64. The numpy reference promotes the
        // waveform + window to float64 before running rfft, then casts
        // the final mel output back to float32. Mirroring that order
        // (float64 STFT, float32 mel projection + log clip) puts the
        // measured mel parity at ~2.4e-7 max-abs across silence /
        // synthetic-speech / synthetic-music / real-speech clips —
        // single-precision unit roundoff after the float32 cast.
        let bins = numFrequencyBins
        var cosD = [Double](repeating: 0.0, count: N * bins)
        var sinD = [Double](repeating: 0.0, count: N * bins)
        let twoPiOverNFFT = 2.0 * Double.pi / Double(config.nFFT)
        for n in 0..<N {
            for k in 0..<bins {
                let angle = twoPiOverNFFT * Double(k) * Double(n)
                cosD[n * bins + k] = cos(angle)
                sinD[n * bins + k] = sin(angle)
            }
        }
        self.cosBasis = cosD
        self.sinBasis = sinD
    }

    // MARK: - Offline API

    /// Compute log-mel features for a complete utterance, padded /
    /// truncated to `numTotalFrames`. Equivalent to a single call to
    /// `WhisperFeatureExtractor(audio, padding="max_length",
    /// return_attention_mask=True)` for one utterance.
    ///
    /// - Parameters:
    ///   - pcm: 16 kHz mono float32 PCM in the range `[-1, 1]`.
    ///   - sampleRate: must equal `config.samplingRate`. Resampling is
    ///     intentionally NOT done here; calling code should resample
    ///     before invoking the frontend so the parity bar is meaningful.
    /// - Returns: `MelFeatures` with the padded mel matrix and
    ///   per-frame attention mask.
    public func computeFeatures(
        pcm: [Float], sampleRate: Int = 16_000
    ) throws -> MelFeatures {
        guard sampleRate == config.samplingRate else {
            throw MelFrontendError.unsupportedSampleRate(
                actual: sampleRate, expected: config.samplingRate
            )
        }
        guard !pcm.isEmpty else {
            throw MelFrontendError.emptyInput
        }

        let nSamples = config.nSamples
        let hopLength = config.hopLength
        let frameLength = config.frameLength
        let numTotalFrames = config.numTotalFrames

        // 1) Right-pad / truncate to nSamples. Reference behavior:
        //    `padding="max_length"` zero-pads on the right;
        //    `truncation=True` (default) crops anything beyond
        //    nSamples. Both branches operate on the host buffer
        //    before the STFT runs.
        var padded = [Float](repeating: 0.0, count: nSamples)
        let copyLen = Swift.min(pcm.count, nSamples)
        padded.withUnsafeMutableBufferPointer { dst in
            pcm.withUnsafeBufferPointer { src in
                guard let dstBase = dst.baseAddress, let srcBase = src.baseAddress
                else { return }
                dstBase.update(from: srcBase, count: copyLen)
            }
        }

        // 2) Center-pad with reflect mode (frame_length // 2 samples on
        //    each side). Reference: `transformers.audio_utils.spectrogram`
        //    has `center=True, pad_mode="reflect"` defaults; the Whisper
        //    feature extractor uses these defaults, so the effective
        //    waveform length the STFT walks over is
        //    `nSamples + frameLength`. After this:
        //      numFrames = (nSamples + frameLength - frameLength) / hop + 1
        //                = nSamples / hop + 1.
        //    Drop the last frame (`log_spec[:, :-1]` in the reference)
        //    and we land on `numTotalFrames = nSamples / hop`, exactly
        //    what the Qwen3-ASR audio tower expects (3000 frames at
        //    30 s × 100 fps).
        let halfFrame = frameLength / 2
        let centeredLength = nSamples + 2 * halfFrame
        var centered = [Float](repeating: 0.0, count: centeredLength)
        Self.reflectPad(
            source: padded,
            destination: &centered,
            leftPad: halfFrame,
            rightPad: halfFrame
        )

        // 3) Build the windowed frame matrix in float64.
        //    `frames` of shape `(numFrames, frameLength)`, row-major.
        //    Reference frame count after centering:
        //    floor((centeredLength - frameLength) / hop) + 1.
        //    For 480400 / 400 / 160 -> 3001 frames.
        //    Memory: 3001 × 400 × 8 B = 9.6 MB; within the budget for
        //    the 30 s offline path. Streaming sessions reuse the same
        //    `computeFeatures` call, so the same buffer applies.
        let numFrames =
            (centeredLength - frameLength) / hopLength + 1
        let layout = FrameLayout(
            numFrames: numFrames,
            frameLength: frameLength,
            hopLength: hopLength
        )
        var frames = [Double](
            repeating: 0.0, count: numFrames * frameLength
        )
        Self.windowedFramesDouble(
            input: centered,
            output: &frames,
            window: windowDouble,
            layout: layout
        )

        // 4) Real DFT via two double-precision GEMMs, then power
        //    spectrum (Re^2 + Im^2) materialised as float32. This
        //    mirrors the numpy reference, which runs the rfft and the
        //    `np.abs(...)**2` step in float64 before casting.
        //    `power` shape: (numFrames, numFrequencyBins).
        var power = [Float](
            repeating: 0.0, count: numFrames * numFrequencyBins
        )
        Self.realPowerSpectrogram(
            inputs: STFTInputs(
                frames: frames,
                cosBasis: cosBasis,
                sinBasis: sinBasis,
                numFrames: numFrames,
                frameLength: frameLength,
                numFrequencyBins: numFrequencyBins
            ),
            power: &power
        )

        // 5) Drop the last frame. Reference: `log_spec[:, :-1]` after
        //    log-mel. Equivalent (and cheaper) to drop here, before the
        //    mel projection runs.
        let numKeptFrames = numFrames - 1
        precondition(
            numKeptFrames == numTotalFrames,
            "Internal invariant: kept frame count must match config.numTotalFrames"
        )

        // 6) Mel projection: melPower = filters @ power[:keepFrames].T,
        //    laid out as `(numMelFilters, numKeptFrames)` to match the
        //    Whisper output shape `(num_mel_filters, num_frames)`.
        var melPower = [Float](
            repeating: 0.0,
            count: config.numMelFilters * numKeptFrames
        )
        Self.applyMelFilters(
            power: power,
            keepFrames: numKeptFrames,
            numFrequencyBins: numFrequencyBins,
            filterBank: filterBank,
            melPower: &melPower
        )

        // 7) Floor at mel_floor = 1e-10, log10, max(log_spec, log_spec.max() - 8), (x+4)/4.
        Self.logFloorAndNormalise(
            melPower: &melPower
        )

        // 8) Build attention mask (num_real_frames = len(pcm) // hop).
        //    Reference: `mask[: len(audio) // hop_length] = 1`.
        //    For the standard case `len(audio) == nSamples` this is
        //    `nSamples / hop = numTotalFrames` (entire mask is 1).
        let numValidFrames = Swift.min(pcm.count / hopLength, numTotalFrames)
        var mask = [Int32](repeating: 0, count: numTotalFrames)
        for i in 0..<numValidFrames {
            mask[i] = 1
        }

        return MelFeatures(
            features: melPower,
            attentionMask: mask,
            numMelFilters: config.numMelFilters,
            numTotalFrames: numTotalFrames,
            numValidFrames: numValidFrames
        )
    }

    // MARK: - Streaming API

    /// Streaming session that buffers PCM across calls so chunked
    /// callers (e.g. WebSocket receivers) can produce STFT frames
    /// incrementally. The session keeps no padding / truncation policy
    /// of its own — the caller flushes with `finish()` when the
    /// utterance is complete, at which point the frontend applies the
    /// same `padding="max_length"` policy as the offline path.
    public final class StreamingSession {
        private let frontend: MelFrontend
        private var buffer: [Float] = []
        private var totalSamples: Int = 0
        private var finished: Bool = false

        fileprivate init(_ frontend: MelFrontend) {
            self.frontend = frontend
        }

        /// Feed a chunk of PCM. Returns the number of full frames
        /// available *if* the caller wanted incremental power
        /// spectrograms; for the parity gate we only emit log-mel at
        /// `finish()`, so this is informational.
        @discardableResult
        public func push(pcm: [Float], sampleRate: Int = 16_000) throws -> Int {
            guard sampleRate == frontend.config.samplingRate else {
                throw MelFrontendError.unsupportedSampleRate(
                    actual: sampleRate, expected: frontend.config.samplingRate
                )
            }
            guard !finished else {
                throw MelFrontendError.streamingSessionClosed
            }
            buffer.append(contentsOf: pcm)
            totalSamples += pcm.count
            // Number of complete (non-overlapping-hop) frames currently
            // synthesizable. Used for callers that want a progress signal.
            let frameLength = frontend.config.frameLength
            let hop = frontend.config.hopLength
            if buffer.count < frameLength { return 0 }
            return (buffer.count - frameLength) / hop + 1
        }

        /// Flush the buffered PCM and produce padded log-mel features.
        /// Equivalent to running `MelFrontend.computeFeatures(pcm:)` on
        /// the concatenation of every chunk pushed since session start.
        ///
        /// `finish()` is single-shot; calling it twice or calling
        /// `push` after it throws `streamingSessionClosed`.
        public func finish() throws -> MelFeatures {
            guard !finished else {
                throw MelFrontendError.streamingSessionClosed
            }
            guard totalSamples > 0 else {
                throw MelFrontendError.emptyInput
            }
            finished = true
            return try frontend.computeFeatures(
                pcm: buffer, sampleRate: frontend.config.samplingRate
            )
        }
    }

    /// Open a streaming session for chunked PCM input.
    public func makeStreamingSession() -> StreamingSession {
        StreamingSession(self)
    }

    // MARK: - Static numeric kernels

    /// Reflect-pad like `numpy.pad(mode="reflect")`: the reflection
    /// excludes the boundary value, so for a source `[a, b, c, d]`
    /// padded by 3 on each side we get `[d, c, b, a, b, c, d, c, b, a]`.
    /// This matches `transformers.audio_utils.spectrogram(center=True,
    /// pad_mode="reflect")` precisely.
    ///
    /// `leftPad` and `rightPad` may not exceed `source.count - 1`,
    /// which holds for our use case (`leftPad = rightPad = 200` on a
    /// 480 000-sample buffer).
    static func reflectPad(
        source: [Float],
        destination: inout [Float],
        leftPad: Int,
        rightPad: Int
    ) {
        precondition(leftPad >= 0 && rightPad >= 0)
        let n = source.count
        precondition(
            destination.count == n + leftPad + rightPad,
            "destination must be sized to source.count + leftPad + rightPad"
        )
        if n == 0 {
            // Pure padding of an empty buffer is undefined for reflect
            // mode; the caller guards against this with `emptyInput`.
            return
        }
        // Mirror without including the boundary samples themselves.
        // For numpy.pad reflect: pad_left[i] = source[leftPad - i].
        // Index range required: 1...leftPad (must be < n).
        precondition(
            leftPad < n,
            "reflect-pad left length (\(leftPad)) must be < source length (\(n))"
        )
        precondition(
            rightPad < n,
            "reflect-pad right length (\(rightPad)) must be < source length (\(n))"
        )
        source.withUnsafeBufferPointer { srcPtr in
            destination.withUnsafeMutableBufferPointer { dstPtr in
                guard
                    let srcBase = srcPtr.baseAddress,
                    let dstBase = dstPtr.baseAddress
                else { return }
                // Left reflection: dst[i] = src[leftPad - i] for i in 0..<leftPad
                for i in 0..<leftPad {
                    dstBase[i] = srcBase[leftPad - i]
                }
                // Body
                dstBase.advanced(by: leftPad).update(
                    from: srcBase, count: n
                )
                // Right reflection: dst[leftPad + n + j] = src[n - 2 - j]
                // for j in 0..<rightPad
                let rightStart = leftPad + n
                for j in 0..<rightPad {
                    dstBase[rightStart + j] = srcBase[n - 2 - j]
                }
            }
        }
    }

    /// Frame layout shared between the windowing and STFT helpers.
    /// Bundling these scalars keeps the helper signatures under
    /// SwiftLint's parameter-count limit and makes the relationship
    /// (`numFrames × frameLength` row-major buffer, hop = `hopLength`)
    /// explicit at every call site.
    struct FrameLayout {
        let numFrames: Int
        let frameLength: Int
        let hopLength: Int
    }

    /// Build the windowed-frame matrix in float64.
    /// `output[f, n] = (Double)input[f * hopLength + n] * window[n]`.
    /// Promoting to float64 before windowing matches the reference
    /// numpy `spectrogram` path, which casts the waveform and the
    /// window to float64 before frame extraction.
    static func windowedFramesDouble(
        input: [Float],
        output: inout [Double],
        window: [Double],
        layout: FrameLayout
    ) {
        precondition(
            output.count == layout.numFrames * layout.frameLength,
            "output buffer must be (numFrames, frameLength)"
        )
        precondition(window.count == layout.frameLength)
        input.withUnsafeBufferPointer { inputPtr in
            window.withUnsafeBufferPointer { windowPtr in
                output.withUnsafeMutableBufferPointer { outputPtr in
                    guard
                        let inBase = inputPtr.baseAddress,
                        let winBase = windowPtr.baseAddress,
                        let outBase = outputPtr.baseAddress
                    else { return }
                    for f in 0..<layout.numFrames {
                        let frameStart = f * layout.hopLength
                        let dstStart = f * layout.frameLength
                        // 1. Promote frame to Float64 in-place at the
                        //    output slot, then multiply by the window.
                        //    `vDSP_vspdp` widens float32 -> float64.
                        vDSP_vspdp(
                            inBase.advanced(by: frameStart), 1,
                            outBase.advanced(by: dstStart), 1,
                            vDSP_Length(layout.frameLength)
                        )
                        // 2. out[f, :] *= window
                        vDSP_vmulD(
                            outBase.advanced(by: dstStart), 1,
                            winBase, 1,
                            outBase.advanced(by: dstStart), 1,
                            vDSP_Length(layout.frameLength)
                        )
                    }
                }
            }
        }
    }

    /// Inputs to the real-input power spectrogram kernel.
    struct STFTInputs {
        let frames: [Double]      // (numFrames, frameLength) row-major
        let cosBasis: [Double]    // (frameLength, numFrequencyBins) row-major
        let sinBasis: [Double]    // same shape as cosBasis
        let numFrames: Int
        let frameLength: Int
        let numFrequencyBins: Int
    }

    /// Compute the real-input power spectrogram via two DGEMMs and
    /// materialise the result as float32 power.
    /// `power[f, k] = (frames[f, :] · cosBasis[:, k])^2 + (frames[f, :] · sinBasis[:, k])^2`.
    /// All intermediates are float64; the cast to float32 happens once
    /// at the end. This matches the numpy reference path
    /// (`np.abs(spectrogram) ** 2` runs in float64).
    static func realPowerSpectrogram(
        inputs: STFTInputs,
        power: inout [Float]
    ) {
        let frames = inputs.frames
        let cosBasis = inputs.cosBasis
        let sinBasis = inputs.sinBasis
        let numFrames = inputs.numFrames
        let frameLength = inputs.frameLength
        let numFrequencyBins = inputs.numFrequencyBins
        precondition(power.count == numFrames * numFrequencyBins)
        let total = numFrames * numFrequencyBins
        var realPart = [Double](repeating: 0.0, count: total)
        var imagPart = [Double](repeating: 0.0, count: total)

        // Re = frames (F, N) @ cosBasis (N, K) -> (F, K)
        // Im = frames (F, N) @ sinBasis (N, K) -> (F, K)
        // For the magnitude squared we don't care about the sign of
        // sinBasis; np.fft.rfft uses exp(-i...), which flips the sign of
        // the imaginary part, but |X|^2 is sign-insensitive.
        frames.withUnsafeBufferPointer { framesPtr in
            cosBasis.withUnsafeBufferPointer { cosPtr in
                sinBasis.withUnsafeBufferPointer { sinPtr in
                    realPart.withUnsafeMutableBufferPointer { rePtr in
                        imagPart.withUnsafeMutableBufferPointer { imPtr in
                            cblas_dgemm(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(numFrames),
                                Int32(numFrequencyBins),
                                Int32(frameLength),
                                1.0,
                                framesPtr.baseAddress, Int32(frameLength),
                                cosPtr.baseAddress, Int32(numFrequencyBins),
                                0.0,
                                rePtr.baseAddress, Int32(numFrequencyBins)
                            )
                            cblas_dgemm(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(numFrames),
                                Int32(numFrequencyBins),
                                Int32(frameLength),
                                1.0,
                                framesPtr.baseAddress, Int32(frameLength),
                                sinPtr.baseAddress, Int32(numFrequencyBins),
                                0.0,
                                imPtr.baseAddress, Int32(numFrequencyBins)
                            )
                        }
                    }
                }
            }
        }

        // power_d = Re^2 + Im^2 (in float64), then narrow to float32.
        var powerDouble = [Double](repeating: 0.0, count: total)
        realPart.withUnsafeBufferPointer { rePtr in
            imagPart.withUnsafeBufferPointer { imPtr in
                powerDouble.withUnsafeMutableBufferPointer { pPtr in
                    guard
                        let reBase = rePtr.baseAddress,
                        let imBase = imPtr.baseAddress,
                        let pBase = pPtr.baseAddress
                    else { return }
                    // power = Re * Re
                    vDSP_vsqD(reBase, 1, pBase, 1, vDSP_Length(total))
                    // power += Im * Im
                    vDSP_vmaD(
                        imBase, 1, imBase, 1,
                        pBase, 1, pBase, 1,
                        vDSP_Length(total)
                    )
                }
            }
        }
        // Narrow to float32. `vDSP_vdpsp` is the matched companion to
        // `vDSP_vspdp` we used to widen earlier.
        powerDouble.withUnsafeBufferPointer { srcPtr in
            power.withUnsafeMutableBufferPointer { dstPtr in
                guard
                    let srcBase = srcPtr.baseAddress,
                    let dstBase = dstPtr.baseAddress
                else { return }
                vDSP_vdpsp(srcBase, 1, dstBase, 1, vDSP_Length(total))
            }
        }
    }

    /// Apply the mel filter bank to the power spectrogram.
    /// `melPower[m, t] = max(mel_floor, sum_k filters[m, k] * power[t, k])`
    /// for `t = 0..<keepFrames`. Numerically equivalent to
    /// `np.maximum(mel_floor, mel_filters.T @ power[:, :keep])` in the
    /// reference numpy path. The `power` buffer may be larger than
    /// `keepFrames` rows (caller drops trailing frames); only the first
    /// `keepFrames` rows are read.
    static func applyMelFilters(
        power: [Float],
        keepFrames: Int,
        numFrequencyBins: Int,
        filterBank: MelFilterBank,
        melPower: inout [Float]
    ) {
        precondition(filterBank.numFrequencyBins == numFrequencyBins)
        precondition(
            melPower.count == filterBank.numMelFilters * keepFrames
        )
        // power is (numFrames, numFrequencyBins) row-major. We want
        //   melPower = filters @ power[:keepFrames].T
        // shape (numMelFilters, keepFrames). With BLAS layouts:
        //   A = filters         (numMelFilters,    numFrequencyBins) row-major
        //   B = power[:keep]    (keepFrames,       numFrequencyBins) row-major
        //   C = melPower        (numMelFilters,    keepFrames)       row-major
        //   C = A @ B^T  -> CblasNoTrans, CblasTrans
        filterBank.weights.withUnsafeBufferPointer { aPtr in
            power.withUnsafeBufferPointer { bPtr in
                melPower.withUnsafeMutableBufferPointer { cPtr in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(filterBank.numMelFilters),
                        Int32(keepFrames),
                        Int32(numFrequencyBins),
                        1.0,
                        aPtr.baseAddress, Int32(numFrequencyBins),
                        bPtr.baseAddress, Int32(numFrequencyBins),
                        0.0,
                        cPtr.baseAddress, Int32(keepFrames)
                    )
                }
            }
        }

        // Floor at 1e-10 (mel_floor in the reference).
        let melFloor: Float = 1e-10
        melPower.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            var lo: Float = melFloor
            var hi: Float = .infinity
            vDSP_vclip(
                base, 1,
                &lo, &hi,
                base, 1,
                vDSP_Length(ptr.count)
            )
        }
    }

    /// In-place log10, `max(log_spec, log_spec.max() - 8.0)` clip,
    /// `(log_spec + 4.0) / 4.0` post-processing. Reference:
    /// the tail of `WhisperFeatureExtractor._np_extract_fbank_features`.
    static func logFloorAndNormalise(
        melPower: inout [Float]
    ) {
        let count = melPower.count
        precondition(count > 0)

        melPower.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }

            // log10 in-place. vForce vvlog10f wants the count by reference.
            var n = Int32(count)
            // vvlog10f reads `src` and writes `dst` — same buffer is OK.
            vvlog10f(base, base, &n)

            // log_max = max(log_spec)
            var logMax: Float = 0
            vDSP_maxv(base, 1, &logMax, vDSP_Length(count))

            // Clip to [log_max - 8, +inf) using vDSP_vclip.
            var lo = logMax - 8.0
            var hi: Float = .infinity
            vDSP_vclip(base, 1, &lo, &hi, base, 1, vDSP_Length(count))

            // (x + 4) / 4 == 0.25 * x + 1.0
            var scale: Float = 0.25
            var bias: Float = 1.0
            // vDSP_vsmsa: result[i] = a[i] * scale + bias
            vDSP_vsmsa(
                base, 1,
                &scale, &bias,
                base, 1,
                vDSP_Length(count)
            )
        }
    }
}
