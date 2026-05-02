import Foundation

/// Slaney-norm Slaney-mel triangular filter bank.
///
/// Bit-for-bit equivalent to
/// `transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")`,
/// which is the construction `WhisperFeatureExtractor` uses for the
/// 128-mel projection matrix. We rebuild it in pure Swift so the engine
/// has no binary asset dependency for the Qwen3-ASR mel frontend.
///
/// The matrix is stored row-major as `(numMelFilters, numFrequencyBins)`
/// because the downstream STFT magnitudes are laid out as
/// `(numFrames, numFrequencyBins)`, and `cblas_sgemm` then computes
/// `mel = magnitudes @ filters^T` with a single right-hand-side
/// transpose, which is what the Whisper / Qwen3-ASR reference does.
public struct MelFilterBank: Sendable {
    public let numMelFilters: Int
    public let numFrequencyBins: Int
    /// Row-major `(numMelFilters, numFrequencyBins)` weights, `Float`
    /// precision. Each row sums (approximately) to a constant area
    /// after Slaney normalisation.
    public let weights: [Float]

    /// Build the canonical Whisper / Qwen3-ASR 128-mel filter bank.
    /// Exposed primarily for tests; production callers should use
    /// `MelFrontend.qwen3ASR()`, which wires the matching defaults.
    public static func whisperDefault(numMelFilters: Int = 128) -> MelFilterBank {
        Self(
            numFrequencyBins: 201,  // 1 + n_fft/2 with n_fft = 400
            numMelFilters: numMelFilters,
            minFrequency: 0.0,
            maxFrequency: 8000.0,
            samplingRate: 16_000
        )
    }

    /// Construct an arbitrary Slaney-norm Slaney-mel filter bank.
    ///
    /// - Parameters:
    ///   - numFrequencyBins: `1 + nFFT / 2`.
    ///   - numMelFilters: number of mel bins.
    ///   - minFrequency: lower edge in Hz (typically 0.0).
    ///   - maxFrequency: upper edge in Hz (typically `samplingRate / 2`).
    ///   - samplingRate: STFT sample rate in Hz.
    public init(
        numFrequencyBins: Int,
        numMelFilters: Int,
        minFrequency: Double,
        maxFrequency: Double,
        samplingRate: Int
    ) {
        precondition(numFrequencyBins >= 2, "numFrequencyBins must be >= 2")
        precondition(numMelFilters >= 1, "numMelFilters must be >= 1")
        precondition(
            minFrequency <= maxFrequency,
            "minFrequency must be <= maxFrequency"
        )

        self.numFrequencyBins = numFrequencyBins
        self.numMelFilters = numMelFilters

        // Center points of the triangular mel filters in mel space, then
        // converted back to Hz. (numMelFilters + 2) points: outer two
        // are the band edges; interior numMelFilters are the centers.
        let melMin = Self.hertzToMelSlaney(minFrequency)
        let melMax = Self.hertzToMelSlaney(maxFrequency)
        let nPts = numMelFilters + 2
        var melFreqs = [Double](repeating: 0.0, count: nPts)
        // np.linspace(melMin, melMax, nPts) -> identical formula:
        //   melMin + i * (melMax - melMin) / (nPts - 1)
        let melStep = (melMax - melMin) / Double(nPts - 1)
        for i in 0..<nPts {
            melFreqs[i] = melMin + Double(i) * melStep
        }
        let filterFreqs: [Double] = melFreqs.map { Self.melToHertzSlaney($0) }

        // FFT bin frequencies: np.linspace(0, samplingRate // 2, numFrequencyBins).
        // np.linspace uses (samplingRate // 2 - 0) / (numFrequencyBins - 1) as the step,
        // matching the reference exactly when samplingRate is even (which it is for 16 kHz).
        let fftMax = Double(samplingRate / 2)
        var fftFreqs = [Double](repeating: 0.0, count: numFrequencyBins)
        let fftStep = fftMax / Double(numFrequencyBins - 1)
        for i in 0..<numFrequencyBins {
            fftFreqs[i] = Double(i) * fftStep
        }

        // _create_triangular_filter_bank:
        //   diff = filter_freqs[1:] - filter_freqs[:-1]      (size = numMelFilters + 1)
        //   slopes = filter_freqs - fft_freqs[:, None]       (numFrequencyBins, numMelFilters+2)
        //   down  = -slopes[:, :numMelFilters]   / diff[:numMelFilters]   (broadcast)
        //   up    =  slopes[:, 2:numMelFilters+2] / diff[1:numMelFilters+1]
        //   bank  = max(0, min(down, up))                    (numFrequencyBins, numMelFilters)
        //
        // We build the bank row-major as `(numMelFilters, numFrequencyBins)`
        // because the dot-product layout in MelFrontend is
        // `magnitudes (frames, freq) @ weights^T (freq, mel)`.
        var diff = [Double](repeating: 0.0, count: nPts - 1)
        for i in 0..<(nPts - 1) {
            diff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        var weightsDouble = [Double](
            repeating: 0.0,
            count: numMelFilters * numFrequencyBins
        )
        for m in 0..<numMelFilters {
            let leftEdge = filterFreqs[m]
            let center = filterFreqs[m + 1]
            let rightEdge = filterFreqs[m + 2]
            let leftDiff = diff[m]  // center - leftEdge
            let rightDiff = diff[m + 1]  // rightEdge - center
            // Defensive: degenerate filter would zero-divide. Reference
            // uses np.diff which never produces zero for a strictly
            // increasing mel grid, but we still avoid NaN/inf if a
            // caller picks a pathological config.
            let safeLeft = leftDiff != 0.0 ? leftDiff : 1.0
            let safeRight = rightDiff != 0.0 ? rightDiff : 1.0
            for k in 0..<numFrequencyBins {
                let fk = fftFreqs[k]
                // slopes[k, m]   = leftEdge   - fk  -> down = -(leftEdge - fk) / leftDiff
                // slopes[k, m+2] = rightEdge  - fk  ->  up  =  (rightEdge - fk) / rightDiff
                let down = -(leftEdge - fk) / safeLeft
                let up = (rightEdge - fk) / safeRight
                let v = min(down, up)
                if v > 0.0 {
                    weightsDouble[m * numFrequencyBins + k] = v
                }
            }
        }

        // Slaney area normalisation: enorm[m] = 2 / (filter_freqs[m+2] - filter_freqs[m])
        for m in 0..<numMelFilters {
            let bandwidth = filterFreqs[m + 2] - filterFreqs[m]
            // bandwidth > 0 because mel grid is strictly increasing.
            let enorm = 2.0 / bandwidth
            for k in 0..<numFrequencyBins {
                weightsDouble[m * numFrequencyBins + k] *= enorm
            }
        }

        // Reference computes the bank in float64 then casts to float32
        // for storage. We do the same; the Whisper preprocessor stores
        // `mel_filters` as float64 internally but the actual mel mat-vec
        // inside `_np_extract_fbank_features` runs in float32 because
        // the spectrogram is float32. The cast happens implicitly via
        // `np.dot(mel_filters.T, spectrogram)` where `spectrogram` is
        // float32. We mirror that ordering exactly: cast to float32
        // here, then keep all downstream math in float32.
        self.weights = weightsDouble.map { Float($0) }
    }

    // MARK: - Mel-scale conversions (Slaney)

    /// `transformers.audio_utils.hertz_to_mel(freq, "slaney")`.
    /// Linear below 1 kHz, log above.
    @inline(__always)
    static func hertzToMelSlaney(_ freq: Double) -> Double {
        let minLogHertz = 1_000.0
        let minLogMel = 15.0
        let logstep = 27.0 / log(6.4)
        if freq >= minLogHertz {
            return minLogMel + log(freq / minLogHertz) * logstep
        }
        return 3.0 * freq / 200.0
    }

    /// `transformers.audio_utils.mel_to_hertz(mels, "slaney")`.
    @inline(__always)
    static func melToHertzSlaney(_ mels: Double) -> Double {
        let minLogHertz = 1_000.0
        let minLogMel = 15.0
        let logstep = log(6.4) / 27.0
        if mels >= minLogMel {
            return minLogHertz * exp(logstep * (mels - minLogMel))
        }
        return 200.0 * mels / 3.0
    }
}
