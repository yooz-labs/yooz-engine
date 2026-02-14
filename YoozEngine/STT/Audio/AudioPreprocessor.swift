// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
@preconcurrency import MLX

// MARK: - Window Functions

/// Window function types for STFT
public enum WindowFunction: String, Sendable {
    case hann
    case hanning
    case hamming
    case blackman
    case bartlett

    /// Generate window array of given size
    public func generate(size: Int) -> MLXArray {
        switch self {
        case .hann, .hanning:
            return hanningWindow(size: size)
        case .hamming:
            return hammingWindow(size: size)
        case .blackman:
            return blackmanWindow(size: size)
        case .bartlett:
            return bartlettWindow(size: size)
        }
    }
}

/// Hanning window: 0.5 * (1 - cos(2*pi*n/(N-1)))
private func hanningWindow(size: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: size)
    for n in 0..<size {
        values[n] = 0.5 * (1 - cos(2 * .pi * Float(n) / Float(size - 1)))
    }
    return MLXArray(values)
}

/// Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
private func hammingWindow(size: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: size)
    for n in 0..<size {
        values[n] = 0.54 - 0.46 * cos(2 * .pi * Float(n) / Float(size - 1))
    }
    return MLXArray(values)
}

/// Blackman window
private func blackmanWindow(size: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: size)
    for n in 0..<size {
        let t = Float(n) / Float(size - 1)
        values[n] = 0.42 - 0.5 * cos(2 * .pi * t) + 0.08 * cos(4 * .pi * t)
    }
    return MLXArray(values)
}

/// Bartlett (triangular) window
private func bartlettWindow(size: Int) -> MLXArray {
    var values = [Float](repeating: 0, count: size)
    for n in 0..<size {
        values[n] = 1 - 2 * abs(Float(n) - Float(size - 1) / 2) / Float(size - 1)
    }
    return MLXArray(values)
}

// MARK: - Mel Filterbank

/// Compute mel filterbank matrix
/// - Parameters:
///   - sampleRate: Audio sample rate (e.g., 16000)
///   - nFft: FFT size
///   - nMels: Number of mel frequency bins
///   - fMin: Minimum frequency (default 0)
///   - fMax: Maximum frequency (default sampleRate/2)
///   - norm: Normalization type ("slaney" or nil)
/// - Returns: Mel filterbank matrix of shape [nMels, nFft/2+1]
public func melFilterbank(
    sampleRate: Int,
    nFft: Int,
    nMels: Int,
    fMin: Float = 0,
    fMax: Float? = nil,
    norm: String? = nil
) -> MLXArray {
    let fMax = fMax ?? Float(sampleRate) / 2

    // Hz to Mel conversion (HTK formula)
    func hzToMel(_ freq: Float) -> Float {
        return 2595.0 * log10(1.0 + freq / 700.0)
    }

    // Mel to Hz conversion
    func melToHz(_ mel: Float) -> Float {
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    let nFreqs = nFft / 2 + 1

    // Generate linearly spaced frequencies
    var allFreqs = [Float](repeating: 0, count: nFreqs)
    for i in 0..<nFreqs {
        allFreqs[i] = Float(i) * Float(sampleRate) / Float(nFft)
    }

    // Generate mel points
    let mMin = hzToMel(fMin)
    let mMax = hzToMel(fMax)
    var melPoints = [Float](repeating: 0, count: nMels + 2)
    for i in 0..<(nMels + 2) {
        let mel = mMin + Float(i) * (mMax - mMin) / Float(nMels + 1)
        melPoints[i] = melToHz(mel)
    }

    // Build filterbank matrix
    var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nFreqs), count: nMels)

    for m in 0..<nMels {
        let fLeft = melPoints[m]
        let fCenter = melPoints[m + 1]
        let fRight = melPoints[m + 2]

        for k in 0..<nFreqs {
            let freq = allFreqs[k]

            if freq >= fLeft && freq <= fCenter {
                // Rising slope
                filterbank[m][k] = (freq - fLeft) / (fCenter - fLeft)
            } else if freq > fCenter && freq <= fRight {
                // Falling slope
                filterbank[m][k] = (fRight - freq) / (fRight - fCenter)
            }
        }

        // Apply slaney normalization if requested
        if norm == "slaney" {
            let enorm = 2.0 / (fRight - fLeft)
            for k in 0..<nFreqs {
                filterbank[m][k] *= enorm
            }
        }
    }

    // Convert to MLXArray [nMels, nFreqs]
    let flatData = filterbank.flatMap { $0 }
    return MLXArray(flatData).reshaped([nMels, nFreqs])
}

// MARK: - STFT

/// Short-time Fourier Transform (vectorized using asStrided)
/// - Parameters:
///   - x: Input audio signal (1D array)
///   - nFft: FFT window size
///   - hopLength: Number of samples between frames
///   - winLength: Window length (typically same as nFft)
///   - window: Window function to apply
///   - center: Whether to pad input to center frames
/// - Returns: Complex STFT matrix of shape [numFrames, nFft/2+1]
public func stft(
    _ x: MLXArray,
    nFft: Int,
    hopLength: Int,
    winLength: Int,
    window: MLXArray,
    center: Bool = true
) -> MLXArray {
    var signal = x

    // Pad signal if centering
    if center {
        let padAmount = nFft / 2
        // Use zero padding (simpler than reflect, minimal impact on quality)
        let padding = MLXArray.zeros([padAmount])
        signal = concatenated([padding, signal, padding], axis: 0)
    }

    // Pad window to nFft if needed
    var win = window
    if win.dim(0) < nFft {
        let padSize = nFft - win.dim(0)
        win = concatenated([win, MLXArray.zeros([padSize])], axis: 0)
    }

    // Calculate number of frames
    let numFrames = 1 + (signal.dim(0) - nFft) / hopLength
    guard numFrames > 0 else {
        fatalError("Input too short for STFT with given parameters")
    }

    // Use asStrided for efficient frame extraction (no data copy)
    // This creates a view with shape [numFrames, nFft] where each row
    // overlaps with the previous by (nFft - hopLength) samples
    let shape = [numFrames, nFft]
    let strides = [hopLength, 1]
    let frames = asStrided(signal, shape, strides: strides)

    // Apply window and compute rfft
    return rfft(frames * win, axis: -1)
}

// MARK: - Audio Preprocessor

/// Audio preprocessor for computing log mel spectrograms
public struct AudioPreprocessor: Sendable {
    public let config: PreprocessConfig

    // Cached mel filterbank
    private let melFilters: MLXArray

    // Cached window function
    private let window: MLXArray

    public init(config: PreprocessConfig) {
        self.config = config

        // Pre-compute mel filterbank
        self.melFilters = melFilterbank(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            norm: config.normalize == "slaney" ? "slaney" : nil
        )

        // Pre-compute window function
        let windowFn = WindowFunction(rawValue: config.window) ?? .hann
        self.window = windowFn.generate(size: config.winLength)
    }

    /// Compute log mel spectrogram from audio samples
    /// - Parameter audio: Raw audio samples (1D array, 16kHz mono)
    /// - Returns: Log mel spectrogram of shape [1, time, melFeatures]
    public func logMelSpectrogram(_ audio: MLXArray) -> MLXArray {
        var x = audio

        // Optional padding
        if config.padTo > 0 && x.dim(0) < config.padTo {
            let padLength = config.padTo - x.dim(0)
            x = padded(x, widths: [.init((0, padLength))], value: MLXArray(config.padValue))
        }

        // Apply preemphasis filter (mode-dependent): y[n] = x[n] - α*x[n-1]
        if config.activePreemph > 0 {
            let first = x[0..<1]
            let rest = x[1...] - config.activePreemph * x[..<(x.dim(0) - 1)]
            x = concatenated([first, rest], axis: 0)
        }

        // Compute STFT
        let stftResult = stft(
            x,
            nFft: config.nFft,
            hopLength: config.hopLength,
            winLength: config.winLength,
            window: window
        )

        // Compute magnitude squared (power spectrum)
        let powerSpec = abs(stftResult).square()

        // Apply filterbank: [nMels, nFreqs] @ [numFrames, nFreqs].T -> [nMels, numFrames]
        let melSpec = matmul(melFilters, powerSpec.T)

        // Log compression
        var logMelSpec = log(melSpec + 1e-5)

        // Apply spectral tilt compensation for whispered speech
        // Adds linear ramp to boost higher frequency bins (compensates for flatter spectral envelope)
        if config.activeSpectralTilt > 0 {
            let numMels = Float(logMelSpec.dim(0))
            // Create linear ramp: [0, tilt/numMels, 2*tilt/numMels, ..., tilt*(numMels-1)/numMels]
            let indices = MLXArray(Array(0..<Int(numMels)).map { Float($0) })
            let tiltRamp = (indices / numMels) * config.activeSpectralTilt
            // Expand to [numMels, 1] for broadcasting across time frames
            let tiltRamp2D = tiltRamp.expandedDimensions(axis: 1)
            logMelSpec = logMelSpec + tiltRamp2D
        }

        // Normalize
        if config.normalize == "per_feature" {
            // Per-feature (per mel bin) normalization
            let mean = logMelSpec.mean(axis: 1, keepDims: true)
            let std = logMelSpec.std(axis: 1, keepDims: true)
            logMelSpec = (logMelSpec - mean) / (std + 1e-5)
        } else {
            // Global normalization
            let mean = logMelSpec.mean()
            let std = logMelSpec.std()
            logMelSpec = (logMelSpec - mean) / (std + 1e-5)
        }

        // Transpose to [time, melFeatures] and add batch dimension
        let result = logMelSpec.T.expandedDimensions(axis: 0)

        return result
    }

    /// Convenience method to process Float array
    public func logMelSpectrogram(_ audio: [Float]) -> MLXArray {
        let x = MLXArray(audio)
        return logMelSpectrogram(x)
    }
}

// MARK: - Extension for std calculation

extension MLXArray {
    /// Compute standard deviation along axis
    func std(axis: Int? = nil, keepDims: Bool = false) -> MLXArray {
        if let axis = axis {
            let mean = self.mean(axis: axis, keepDims: true)
            let variance = ((self - mean).square()).mean(axis: axis, keepDims: keepDims)
            return MLX.sqrt(variance)
        } else {
            let mean = self.mean()
            let variance = ((self - mean).square()).mean()
            return MLX.sqrt(variance)
        }
    }
}
