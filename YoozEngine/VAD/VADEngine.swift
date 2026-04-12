// VADEngine.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import CoreML
import Accelerate
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "VADEngine")

/// Voice Activity Detection using Silero VAD v6.0.0 CoreML model.
///
/// Detects speech segments in audio by sliding 512-sample windows (32ms at 16kHz)
/// over the input and returning time-stamped speech regions.
///
/// ## Client-side chunk processing
///
/// Client apps (e.g. yooz-whisper) use the engine's VAD to drive chunk-based
/// transcription pipelines. The recommended pattern is:
///
/// 1. **Stream audio** to the VAD endpoint in real time.
/// 2. **Determine chunk boundaries** using one of:
///    - **Silence**: VAD detects sustained silence (post-speech smoothing).
///    - **Word count**: 25-50 words accumulated from streaming STT.
///    - **Punctuation**: sentence-ending punctuation (`.`, `!`, `?`) after 25+ words.
///    - **Recording ended**: user stops recording.
/// 3. **Overlap context**: prepend 0.8 seconds (12,800 samples at 16kHz) from the
///    previous chunk to provide acoustic context for edge-word transcription.
/// 4. **Process pipeline**: VAD -> STT (batch) -> Grammar -> TouchUp.
/// 5. **Track pending chunks** with a 30-second timeout to avoid blocking on
///    slow batch transcriptions.
///
/// This pattern is implemented in yooz-whisper's `ChunkProcessor` and should be
/// replicated in any thin client that performs real-time transcription via the engine.
actor VADEngine {

    // MARK: - Singleton

    static let shared = VADEngine()

    // MARK: - Constants

    static let sampleRate: Int = 16000
    static let windowSize: Int = 512
    private let speechThreshold: Float = 0.5
    private let preSpeechFrames: Int
    private let postSpeechFrames: Int

    // MARK: - State

    private var model: MLModel?
    private(set) var isLoaded: Bool = false
    private var hiddenState: MLMultiArray?
    private var cellState: MLMultiArray?

    // MARK: - Lifecycle

    /// Create a VAD engine with configurable smoothing parameters.
    ///
    /// - Parameters:
    ///   - preSpeechFrames: Number of consecutive speech frames required before
    ///     transitioning to speech state. Default is 2 (64ms).
    ///   - postSpeechFrames: Number of consecutive silence frames required before
    ///     transitioning to silence state. Default is 8 (256ms).
    init(preSpeechFrames: Int = 2, postSpeechFrames: Int = 8) {
        self.preSpeechFrames = preSpeechFrames
        self.postSpeechFrames = postSpeechFrames
    }

    /// Load the Silero VAD CoreML model from the app bundle.
    func load() throws {
        guard !isLoaded else { return }

        guard let modelURL = Bundle.main.url(
            forResource: "silero-vad-unified-v6.0.0",
            withExtension: "mlpackage"
        ) ?? Bundle.main.url(
            forResource: "silero-vad-unified-v6.0.0",
            withExtension: "mlmodelc"
        ) else {
            throw VADError.modelNotFound
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        model = try MLModel(contentsOf: modelURL, configuration: config)
        try resetState()
        isLoaded = true
        logger.info("Silero VAD model loaded")
    }

    /// Reset RNN hidden state (call between recordings).
    func reset() throws {
        try resetState()
    }

    // MARK: - Detection

    /// Detect speech segments in audio samples.
    ///
    /// Slides a 512-sample window across the input, runs VAD inference on each
    /// window, applies pre/post-speech smoothing, and returns contiguous speech
    /// regions. Falls back to energy-based detection if the CoreML model is
    /// unavailable or inference fails.
    ///
    /// - Parameters:
    ///   - samples: Audio samples at 16kHz.
    ///   - resetState: When true (default), resets the RNN hidden/cell state
    ///     before detection. Set to false when sending consecutive chunks from
    ///     the same recording to preserve inter-frame state continuity.
    /// - Returns: Array of detected speech segments with start/end in milliseconds.
    /// - Throws: ``VADError/modelNotLoaded`` if the model has not been loaded.
    func detect(samples: [Float], resetState shouldReset: Bool = true) throws -> [VADSegmentResult] {
        guard isLoaded else {
            throw VADError.modelNotLoaded
        }

        if shouldReset {
            try resetState()
        }

        let windowSize = Self.windowSize
        let sampleRate = Self.sampleRate
        guard samples.count >= windowSize else { return [] }

        var segments: [VADSegmentResult] = []
        var speechStart: Int?
        var speechFrameCount = 0
        var silenceFrameCount = 0
        var inSpeech = false
        var fallbackCount = 0

        let frameCount = samples.count / windowSize
        for frameIndex in 0..<frameCount {
            let offset = frameIndex * windowSize
            let window = Array(samples[offset..<offset + windowSize])
            let probability = getSpeechProbability(window, fallbackCount: &fallbackCount)
            let rawSpeech = probability > speechThreshold

            if rawSpeech {
                speechFrameCount += 1
                silenceFrameCount = 0
                if speechFrameCount >= preSpeechFrames && !inSpeech {
                    inSpeech = true
                    // Backtrack to include pre-speech frames
                    let startFrame = max(0, frameIndex - preSpeechFrames + 1)
                    speechStart = startFrame * windowSize * 1000 / sampleRate
                }
            } else {
                silenceFrameCount += 1
                speechFrameCount = 0
                if silenceFrameCount >= postSpeechFrames && inSpeech {
                    inSpeech = false
                    let endFrame = max(0, frameIndex - postSpeechFrames + 1)
                    let endMs = endFrame * windowSize * 1000 / sampleRate
                    if let start = speechStart {
                        segments.append(VADSegmentResult(
                            startMs: start,
                            endMs: endMs,
                            probability: probability
                        ))
                    }
                    speechStart = nil
                }
            }
        }

        // Close any open segment; use threshold as probability since
        // the last per-frame value is unavailable after loop exit
        if inSpeech, let start = speechStart {
            let endMs = frameCount * windowSize * 1000 / sampleRate
            segments.append(VADSegmentResult(
                startMs: start,
                endMs: endMs,
                probability: speechThreshold
            ))
        }

        if fallbackCount > 0 {
            logger.warning("VAD used energy-based fallback for \(fallbackCount)/\(frameCount) frames")
        }
        logger.debug("VAD detected \(segments.count) segment(s) in \(samples.count) samples")
        return segments
    }

    // MARK: - Private

    /// Get speech probability for a single window of audio samples.
    ///
    /// Uses the Silero CoreML model for inference; falls back to energy-based
    /// detection (RMS threshold) if inference fails at runtime.
    /// Called from detect() which guards on isLoaded, so model/state should be non-nil.
    private func getSpeechProbability(_ samples: [Float], fallbackCount: inout Int) -> Float {
        guard let model, let h = hiddenState, let c = cellState else {
            if fallbackCount == 0 {
                logger.error("VAD state unexpectedly nil, using energy-based fallback")
            }
            fallbackCount += 1
            return getEnergyBasedProbability(samples)
        }

        do {
            let inputArray = try MLMultiArray(
                shape: [1, NSNumber(value: samples.count)],
                dataType: .float32
            )
            for (i, sample) in samples.enumerated() {
                inputArray[i] = NSNumber(value: sample)
            }

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input": inputArray,
                "h": h,
                "c": c
            ])

            let output = try model.prediction(from: input)

            guard let outputArray = output.featureValue(for: "output")?.multiArrayValue else {
                if fallbackCount == 0 {
                    logger.error("VAD output extraction failed, using energy-based fallback")
                }
                fallbackCount += 1
                return getEnergyBasedProbability(samples)
            }

            let probability = outputArray[0].floatValue

            if let newH = output.featureValue(for: "hn")?.multiArrayValue,
               let newC = output.featureValue(for: "cn")?.multiArrayValue {
                hiddenState = newH
                cellState = newC
            }

            return probability
        } catch {
            if fallbackCount == 0 {
                logger.error("VAD inference failed: \(error.localizedDescription), using energy-based fallback")
            }
            fallbackCount += 1
            return getEnergyBasedProbability(samples)
        }
    }

    /// Energy-based fallback when the CoreML model is unavailable.
    ///
    /// Computes the root mean square (RMS) of the samples and returns a high
    /// probability (0.9) when above a noise-floor threshold, or low (0.1) otherwise.
    private func getEnergyBasedProbability(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0.0 }

        var rms: Float = 0
        vDSP_rmsqv(samples, 1, &rms, vDSP_Length(samples.count))

        let threshold: Float = 0.005
        return rms > threshold ? 0.9 : 0.1
    }

    private func resetState() throws {
        hiddenState = try MLMultiArray(shape: [2, 1, 64], dataType: .float32)
        cellState = try MLMultiArray(shape: [2, 1, 64], dataType: .float32)
        for i in 0..<hiddenState!.count {
            hiddenState![i] = 0.0
            cellState![i] = 0.0
        }
    }
}

// MARK: - Types

struct VADSegmentResult {
    let startMs: Int
    let endMs: Int
    let probability: Float
}

enum VADError: LocalizedError {
    case modelNotFound
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Silero VAD model not found in app bundle"
        case .modelNotLoaded:
            return "VAD model not loaded; call load() first"
        }
    }
}
