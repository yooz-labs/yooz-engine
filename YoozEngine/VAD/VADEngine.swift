// VADEngine.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import CoreML
import Accelerate
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "VADEngine")

/// Voice Activity Detection using Silero VAD CoreML model.
///
/// Detects speech segments in audio by sliding 512-sample windows (32ms at 16kHz)
/// over the input and returning time-stamped speech regions.
actor VADEngine {

    // MARK: - Singleton

    static let shared = VADEngine()

    // MARK: - Constants

    static let sampleRate: Int = 16000
    static let windowSize: Int = 512
    private let speechThreshold: Float = 0.5
    private let preSpeechFrames: Int = 2
    private let postSpeechFrames: Int = 8

    // MARK: - State

    private var model: MLModel?
    private(set) var isLoaded: Bool = false
    private var hiddenState: MLMultiArray?
    private var cellState: MLMultiArray?

    // MARK: - Lifecycle

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
        isLoaded = true
        resetState()
        logger.info("Silero VAD model loaded")
    }

    /// Reset RNN hidden state (call between recordings).
    func reset() {
        resetState()
    }

    // MARK: - Detection

    /// Detect speech segments in audio samples.
    ///
    /// Slides a 512-sample window across the input, runs VAD inference on each
    /// window, applies smoothing, and returns contiguous speech regions.
    ///
    /// - Parameter samples: Audio samples at 16kHz.
    /// - Returns: Array of detected speech segments with start/end in milliseconds.
    func detect(samples: [Float]) -> [VADSegmentResult] {
        guard isLoaded, model != nil else {
            logger.warning("VAD model not loaded, returning empty segments")
            return []
        }

        resetState()

        let windowSize = Self.windowSize
        let sampleRate = Self.sampleRate
        guard samples.count >= windowSize else { return [] }

        var segments: [VADSegmentResult] = []
        var speechStart: Int?
        var speechFrameCount = 0
        var silenceFrameCount = 0
        var inSpeech = false

        let frameCount = samples.count / windowSize
        for frameIndex in 0..<frameCount {
            let offset = frameIndex * windowSize
            let window = Array(samples[offset..<offset + windowSize])
            let probability = getSpeechProbability(window)
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

        // Close any open segment
        if inSpeech, let start = speechStart {
            let endMs = frameCount * windowSize * 1000 / sampleRate
            segments.append(VADSegmentResult(
                startMs: start,
                endMs: endMs,
                probability: speechThreshold
            ))
        }

        logger.debug("VAD detected \(segments.count) segment(s) in \(samples.count) samples")
        return segments
    }

    // MARK: - Private

    private func getSpeechProbability(_ samples: [Float]) -> Float {
        guard let model = model else {
            return getEnergyFallback(samples)
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
                "h": hiddenState!,
                "c": cellState!
            ])

            let output = try model.prediction(from: input)

            if let outputArray = output.featureValue(for: "output")?.multiArrayValue {
                let probability = outputArray[0].floatValue

                if let newH = output.featureValue(for: "hn")?.multiArrayValue,
                   let newC = output.featureValue(for: "cn")?.multiArrayValue {
                    hiddenState = newH
                    cellState = newC
                }

                return probability
            }
        } catch {
            logger.error("VAD inference error: \(error.localizedDescription)")
            return getEnergyFallback(samples)
        }

        return 0.0
    }

    private func getEnergyFallback(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0.0 }
        var rms: Float = 0
        vDSP_rmsqv(samples, 1, &rms, vDSP_Length(samples.count))
        return rms > 0.005 ? 0.9 : 0.1
    }

    private func resetState() {
        do {
            hiddenState = try MLMultiArray(shape: [2, 1, 64], dataType: .float32)
            cellState = try MLMultiArray(shape: [2, 1, 64], dataType: .float32)
            for i in 0..<hiddenState!.count {
                hiddenState![i] = 0.0
                cellState![i] = 0.0
            }
        } catch {
            logger.error("Failed to reset VAD state: \(error.localizedDescription)")
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

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Silero VAD model not found in app bundle"
        }
    }
}
