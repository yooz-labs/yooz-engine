// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Parakeet TDT Speech-to-Text Model
/// Combines Conformer encoder with RNNT decoder using Token-and-Duration Transducer
public final class ParakeetModel: Module {
    public let config: ParakeetTDTConfig

    @ModuleInfo(key: "encoder") var encoder: Conformer
    @ModuleInfo(key: "decoder") var decoder: PredictNetwork
    @ModuleInfo(key: "joint") var joint: JointNetwork

    private let preprocessor: AudioPreprocessor
    private let tdtDecoder: TDTDecoder

    public init(config: ParakeetTDTConfig) {
        self.config = config

        self._encoder.wrappedValue = Conformer(config: config.encoder)
        self._decoder.wrappedValue = PredictNetwork(config: config.decoder, blankAsPad: true)
        self._joint.wrappedValue = JointNetwork(config: config.joint)

        self.preprocessor = AudioPreprocessor(config: config.preprocessor)
        self.tdtDecoder = TDTDecoder(
            vocabulary: config.joint.vocabulary,
            durations: config.decoding.durations,
            subsamplingFactor: config.encoder.subsamplingFactor,
            sampleRate: config.preprocessor.sampleRate,
            hopLength: config.preprocessor.hopLength,
            maxSymbols: config.decoding.maxSymbols
        )
    }

    /// Load model from local directory
    /// - Parameters:
    ///   - directory: Path to directory containing config.json and model.safetensors
    ///   - dtype: Data type for model weights (default: bfloat16)
    /// - Returns: Loaded ParakeetModel
    public static func fromDirectory(
        _ directory: URL,
        dtype: DType = .bfloat16
    ) throws -> ParakeetModel {
        // Load config using NeMo parser (handles nested structure)
        let configURL = directory.appendingPathComponent("config.json")
        let config = try NeMoConfigParser.parse(from: configURL)

        // Initialize model
        let model = ParakeetModel(config: config)

        // Load weights
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        var weights = try loadArrays(url: weightsURL)

        // Sanitize weight keys
        weights = model.sanitizeWeights(weights)

        // Convert to target dtype
        weights = weights.mapValues { $0.asType(dtype) }

        // Update model parameters
        let params = ModuleParameters.unflattened(weights)
        try model.update(parameters: params, verify: .noUnusedKeys)

        eval(model)
        return model
    }

    /// Sanitize weight keys from Python/safetensors format to Swift module format
    /// - Parameter weights: Raw weights dictionary from safetensors
    /// - Returns: Sanitized weights dictionary matching our module structure
    private func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            // === Decoder/Prediction Network ===
            // decoder.prediction.embed.weight -> decoder.embed.weight
            // decoder.prediction.dec_rnn.lstm.X.* -> decoder.dec_rnn.lstm.X.*
            newKey = newKey.replacingOccurrences(of: "decoder.prediction.", with: "decoder.")

            // === Joint Network ===
            // joint.joint_net.2.* -> joint.out.*
            // (joint.enc.* and joint.pred.* already match our keys)
            newKey = newKey.replacingOccurrences(of: "joint.joint_net.2.", with: "joint.out.")

            // === Subsampling conv layer index remapping ===
            // Python uses indices with gaps (for ReLU): conv.0, conv.2, conv.3, conv.5, conv.6
            // Swift uses consecutive: conv.0, conv.1, conv.2, conv.3, conv.4
            // Map: 0->0, 2->1, 3->2, 5->3, 6->4, etc.
            if newKey.contains("pre_encode.conv.") {
                // Extract the layer index and remap
                let pythonIndices = [0, 2, 3, 5, 6, 8, 9]  // Add more if needed
                let swiftIndices =  [0, 1, 2, 3, 4, 5, 6]
                for (pythonIdx, swiftIdx) in zip(pythonIndices, swiftIndices) {
                    let pythonPattern = "pre_encode.conv.\(pythonIdx)."
                    let swiftPattern = "pre_encode.conv.\(swiftIdx)."
                    if newKey.contains(pythonPattern) {
                        newKey = newKey.replacingOccurrences(of: pythonPattern, with: swiftPattern)
                        break
                    }
                }
            }

            sanitized[newKey] = value
        }

        return sanitized
    }

    /// Transcribe audio samples to text
    /// - Parameter audio: Audio samples [time] at config.preprocessor.sampleRate Hz
    /// - Returns: Transcription result with aligned tokens
    public func transcribe(_ audio: MLXArray) -> TranscriptionResult {
        // Compute mel spectrogram
        let mel = preprocessor.logMelSpectrogram(audio)

        // Encode
        let (features, lengths) = encoder(mel)
        eval(features, lengths)

        // Decode
        let tokens = tdtDecoder.decode(
            features: features,
            lengths: lengths,
            decoder: decoder,
            joint: joint
        )

        // Build result
        return TranscriptionResult(tokens: tokens)
    }

    /// Transcribe audio from Float array
    /// - Parameter audio: Audio samples at config.preprocessor.sampleRate Hz
    /// - Returns: Transcription result
    public func transcribe(_ audio: [Float]) -> TranscriptionResult {
        let x = MLXArray(audio)
        return transcribe(x)
    }

    // MARK: - Streaming Transcription

    /// Streaming transcription with encoder cache
    /// Computes mel for ALL audio, but only encodes NEW frames (using cache for context)
    /// - Parameters:
    ///   - allAudio: All accumulated audio samples (not just new ones)
    ///   - encoderCache: Array of ConformerCache for each encoder layer
    ///   - previousEncoderOutput: Accumulated encoder outputs from previous calls
    /// - Returns: (all tokens, new encoder output to append)
    public func transcribeStreaming(
        _ allAudio: [Float],
        encoderCache: [ConformerCache],
        previousEncoderOutput: MLXArray?
    ) -> (tokens: [AlignedToken], newEncoderOutput: MLXArray) {
        // Compute mel spectrogram for ALL audio (needed for correct STFT overlap)
        let allMel = preprocessor.logMelSpectrogram(MLXArray(allAudio))

        // Get cache offset (how many encoder frames we've already encoded)
        // Note: offset is in encoder frames, we need to convert to mel frames
        let encoderOffset = encoderCache.first?.offset ?? 0
        let subsamplingFactor = config.encoder.subsamplingFactor
        let melOffset = encoderOffset * subsamplingFactor

        // If cache has frames, slice mel to get only NEW frames
        let melToEncode: MLXArray
        if melOffset > 0 && allMel.dim(1) > melOffset {
            // Only encode new frames (after mel offset)
            melToEncode = allMel[0..., melOffset..., 0...]
        } else {
            melToEncode = allMel
        }

        // Encode new mel frames with cache (cache provides attention context)
        let (newEncoderOutput, _) = encoder(melToEncode, cache: encoderCache)
        eval(newEncoderOutput)

        // Combine with previous encoder outputs for decoding
        let allEncoderOutput: MLXArray
        if let prevOutput = previousEncoderOutput {
            allEncoderOutput = concatenated([prevOutput, newEncoderOutput], axis: 1)
        } else {
            allEncoderOutput = newEncoderOutput
        }

        // Get total length
        let totalLen = full([allEncoderOutput.dim(0)], values: Int32(allEncoderOutput.dim(1)))

        // Decode from all encoder outputs
        let tokens = tdtDecoder.decode(
            features: allEncoderOutput,
            lengths: totalLen,
            decoder: decoder,
            joint: joint
        )

        return (tokens, newEncoderOutput)
    }

    /// Create encoder caches for streaming
    public func createEncoderCaches() -> [ConformerCache] {
        return encoder.createCaches()
    }

    /// Create rotating encoder caches for long audio streaming
    public func createRotatingEncoderCaches(capacity: Int, dropSize: Int = 0) -> [RotatingConformerCache] {
        return encoder.createRotatingCaches(capacity: capacity, dropSize: dropSize)
    }

    /// Decode encoder features to tokens using TDT greedy search
    /// - Parameters:
    ///   - features: Encoder output [1, time, hidden]
    ///   - length: Number of valid encoder frames
    /// - Returns: Array of aligned tokens
    public func tdtDecode(features: MLXArray, length: Int) -> [AlignedToken] {
        let lengths = full([features.dim(0)], values: Int32(length))
        return tdtDecoder.decode(
            features: features,
            lengths: lengths,
            decoder: decoder,
            joint: joint
        )
    }
}

/// Result of transcription
public struct TranscriptionResult: Sendable {
    public let tokens: [AlignedToken]

    /// Full transcription text
    public var text: String {
        tokens.map { $0.text }.joined().trimmingCharacters(in: .whitespaces)
    }

    /// Group tokens into sentences (by punctuation or pauses)
    public var sentences: [AlignedSentence] {
        // Simple sentence grouping by pauses > 0.5s
        var sentences: [AlignedSentence] = []
        var currentTokens: [AlignedToken] = []

        for token in tokens {
            if let last = currentTokens.last {
                let gap = token.start - last.end
                if gap > 0.5 && !currentTokens.isEmpty {
                    let text = currentTokens.map { $0.text }.joined()
                    sentences.append(AlignedSentence(text: text, tokens: currentTokens))
                    currentTokens = []
                }
            }
            currentTokens.append(token)
        }

        if !currentTokens.isEmpty {
            let text = currentTokens.map { $0.text }.joined()
            sentences.append(AlignedSentence(text: text, tokens: currentTokens))
        }

        return sentences
    }
}
