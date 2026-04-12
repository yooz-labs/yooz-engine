// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// RNNT (Recurrent Neural Network Transducer) Greedy Decoder
/// Standard RNNT decoding without duration prediction (unlike TDT)
/// Advances by 1 frame when blank is predicted
public struct RNNTDecoder {
    public let vocabulary: [String]
    public let subsamplingFactor: Int
    public let sampleRate: Int
    public let hopLength: Int
    public let maxSymbols: Int?

    public init(
        vocabulary: [String],
        subsamplingFactor: Int,
        sampleRate: Int,
        hopLength: Int,
        maxSymbols: Int? = nil
    ) {
        self.vocabulary = vocabulary
        self.subsamplingFactor = subsamplingFactor
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.maxSymbols = maxSymbols
    }

    /// Decode encoded features using RNNT greedy search
    /// - Parameters:
    ///   - features: Encoder output [1, time, hidden]
    ///   - lengths: Sequence lengths [1]
    ///   - decoder: PredictNetwork for token prediction
    ///   - joint: JointNetwork for combining encoder and decoder outputs
    /// - Returns: Array of aligned tokens with timing information
    public func decode(
        features: MLXArray,
        lengths: MLXArray,
        decoder: PredictNetwork,
        joint: JointNetwork
    ) -> [AlignedToken] {
        // Input validation
        guard lengths.size > 0 else {
            #if DEBUG
            print("[RNNTDecoder] Warning: Empty lengths array, returning empty result")
            #endif
            return []
        }

        guard !vocabulary.isEmpty else {
            #if DEBUG
            print("[RNNTDecoder] Error: Empty vocabulary")
            #endif
            return []
        }

        guard features.ndim >= 2 else {
            #if DEBUG
            print("[RNNTDecoder] Error: Features must have at least 2 dimensions, got \(features.ndim)")
            #endif
            return []
        }

        let maxLength = Int(lengths[0].item(Int32.self))
        guard maxLength > 0 else {
            #if DEBUG
            print("[RNNTDecoder] Warning: Max length is 0, returning empty result")
            #endif
            return []
        }

        let blankToken = vocabulary.count

        // Pre-compute encoder projection for all time steps
        let allEncProjected = joint.projectEncoder(features)
        eval(allEncProjected)

        var lastToken = blankToken
        var hypothesis: [AlignedToken] = []
        hypothesis.reserveCapacity(maxLength / 4)
        var time = 0
        var symbolsAtTime = 0
        var decoderHidden: (MLXArray, MLXArray)? = nil

        let tokenizer = Tokenizer(vocabulary: vocabulary)

        // Time conversion factor
        let timeToSeconds = Float(subsamplingFactor) / Float(sampleRate) * Float(hopLength)

        while time < maxLength {
            // Get encoder projection for current time step
            let encProjSlice = allEncProjected[0..., time..<(time + 1), 0...]

            // Prepare input token (nil for blank/start)
            let currentToken: MLXArray? = (lastToken != blankToken)
                ? MLXArray([Int32(lastToken)]).reshaped([1, 1])
                : nil

            // Run decoder
            let (decoderOutput, (hidden, cell)) = decoder(currentToken, hc: decoderHidden)

            // Ensure dtype compatibility
            let targetDtype = encProjSlice.dtype
            let decoderOutputCast = decoderOutput.dtype == targetDtype
                ? decoderOutput
                : decoderOutput.asType(targetDtype)
            let proposedHidden = decoderHidden?.0.dtype == targetDtype
                ? (hidden, cell)
                : (hidden.asType(targetDtype), cell.asType(targetDtype))

            // Run joint network
            let jointOutput = joint.forwardWithProjectedEncoder(encProjSlice, pred: decoderOutputCast)

            // Get token logits (excluding duration outputs since RNNT has none)
            let tokenLogits = jointOutput[0, 0, 0, 0..<(blankToken + 1)]
            let tokenArgmax = argMax(tokenLogits)
            eval(tokenArgmax)

            let predToken = Int(tokenArgmax.item(Int32.self))

            if predToken != blankToken {
                // Non-blank token: emit and update decoder state
                let startTime = Float(time) * timeToSeconds
                // Default duration is 1 frame (will be refined by next token or end)
                let duration = timeToSeconds

                let token = AlignedToken(
                    id: predToken,
                    text: tokenizer.decode(token: predToken),
                    start: startTime,
                    duration: duration
                )
                hypothesis.append(token)

                lastToken = predToken
                decoderHidden = proposedHidden
                symbolsAtTime += 1

                // Check max symbols per time step
                let maxSyms = maxSymbols ?? 10
                if symbolsAtTime >= maxSyms {
                    time += 1
                    symbolsAtTime = 0
                }
            } else {
                // Blank: advance to next time step
                time += 1
                symbolsAtTime = 0
            }
        }

        return hypothesis
    }
}
