// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// TDT (Token-and-Duration Transducer) Greedy Decoder
/// Performs frame-by-frame decoding with duration prediction
public struct TDTDecoder {
    public let vocabulary: [String]
    public let durations: [Int]
    public let subsamplingFactor: Int
    public let sampleRate: Int
    public let hopLength: Int
    public let maxSymbols: Int?

    public init(
        vocabulary: [String],
        durations: [Int],
        subsamplingFactor: Int,
        sampleRate: Int,
        hopLength: Int,
        maxSymbols: Int? = nil
    ) {
        self.vocabulary = vocabulary
        self.durations = durations
        self.subsamplingFactor = subsamplingFactor
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.maxSymbols = maxSymbols
    }

    /// Decode encoded features using TDT greedy search
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
        let maxLength = Int(lengths[0].item(Int32.self))
        let blankToken = vocabulary.count

        // H1 Optimization: Pre-compute encoder projection for all time steps
        // This avoids redundant projection computation in the loop
        let allEncProjected = joint.projectEncoder(features)
        eval(allEncProjected)

        var lastToken = blankToken
        var hypothesis: [AlignedToken] = []
        hypothesis.reserveCapacity(maxLength / 4)  // H2: Pre-allocate with estimate
        var time = 0
        var newSymbols = 0
        var decoderHidden: (MLXArray, MLXArray)? = nil

        let tokenizer = Tokenizer(vocabulary: vocabulary)

        // Pre-compute time conversion factor (H7: avoid repeated computation)
        let timeToSeconds = Float(subsamplingFactor) / Float(sampleRate) * Float(hopLength)

        while time < maxLength {
            // Use pre-computed encoder projection slice
            let encProjSlice = allEncProjected[0..., time..<(time + 1), 0...]

            // Prepare input token (nil for blank/start)
            let currentToken: MLXArray? = (lastToken != blankToken)
                ? MLXArray([Int32(lastToken)]).reshaped([1, 1])
                : nil

            // Run decoder
            let (decoderOutput, (hidden, cell)) = decoder(currentToken, hc: decoderHidden)

            // H7 Optimization: Only cast if dtypes differ
            let targetDtype = encProjSlice.dtype
            let decoderOutputCast = decoderOutput.dtype == targetDtype
                ? decoderOutput
                : decoderOutput.asType(targetDtype)
            let proposedHidden = decoderHidden?.0.dtype == targetDtype
                ? (hidden, cell)
                : (hidden.asType(targetDtype), cell.asType(targetDtype))

            // Run joint network with pre-computed encoder projection
            let jointOutput = joint.forwardWithProjectedEncoder(encProjSlice, pred: decoderOutputCast)

            // H3 Optimization: Batch argmax operations to reduce GPU-CPU sync
            // Extract logits
            let tokenLogits = jointOutput[0, 0, 0, 0..<(blankToken + 1)]
            let durationLogits = jointOutput[0, 0, 0, (blankToken + 1)...]

            // Compute both argmax in one batch
            let tokenArgmax = argMax(tokenLogits)
            let durationArgmax = argMax(durationLogits)

            // Single eval call for both results
            eval(tokenArgmax, durationArgmax)

            let predToken = Int(tokenArgmax.item(Int32.self))
            let durationIdx = Int(durationArgmax.item(Int32.self))

            // Non-blank token: emit and update decoder state
            if predToken != blankToken {
                let startTime = Float(time) * timeToSeconds
                let duration = Float(durations[durationIdx]) * timeToSeconds

                let token = AlignedToken(
                    id: predToken,
                    text: tokenizer.decode(token: predToken),
                    start: startTime,
                    duration: duration
                )
                hypothesis.append(token)

                lastToken = predToken
                decoderHidden = proposedHidden
            }

            // Advance time by predicted duration
            time += durations[durationIdx]
            newSymbols += 1

            // Handle max symbols limit (default to 10 if not set)
            if durations[durationIdx] != 0 {
                newSymbols = 0
            } else {
                let maxSyms = maxSymbols ?? 10
                if newSymbols >= maxSyms {
                    time += 1
                    newSymbols = 0
                }
            }
        }

        return hypothesis
    }
}
