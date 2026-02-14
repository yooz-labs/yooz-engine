// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Prediction Network for RNNT/TDT
/// Processes previous token embeddings through LSTM layers
public class PredictNetwork: Module {
    public let predHidden: Int
    public let blankAsPad: Bool

    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "dec_rnn") var decRnn: MultiLayerLSTM

    public init(config: PredictConfig, blankAsPad: Bool = true) {
        self.predHidden = config.predHidden
        self.blankAsPad = blankAsPad

        let vocabSize = blankAsPad ? config.vocabSize + 1 : config.vocabSize
        let rnnHidden = config.rnnHiddenSize

        self._embed.wrappedValue = Embedding(
            embeddingCount: vocabSize,
            dimensions: config.predHidden
        )

        self._decRnn.wrappedValue = MultiLayerLSTM(
            inputSize: config.predHidden,
            hiddenSize: rnnHidden,
            numLayers: config.predRnnLayers,
            bias: true,
            batchFirst: true
        )
    }

    /// Forward pass
    /// - Parameters:
    ///   - y: Previous token indices [batch, seq] or nil for initial state
    ///   - hc: Optional tuple of (hidden, cell) states
    /// - Returns: (output, (hidden, cell))
    public func callAsFunction(
        _ y: MLXArray?,
        hc: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let embeddedY: MLXArray

        if let y = y {
            embeddedY = embed(y)
        } else {
            // Initial state: use zeros
            let batch = hc?.0.dim(1) ?? 1
            embeddedY = MLXArray.zeros([batch, 1, predHidden])
        }

        return decRnn(embeddedY, hc: hc)
    }
}
