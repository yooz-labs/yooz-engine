// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Multi-layer LSTM wrapper
/// Stacks multiple LSTM layers with proper hidden state handling
public class MultiLayerLSTM: Module {
    public let hiddenSize: Int
    public let numLayers: Int
    public let batchFirst: Bool

    @ModuleInfo(key: "lstm") var lstmLayers: [LSTM]

    public init(
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        bias: Bool = true,
        batchFirst: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.batchFirst = batchFirst

        // Build stacked LSTM layers
        var layers: [LSTM] = []
        for i in 0..<numLayers {
            let layerInputSize = (i == 0) ? inputSize : hiddenSize
            layers.append(LSTM(inputSize: layerInputSize, hiddenSize: hiddenSize, bias: bias))
        }
        self._lstmLayers.wrappedValue = layers
    }

    /// Forward pass through stacked LSTM layers
    /// - Parameters:
    ///   - x: Input tensor [batch, seq, input_size] if batchFirst, else [seq, batch, input_size]
    ///   - hc: Optional tuple of (hidden, cell) states, each [num_layers, batch, hidden_size]
    /// - Returns: (output, (final_hidden, final_cell))
    ///   - output: [batch, seq, hidden_size] if batchFirst
    ///   - final_hidden/cell: [num_layers, batch, hidden_size]
    public func callAsFunction(
        _ x: MLXArray,
        hc: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        var input = x

        // Transpose if batch_first to [seq, batch, features] for LSTM
        if batchFirst {
            input = input.transposed(1, 0, 2)
        }

        // Extract per-layer hidden/cell states
        var layerH: [MLXArray?]
        var layerC: [MLXArray?]

        if let (h, c) = hc {
            layerH = (0..<numLayers).map { h[$0] }
            layerC = (0..<numLayers).map { c[$0] }
        } else {
            layerH = Array(repeating: nil, count: numLayers)
            layerC = Array(repeating: nil, count: numLayers)
        }

        var outputs = input
        var nextH: [MLXArray] = []
        var nextC: [MLXArray] = []

        // Pass through each layer
        for i in 0..<numLayers {
            let layer = lstmLayers[i]
            let (allH, allC) = layer(outputs, hidden: layerH[i], cell: layerC[i])

            outputs = allH
            // Take the last hidden/cell state from sequence dimension
            nextH.append(allH[allH.dim(0) - 1])
            nextC.append(allC[allC.dim(0) - 1])
        }

        // Transpose back if batch_first
        if batchFirst {
            outputs = outputs.transposed(1, 0, 2)
        }

        // Stack layer states: [num_layers, batch, hidden_size]
        let finalH = stacked(nextH, axis: 0)
        let finalC = stacked(nextC, axis: 0)

        return (outputs, (finalH, finalC))
    }
}
