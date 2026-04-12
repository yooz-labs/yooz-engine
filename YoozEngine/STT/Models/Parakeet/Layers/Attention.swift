// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

// MARK: - Relative Positional Encoding

/// Relative positional encoding for Conformer
/// Creates sinusoidal position embeddings centered around current position
public class RelPositionalEncoding: Module {
    public let dModel: Int
    public var maxLen: Int
    public let scale: Float

    private var pe: MLXArray

    public init(dModel: Int, maxLen: Int = 5000, scaleInput: Bool = true) {
        precondition(dModel % 2 == 0, "dModel must be even")
        precondition(maxLen > 0, "maxLen must be positive")

        self.dModel = dModel
        self.maxLen = maxLen
        self.scale = scaleInput ? sqrt(Float(dModel)) : 1.0
        self.pe = MLXArray.zeros([1, 1, dModel])

        super.init()
        calculatePE()
    }

    private func calculatePE() {
        // Create positions from maxLen-1 down to -maxLen+1
        let positions = MLXArray(stride(from: maxLen - 1, through: -(maxLen - 1), by: -1))
            .expandedDimensions(axis: 1)
            .asType(.float32)

        // Compute division term: exp(2i * -log(10000) / d_model)
        let divTerm = exp(
            MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32)
                * Float(-log(10000.0) / Float(dModel))
        )

        // sin for even indices, cos for odd indices
        let sinValues = sin(positions * divTerm)
        let cosValues = cos(positions * divTerm)

        // Interleave sin and cos values
        // Note: MLX doesn't have direct slice assignment, so we build it differently
        var peArray = [[Float]](repeating: [Float](repeating: 0, count: dModel), count: 2 * maxLen - 1)

        let sinData = sinValues.asArray(Float.self)
        let cosData = cosValues.asArray(Float.self)

        for row in 0..<(2 * maxLen - 1) {
            for col in 0..<(dModel / 2) {
                peArray[row][2 * col] = sinData[row * (dModel / 2) + col]
                peArray[row][2 * col + 1] = cosData[row * (dModel / 2) + col]
            }
        }

        let flatPE = peArray.flatMap { $0 }
        pe = MLXArray(flatPE).reshaped([1, 2 * maxLen - 1, dModel]).asType(.float32)
        eval(pe)
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let inputLen = x.dim(1) + offset

        // Extend PE if needed
        if inputLen > maxLen {
            maxLen = inputLen + 1
            calculatePE()
        }

        // Scale input
        let scaledX = x * scale

        // Extract relevant position embeddings
        let bufferLen = pe.dim(1)
        let startIdx = bufferLen / 2 - (inputLen - 1)
        let endIdx = bufferLen / 2 + inputLen

        let posEmb = pe[0..., startIdx..<endIdx, 0...].asType(x.dtype)

        return (scaledX, posEmb)
    }
}

// MARK: - Relative Position Multi-Head Attention

/// Multi-head attention with relative position encoding
/// Used in Conformer blocks for position-aware self-attention
public class RelPositionMultiHeadAttention: Module {
    public let nHead: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    public init(
        nHead: Int,
        nFeat: Int,
        bias: Bool = true,
        posBiasU: MLXArray? = nil,
        posBiasV: MLXArray? = nil
    ) {
        self.nHead = nHead
        self.headDim = nFeat / nHead
        self.scale = pow(Float(headDim), -0.5)

        self._linearQ.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat, bias: bias)
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)

        self._posBiasU.wrappedValue = posBiasU ?? MLXArray.zeros([nHead, headDim])
        self._posBiasV.wrappedValue = posBiasV ?? MLXArray.zeros([nHead, headDim])
    }

    /// Relative shift operation for position attention
    private func relShift(_ x: MLXArray) -> MLXArray {
        let (B, H, Tq, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        // Pad with zeros on the left
        var padded = padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((1, 0))])

        // Reshape and slice
        padded = padded.reshaped([B, H, posLen + 1, Tq])
        padded = padded[0..., 0..., 1..., 0...]
        padded = padded.reshaped([B, H, Tq, posLen])

        return padded
    }

    public func callAsFunction(
        _ q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        posEmb: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil,
        localContext: (left: Int, right: Int)? = nil
    ) -> MLXArray {
        // Project Q, K, V
        var qProj = linearQ(q)
        var kProj = linearK(k)
        var vProj = linearV(v)

        // Project position embeddings
        let pProj = linearPos(posEmb)

        let (batch, qSeq, _) = (qProj.dim(0), qProj.dim(1), qProj.dim(2))
        let kSeq = kProj.dim(1)
        let posLen = pProj.dim(1)

        // Reshape Q with position biases
        qProj = qProj.reshaped([batch, qSeq, nHead, headDim])
        let qU = (qProj + posBiasU).transposed(0, 2, 1, 3)
        let qV = (qProj + posBiasV).transposed(0, 2, 1, 3)

        // Reshape K, V, P
        kProj = kProj.reshaped([batch, kSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        vProj = vProj.reshaped([batch, kSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        let p = pProj.reshaped([batch, posLen, nHead, headDim]).transposed(0, 2, 1, 3)

        // Update cache if present
        if let cache {
            (kProj, vProj) = cache.update(keys: kProj, values: vProj)
        }

        // Compute position-aware attention scores
        var matrixBD = matmul(qV, p.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., 0..<kProj.dim(-2)] * scale

        // Apply mask if provided
        var effectiveMask = matrixBD
        if let mask {
            let expandedMask = mask.expandedDimensions(axis: 0)
            effectiveMask = which(expandedMask, -Float.infinity, effectiveMask)
        }

        // Apply local attention mask if specified
        if let local = localContext {
            // Get cache offset for streaming (position of first key relative to first query)
            let cacheOffset = (cache as? ConformerCache)?.offset ?? 0
            let localMask = LocalAttentionMask.create(
                queryLen: qSeq,
                keyLen: kProj.dim(-2),
                leftContext: local.left,
                rightContext: local.right,
                offset: cacheOffset
            )
            effectiveMask = LocalAttentionMask.combine(localMask: localMask, existingMask: effectiveMask)
        }

        // Compute attention output using scaled dot product attention
        let o = MLX.scaledDotProductAttention(
            queries: qU,
            keys: kProj,
            values: vProj,
            scale: scale,
            mask: effectiveMask
        )

        // Reshape and project output
        let output = o.transposed(0, 2, 1, 3).reshaped([batch, qSeq, -1])
        return linearOut(output)
    }
}
