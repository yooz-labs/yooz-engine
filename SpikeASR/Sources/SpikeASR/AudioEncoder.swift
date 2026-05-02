import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Sinusoidal positional embedding

/// Sinusoidal positional embedding identical to the Python reference
/// (`mlx_audio.stt.models.qwen3_asr.qwen3_asr.SinusoidalPositionEmbedding`).
///
/// Caches the table at construction time in float32 so adds against a
/// bf16 hidden state get implicit MLX up-promotion, matching the
/// reference numerics exactly. Implemented as a plain `final class`
/// rather than a `Module` subclass, because the table is a derived
/// constant — not a learnable parameter — and exposing it via
/// `parameters()` would force the safetensors loader to invent a
/// non-existent `positional_embedding.table` key.
final class SinusoidalPositionEmbedding {
    let table: MLXArray  // shape (length, channels) in float32

    init(length: Int, channels: Int, maxTimescale: Float = 10_000.0) {
        precondition(
            channels % 2 == 0,
            "SinusoidalPositionEmbedding requires even channels"
        )
        let halfChannels = channels / 2
        let logTimescaleIncrement = log(maxTimescale) / Float(halfChannels - 1)
        let invTimescales = MLX.exp(
            -logTimescaleIncrement
                * MLXArray(0..<halfChannels).asType(.float32)
        )  // (halfChannels,)
        let positions = MLXArray(0..<length).asType(.float32)
            .expandedDimensions(axis: 1)  // (length, 1)
        let scaledTime = positions * invTimescales.expandedDimensions(axis: 0)
        // (length, channels)
        self.table = MLX.concatenated(
            [MLX.sin(scaledTime), MLX.cos(scaledTime)],
            axis: 1
        )
    }

    func callAsFunction(_ seqLen: Int) -> MLXArray {
        table[0..<seqLen]
    }
}

// MARK: - Audio attention

/// Pre-norm multi-head self-attention, key/query/value linear with bias.
///
/// Matches `mlx_audio` exactly:
///  - q is pre-scaled by `head_dim ** -0.5` BEFORE being passed into
///    `MLXFast.scaledDotProductAttention(scale: 1.0)`, not after.
///    Passing the scale into the fused kernel changes the order of
///    multiplications and produces a tiny bf16 drift relative to the
///    Python path; the parity check enforces this.
final class AudioAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scaling: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: AudioEncoderConfig) {
        self.embedDim = config.dModel
        self.numHeads = config.encoderAttentionHeads
        self.headDim = embedDim / numHeads
        self.scaling = pow(Float(headDim), -0.5)

        precondition(
            headDim * numHeads == embedDim,
            "embedDim must be divisible by numHeads"
        )

        self._qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._kProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let bsz = x.dim(0)
        let seqLen = x.dim(1)

        let q = (qProj(x) * scaling)
            .reshaped(bsz, seqLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let k = kProj(x)
            .reshaped(bsz, seqLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let v = vProj(x)
            .reshaped(bsz, seqLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0,  // see doc-comment above
            mask: mask
        )
        let out =
            attended
            .transposed(0, 2, 1, 3)
            .reshaped(bsz, seqLen, embedDim)
        return outProj(out)
    }
}

// MARK: - Encoder layer

/// Pre-norm transformer block: LN -> self-attn -> residual,
/// then LN -> fc1 -> GELU -> fc2 -> residual.
final class AudioEncoderLayer: Module {
    let embedDim: Int

    @ModuleInfo(key: "self_attn") var selfAttn: AudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: AudioEncoderConfig) {
        self.embedDim = config.dModel
        self._selfAttn.wrappedValue = AudioAttention(config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
        self._fc1.wrappedValue = Linear(embedDim, config.encoderFFNDim)
        self._fc2.wrappedValue = Linear(config.encoderFFNDim, embedDim)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        var residual = h

        h = selfAttnLayerNorm(h)
        h = selfAttn(h, mask: mask)
        h = residual + h

        residual = h
        h = finalLayerNorm(h)
        h = MLXNN.gelu(fc1(h))  // exact erf-based GELU; do not approximate
        h = fc2(h)
        h = residual + h
        return h
    }
}

// MARK: - Audio encoder

/// Qwen3-ASR audio tower: 3-layer Conv2d frontend (downsample 8x in
/// time, 8x in frequency) -> linear projection -> sinusoidal pos emb
/// -> 24 transformer blocks with chunked block-attention -> LayerNorm
/// -> proj1 + GELU -> proj2 (2048-d audio token stream).
public final class AudioEncoder: Module {
    public let config: AudioEncoderConfig
    let embedScale: Float
    public let maxLenAfterCNN: Int

    @ModuleInfo(key: "conv2d1") var conv2d1: Conv2d
    @ModuleInfo(key: "conv2d2") var conv2d2: Conv2d
    @ModuleInfo(key: "conv2d3") var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear
    @ModuleInfo(key: "layers") var layers: [AudioEncoderLayer]

    /// Positional embedding is a cached buffer, not a trainable
    /// parameter; kept on the module so it travels with the encoder
    /// but excluded from `parameters()` via the underscore prefix.
    let positionalEmbedding: SinusoidalPositionEmbedding

    public init(_ config: AudioEncoderConfig) {
        self.config = config
        self.embedScale =
            config.scaleEmbedding ? sqrt(Float(config.dModel)) : 1.0

        // Per-chunk feat length after the 3 stride-2 convs.
        // Matches `_get_feat_extract_output_lengths` for the
        // canonical chunk size of `n_window * 2 = 100`. The encoder
        // pads each chunk to `chunk_size = 100` mel frames in the
        // batched offline path.
        let chunkSize = config.nWindow * 2  // 100
        self.maxLenAfterCNN = AudioEncoder.featExtractOutputLength(chunkSize)

        self._conv2d1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            bias: true
        )
        self._conv2d2.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            bias: true
        )
        self._conv2d3.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            bias: true
        )
        let flattenedConvOutDim =
            config.downsampleHiddenSize * config.freqAfterConv
        self._convOut.wrappedValue = Linear(
            flattenedConvOutDim, config.dModel, bias: false
        )
        self.positionalEmbedding = SinusoidalPositionEmbedding(
            length: config.maxSourcePositions, channels: config.dModel
        )
        self._layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            AudioEncoderLayer(config)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._proj1.wrappedValue = Linear(config.dModel, config.dModel)
        self._proj2.wrappedValue = Linear(config.dModel, config.outputDim)
        super.init()
    }

    /// Mirrors `_get_feat_extract_output_lengths` in the Python
    /// reference. Pure Swift Int arithmetic, but using Python-style
    /// floor division (rounds toward -inf, not toward 0) to stay
    /// numerically identical for negative intermediates that arise
    /// when `L % 100 == 0`.
    public static func featExtractOutputLength(_ L: Int) -> Int {
        let leave = L % 100
        let feat = floorDiv(leave - 1, 2) + 1
        let half = floorDiv(feat - 1, 2) + 1
        let cnnPart = floorDiv(half - 1, 2) + 1
        let perWindow13 = (L / 100) * 13
        return cnnPart + perWindow13
    }

    /// Floor-division matching Python's `//` semantics (rounds toward
    /// negative infinity), unlike Swift's `/` which truncates toward
    /// zero. Required for numerical parity with the Python reference
    /// when intermediates go negative (e.g. `L % 100 == 0`).
    @inline(__always)
    static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        let r = a % b
        if (r != 0) && ((r < 0) != (b < 0)) {
            return q - 1
        }
        return q
    }

    /// Build the cu_seqlens-style block-attention mask used by the
    /// chunked encoder. Returns a `(seqLen, seqLen)` additive mask
    /// where in-chunk positions get 0.0 and cross-chunk positions get
    /// `-1e9` (NOT `-Float.infinity` — bf16 softmax will NaN with inf).
    static func blockAttentionMask(
        seqLen: Int, cuSeqLens: [Int], dtype: DType
    ) -> MLXArray {
        let mask = MLXArray.full(
            [seqLen, seqLen], values: MLXArray(Float(-1e9))
        ).asType(dtype)
        for i in 0..<(cuSeqLens.count - 1) {
            let start = cuSeqLens[i]
            let end = cuSeqLens[i + 1]
            if end > start {
                let zeros = MLXArray.zeros(
                    [end - start, end - start], dtype: dtype
                )
                mask[start..<end, start..<end] = zeros
            }
        }
        return mask
    }

    /// Offline forward pass.
    ///
    /// - Parameter inputFeatures: `(batch, numMelBins, numFrames)`
    ///   float32 mel features (Whisper-style 128-bin log-mel).
    /// - Parameter featureAttentionMask: optional `(batch, numFrames)`
    ///   int32 mask (1 = real frame, 0 = padding). When `nil` all
    ///   frames are treated as valid.
    public func callAsFunction(
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        let bsz = inputFeatures.dim(0)
        let totalFrames = inputFeatures.dim(2)

        // 1) Per-utterance feature lengths (real, before padding).
        let featureLensArray: MLXArray
        if let mask = featureAttentionMask {
            featureLensArray = mask.sum(axis: -1).asType(.int32)
        } else {
            featureLensArray = MLXArray(
                Array(repeating: Int32(totalFrames), count: bsz)
            )
        }
        // Pull to host so we can drive the (batch-variable) chunking
        // loop with plain Swift Ints. Encoder is offline; this
        // sync is the same one the Python path performs.
        let featureLens: [Int] = featureLensArray
            .asArray(Int32.self)
            .map { Int($0) }

        // 2) Compute aftercnn lengths (per-utterance, full-length).
        let aftercnnLens: [Int] = featureLens.map(
            AudioEncoder.featExtractOutputLength
        )

        // 3) Slice each utterance into chunkSize-length chunks; pad
        //    last chunk per utterance up to chunkSize (== nWindow * 2).
        let chunkSize = config.nWindow * 2
        var chunks: [MLXArray] = []
        var chunkLengths: [Int] = []
        for i in 0..<bsz {
            let featLen = featureLens[i]
            let numChunks = (featLen + chunkSize - 1) / chunkSize
            var pos = 0
            for j in 0..<numChunks {
                let cLen: Int
                if j == numChunks - 1 {
                    let remainder = featLen % chunkSize
                    cLen = (remainder == 0) ? chunkSize : remainder
                } else {
                    cLen = chunkSize
                }
                let slice = inputFeatures[i][0..., pos..<(pos + cLen)]
                chunks.append(slice)
                chunkLengths.append(cLen)
                pos += cLen
            }
        }
        let maxChunkLen = chunkLengths.max() ?? chunkSize

        // 4) Pad chunks to maxChunkLen on the time axis.
        let paddedChunks: [MLXArray] = chunks.enumerated().map {
            (idx, chunk) -> MLXArray in
            let cLen = chunkLengths[idx]
            if cLen < maxChunkLen {
                return MLX.padded(
                    chunk,
                    widths: [
                        IntOrPair(0),  // mel-bin axis
                        IntOrPair((0, maxChunkLen - cLen)),  // time axis
                    ],
                    value: MLXArray(Float(0))
                )
            }
            return chunk
        }
        // (chunkCount, melBins, maxChunkLen)
        let paddedFeature = MLX.stacked(paddedChunks, axis: 0)

        // 5) Per-chunk lengths after CNN, used for valid-region pruning.
        let featLensAfterCNN: [Int] = chunkLengths.map(
            AudioEncoder.featExtractOutputLength
        )
        let maxLenAfterCNNRuntime = featLensAfterCNN.max() ?? 0

        // 6) Conv frontend. MLX Conv2d wants
        //    NHWC layout `(batch, H, W, C_in)`, with H = melBins,
        //    W = time, C_in = 1. After 3 stride-2 convs:
        //    H' = freqAfterConv, W' = ceil(maxChunkLen / 8).
        var x = paddedFeature.expandedDimensions(axis: 3)  // (B, H, W, 1)
        x = MLXNN.gelu(conv2d1(x))
        x = MLXNN.gelu(conv2d2(x))
        x = MLXNN.gelu(conv2d3(x))

        // 7) (B, H', W', C) -> (B, W', C * H')
        let b = x.dim(0)
        let f = x.dim(1)  // freq after conv
        let t = x.dim(2)  // time after conv
        let c = x.dim(3)  // channels after conv
        x = x.transposed(0, 2, 3, 1).reshaped(b, t, c * f)
        x = convOut(x)  // (B, W', dModel)

        // 8) Add positional embedding (broadcast over batch).
        let pos = positionalEmbedding(x.dim(1))
        x = x + pos.expandedDimensions(axis: 0)

        // 9) Concatenate the valid-region of each chunk along the
        //    time axis, then prepend a batch dim of 1.
        var hiddenList: [MLXArray] = []
        for i in 0..<x.dim(0) {
            let validLen = featLensAfterCNN[i]
            hiddenList.append(x[i][0..<validLen])
        }
        var hidden = MLX.concatenated(hiddenList, axis: 0)
        // (totalValid, dModel)

        // 10) Build cu_seqlens for the block mask using the per-utterance
        //     aftercnn lengths and `n_window_infer`.
        let windowAfterCNN =
            maxLenAfterCNNRuntime
            * (config.nWindowInfer / (config.nWindow * 2))
        var cuChunkLens: [Int] = [0]
        for cnnLen in aftercnnLens {
            let numFullWindows = cnnLen / windowAfterCNN
            for _ in 0..<numFullWindows {
                cuChunkLens.append(windowAfterCNN)
            }
            let remainder = cnnLen % windowAfterCNN
            if remainder != 0 {
                cuChunkLens.append(remainder)
            }
        }
        var cuSeqLens: [Int] = []
        var running = 0
        for v in cuChunkLens {
            running += v
            cuSeqLens.append(running)
        }

        let seqLen = hidden.dim(0)
        var attentionMask = AudioEncoder.blockAttentionMask(
            seqLen: seqLen,
            cuSeqLens: cuSeqLens,
            dtype: hidden.dtype
        )
        // SDPA expects (batch, heads, q, k); broadcast over heads.
        attentionMask = attentionMask.expandedDimensions(axes: [0, 1])

        // 11) Add a leading batch dim of 1, run encoder layers, drop
        //     the batch dim, ln_post, two projections (with GELU).
        hidden = hidden.expandedDimensions(axis: 0)
        for layer in layers {
            hidden = layer(hidden, mask: attentionMask)
        }
        hidden = hidden[0]

        hidden = lnPost(hidden)
        hidden = MLXNN.gelu(proj1(hidden))
        hidden = proj2(hidden)
        return hidden
    }
}
