// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Sinusoidal positional embedding

/// Cached sinusoidal positional embedding identical to the
/// `mlx_audio.stt.models.qwen3_asr.qwen3_asr.SinusoidalPositionEmbedding`
/// reference.
///
/// The lookup table is materialized in float32 at construction time
/// so that adds against a bf16 hidden state get implicit MLX
/// up-promotion, which matches the reference numerics exactly.
///
/// Implemented as a plain `final class` (not an `MLXNN.Module`)
/// because the table is a derived constant — not a learnable
/// parameter. Exposing it as a child module would force the
/// safetensors loader to invent a `positional_embedding.table` key
/// that does not exist in the upstream checkpoint.
final class Qwen3SinusoidalPositionEmbedding {
    let table: MLXArray  // (length, channels) float32

    init(length: Int, channels: Int, maxTimescale: Float = 10_000.0) {
        precondition(
            channels % 2 == 0,
            "Qwen3SinusoidalPositionEmbedding requires even channels, "
                + "got \(channels)"
        )
        precondition(
            length > 0,
            "Qwen3SinusoidalPositionEmbedding requires positive length, "
                + "got \(length)"
        )
        let halfChannels = channels / 2
        let logTimescaleIncrement =
            log(maxTimescale) / Float(halfChannels - 1)
        let invTimescales = MLX.exp(
            -logTimescaleIncrement
                * MLXArray(0..<halfChannels).asType(.float32)
        )
        let positions = MLXArray(0..<length).asType(.float32)
            .expandedDimensions(axis: 1)
        let scaledTime = positions * invTimescales.expandedDimensions(axis: 0)
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

/// Pre-norm multi-head self-attention with bias on q/k/v/out.
///
/// Bit-exact match to `mlx_audio` requires the query to be pre-scaled
/// by `head_dim ** -0.5` BEFORE entering
/// `MLXFast.scaledDotProductAttention(scale: 1.0)`. Passing the
/// scale into the fused kernel changes the order of multiplications
/// and produces a tiny bf16 drift relative to the reference path.
/// Phase 1 parity (9.6e-7 max-abs) confirmed this ordering; the
/// per-layer test bar (≤1e-4) further enforces it.
final class Qwen3AudioAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scaling: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Qwen3ASRConfig) {
        self.embedDim = config.dModel
        self.numHeads = config.encoderAttentionHeads
        self.headDim = embedDim / numHeads
        self.scaling = pow(Float(headDim), -0.5)

        // Config validation guarantees divisibility, but we still
        // assert here so a misuse surfaces immediately.
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
            scale: 1.0,
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

/// Pre-norm transformer block:
///   `x + selfAttn(LN(x))`, then `x + fc2(GELU(fc1(LN(x))))`.
///
/// Uses the exact erf-based GELU. Do not substitute the tanh
/// approximation — the reference uses `nn.gelu` (erf form) and
/// approximate-GELU silently widens the parity envelope to roughly
/// 5e-4 in late layers.
final class Qwen3AudioEncoderLayer: Module {
    let embedDim: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3AudioAttention
    @ModuleInfo(key: "self_attn_layer_norm")
    var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: Qwen3ASRConfig) {
        self.embedDim = config.dModel
        self._selfAttn.wrappedValue = Qwen3AudioAttention(config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(
            dimensions: embedDim
        )
        self._fc1.wrappedValue = Linear(embedDim, config.encoderFFNDim)
        self._fc2.wrappedValue = Linear(config.encoderFFNDim, embedDim)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: embedDim)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var h = x
        var residual = h

        h = selfAttnLayerNorm(h)
        h = selfAttn(h, mask: mask)
        h = residual + h

        residual = h
        h = finalLayerNorm(h)
        h = MLXNN.gelu(fc1(h))
        h = fc2(h)
        h = residual + h
        return h
    }
}

// MARK: - Audio encoder

/// Captures every per-layer intermediate the production parity
/// suite checks against the Python reference. Used only by tests
/// and diagnostics — `Qwen3AudioEncoder.callAsFunction` returns the
/// final hidden states for the engine path.
///
/// Not `Sendable` because `MLXArray` itself is not Sendable; a
/// trace is meant to live within a single thread (test fixture or
/// diagnostic dump) and never crosses an actor boundary in
/// production code.
public struct Qwen3EncoderTrace {
    public let afterConv2d1: MLXArray
    public let afterConv2d2: MLXArray
    public let afterConv2d3: MLXArray
    public let afterConvOut: MLXArray
    public let afterPositionalEmbedding: MLXArray
    public let afterConcatValid: MLXArray
    public let afterLayer: [MLXArray]
    public let afterLnPost: MLXArray
    public let afterProj1: MLXArray
    public let final: MLXArray
}

/// Qwen3-ASR audio tower in pure Swift / MLX.
///
/// Architecture:
///   - 3-layer Conv2d frontend with stride 2, padding 1
///     (downsample 8× in time and 8× in frequency)
///   - flatten + Linear projection to `dModel`
///   - cached sinusoidal positional embedding (additive)
///   - chunked block-attention transformer stack
///     (`encoderLayers` × pre-norm blocks)
///   - LayerNorm + 2-layer MLP projection to `outputDim`
///
/// Phase 1 spike achieved 9.6e-7 max-abs delta against the Python
/// reference end-to-end. This production encoder targets ≤1e-4 at
/// every per-layer cut point.
///
/// Forward arguments:
///   - `inputFeatures`: `(batch, numMelBins, numFrames)` float32
///     log-mel features (Whisper-style 128-bin).
///   - `featureAttentionMask`: optional `(batch, numFrames)` int32
///     mask (1 = real frame, 0 = padding). Defaults to all-real.
public final class Qwen3AudioEncoder: Module {
    public let config: Qwen3ASRConfig

    @ModuleInfo(key: "conv2d1") var conv2d1: Conv2d
    @ModuleInfo(key: "conv2d2") var conv2d2: Conv2d
    @ModuleInfo(key: "conv2d3") var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen3AudioEncoderLayer]

    /// Pre-built positional embedding (see
    /// `Qwen3SinusoidalPositionEmbedding`). Stored as a non-`Module`
    /// reference so MLXNN's parameter walk does not search the
    /// safetensors checkpoint for a non-existent
    /// `positional_embedding.table` key.
    let positionalEmbedding: Qwen3SinusoidalPositionEmbedding

    /// Throwing convenience initializer that runs config validation
    /// before building the graph. Use this in production paths;
    /// the non-throwing `init` is retained for tests that supply
    /// trusted in-memory configs.
    public convenience init(checked config: Qwen3ASRConfig) throws {
        try config.validate()
        self.init(config)
    }

    public init(_ config: Qwen3ASRConfig) {
        self.config = config

        self._conv2d1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3, stride: 2, padding: 1, bias: true
        )
        self._conv2d2.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3, stride: 2, padding: 1, bias: true
        )
        self._conv2d3.wrappedValue = Conv2d(
            inputChannels: config.downsampleHiddenSize,
            outputChannels: config.downsampleHiddenSize,
            kernelSize: 3, stride: 2, padding: 1, bias: true
        )
        let flattenedConvOutDim =
            config.downsampleHiddenSize * config.freqAfterConv
        self._convOut.wrappedValue = Linear(
            flattenedConvOutDim, config.dModel, bias: false
        )
        self.positionalEmbedding = Qwen3SinusoidalPositionEmbedding(
            length: config.maxSourcePositions, channels: config.dModel
        )
        self._layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            Qwen3AudioEncoderLayer(config)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._proj1.wrappedValue = Linear(config.dModel, config.dModel)
        self._proj2.wrappedValue = Linear(config.dModel, config.outputDim)
        super.init()
    }

    // MARK: - Frame-length math

    /// Mirrors `_get_feat_extract_output_lengths` from the Python
    /// reference. Pure Swift `Int` arithmetic with Python-style
    /// floor division so negative intermediates (which arise when
    /// `L % 100 == 0`) round identically.
    public static func featExtractOutputLength(_ L: Int) -> Int {
        let leave = L % 100
        let feat = floorDiv(leave - 1, 2) + 1
        let half = floorDiv(feat - 1, 2) + 1
        let cnnPart = floorDiv(half - 1, 2) + 1
        let perWindow13 = (L / 100) * 13
        return cnnPart + perWindow13
    }

    /// Floor-division with Python `//` semantics (rounds toward
    /// negative infinity). Required for parity when intermediates
    /// go negative; Swift `/` truncates toward zero.
    @inline(__always)
    static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        let r = a % b
        if (r != 0) && ((r < 0) != (b < 0)) {
            return q - 1
        }
        return q
    }

    // MARK: - Block attention mask

    /// Build the cu_seqlens-style block-attention mask used by the
    /// chunked encoder. Returns a `(seqLen, seqLen)` additive mask
    /// where in-chunk positions get 0.0 and cross-chunk positions get
    /// `-1e9` (NOT `-Float.infinity` — bf16 softmax NaNs with `inf`).
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

    // MARK: - Forward (engine path)

    /// Standard forward pass returning the final encoder hidden
    /// states. This is the only call the Phase 4 decoder bridge
    /// needs. Throws `Qwen3ASRError.invalidInput` on shape problems
    /// rather than crashing the engine.
    public func forward(
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) throws -> MLXArray {
        return try forwardImpl(
            inputFeatures: inputFeatures,
            featureAttentionMask: featureAttentionMask,
            traceCollector: nil
        )
    }

    /// Compatibility wrapper preserving the spike's call signature.
    /// New callers should prefer `forward(inputFeatures:...)` (which
    /// exposes the typed error path) — this entry point preserves
    /// the original `MLXArray`-returning shape so engine glue can
    /// adopt it without a typed-error rewrite. Crashes on the same
    /// preconditions the spike enforced; production paths must
    /// already have validated input.
    public func callAsFunction(
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        do {
            return try forward(
                inputFeatures: inputFeatures,
                featureAttentionMask: featureAttentionMask
            )
        } catch {
            // Surface the underlying typed error in the crash so
            // diagnostics (test failure traces, crash reports) carry
            // it, but never silently substitute a value.
            preconditionFailure(
                "Qwen3AudioEncoder forward failed: \(error)"
            )
        }
    }

    // MARK: - Trace path (tests / diagnostics)

    /// Same forward pass as `forward(...)` but additionally captures
    /// the intermediate tensors the per-layer parity suite checks.
    /// Not for the production hot path — the trace forces extra
    /// `eval` to keep tensor identity stable during MLX scheduling.
    public func traceForward(
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) throws -> Qwen3EncoderTrace {
        var collector = TraceCollector(reserve: config.encoderLayers)
        _ = try forwardImpl(
            inputFeatures: inputFeatures,
            featureAttentionMask: featureAttentionMask,
            traceCollector: &collector
        )
        return collector.finalize()
    }

    // MARK: - Implementation

    /// Internal collector captures intermediate snapshots without
    /// branching the production forward path. The forward routine
    /// guards every store with `if let trace = traceCollector`, so
    /// when the pointer is `nil` the diagnostic stores compile away
    /// to a single null check per cut and add zero allocation.
    fileprivate struct TraceCollector {
        var afterConv2d1: MLXArray?
        var afterConv2d2: MLXArray?
        var afterConv2d3: MLXArray?
        var afterConvOut: MLXArray?
        var afterPos: MLXArray?
        var afterConcat: MLXArray?
        var afterLayer: [MLXArray]
        var afterLnPost: MLXArray?
        var afterProj1: MLXArray?
        var final: MLXArray?

        init(reserve: Int) {
            self.afterLayer = []
            self.afterLayer.reserveCapacity(reserve)
        }

        func finalize() -> Qwen3EncoderTrace {
            // All slots must be filled by the time we get here; the
            // forward path writes them in lock-step.
            guard
                let c1 = afterConv2d1, let c2 = afterConv2d2,
                let c3 = afterConv2d3, let co = afterConvOut,
                let pe = afterPos, let cv = afterConcat,
                let ln = afterLnPost, let p1 = afterProj1,
                let fn = final
            else {
                preconditionFailure(
                    "Qwen3EncoderTrace finalize called with missing slots"
                )
            }
            return Qwen3EncoderTrace(
                afterConv2d1: c1,
                afterConv2d2: c2,
                afterConv2d3: c3,
                afterConvOut: co,
                afterPositionalEmbedding: pe,
                afterConcatValid: cv,
                afterLayer: afterLayer,
                afterLnPost: ln,
                afterProj1: p1,
                final: fn
            )
        }
    }

    private func forwardImpl(
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray?,
        traceCollector: UnsafeMutablePointer<TraceCollector>?
    ) throws -> MLXArray {
        // Shape validation
        guard inputFeatures.ndim == 3 else {
            throw Qwen3ASRError.invalidInput(
                "inputFeatures must be (batch, numMelBins, numFrames); "
                    + "got rank-\(inputFeatures.ndim) shape "
                    + "\(inputFeatures.shape)"
            )
        }
        let bsz = inputFeatures.dim(0)
        let melBins = inputFeatures.dim(1)
        let totalFrames = inputFeatures.dim(2)
        guard bsz > 0 else {
            throw Qwen3ASRError.invalidInput(
                "inputFeatures batch dimension must be > 0"
            )
        }
        guard totalFrames > 0 else {
            throw Qwen3ASRError.invalidInput(
                "inputFeatures frame dimension must be > 0"
            )
        }
        guard melBins == config.numMelBins else {
            throw Qwen3ASRError.invalidInput(
                "inputFeatures mel-bin dim \(melBins) != config.numMelBins "
                    + "\(config.numMelBins)"
            )
        }

        // Per-utterance feature lengths (real, before padding).
        let featureLensArray: MLXArray
        if let mask = featureAttentionMask {
            guard mask.ndim == 2,
                mask.dim(0) == bsz,
                mask.dim(1) == totalFrames
            else {
                throw Qwen3ASRError.invalidInput(
                    "featureAttentionMask must be (batch=\(bsz), "
                        + "numFrames=\(totalFrames)); got "
                        + "\(mask.shape)"
                )
            }
            featureLensArray = mask.sum(axis: -1).asType(.int32)
        } else {
            featureLensArray = MLXArray(
                Array(repeating: Int32(totalFrames), count: bsz)
            )
        }
        // Pull to host so we can drive the chunking loop with plain
        // Ints; encoder is offline so this matches the reference.
        let featureLens: [Int] = featureLensArray
            .asArray(Int32.self)
            .map { Int($0) }
        guard featureLens.allSatisfy({ $0 > 0 }) else {
            throw Qwen3ASRError.invalidInput(
                "Qwen3AudioEncoder cannot process zero-length utterances; "
                    + "got featureLens \(featureLens)"
            )
        }

        let aftercnnLens: [Int] = featureLens.map(
            Qwen3AudioEncoder.featExtractOutputLength
        )

        // Slice each utterance into chunkSize-length chunks, padding
        // the final one (per utterance) up to chunkSize.
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
        // chunkLengths is non-empty by the precondition on
        // featureLens; the `?? chunkSize` is defensive only.
        let maxChunkLen = chunkLengths.max() ?? chunkSize

        let paddedChunks: [MLXArray] = chunks.enumerated().map {
            (idx, chunk) -> MLXArray in
            let cLen = chunkLengths[idx]
            if cLen < maxChunkLen {
                return MLX.padded(
                    chunk,
                    widths: [
                        IntOrPair(0),
                        IntOrPair((0, maxChunkLen - cLen)),
                    ],
                    value: MLXArray(Float(0))
                )
            }
            return chunk
        }
        let paddedFeature = MLX.stacked(paddedChunks, axis: 0)

        let featLensAfterCNN: [Int] = chunkLengths.map(
            Qwen3AudioEncoder.featExtractOutputLength
        )
        let maxLenAfterCNNRuntime = featLensAfterCNN.max() ?? 0

        // Conv frontend. MLX Conv2d expects NHWC layout
        // (batch, H=mel, W=time, C_in=1).
        var x = paddedFeature.expandedDimensions(axis: 3)
        x = MLXNN.gelu(conv2d1(x))
        if let trace = traceCollector {
            trace.pointee.afterConv2d1 = x
        }
        x = MLXNN.gelu(conv2d2(x))
        if let trace = traceCollector {
            trace.pointee.afterConv2d2 = x
        }
        x = MLXNN.gelu(conv2d3(x))
        if let trace = traceCollector {
            trace.pointee.afterConv2d3 = x
        }

        // (B, H', W', C) -> (B, W', C * H')
        let convB = x.dim(0)
        let convF = x.dim(1)
        let convT = x.dim(2)
        let convC = x.dim(3)
        x = x.transposed(0, 2, 3, 1).reshaped(convB, convT, convC * convF)
        x = convOut(x)
        if let trace = traceCollector {
            trace.pointee.afterConvOut = x
        }

        // Add positional embedding (broadcast over batch).
        let pos = positionalEmbedding(x.dim(1))
        x = x + pos.expandedDimensions(axis: 0)
        if let trace = traceCollector {
            trace.pointee.afterPos = x
        }

        // Concatenate the valid-region of each chunk along time;
        // prepend a batch dim of 1 below.
        var hiddenList: [MLXArray] = []
        for i in 0..<x.dim(0) {
            let validLen = featLensAfterCNN[i]
            hiddenList.append(x[i][0..<validLen])
        }
        var hidden = MLX.concatenated(hiddenList, axis: 0)
        if let trace = traceCollector {
            trace.pointee.afterConcat = hidden
        }

        // Build cu_seqlens for the block mask using the per-utterance
        // aftercnn lengths and `n_window_infer`.
        let windowAfterCNN =
            maxLenAfterCNNRuntime
            * (config.nWindowInfer / (config.nWindow * 2))
        var cuChunkLens: [Int] = [0]
        for cnnLen in aftercnnLens {
            // windowAfterCNN can theoretically be 0 if the mel input
            // was so short that maxLenAfterCNNRuntime collapsed to 0;
            // the input-length precondition above prevents that, but
            // we guard anyway.
            guard windowAfterCNN > 0 else { break }
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
        var attentionMask = Qwen3AudioEncoder.blockAttentionMask(
            seqLen: seqLen,
            cuSeqLens: cuSeqLens,
            dtype: hidden.dtype
        )
        attentionMask = attentionMask.expandedDimensions(axes: [0, 1])

        hidden = hidden.expandedDimensions(axis: 0)
        for layer in layers {
            hidden = layer(hidden, mask: attentionMask)
            if let trace = traceCollector {
                trace.pointee.afterLayer.append(hidden)
            }
        }
        hidden = hidden[0]

        hidden = lnPost(hidden)
        if let trace = traceCollector {
            trace.pointee.afterLnPost = hidden
        }
        hidden = MLXNN.gelu(proj1(hidden))
        if let trace = traceCollector {
            trace.pointee.afterProj1 = hidden
        }
        hidden = proj2(hidden)
        if let trace = traceCollector {
            trace.pointee.final = hidden
        }
        return hidden
    }
}

// MARK: - Trace forwarding glue

extension Qwen3AudioEncoder {
    /// Wrapper that lets `forwardImpl` accept the trace collector
    /// by `inout`. `UnsafeMutablePointer` is only used internally
    /// and never escapes; trace tensors live in MLX's regular
    /// reference-counted graph.
    fileprivate func forwardImpl(
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray?,
        traceCollector: inout TraceCollector
    ) throws -> MLXArray {
        return try withUnsafeMutablePointer(to: &traceCollector) { ptr in
            try forwardImpl(
                inputFeatures: inputFeatures,
                featureAttentionMask: featureAttentionMask,
                traceCollector: ptr
            )
        }
    }
}
