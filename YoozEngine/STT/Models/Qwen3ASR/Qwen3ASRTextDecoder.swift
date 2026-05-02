// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Attention

/// Qwen3 self-attention block adapted for the ASR decoder.
///
/// Layout matches the published `mlx-community/Qwen3-ASR-1.7B-8bit`
/// safetensors keys:
///   `model.layers.<i>.self_attn.{q,k,v,o}_proj.{weight,scales,biases}`
///   `model.layers.<i>.self_attn.{q,k}_norm.weight`
///
/// We don't reuse `MLXLLM.Qwen3Attention` because the full pipeline
/// needs an embedding-only forward path (audio embeddings are spliced
/// in *before* the decoder runs), and that path lives in
/// `Qwen3ASRTextDecoderInner`. Keeping the attention class private to
/// this file isolates the parity surface from the public LLM module
/// (which evolves on its own cadence).
final class Qwen3ASRTextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ config: Qwen3ASRTextConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            numHeads * headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            numKVHeads * headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            numKVHeads * headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(
            numHeads * headDim,
            config.hiddenSize,
            bias: false
        )

        self._qNorm.wrappedValue = RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps
        )
        self._kNorm.wrappedValue = RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps
        )

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: config.ropeTheta
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: MLXLMCommon.KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape and apply Q/K norm before the head transpose.
        // Python reference applies q_norm/k_norm in (B, L, H, D) layout
        // and then transposes; we follow the same sequence so RMSNorm
        // sees the same per-head slices.
        queries = qNorm(queries.reshaped(B, L, numHeads, headDim))
            .transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, numKVHeads, headDim))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        // Apply RoPE positioning. Older mlx-swift-lm versions (the
        // 2.30.x branch we pin to) don't ship `applyRotaryPosition`,
        // so call the layer directly with the cache offset.
        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MLP (SwiGLU)

final class Qwen3ASRTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: Qwen3ASRTextConfig) {
        self._gate.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false
        )
        self._up.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false
        )
        self._down.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Decoder layer

final class Qwen3ASRTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3ASRTextAttention
    let mlp: Qwen3ASRTextMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm")
    var postAttentionLayerNorm: RMSNorm

    init(_ config: Qwen3ASRTextConfig) {
        self._selfAttn.wrappedValue = Qwen3ASRTextAttention(config)
        self.mlp = Qwen3ASRTextMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: MLXLMCommon.KVCache?
    ) -> MLXArray {
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Inner text model

/// Token-only forward path mirroring the Python `TextModel`. Exposes
/// both `embedTokens` (so the pipeline can build inputs_embeds and
/// splice in audio features) and a `callAsEmbeddings(...)` entry that
/// runs the transformer stack from a precomputed embedding tensor.
public final class Qwen3ASRTextDecoderInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3ASRTextDecoderLayer]
    let norm: RMSNorm

    let config: Qwen3ASRTextConfig

    public init(_ config: Qwen3ASRTextConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0..<config.numHiddenLayers).map { _ in
            Qwen3ASRTextDecoderLayer(config)
        }
        self.norm = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    /// Forward from token IDs (no audio splicing path). Matches the
    /// signature `MLXLLM.Qwen3` exposes so this module can be passed
    /// through `MLXLMCommon.TokenIterator` for plain LLM use cases.
    public func callAsFunction(
        _ inputs: MLXArray, cache: [MLXLMCommon.KVCache]? = nil
    ) -> MLXArray {
        let h = embedTokens(inputs)
        return runStack(h: h, cache: cache)
    }

    /// Forward from precomputed embeddings. The pipeline calls this
    /// after splicing audio_tower hidden states into the `<|audio_pad|>`
    /// positions of the prompt embedding sequence.
    public func callAsEmbeddings(
        _ inputsEmbeds: MLXArray, cache: [MLXLMCommon.KVCache]? = nil
    ) -> MLXArray {
        runStack(h: inputsEmbeds, cache: cache)
    }

    private func runStack(
        h: MLXArray, cache: [MLXLMCommon.KVCache]?
    ) -> MLXArray {
        var hidden = h
        // Use the symbolic-mask form (`.causal`) when possible so MLX
        // can fuse the SDPA kernel; this matches the convention every
        // mlx-swift-lm Qwen3-family model uses.
        let mask = makeAttentionMask(
            n: hidden.dim(1),
            cache: cache?.first
        )
        for (i, layer) in layers.enumerated() {
            hidden = layer(
                hidden, mask: mask, cache: cache?[i]
            )
        }
        return norm(hidden)
    }
}

// MARK: - Top-level decoder

/// Qwen3-ASR text decoder. Exposes the language-model surface
/// (`callAsFunction(inputs:cache:) -> logits`) plus an embedding
/// forward (`callAsEmbeddings(...) -> logits`) used by the audio
/// bridge. Conforms to `LanguageModel` from `MLXLMCommon` so the
/// engine can reuse `TokenIterator` machinery if it ever wants
/// pure-text generation (Phase 5+ may piggyback this for diagnostics).
public final class Qwen3ASRTextDecoder: Module, LanguageModel,
    KVCacheDimensionProvider
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3ASRTextDecoderInner
    public let configuration: Qwen3ASRTextConfig

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Qwen3ASRTextConfig) {
        self.configuration = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0..<config.numHiddenLayers).map { _ in
            config.numKeyValueHeads
        }
        self.model = Qwen3ASRTextDecoderInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabSize, bias: false
            )
        }
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [MLXLMCommon.KVCache]?
    ) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Forward from precomputed embeddings (one decode step or full
    /// prompt prefill). Returns logits of shape
    /// `(batch, seqLen, vocab)`.
    public func callAsEmbeddings(
        _ inputsEmbeds: MLXArray, cache: [MLXLMCommon.KVCache]?
    ) -> MLXArray {
        var out = model.callAsEmbeddings(inputsEmbeds, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Strip `lm_head.weight` when word embeddings are tied; the
    /// canonical checkpoint never ships an explicit head in that
    /// configuration. Mirrors the Python reference's `sanitize`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
            weights["lm_head.scales"] = nil
            weights["lm_head.biases"] = nil
        }
        return weights
    }

    // MARK: - LanguageModel conformance

    /// Default `prepare` runs the entire prompt through the decoder
    /// and hands the iterator the last token's logits. Phase 4 uses
    /// the embeddings path directly (audio splicing happens *before*
    /// the decoder runs), so this conformance only matters for
    /// pure-text uses of the decoder — e.g. Phase 6+ diagnostics that
    /// drop it into the stock `TokenIterator`.
    public func prepare(
        _ input: LMInput,
        cache: [MLXLMCommon.KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        let logits = callAsFunction(
            input.text.tokens.expandedDimensions(axis: 0),
            cache: cache
        )
        return .logits(.init(logits: logits))
    }
}
