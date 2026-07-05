// InfiniteModelSelection.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

public enum InfiniteRAMTier: String, Codable, Sendable {
    case belowMinimum = "below_minimum"
    case reduced
    case full

    public var minimumPhysicalMemoryBytes: Int64 {
        switch self {
        case .belowMinimum:
            return 0
        case .reduced:
            return 32 * 1024 * 1024 * 1024
        case .full:
            return 64 * 1024 * 1024 * 1024
        }
    }

    public static var current: InfiniteRAMTier {
        let gib = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)
        if gib >= 64 {
            return .full
        }
        if gib >= 32 {
            return .reduced
        }
        return .belowMinimum
    }

    public func supports(required tier: InfiniteRAMTier) -> Bool {
        switch (self, tier) {
        case (.full, .full), (.full, .reduced):
            return true
        case (.reduced, .reduced):
            return true
        default:
            return false
        }
    }
}

/// Engine-owned model/mode catalogue for the Infinite long-context module.
///
/// Stable raw values are the wire ids used by `/v1/infinite/model[s]`.
/// Catalog rows are owned by the engine, while adapter implementations
/// live behind `InfiniteBackendAdapter` so Infinite lends capability
/// without becoming another serving surface.
public enum InfiniteModelSelection: String, CaseIterable, Codable, Sendable {
    /// Light/reduced-tier 1M-capable Gemma4 model proven in infinite.
    case gemma4E4B1M = "gemma4-e4b-1m"
    /// Full-tier Gemma4 MoE model proven in infinite.
    case gemma4_26B_A4B1M = "gemma4-26b-a4b-1m"
    /// Reduced-tier dense Gemma4 12B (`gemma4_unified`) for long-context coding.
    case gemma4_12B1M = "gemma4-12b-1m"
    /// Qwen/d1 paged long-context flagship path.
    case qwen35B1M = "qwen3-35b-1m"
    /// Fast retrieval-backed mode from the s3 carry track.
    case s3Retrieval = "s3-retrieval"

    public var displayName: String {
        switch self {
        case .gemma4E4B1M:
            return "Gemma4 E4B 1M"
        case .gemma4_26B_A4B1M:
            return "Gemma4 26B-A4B 1M"
        case .gemma4_12B1M:
            return "Gemma4 12B 1M"
        case .qwen35B1M:
            return "Qwen3.6 35B-A3B 1M"
        case .s3Retrieval:
            return "S3 Retrieval"
        }
    }

    public var description: String {
        switch self {
        case .gemma4E4B1M:
            return "Reduced-tier long-context model. Single-needle retrieval "
                + "validated near 1M tokens; interactive tier ~256K (multi-hop degrades beyond)."
        case .gemma4_26B_A4B1M:
            return "Full-tier Gemma4 long-context model. Single-needle retrieval "
                + "validated near 1M tokens; interactive tier ~256K (multi-hop degrades beyond)."
        case .gemma4_12B1M:
            return "Dense Gemma4 12B for long-context coding. Reduced-tier viable "
                + "(~7 GB 4-bit); native window 262K, interactive coding recall the primary use."
        case .qwen35B1M:
            return "Paged-cache Qwen3.6 flagship path. 1M context is memory-feasible "
                + "but latency-bound; interactive tier ~256K."
        case .s3Retrieval:
            return "Fast retrieval-backed mode for lexical and semantic recall. "
                + "The 10M figure is retrieval index capacity, not an attention window."
        }
    }

    public var tier: ModelTier {
        switch self {
        case .gemma4E4B1M:
            return .light
        case .gemma4_26B_A4B1M, .gemma4_12B1M, .s3Retrieval:
            return .quality
        case .qwen35B1M:
            return .premium
        }
    }

    /// Approximate model artifact size. `nil` means the mode is not a
    /// single model artifact in this phase (for example retrieval mode).
    public var sizeBytes: Int64? {
        switch self {
        case .gemma4E4B1M:
            return 3 * 1024 * 1024 * 1024
        case .gemma4_26B_A4B1M:
            return 14 * 1024 * 1024 * 1024
        case .gemma4_12B1M:
            return 7 * 1024 * 1024 * 1024
        case .qwen35B1M:
            return 20 * 1024 * 1024 * 1024
        case .s3Retrieval:
            return nil
        }
    }

    public var maxContextTokens: Int {
        descriptor.targetContextTokens
    }

    public var nativeContextTokens: Int {
        descriptor.nativeContextTokens
    }

    public var requiredRAMTier: InfiniteRAMTier {
        descriptor.requiredRAMTier
    }

    public var ramTier: String {
        requiredRAMTier.rawValue
    }

    /// Whether the model can actually be loaded + run by the engine's
    /// MLX-Swift runtime today. The catalog advertises models proven in the
    /// Python harness; the Swift `mlx-swift-lm` fork loads all four MLX rows:
    /// `qwen3_5_moe` (the Qwen row) plus the three `gemma4`-family rows —
    /// 26B-A4B (`gemma4`, #184), the E4B OptiQ-4bit build (`gemma4`, #186, once
    /// its per-layer-input projection + KV-shared layers became loadable), and
    /// the dense 12B (`gemma4_unified`, #187, registered + its K-eq-V value path
    /// fixed). All verified at native context vs the Python reference (mlx-lm for
    /// the `gemma4` rows, mlx-vlm for `gemma4_unified`). Only retrieval mode has
    /// no MLX backend wired here. A row is selectable in the picker for
    /// discovery, but load/generate refuses cleanly when this is false.
    public var swiftRuntimeSupported: Bool {
        switch self {
        case .qwen35B1M, .gemma4_26B_A4B1M, .gemma4E4B1M, .gemma4_12B1M:
            return true
        case .s3Retrieval:
            return false
        }
    }

    public var backendKind: String {
        descriptor.backendKind.rawValue
    }

    public var adapterKind: String {
        descriptor.adapterKind.rawValue
    }

    public var huggingFaceID: String? {
        descriptor.repository?.id
    }

    public var revision: String? {
        descriptor.repository?.revision
    }

    public var evidenceRef: String {
        switch self {
        case .gemma4E4B1M, .gemma4_26B_A4B1M:
            return "infinite:research/18-gemma-support-matrix.md"
        case .gemma4_12B1M:
            return "engine:YoozEngine/Infinite/results/PARITY.md (#187)"
        case .qwen35B1M:
            return "infinite:research/26-flagship-1m.md"
        case .s3Retrieval:
            return "infinite:research/24-dense-retrieval.md"
        }
    }

    /// Turn-commit's chat-template composer for this model family (engine#267):
    /// Qwen's ChatML `<think>` framing vs Gemma4's `<|turn>`/`<turn|>` framing
    /// with no per-turn think toggle. `.s3Retrieval` never actually reaches a
    /// loaded `MLXInfiniteBackend` (`swiftRuntimeSupported == false`, and its
    /// `descriptor.repository == nil` makes `MLXInfiniteBackend.load` throw
    /// first), so that arm is unreachable in practice — Qwen's composer is
    /// returned there only to keep this switch exhaustive.
    public var turnComposer: any InfiniteTurnComposer {
        switch self {
        case .qwen35B1M:
            return Qwen35ChatMLComposer()
        case .gemma4E4B1M, .gemma4_26B_A4B1M, .gemma4_12B1M:
            return Gemma4Composer()
        case .s3Retrieval:
            return Qwen35ChatMLComposer()
        }
    }

    /// Whether a session on this model may opt into a quantized KV cache
    /// (engine#268). Qwen-only in v1: Qwen's attention path
    /// (`attentionWithCacheUpdate`, mlx-swift-lm's `AttentionUtils.swift`)
    /// detects `QuantizedKVCacheProtocol` and dispatches to
    /// `updateQuantized`, but Gemma4Text's attention path
    /// (`Gemma4Text.swift`) calls `cache.update()` directly — which
    /// `fatalError`s on a `QuantizedKVCache` (`update` is only a stub there;
    /// `updateQuantized` is the real entry point). Same family grouping as
    /// `turnComposer` above, kept as its own switch since it gates a
    /// different concern (cache class safety, not chat-template framing).
    /// `.s3Retrieval` never reaches a loaded backend (`swiftRuntimeSupported
    /// == false`), so `false` here is defensive, not load-bearing.
    public var supportsQuantizedKVCache: Bool {
        switch self {
        case .qwen35B1M:
            return true
        case .gemma4E4B1M, .gemma4_26B_A4B1M, .gemma4_12B1M, .s3Retrieval:
            return false
        }
    }

    public var descriptor: InfiniteBackendDescriptor {
        switch self {
        case .gemma4E4B1M:
            return InfiniteBackendDescriptor(
                selection: self,
                repository: InfiniteModelRepository(
                    id: "mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit",
                    revision: "b4966f32e71f9f4976a78f74bc8944b1d064bcbf"
                ),
                backendKind: .pagedKV,
                adapterKind: .pagedKVMLX,
                nativeContextTokens: 131_072,
                targetContextTokens: 1_000_000,
                requiredRAMTier: .reduced
            )
        case .gemma4_26B_A4B1M:
            return InfiniteBackendDescriptor(
                selection: self,
                repository: InfiniteModelRepository(
                    id: "mlx-community/gemma-4-26b-a4b-it-4bit",
                    revision: "efbeee6e582ebfd06abc9d65e90839c4b5d2116b"
                ),
                backendKind: .pagedKV,
                adapterKind: .pagedKVMLX,
                nativeContextTokens: 262_144,
                targetContextTokens: 1_000_000,
                requiredRAMTier: .full
            )
        case .gemma4_12B1M:
            return InfiniteBackendDescriptor(
                selection: self,
                repository: InfiniteModelRepository(
                    id: "mlx-community/gemma-4-12B-it-4bit",
                    revision: "73bcf09092aa277861d5a191b989b666f7f32e8f"
                ),
                backendKind: .pagedKV,
                adapterKind: .pagedKVMLX,
                nativeContextTokens: 262_144,
                targetContextTokens: 1_000_000,
                requiredRAMTier: .reduced
            )
        case .qwen35B1M:
            return InfiniteBackendDescriptor(
                selection: self,
                repository: InfiniteModelRepository(
                    id: "mlx-community/Qwen3.6-35B-A3B-4bit",
                    revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46"
                ),
                backendKind: .pagedKV,
                adapterKind: .pagedKVMLX,
                nativeContextTokens: 262_144,
                targetContextTokens: 1_000_000,
                requiredRAMTier: .full
            )
        case .s3Retrieval:
            return InfiniteBackendDescriptor(
                selection: self,
                repository: nil,
                backendKind: .retrieval,
                adapterKind: .retrievalIndex,
                nativeContextTokens: 0,
                targetContextTokens: 10_000_000,
                requiredRAMTier: .full,
                requiredCachedFiles: []
            )
        }
    }
}
