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
        case .gemma4_26B_A4B1M, .s3Retrieval:
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
        case .qwen35B1M:
            return "infinite:research/26-flagship-1m.md"
        case .s3Retrieval:
            return "infinite:research/24-dense-retrieval.md"
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
