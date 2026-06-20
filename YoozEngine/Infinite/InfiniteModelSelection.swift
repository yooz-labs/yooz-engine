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
/// Phase 1 exposes the contract and evidence-backed rows only; real
/// backend launch/load behavior lands in later phases.
public enum InfiniteModelSelection: String, CaseIterable, Codable, Sendable {
    /// Light/reduced-tier 1M-capable Gemma4 model proven in infinite.
    case gemma4E4B1M = "gemma4-e4b-1m"
    /// Full-tier Gemma4 MoE model proven in infinite.
    case gemma4_26BA4B1M = "gemma4-26b-a4b-1m"
    /// Qwen/d1 paged long-context flagship path.
    case qwen35B1M = "qwen3-35b-1m"
    /// Fast retrieval-backed mode from the s3 carry track.
    case s3Retrieval = "s3-retrieval"

    public var displayName: String {
        switch self {
        case .gemma4E4B1M:
            return "Gemma4 E4B 1M"
        case .gemma4_26BA4B1M:
            return "Gemma4 26B-A4B 1M"
        case .qwen35B1M:
            return "Qwen3 35B 1M"
        case .s3Retrieval:
            return "S3 Retrieval"
        }
    }

    public var description: String {
        switch self {
        case .gemma4E4B1M:
            return "Reduced-tier long-context model, proven at ~1M tokens."
        case .gemma4_26BA4B1M:
            return "Full-tier Gemma4 long-context model, proven at ~1M tokens."
        case .qwen35B1M:
            return "Paged-cache Qwen flagship path for 1M-token contexts."
        case .s3Retrieval:
            return "Fast retrieval-backed long-context mode for lexical and semantic recall."
        }
    }

    public var tier: ModelTier {
        switch self {
        case .gemma4E4B1M:
            return .light
        case .gemma4_26BA4B1M, .s3Retrieval:
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
        case .gemma4_26BA4B1M:
            return 14 * 1024 * 1024 * 1024
        case .qwen35B1M:
            return 20 * 1024 * 1024 * 1024
        case .s3Retrieval:
            return nil
        }
    }

    public var maxContextTokens: Int {
        switch self {
        case .gemma4E4B1M, .gemma4_26BA4B1M, .qwen35B1M:
            return 1_000_000
        case .s3Retrieval:
            return 10_000_000
        }
    }

    public var requiredRAMTier: InfiniteRAMTier {
        switch self {
        case .gemma4E4B1M:
            return .reduced
        case .gemma4_26BA4B1M, .qwen35B1M, .s3Retrieval:
            return .full
        }
    }

    public var ramTier: String {
        requiredRAMTier.rawValue
    }

    public var backendKind: String {
        switch self {
        case .gemma4E4B1M, .gemma4_26BA4B1M, .qwen35B1M:
            return "paged-kv"
        case .s3Retrieval:
            return "retrieval"
        }
    }

    public var evidenceRef: String {
        switch self {
        case .gemma4E4B1M, .gemma4_26BA4B1M:
            return "infinite:research/18-gemma-support-matrix.md"
        case .qwen35B1M:
            return "infinite:research/26-flagship-1m.md"
        case .s3Retrieval:
            return "infinite:research/24-dense-retrieval.md"
        }
    }
}
