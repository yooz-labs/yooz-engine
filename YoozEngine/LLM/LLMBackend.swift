// LLMBackend.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

// MARK: - Model Types

/// Yooz LLM model types for touch-up processing.
///
/// Identifiers are stable wire values; `/v1/llm/generate` accepts the raw
/// value as the `model` field. Order of cases matches the public model
/// lineup (light first, quality second).
///
/// Both tiers download from Hugging Face on first use via the
/// `#huggingFaceLoadModelContainer` macro in `MLXLLMBackend`. There is no
/// embedded / bundled model path — packaged builds and fresh installs
/// behave identically. Cached snapshots land under
/// `~/.cache/huggingface/hub/` per swift-transformers `Hub` defaults.
///
/// `public` because `APIServer` (a different target on the modular
/// build) consumes this enum directly via the picker routes.
public enum LLMModelType: String, CaseIterable, Sendable {
    /// Fast proofread tier. Yooz-Light v3, fused 6-bit on the KD
    /// Qwen3.5-0.8B QAT base (yooz-benchmark#29).
    case yoozLight = "yooz-light-v3"
    /// High-quality rewrite tier. Yooz-Quality v3, fused 6-bit on the KD
    /// Qwen3.5-4B QAT base. Replaces the former Qwen3.5-9B fallback.
    case yoozQuality = "yooz-quality-v3"

    public var displayName: String {
        switch self {
        case .yoozLight:
            return "Yooz-Light"
        case .yoozQuality:
            return "Yooz-Quality"
        }
    }

    public var description: String {
        switch self {
        case .yoozLight:
            return "Fast proofreading (~300ms)"
        case .yoozQuality:
            return "High quality rewriting (~1s)"
        }
    }

    /// Approximate on-disk size after HF download (used for picker UX
    /// hints in consumer apps). Numbers are the published 4-bit MLX
    /// snapshot sizes, not raw weights.
    public var estimatedSize: Int64 {
        switch self {
        case .yoozLight:
            return 605 * 1024 * 1024   // ~605 MB (Yooz-Light-v3 fused 6-bit)
        case .yoozQuality:
            return 3277 * 1024 * 1024  // ~3.2 GB (Yooz-Quality-v3 fused 6-bit)
        }
    }

    /// Hugging Face model identifier. Pulled by
    /// `loadModelContainer(from: #hubDownloader(), …, configuration:)`
    /// on first load. `revision` defaults to `main`; pin a commit here
    /// only if a future upstream change breaks compatibility with our
    /// backend assumptions.
    public var huggingFaceID: String {
        switch self {
        case .yoozLight:
            return "YoozLabs/Yooz-Light-v3-Qwen3.5-0.8B"
        case .yoozQuality:
            // v3 publishes fused weights only (publish-gated: no adapter
            // files or lora_* keys); the adapter-pollution loader crash
            // history is regression-guarded by
            // `testPreloadLoadsQualityModelV2`.
            return "YoozLabs/Yooz-Quality-v3-Qwen3.5-4B"
        }
    }
}

// MARK: - Errors

public enum LLMError: Error, LocalizedError, Sendable {
    case notLoaded
    case loadFailed(String)
    case generationFailed(String)
    case notAvailable(String)
    case downloadFailed(String)
    case parsingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "Model not loaded"
        case .loadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .generationFailed(let reason):
            return "Generation failed: \(reason)"
        case .notAvailable(let reason):
            return "Model not available: \(reason)"
        case .downloadFailed(let reason):
            return "Download failed: \(reason)"
        case .parsingFailed(let reason):
            return "Failed to parse response: \(reason)"
        }
    }
}

// MARK: - Protocol

/// Protocol for LLM backends used in touch-up processing.
///
/// Kept `internal` on purpose; `TouchUpEngine` is the only out-of-module
/// caller and it exposes its own domain API, not the backend abstraction.
protocol LLMBackend: Actor {
    var identifier: String { get }
    var modelType: LLMModelType { get }
    var isLoaded: Bool { get }

    func load() async throws
    func unload()
    /// `workloadClass` (engine#228) tells the backend whether this call is
    /// latency-sensitive (`.interactive`, admitted immediately) or
    /// throughput work that can queue/yield behind interactive activity
    /// (`.background`, the default every existing caller gets via
    /// `TouchUpProcessor.process`). No default here — the one existential
    /// call site (`TouchUpProcessor.process`) passes it explicitly so the
    /// classification is never accidentally implicit.
    func generate(
        prompt: String,
        systemPrompt: String,
        workloadClass: MLXWorkloadClass
    ) async throws -> String
}
