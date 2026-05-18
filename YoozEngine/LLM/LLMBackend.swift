// LLMBackend.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

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
    /// Fast proofread tier. Yooz-Light v2 LoRA on Qwen2.5-0.5B base.
    case yoozLight = "yooz-light-v2"
    /// High-quality proofread tier. Yooz-Quality v2 LoRA on Qwen3.5-0.8B base.
    case yoozQuality = "yooz-quality-v2"

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
            return "Fast proofreading (~200ms)"
        case .yoozQuality:
            return "High quality proofreading (~310ms)"
        }
    }

    /// Approximate on-disk size after HF download (used for picker UX
    /// hints in consumer apps). Numbers are the published 4-bit MLX
    /// snapshot sizes, not raw weights.
    public var estimatedSize: Int64 {
        switch self {
        case .yoozLight:
            return 276 * 1024 * 1024   // ~276 MB (Qwen2.5-0.5B-Instruct-4bit)
        case .yoozQuality:
            return 424 * 1024 * 1024   // ~424 MB (Yooz-Quality-v2 fused 4-bit)
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
            return "YoozLabs/Yooz-Light-v2-Qwen2.5-0.5B-LoRA"
        case .yoozQuality:
            // v2 publishes fused weights only; the adapter-pollution loader
            // crash history is regression-guarded by
            // `testPreloadLoadsQualityModelV2`.
            return "YoozLabs/Yooz-Quality-v2-Qwen3.5-0.8B-LoRA"
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
    func generate(prompt: String, systemPrompt: String) async throws -> String
}
