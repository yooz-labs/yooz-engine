// LLMBackend.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - Model Types

/// Yooz LLM model types for touch-up processing.
///
/// Both tiers download from Hugging Face on first use via the
/// `#huggingFaceLoadModelContainer` macro in `MLXLLMBackend`. There is no
/// embedded / bundled model path — packaged builds and fresh installs
/// behave identically (no `/Volumes/S1` dev-cache fallback). Cached
/// snapshots land under `~/.cache/huggingface/hub/` per swift-transformers
/// `Hub` defaults.
enum LLMModelType: String, CaseIterable, Sendable {
    /// Fast proofread tier. Currently unfinetuned base — see issue #91 for
    /// the planned `Yooz-Light v2` LoRA on the gold_standard_v3 corpus.
    case yoozLight = "yooz-light-v3"
    /// High-quality proofread tier. Fine-tuned LoRA fused into the base
    /// (Qwen3.5-0.8B-MLX-4bit).
    case yoozQuality = "yooz-quality-v3"

    var displayName: String {
        switch self {
        case .yoozLight:
            return "Yooz-Light"
        case .yoozQuality:
            return "Yooz-Quality"
        }
    }

    var description: String {
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
    var estimatedSize: Int64 {
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
    var huggingFaceID: String {
        switch self {
        case .yoozLight:
            // Stock Qwen2.5-0.5B-Instruct 4-bit MLX. No fine-tune yet
            // (tracked by #91). Wired here so the engine builds out of
            // the box and consumers can switch to the fine-tuned LoRA
            // by changing this single string when Light v2 ships on HF.
            return "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        case .yoozQuality:
            // TEMPORARY: stock Qwen3.5-0.8B base. The fine-tuned
            // checkpoint at `YoozLabs/Yooz-Quality-v2-Qwen3.5-0.8B-LoRA`
            // ships an `adapters/` subdirectory alongside the fused
            // `model.safetensors`. mlx-swift-lm's loader auto-applies
            // the adapter on top of the already-fused weights, which
            // throws `Unhandled keys [lora_a, lora_b] in QuantizedLinear`.
            // Switch this to `YoozLabs/Yooz-Quality-v3-...` once the v3
            // sweep winner (issue #82) is republished without the
            // adapter pollution (tracking issue #92).
            return "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        }
    }
}

// MARK: - Errors

enum LLMError: Error, LocalizedError, Sendable {
    case notLoaded
    case loadFailed(String)
    case generationFailed(String)
    case notAvailable(String)
    case downloadFailed(String)
    case parsingFailed(String)

    var errorDescription: String? {
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

/// Protocol for LLM backends used in touch-up processing
protocol LLMBackend: Actor {
    var identifier: String { get }
    var modelType: LLMModelType { get }
    var isLoaded: Bool { get }

    func load() async throws
    func unload()
    func generate(prompt: String, systemPrompt: String) async throws -> String
}
