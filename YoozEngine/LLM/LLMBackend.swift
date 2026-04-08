// LLMBackend.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - Model Types

/// Yooz LLM model types for touch-up processing
enum LLMModelType: String, CaseIterable, Sendable {
    case yoozLight = "yooz-light-v3"      // Fast, fine-tuned (Qwen2.5-0.5B-4bit, 276MB)
    case yoozQuality = "yooz-quality-v3"  // High quality, fine-tuned (Qwen3-1.7B-4bit, 1037MB)

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
            return "High quality proofreading (~490ms)"
        }
    }

    var estimatedSize: Int64 {
        switch self {
        case .yoozLight:
            return 276 * 1024 * 1024   // ~276 MB
        case .yoozQuality:
            return 1037 * 1024 * 1024  // ~1037 MB
        }
    }

    var isEmbedded: Bool {
        switch self {
        case .yoozLight:
            return true   // Bundled with app
        case .yoozQuality:
            return false  // Downloaded from GHCR on-demand
        }
    }

    var baseModelId: String {
        switch self {
        case .yoozLight:
            return "qwen2.5-0.5b-instruct-4bit"
        case .yoozQuality:
            return "qwen3-1.7b-instruct-ojus-4bit"
        }
    }

    /// GHCR package name for downloading
    var packageName: String {
        "yooz-models"
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
