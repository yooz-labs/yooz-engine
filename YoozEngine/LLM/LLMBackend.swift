// LLMBackend.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

// MARK: - Model Types

/// Yooz LLM model types for touch-up processing
enum LLMModelType: String, CaseIterable, Sendable {
    case yoozLightV1 = "yooz-light-v1"      // Fast, default (Qwen2.5-0.5B-4bit, 278MB)
    case yoozQualityV1 = "yooz-quality-v1"  // Higher quality (Qwen3-1.7B-4bit, 1.0GB)

    var displayName: String {
        switch self {
        case .yoozLightV1:
            return "Yooz-Light"
        case .yoozQualityV1:
            return "Yooz-Quality"
        }
    }

    var description: String {
        switch self {
        case .yoozLightV1:
            return "Fast proofreading (~120ms)"
        case .yoozQualityV1:
            return "High quality validation (~300ms)"
        }
    }

    var estimatedSize: Int64 {
        switch self {
        case .yoozLightV1:
            return 278 * 1024 * 1024   // ~278 MB
        case .yoozQualityV1:
            return 1024 * 1024 * 1024  // ~1.0 GB
        }
    }

    var isEmbedded: Bool {
        switch self {
        case .yoozLightV1:
            return true   // Bundled with app
        case .yoozQualityV1:
            return false  // Downloaded from GHCR on-demand
        }
    }

    var baseModelId: String {
        switch self {
        case .yoozLightV1:
            return "qwen2.5-0.5b-instruct-4bit"
        case .yoozQualityV1:
            return "qwen3-1.7b-instruct-ojus-4bit"
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
