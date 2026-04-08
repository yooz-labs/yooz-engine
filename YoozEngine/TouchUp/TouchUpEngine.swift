// TouchUpEngine.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "TouchUpEngine")

/// Main entry point for AI touch-up processing.
///
/// The TouchUpEngine manages LLM models and provides smart routing
/// for transcription cleanup. It uses up to three backends:
/// - **Yooz-Light** (Qwen2.5-0.5B): Fast proofreading, ~200ms latency
/// - **Yooz-Quality** (Qwen3-1.7B): Higher quality proofreading, ~490ms latency
/// - **Apple Intelligence** (Foundation Models 3B): macOS 26+, structured generation
actor TouchUpEngine {

    // MARK: - Singleton

    static let shared = TouchUpEngine()

    // MARK: - Properties

    /// The light model backend (Yooz-Light, Qwen2.5-0.5B)
    private var lightModel: MLXLLMBackend?

    /// The quality model backend (Yooz-Quality, Qwen3-1.7B)
    private var qualityModel: MLXLLMBackend?

    /// The Apple Intelligence backend (Foundation Models, macOS 26+)
    private var foundationModelsBackend: FoundationModelsBackend?

    /// Bundle identifier for model loading
    private let bundleIdentifier: String

    /// Whether the engine has been preloaded
    private(set) var isPreloaded: Bool = false

    /// Whether the light model is loaded
    var isLightModelLoaded: Bool {
        get async {
            guard let model = lightModel else { return false }
            return await model.isLoaded
        }
    }

    /// Whether the quality model is loaded
    var isQualityModelLoaded: Bool {
        get async {
            guard let model = qualityModel else { return false }
            return await model.isLoaded
        }
    }

    /// Whether Apple Intelligence is available and loaded
    var isFoundationModelsLoaded: Bool {
        get async {
            guard let backend = foundationModelsBackend else { return false }
            return await backend.isLoaded
        }
    }

    // MARK: - Initialization

    init(bundleIdentifier: String = "live.yooz.engine") {
        self.bundleIdentifier = bundleIdentifier
    }

    // MARK: - Lifecycle

    /// Preload models for immediate use.
    ///
    /// This loads the light model (Yooz-Light) which is embedded in the app bundle.
    /// The quality model (Yooz-Quality) is loaded on-demand when needed.
    /// Apple Intelligence is loaded if available (macOS 26+).
    func preload(loadQuality: Bool = false) async throws {
        logger.info("Preloading TouchUpEngine...")

        if lightModel == nil {
            lightModel = MLXLLMBackend.createLight(bundleIdentifier: bundleIdentifier)
        }

        if let light = lightModel {
            try await light.load()
            logger.info("Yooz-Light model loaded")
        }

        if loadQuality {
            if qualityModel == nil {
                qualityModel = MLXLLMBackend.createQuality(bundleIdentifier: bundleIdentifier)
            }
            if let quality = qualityModel {
                try await quality.load()
                logger.info("Yooz-Quality model loaded")
            }
        }

        // Try to load Apple Intelligence if available
        await loadFoundationModelsIfAvailable()

        isPreloaded = true
        logger.info("TouchUpEngine preloaded successfully")
    }

    /// Ensure the quality model is loaded.
    /// Downloads from GHCR if not cached.
    func loadQualityModel() async throws {
        if qualityModel == nil {
            qualityModel = MLXLLMBackend.createQuality(bundleIdentifier: bundleIdentifier)
        }

        guard let quality = qualityModel else { return }

        if await !quality.isLoaded {
            try await quality.load()
            logger.info("Yooz-Quality model loaded on-demand")
        }
    }

    /// Try to load the Foundation Models backend if available on this system.
    private func loadFoundationModelsIfAvailable() async {
        let backend = FoundationModelsBackend()
        guard backend.isAvailable() else {
            logger.info("Apple Intelligence not available on this system")
            return
        }

        do {
            try await backend.load()
            foundationModelsBackend = backend
            logger.info("Apple Intelligence backend loaded")
        } catch {
            logger.warning("Failed to load Apple Intelligence: \(error.localizedDescription)")
        }
    }

    /// Unload all models from memory.
    func unload() async {
        if let light = lightModel {
            await light.unload()
        }
        if let quality = qualityModel {
            await quality.unload()
        }
        if let fm = foundationModelsBackend {
            await fm.unload()
        }
        isPreloaded = false
        logger.info("TouchUpEngine unloaded")
    }

    // MARK: - Raw LLM Generation

    /// Generate text using a specified model.
    /// Used by the `/v1/llm/generate` endpoint.
    ///
    /// - Parameters:
    ///   - prompt: The user prompt
    ///   - systemPrompt: System prompt for the model
    ///   - modelType: Which model to use (defaults to light)
    /// - Returns: Generated text
    func generate(
        prompt: String,
        systemPrompt: String,
        modelType: LLMModelType = .yoozLight
    ) async throws -> String {
        let model: MLXLLMBackend

        switch modelType {
        case .yoozLight:
            if lightModel == nil {
                lightModel = MLXLLMBackend.createLight(bundleIdentifier: bundleIdentifier)
            }
            guard let light = lightModel else {
                throw LLMError.notLoaded
            }
            if await !light.isLoaded {
                try await light.load()
            }
            model = light

        case .yoozQuality:
            try await loadQualityModel()
            guard let quality = qualityModel else {
                throw LLMError.notLoaded
            }
            model = quality
        }

        return try await model.generate(prompt: prompt, systemPrompt: systemPrompt)
    }

    /// Generate text using Apple Intelligence (Foundation Models).
    /// Used when the caller specifically wants the Apple backend.
    ///
    /// - Parameters:
    ///   - prompt: The user prompt
    ///   - systemPrompt: Optional system prompt
    /// - Returns: Generated text
    func generateWithFoundationModels(
        prompt: String,
        systemPrompt: String? = nil
    ) async throws -> String {
        guard let backend = foundationModelsBackend, await backend.isLoaded else {
            throw LLMError.notAvailable("Apple Intelligence not available or not loaded")
        }
        return try await backend.generate(prompt: prompt, systemPrompt: systemPrompt)
    }

    // MARK: - TouchUp Processing

    /// Process text with mode-aware prompt selection and smart two-model routing.
    ///
    /// - Parameters:
    ///   - text: Transcribed text (with replacements already applied)
    ///   - mode: Processing mode controlling prompt and model selection
    ///   - replacements: List of (original, replacement) tuples to validate
    /// - Returns: ProcessResult with cleaned text and metadata
    func process(
        text: String,
        mode: ServerTouchUpMode,
        replacements: [(original: String, replacement: String)] = []
    ) async -> TouchUpProcessor.ProcessResult {
        let replacementStructs = replacements.map {
            TouchUpProcessor.Replacement(original: $0.original, replacement: $0.replacement)
        }

        // Ensure light model is loaded
        guard let light = lightModel, await light.isLoaded else {
            logger.warning("Light model not loaded, using regex-only processing")
            return TouchUpProcessor.processRegexOnly(text: text, replacements: replacementStructs)
        }

        // If we have replacements, ensure quality model is loaded
        var qualityLoadError: String?
        if !replacements.isEmpty {
            do {
                try await loadQualityModel()
            } catch {
                qualityLoadError = error.localizedDescription
                logger.error("Failed to load quality model: \(error.localizedDescription)")
            }
        }

        // Check if quality model is available and loaded
        let qualityAvailable: Bool
        if let quality = qualityModel {
            qualityAvailable = await quality.isLoaded
        } else {
            qualityAvailable = false
        }


        // Select the proofread prompt based on mode and available model
        let proofreadPrompt = selectPrompt(
            for: mode,
            qualityAvailable: qualityAvailable
        )

        // Route to appropriate processing
        if replacements.isEmpty || !qualityAvailable {
            var result = await TouchUpProcessor.process(
                text: text,
                replacements: [],
                lightModel: light,
                qualityModel: light,
                proofreadPrompt: proofreadPrompt
            )
            // If quality was needed but failed to load, report degraded service
            if !replacements.isEmpty, let loadError = qualityLoadError {
                result = TouchUpProcessor.ProcessResult(
                    text: result.text,
                    keepDecisions: result.keepDecisions,
                    modelUsed: result.modelUsed,
                    latencyMs: result.latencyMs,
                    fallbackReason: "Quality model unavailable: \(loadError)"
                )
            }
            return result
        } else if let quality = qualityModel {
            return await TouchUpProcessor.process(
                text: text,
                replacements: replacementStructs,
                lightModel: light,
                qualityModel: quality,
                proofreadPrompt: proofreadPrompt
            )
        } else {
            // Should not reach here since qualityAvailable was true,
            // but fall back to light model to be safe
            return await TouchUpProcessor.process(
                text: text,
                replacements: [],
                lightModel: light,
                qualityModel: light,
                proofreadPrompt: proofreadPrompt
            )
        }
    }

    /// Process text using Apple Intelligence backend directly.
    /// Falls back to MLX models if Foundation Models unavailable.
    func processWithFoundationModels(
        text: String,
        mode: ServerTouchUpMode
    ) async -> TouchUpProcessor.ProcessResult {
        let startTime = CFAbsoluteTimeGetCurrent()

        guard let backend = foundationModelsBackend, await backend.isLoaded else {
            logger.info("Foundation Models not available, falling back to MLX")
            return await process(text: text, mode: mode)
        }

        let systemPrompt: String
        switch mode {
        case .off:
            return processRegexOnly(text: text)
        case .light, .standard:
            systemPrompt = YoozPrompts.appleStandard
        case .full:
            systemPrompt = YoozPrompts.appleFull
        }

        do {
            let result = try await backend.generate(prompt: text, systemPrompt: systemPrompt)
            let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            return TouchUpProcessor.ProcessResult(
                text: result,
                keepDecisions: [],
                modelUsed: .foundationModels,
                latencyMs: latencyMs,
                fallbackReason: nil
            )
        } catch {
            logger.error("Foundation Models generation failed: \(error.localizedDescription)")
            let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            return TouchUpProcessor.ProcessResult(
                text: text,
                keepDecisions: [],
                modelUsed: .fallbackRegex,
                latencyMs: latencyMs,
                fallbackReason: "Foundation Models failed: \(error.localizedDescription)"
            )
        }
    }

    /// Process text with regex only (no LLM).
    nonisolated func processRegexOnly(
        text: String,
        replacements: [(original: String, replacement: String)] = []
    ) -> TouchUpProcessor.ProcessResult {
        let replacementStructs = replacements.map {
            TouchUpProcessor.Replacement(original: $0.original, replacement: $0.replacement)
        }
        return TouchUpProcessor.processRegexOnly(text: text, replacements: replacementStructs)
    }

    // MARK: - Prompt Selection

    /// Select the appropriate proofread prompt based on mode and model availability.
    private func selectPrompt(
        for mode: ServerTouchUpMode,
        qualityAvailable: Bool
    ) -> String {
        switch mode {
        case .off:
            // Should not reach here; off mode skips LLM entirely
            return TouchUpPrompts.proofread
        case .light:
            return YoozPrompts.lightStandard
        case .standard:
            if qualityAvailable {
                return YoozPrompts.qualityStandard
            } else {
                return YoozPrompts.lightStandard
            }
        case .full:
            if qualityAvailable {
                return YoozPrompts.qualityFull
            } else {
                return YoozPrompts.lightFull
            }
        }
    }

    // MARK: - Model Info

    /// Check if the quality model is cached locally
    var isQualityModelCached: Bool {
        get async {
            guard let quality = qualityModel else {
                let temp = MLXLLMBackend.createQuality(bundleIdentifier: bundleIdentifier)
                return await temp.isModelCached
            }
            return await quality.isModelCached
        }
    }

    /// Get model info for display
    func getModelInfo() async -> (light: LLMModelInfo, quality: LLMModelInfo) {
        let lightLoaded = await isLightModelLoaded
        let qualityLoaded = await isQualityModelLoaded
        let qualityCached = await isQualityModelCached

        return (
            light: LLMModelInfo(
                type: .yoozLight,
                isLoaded: lightLoaded,
                isCached: true  // Always embedded
            ),
            quality: LLMModelInfo(
                type: .yoozQuality,
                isLoaded: qualityLoaded,
                isCached: qualityCached
            )
        )
    }
}

// MARK: - LLM Model Info

/// Information about a model's status (renamed from ModelInfo to avoid collision with APITypes.ModelInfo)
struct LLMModelInfo: Sendable {
    let type: LLMModelType
    let isLoaded: Bool
    let isCached: Bool

    init(type: LLMModelType, isLoaded: Bool, isCached: Bool) {
        self.type = type
        self.isLoaded = isLoaded
        self.isCached = isCached
    }
}
