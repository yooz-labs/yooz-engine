// MLXLLMBackend.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

#if canImport(MLXLMCommon)
import MLX
import MLXLMCommon
#endif

private let logger = Logger(subsystem: "live.yooz.engine", category: "MLXLLMBackend")

/// MLX-Swift backend for Yooz LLM models.
/// Supports both embedded (Yooz-Light) and downloaded (Yooz-Quality) models.
/// Implements system prompt KV cache optimization to skip re-computing the
/// system prompt tokens on subsequent calls with the same system prompt.
actor MLXLLMBackend: LLMBackend {

    // MARK: - Properties

    let identifier: String
    let modelType: LLMModelType

    /// Whether a model is loaded and ready for generation.
    private(set) var isLoaded = false

    /// Download progress (0.0 to 1.0) for non-embedded models
    private(set) var downloadProgress: Double = 0

    private let downloader: ModelDownloader

    private let bundleIdentifier: String

    #if canImport(MLXLMCommon)
    private var modelContainer: ModelContainer?

    /// Cached KV state containing only system prompt tokens.
    /// After the first call with a given system prompt, we snapshot the KV cache
    /// at the system prompt boundary so subsequent calls skip re-computing it.
    private var cachedPromptKVState: [[MLXArray]]?

    /// Number of tokens in the cached system prompt KV state
    private var cachedPromptTokenCount: Int = 0

    /// System prompt that produced the cached KV state
    private var cachedSystemPrompt: String?
    #endif

    // MARK: - Initialization

    init(
        modelType: LLMModelType,
        bundleIdentifier: String = "live.yooz.engine"
    ) {
        self.identifier = modelType.rawValue
        self.modelType = modelType
        self.bundleIdentifier = bundleIdentifier
        self.downloader = ModelDownloader(bundleIdentifier: bundleIdentifier)
    }

    // MARK: - LLMBackend Protocol

    func load() async throws {
        guard !isLoaded else { return }

        #if canImport(MLXLMCommon)
        logger.info("Loading model \(self.modelType.rawValue)...")

        do {
            let modelDirectory: URL = if modelType.isEmbedded {
                try getEmbeddedModelDirectory()
            } else {
                try await downloader.downloadModel(modelType) { [weak self] progress in
                    Task {
                        await self?.setDownloadProgress(progress)
                    }
                }
            }

            logger.info("Model directory: \(modelDirectory.path)")

            let configuration = ModelConfiguration(directory: modelDirectory)
            modelContainer = try await loadModelContainer(configuration: configuration)

            isLoaded = true
            logger.info("Model \(self.modelType.rawValue) loaded successfully")
        } catch let error as LLMError {
            throw error
        } catch {
            logger.error("Failed to load model: \(error.localizedDescription)")
            throw LLMError.loadFailed(error.localizedDescription)
        }
        #else
        logger.error("MLXLMCommon not available")
        throw LLMError.notAvailable("MLX framework not linked. Please rebuild with mlx-swift-lm package.")
        #endif
    }

    func unload() {
        #if canImport(MLXLMCommon)
        modelContainer = nil
        cachedPromptKVState = nil
        cachedPromptTokenCount = 0
        cachedSystemPrompt = nil
        #endif
        isLoaded = false
        downloadProgress = 0
        logger.info("Model \(self.modelType.rawValue) unloaded")
    }

    /// Clear the cached system prompt KV state, forcing re-computation on the next call.
    func clearSession() {
        #if canImport(MLXLMCommon)
        cachedPromptKVState = nil
        cachedPromptTokenCount = 0
        cachedSystemPrompt = nil
        #endif
        logger.debug("Prompt cache cleared")
    }

    func generate(prompt: String, systemPrompt: String) async throws -> String {
        guard isLoaded else {
            throw LLMError.notLoaded
        }

        #if canImport(MLXLMCommon)
        guard let container = modelContainer else {
            throw LLMError.notLoaded
        }

        do {
            let estimatedTokens = max(100, (prompt.count / 3) + 50)

            // Invalidate cache if system prompt changed
            if systemPrompt != cachedSystemPrompt {
                cachedPromptKVState = nil
                cachedPromptTokenCount = 0
                cachedSystemPrompt = nil
            }

            // Tokenize the full [system, user] message sequence
            let messages: [Chat.Message] = [
                .system(systemPrompt),
                .user(prompt)
            ]
            let userInput = UserInput(chat: messages)
            let fullInput = try await container.prepare(input: userInput)

            let params = GenerateParameters(
                maxTokens: estimatedTokens,
                temperature: 0.1,
                topP: 0.9
            )

            let savedKVState = cachedPromptKVState
            let savedTokenCount = cachedPromptTokenCount

            // On the first call, find the system prompt token boundary by comparing
            // two full sequences with different user messages. Chat templates are not
            // additive (system-only tokenization differs from the system portion of a
            // full sequence), so we find the common prefix of two full sequences instead.
            var sysOnlyTokenCount = savedTokenCount
            if savedKVState == nil {
                let probeMessages: [Chat.Message] = [
                    .system(systemPrompt),
                    .user("_")
                ]
                let probeInput = UserInput(chat: probeMessages)
                let probeLMInput = try await container.prepare(input: probeInput)

                let fullTokens = fullInput.text.tokens
                let probeTokens = probeLMInput.text.tokens
                let minLen = min(fullTokens.size, probeTokens.size)

                // Find where the two sequences diverge; that's the user content boundary
                if minLen > 0 {
                    let match = (fullTokens[..<minLen] .== probeTokens[..<minLen])
                    let matchArray = match.asArray(Bool.self)
                    var prefixLen = 0
                    for v in matchArray {
                        if v { prefixLen += 1 } else { break }
                    }
                    sysOnlyTokenCount = prefixLen
                }
            }
            let sysCount = sysOnlyTokenCount

            // If we have a cached system prompt KV state, skip the system tokens and
            // only feed the user portion. Otherwise, feed all tokens.
            let hasCachedState = savedKVState != nil && sysCount > 0
            let inputForModel: LMInput
            if hasCachedState {
                let userTokens = fullInput.text.tokens[sysCount...]
                inputForModel = LMInput(text: .init(tokens: userTokens))
            } else {
                inputForModel = fullInput
            }

            let (response, kvSnapshot): (String, [[MLXArray]]?) = try await container.perform { context in
                // Create fresh KV cache and restore cached system prompt state if available
                var cache = context.model.newCache(parameters: params)
                if let savedState = savedKVState {
                    for i in 0..<min(cache.count, savedState.count) {
                        cache[i].state = savedState[i]
                    }
                    eval(cache)
                }

                // Generate using the lower-level API with our managed cache
                let stream = try MLXLMCommon.generate(
                    input: inputForModel,
                    cache: cache,
                    parameters: params,
                    context: context
                )

                var text = ""
                for await generation in stream {
                    if case let .chunk(chunk) = generation {
                        text += chunk
                    }
                }

                // Trim the cache back to just the system prompt tokens and snapshot
                var snapshot: [[MLXArray]]? = nil
                if sysCount > 0 {
                    let currentOffset = cache.first?.offset ?? 0
                    let trimAmount = currentOffset - sysCount
                    if trimAmount > 0 {
                        for layer in cache {
                            layer.trim(trimAmount)
                        }
                        eval(cache)
                    }
                    snapshot = cache.map(\.state)
                }
                return (text, snapshot)
            }

            // Update cached state
            if let snapshot = kvSnapshot {
                cachedPromptKVState = snapshot
                cachedPromptTokenCount = sysCount
                cachedSystemPrompt = systemPrompt
                if savedKVState == nil {
                    logger.info("Cached system prompt KV (\(sysCount) tokens)")
                }
            }

            logger.debug("Generation complete, got \(response.count) chars")
            return postProcessResponse(response, originalInput: prompt)
        } catch {
            cachedPromptKVState = nil
            cachedPromptTokenCount = 0
            cachedSystemPrompt = nil
            logger.error("Generation failed: \(error.localizedDescription)")
            throw LLMError.generationFailed(error.localizedDescription)
        }
        #else
        throw LLMError.notAvailable("MLX framework not linked")
        #endif
    }

    // MARK: - Progress

    private func setDownloadProgress(_ value: Double) {
        downloadProgress = value
        logger.debug("Download progress: \(Int(value * 100))%")
    }

    // MARK: - Embedded Model

    private func getEmbeddedModelDirectory() throws -> URL {
        if let bundleURL = Bundle.main.url(forResource: modelType.rawValue, withExtension: nil) {
            logger.info("Found embedded model in bundle: \(bundleURL.path)")
            return bundleURL
        }

        if let resourcesURL = Bundle.main.resourceURL?.appendingPathComponent(modelType.rawValue) {
            if FileManager.default.fileExists(atPath: resourcesURL.path) {
                logger.info("Found embedded model in resources: \(resourcesURL.path)")
                return resourcesURL
            }
        }

        // Look in Application Support (for dev builds)
        if let appSupportURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let modelsDir = appSupportURL.appendingPathComponent(bundleIdentifier).appendingPathComponent("Models")
            let modelDir = modelsDir.appendingPathComponent(modelType.rawValue)
            if FileManager.default.fileExists(atPath: modelDir.path) {
                logger.info("Found model in Application Support: \(modelDir.path)")
                return modelDir
            }

            let baseModelDir = modelsDir.appendingPathComponent(modelType.baseModelId)
            if FileManager.default.fileExists(atPath: baseModelDir.path) {
                logger.info("Found model (base ID) in Application Support: \(baseModelDir.path)")
                return baseModelDir
            }
        }

        // Also check EngineConfig.modelsDirectory
        let engineModelsDir = EngineConfig.modelsDirectory.appendingPathComponent(modelType.rawValue)
        if FileManager.default.fileExists(atPath: engineModelsDir.path) {
            logger.info("Found model in engine models directory: \(engineModelsDir.path)")
            return engineModelsDir
        }

        #if DEBUG
        let devPath = "/Volumes/S1/HuggingFace/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots"
        if FileManager.default.fileExists(atPath: devPath) {
            let contents = try FileManager.default.contentsOfDirectory(atPath: devPath)
            if let firstSnapshot = contents.first {
                let snapshotPath = URL(fileURLWithPath: devPath).appendingPathComponent(firstSnapshot)
                logger.info("Found model in dev HuggingFace cache: \(snapshotPath.path)")
                return snapshotPath
            }
        }
        #endif

        throw LLMError.notAvailable("Embedded model \(modelType.rawValue) not found in bundle. For development, copy model to ~/Library/Application Support/\(bundleIdentifier)/Models/\(modelType.rawValue)/")
    }

    // MARK: - Post-Processing

    private func postProcessResponse(_ response: String, originalInput: String) -> String {
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)

        let commentaryPrefixes = [
            "here's", "here is", "i'm sorry", "i apologize",
            "as a transcription", "as an ai", "the corrected",
            "the cleaned", "the revised", "i think", "i believe",
            "it seems", "let me", "sure,", "certainly,", "of course,"
        ]

        let lowercased = trimmed.lowercased()
        let hasCommentary = commentaryPrefixes.contains { lowercased.hasPrefix($0) }

        if hasCommentary {
            logger.debug("Detected commentary, attempting to extract clean text...")

            if let extracted = extractQuotedText(from: trimmed) {
                return extracted
            }

            if let extracted = extractAfterColon(from: trimmed) {
                return extracted
            }

            logger.warning("Could not extract clean text, returning original input")
            return originalInput
        }

        if trimmed.count > originalInput.count * 3 && trimmed.count > 200 {
            logger.warning("Response suspiciously long, returning original input")
            return originalInput
        }

        return trimmed
    }

    private func extractQuotedText(from text: String) -> String? {
        let pattern = "\"([^\"]+)\""
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(in: text, options: [], range: NSRange(text.startIndex..., in: text)),
              let range = Range(match.range(at: 1), in: text) else {
            return nil
        }
        return String(text[range])
    }

    private func extractAfterColon(from text: String) -> String? {
        guard let colonIndex = text.firstIndex(of: ":") else { return nil }
        let afterColon = text[text.index(after: colonIndex)...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if afterColon.count > 10 {
            return afterColon
        }
        return nil
    }

    // MARK: - Model Info

    var isModelCached: Bool {
        get async {
            if modelType.isEmbedded {
                return true
            }
            return await downloader.isModelCached(modelType)
        }
    }
}

// MARK: - Factory

extension MLXLLMBackend {
    static func create(
        for type: LLMModelType,
        bundleIdentifier: String = "live.yooz.engine"
    ) -> MLXLLMBackend {
        return MLXLLMBackend(modelType: type, bundleIdentifier: bundleIdentifier)
    }

    static func createLight(bundleIdentifier: String = "live.yooz.engine") -> MLXLLMBackend {
        return create(for: .yoozLight, bundleIdentifier: bundleIdentifier)
    }

    static func createQuality(bundleIdentifier: String = "live.yooz.engine") -> MLXLLMBackend {
        return create(for: .yoozQuality, bundleIdentifier: bundleIdentifier)
    }
}
