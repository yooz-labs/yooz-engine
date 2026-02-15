// MLXLLMBackend.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

#if canImport(MLXLMCommon)
import MLXLMCommon
#endif

private let logger = Logger(subsystem: "live.yooz.engine", category: "MLXLLMBackend")

/// MLX-Swift backend for Yooz LLM models.
/// Supports both embedded (Yooz-Light) and downloaded (Yooz-Quality) models.
actor MLXLLMBackend: LLMBackend {

    // MARK: - Properties

    let identifier: String
    let modelType: LLMModelType

    /// Whether a model is loaded and ready for generation.
    /// Derived from the presence of a loaded model container.
    var isLoaded: Bool {
        #if canImport(MLXLMCommon)
        return modelContainer != nil
        #else
        return false
        #endif
    }

    /// Download progress (0.0 to 1.0) for non-embedded models
    private(set) var downloadProgress: Double = 0

    private let downloader: ModelDownloader

    private let bundleIdentifier: String

    #if canImport(MLXLMCommon)
    private var modelContainer: ModelContainer?
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
            let modelDirectory: URL

            if modelType.isEmbedded {
                modelDirectory = try getEmbeddedModelDirectory()
            } else {
                modelDirectory = try await downloader.downloadModel(modelType) { [weak self] progress in
                    Task {
                        await self?.setDownloadProgress(progress)
                    }
                }
            }

            logger.info("Model directory: \(modelDirectory.path)")

            let configuration = ModelConfiguration(directory: modelDirectory)
            modelContainer = try await loadModelContainer(configuration: configuration)

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
        #endif
        downloadProgress = 0
        logger.info("Model \(self.modelType.rawValue) unloaded")
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
            logger.debug("Creating chat session...")

            // Rough token estimate: ~3 chars per token + 50 for JSON overhead.
            // Floor at 100 to avoid truncating short inputs.
            let estimatedTokens = max(100, (prompt.count / 3) + 50)

            let session = ChatSession(
                container,
                instructions: systemPrompt,
                generateParameters: GenerateParameters(
                    maxTokens: estimatedTokens,
                    temperature: 0.1,
                    topP: 0.9
                )
            )

            logger.debug("Generating for: \(prompt.prefix(50))...")

            let response = try await session.respond(to: prompt)

            logger.debug("Generation complete, got \(response.count) chars")

            let cleaned = postProcessResponse(response, originalInput: prompt)
            return cleaned
        } catch let error as LLMError {
            throw error
        } catch {
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
        return create(for: .yoozLightV1, bundleIdentifier: bundleIdentifier)
    }

    static func createQuality(bundleIdentifier: String = "live.yooz.engine") -> MLXLLMBackend {
        return create(for: .yoozQualityV1, bundleIdentifier: bundleIdentifier)
    }
}
