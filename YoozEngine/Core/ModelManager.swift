import Foundation
import Logging

/// Centralized model lifecycle management.
///
/// Handles lazy loading, keeping models warm, and memory management
/// for all AI modules (STT, LLM, VAD, TTS).
actor ModelManager {
    static let shared = ModelManager()

    private let logger = Logger(label: "live.yooz.engine.models")

    enum ModelState: Equatable {
        case notLoaded
        case loading
        case ready
        case error(String)
    }

    private(set) var loadedModels: [String: ModelState] = [:]

    func ensureModelsDirectory() throws {
        try FileManager.default.createDirectory(
            at: EngineConfig.modelsDirectory,
            withIntermediateDirectories: true
        )
    }

    // MARK: - STT Module

    func loadSTTModel(language: STTLanguage) async throws {
        let key = "stt:\(language.modelIdentifier)"

        if loadedModels[key] == .ready {
            logger.info("STT model already loaded: \(language.modelIdentifier)")
            return
        }

        loadedModels[key] = .loading
        logger.info("Loading STT model: \(language.modelIdentifier)")

        do {
            try await YoozSTTEngine.shared.start(language: language)
            loadedModels[key] = .ready
            logger.info("STT model loaded: \(language.modelIdentifier)")
        } catch {
            loadedModels[key] = .error(error.localizedDescription)
            logger.error("Failed to load STT model: \(error)")
            throw error
        }
    }

    func unloadSTTModel() {
        YoozSTTEngine.shared.stop()

        // Remove all STT model entries
        for key in loadedModels.keys where key.hasPrefix("stt:") {
            loadedModels[key] = .notLoaded
        }

        logger.info("STT model unloaded")
    }

    var sttStatus: (loaded: Bool, language: String?) {
        let engine = YoozSTTEngine.shared
        return (engine.isRunning, engine.isRunning ? engine.currentLanguage.rawValue : nil)
    }
}
