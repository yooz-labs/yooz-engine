import EngineCore
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

    // Module registration will be added as modules are migrated
}
