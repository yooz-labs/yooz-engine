import Foundation

/// Centralized model lifecycle management.
///
/// Handles lazy loading, keeping models warm, and memory management
/// for all AI modules (STT, LLM, VAD, TTS).
@MainActor
final class ModelManager: ObservableObject {
    static let shared = ModelManager()

    @Published var loadedModels: [String: ModelState] = [:]

    enum ModelState {
        case notLoaded
        case loading
        case ready
        case error(String)
    }

    func ensureModelsDirectory() {
        try? FileManager.default.createDirectory(
            at: EngineConfig.modelsDirectory,
            withIntermediateDirectories: true
        )
    }

    // Module registration will be added as modules are migrated
}
