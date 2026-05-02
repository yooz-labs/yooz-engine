import Foundation

enum EngineConfig {
    static let port: Int = 19920
    static let host: String = "127.0.0.1"
    static let version: String = "0.5.0"

    /// Default STT backend resolved at startup. Driven by the
    /// `YOOZ_STT_BACKEND` env var so dev and tests can flip the flag
    /// without writing a config file. Unknown values fall back to
    /// `.parakeet` rather than crashing.
    static var sttBackend: STTBackendID {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "YOOZ_STT_BACKEND"
            ],
            let parsed = STTBackendID(rawValue: raw)
        else {
            return .parakeet
        }
        return parsed
    }

    static let modelsDirectory: URL = {
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            fatalError("EngineConfig: Application Support directory not found")
        }
        return appSupport.appendingPathComponent("YoozEngine/Models")
    }()

    static let cacheDirectory: URL = {
        guard let caches = FileManager.default.urls(
            for: .cachesDirectory,
            in: .userDomainMask
        ).first else {
            fatalError("EngineConfig: Caches directory not found")
        }
        return caches.appendingPathComponent("live.yooz.engine")
    }()
}
