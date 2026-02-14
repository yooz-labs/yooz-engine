import Foundation

enum EngineConfig {
    static let port: Int = 19920
    static let host: String = "127.0.0.1"
    static let version: String = "0.1.0"

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
