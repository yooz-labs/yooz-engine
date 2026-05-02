import Foundation

/// KV cache compression mode for the MLX LLM backend.
///
/// `.off` keeps the upstream FP16 KV path (default; no behavioral change).
/// `.turbo3` enables SharpAI's TurboQuant 3-bit packing on KV cache layers
/// whose head_dim is 128 or 256, gated above 2048 tokens by the upstream
/// `KVCacheSimple.turboMinActivationTokens`. Short prompts (TouchUp, chat
/// turns) stay on the FP16 path even with `.turbo3` selected — the upstream
/// gate only flips compression on once the cache exceeds the activation
/// threshold, so short workloads pay zero overhead.
public enum KVCompressionMode: String, Codable, Sendable {
    case off
    case turbo3
}

enum EngineConfig {
    static let port: Int = 19920
    static let host: String = "127.0.0.1"
    static let version: String = "0.5.0"

    /// Default KV cache compression mode for new MLX LLM backends.
    /// Can be overridden per-backend via the `kvCompression` init parameter
    /// or per-request via the `/v1/llm/generate` request body.
    static let kvCompression: KVCompressionMode = .off

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
