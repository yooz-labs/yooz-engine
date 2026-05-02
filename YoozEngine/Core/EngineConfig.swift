import Foundation

/// KV cache compression mode for the MLX LLM backend.
///
/// `.off` keeps the upstream FP16 KV path (default; no behavioral change).
/// `.turbo3` enables SharpAI's TurboQuant 3-bit packing on KV cache layers
/// whose head_dim is 128 or 256 (and 512 split into 2x256 virtual heads),
/// gated above 2048 tokens by the upstream
/// `KVCacheSimple.turboMinActivationTokens`. Short prompts (TouchUp, chat
/// turns) stay on the FP16 path even with `.turbo3` selected — the upstream
/// gate only flips compression on once the cache exceeds the activation
/// threshold, so short workloads pay zero overhead.
///
/// Layers whose `head_dim` is not in `{128, 256, 512}` self-disable the
/// flag at runtime and emit a one-time fallback log; the engine separately
/// counts how many cache layers accepted the flag and logs an error if the
/// total is zero. See `MLXLLMBackend.lastTurboLayersEnabled`.
public enum KVCompressionMode: String, Codable, Sendable {
    case off
    case turbo3
}

enum EngineConfig {
    static let port: Int = 19920
    static let host: String = "127.0.0.1"
    static let version: String = "0.6.0"

    /// Default KV cache compression mode for new MLX LLM backends.
    ///
    /// Resolution order (highest priority first):
    /// 1. Per-request override via `/v1/llm/generate` request body
    ///    (`kv_compression` or `kvCompression` key, decoded into
    ///    `LLMGenerateServerRequest.kvCompression`).
    /// 2. Per-backend `kvCompression` argument to `MLXLLMBackend.init` or
    ///    the `MLXLLMBackend.create*` factories.
    /// 3. This engine-wide default (currently `.off`).
    ///
    /// All three paths are exercised by `KVCompressionTests`. To roll
    /// turbo3 out globally, flip this to `.turbo3` — every cached backend
    /// (TouchUp's `lightModel` / `qualityModel`) picks it up at next
    /// construction. Per-request overrides bypass the cached models and
    /// build a fresh backend for that single call.
    static let kvCompression: KVCompressionMode = .off

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

    // MARK: - Telemetry (Phase 6)

    /// Whether the user has opted into local STT telemetry. Default
    /// `false`. Driven by the `YOOZ_TELEMETRY_STT` env var:
    /// `"local"` opts in to the on-disk JSONL sink; any other value
    /// (including unset) opts out.
    ///
    /// No HTTP route surfaces this flag or the recorded metrics —
    /// telemetry consumption is local-file-only.
    static var telemetryOptedIn: Bool {
        let raw = ProcessInfo.processInfo.environment["YOOZ_TELEMETRY_STT"]
        return raw == "local"
    }

    /// Directory the on-disk JSONL metrics file lives in. Defaults
    /// to `~/Library/Application Support/YoozEngine/telemetry/`.
    /// Tests redirect via `YOOZ_TELEMETRY_DIR`.
    static var telemetryDirectory: URL {
        if let override = ProcessInfo.processInfo.environment[
            "YOOZ_TELEMETRY_DIR"
        ] {
            return URL(fileURLWithPath: override)
        }
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            // Fall back to a temp location rather than crashing — a
            // sink failure must never take down the engine.
            return URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("YoozEngine/telemetry")
        }
        return appSupport
            .appendingPathComponent("YoozEngine/telemetry")
    }

    /// Path to the JSONL metrics file under `telemetryDirectory`.
    static var sttMetricsFileURL: URL {
        telemetryDirectory.appendingPathComponent("stt_metrics.jsonl")
    }
}
