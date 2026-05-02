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
