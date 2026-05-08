// EngineConfig.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

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

/// Process-wide engine configuration.
///
/// Lives in `EngineCore` so every module target (STT, LLM, VAD, Grammar)
/// can read ports, version, and on-disk locations without pulling in the
/// `YoozEngine` app target.
public enum EngineConfig {
    public static let port: Int = 19920
    public static let host: String = "127.0.0.1"
    public static let version: String = "0.6.0"

    /// Convenience accessor for the active build variant. Mirrors
    /// `BuildVariant.current` so callers reading other engine config
    /// values can stay on the `EngineConfig` namespace.
    public static var variant: BuildVariant { BuildVariant.current }

    /// Environment variable host apps set when launching the engine as an
    /// embedded helper. When `YOOZ_ENGINE_HEADLESS=1` the app suppresses its
    /// menu-bar UI and Settings scene, but still starts the API server and
    /// writes to the same OSLog subsystem. This contract is consumed by
    /// `YoozEngineApp` and `EngineAppDelegate`; host apps (e.g. yooz-whisper's
    /// `EngineHelperController`) must pass the variable through
    /// `NSWorkspace.OpenConfiguration.environment` when spawning the helper.
    public static let headlessEnvVar = "YOOZ_ENGINE_HEADLESS"

    /// `true` when the engine process was launched with
    /// `YOOZ_ENGINE_HEADLESS=1`. Used to gate menu-bar/Settings UI, modal
    /// alerts, and any other host-app-inappropriate behaviour.
    public static var isHelper: Bool {
        ProcessInfo.processInfo.environment[headlessEnvVar] == "1"
    }

    /// `~/Library/Application Support/YoozEngine/Models` — long-lived model
    /// artifacts keyed by `LLMModelType.rawValue` or STT model identifier.
    public static let modelsDirectory: URL = {
        guard let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            fatalError("EngineConfig: Application Support directory not found")
        }
        return appSupport.appendingPathComponent("YoozEngine/Models")
    }()

    /// `~/Library/Caches/live.yooz.engine` — ephemeral downloads, tarballs,
    /// partial models. Safe for the OS to evict.
    public static let cacheDirectory: URL = {
        guard let caches = FileManager.default.urls(
            for: .cachesDirectory,
            in: .userDomainMask
        ).first else {
            fatalError("EngineConfig: Caches directory not found")
        }
        return caches.appendingPathComponent("live.yooz.engine")
    }()

    /// Whether the eager-load loop runs when `APIServer.start()`
    /// completes. Production keeps this on; tests turn it off because
    /// the route tests boot a real `APIServer` and a kickoff would
    /// drag in MLX weight loads on every test invocation (the eager
    /// loader would still finish quickly for VAD / Grammar but the
    /// MLX STT + LLM paths take seconds and run network fetches).
    ///
    /// Resolution order:
    /// 1. `YOOZ_DISABLE_EAGER_LOAD=1` env var → off (explicit
    ///    override; useful when running the engine binary against
    ///    a remote model store you don't want pre-warmed).
    /// 2. Detected XCTest invocation → off (`XCInjectBundleInto` is
    ///    set by xctest; the inject env keys are documented stable).
    /// 3. Otherwise → on.
    public static var eagerLoadOnLaunch: Bool {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["YOOZ_DISABLE_EAGER_LOAD"], raw != "0" {
            return false
        }
        // XCTest injects these env vars when running tests via
        // `xcodebuild test`. Either is sufficient to identify a
        // test process; the loader stays off so route tests boot
        // fast and don't pay the model-fetch cost.
        if env["XCTestConfigurationFilePath"] != nil ||
            env["XCInjectBundleInto"] != nil {
            return false
        }
        return true
    }

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
    public static let kvCompression: KVCompressionMode = .off

    // MARK: - Telemetry

    /// Whether the user has opted into local STT telemetry. Default
    /// `false`. Driven by the `YOOZ_TELEMETRY_STT` env var:
    /// `"local"` opts in to the on-disk JSONL sink; any other value
    /// (including unset) opts out.
    ///
    /// No HTTP route surfaces this flag or the recorded metrics —
    /// telemetry consumption is local-file-only.
    public static var telemetryOptedIn: Bool {
        let raw = ProcessInfo.processInfo.environment["YOOZ_TELEMETRY_STT"]
        return raw == "local"
    }

    /// Directory the on-disk JSONL metrics file lives in. Defaults
    /// to `~/Library/Application Support/YoozEngine/telemetry/`.
    /// Tests redirect via `YOOZ_TELEMETRY_DIR`.
    public static var telemetryDirectory: URL {
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
    public static var sttMetricsFileURL: URL {
        telemetryDirectory.appendingPathComponent("stt_metrics.jsonl")
    }
}
