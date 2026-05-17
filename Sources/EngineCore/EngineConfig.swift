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
    public static let defaultPort: Int = 19920
    public static let portEnvVar = "YOOZ_ENGINE_PORT"

    public static var port: Int {
        // Use `getenv()` rather than `ProcessInfo.processInfo.environment`
        // so this accessor reads the libc env table directly. The
        // YoozEngineTests suite mutates `YOOZ_ENGINE_PORT` via `setenv()`
        // between cases (see `UniqueEnginePort`, yooz-engine#122) so each
        // `APIServer` boot binds its own loopback port. `getenv()` is the
        // matching read primitive for `setenv()` — same libc table, no
        // intermediate `[String: String]` rebuild, no risk that a future
        // Foundation cache layer (Darwin's `ProcessInfo.environment` has
        // historically been live, but swift-corelibs-foundation snapshots)
        // would silently freeze the value. Production callers see no
        // behavior change: `YOOZ_ENGINE_PORT` is set once before launch
        // via `NSWorkspace.OpenConfiguration.environment` and never
        // mutated thereafter.
        guard let rawPointer = getenv(portEnvVar) else {
            return defaultPort
        }
        let raw = String(cString: rawPointer)
        guard let parsed = Int(raw), (1...65535).contains(parsed) else {
            return defaultPort
        }
        return parsed
    }

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
    /// `EngineHelperController`) pass the variable through
    /// `NSWorkspace.OpenConfiguration.environment` when spawning the helper.
    ///
    /// Note: `NSWorkspace.OpenConfiguration.environment` is NOT reliably
    /// propagated to nested helper bundles on macOS 26 (LaunchServices
    /// quirk verified in yooz-whisper#179, this repo #117). The env-var
    /// channel is kept for backward compat with direct shell exec, scripts,
    /// and test harnesses that set the environment. Host apps launching
    /// the helper via `NSWorkspace.openApplication` must additionally pass
    /// `--headless` through `OpenConfiguration.arguments`, which
    /// LaunchServices DOES preserve. See `helperModeArg`.
    public static let headlessEnvVar = "YOOZ_ENGINE_HEADLESS"

    /// Command-line argument that flips the engine into helper mode. This
    /// is the reliable channel for `NSWorkspace.openApplication` launches:
    /// `OpenConfiguration.arguments` IS propagated by LaunchServices, while
    /// `OpenConfiguration.environment` is not (#117). Host apps should pass
    /// both `--headless` (argv) and `YOOZ_ENGINE_HEADLESS=1` (env) so the
    /// helper enters headless mode regardless of which channel survives.
    public static let helperModeArg = "--headless"

    /// `true` when the engine process should run as a background helper
    /// (no menu-bar icon, no Settings scene). Helper mode is signaled
    /// from two channels:
    ///
    /// 1. Env var `YOOZ_ENGINE_HEADLESS=1` (preserved for direct shell
    ///    exec / scripts / test harnesses that set the environment).
    /// 2. Command-line argument `--headless` (the reliable channel when
    ///    the process is spawned via `NSWorkspace.openApplication`).
    ///
    /// Either source alone is sufficient. Used to gate menu-bar/Settings
    /// UI, modal alerts, and any other host-app-inappropriate behaviour.
    public static var isHelper: Bool {
        isHelperMode(
            environment: ProcessInfo.processInfo.environment,
            arguments: CommandLine.arguments
        )
    }

    /// Pure predicate that decides helper mode from a given environment
    /// and argument vector. Extracted so unit tests can drive both
    /// channels without mutating process-global state (`CommandLine.arguments`
    /// is read-only at the language level). The runtime accessor
    /// `isHelper` simply binds this to the live process. Public so
    /// callers that already have an environment dictionary in hand (e.g.
    /// future spawn helpers, integration tests) can reuse the contract.
    public static func isHelperMode(
        environment: [String: String],
        arguments: [String]
    ) -> Bool {
        if environment[headlessEnvVar] == "1" {
            return true
        }
        if arguments.contains(helperModeArg) {
            return true
        }
        return false
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
