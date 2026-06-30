// EngineConfig.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

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
    public static let version: String = "0.7.5"

    /// Budget for MLX's Metal buffer cache **per resident model category**. The
    /// process-global `MLX.Memory.cacheLimit` is set to this value times the
    /// number of resident categories, by `MLXResidency` at the MLX model-load /
    /// teardown paths (`MLXLLMBackend`, `YoozSTTEngine`, the Qwen3 backend).
    ///
    /// mlx-swift's default `cacheLimit` equals its memory limit (~1.5x the
    /// device's recommended working set), which scales with installed RAM and
    /// lets the cache of freed-but-retained buffers grow into the tens of GB.
    /// The loopback packaging hid this by running the engine in a separate,
    /// kill-able helper process; in-process that growth is charged to the
    /// consumer app's own RSS for the app's lifetime (the observed multi-tens-
    /// of-GB runaway). Capping the cache keeps steady-state RAM at roughly
    /// (resident model weights + this cache).
    ///
    /// 512 MB is ample scratch for **one** model category (the STT encoder, or
    /// the LLM KV). It was the original whole-process cap, which starved a
    /// coexisting second MLX model — Parakeet STT + the Quality LLM share one
    /// process, and a single 512 MB pool made them evict each other's buffers
    /// every cycle. Treating it as a *per-category* budget that `MLXResidency`
    /// sums across resident categories (e.g. STT + LLM -> 1 GB) keeps each
    /// category's scratch warm while RAM stays bounded and scales with how many
    /// models actually coexist. See `MLXResidency`.
    ///
    /// Applied at the load paths (not at process start) on purpose: setting it
    /// touches the Metal allocator, which needs `default.metallib` — present in
    /// the real app and under xcodebuild, absent under a plain `swift test`
    /// run. The load paths only execute when a model is actually being loaded,
    /// so the cap lands before the cache-growing inference begins and never
    /// fires in the non-GPU structural tests.
    public static let mlxCacheBudgetPerCategoryBytes: Int = 512 * 1024 * 1024

    /// Upper bound, in seconds, for a single blocking model-load await on the
    /// in-process transport. The in-process path has no HTTP-client timeout to
    /// fall back on, so a blocking caller (`loadModel(wait:true)`, the TouchUp
    /// model switch, a stream open) wraps its load `Task` in this deadline via
    /// `awaitLoadTask(_:deadlineSeconds:)`. Generous on purpose — a legitimate
    /// first-run multi-GB Hugging Face pull on a slow link can run many minutes;
    /// the deadline only exists so a genuinely wedged load surfaces an error
    /// instead of blocking the caller forever. Matches the consumer-side
    /// `awaitModelReady` poll ceiling in yooz-whisper.
    public static let modelLoadDeadlineSeconds: Double = 600

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

    // MARK: - STT streaming cadence

    /// Default minimum interval (seconds) between successive streaming
    /// partial emissions from `StreamingTranscriber`. Host apps that
    /// don't override the value in `StreamingTranscriber.init(...)`
    /// inherit this. `2.0` gives users a partial roughly every two
    /// seconds during continuous speech — substantially more
    /// responsive than the prior "process every WS frame" behaviour
    /// (effective ~5 s when the upstream chunks are large) without
    /// burning encode cycles on every 64 ms audio buffer.
    ///
    /// Override at runtime via `YOOZ_STT_PARTIAL_INTERVAL_SEC`. Values
    /// below `0.1` are clamped up (preventing accidental no-throttle
    /// configs that would re-encode on every frame); `0` is honored
    /// as an explicit "disable throttle" opt-out. Non-numeric values
    /// fall back to the compiled default.
    public static let defaultStreamingPartialIntervalSec: Float = 2.0

    /// Resolved STT partial-emission cadence in seconds. Reads
    /// `YOOZ_STT_PARTIAL_INTERVAL_SEC` via `getenv` (matching the rest
    /// of this enum's env-var pattern). Returns
    /// `defaultStreamingPartialIntervalSec` on unset, empty, or
    /// non-numeric values. `0` is honored as a literal "disable" opt-out;
    /// positive values below `0.1` are clamped to `0.1` so a typo can't
    /// accidentally turn the throttle off.
    public static var streamingPartialIntervalSec: Float {
        guard let rawPointer = getenv("YOOZ_STT_PARTIAL_INTERVAL_SEC") else {
            return defaultStreamingPartialIntervalSec
        }
        let raw = String(cString: rawPointer)
        guard let parsed = Float(raw), parsed >= 0 else {
            return defaultStreamingPartialIntervalSec
        }
        if parsed == 0 {
            return 0
        }
        return max(0.1, parsed)
    }

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
