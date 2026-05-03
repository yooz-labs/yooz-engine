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

/// Build variant the engine is compiled for. Drives variant-aware
/// eager-loading (`ModuleEagerLoader`) and per-module compile-time
/// gating. The active variant is `EngineConfig.variant`, resolved
/// once from compile-time flags so the runtime read is free.
///
/// The Xcode targets that flip these flags (`YoozEngineWhisper`,
/// `YoozEngineLite`) are added in Phase 5 hardening; today only
/// `.full` ships, but the gate exists so each new target only needs
/// `OTHER_SWIFT_FLAGS=-DYOOZ_ENGINE_WHISPER` (or `_LITE`) and
/// inherits all module gating for free.
public enum EngineVariant: String, Codable, Sendable {
    /// Full engine — STT (MLX) + Apple STT + Grammar + LLM + VAD.
    /// Standalone menu-bar service. Default when no flag is set.
    case full
    /// Whisper-helper variant — STT (MLX) + Apple STT + Grammar + LLM.
    /// VAD is whisper-embedded (not bundled here) because its
    /// ~64ms call rate makes an HTTP round-trip non-viable.
    case whisper
    /// Lite variant — Apple STT + Grammar + LLM. No MLX STT, no VAD.
    /// Targets Remi-class apps and iOS hosts where the sub-GB binary
    /// matters more than throughput.
    case lite

    /// Whether MLX-based STT (Parakeet / FastConformer / Qwen3) is
    /// compiled into this variant. Lite drops it entirely and relies
    /// on Apple STT for transcription.
    public var includesMLXSTT: Bool {
        switch self {
        case .full, .whisper: return true
        case .lite: return false
        }
    }

    /// Whether the CoreML VAD model (`silero-vad-unified-v6.0.0`) is
    /// bundled into this variant. Whisper hosts its own embedded VAD
    /// (out-of-process latency is too high for ~64ms windows); lite
    /// has no need for VAD on its hot path.
    public var includesVAD: Bool {
        switch self {
        case .full: return true
        case .whisper, .lite: return false
        }
    }

    /// Whether the LLM stack (MLX-Swift backends + Apple Intelligence
    /// when available) is compiled in. All three variants ship LLM —
    /// it is the engine's primary value-add over native OS APIs.
    public var includesLLM: Bool { true }

    /// Grammar (`YoozTextCleanup` xcframework) is always linked. The
    /// Rust FFI loads on first reference; engine-side cost is ~0.
    public var includesGrammar: Bool { true }
}

enum EngineConfig {
    static let port: Int = 19920
    static let host: String = "127.0.0.1"
    static let version: String = "0.6.0"

    // MARK: - Helper-mode contract (issue #42)

    /// Env-var name host apps (yooz-whisper, yooz-notes, …) set on the
    /// helper subprocess to request headless behaviour: no MenuBarExtra,
    /// no Dock entry, `.prohibited` activation policy. Pinned by
    /// `EngineConfigHelperModeTests` because renaming it silently
    /// breaks every host's launch code (and brings the brain icon back).
    static let headlessEnvVar: String = "YOOZ_ENGINE_HEADLESS"

    /// `true` iff `YOOZ_ENGINE_HEADLESS` is set to literal `"1"`.
    /// Strict equality — `"0"`, `"true"`, empty, and unset all mean
    /// standalone mode. Read lazily on every call so tests can drive
    /// the predicate via `setenv` / `unsetenv` without spawning a
    /// subprocess.
    static var isHelper: Bool {
        ProcessInfo.processInfo.environment[headlessEnvVar] == "1"
    }

    /// Active build variant. Resolved from compile-time flags so the
    /// runtime read is a constant load. Override with
    /// `OTHER_SWIFT_FLAGS=-DYOOZ_ENGINE_WHISPER` or
    /// `-DYOOZ_ENGINE_LITE` per Xcode target.
    static let variant: EngineVariant = {
        #if YOOZ_ENGINE_LITE
        return .lite
        #elseif YOOZ_ENGINE_WHISPER
        return .whisper
        #else
        return .full
        #endif
    }()

    /// Default STT language to eager-load on the full / whisper
    /// variants. Driven by `YOOZ_DEFAULT_STT_LANG` so a user with a
    /// different primary language pays the eager-load cost on the
    /// right model. Falls back to English on unknown values.
    static var defaultSTTLanguage: STTLanguage {
        guard
            let raw = ProcessInfo.processInfo.environment[
                "YOOZ_DEFAULT_STT_LANG"
            ],
            let parsed = STTLanguage.fromCode(raw),
            parsed.isImplemented
        else {
            return .english
        }
        return parsed
    }

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
    static var eagerLoadOnLaunch: Bool {
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

    // MARK: - Telemetry

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
