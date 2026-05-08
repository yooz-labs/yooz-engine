// ModuleEagerLoader.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log
#if canImport(GrammarModule)
import GrammarModule
#endif
#if canImport(LLMModule)
import LLMModule
#endif
#if canImport(STTModule)
import STTModule
#endif
#if canImport(VADModule)
import VADModule
#endif

private let logger = Logger(
    subsystem: "live.yooz.engine",
    category: "ModuleEagerLoader"
)

/// Per-module readiness state surfaced via `/v1/health` and
/// `/v1/modules`. The string `rawValue` is the wire format clients
/// branch on; do not rename without bumping the API contract.
public enum ModuleReadiness: String, Codable, Sendable {
    /// Module is not compiled into this build variant. Whisper +
    /// Lite expose this for VAD; Lite also exposes it for MLX STT.
    /// Clients should render this as a neutral tag, not a red dot —
    /// the absence is by design, not a failure.
    case unavailable
    /// Module is compiled in but has not been asked to load. Pre-
    /// kickoff state; should not be observable in production once the
    /// eager loader has run.
    case notLoaded = "not_loaded"
    /// Load task is in flight. Clients render a spinner.
    case loading
    /// Module is loaded and serving requests.
    case ready
    /// Load was attempted and threw. Detail message is in
    /// `ModuleDetail.detail`.
    case error
}

/// Identifier for a module the engine eager-loads. Stable wire names
/// — clients pull these out of the `/v1/health` `detail` map and
/// `/v1/modules` response.
public enum ModuleID: String, CaseIterable, Codable, Sendable {
    case stt
    case llm
    case touchup
    case grammar
    case vad
    case tts
}

/// Per-module readiness record. `state` is the public-facing flag;
/// `detail` carries an error message for `error` state, otherwise
/// nil.
public struct ModuleDetail: Codable, Sendable, Equatable {
    public let state: ModuleReadiness
    public let detail: String?

    public init(state: ModuleReadiness, detail: String? = nil) {
        self.state = state
        self.detail = detail
    }
}

/// Map from `ModuleID.rawValue` to the per-module record. Codable
/// dictionary so the wire shape is `{"stt": {"state": "ready"}, ...}`.
public typealias ModuleDetailMap = [String: ModuleDetail]

/// Variant-aware eager-loader. Fires once after `APIServer.start()`
/// returns and primes every module the active build variant compiles
/// in. Tracks per-module readiness so `/v1/health` (and the new
/// `/v1/modules`) can report `loading` -> `ready` / `error` instead
/// of the old `false` -> `true` jump that left the whisper Engine tab
/// red until first request.
///
/// Concurrency model: each module loads in its own child `Task` so
/// MLX STT, MLX LLM, and CoreML VAD warm up in parallel — they touch
/// disjoint executors and won't contend on first-load. Grammar is
/// already loaded by the time we run (`GrammarEngine.shared` does the
/// FFI work in `init`); we just record it.
public actor ModuleEagerLoader {

    // MARK: - Singleton

    public static let shared = ModuleEagerLoader()

    // MARK: - State

    private var states: ModuleDetailMap = [:]
    private var kickoffTask: Task<Void, Never>?
    private var hasKickedOff = false

    // MARK: - Init

    public init() {
        // Seed all modules with the pre-kickoff state. The `kickoff`
        // call replaces these with `unavailable` for the modules the
        // active variant excludes, then transitions in-variant
        // modules through `loading` -> `ready` / `error`.
        for module in ModuleID.allCases {
            states[module.rawValue] = ModuleDetail(state: .notLoaded)
        }
    }

    // MARK: - Public API

    /// Start the per-module eager loads for the given variant.
    /// Idempotent: calling twice on the same loader returns the same
    /// task. Returns immediately once the child tasks are spawned;
    /// `await waitForCompletion()` to block until they all finish.
    public func kickoff(variant: BuildVariant) {
        if hasKickedOff { return }
        hasKickedOff = true

        // Mark out-of-variant modules as `unavailable` up front so a
        // caller that polls `/v1/health` mid-kickoff doesn't see a
        // transient `notLoaded` state for a module that will never
        // load on this variant.
        applyVariantGating(variant: variant)

        kickoffTask = Task { [variant] in
            await self.runLoadGroup(variant: variant)
        }

        logger.info(
            "Eager-load kickoff: variant=\(variant.rawValue, privacy: .public)"
        )
    }

    /// Apply variant gating only — mark out-of-variant modules as
    /// `unavailable` without spawning the heavy load tasks. Used by
    /// tests and by the `eagerLoadOnLaunch=false` boot path so the
    /// snapshot reflects the variant policy even when the engine
    /// stays cold. Idempotent across `applyVariantGating` and
    /// `kickoff`: the first call wins; the second is a no-op.
    public func markVariantUnavailableModules(variant: BuildVariant) {
        if hasKickedOff { return }
        applyVariantGating(variant: variant)
    }

    /// Wait until the kickoff task completes. Used by tests; not
    /// called on the production boot path (the server is up and
    /// serving while loads run).
    public func waitForCompletion() async {
        await kickoffTask?.value
    }

    /// Snapshot the current per-module readiness map. Safe to call
    /// from any context (each call returns a fresh copy).
    public func snapshot() -> ModuleDetailMap {
        return states
    }

    /// Read a single module's readiness state. Used by the legacy
    /// bool-shape fields on `/v1/health` so we don't double-walk the
    /// map for each module field.
    public func readiness(for module: ModuleID) -> ModuleReadiness {
        states[module.rawValue]?.state ?? .notLoaded
    }

    /// Reset the loader to the pre-kickoff state. Used by
    /// `APIServer.stop()` so a subsequent `start()` re-runs the
    /// load policy against unloaded engines, and by tests for
    /// state isolation. Cancels any in-flight task but does NOT
    /// wait for it to finish — MLX model loads aren't cancellation-
    /// aware, so a running preload would block reset for tens of
    /// seconds. The cancelled task drops its writes into the
    /// already-cleared map silently (Swift actors serialize the
    /// writes; the task either runs before reset and is overwritten,
    /// or runs after and writes into a fresh map that's about to be
    /// overwritten by the next kickoff).
    public func reset() {
        kickoffTask?.cancel()
        kickoffTask = nil
        hasKickedOff = false
        states.removeAll()
        for module in ModuleID.allCases {
            states[module.rawValue] = ModuleDetail(state: .notLoaded)
        }
    }

    // MARK: - Internals

    private func applyVariantGating(variant: BuildVariant) {
        // STT — `unavailable` on .lite; otherwise leave the
        // pre-kickoff state alone (becomes `loading` once kickoff
        // spawns the loadSTT task, then `ready` / `error`).
        if !variant.includesMLXSTT {
            states[ModuleID.stt.rawValue] = ModuleDetail(
                state: .unavailable,
                detail: "MLX STT not compiled into the \(variant.rawValue) variant"
            )
        } else {
            // Reset to `notLoaded` so a second gating call (e.g.
            // after a Stop -> Start cycle that flipped variant via
            // env var, or a unit test that re-gates) doesn't leak
            // a stale `unavailable` from the previous variant.
            states[ModuleID.stt.rawValue] = ModuleDetail(state: .notLoaded)
        }
        // VAD — `unavailable` on .whisper / .lite. Same reset
        // pattern as STT.
        if !variant.includesVAD {
            states[ModuleID.vad.rawValue] = ModuleDetail(
                state: .unavailable,
                detail: variant == .whisper
                    ? "VAD is whisper-embedded (out-of-process latency unsuitable for 64ms windows)"
                    : "VAD not compiled into the \(variant.rawValue) variant"
            )
        } else {
            states[ModuleID.vad.rawValue] = ModuleDetail(state: .notLoaded)
        }
        // TTS isn't shipped on any variant yet (Phase 7). Always
        // `unavailable` regardless of variant.
        states[ModuleID.tts.rawValue] = ModuleDetail(
            state: .unavailable,
            detail: "TTS module not yet shipped"
        )
    }

    private func setState(
        _ module: ModuleID,
        _ state: ModuleReadiness,
        detail: String? = nil
    ) {
        states[module.rawValue] = ModuleDetail(state: state, detail: detail)
    }

    private func runLoadGroup(variant: BuildVariant) async {
        await withTaskGroup(of: Void.self) { group in
            // Grammar — already loaded on first reference to
            // `GrammarEngine.shared` (FFI in init). We just probe and
            // record.
            group.addTask { [weak self] in
                await self?.loadGrammar()
            }

            if variant.includesLLM {
                group.addTask { [weak self] in
                    await self?.loadLLM()
                }
            }

            if variant.includesMLXSTT {
                group.addTask { [weak self] in
                    await self?.loadSTT(
                        language: EngineConfig.defaultSTTLanguage
                    )
                }
            }

            if variant.includesVAD {
                group.addTask { [weak self] in
                    await self?.loadVAD()
                }
            }

            // No need to await individual results — the TaskGroup
            // joins them. Each child writes its own readiness state
            // through `setState`.
        }

        logger.info("Eager-load complete; states=\(self.summary(), privacy: .public)")
    }

    private func summary() -> String {
        states
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value.state.rawValue)" }
            .joined(separator: ",")
    }

    // MARK: - Per-module load paths

    private func loadGrammar() async {
        // GrammarEngine loads its rule counts in `init`. Touching the
        // singleton triggers it. `isAvailable` is `nonisolated` so we
        // can read it directly.
        if GrammarEngine.shared.isAvailable {
            setState(.grammar, .ready)
        } else {
            setState(
                .grammar,
                .error,
                detail: "Grammar FFI loaded but no rules present"
            )
        }
    }

    private func loadLLM() async {
        setState(.llm, .loading)
        setState(.touchup, .loading)
        do {
            // Preload the light model only — quality is loaded on
            // demand (it can be 1+ GB and not every TouchUp request
            // needs it). `preload` also tries to bring up Apple
            // Intelligence on macOS 26+.
            try await TouchUpEngine.shared.preload(loadQuality: false)
            setState(.llm, .ready)
            setState(.touchup, .ready)
            logger.info("LLM + TouchUp eager-load: ready")
        } catch {
            let msg = error.localizedDescription
            setState(.llm, .error, detail: msg)
            setState(.touchup, .error, detail: msg)
            logger.error("LLM eager-load failed: \(msg, privacy: .public)")
        }
    }

    private func loadSTT(language: STTLanguage) async {
        setState(.stt, .loading)
        do {
            try await YoozSTTEngine.shared.start(language: language)
            setState(.stt, .ready)
            logger.info(
                "STT eager-load: ready (lang=\(language.rawValue, privacy: .public))"
            )
        } catch {
            // Model assets may not be on disk in dev / CI; surface
            // the message rather than crashing. The lazy-load path
            // through `POST /v1/stt/load` still works for clients
            // that bring their own models.
            let msg = error.localizedDescription
            setState(.stt, .error, detail: msg)
            logger.error("STT eager-load failed: \(msg, privacy: .public)")
        }
    }

    private func loadVAD() async {
        #if canImport(VADModule)
        setState(.vad, .loading)
        do {
            try await VADEngine.shared.load()
            setState(.vad, .ready)
            logger.info("VAD eager-load: ready")
        } catch {
            let msg = error.localizedDescription
            setState(.vad, .error, detail: msg)
            logger.error("VAD eager-load failed: \(msg, privacy: .public)")
        }
        #else
        // Variant doesn't bundle VAD (e.g. YoozEngineWhisper). The
        // variant-gating pass already marked .vad as .unavailable;
        // skip the load attempt so we don't fault the readiness map.
        setState(.vad, .unavailable, detail: "VAD module not bundled")
        #endif
    }
}
