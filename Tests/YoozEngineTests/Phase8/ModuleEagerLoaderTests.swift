// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

import EngineCore
@testable import YoozEngine

/// Phase 8 — variant-aware module eager-load (issue #43). The loader
/// fires once after the API server starts and primes every module
/// the active build variant compiles in.
///
/// These tests exercise the policy surface that does NOT trigger
/// real MLX / CoreML loads — variant gating, idempotency, reset,
/// wire-format rawValues, the pre-kickoff snapshot. The "real load"
/// path is exercised by the live engine smoke test (see PR
/// description); we deliberately do not boot MLX in unit tests
/// because (a) model files are not on CI runners and (b) the MLX
/// load takes seconds and pulls in TouchUpEngine.shared singleton
/// state that bleeds across tests.
final class ModuleEagerLoaderTests: XCTestCase {

    // MARK: - Variant-gating

    func testFullVariantMarksTTSUnavailable() async throws {
        // `markVariantUnavailableModules` runs the cheap part of
        // kickoff (just sets unavailable rows) without spawning the
        // load TaskGroup. This is what `APIServer.start()` calls
        // when `eagerLoadOnLaunch` is off (XCTest path).
        let loader = ModuleEagerLoader()
        await loader.markVariantUnavailableModules(variant: .full)

        let snapshot = await loader.snapshot()

        // TTS is unavailable on every variant today (Phase 7 work).
        XCTAssertEqual(
            snapshot[ModuleID.tts.rawValue]?.state, .unavailable
        )
        XCTAssertNotNil(snapshot[ModuleID.tts.rawValue]?.detail)

        // Other modules stay in the pre-kickoff `notLoaded` state on
        // `.full` — none are out-of-variant.
        for id: ModuleID in [.stt, .llm, .touchup, .grammar, .vad] {
            XCTAssertEqual(
                snapshot[id.rawValue]?.state, .notLoaded,
                "\(id.rawValue) should remain `notLoaded` on .full pre-kickoff"
            )
        }
    }

    func testWhisperVariantMarksVADUnavailable() async throws {
        let loader = ModuleEagerLoader()
        await loader.markVariantUnavailableModules(variant: .whisper)

        let snapshot = await loader.snapshot()

        // VAD is whisper-embedded; the engine should NOT try to load
        // it. Detail message should explain why.
        XCTAssertEqual(
            snapshot[ModuleID.vad.rawValue]?.state, .unavailable,
            "VAD should be `unavailable` on the whisper variant"
        )
        let detail = snapshot[ModuleID.vad.rawValue]?.detail ?? ""
        XCTAssertTrue(
            detail.contains("whisper-embedded"),
            "VAD detail should mention whisper-embedded reason; got `\(detail)`"
        )

        // STT is still in-variant on .whisper — pre-kickoff it stays
        // `notLoaded`, not `unavailable`.
        XCTAssertEqual(
            snapshot[ModuleID.stt.rawValue]?.state, .notLoaded
        )
    }

    func testLiteVariantMarksMLXSTTAndVADUnavailable() async throws {
        let loader = ModuleEagerLoader()
        await loader.markVariantUnavailableModules(variant: .lite)

        let snapshot = await loader.snapshot()

        XCTAssertEqual(
            snapshot[ModuleID.stt.rawValue]?.state, .unavailable
        )
        XCTAssertEqual(
            snapshot[ModuleID.vad.rawValue]?.state, .unavailable
        )

        // LLM is still in-variant on .lite — pre-kickoff it stays
        // `notLoaded`.
        XCTAssertEqual(
            snapshot[ModuleID.llm.rawValue]?.state, .notLoaded
        )
    }

    // MARK: - Idempotency + reset

    func testMarkVariantUnavailableModulesReGates() async throws {
        // `markVariantUnavailableModules` is safe to call multiple
        // times before kickoff. Each call FULLY re-applies the
        // variant policy (in-variant rows reset to `.notLoaded`,
        // out-of-variant rows set to `.unavailable`). This matches
        // the APIServer Stop -> Start path where stop calls reset
        // and the next start re-gates.
        let loader = ModuleEagerLoader()
        await loader.markVariantUnavailableModules(variant: .lite)
        await loader.markVariantUnavailableModules(variant: .full)

        let snapshot = await loader.snapshot()
        // After the second call with .full, STT + VAD should be back
        // to `.notLoaded` (in-variant for .full); TTS stays
        // `.unavailable` (universal).
        XCTAssertEqual(
            snapshot[ModuleID.stt.rawValue]?.state, .notLoaded,
            "STT should be `notLoaded` after re-gating to .full"
        )
        XCTAssertEqual(
            snapshot[ModuleID.vad.rawValue]?.state, .notLoaded,
            "VAD should be `notLoaded` after re-gating to .full"
        )
        XCTAssertEqual(
            snapshot[ModuleID.tts.rawValue]?.state, .unavailable
        )
    }

    func testResetClearsState() async throws {
        let loader = ModuleEagerLoader()
        await loader.markVariantUnavailableModules(variant: .lite)
        await loader.reset()

        let snapshot = await loader.snapshot()
        // After reset, every module should be back to `.notLoaded`
        // — including the previously-`unavailable` ones.
        for module in ModuleID.allCases {
            XCTAssertEqual(
                snapshot[module.rawValue]?.state, .notLoaded,
                "\(module.rawValue) should be `notLoaded` after reset"
            )
        }
    }

    func testResetAllowsReGating() async throws {
        // After reset, a fresh `markVariantUnavailableModules` call
        // should take effect (the `hasKickedOff` short-circuit must
        // be cleared). Mirrors the APIServer Stop -> Start path.
        let loader = ModuleEagerLoader()
        await loader.markVariantUnavailableModules(variant: .lite)
        await loader.reset()
        await loader.markVariantUnavailableModules(variant: .whisper)

        let snapshot = await loader.snapshot()
        // Now we should see whisper's gating, not lite's.
        XCTAssertEqual(
            snapshot[ModuleID.stt.rawValue]?.state, .notLoaded,
            "STT should be `notLoaded` (in-variant for .whisper) after re-gating"
        )
        XCTAssertEqual(
            snapshot[ModuleID.vad.rawValue]?.state, .unavailable,
            "VAD should be `unavailable` for .whisper after re-gating"
        )
    }

    // MARK: - Variant flag table

    func testBuildVariantIncludesGates() {
        XCTAssertTrue(BuildVariant.full.includesMLXSTT)
        XCTAssertTrue(BuildVariant.whisper.includesMLXSTT)
        XCTAssertFalse(BuildVariant.lite.includesMLXSTT)

        XCTAssertTrue(BuildVariant.full.includesVAD)
        XCTAssertFalse(BuildVariant.whisper.includesVAD)
        XCTAssertFalse(BuildVariant.lite.includesVAD)

        // LLM + Grammar are universal today.
        for variant: BuildVariant in [.full, .whisper, .lite] {
            XCTAssertTrue(variant.includesLLM, "\(variant) should include LLM")
            XCTAssertTrue(
                variant.includesGrammar, "\(variant) should include Grammar"
            )
        }
    }

    // MARK: - Pre-kickoff state

    func testPreKickoffStateIsNotLoaded() async {
        let loader = ModuleEagerLoader()
        let snapshot = await loader.snapshot()
        for module in ModuleID.allCases {
            XCTAssertEqual(
                snapshot[module.rawValue]?.state, .notLoaded,
                "\(module.rawValue) should be `notLoaded` before any gating"
            )
        }
    }

    // MARK: - Wire format

    func testModuleReadinessRawValuesAreStable() {
        // Clients branch on these strings; renaming them is an API
        // break. Lock them in.
        XCTAssertEqual(ModuleReadiness.unavailable.rawValue, "unavailable")
        XCTAssertEqual(ModuleReadiness.notLoaded.rawValue, "not_loaded")
        XCTAssertEqual(ModuleReadiness.loading.rawValue, "loading")
        XCTAssertEqual(ModuleReadiness.ready.rawValue, "ready")
        XCTAssertEqual(ModuleReadiness.error.rawValue, "error")
    }

    func testModuleIDRawValuesAreStable() {
        let expected: [ModuleID: String] = [
            .stt: "stt",
            .llm: "llm",
            .touchup: "touchup",
            .grammar: "grammar",
            .vad: "vad",
            .tts: "tts"
        ]
        for (id, raw) in expected {
            XCTAssertEqual(id.rawValue, raw)
        }
    }

    // MARK: - Engine variant detection

    func testEngineConfigVariantIsKnownVariant() {
        // The compile-time gate must resolve to one of the three
        // variants. Default (no flag) is `.full`.
        let variant = EngineConfig.variant
        XCTAssertTrue(
            [.full, .whisper, .lite].contains(variant),
            "EngineConfig.variant should be a known BuildVariant; got \(variant)"
        )
    }

    func testEagerLoadOnLaunchIsOffInTests() {
        // The XCTest detection should keep the loader off during
        // route tests. Without this, route tests would spin up MLX
        // model loads on every test boot.
        XCTAssertFalse(
            EngineConfig.eagerLoadOnLaunch,
            "eagerLoadOnLaunch must be false in XCTest runs"
        )
    }
}
