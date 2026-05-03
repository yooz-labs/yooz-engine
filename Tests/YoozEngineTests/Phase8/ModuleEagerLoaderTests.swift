// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import XCTest

@testable import YoozEngine

/// Phase 8 — variant-aware module eager-load (issue #43). The loader
/// fires once after the API server starts and primes every module
/// the active build variant compiles in. These tests exercise the
/// variant-gating policy plus the readiness-state transitions; the
/// tests deliberately do NOT assert that MLX models actually load on
/// CI (model files may not be on disk), only that the loader records
/// a terminal state (`ready` or `error`) for the modules it should
/// have attempted, and `unavailable` for the modules it should have
/// skipped.
final class ModuleEagerLoaderTests: XCTestCase {

    // MARK: - Variant-gating

    func testFullVariantAttemptsAllInVariantModules() async throws {
        let loader = ModuleEagerLoader()
        await loader.kickoff(variant: .full)
        await loader.waitForCompletion()

        let snapshot = await loader.snapshot()

        // Grammar should always reach a terminal state on .full
        // (Rust FFI loaded — `ready`, or zero rules — `error`).
        let grammar = snapshot[ModuleID.grammar.rawValue]?.state
        XCTAssertTrue(
            grammar == .ready || grammar == .error,
            "Grammar should reach a terminal state on .full; got \(String(describing: grammar))"
        )

        // STT, LLM, TouchUp, VAD all attempted on .full. Either
        // `.ready` (model files present) or `.error` (model files
        // missing — expected on CI). They must NOT remain in
        // `.notLoaded` or `.loading`.
        for id: ModuleID in [.stt, .llm, .touchup, .vad] {
            let state = snapshot[id.rawValue]?.state
            XCTAssertTrue(
                state == .ready || state == .error,
                "\(id.rawValue) should reach a terminal state on .full; got \(String(describing: state))"
            )
        }

        // TTS is unavailable on every variant today (Phase 7 work).
        XCTAssertEqual(
            snapshot[ModuleID.tts.rawValue]?.state, .unavailable,
            "TTS should be `unavailable` on all variants until Phase 7 ships"
        )
    }

    func testWhisperVariantSkipsVAD() async throws {
        let loader = ModuleEagerLoader()
        await loader.kickoff(variant: .whisper)
        await loader.waitForCompletion()

        let snapshot = await loader.snapshot()

        // VAD is whisper-embedded; the engine should NOT try to load
        // it. Detail message should explain why.
        XCTAssertEqual(
            snapshot[ModuleID.vad.rawValue]?.state, .unavailable,
            "VAD should be `unavailable` on the whisper variant"
        )
        XCTAssertNotNil(
            snapshot[ModuleID.vad.rawValue]?.detail,
            "VAD `unavailable` state should carry an explanatory detail string"
        )

        // STT is still in-variant on whisper — it should reach a
        // terminal state.
        let stt = snapshot[ModuleID.stt.rawValue]?.state
        XCTAssertTrue(
            stt == .ready || stt == .error,
            "STT should reach a terminal state on .whisper; got \(String(describing: stt))"
        )
    }

    func testLiteVariantSkipsMLXSTTAndVAD() async throws {
        let loader = ModuleEagerLoader()
        await loader.kickoff(variant: .lite)
        await loader.waitForCompletion()

        let snapshot = await loader.snapshot()

        XCTAssertEqual(
            snapshot[ModuleID.stt.rawValue]?.state, .unavailable,
            "STT should be `unavailable` on the lite variant"
        )
        XCTAssertEqual(
            snapshot[ModuleID.vad.rawValue]?.state, .unavailable,
            "VAD should be `unavailable` on the lite variant"
        )

        // LLM is still in-variant — should reach a terminal state.
        let llm = snapshot[ModuleID.llm.rawValue]?.state
        XCTAssertTrue(
            llm == .ready || llm == .error,
            "LLM should reach a terminal state on .lite; got \(String(describing: llm))"
        )
    }

    // MARK: - Idempotency + reset

    func testKickoffIsIdempotent() async throws {
        let loader = ModuleEagerLoader()
        await loader.kickoff(variant: .lite)
        // Second call should be a no-op (no second TaskGroup spawned)
        await loader.kickoff(variant: .lite)
        await loader.waitForCompletion()

        let snapshot = await loader.snapshot()
        // Loader still produced a coherent snapshot (didn't lose state
        // because of the duplicate kickoff).
        XCTAssertEqual(
            snapshot[ModuleID.stt.rawValue]?.state, .unavailable
        )
        XCTAssertEqual(
            snapshot[ModuleID.vad.rawValue]?.state, .unavailable
        )
    }

    func testResetClearsState() async throws {
        let loader = ModuleEagerLoader()
        await loader.kickoff(variant: .lite)
        await loader.waitForCompletion()
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

    // MARK: - Variant flags

    func testEngineVariantIncludesGates() {
        XCTAssertTrue(EngineVariant.full.includesMLXSTT)
        XCTAssertTrue(EngineVariant.whisper.includesMLXSTT)
        XCTAssertFalse(EngineVariant.lite.includesMLXSTT)

        XCTAssertTrue(EngineVariant.full.includesVAD)
        XCTAssertFalse(EngineVariant.whisper.includesVAD)
        XCTAssertFalse(EngineVariant.lite.includesVAD)

        // LLM + Grammar are universal today.
        for variant: EngineVariant in [.full, .whisper, .lite] {
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
                "\(module.rawValue) should be `notLoaded` before kickoff"
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
}
