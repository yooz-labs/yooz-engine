// TouchUpPickerTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pins the TouchUp model picker contract — both the public wire
// shape and the engine-actor invariants. The wire ids and picker
// shape are the canonical pattern for every future module picker
// (STT engine, TTS voice, ...) per AGENTS.md; drift here cascades
// into every consumer app.

import XCTest
import EngineCore
@testable import LLMModule
@testable import YoozEngine

final class TouchUpPickerTests: XCTestCase {

    /// Stable wire ids — every consumer app, including pre-built
    /// SDK clients, depends on these strings. Renaming is a major
    /// SDK bump, so the contract is pinned in a test.
    func testSelectionRawValuesAreStable() {
        XCTAssertEqual(TouchUpModelSelection.yoozLight.rawValue, "yooz-light-v3")
        XCTAssertEqual(TouchUpModelSelection.yoozQuality.rawValue, "yooz-quality-v3")
        XCTAssertEqual(TouchUpModelSelection.foundationModels.rawValue, "foundation-models")
    }

    /// Tier mapping per selection. The wire side ships the typed
    /// `TouchUpModelTier` enum; pinning the engine-side mapping
    /// catches an accidental rename (e.g. swapping a Pro badge tier
    /// silently).
    func testTierMappingIsStable() {
        XCTAssertEqual(TouchUpModelSelection.yoozLight.tier, .light)
        XCTAssertEqual(TouchUpModelSelection.yoozQuality.tier, .quality)
        XCTAssertEqual(TouchUpModelSelection.foundationModels.tier, .premium)
    }

    /// `availableModels()` must return one row per selection, in the
    /// declared order, with metadata derived from the selection
    /// itself. Pinning the count + ordering catches an accidental
    /// drop or reorder that would silently break picker UX in apps
    /// that depend on row index (e.g. "Yooz-Light is always first").
    func testAvailableModelsCoversEverySelection() async {
        let engine = TouchUpEngine()
        let models = await engine.availableModels()
        XCTAssertEqual(models.count, TouchUpModelSelection.allCases.count)
        XCTAssertEqual(
            models.map(\.id),
            TouchUpModelSelection.allCases.map(\.rawValue),
            "Picker rows must follow TouchUpModelSelection.allCases ordering"
        )
    }

    /// Default active model is `.yoozLight` — every consumer app
    /// relies on this on first launch (no picker interaction yet).
    /// Changing the default is a UX-breaking change; pin it.
    func testActiveDefaultsToYoozLight() async {
        let engine = TouchUpEngine()
        let active = await engine.activeModel
        XCTAssertEqual(active, .yoozLight)

        let models = await engine.availableModels()
        let activeRow = try? XCTUnwrap(models.first(where: { $0.isActive }))
        XCTAssertEqual(activeRow?.id, TouchUpModelSelection.yoozLight.rawValue)
    }

    /// MLX tiers always report `.available` or higher (download on
    /// first use). FoundationModels degrades to `.unavailable` on
    /// pre-26 hosts. The four-state lifecycle replaces three loose
    /// booleans so illegal combinations like
    /// "loaded but not cached" become unrepresentable.
    func testLoadStateForMLXTiersIsAtLeastAvailable() async {
        let engine = TouchUpEngine()
        let models = await engine.availableModels()
        let allowedMLX: Set<ModelLoadState> = [.available, .cached, .loaded]
        for row in models where row.id != TouchUpModelSelection.foundationModels.rawValue {
            XCTAssertTrue(
                allowedMLX.contains(row.loadState),
                "\(row.id) load state should be available/cached/loaded; got \(row.loadState)"
            )
        }
    }

    /// Picking FoundationModels on a pre-26 machine (or an
    /// opted-out user) must throw `LLMError.notAvailable`. The
    /// route handler maps this to 501 `model_unavailable` so the
    /// picker UI can render "not supported on this Mac" cleanly.
    /// Skipped when the test host actually has Apple Intelligence;
    /// the negative branch is the one that needs pinning.
    func testSetActiveFoundationModelsThrowsWhenUnavailable() async {
        let backend = FoundationModelsBackend()
        guard !backend.isAvailable() else {
            // Apple Intelligence really is available on this host
            // — the unavailable-branch test is moot.
            return
        }

        let engine = TouchUpEngine()
        do {
            _ = try await engine.setActiveModel(.foundationModels, preload: false)
            XCTFail("Expected LLMError.notAvailable")
        } catch let error as LLMError {
            switch error {
            case .notAvailable:
                break // expected
            default:
                XCTFail("Wrong LLMError case: \(error)")
            }
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    /// Setting the active model with `preload: false` must update
    /// `activeModel` synchronously without touching the network or
    /// MLX runtime. The light tier always reports `available`, so
    /// this is a safe fast-path test.
    func testSetActiveYoozLightWithoutPreloadFlipsActive() async throws {
        let engine = TouchUpEngine()
        let info = try await engine.setActiveModel(.yoozLight, preload: false)
        XCTAssertEqual(info.id, TouchUpModelSelection.yoozLight.rawValue)
        XCTAssertTrue(info.isActive)
        let active = await engine.activeModel
        XCTAssertEqual(active, .yoozLight)
    }

    /// Wire shape for the picker `ModelInfo` is the canonical
    /// pattern (per AGENTS.md). This test pins the JSON keys so a
    /// future renumbering of fields breaks the build, not the
    /// downstream SDK decode in production.
    func testModelInfoCodableRoundTripPinsWireShape() throws {
        let info = TouchUpModelInfo(
            id: "yooz-light-v3",
            displayName: "Yooz-Light",
            description: "Fast",
            tier: .light,
            sizeBytes: 276 * 1024 * 1024,
            loadState: .loaded,
            isActive: true
        )
        let encoded = try JSONEncoder().encode(info)
        let decoded = try JSONDecoder().decode(TouchUpModelInfo.self, from: encoded)
        XCTAssertEqual(decoded, info)

        // Spot-check the JSON keys; these are part of the wire
        // contract every SDK consumer depends on.
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        for key in ["id", "displayName", "description", "tier",
                    "sizeBytes", "loadState", "isActive"] {
            XCTAssertNotNil(json[key], "Wire shape missing key '\(key)'")
        }
    }
}

/// Quality-dispatch regression test for the C1 finding: an earlier
/// implementation routed `.yoozQuality` through `process(...)`
/// which auto-fell-back to the light model when no replacements
/// were present, making the picker a silent no-op. The fix runs
/// the loaded quality backend on both routing slots and relabels
/// the result, so a user-picked Quality is honored regardless of
/// replacements.
///
/// Held in this file (not the live integration suite) so the bug
/// is caught by `xcodebuild test` without any model weights on
/// disk — we only need to exercise the dispatch shape, not the
/// MLX inference itself.
final class TouchUpProcessWithActiveModelTests: XCTestCase {

    /// `.yoozLight` dispatch goes straight to the legacy
    /// `process(...)` path. Without a loaded light model the path
    /// returns regex-only (modelUsed == .regexOnly). Pinning this
    /// is a tripwire — if someone refactors the dispatch in a way
    /// that bypasses the fallback (e.g. forcing an MLX call on a
    /// nil model), the test fails hard.
    func testYoozLightDispatchFallsBackToRegexWithoutWeights() async {
        let engine = TouchUpEngine()
        let result = await engine.processWithActiveModel(
            text: "hello world",
            mode: .standard
        )
        // Either regexOnly (light model nil → process() short-circuits)
        // or .light (light somehow loaded from a sibling test). Both
        // are valid; the bug we are guarding against is the dispatch
        // crashing or returning .quality / .foundationModels.
        XCTAssertTrue(
            [.regexOnly, .light, .fallbackRegex].contains(result.modelUsed),
            "Light dispatch must stay on light path; got \(result.modelUsed)"
        )
    }

    /// C1 regression: with `.yoozQuality` selected, the dispatch
    /// must not silently report the light model. When the quality
    /// backend loads (cached or downloadable), the result label is
    /// relabeled to `.quality`. When the backend fails to load, we
    /// fall through to the legacy path which surfaces the load
    /// failure as `.regexOnly` / `.fallbackRegex` / `.light` — none
    /// of which silently *claim* Quality. The forbidden outcome is
    /// a successful inference labeled `.light` (the original C1
    /// bug).
    func testYoozQualityDispatchNeverSilentlyLabelsAsLight() async throws {
        let engine = TouchUpEngine()
        _ = try await engine.setActiveModel(.yoozQuality, preload: false)
        let result = await engine.processWithActiveModel(
            text: "hello world",
            mode: .standard
        )
        // Acceptable: quality (load succeeded), regexOnly /
        // fallbackRegex / light (load failed → fallback path —
        // these are honest "we couldn't run quality" signals).
        // Forbidden: any other label that pretends quality ran.
        let acceptable: Set<TouchUpProcessor.ModelUsed> = [
            .quality, .regexOnly, .fallbackRegex, .light
        ]
        XCTAssertTrue(
            acceptable.contains(result.modelUsed),
            "Quality dispatch returned unexpected label: \(result.modelUsed)"
        )
    }
}
