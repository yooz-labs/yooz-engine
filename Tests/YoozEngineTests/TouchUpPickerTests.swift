// TouchUpPickerTests.swift
// YoozEngineTests
//
// Copyright 2026 Yooz Labs. All rights reserved.
//
// Pins the TouchUp model picker contract introduced by issue #97.
// The wire ids and picker shape are the canonical pattern for every
// future module picker (STT engine, TTS voice, ...) per AGENTS.md;
// drift here cascades into every consumer app.

import XCTest
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

    /// MLX selections must round-trip cleanly to the underlying
    /// `LLMModelType`; FoundationModels intentionally has no MLX
    /// counterpart. A future MLX-side rename of `LLMModelType` cases
    /// would break this test before the picker ships a broken id.
    func testMLXMappingMatchesLLMModelType() {
        XCTAssertEqual(TouchUpModelSelection.yoozLight.mlxModelType, .yoozLight)
        XCTAssertEqual(TouchUpModelSelection.yoozQuality.mlxModelType, .yoozQuality)
        XCTAssertNil(TouchUpModelSelection.foundationModels.mlxModelType)
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
    /// MLX runtime. The light tier always reports `isAvailable`, so
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
            tier: "light",
            sizeBytes: 276 * 1024 * 1024,
            isAvailable: true,
            isCached: false,
            isLoaded: false,
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
        for key in ["id", "displayName", "description", "tier", "sizeBytes",
                    "isAvailable", "isCached", "isLoaded", "isActive"] {
            XCTAssertNotNil(json[key], "Wire shape missing key '\(key)'")
        }
    }
}
