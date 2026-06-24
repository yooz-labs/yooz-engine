import EngineCore
import MLX
import XCTest
import YoozEngineClient
@testable import YoozEngineInProcess

/// The in-process residency invariant (epic #192): in-process RAM tracks the
/// resident model set, not an unbounded runaway. Two mechanisms:
///   - Tier 1: MLX's Metal buffer cache is capped (`EngineConfig.mlxCacheLimitBytes`).
///   - Tier 2: switching a TouchUp tier evicts the previous one (single-resident).
///
/// Background: mlx-swift's default `cacheLimit` equals its memory limit (~1.5x
/// the device working set, scaling with RAM), so freed inference buffers
/// accumulate without bound. The loopback packaging hid this in a separate,
/// kill-able helper process; in-process it is charged to the consumer app's own
/// RSS for the app's lifetime (the 46 GB regression).
///
/// Gating: tests that touch the MLX allocator / load weights need
/// `default.metallib`, present only under the app-hosted xctest (xcodebuild),
/// not a plain `swift test` run. STT model paths gate on `YOOZ_STT_LOAD_MODELS`;
/// LLM/TouchUp paths gate on `YOOZ_LLM_LOAD_MODELS` — mirroring
/// `InProcessLiveModelTests`.
final class InProcessMemoryTests: XCTestCase {
    private var sttEnabled: Bool {
        ProcessInfo.processInfo.environment["YOOZ_STT_LOAD_MODELS"] == "1"
    }

    private var llmEnabled: Bool {
        ProcessInfo.processInfo.environment["YOOZ_LLM_LOAD_MODELS"] == "1"
    }

    /// Guards intent, not the literal value: the cap must stay conservative so
    /// steady-state RAM is (resident weights + a small scratch cache). Asserting
    /// a range rather than `== 512 MB` lets the cap be tuned for a future device
    /// tier without a mechanical test edit, while still failing if someone drops
    /// it so low models churn or raises it past the in-process RSS budget.
    func testCacheLimitConstantIsConservativelyBounded() {
        XCTAssertGreaterThanOrEqual(
            EngineConfig.mlxCacheLimitBytes, 256 * 1024 * 1024,
            "cache cap must not drop below 256 MB (would churn re-allocation)"
        )
        XCTAssertLessThanOrEqual(
            EngineConfig.mlxCacheLimitBytes, 1024 * 1024 * 1024,
            "cache cap must not exceed 1 GB (violates the in-process RSS budget)"
        )
    }

    /// Tier 1, live: a real model load applies the cap. Gated behind
    /// `YOOZ_STT_LOAD_MODELS=1` (downloads Parakeet) and only meaningful under
    /// the app-hosted xctest where `default.metallib` is present.
    func testLoadingAModelBoundsTheMLXBufferCache() async throws {
        try XCTSkipUnless(
            sttEnabled,
            "Set YOOZ_STT_LOAD_MODELS=1 (app-hosted xctest) to exercise the "
                + "in-process load path that applies the MLX cache cap."
        )
        let client = YoozEngineClient.inProcess()
        try await client.connect()

        // Drives SDK -> InProcessTransport -> YoozSTTEngine load, which sets
        // `Memory.cacheLimit` before allocating weights.
        let loaded = try await client.stt.loadModel(language: .english)
        XCTAssertTrue(loaded.loaded, "loadModel must return a loaded status in-process")

        XCTAssertEqual(
            Memory.cacheLimit,
            EngineConfig.mlxCacheLimitBytes,
            "model load must bound the MLX buffer cache to the engine cap"
        )
    }

    /// Tier 2, live: switching the TouchUp tier evicts the previous one, so only
    /// one MLX tier is resident. Loads both Light and Quality, so gated behind
    /// `YOOZ_LLM_LOAD_MODELS=1` (app-hosted xctest, ~2 GB of weights). Observes
    /// residency through the public picker `loadState`, no `@testable` needed.
    func testSwitchingTouchUpTierEvictsThePrevious() async throws {
        try XCTSkipUnless(
            llmEnabled,
            "Set YOOZ_LLM_LOAD_MODELS=1 (app-hosted xctest) to load both LLM tiers."
        )
        let client = YoozEngineClient.inProcess()
        try await client.connect()

        // Load Light, then switch to Quality with preload. Single-resident
        // eviction must unload Light so Quality is the only resident tier.
        _ = try await client.touchUp.setModel(id: "yooz-light-v2", preload: true)
        _ = try await client.touchUp.setModel(id: "yooz-quality-v2", preload: true)

        // Explicit type: TouchUpClient overloads `availableModels()` by return
        // type (TouchUpModelsResponse vs LLMModelsResponse).
        let picker: TouchUpModelsResponse = try await client.touchUp.availableModels()
        XCTAssertEqual(picker.activeId, "yooz-quality-v2", "Quality must be active")

        let light = picker.models.first { $0.id == "yooz-light-v2" }
        let quality = picker.models.first { $0.id == "yooz-quality-v2" }
        XCTAssertNotEqual(
            light?.loadState, .loaded,
            "Light must be evicted (not resident) after switching to Quality"
        )
        XCTAssertEqual(
            quality?.loadState, .loaded,
            "Quality must be the single resident tier"
        )
    }
}
