import EngineCore
import MLX
import XCTest
import YoozEngineClient
@testable import YoozEngineInProcess

/// Tier 1 of the in-process residency invariant: the engine bounds MLX's Metal
/// buffer cache so the in-process runtime can't run the consumer app's RSS into
/// the tens of GB.
///
/// Background: mlx-swift's default `cacheLimit` equals its memory limit (~1.5x
/// the device's recommended working set), which scales with installed RAM, so
/// the cache of freed-but-retained buffers can grow into the tens of GB. The
/// loopback packaging hid this by running the engine in a separate, kill-able
/// helper process; in-process that growth is charged to the consumer app's own
/// RSS for the app's lifetime (the 46 GB regression). The cap is applied at the
/// MLX model-load paths (see `EngineConfig.mlxCacheLimitBytes`), so it lands
/// before the cache-growing inference and never touches the Metal allocator in
/// the non-GPU structural tests.
final class InProcessMemoryTests: XCTestCase {
    private var sttEnabled: Bool {
        ProcessInfo.processInfo.environment["YOOZ_STT_LOAD_MODELS"] == "1"
    }

    func testCacheLimitConstantIsConservativelyBounded() {
        // 512 MB: ample scratch for the STT encoder + LLM KV without pinning
        // GBs. If this constant is bumped, it is a deliberate memory-budget
        // decision, not an accident.
        XCTAssertEqual(EngineConfig.mlxCacheLimitBytes, 512 * 1024 * 1024)
    }

    /// Live proof that a real model load applies the cap. Gated behind
    /// `YOOZ_STT_LOAD_MODELS=1` (downloads Parakeet) and only meaningful under
    /// the app-hosted xctest where `default.metallib` is present — a plain
    /// `swift test` run lacks the metallib and would fault on the first MLX
    /// allocator call, the same reason the engine's other live MLX tests are
    /// xcodebuild-only.
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
}
