import Foundation
import XCTest
@testable import YoozEngineClient

/// Tests for `YoozEngineClient.connect()` and the probe / error shapes.
///
/// Per repo policy **no mocks**. The HTTP/live-engine interaction tests
/// are gated behind environment flags so CI doesn't spuriously launch
/// the menu bar helper. To run the live tests locally:
///
/// ```
/// YOOZ_ENGINE_LIVE_TESTS=1 \
///     xcodebuild -scheme YoozEngineClientTests test
/// ```
final class ConnectTests: XCTestCase {

    // MARK: - Error shape

    func testPortHeldByStaleEngineErrorCarriesPort() {
        let error = YoozEngineError.portHeldByStaleEngine(port: 19920)
        XCTAssertTrue((error.errorDescription ?? "").contains("19920"))
        XCTAssertTrue((error.errorDescription ?? "").contains("YOOZ_ENGINE_AUTO_RECOVER"))
    }

    func testYoozEngineErrorEquatable() {
        XCTAssertEqual(
            YoozEngineError.portHeldByStaleEngine(port: 19920),
            YoozEngineError.portHeldByStaleEngine(port: 19920)
        )
        XCTAssertNotEqual(
            YoozEngineError.portHeldByStaleEngine(port: 19920),
            YoozEngineError.portHeldByStaleEngine(port: 1)
        )
        XCTAssertNotEqual(
            YoozEngineError.portHeldByStaleEngine(port: 19920),
            YoozEngineError.engineNotReachable
        )
    }

    func testEngineNotInstalledHasUserGuidance() {
        let error = YoozEngineError.engineNotInstalled
        XCTAssertTrue((error.errorDescription ?? "").contains("install"))
    }

    // MARK: - Probe outcome on a closed port (no mocks — uses real socket)

    /// Probing a closed port must resolve to `.refused`, never `.staleHolder`.
    /// This differentiates the connect() branches: `.refused` triggers a
    /// launch, `.staleHolder` raises `portHeldByStaleEngine`. If this test
    /// flips we risk launching the engine in "stale holder" paths.
    func testProbeRefusedOnClosedPort() async throws {
        // Pick a port that is almost certainly not in use in a test
        // environment. Port 1 requires root to bind, so regular
        // processes will not hold it.
        let client = YoozEngineClient(port: 1)
        let outcome = await client.probeEngine()
        XCTAssertEqual(outcome, .refused)
    }

    // MARK: - Live engine (env-gated)

    /// If an engine is already running, connect() short-circuits without
    /// launching a new one. This test only runs when an engine is
    /// actually live on :19920 — otherwise it skips.
    func testConnectShortCircuitsWhenEngineAlreadyRunning() async throws {
        guard ProcessInfo.processInfo.environment["YOOZ_ENGINE_LIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires YOOZ_ENGINE_LIVE_TESTS=1 and a running engine")
        }
        let client = YoozEngineClient()
        // Should not throw; should not time out. Engine must already be
        // listening on :19920.
        try await client.connect()
        let health = try await client.health()
        XCTAssertEqual(health.isHealthy, true)
    }

    /// When no engine is running AND the engine is not installed (no
    /// bundled helper + no system install), connect() must raise
    /// `engineNotInstalled` rather than hang or fall through.
    func testConnectErrorsWhenEngineAbsent() async throws {
        guard ProcessInfo.processInfo.environment["YOOZ_ENGINE_OFFLINE_TESTS"] == "1" else {
            throw XCTSkip(
                "Requires YOOZ_ENGINE_OFFLINE_TESTS=1 AND no engine running AND "
                    + "no Yooz Engine.app installed. Set up the env explicitly "
                    + "before running (we can't detect system installs reliably)."
            )
        }
        // Use a non-standard port to ensure we hit .refused. If a real
        // engine is on 19920 we'd short-circuit and defeat the test.
        let client = YoozEngineClient(port: 54321)
        do {
            try await client.connect()
            XCTFail("connect() should throw when engine is absent")
        } catch YoozEngineError.engineNotInstalled {
            // Expected
        } catch YoozEngineError.engineLaunchFailed {
            // Also acceptable: we found an app but launch failed.
        } catch YoozEngineError.engineNotReachable {
            // Also acceptable: launched but never became ready within 10s.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
}
