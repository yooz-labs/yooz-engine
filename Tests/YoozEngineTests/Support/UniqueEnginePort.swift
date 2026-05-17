// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation

/// Per-test unique port assignment helper.
///
/// `APIServer.stop()` awaits `serverTask` to return but Hummingbird /
/// swift-NIO can leave the listening socket bound briefly after
/// `app.run()` unwinds, so a back-to-back `start()` on the fixed
/// `EngineConfig.defaultPort` (19920) races the previous bind and the
/// next test fails with `portInUse(pid: nil)` — the socket is held by
/// the same xctest process but lsof doesn't see it as `LISTEN`.
/// See engine#122.
///
/// Instead of trying to make `stop()` synchronously release the
/// kernel socket (a Hummingbird-internal property we can't guarantee
/// from the caller), each test reserves a fresh port via this helper.
/// The helper bumps `YOOZ_ENGINE_PORT` to a new value every call;
/// `EngineConfig.port` reads the env var on every access, so both
/// `APIServer.start()` and the test's URL construction pick up the
/// same port without further plumbing.
enum UniqueEnginePort {
    /// Process-wide monotonic counter for port assignment. Starts at
    /// the default engine port + 1 so the first test in a run uses
    /// 19921, leaving 19920 free for any concurrent dev-machine
    /// engine the developer may have running.
    private static let counter = Counter(start: EngineConfig.defaultPort + 1)

    /// Reserve a fresh port for the next `APIServer()` boot in the
    /// current test. Sets `YOOZ_ENGINE_PORT` so `EngineConfig.port`
    /// returns the chosen value. Call from `setUp()` (or before
    /// instantiating the server) in every XCTestCase that boots a
    /// real `APIServer`.
    ///
    /// Wraps at 65000 back to `defaultPort + 1` — a single test run
    /// will never produce 45000 unique tests, but the wrap keeps the
    /// counter inside the user-port range.
    static func assignFreshPort() {
        let next = counter.next()
        let bounded = ((next - (EngineConfig.defaultPort + 1)) % 45000)
            + (EngineConfig.defaultPort + 1)
        setenv(EngineConfig.portEnvVar, String(bounded), 1)
    }

    /// Monotonic counter. Plain `NSLock`-guarded `Int` rather than an
    /// `actor` so callers don't need to `await` for a port assignment
    /// (the helper has to run synchronously inside `setUp` and the
    /// `withServer` body).
    private final class Counter: @unchecked Sendable {
        private var value: Int
        private let lock = NSLock()

        init(start: Int) {
            value = start
        }

        func next() -> Int {
            lock.lock()
            defer { lock.unlock() }
            let current = value
            value += 1
            return current
        }
    }
}
