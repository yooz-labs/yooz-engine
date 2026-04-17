// TimeoutGuard.swift
// IntegrationTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Polls a predicate until it returns true or a deadline is reached.
///
/// Small, dependency-free helper used by `IntegrationTestCase` to wait for
/// `/v1/health` to come up without relying on arbitrary `sleep` durations.
public enum TimeoutGuard {

    public struct TimedOut: Error, CustomStringConvertible {
        public let description: String
        public init(_ description: String) { self.description = description }
    }

    /// Runs `check()` every `poll` until it returns true, or throws
    /// `TimedOut` after `timeout` wall-clock has elapsed.
    ///
    /// - Parameters:
    ///   - description: Human-readable label used in the timeout error.
    ///   - timeout: Maximum wall-clock duration to wait.
    ///   - poll: Interval between predicate evaluations.
    ///   - check: Async predicate. Thrown errors are swallowed and treated
    ///     as "not yet ready"; only a hard timeout surfaces to the caller.
    public static func waitUntil(
        _ description: String,
        timeout: Duration,
        poll: Duration = .milliseconds(250),
        check: @escaping @Sendable () async throws -> Bool
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while ContinuousClock.now < deadline {
            do {
                if try await check() { return }
            } catch {
                // treat transient errors as "not yet ready"; loop until deadline.
            }
            try await Task.sleep(for: poll)
        }
        // One final attempt after deadline — useful if the sleep overshoots.
        if (try? await check()) == true { return }
        throw TimedOut("timed out after \(timeout) waiting for: \(description)")
    }
}
