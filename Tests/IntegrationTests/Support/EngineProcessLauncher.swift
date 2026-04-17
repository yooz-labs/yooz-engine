// EngineProcessLauncher.swift
// IntegrationTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Locates and runs a freshly-built `Yooz Engine.app` helper as a child
/// process for integration tests.
///
/// Lifecycle:
///   1. `locateAppBundle(...)` finds the `.app` bundle. Priority:
///      - `YOOZ_ENGINE_APP_PATH` env var (exported by `run-integration.sh`).
///      - Glob under `~/Library/Developer/Xcode/DerivedData` matching
///        `YoozEngine-*/Build/Products/Debug/Yooz Engine.app`, newest first.
///   2. `launch()` spawns the inner executable
///      (`Yooz Engine.app/Contents/MacOS/Yooz Engine`). We launch the
///      binary directly rather than via `open(1)` so we own the child's
///      lifetime and can `terminate()` it on teardown.
///   3. `terminate()` sends SIGTERM, then waits up to 5s, then SIGKILL.
///
/// The launcher is deliberately dumb about health: the caller
/// (`IntegrationTestCase`) combines `launch()` with `TimeoutGuard` against
/// the `/v1/health` URL.
public final class EngineProcessLauncher: @unchecked Sendable {

    public enum LaunchError: Error, CustomStringConvertible {
        case bundleNotFound(String)
        case executableMissing(URL)
        case notAppBundle(URL)
        case alreadyRunning

        public var description: String {
            switch self {
            case .bundleNotFound(let hint):
                return "Engine app bundle not found: \(hint)"
            case .executableMissing(let url):
                return "Engine executable missing inside bundle: \(url.path)"
            case .notAppBundle(let url):
                return "Path is not an .app bundle: \(url.path)"
            case .alreadyRunning:
                return "Engine subprocess already running"
            }
        }
    }

    public let appBundleURL: URL
    public let executableURL: URL

    private let process = Process()
    private var didLaunch = false

    /// Designated initializer. Verifies the bundle layout up-front so bogus
    /// paths fail fast in unit tests.
    public init(appBundleURL: URL) throws {
        guard appBundleURL.pathExtension == "app" else {
            throw LaunchError.notAppBundle(appBundleURL)
        }
        self.appBundleURL = appBundleURL

        // PRODUCT_NAME="Yooz Engine" -> Contents/MacOS/Yooz Engine.
        let macOS = appBundleURL.appendingPathComponent("Contents/MacOS", isDirectory: true)
        let name = appBundleURL.deletingPathExtension().lastPathComponent  // "Yooz Engine"
        let exe = macOS.appendingPathComponent(name, isDirectory: false)
        guard FileManager.default.isExecutableFile(atPath: exe.path) else {
            throw LaunchError.executableMissing(exe)
        }
        self.executableURL = exe
    }

    /// Spawns the engine subprocess. Safe to call only once per instance.
    public func launch() throws {
        guard !didLaunch else { throw LaunchError.alreadyRunning }
        process.executableURL = executableURL
        process.arguments = []
        // Inherit stdout/stderr so test output surfaces Hummingbird logs.
        process.standardInput = FileHandle.nullDevice
        try process.run()
        didLaunch = true
    }

    public var isRunning: Bool { didLaunch && process.isRunning }

    /// Gracefully terminate the subprocess. Idempotent; safe to call if the
    /// process already exited.
    public func terminate() {
        guard didLaunch else { return }
        if process.isRunning {
            process.terminate()
            // Wait up to 5s, polling 100ms, then SIGKILL.
            let deadline = Date().addingTimeInterval(5.0)
            while process.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.1)
            }
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
                process.waitUntilExit()
            }
        }
    }

    // MARK: - Bundle discovery

    /// Resolves an `Yooz Engine.app` URL in priority order.
    public static func locateAppBundle() throws -> URL {
        if let env = ProcessInfo.processInfo.environment["YOOZ_ENGINE_APP_PATH"],
           !env.isEmpty {
            let url = URL(fileURLWithPath: env)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
            throw LaunchError.bundleNotFound("YOOZ_ENGINE_APP_PATH=\(env) does not exist")
        }
        if let derived = try locateInDerivedData() {
            return derived
        }
        throw LaunchError.bundleNotFound(
            "set YOOZ_ENGINE_APP_PATH or build YoozEngine scheme first"
        )
    }

    /// Walks `~/Library/Developer/Xcode/DerivedData/YoozEngine-*/Build/Products/Debug/`
    /// and returns the newest `Yooz Engine.app` found.
    static func locateInDerivedData() throws -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let derived = home.appendingPathComponent(
            "Library/Developer/Xcode/DerivedData", isDirectory: true
        )
        guard FileManager.default.fileExists(atPath: derived.path) else { return nil }

        let entries = (try? FileManager.default.contentsOfDirectory(
            at: derived,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        )) ?? []

        let candidates: [URL] = entries
            .filter { $0.lastPathComponent.hasPrefix("YoozEngine-") }
            .map { $0.appendingPathComponent("Build/Products/Debug/Yooz Engine.app") }
            .filter { FileManager.default.fileExists(atPath: $0.path) }

        // Newest first.
        let sorted = candidates.sorted { lhs, rhs in
            let lm = (try? lhs.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            let rm = (try? rhs.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            return lm > rm
        }
        return sorted.first
    }
}
