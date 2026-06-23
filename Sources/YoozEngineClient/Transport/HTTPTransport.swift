import Foundation
#if canImport(AppKit)
import AppKit
#endif
#if canImport(Darwin)
import Darwin
#endif
import OSLog

/// Default `EngineTransport`: talks to a Yooz Engine helper over an
/// HTTP/WebSocket loopback socket (`127.0.0.1:<port>`), auto-launching a
/// bundled helper when the engine is not already running.
///
/// This is the original `YoozEngineClient` networking implementation, lifted
/// behind the `EngineTransport` seam (epic #192 Phase 2). Behavior is
/// unchanged; loopback consumers get exactly the same launch / probe /
/// stale-recovery semantics they had before the seam existed.
public final class HTTPTransport: EngineTransport {
    public let baseURL: URL
    public let port: Int
    private let session: URLSession
    private let engineBundleID = "live.yooz.engine"
    static let headlessEnvVar = "YOOZ_ENGINE_HEADLESS"
    static let portEnvVar = "YOOZ_ENGINE_PORT"
    /// Command-line flag the helper accepts as a second signal to enter
    /// headless mode. Must stay in sync with `EngineConfig.helperModeArg`
    /// (engine-side). The argv channel is the reliable path when launching
    /// via `NSWorkspace.openApplication`: `OpenConfiguration.arguments` is
    /// propagated by LaunchServices, while `OpenConfiguration.environment`
    /// is NOT reliably propagated to nested helper bundles on macOS 26
    /// (engine#117 / whisper#179).
    static let helperModeArg = "--headless"

    /// Structured log channel for lifecycle events (connect, launch,
    /// stale-engine recovery). Uses OSLog so Console.app can filter by
    /// subsystem `live.yooz.engine.client`.
    private let logger = Logger(subsystem: "live.yooz.engine.client", category: "lifecycle")

    public init(
        host: String = "127.0.0.1",
        port: Int = 19920
    ) {
        guard let url = URL(string: "http://\(host):\(port)") else {
            preconditionFailure("HTTPTransport: invalid host '\(host)' or port \(port)")
        }
        self.baseURL = url
        self.port = port

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        self.session = URLSession(configuration: config)
    }

    /// Check if the engine is reachable, launching the helper if not.
    ///
    /// Resolution order:
    /// 1. Probe `/v1/health`. If 200 OK, return.
    /// 2. If the TCP socket is **refused**, locate the engine binary
    ///    (bundled helper first, then system-installed app) and launch
    ///    it. Poll `/v1/health` for up to 10s.
    /// 3. If the TCP socket is **accepted** but `/v1/health` does not
    ///    respond, a stale/crashed engine is holding the port. Throw
    ///    `YoozEngineError.portHeldByStaleEngine`, unless the env var
    ///    `YOOZ_ENGINE_AUTO_RECOVER=1` is set — in which case attempt
    ///    to terminate the holder via `kill -9` and relaunch.
    public func connect() async throws {
        try Task.checkCancellation()

        logger.debug("connect: probing engine at \(self.baseURL.absoluteString, privacy: .public)")
        switch await probeEngine() {
        case .healthy:
            logger.info("connect: engine already running — reusing existing instance")
            return
        case .refused:
            logger.info("connect: engine not running — launching helper")
            try await launchAndWaitForReady()
            return
        case .staleHolder:
            logger.error("connect: port \(self.port, privacy: .public) held by stale engine")
            if ProcessInfo.processInfo.environment["YOOZ_ENGINE_AUTO_RECOVER"] == "1" {
                logger.warning("connect: YOOZ_ENGINE_AUTO_RECOVER=1 — terminating stale holder")
                try await recoverStaleEngine()
                return
            }
            throw YoozEngineError.portHeldByStaleEngine(port: port)
        }
    }

    /// Returns `true` iff `/v1/health` answered. Any `URLError` is treated as
    /// "not reachable"; `DecodingError` is treated as "reachable but wire
    /// format drift" (engine is up with a newer schema). Throws
    /// `CancellationError` when cancelled.
    public func isReachable() async throws -> Bool {
        do {
            let data = try await get("/v1/health")
            _ = try JSONDecoder().decode(HealthStatus.self, from: data)
            return true
        } catch is CancellationError {
            throw CancellationError()
        } catch is URLError {
            return false
        } catch is DecodingError {
            // Server is up but returned unexpected format (version mismatch)
            return true
        } catch {
            return false
        }
    }

    // MARK: - HTTP helpers

    /// Build a URL from `baseURL` + a path that may contain a `?<query>`
    /// suffix. `URL.appendingPathComponent` percent-encodes the `?` into
    /// the path itself (producing e.g. `/v1/stt/load%3Fwait%3Dtrue`),
    /// which hits a 404 because Hummingbird sees a literal path with
    /// no route registered. Split on `?` and assign the query string
    /// component explicitly via URLComponents so the engine sees the
    /// real query parameters (engine#125's `?wait=true` contract
    /// depended on this and was silently regressing every caller).
    func resolveURL(_ path: String) -> URL {
        let (rawPath, rawQuery): (String, String?)
        if let separator = path.firstIndex(of: "?") {
            rawPath = String(path[..<separator])
            rawQuery = String(path[path.index(after: separator)...])
        } else {
            rawPath = path
            rawQuery = nil
        }
        let pathOnly = baseURL.appendingPathComponent(rawPath)
        guard let rawQuery, !rawQuery.isEmpty else {
            return pathOnly
        }
        guard var components = URLComponents(
            url: pathOnly, resolvingAgainstBaseURL: false
        ) else {
            return pathOnly
        }
        components.percentEncodedQuery = rawQuery
        return components.url ?? pathOnly
    }

    public func get(_ path: String) async throws -> Data {
        let url = resolveURL(path)
        let (data, response) = try await session.data(from: url)
        try validateResponse(response, data: data)
        return data
    }

    public func post(_ path: String, body: Data) async throws -> Data {
        let url = resolveURL(path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await session.data(for: request)
        try validateResponse(response, data: data)
        return data
    }

    public func delete(_ path: String) async throws -> Data {
        let url = resolveURL(path)
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        let (data, response) = try await session.data(for: request)
        try validateResponse(response, data: data)
        return data
    }

    @available(macOS 14.0, iOS 17.0, *)
    public func openSTTStream(language: String, mode: String) async throws -> any STTStreamSession {
        let wsURL = baseURL.appendingPathComponent("v1/stt/stream")
        guard var components = URLComponents(url: wsURL, resolvingAgainstBaseURL: false) else {
            throw YoozEngineError.invalidResponse
        }
        components.scheme = "ws"
        guard let url = components.url else {
            throw YoozEngineError.invalidResponse
        }

        let session = URLSession(configuration: .default)
        let wsTask = session.webSocketTask(with: url)
        wsTask.resume()

        do {
            // Send config frame, then wait for the engine's `ready` ack.
            let config = STTStreamConfig(type: "config", language: language, mode: mode)
            let configData = try JSONEncoder().encode(config)
            let configStr = String(data: configData, encoding: .utf8)!
            try await wsTask.send(.string(configStr))

            let readyMsg = try await wsTask.receive()
            switch readyMsg {
            case .string(let text):
                if let data = text.data(using: .utf8) {
                    let response = try JSONDecoder().decode(WSReadyResponse.self, from: data)
                    if response.type == "error" {
                        throw YoozEngineError.webSocketError(response.message ?? "Unknown error")
                    }
                }
            case .data:
                break
            @unknown default:
                break
            }
        } catch {
            wsTask.cancel(with: .normalClosure, reason: nil)
            session.invalidateAndCancel()
            throw error
        }

        return WebSocketSTTStreamSession(task: wsTask, session: session)
    }

    /// Engine error envelope: every non-2xx response carries `{error, code}`.
    private struct ServerErrorBody: Decodable {
        let error: String
        let code: String
    }

    private func validateResponse(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw YoozEngineError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            // Surface the engine's structured error code/message when present so
            // callers can branch on it (e.g. generation_unavailable vs
            // module_not_bundled, both 501). Fall back to a bare httpError when
            // the body is missing or not the structured shape.
            if let body = try? JSONDecoder().decode(ServerErrorBody.self, from: data) {
                throw YoozEngineError.serverError(
                    statusCode: http.statusCode,
                    code: body.code,
                    message: body.error
                )
            }
            throw YoozEngineError.httpError(statusCode: http.statusCode)
        }
    }

    // MARK: - Probe

    /// Three-outcome probe that distinguishes healthy / refused / stale.
    ///
    /// - `healthy`: `/v1/health` returned 200.
    /// - `refused`: TCP connect refused — no process on the port.
    /// - `staleHolder`: TCP connect accepted, but `/v1/health` timed out
    ///    or returned an unexpected response. Engine is dead or wedged.
    enum ProbeOutcome: Sendable, Equatable {
        case healthy
        case refused
        case staleHolder
    }

    func probeEngine() async -> ProbeOutcome {
        // Short TCP connect check first. If the connection is refused
        // outright we short-circuit — no need to fire an HTTP request.
        let tcpOpen = await isTCPOpen(host: baseURL.host ?? "127.0.0.1", port: port, timeout: 0.5)
        if !tcpOpen {
            return .refused
        }

        // Port is accepting connections — now see whether the process
        // behind it speaks our /v1/health contract.
        let healthConfig = URLSessionConfiguration.ephemeral
        healthConfig.timeoutIntervalForRequest = 1.0
        healthConfig.timeoutIntervalForResource = 2.0
        let probeSession = URLSession(configuration: healthConfig)
        defer { probeSession.invalidateAndCancel() }

        let url = baseURL.appendingPathComponent("/v1/health")
        do {
            let (_, response) = try await probeSession.data(from: url)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return .staleHolder
            }
            return .healthy
        } catch {
            return .staleHolder
        }
    }

    /// Best-effort TCP reachability check; returns true when a socket
    /// `connect()` succeeds within `timeout`. Used as the "refused vs
    /// stale" discriminator. Non-throwing — any failure short-circuits
    /// to `false` (refused).
    private func isTCPOpen(host: String, port: Int, timeout: TimeInterval) async -> Bool {
        await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                #if canImport(Darwin)
                let fd = socket(AF_INET, SOCK_STREAM, 0)
                guard fd >= 0 else { continuation.resume(returning: false); return }
                defer { close(fd) }

                // Non-blocking so we can enforce a connect timeout.
                let flags = fcntl(fd, F_GETFL, 0)
                _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)

                var addr = sockaddr_in()
                addr.sin_family = sa_family_t(AF_INET)
                addr.sin_port = in_port_t(UInt16(port).bigEndian)
                addr.sin_addr.s_addr = inet_addr(host)
                if addr.sin_addr.s_addr == INADDR_NONE {
                    continuation.resume(returning: false)
                    return
                }

                let rc = withUnsafePointer(to: &addr) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { saPtr in
                        Darwin.connect(fd, saPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
                    }
                }
                if rc == 0 {
                    continuation.resume(returning: true)
                    return
                }
                guard errno == EINPROGRESS else {
                    continuation.resume(returning: false)
                    return
                }

                // poll for writability within timeout
                var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
                let ms = Int32(timeout * 1000)
                let pollRc = poll(&pfd, 1, ms)
                if pollRc <= 0 {
                    continuation.resume(returning: false)
                    return
                }

                var err: Int32 = 0
                var errLen = socklen_t(MemoryLayout<Int32>.size)
                let soRc = getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &errLen)
                continuation.resume(returning: soRc == 0 && err == 0)
                #else
                continuation.resume(returning: false)
                #endif
            }
        }
    }

    // MARK: - Engine launch / recovery

    private func launchAndWaitForReady() async throws {
        try await launchEngine()

        logger.debug("launch: polling /v1/health for up to 10s")
        // Wait for engine to become ready (up to 10 seconds)
        for _ in 0..<20 {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(500))
            if case .healthy = await probeEngine() {
                logger.info("launch: engine reported healthy")
                return
            }
        }

        logger.error("launch: engine did not become ready within 10s")
        throw YoozEngineError.engineNotReachable
    }

    private func recoverStaleEngine() async throws {
        #if canImport(Darwin)
        // Find the PID holding the port and SIGKILL it.
        if let pid = pidHoldingPort(port) {
            logger.warning("recover: terminating pid \(pid, privacy: .public) holding port \(self.port, privacy: .public)")
            _ = kill(pid_t(pid), SIGKILL)
            // Give the OS a moment to release the socket.
            try await Task.sleep(for: .milliseconds(500))
        } else {
            logger.error("recover: could not resolve pid holding port; launching anyway")
        }

        try await launchAndWaitForReady()
        #else
        throw YoozEngineError.engineNotReachable
        #endif
    }

    /// Ask `lsof` which PID is listening on the given TCP port. Returns
    /// nil on any failure — diagnostic only.
    private func pidHoldingPort(_ port: Int) -> Int? {
        let task = Process()
        task.launchPath = "/usr/sbin/lsof"
        task.arguments = ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()

        do {
            try task.run()
        } catch {
            return nil
        }
        task.waitUntilExit()
        guard task.terminationStatus == 0 else { return nil }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard let output = String(data: data, encoding: .utf8) else { return nil }
        let firstLine = output
            .split(separator: "\n", omittingEmptySubsequences: true)
            .first
            .map(String.init)?
            .trimmingCharacters(in: .whitespaces) ?? ""
        return Int(firstLine)
    }

    private func launchEngine() async throws {
        #if canImport(AppKit)
        // Prefer the bundled helper when the SDK is embedded in a host
        // app (e.g. Yooz Whisper ships the engine under
        // Contents/Helpers/Yooz Engine (Whisper).app). Fall back to the
        // system-installed standalone engine.
        if let helperURL = bundledHelperURL() {
            logger.info("launch: using bundled helper at \(helperURL.path, privacy: .public)")
            try await openApplication(at: helperURL, createsNewInstance: true)
            return
        }

        guard let engineURL = NSWorkspace.shared.urlForApplication(
            withBundleIdentifier: engineBundleID
        ) else {
            logger.error("launch: no bundled helper and no installed engine for bundle id \(self.engineBundleID, privacy: .public)")
            throw YoozEngineError.engineNotInstalled
        }
        logger.info("launch: using installed engine at \(engineURL.path, privacy: .public)")
        try await openApplication(at: engineURL, createsNewInstance: false)
        #else
        throw YoozEngineError.engineNotInstalled
        #endif
    }

    #if canImport(AppKit)
    var helperLaunchEnvironment: [String: String] {
        [
            Self.headlessEnvVar: "1",
            Self.portEnvVar: "\(port)"
        ]
    }

    /// Argv vector passed to the helper through
    /// `NSWorkspace.OpenConfiguration.arguments`. Carries `--headless` as
    /// the reliable headless signal on macOS 26, where the env-var channel
    /// is not propagated to nested helper bundles by LaunchServices
    /// (engine#117 / whisper#179). Belt-and-suspenders with
    /// `helperLaunchEnvironment` — the engine treats either channel as
    /// sufficient (`EngineConfig.isHelperMode`).
    var helperLaunchArguments: [String] {
        [Self.helperModeArg]
    }

    func helperOpenConfiguration(createsNewInstance: Bool) -> NSWorkspace.OpenConfiguration {
        let config = NSWorkspace.OpenConfiguration()
        config.activates = false
        config.createsNewApplicationInstance = createsNewInstance
        // Run the helper as a headless service: skip the menu-bar
        // status item + Settings scene so the host app's UI is the
        // only surface the user sees. `EngineConfig.isHelper` reads
        // both env and argv channels at startup and forces `.prohibited`
        // activation policy when either signals headless.
        //
        // Both channels are populated because `OpenConfiguration.environment`
        // is NOT reliably propagated to nested helper bundles on macOS 26
        // (engine#117 / whisper#179) — the argv channel is the reliable
        // one. The env channel is preserved for backward compat with
        // engine builds that pre-date the argv path.
        //
        // Port isolation uses the same launch-time contract: a host app
        // with `YoozEngineClient(port:)` launches its bundled helper with
        // `YOOZ_ENGINE_PORT` so each app binds its own loopback port.
        // Port has no argv equivalent yet; if env is dropped the helper
        // falls back to the default port, which is acceptable for now
        // because the SDK and helper share the same default.
        config.environment = helperLaunchEnvironment
        config.arguments = helperLaunchArguments
        return config
    }

    private func openApplication(at url: URL, createsNewInstance: Bool) async throws {
        let config = helperOpenConfiguration(createsNewInstance: createsNewInstance)
        do {
            _ = try await NSWorkspace.shared.openApplication(
                at: url,
                configuration: config
            )
        } catch {
            throw YoozEngineError.engineLaunchFailed(error.localizedDescription)
        }
    }
    #endif

    /// Check whether the SDK is embedded in a host app that ships an
    /// engine helper under `Contents/Helpers/`.
    ///
    /// Convention (see phase5_epic.md, ship model):
    /// `<host>.app/Contents/Helpers/Yooz Engine*.app`
    private func bundledHelperURL() -> URL? {
        #if canImport(AppKit)
        // `Bundle.main.bundleURL` → `<host>.app`
        // We look in `<host>.app/Contents/Helpers/` for any `.app`
        // whose name matches the Yooz Engine family.
        let helpersURL = Bundle.main.bundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("Helpers", isDirectory: true)

        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: helpersURL.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }

        guard let entries = try? fm.contentsOfDirectory(
            at: helpersURL,
            includingPropertiesForKeys: nil
        ) else { return nil }

        return entries.first { url in
            url.pathExtension == "app" && url.lastPathComponent.lowercased().contains("yooz engine")
        }
        #else
        return nil
        #endif
    }
}
