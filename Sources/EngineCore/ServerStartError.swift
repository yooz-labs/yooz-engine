// ServerStartError.swift
// EngineCore
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Errors thrown by `APIServer.start()` / `APIServer.restart()`.
///
/// Lives in `EngineCore` (not the app target) so unit tests and thin
/// clients can decode/match on the cases without pulling the Hummingbird
/// server stack in.
///
/// `portInUse` carries the conflicting port and, when available, the PID
/// of the holder (from `lsof`). `pid` is nil when lookup fails or is
/// unavailable — callers should not treat a nil PID as "no conflict",
/// the mere presence of this case means the bind failed with `EADDRINUSE`.
public enum ServerStartError: LocalizedError, Sendable, Equatable {
    case portInUse(port: Int, pid: Int?)
    case failedToBind(String)
    case healthCheckFailed
    case crashed(String)

    public var errorDescription: String? {
        switch self {
        case .portInUse(let port, let pid):
            if let pid = pid {
                return "Port \(port) is already in use by process \(pid). "
                    + "Stop that process or set YOOZ_ENGINE_AUTO_RECOVER=1."
            }
            return "Port \(port) is already in use by another process. "
                + "Stop that process and try again."
        case .failedToBind(let reason):
            return "Failed to bind server: \(reason)"
        case .healthCheckFailed:
            return "Server started but health check failed"
        case .crashed(let reason):
            return "Engine server terminated unexpectedly: \(reason)"
        }
    }

    /// Stable machine-readable identifier — matches the `code` field in
    /// HTTP error bodies produced elsewhere in the server.
    public var code: String {
        switch self {
        case .portInUse: return "port_in_use"
        case .failedToBind: return "failed_to_bind"
        case .healthCheckFailed: return "health_check_failed"
        case .crashed: return "crashed"
        }
    }
}

/// Utilities for diagnosing an `EADDRINUSE` bind failure.
///
/// These helpers are split out so they can be exercised without booting
/// the server — tests drive `PortDiagnostics.isAddressInUse(_:)` with
/// synthesized errors.
public enum PortDiagnostics {
    /// True if the given error indicates the bind address was already in
    /// use. Recognises the raw POSIX `EADDRINUSE` errno (48 on Darwin,
    /// 98 on Linux) and also falls back to a string match on the error
    /// description, since NIO wraps the errno inside `IOError` which is
    /// not publicly imported here.
    public static func isAddressInUse(_ error: Error) -> Bool {
        let nsError = error as NSError
        // Darwin: 48, Linux: 98
        if nsError.code == Int(EADDRINUSE) { return true }
        let description = String(describing: error).lowercased()
        if description.contains("address already in use") { return true }
        if description.contains("eaddrinuse") { return true }
        return false
    }

    /// True iff the given TCP port on the loopback interface is already
    /// bound by another process. Implementation opens an IPv4 TCP
    /// socket and attempts to `bind` it; on `EADDRINUSE` we return
    /// true. Any other failure returns false (treat "unknown" as
    /// "not in use" so unrelated errors don't block startup).
    ///
    /// Used by `APIServer.start()` as a pre-check before spawning the
    /// server task so the real failure can be diagnosed instead of
    /// masked by a downstream `URLError` from the health probe.
    public static func isPortInUse(_ port: Int) -> Bool {
        #if canImport(Darwin) || canImport(Glibc)
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let result = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { rawPtr in
                #if canImport(Darwin)
                return Darwin.bind(fd, rawPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
                #else
                return Glibc.bind(fd, rawPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
                #endif
            }
        }

        if result == 0 { return false }
        return errno == EADDRINUSE
        #else
        return false
        #endif
    }

    /// Query `lsof` for the PID holding a TCP port on localhost. Returns
    /// `nil` when `lsof` is not available, times out, or reports no
    /// holder. Never throws — diagnostic helper only.
    public static func pidHoldingPort(_ port: Int) -> Int? {
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

        // lsof may list multiple PIDs (one per line); take the first.
        let firstLine = output
            .split(separator: "\n", omittingEmptySubsequences: true)
            .first
            .map(String.init)?
            .trimmingCharacters(in: .whitespaces) ?? ""
        return Int(firstLine)
    }
}
