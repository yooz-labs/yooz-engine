// PortAvailability.swift
// IntegrationTests
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Darwin
import Foundation

/// Checks whether a TCP port on the loopback interface is currently bound.
///
/// Used by the integration harness to (a) decide whether to launch its own
/// engine subprocess or reuse a dev instance, and (b) fail fast if the port
/// is held by a non-engine process.
///
/// Implementation: opens an IPv4 TCP socket and tries to `bind` to
/// 127.0.0.1:port with `SO_REUSEADDR=0`. `EADDRINUSE` means "in use".
/// Any other failure is surfaced via the throwing variant; the
/// non-throwing convenience treats errors other than EADDRINUSE as "free"
/// so missing privileges don't cause false positives.
public enum PortAvailability {

    public enum PortError: Error, Equatable {
        case socketCreateFailed(errno: Int32)
        case unexpectedBindFailure(errno: Int32)
    }

    /// Returns true if some process is listening on `127.0.0.1:port`.
    /// Throws on unexpected socket errors (EMFILE, etc.).
    public static func isInUse(_ port: Int) throws -> Bool {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw PortError.socketCreateFailed(errno: errno)
        }
        defer { close(fd) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = UInt16(port).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let result = withUnsafePointer(to: &addr) { ptr -> Int32 in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { rawPtr in
                Darwin.bind(fd, rawPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }

        if result == 0 { return false }
        let err = errno
        if err == EADDRINUSE { return true }
        throw PortError.unexpectedBindFailure(errno: err)
    }
}
