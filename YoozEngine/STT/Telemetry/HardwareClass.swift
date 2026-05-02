// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Coarse-grained Apple Silicon / Intel chip class. The metrics sink
/// records this — never a free-form CPU brand string — so dashboards
/// can bucket runs without identifying a specific user's machine.
///
/// The mapping is intentionally lossy: we only distinguish the four
/// Apple Silicon major generations the engine targets (M1, M2, M3,
/// M4) plus a fall-through bucket. The chip's variant (Pro / Max /
/// Ultra) and the Mac product line are not recorded.
public enum HardwareClass: String, Codable, Sendable, Equatable, Hashable, CaseIterable {
    case appleSiliconM1 = "apple_silicon_m1"
    case appleSiliconM2 = "apple_silicon_m2"
    case appleSiliconM3 = "apple_silicon_m3"
    case appleSiliconM4 = "apple_silicon_m4"
    case appleSiliconUnknown = "apple_silicon_unknown"
    case intel
    case unknown

    /// Pure classification. Public so tests can drive every branch
    /// from a stub `HardwareClassResolver` without needing a real
    /// `sysctl` call.
    ///
    /// - Parameter brandString: the value `sysctlbyname(
    ///   "machdep.cpu.brand_string", ...)` would return on a real
    ///   Mac. May be empty.
    public static func classify(brandString: String) -> HardwareClass {
        let trimmed = brandString.trimmingCharacters(
            in: .whitespacesAndNewlines
        )
        if trimmed.isEmpty {
            return .unknown
        }

        // Match the four Apple Silicon major generations by prefix.
        // Order matters: "Apple M1 Pro" must hit M1, not Apple-Silicon-
        // unknown.
        if trimmed.hasPrefix("Apple M1") {
            return .appleSiliconM1
        }
        if trimmed.hasPrefix("Apple M2") {
            return .appleSiliconM2
        }
        if trimmed.hasPrefix("Apple M3") {
            return .appleSiliconM3
        }
        if trimmed.hasPrefix("Apple M4") {
            return .appleSiliconM4
        }
        if trimmed.hasPrefix("Apple") {
            // Future Apple Silicon (M5, etc.) — record the family but
            // not a specific number we don't know how to interpret.
            return .appleSiliconUnknown
        }

        let lower = trimmed.lowercased()
        if lower.contains("intel") || lower.contains("genuine intel") {
            return .intel
        }
        return .unknown
    }
}

/// Resolves the machine's coarse-grained hardware class. The default
/// implementation reads `sysctlbyname`. Tests inject a static stub so
/// they can exercise every classification branch without depending on
/// the host machine.
public protocol HardwareClassResolver: Sendable {
    func resolve() -> HardwareClass
}

/// Production resolver: reads `machdep.cpu.brand_string` via
/// `sysctlbyname`. Returns `.unknown` if the syscall fails (rather
/// than crashing or surfacing the underlying errno).
public struct SystemHardwareClassResolver: HardwareClassResolver {
    public init() {}

    public func resolve() -> HardwareClass {
        HardwareClass.classify(brandString: Self.readBrandString())
    }

    /// Read the CPU brand string via `sysctlbyname`. Empty string on
    /// any failure path so the classifier returns `.unknown`.
    static func readBrandString() -> String {
        var size: Int = 0
        let key = "machdep.cpu.brand_string"
        // First call: discover the buffer size.
        if sysctlbyname(key, nil, &size, nil, 0) != 0 || size == 0 {
            return ""
        }
        var buffer = [CChar](repeating: 0, count: size)
        if sysctlbyname(key, &buffer, &size, nil, 0) != 0 {
            return ""
        }
        return String(cString: buffer)
    }
}

/// Test-only resolver. Inject a brand string to exercise every
/// classification branch.
public struct StaticHardwareClassResolver: HardwareClassResolver {
    public let brandString: String

    public init(brandString: String) {
        self.brandString = brandString
    }

    public func resolve() -> HardwareClass {
        HardwareClass.classify(brandString: brandString)
    }
}
