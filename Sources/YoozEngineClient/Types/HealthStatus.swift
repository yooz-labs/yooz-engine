import Foundation

public struct HealthStatus: Codable, Sendable {
    public let status: String
    public let version: String
    public let modules: ModuleStatus

    public var isHealthy: Bool { status == "ok" }
}

public struct ModuleStatus: Codable, Sendable {
    public let stt: Bool
    public let llm: Bool
    public let touchup: Bool
    public let grammar: Bool
    public let vad: Bool
    public let tts: Bool
    /// Optional for back-compat: older engines that predate the Infinite
    /// module omit this key, decoding to `nil`. `true` once Infinite is
    /// bundled and loaded.
    public let infinite: Bool?
}
