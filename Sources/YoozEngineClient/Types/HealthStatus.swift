import Foundation

public struct HealthStatus: Codable, Sendable {
    public let status: String
    public let version: String
    public let modules: ModuleStatus

    public var isHealthy: Bool { status == "ok" }

    public init(status: String, version: String, modules: ModuleStatus) {
        self.status = status
        self.version = version
        self.modules = modules
    }
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

    public init(
        stt: Bool,
        llm: Bool,
        touchup: Bool,
        grammar: Bool,
        vad: Bool,
        tts: Bool,
        infinite: Bool?
    ) {
        self.stt = stt
        self.llm = llm
        self.touchup = touchup
        self.grammar = grammar
        self.vad = vad
        self.tts = tts
        self.infinite = infinite
    }
}
