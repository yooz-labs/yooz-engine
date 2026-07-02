import Foundation

/// SDK-side mirror of the engine's `EngineCore.MLXWorkloadClass` (engine#228).
/// `YoozEngineClient` is a separate SwiftPM target that can't import
/// `EngineCore`, so the contract is held by matching `rawValue`s — same
/// pattern as `LoadState` in `LLMTypes.swift`.
///
/// Optional, additive wire field on `TouchUpRequest` / `LLMGenerateRequest`:
/// omitting it preserves today's behavior (the engine defaults both routes
/// to `.background`). Set `.interactive` only for a genuinely
/// latency-sensitive one-off call that should never queue behind another
/// interactive workload.
public enum GPUWorkloadClass: String, Codable, Sendable {
    /// Latency-sensitive, small — never queues at the engine's admission
    /// gate. Reserve for calls that must not wait behind a concurrently
    /// running background submission.
    case interactive
    /// Throughput work that can tolerate queuing behind an active
    /// interactive workload (the default for touch-up / raw generation).
    case background
}
