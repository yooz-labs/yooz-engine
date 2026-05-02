// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Privacy-first telemetry record describing a single STT
/// transcription run.
///
/// The record carries timing, the active backend, the model variant,
/// the host's coarse-grained hardware class, and a flag indicating
/// whether the run came from an auto-fallback after a preview
/// cold-start failure. **It does not — and tests assert it does not —
/// carry any of:**
///
/// - the transcribed text or any token thereof
/// - audio bytes or any derivative (mel frames, spectrograms, etc.)
/// - user identifiers (Apple ID, hostname, account, etc.)
/// - file system paths
/// - free-form strings derived from user content
///
/// The `modelVariant` field is a constant identifier set by the
/// engine (e.g. `"qwen3-asr-preview-int4"`), not a path or a
/// user-supplied value.
///
/// Telemetry is opt-in. When `EngineConfig.telemetryOptedIn` is
/// `false` (default), the engine constructs metrics records but
/// neither persists nor logs them. See `STTMetricsSink`.
public struct STTBackendMetrics: Codable, Sendable, Equatable, Hashable {

    /// Active backend that produced this transcription.
    public let backend: STTBackendID

    /// Model variant tag chosen by the engine. Constant per backend
    /// (e.g. `"parakeet-tdt-v3"`, `"qwen3-asr-preview-int4"`); never
    /// derived from user content.
    public let modelVariant: String

    /// Length of the audio buffer that was transcribed, in
    /// milliseconds.
    public let audioDurationMs: UInt32

    /// Time-to-first-token in milliseconds (streaming runs).
    /// Optional: backends that buffer-then-finalize (Qwen3 ASR
    /// preview) leave this `nil`.
    public let timeToFirstTokenMs: UInt32?

    /// Wall-clock latency from request received to result returned,
    /// in milliseconds.
    public let endToEndLatencyMs: UInt32

    /// Coarse hardware bucket — see `HardwareClass`. Bucketed so
    /// dashboards cannot fingerprint a specific machine.
    public let hardwareClass: HardwareClass

    /// `true` when this transcription is the result of an
    /// auto-fallback from `qwen3_asr_preview` to Parakeet after a
    /// preview cold-start failure. `false` for every other run.
    public let fellBackFromPreview: Bool

    /// UTC timestamp of the metric event.
    public let timestampUTC: Date

    public init(
        backend: STTBackendID,
        modelVariant: String,
        audioDurationMs: UInt32,
        timeToFirstTokenMs: UInt32?,
        endToEndLatencyMs: UInt32,
        hardwareClass: HardwareClass,
        fellBackFromPreview: Bool,
        timestampUTC: Date
    ) {
        self.backend = backend
        self.modelVariant = modelVariant
        self.audioDurationMs = audioDurationMs
        self.timeToFirstTokenMs = timeToFirstTokenMs
        self.endToEndLatencyMs = endToEndLatencyMs
        self.hardwareClass = hardwareClass
        self.fellBackFromPreview = fellBackFromPreview
        self.timestampUTC = timestampUTC
    }

    // MARK: - Codable

    /// Snake-cased JSON keys. The privacy test reads
    /// `canonicalJSONKeys` (derived from this enum) so adding a case
    /// here without updating the test is a hard CI failure — single
    /// source of truth.
    fileprivate enum CodingKeys: String, CodingKey, CaseIterable {
        case backend
        case modelVariant = "model_variant"
        case audioDurationMs = "audio_duration_ms"
        case timeToFirstTokenMs = "time_to_first_token_ms"
        case endToEndLatencyMs = "end_to_end_latency_ms"
        case hardwareClass = "hardware_class"
        case fellBackFromPreview = "fell_back_from_preview"
        case timestampUTC = "timestamp_utc"
    }

    /// Canonical set of JSON keys this record produces. Derived from
    /// `CodingKeys.allCases` so adding a Swift field automatically
    /// updates the wire-shape tripwire used by the privacy test.
    public static let canonicalJSONKeys: Set<String> = Set(
        CodingKeys.allCases.map(\.rawValue)
    )
}

extension STTBackendMetrics {

    /// Default model-variant tag for each known backend. Constant
    /// strings under engine control, never user-derived.
    public static func defaultModelVariant(
        for backend: STTBackendID
    ) -> String {
        switch backend {
        case .parakeet:
            return "parakeet-tdt-v3"
        case .fastConformer:
            return "fast-conformer-hybrid"
        case .appleSTT:
            return "apple-sfspeech"
        case .qwen3ASRPreview:
            return "qwen3-asr-preview-int4"
        }
    }
}
