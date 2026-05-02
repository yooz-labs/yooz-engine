// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// # Preview Backend Usage — Sample Client Snippet
///
/// This file demonstrates how a downstream Yooz app (Whisper, Notes,
/// future Crisp / Remi) consumes the Phase 6 engine facilities:
///
/// 1. Switching to the multilingual preview backend (`qwen3_asr_preview`)
///    via the engine's `POST /v1/stt/engine` route.
/// 2. Surfacing the engine-canonical display string and download size
///    estimate to the user before triggering a 3.5 GB first-run fetch.
/// 3. Reading opt-in telemetry off disk (no HTTP route — see "Decision"
///    below).
/// 4. Watching first-run download progress (callers wire this through
///    their own UI; the example shows the consumer-side shape of the
///    data).
///
/// ## Decision: telemetry consumption path
///
/// Phase 6 does **not** add a new HTTP route for metrics. The recorded
/// JSONL file lives at:
///
/// ```
/// ~/Library/Application Support/YoozEngine/telemetry/stt_metrics.jsonl
/// ```
///
/// Rationale:
///
/// - **Privacy-first.** A local file is opt-in, deletable by the user,
///   and never crosses a process boundary. An HTTP route would expose
///   the records to anything that can dial localhost:19920.
/// - **Smaller engine surface area.** No new route, no new auth story,
///   no new schema versioning headache.
/// - **Easier downstream UX.** Whisper / Notes can read the file
///   directly when they want to render a "this run took 380ms"
///   indicator, or they can ignore the file entirely.
///
/// If a future phase needs cross-machine telemetry (it shouldn't —
/// telemetry stays local-first), an HTTP route can be added then.
///
/// ## Example flows
///
/// All example methods are `static`. The struct exists purely to
/// scope the namespace; instantiate the production `YoozEngineClient`
/// to call into the engine for real.
public enum PreviewBackendUsageExample {

    // MARK: - 1. Switch to preview backend

    /// HTTP request body for `POST /v1/stt/engine`. The wire field
    /// name is `engine` (matching the engine's `STTEnginePostRequest`
    /// shape). The engine validates the value against
    /// `STTBackendID.allCases` and rejects unknown identifiers with a
    /// 400 `invalid_engine` error.
    public struct STTBackendSwitchRequest: Codable {
        public let engine: String

        public init(backend: STTBackendIdentifier) {
            self.engine = backend.rawValue
        }
    }

    /// Switch the engine's active STT backend to the multilingual
    /// preview. Caller-side: confirm the user is okay with the
    /// download via `previewDownloadConfirmation()` before invoking
    /// this — once the engine reports "loaded" it has fetched 3.5 GB
    /// of weights into Application Support.
    public static func switchToMultilingualPreview(
        client: YoozEngineClient
    ) async throws {
        let request = STTBackendSwitchRequest(backend: .qwen3ASRPreview)
        let body = try JSONEncoder().encode(request)
        _ = try await client.post("/v1/stt/engine", body: body)
    }

    // MARK: - 2. User-facing labeling

    /// Consumer-visible labels mirroring the engine-side source of
    /// truth in `STTBackendID`. Whisper / Notes UIs read these so
    /// they don't drift from the engine's canonical naming.
    public static func displayName(
        for backend: STTBackendIdentifier
    ) -> String {
        switch backend {
        case .parakeet:        return "Parakeet (Recommended)"
        case .fastConformer:   return "FastConformer (Arabic / Persian / Hebrew)"
        case .appleSTT:        return "Apple Speech (On-device)"
        case .qwen3ASRPreview: return "Multilingual (Preview)"
        }
    }

    /// `true` for backends explicitly tagged as preview. Display the
    /// "Preview — may auto-fallback" disclosure in the UI.
    public static func isPreview(
        _ backend: STTBackendIdentifier
    ) -> Bool {
        backend == .qwen3ASRPreview
    }

    /// Approximate first-run download size in megabytes. `nil` for
    /// backends that ship with the engine.
    public static func estimatedDownloadMB(
        for backend: STTBackendIdentifier
    ) -> Int? {
        switch backend {
        case .parakeet, .fastConformer, .appleSTT: return nil
        case .qwen3ASRPreview:                     return 3_500
        }
    }

    /// Compose a confirmation prompt for the preview backend. UIs are
    /// free to render their own copy; this example gives a reasonable
    /// default.
    public static func previewDownloadConfirmation(
        for backend: STTBackendIdentifier
    ) -> String? {
        guard isPreview(backend), let mb = estimatedDownloadMB(
            for: backend
        ) else {
            return nil
        }
        return
            "\(displayName(for: backend)) needs a \(mb) MB one-time "
            + "download. The model runs entirely on this machine; no "
            + "audio leaves your device."
    }

    // MARK: - 3. Reading opt-in telemetry off disk

    /// Default location of the engine's opt-in JSONL metrics file.
    /// Mirrors `EngineConfig.sttMetricsFileURL`.
    public static var defaultMetricsFileURL: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return appSupport
            .appendingPathComponent("YoozEngine/telemetry")
            .appendingPathComponent("stt_metrics.jsonl")
    }

    /// Decoded metrics record. Schema mirrors the engine-side
    /// `STTBackendMetrics`. Kept in this client SDK so downstream
    /// apps don't need to depend on the engine target.
    public struct DecodedMetric: Codable, Sendable, Equatable {
        public let backend: String
        public let modelVariant: String
        public let audioDurationMs: UInt32
        public let timeToFirstTokenMs: UInt32?
        public let endToEndLatencyMs: UInt32
        public let hardwareClass: String
        public let fellBackFromPreview: Bool
        public let timestampUTC: Date

        private enum CodingKeys: String, CodingKey {
            case backend
            case modelVariant = "model_variant"
            case audioDurationMs = "audio_duration_ms"
            case timeToFirstTokenMs = "time_to_first_token_ms"
            case endToEndLatencyMs = "end_to_end_latency_ms"
            case hardwareClass = "hardware_class"
            case fellBackFromPreview = "fell_back_from_preview"
            case timestampUTC = "timestamp_utc"
        }
    }

    /// Read recent metrics from the on-disk JSONL file. Returns at
    /// most `limit` entries from the tail of the file, decoded with
    /// ISO-8601 dates. Skips malformed lines silently — telemetry
    /// must never break the consumer.
    public static func readRecentMetrics(
        from fileURL: URL = defaultMetricsFileURL,
        limit: Int = 50
    ) -> [DecodedMetric] {
        guard let data = try? Data(contentsOf: fileURL),
              let text = String(data: data, encoding: .utf8) else {
            return []
        }
        let lines = text
            .split(separator: "\n", omittingEmptySubsequences: true)
            .suffix(limit)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return lines.compactMap { line in
            guard let lineData = line.data(using: .utf8) else { return nil }
            return try? decoder.decode(
                DecodedMetric.self, from: lineData
            )
        }
    }

    // MARK: - 4. First-run download progress

    /// Consumer-side shape of `Qwen3ASRModelFetcher.DownloadProgress`.
    /// The engine streams these via its own internal API; downstream
    /// apps that want to mirror the progress (e.g., a "Downloading
    /// Multilingual model — 1.4 / 3.5 GB" toast) should subscribe to
    /// the future progress endpoint or invoke the fetcher directly
    /// via the engine app's plugin surface.
    public enum DownloadProgress: Sendable, Equatable {
        case manifestResolved(totalBytes: Int64, fileCount: Int)
        case fileStarted(path: String, bytes: Int64?)
        case fileBytes(path: String, completed: Int64, total: Int64?)
        case fileFinished(path: String, bytes: Int64)
        case tokenizerPrepStarted
        case tokenizerPrepFinished
        case done
    }

    /// Reduce a sequence of progress events into a render-ready
    /// percentage (0.0 ... 1.0). The example keeps state in a small
    /// struct so callers can drive a SwiftUI progress bar.
    public struct DownloadProgressTracker: Sendable {
        public private(set) var totalBytes: Int64 = 0
        public private(set) var completedBytes: Int64 = 0
        public private(set) var done: Bool = false

        public init() {}

        public mutating func consume(_ event: DownloadProgress) {
            switch event {
            case .manifestResolved(let total, _):
                totalBytes = total
                completedBytes = 0
            case .fileBytes(_, let completed, _):
                if completed > completedBytes {
                    completedBytes = completed
                }
            case .fileFinished:
                break
            case .done:
                done = true
                if totalBytes > 0 {
                    completedBytes = totalBytes
                }
            case .fileStarted, .tokenizerPrepStarted,
                 .tokenizerPrepFinished:
                break
            }
        }

        /// Render-ready fraction in `[0, 1]`. Returns `1.0` once
        /// `.done` has been observed.
        public var fraction: Double {
            if done { return 1.0 }
            guard totalBytes > 0 else { return 0.0 }
            return min(1.0, Double(completedBytes) / Double(totalBytes))
        }
    }

    // MARK: - Identifier (mirrors engine STTBackendID)

    /// Backend identifier mirroring the engine-side enum. Kept here
    /// so the client SDK doesn't have to depend on the engine target.
    public enum STTBackendIdentifier: String, Codable, Sendable, CaseIterable {
        case parakeet
        case fastConformer = "fast_conformer"
        case appleSTT = "apple_stt"
        case qwen3ASRPreview = "qwen3_asr_preview"
    }
}
