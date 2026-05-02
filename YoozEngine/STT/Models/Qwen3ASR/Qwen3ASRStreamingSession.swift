// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

/// Per-WebSocket streaming session for the `qwen3_asr_preview`
/// backend.
///
/// **Streaming strategy.** The Qwen3-Omni audio tower is a Whisper-
/// style non-causal encoder operating on 30 s padded log-mel chunks.
/// The Python reference's `stream_transcribe` is token-streaming
/// (split long audio into seconds-to-minutes chunks at low-energy
/// boundaries, run the offline encoder once per chunk, stream decoder
/// tokens). There is no incremental-encoder code path in the
/// `mlx_audio` reference and the architecture (block attention, not
/// unidirectional) does not support one without forcing a re-prefill
/// on every receive callback — for which the published WER is
/// unknown.
///
/// For our WS path we therefore implement the safe option: buffer the
/// PCM as it arrives, run the offline pipeline once on `finalize()`,
/// and return the result as a `final` frame. Heartbeat `partial`
/// frames are emitted from the WS handler per receive callback so the
/// client can confirm the connection is alive; their text fields are
/// empty until the final pass runs. This trivially satisfies the
/// streaming-vs-offline WER bound (delta is 0 by construction) and
/// keeps per-receive-callback latency in the microseconds range.
///
/// **Per-stream isolation.** Each WS connection holds one session.
/// The session owns its audio buffer; the only shared resource is the
/// `Qwen3ASRBackend` actor singleton, which serializes concurrent
/// `transcribe` calls naturally. Two simultaneous WS sessions on
/// different audio see no cross-contamination because the actor's
/// transcribe call only consumes its argument PCM, not any internal
/// per-utterance state.
public actor Qwen3ASRStreamingSession {

    // MARK: - Public types

    /// Outcome of the offline pass run on `finalize`.
    public struct FinalResult: Sendable, Equatable {
        public let text: String
        public let language: String
        /// Cumulative audio buffered before the offline pass ran, in
        /// milliseconds. Carried into the telemetry record by the
        /// caller.
        public let audioDurationMs: UInt32

        public init(
            text: String,
            language: String,
            audioDurationMs: UInt32
        ) {
            self.text = text
            self.language = language
            self.audioDurationMs = audioDurationMs
        }
    }

    /// Errors surfaced to the WS layer. The handler translates these
    /// into `error` frames; the `Qwen3ASRError` payload, if any, is
    /// reported verbatim for diagnostics.
    public enum SessionError: Error, CustomStringConvertible {
        /// Pipeline was never loaded for this connection. Call
        /// `/v1/stt/load` first.
        case backendNotLoaded
        /// `finalize()` was called with no audio buffered.
        case empty
        /// Session was already finalized or discarded.
        case finalized
        /// Backend was unloaded mid-stream by a concurrent
        /// `POST /v1/stt/engine` switch. Distinct from
        /// `backendNotLoaded` ("never loaded") so the client can
        /// surface "your selection changed; reconnect" instead of
        /// "wasn't loaded".
        case backendChangedDuringStream
        case underlying(Error)

        public var description: String {
            switch self {
            case .backendNotLoaded:
                return "Qwen3-ASR pipeline is not loaded; call /v1/stt/load before streaming."
            case .empty:
                return "Streaming session received no audio before finalize()."
            case .finalized:
                return "Streaming session has already been finalized; open a new connection to continue."
            case .backendChangedDuringStream:
                return "Backend was changed by /v1/stt/engine while this stream was open; reconnect to continue."
            case let .underlying(error):
                return String(describing: error)
            }
        }
    }

    // MARK: - State

    private static let logger = Logger(
        subsystem: "live.yooz.engine",
        category: "Qwen3ASRStreamingSession"
    )

    /// Canonical Qwen3-ASR language label (`"English"`, `"Persian"`,
    /// `"Arabic"`, …). The WS handler resolves the `STTLanguage` ↔
    /// hint mapping via `STTLanguage.qwen3LanguageHint`; this actor
    /// only needs the resolved string. Keeping the dependency surface
    /// at `String` lets the type compile inside the SwiftPM
    /// `Qwen3ASR` target (which does not pull in `STTLanguage`).
    private let languageHint: String
    private let backend: Qwen3ASRBackend
    private let sampleRate: Int
    /// Hard cap on buffered audio. Qwen3-ASR's audio tower pads to
    /// `chunkLength` (30 s) per call; very long buffers get truncated
    /// inside the pipeline. We cap conservatively to bound memory on
    /// the WS path: 5 minutes at 16 kHz Float32 ≈ 19 MB.
    private let maxBufferedSamples: Int

    private var buffer: [Float] = []
    private var totalSamples: Int = 0
    private var finalized: Bool = false

    // MARK: - Init

    public init(
        languageHint: String,
        backend: Qwen3ASRBackend = .shared,
        sampleRate: Int = 16_000,
        maxBufferedSeconds: Int = 300
    ) {
        self.languageHint = languageHint
        self.backend = backend
        self.sampleRate = sampleRate
        self.maxBufferedSamples = sampleRate * maxBufferedSeconds
    }

    // MARK: - Public surface

    /// Cumulative audio buffered so far, in milliseconds.
    public var audioDurationMs: UInt32 {
        guard sampleRate > 0 else { return 0 }
        let ms = (UInt64(totalSamples) * 1_000) / UInt64(sampleRate)
        return UInt32(min(ms, UInt64(UInt32.max)))
    }

    /// `true` once `finalize()` has been called. Subsequent `push` /
    /// `finalize` calls throw `SessionError.finalized`.
    public var isFinalized: Bool { finalized }

    /// Number of samples currently buffered. Useful for telemetry and
    /// tests; not part of the WS protocol.
    public var bufferedSampleCount: Int { totalSamples }

    /// Outcome of a `push(samples:)` call. Carries the post-append
    /// cumulative total and the number of samples that were *actually*
    /// accepted into the buffer (which may be less than the input
    /// length when the soft cap was hit). The WS handler reads
    /// `accepted < requested` to fire the one-shot
    /// `buffer_cap_reached` warning frame on the chunk that crossed
    /// the cap, not just on subsequent fully-rejected chunks.
    public struct PushOutcome: Sendable, Equatable {
        public let totalSamples: Int
        public let accepted: Int
        public let requested: Int

        public var truncated: Bool { accepted < requested }
    }

    /// Append PCM samples to the session buffer. Returns a
    /// `PushOutcome` describing how many samples were actually
    /// accepted (post-cap truncation) and the cumulative buffered
    /// total. Throws if the session has already been finalized.
    @discardableResult
    public func push(samples: [Float]) throws -> PushOutcome {
        guard !finalized else { throw SessionError.finalized }
        let requested = samples.count
        guard requested > 0 else {
            return PushOutcome(
                totalSamples: totalSamples,
                accepted: 0,
                requested: 0
            )
        }

        let remainingCapacity = maxBufferedSamples - totalSamples
        if remainingCapacity <= 0 {
            // Soft cap: silently drop further audio rather than
            // throwing. The user is past 5 minutes of streamed audio
            // on a backend the docs label "preview"; the right call is
            // to keep the connection alive and return what we have on
            // close.
            Self.logger.warning(
                "Qwen3 streaming session at buffer cap (\(self.maxBufferedSamples) samples); dropping further audio."
            )
            return PushOutcome(
                totalSamples: totalSamples,
                accepted: 0,
                requested: requested
            )
        }

        let take = Swift.min(requested, remainingCapacity)
        if take == requested {
            buffer.append(contentsOf: samples)
        } else {
            buffer.append(contentsOf: samples.prefix(take))
        }
        totalSamples += take
        return PushOutcome(
            totalSamples: totalSamples,
            accepted: take,
            requested: requested
        )
    }

    /// Run the offline pipeline on the accumulated PCM and return the
    /// transcription. The session is single-shot; subsequent calls
    /// throw `SessionError.finalized`.
    public func finalize() async throws -> FinalResult {
        guard !finalized else { throw SessionError.finalized }
        guard totalSamples > 0 else {
            finalized = true
            throw SessionError.empty
        }

        finalized = true
        let pcm = buffer
        // Drop the host-side buffer immediately so the actor isn't
        // holding the audio in memory across the (potentially long)
        // backend transcribe call.
        buffer = []

        do {
            let result = try await backend.transcribe(
                pcm: pcm,
                language: languageHint
            )
            return FinalResult(
                text: result.text,
                language: result.language,
                audioDurationMs: audioDurationMs
            )
        } catch let error as Qwen3ASRError where error == .pipelineNotLoaded {
            // `pipelineNotLoaded` mid-finalize means the singleton
            // backend was unloaded under us — most commonly by a
            // concurrent `POST /v1/stt/engine` switch. We had
            // buffered audio (totalSamples > 0 was the precondition
            // for entering this branch), so this isn't the
            // "never-loaded" case.
            throw SessionError.backendChangedDuringStream
        } catch {
            throw SessionError.underlying(error)
        }
    }

    /// Drop any buffered audio without running the pipeline. Used by
    /// the WS handler when a client disconnects before any audio has
    /// been streamed (or when an error closes the session early).
    /// Idempotent: calling `discard()` on an already-finalized
    /// session is a no-op rather than an error, so the WS handler
    /// can defensively call it from cleanup paths without state
    /// checks.
    public func discard() {
        buffer = []
        totalSamples = 0
        finalized = true
    }
}
