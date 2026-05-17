// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import os

private let logger = Logger(subsystem: "live.yooz.engine", category: "stt-cadence")

/// Streaming transcriber - accumulates audio and transcribes
/// Note: Real-time streaming preview is a work in progress. Currently uses batch mode.
public final class StreamingTranscriber {

    // MARK: - Configuration

    /// Minimum mel frames before processing
    private let minMelFrames: Int

    /// Minimum number of newly-buffered samples between successive
    /// emissions, i.e. the partial-emission cadence floor. Once the
    /// transcriber has emitted a partial, the next call is treated as a
    /// cache hit (returning the prior `currentResult()`) until this many
    /// additional samples have accumulated. Set to 0 to disable
    /// throttling and re-encode on every call.
    private let partialEmissionIntervalSamples: Int

    /// Context size for draft region
    private let contextSize: (left: Int, right: Int)

    /// How many encoder frames are in draft region
    private var dropSize: Int { contextSize.right }

    /// Sample rate
    private let sampleRate: Int

    /// Subsampling factor (mel frames -> encoder frames)
    private let subsamplingFactor: Int

    /// Hop length for mel spectrogram
    private let hopLength: Int

    // MARK: - State

    private let model: ParakeetModel
    private let preprocessor: AudioPreprocessor

    /// Accumulated audio buffer
    private var audioBuffer: [Float] = []

    /// `audioBuffer.count` at the time of the last successful encode
    /// + decode pass. Used together with
    /// `partialEmissionIntervalSamples` to throttle the per-frame
    /// re-encode rate when the caller feeds audio in short
    /// (~64 ms) frames.
    private var lastEmittedBufferCount: Int = 0

    /// Finalized tokens (stable, won't change)
    private var finalizedTokens: [AlignedToken] = []

    /// Draft tokens (may change on next call)
    private var draftTokens: [AlignedToken] = []

    // MARK: - Initialization

    /// - Parameters:
    ///   - model: Loaded Parakeet model.
    ///   - contextSize: `(left, right)` encoder frames; `right` is the
    ///     draft window size dropped from the finalized split.
    ///   - minChunkDuration: Minimum audio duration (seconds) the
    ///     accumulated buffer must reach before the first partial is
    ///     emitted. Acts as a warm-up floor; once cleared, subsequent
    ///     emissions are gated by `partialEmissionInterval` instead.
    ///   - partialEmissionInterval: Minimum new audio (seconds) between
    ///     successive partial emissions. Default `2.0` produces a
    ///     ~2-second partial cadence regardless of how frequently the
    ///     caller calls `addAudio`. Lower values (0.5–1.0) give more
    ///     visual feedback at higher CPU cost; `0` disables the throttle
    ///     and re-encodes on every call (legacy behaviour). See
    ///     `EngineConfig.streamingPartialIntervalSec` for the
    ///     process-wide default and the `YOOZ_STT_PARTIAL_INTERVAL_SEC`
    ///     env-var override.
    ///   - preprocessConfig: Optional override of the model's default
    ///     preprocessor config (e.g. for `audioMode`).
    public init(
        model: ParakeetModel,
        contextSize: (Int, Int) = (24, 24),
        minChunkDuration: Float = 0.5,
        partialEmissionInterval: Float = 2.0,
        preprocessConfig: PreprocessConfig? = nil
    ) {
        self.model = model

        // Use override config if provided, otherwise use model's config
        let config = preprocessConfig ?? model.config.preprocessor
        self.preprocessor = AudioPreprocessor(config: config)

        self.sampleRate = config.sampleRate
        self.subsamplingFactor = model.config.encoder.subsamplingFactor
        self.hopLength = config.hopLength
        self.contextSize = contextSize

        // Minimum mel frames = minChunkDuration seconds worth
        self.minMelFrames = Int(minChunkDuration * Float(sampleRate) / Float(hopLength))

        // Cadence floor in samples. Clamp negatives to `0` so a negative
        // interval collapses onto the documented `0` opt-out (gate
        // disabled, re-encode on every call) instead of relying on the
        // `> 0` check in `addAudio` to absorb the surprise. Keeps the
        // contract symmetric: anything <= 0 means "no throttle".
        let intervalSamples = Int(max(0, partialEmissionInterval) * Float(sampleRate))
        self.partialEmissionIntervalSamples = intervalSamples
    }

    // MARK: - Public Methods

    /// Add audio samples and get current transcription
    /// Uses batch mode for accuracy (streaming preview disabled for now)
    public func addAudio(samples: [Float]) -> ParakeetResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        // Accumulate audio
        audioBuffer.append(contentsOf: samples)

        // Cadence throttle (engine #135). When the caller feeds audio
        // in short frames (e.g. 1024 samples / ~64 ms from an
        // `AVAudioEngine` tap), the buffer-accumulation pattern in the
        // rest of this method would otherwise re-encode the entire
        // accumulated buffer on every call. The throttle keeps the
        // per-frame fast path cheap (just an `append`) and gates the
        // mel + encode + decode pass to once per
        // `partialEmissionIntervalSamples`. `lastEmittedBufferCount`
        // starts at `0`, so the first call still pays the encode cost
        // as soon as `minMelFrames` is reached, regardless of the
        // interval; only subsequent emissions are throttled.
        if partialEmissionIntervalSamples > 0
            && lastEmittedBufferCount > 0
            && audioBuffer.count - lastEmittedBufferCount < partialEmissionIntervalSamples {
            return currentResult()
        }

        // Align to hop length
        let alignedLength = (audioBuffer.count / hopLength) * hopLength
        guard alignedLength >= hopLength else {
            return currentResult()
        }

        // Compute mel for ALL accumulated audio
        let alignedAudio = Array(audioBuffer[0..<alignedLength])
        let mel = preprocessor.logMelSpectrogram(MLXArray(alignedAudio))

        // Check if we have enough mel frames
        guard mel.dim(1) >= minMelFrames else {
            return currentResult()
        }

        // Pad mel to subsampling factor (instead of truncating to avoid losing audio)
        let melLength = mel.dim(1)
        let remainder = melLength % subsamplingFactor
        let alignedMel: MLXArray
        if remainder == 0 {
            alignedMel = mel
        } else {
            // Pad with zeros to reach next multiple of subsamplingFactor
            let padSize = subsamplingFactor - remainder
            let padding = MLXArray.zeros([mel.dim(0), padSize, mel.dim(2)])
            alignedMel = concatenated([mel, padding], axis: 1)
        }

        guard alignedMel.dim(1) > 0 else {
            return currentResult()
        }

        // Encode full sequence (batch mode for accuracy)
        let (features, lengths) = model.encoder(alignedMel)
        eval(features, lengths)

        let encoderLength = features.dim(1)

        // Decode entire feature sequence
        let allTokens = model.tdtDecode(
            features: features,
            length: encoderLength
        )

        // Split tokens into finalized vs draft
        let finalizedLength = max(0, encoderLength - dropSize)
        let finalizedDuration = Float(finalizedLength * subsamplingFactor * hopLength) / Float(sampleRate)

        // Clear and rebuild token lists
        finalizedTokens = []
        draftTokens = []

        for token in allTokens {
            if token.start + token.duration <= finalizedDuration {
                finalizedTokens.append(token)
            } else {
                draftTokens.append(token)
            }
        }

        // Mark this emission point so the cadence throttle suppresses
        // re-encodes until another `partialEmissionIntervalSamples`
        // worth of audio has been appended.
        lastEmittedBufferCount = audioBuffer.count

        let result = currentResult()
        let encodeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        // Cadence telemetry (engine #118). `.debug` so production users
        // don't accumulate per-frame log records on disk; elevate with
        // `log config --subsystem live.yooz.engine --mode level:debug`
        // when investigating. Numeric fields marked `.public` for
        // consistency with the rest of the codebase and to ensure they
        // survive log archive redaction.
        logger.debug(
            """
            frame samples=\(samples.count, privacy: .public) \
            buffer=\(self.audioBuffer.count, privacy: .public) \
            encode_ms=\(encodeMs, format: .fixed(precision: 2), privacy: .public) \
            text_len=\(result.text.count, privacy: .public) \
            finalized_len=\(result.finalized.count, privacy: .public) \
            draft_len=\(result.draft.count, privacy: .public)
            """
        )
        return result
    }

    /// Finalize transcription
    public func finalize() -> ParakeetResult {
        let tokens = processAllAudio()
        let text = tokens.map(\.text).joined()
            .trimmingCharacters(in: .whitespaces)

        return ParakeetResult(
            text: text,
            finalized: text,
            draft: ""
        )
    }

    /// Finalize transcription with full token information including timestamps
    /// - Returns: TranscriptionResult with aligned tokens containing start/duration
    public func finalizeWithTimestamps() -> TranscriptionResult {
        TranscriptionResult(tokens: processAllAudio())
    }

    /// Reset all state
    public func reset() {
        audioBuffer = []
        finalizedTokens = []
        draftTokens = []
        lastEmittedBufferCount = 0
    }

    // MARK: - Private Methods

    private func currentResult() -> ParakeetResult {
        let finalizedText = finalizedTokens.map(\.text).joined()
            .trimmingCharacters(in: .whitespaces)
        let draftText = draftTokens.map(\.text).joined()
            .trimmingCharacters(in: .whitespaces)

        let fullText: String = if finalizedText.isEmpty {
            draftText
        } else if draftText.isEmpty {
            finalizedText
        } else {
            finalizedText + " " + draftText
        }

        return ParakeetResult(
            text: fullText.trimmingCharacters(in: .whitespaces),
            finalized: finalizedText,
            draft: draftText
        )
    }

    /// Process all audio and return aligned tokens (shared logic for finalize methods)
    private func processAllAudio() -> [AlignedToken] {
        guard !audioBuffer.isEmpty else {
            return finalizedTokens + draftTokens
        }

        // Add silence padding at the end (200ms) to give model context for final words
        let silencePadding = [Float](repeating: 0.0, count: sampleRate / 5)
        let paddedAudio = audioBuffer + silencePadding

        // Process ALL accumulated audio (batch mode)
        let mel = preprocessor.logMelSpectrogram(MLXArray(paddedAudio))

        guard mel.dim(1) > 0 else {
            return finalizedTokens + draftTokens
        }

        // Pad mel to subsampling factor
        let melLength = mel.dim(1)
        let remainder = melLength % subsamplingFactor
        let alignedMel: MLXArray
        if remainder == 0 {
            alignedMel = mel
        } else {
            let padSize = subsamplingFactor - remainder
            let padding = MLXArray.zeros([mel.dim(0), padSize, mel.dim(2)])
            alignedMel = concatenated([mel, padding], axis: 1)
        }

        guard alignedMel.dim(1) > 0 else {
            return finalizedTokens + draftTokens
        }

        let (features, _) = model.encoder(alignedMel)
        eval(features)

        return model.tdtDecode(
            features: features,
            length: features.dim(1)
        )
    }
}
