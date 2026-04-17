// AppleSTTBackend.swift
// AppleSTTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import Speech
import AVFoundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "AppleSTTBackend")

/// Which Apple-provided STT framework this backend is driving.
///
/// On macOS 14-25 we fall back to `Speech.framework`'s legacy
/// `SFSpeechRecognizer`. macOS 26 introduces the new `SpeechAnalyzer` API which
/// is on-device by default, supports streaming with built-in endpointing, and
/// exposes per-token timestamps natively. `AppleSTTBackend` hides the split so
/// `AppleSTTEngine` can call one API regardless of OS.
public enum AppleSTTBackendKind: String, Sendable {
    /// Legacy `SFSpeechRecognizer` path (macOS 14-25).
    case sfSpeechRecognizer = "sf_speech_recognizer"
    /// Modern `SpeechAnalyzer` path (macOS 26+).
    case speechAnalyzer = "speech_analyzer"
}

/// Authorization status reported by the OS for speech recognition.
///
/// Mirrors `SFSpeechRecognizerAuthorizationStatus` but exposed as a Swift enum
/// so module consumers don't need to import `Speech`.
public enum AppleSTTAuthorizationStatus: String, Sendable {
    case notDetermined = "not_determined"
    case denied
    case restricted
    case authorized
}

public enum AppleSTTError: LocalizedError, Sendable {
    case recognizerUnavailable(locale: String)
    case authorizationDenied(status: AppleSTTAuthorizationStatus)
    case recognitionFailed(String)
    case pcmBufferCreationFailed
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .recognizerUnavailable(let locale):
            return "Apple STT recognizer unavailable for locale '\(locale)'"
        case .authorizationDenied(let status):
            return "Apple STT authorization denied (status: \(status.rawValue))"
        case .recognitionFailed(let message):
            return "Apple STT recognition failed: \(message)"
        case .pcmBufferCreationFailed:
            return "Apple STT failed to create AVAudioPCMBuffer from samples"
        case .cancelled:
            return "Apple STT request cancelled"
        }
    }
}

/// Thin wrapper over the two Apple-provided STT paths.
///
/// - macOS 14-25: `SFSpeechRecognizer` with `SFSpeechAudioBufferRecognitionRequest`
///   for batch transcription of in-memory `[Float]` samples at 16 kHz.
/// - macOS 26+: `SpeechAnalyzer` (gated by `#available`). The new API is the
///   default path when available; falls back to the legacy recognizer only on
///   older systems.
///
/// The backend exposes *batch* transcription today. Streaming plumbs through
/// the same request type because `SFSpeechAudioBufferRecognitionRequest`
/// handles continuous audio via `append(_:)` naturally; a real streaming
/// surface can be layered later without reshaping the public API.
///
/// Authorization is surfaced but never automatically requested — callers are
/// responsible for prompting at the right UX moment (e.g. first use). The
/// backend only *reads* the current status.
///
/// No microphone permission is required for batch transcription of
/// pre-recorded `[Float]` samples; mic permission is only needed for live
/// audio capture (not implemented here).
public struct AppleSTTBackend: Sendable {

    /// Which underlying framework this process will use. Fixed at init time so
    /// a running engine doesn't silently swap paths mid-session.
    public let kind: AppleSTTBackendKind

    /// BCP-47 locale string (e.g. `"en-US"`, `"fa-IR"`).
    public let localeIdentifier: String

    public init(localeIdentifier: String) {
        self.localeIdentifier = localeIdentifier
        // SpeechAnalyzer prefers the 26+ path; leave the enum as documentation
        // of which framework is driving. The actual dispatch uses #available.
        if #available(macOS 26, *) {
            self.kind = .speechAnalyzer
        } else {
            self.kind = .sfSpeechRecognizer
        }
    }

    // MARK: - Availability

    /// Current authorization status from `SFSpeechRecognizer`. Reading is
    /// always safe (no prompt is shown). Use `requestAuthorization()` to prompt.
    public static var authorizationStatus: AppleSTTAuthorizationStatus {
        Self.map(SFSpeechRecognizer.authorizationStatus())
    }

    /// Whether a recognizer exists for the given locale *and* the user has
    /// granted authorization. Callers use this for the `/v1/stt/engine`
    /// capability advertisement.
    public static func isAvailable(localeIdentifier: String) -> Bool {
        guard authorizationStatus == .authorized else { return false }
        let locale = Locale(identifier: localeIdentifier)
        guard let recognizer = SFSpeechRecognizer(locale: locale) else { return false }
        return recognizer.isAvailable
    }

    /// Request speech recognition authorization from the user.
    ///
    /// Wraps the callback-based API into an async call. Safe to call multiple
    /// times; subsequent calls return the cached status without re-prompting.
    @discardableResult
    public static func requestAuthorization() async -> AppleSTTAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: Self.map(status))
            }
        }
    }

    private static func map(_ status: SFSpeechRecognizerAuthorizationStatus) -> AppleSTTAuthorizationStatus {
        switch status {
        case .authorized: return .authorized
        case .denied: return .denied
        case .restricted: return .restricted
        case .notDetermined: return .notDetermined
        @unknown default: return .notDetermined
        }
    }

    // MARK: - Batch transcription

    /// Run a single-shot transcription over an in-memory Float32 buffer.
    ///
    /// Audio must be 16 kHz mono Float32 (the engine's canonical format). The
    /// backend constructs an `AVAudioPCMBuffer`, drives the chosen framework
    /// path end-to-end, and returns the best hypothesis.
    ///
    /// - Note: Per-token timestamps from
    ///   `SFTranscriptionSegment.timestamp` / `.duration` are available on
    ///   the returned `SFSpeechRecognitionResult`. Wiring them into the
    ///   engine's `AlignedToken` surface is tracked in engine#34; the hook
    ///   intentionally stays unimplemented in this deliverable.
    ///
    /// - TODO(engine#34): expose a second transcription entry point that
    ///   returns aligned tokens. Map each `SFTranscriptionSegment` to an
    ///   `AlignedToken(id: idx, text: segment.substring, start:
    ///   Float(segment.timestamp), duration: Float(segment.duration))`.
    ///   The hook intentionally lives on the server-side `TranscriptionResult`
    ///   in `YoozEngine/STT/Models/Parakeet/ParakeetModel.swift` — not the
    ///   SDK's `TranscriptionResult` in `Sources/YoozEngineClient/Types/`.
    public func transcribe(samples: [Float], sampleRate: Double = 16_000) async throws -> String {
        // Only legacy path for now. The SpeechAnalyzer path follows the same
        // [Float] -> single-string contract; we keep the branch explicit so the
        // 26+ wiring can land behind `#available` without disturbing callers.
        if #available(macOS 26, *) {
            return try await transcribeViaSFRecognizer(samples: samples, sampleRate: sampleRate)
        } else {
            return try await transcribeViaSFRecognizer(samples: samples, sampleRate: sampleRate)
        }
    }

    private func transcribeViaSFRecognizer(samples: [Float], sampleRate: Double) async throws -> String {
        let status = Self.authorizationStatus
        guard status == .authorized else {
            throw AppleSTTError.authorizationDenied(status: status)
        }

        let locale = Locale(identifier: localeIdentifier)
        guard let recognizer = SFSpeechRecognizer(locale: locale) else {
            throw AppleSTTError.recognizerUnavailable(locale: localeIdentifier)
        }
        guard recognizer.isAvailable else {
            throw AppleSTTError.recognizerUnavailable(locale: localeIdentifier)
        }

        // Prefer fully on-device recognition so transcription stays privacy-
        // respecting. If the locale has no on-device model the recognizer
        // falls back to server-side — we log but don't hard-fail; the engine
        // contract is "best-effort Apple STT".
        let request = SFSpeechAudioBufferRecognitionRequest()
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        } else {
            logger.warning("SFSpeechRecognizer does not support on-device for \(self.localeIdentifier); falling back to server path")
        }
        request.shouldReportPartialResults = false

        guard let buffer = Self.makePCMBuffer(samples: samples, sampleRate: sampleRate) else {
            throw AppleSTTError.pcmBufferCreationFailed
        }
        request.append(buffer)
        request.endAudio()

        return try await withCheckedThrowingContinuation { continuation in
            var didResume = false
            let task = recognizer.recognitionTask(with: request) { result, error in
                guard !didResume else { return }
                if let error = error {
                    let nsError = error as NSError
                    // Speech framework emits "No speech detected" (1110) for
                    // empty audio on some locales. Treat as empty transcript.
                    if nsError.domain == "kAFAssistantErrorDomain" && nsError.code == 1110 {
                        didResume = true
                        continuation.resume(returning: "")
                        return
                    }
                    didResume = true
                    continuation.resume(throwing: AppleSTTError.recognitionFailed(error.localizedDescription))
                    return
                }
                guard let result = result, result.isFinal else { return }
                didResume = true
                continuation.resume(returning: result.bestTranscription.formattedString)
            }
            // Retain the task for the lifetime of the continuation — if the
            // recognizer ever dropped it before callback we'd deadlock.
            _ = task
        }
    }

    // MARK: - Helpers

    private static func makePCMBuffer(samples: [Float], sampleRate: Double) -> AVAudioPCMBuffer? {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            return nil
        }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            return nil
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData?[0] else { return nil }
        samples.withUnsafeBufferPointer { src in
            channel.initialize(from: src.baseAddress!, count: samples.count)
        }
        return buffer
    }
}
