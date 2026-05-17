// AppleSTTEngine.swift
// AppleSTTModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "AppleSTTEngine")

/// Language selected for the Apple STT backend.
///
/// Kept intentionally minimal — the Apple frameworks accept any BCP-47 locale
/// the OS ships a recognizer for. This enum mirrors `STTLanguage` entries so
/// callers can pass the same raw codes through the server layer, but the
/// `Locale` we hand to `SFSpeechRecognizer` is built from `bcp47` directly.
public enum AppleSTTLanguage: String, Sendable, CaseIterable {
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case dutch = "nl"
    case polish = "pl"
    case russian = "ru"
    case ukrainian = "uk"
    case arabic = "ar"
    case persian = "fa"
    case hebrew = "he"
    case chinese = "zh"
    case japanese = "ja"
    case korean = "ko"
    case cantonese = "yue"

    /// BCP-47 identifier passed to `Locale`. Uses the common region where
    /// Apple ships recognizers; callers can still override by constructing
    /// their own locale string and handing it directly to `AppleSTTBackend`.
    public var bcp47: String {
        switch self {
        case .english: return "en-US"
        case .spanish: return "es-ES"
        case .french: return "fr-FR"
        case .german: return "de-DE"
        case .italian: return "it-IT"
        case .portuguese: return "pt-BR"
        case .dutch: return "nl-NL"
        case .polish: return "pl-PL"
        case .russian: return "ru-RU"
        case .ukrainian: return "uk-UA"
        case .arabic: return "ar-SA"
        case .persian: return "fa-IR"
        case .hebrew: return "he-IL"
        case .chinese: return "zh-CN"
        case .japanese: return "ja-JP"
        case .korean: return "ko-KR"
        case .cantonese: return "yue-CN"
        }
    }

    public static func from(rawCode: String) -> AppleSTTLanguage? {
        AppleSTTLanguage(rawValue: rawCode.lowercased())
    }
}

/// Speech-to-text engine backed by Apple's on-device frameworks.
///
/// Shape mirrors `YoozSTTEngine` (the MLX backend): a process-wide singleton
/// with explicit load (`start(language:)`), an `isReady` flag, and
/// batch + streaming entry points. The key contract difference vs. the MLX
/// engine is that Apple STT has *built-in* voice activity detection — the
/// `/v1/stt/engine` capability response advertises this so clients can skip
/// a parallel VAD pipeline when this backend is active.
///
/// No model weights ship in the app bundle for this engine; load is a
/// cheap capability probe against the OS. That's what lets
/// `YoozEngineLite` drop the MLX runtime entirely while still serving
/// `/v1/stt/*`.
public actor AppleSTTEngine {

    // MARK: - Singleton

    public static let shared = AppleSTTEngine()

    // MARK: - State

    public private(set) var currentLanguage: AppleSTTLanguage = .english
    public private(set) var isLoaded: Bool = false
    public private(set) var isStreaming: Bool = false

    /// The active backend configuration. `nil` until `start(language:)` is
    /// called successfully; future `transcribe` calls without a prior start
    /// return an error, matching the MLX engine's not-ready semantics.
    private var backend: AppleSTTBackend?

    // MARK: - Init

    private init() {}

    // MARK: - Capability

    /// `true` when the OS reports a recognizer for `currentLanguage` and the
    /// user has authorized speech recognition. Used by `AIModule.isReady`.
    public var isReady: Bool {
        guard isLoaded, let backend else { return false }
        return AppleSTTBackend.isAvailable(localeIdentifier: backend.localeIdentifier)
    }

    /// `true` — Apple STT performs its own endpointing. This is the signal
    /// thin clients read via `STTClient.hasBuiltInVAD()` to decide whether to
    /// run their own VAD pipeline on top of the engine.
    public static let hasBuiltInVAD: Bool = true

    /// Current authorization status as reported by the OS.
    public static var authorizationStatus: AppleSTTAuthorizationStatus {
        AppleSTTBackend.authorizationStatus
    }

    /// Which underlying framework the active backend is wired to. `nil`
    /// before `start(language:)`.
    public var backendKind: AppleSTTBackendKind? {
        backend?.kind
    }

    // MARK: - Lifecycle

    /// Prepare the engine for a given language. Idempotent — calling with the
    /// same language after a successful load is a no-op.
    ///
    /// The call performs no network I/O and no weight loading: Apple STT's
    /// models live in the OS. This "start" exists to validate that a
    /// recognizer exists for the requested locale and that the user has
    /// granted authorization.
    public func start(language: AppleSTTLanguage = .english) async throws {
        if isLoaded, currentLanguage == language, backend != nil {
            return
        }
        // Reading status is safe; we don't trigger an unsolicited prompt.
        let status = AppleSTTBackend.authorizationStatus
        guard status == .authorized else {
            throw AppleSTTError.authorizationDenied(status: status)
        }

        let candidate = AppleSTTBackend(localeIdentifier: language.bcp47)
        guard AppleSTTBackend.isAvailable(localeIdentifier: language.bcp47) else {
            throw AppleSTTError.recognizerUnavailable(locale: language.bcp47)
        }
        self.backend = candidate
        self.currentLanguage = language
        self.isLoaded = true
        logger.info("AppleSTTEngine ready: language=\(language.rawValue, privacy: .public) kind=\(candidate.kind.rawValue, privacy: .public)")
    }

    /// Release the backend. Subsequent transcribe calls will throw
    /// `AppleSTTError.recognitionFailed` until `start` is called again.
    public func stop() {
        backend = nil
        isLoaded = false
        isStreaming = false
    }

    /// Per-recording-session reset (engine issue #114). Today's engine is
    /// batch-only (`startStream` throws), so there is no streaming buffer to
    /// drop. We still conform: when the real streaming path lands behind
    /// `startStream`, recognition-task teardown hooks in here and the
    /// `/v1/session/begin` + `/v1/session/end` fan-out keeps working with
    /// zero new wiring.
    ///
    /// Defensively flips `isStreaming` to `false` so a session boundary is
    /// always a clean slate; `backend` + `isLoaded` are preserved because
    /// "loaded" here means "we hold an authorized recognizer for a locale",
    /// which is a long-lived capability, not per-recording state.
    public func resetForNewSession() async {
        isStreaming = false
    }

    // MARK: - Batch

    /// Transcribe an in-memory audio buffer. Parity with
    /// `YoozSTTEngine.batchTranscribe(samples:mode:)` — returns best-effort
    /// text and empty string on recognizer-emitted "no speech detected".
    public func batchTranscribe(samples: [Float]) async throws -> String {
        guard let backend else {
            throw AppleSTTError.recognitionFailed("engine not started; call start(language:) first")
        }
        return try await backend.transcribe(samples: samples)
    }

    /// Transcribe with per-token alignment. Parity with
    /// `YoozSTTEngine.batchTranscribeAligned(samples:mode:)` for the Apple
    /// backend; each `SFTranscriptionSegment` becomes one `AppleAlignedToken`
    /// with seconds-from-buffer-start timestamps.
    ///
    /// Backs `/v1/stt/batch?aligned=true` when Apple STT is the active
    /// backend (engine#34). Returns `AppleAlignedTranscription.empty` on
    /// recognizer-emitted "no speech detected" so callers can treat silent
    /// audio uniformly with the text-only path.
    public func batchTranscribeAligned(samples: [Float]) async throws -> AppleAlignedTranscription {
        guard let backend else {
            throw AppleSTTError.recognitionFailed("engine not started; call start(language:) first")
        }
        return try await backend.transcribeAligned(samples: samples)
    }

    /// Streaming entry point. Not implemented in the initial deliverable; the
    /// server layer returns an error when called. `SFSpeechAudioBufferRecognitionRequest`
    /// supports `append(_:)` and partial results, so a real streaming path
    /// lands behind this same method signature in a follow-up without
    /// reshaping the public API.
    public func startStream() async throws {
        throw AppleSTTError.recognitionFailed("AppleSTTEngine streaming is not implemented; use batchTranscribe")
    }

    /// Cancel whatever in-flight work the engine is tracking. Called by the
    /// API server when a client issues `POST /v1/stt/engine` while a stream is
    /// active. Safe to call concurrently with normal transcription — the
    /// underlying recognition task is retained by the continuation in
    /// `AppleSTTBackend.transcribe(samples:)` and isn't exposed here, so this
    /// is currently a state-reset only.
    public func cancelInFlight() {
        isStreaming = false
    }
}
