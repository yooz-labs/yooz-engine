import Foundation

// `AlignedToken`, `TranscriptionResult`, `STTBackendInfo`,
// `STTBackendsResponse`, and `STTSetBackendRequest` moved to
// `YoozEngineWire` (#225) — visible here via `YoozEngineClient`'s
// `WireReexport.swift`.

public enum STTLanguage: String, Codable, Sendable, CaseIterable {
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
}

// MARK: - STT Backend Picker (canonical pattern, second adopter)
//
// `STTBackendID` stays SDK-owned (a routing/domain identifier, not a plain
// wire DTO) mirroring the engine-side `STTBackendID` in
// `YoozEngine/STT/Engine/STTBackendID.swift` by rawValue contract.

/// Stable wire id for an STT backend. Mirrors engine-side
/// `STTBackendID`. Renaming a case is a major SDK bump.
///
/// Yooz Engine exposes four speech backends today:
/// - `.parakeet` — MLX Parakeet TDT (Latin/European languages, high accuracy,
///    ~600 MB runtime)
/// - `.fastConformer` — MLX FastConformer (Arabic, Persian; RTL scripts)
/// - `.appleSTT` — Apple's built-in STT (`SFSpeechRecognizer` on macOS 14-25,
///    `SpeechAnalyzer` on macOS 26+). Zero MLX footprint; the only backend
///    linked into `YoozEngineLite`.
/// - `.qwen3ASRPreview` — preview Qwen3-ASR backend (variant-gated).
///
/// Picker lives server-side; clients consult `STTClient.availableEngines()`
/// to discover which are linked into the running build variant. Requesting an
/// unbundled engine returns HTTP 501 `module_not_bundled`.
public enum STTBackendID: String, Codable, Sendable, CaseIterable {
    case parakeet
    case fastConformer = "fast_conformer"
    case appleSTT = "apple_stt"
    case qwen3ASRPreview = "qwen3_asr_preview"
}
