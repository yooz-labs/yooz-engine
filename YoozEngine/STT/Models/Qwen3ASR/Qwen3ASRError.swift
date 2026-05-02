// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Errors surfaced by the Qwen3-ASR audio encoder pipeline.
///
/// Every error path a caller may encounter — config validation,
/// weight loading, malformed safetensors, runtime input shape — funnels
/// through this type so the encoder ↔ decoder bridge and the engine
/// HTTP layer can map cases to typed API errors without re-parsing
/// strings.
public enum Qwen3ASRError: Error, Equatable, CustomStringConvertible {
    /// `Qwen3ASRConfig.validate()` rejected the supplied config.
    case invalidConfig(String)
    /// Safetensors checkpoint not on disk at the supplied URL.
    case fileNotFound(URL)
    /// Loaded checkpoint contains no keys with the `audio_tower.`
    /// prefix. Either the file is the wrong slice (e.g. text decoder
    /// only) or has been re-keyed.
    case noAudioTowerWeights(URL)
    /// Required `audio_tower.<key>` key is missing from the loaded
    /// checkpoint after stripping the prefix.
    case missingTensor(String)
    /// Tensor `key` shape disagreed with the encoder's parameter
    /// shape. `expected` and `actual` are dumped verbatim so the
    /// message points at the offending shape pair.
    case shapeMismatch(key: String, expected: [Int], actual: [Int])
    /// Tensor `key` dtype disagreed with the encoder's parameter
    /// dtype. The reference checkpoint stores `audio_tower.*` as
    /// `bfloat16` (or `float16`/`float32` after cast); anything else
    /// is rejected up front rather than silently up-cast.
    case dtypeMismatch(key: String, expected: String, actual: String)
    /// Checkpoint contained `audio_tower.*` keys the encoder does
    /// not declare. Surfaced rather than silently dropped because
    /// it indicates the checkpoint format diverged from the
    /// encoder schema; the bridge layer should know.
    case unexpectedTensor(String)
    /// Underlying `MLX.loadArrays` (safetensors header) failed.
    /// Wrapped as `String` so any provider-specific error surface
    /// (truncated header, unknown dtype tag, etc.) is readable.
    case malformedSafetensors(URL, String)
    /// The forward pass was called with an input the encoder cannot
    /// process (zero-length frames, batch == 0, etc.).
    case invalidInput(String)
    /// `Qwen3ASRBackend.transcribe` was called before the pipeline
    /// was loaded via `ensureLoaded(modelDir:)`.
    case pipelineNotLoaded
    /// First-run model fetch could not contact the manifest endpoint.
    /// The wrapped string carries the underlying transport error.
    case fetchFailed(String)
    /// Downloaded artifact did not match the expected size or hash.
    case fetchValidationFailed(String)

    public var description: String {
        switch self {
        case .invalidConfig(let detail):
            return "Qwen3ASRError.invalidConfig: \(detail)"
        case .fileNotFound(let url):
            return "Qwen3ASRError.fileNotFound: \(url.path)"
        case .noAudioTowerWeights(let url):
            return
                "Qwen3ASRError.noAudioTowerWeights: no 'audio_tower.*' "
                + "tensors in \(url.lastPathComponent)"
        case .missingTensor(let key):
            return "Qwen3ASRError.missingTensor: \(key)"
        case .shapeMismatch(let key, let expected, let actual):
            return
                "Qwen3ASRError.shapeMismatch: \(key) expected "
                + "\(expected) got \(actual)"
        case .dtypeMismatch(let key, let expected, let actual):
            return
                "Qwen3ASRError.dtypeMismatch: \(key) expected "
                + "\(expected) got \(actual)"
        case .unexpectedTensor(let key):
            return "Qwen3ASRError.unexpectedTensor: \(key)"
        case .malformedSafetensors(let url, let detail):
            return
                "Qwen3ASRError.malformedSafetensors(\(url.path)): "
                + detail
        case .invalidInput(let detail):
            return "Qwen3ASRError.invalidInput: \(detail)"
        case .pipelineNotLoaded:
            return
                "Qwen3ASRError.pipelineNotLoaded: call "
                + "Qwen3ASRBackend.ensureLoaded(modelDir:) first"
        case .fetchFailed(let detail):
            return "Qwen3ASRError.fetchFailed: \(detail)"
        case .fetchValidationFailed(let detail):
            return "Qwen3ASRError.fetchValidationFailed: \(detail)"
        }
    }
}
