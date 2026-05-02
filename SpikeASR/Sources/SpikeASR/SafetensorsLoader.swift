import Foundation
import MLX
import MLXNN

/// Errors surfaced while loading the audio_tower slice of a Qwen3-ASR
/// safetensors checkpoint.
public enum SpikeLoaderError: Error, CustomStringConvertible {
    case fileNotFound(URL)
    case missingTensor(String)
    case shapeMismatch(name: String, expected: [Int], got: [Int])
    case noAudioTowerWeights(URL)

    public var description: String {
        switch self {
        case .fileNotFound(let url):
            return "Safetensors file not found: \(url.path)"
        case .missingTensor(let name):
            return "Missing tensor in checkpoint: \(name)"
        case .shapeMismatch(let name, let expected, let got):
            return
                "Shape mismatch for \(name): expected \(expected), got \(got)"
        case .noAudioTowerWeights(let url):
            return
                "No 'audio_tower.*' tensors found in \(url.lastPathComponent)"
        }
    }
}

public enum SpikeLoader {

    /// Load the audio_tower slice of a Qwen3-ASR checkpoint, drop the
    /// `audio_tower.` prefix to match `AudioEncoder`'s parameter
    /// names, and apply weights to a freshly-constructed encoder.
    ///
    /// The 8-bit Qwen3-ASR checkpoint stores the audio encoder as
    /// plain bf16 (only `model.*` is quantized), so this is a direct
    /// round-trip — no quantization plumbing required.
    public static func loadAudioTower(
        from url: URL,
        into encoder: AudioEncoder
    ) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SpikeLoaderError.fileNotFound(url)
        }
        let raw = try MLX.loadArrays(url: url)
        return try applyAudioTowerWeights(raw, to: encoder, sourceURL: url)
    }

    /// Same as `loadAudioTower(from:into:)` but takes pre-loaded
    /// arrays; useful for tests that build a deterministic in-memory
    /// dictionary without touching disk.
    public static func applyAudioTowerWeights(
        _ raw: [String: MLXArray],
        to encoder: AudioEncoder,
        sourceURL: URL? = nil
    ) throws {
        var stripped: [String: MLXArray] = [:]
        for (key, value) in raw {
            if key.hasPrefix("audio_tower.") {
                let trimmed = String(key.dropFirst("audio_tower.".count))
                stripped[trimmed] = value
            }
        }
        if stripped.isEmpty {
            throw SpikeLoaderError.noAudioTowerWeights(
                sourceURL ?? URL(fileURLWithPath: "<memory>")
            )
        }
        let nested = ModuleParameters.unflattened(stripped)
        try encoder.update(parameters: nested, verify: [.all])
        eval(encoder)
    }

    /// Convenience: build an encoder from a config and immediately
    /// apply the weights from a checkpoint.
    public static func encoder(
        from url: URL, config: AudioEncoderConfig
    ) throws -> AudioEncoder {
        let model = AudioEncoder(config)
        try loadAudioTower(from: url, into: model)
        return model
    }
}
