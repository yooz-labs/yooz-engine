// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXNN

/// Production loader for the `audio_tower.*` slice of a Qwen3-ASR
/// safetensors checkpoint.
///
/// `MLXNN`'s `Module.update(parameters:verify:)` reports parameter
/// mismatches behind an opaque error string and gives the caller no
/// way to distinguish "missing tensor" from "wrong shape" from
/// "wrong dtype" — three distinct production failure modes.
///
/// This loader pre-flights every tensor against the encoder's
/// declared parameter map and emits a typed `Qwen3ASRError`
/// describing the exact mismatch, so the encoder ↔ decoder bridge
/// can map cases directly to typed API responses without re-parsing
/// strings.
public enum Qwen3SafetensorsLoader {

    /// Load the audio_tower slice from `url` into `encoder`.
    ///
    /// Performs, in order:
    ///   1. file existence check
    ///   2. safetensors decode (wrapped errors → `malformedSafetensors`)
    ///   3. `audio_tower.` prefix extraction
    ///   4. typed key/shape/dtype pre-flight
    ///   5. `MLXNN.Module.update(parameters:verify:)` apply
    ///   6. `eval(encoder)` to materialize weights up front
    public static func loadAudioTower(
        from url: URL,
        into encoder: Qwen3AudioEncoder
    ) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Qwen3ASRError.fileNotFound(url)
        }

        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: url)
        } catch {
            // `MLX.loadArrays` wraps every header parse / IO failure
            // in an opaque error; re-wrap with the original string so
            // tests can assert it surfaced something actionable.
            throw Qwen3ASRError.malformedSafetensors(
                url, String(describing: error)
            )
        }

        try applyAudioTowerWeights(raw, to: encoder, sourceURL: url)
    }

    /// Apply pre-loaded arrays. Useful for tests that want to feed
    /// a deterministic in-memory dictionary without touching disk.
    /// `sourceURL` is used only for error messages.
    public static func applyAudioTowerWeights(
        _ raw: [String: MLXArray],
        to encoder: Qwen3AudioEncoder,
        sourceURL: URL? = nil
    ) throws {
        let originURL = sourceURL ?? URL(fileURLWithPath: "<memory>")

        // 1) Strip `audio_tower.` prefix; reject checkpoints that
        //    don't contain any audio-tower keys (e.g. text decoder
        //    only).
        var stripped: [String: MLXArray] = [:]
        stripped.reserveCapacity(raw.count)
        for (key, value) in raw {
            if key.hasPrefix("audio_tower.") {
                let trimmed = String(
                    key.dropFirst("audio_tower.".count)
                )
                stripped[trimmed] = value
            }
        }
        if stripped.isEmpty {
            throw Qwen3ASRError.noAudioTowerWeights(originURL)
        }

        // 2) Build the expected (key -> shape, dtype) map from the
        //    encoder's own parameter graph and pre-flight every
        //    tensor before handing it to MLXNN.
        let expectations = expectedParameterMap(of: encoder)

        for (expectedKey, info) in expectations {
            guard let provided = stripped[expectedKey] else {
                throw Qwen3ASRError.missingTensor(expectedKey)
            }
            if provided.shape != info.shape {
                throw Qwen3ASRError.shapeMismatch(
                    key: expectedKey,
                    expected: info.shape,
                    actual: provided.shape
                )
            }
            // Accept the canonical bf16 dtype, or any of the
            // float-family dtypes that round-trip cleanly into
            // bf16 during MLXNN's parameter assignment. Anything
            // else (int32, complex, etc.) is rejected up front.
            if !isAcceptableFloatDType(provided.dtype) {
                throw Qwen3ASRError.dtypeMismatch(
                    key: expectedKey,
                    expected: "float family (bf16/f16/f32)",
                    actual: String(describing: provided.dtype)
                )
            }
        }

        // Reject extra `audio_tower.*` keys the encoder does not
        // declare. MLXNN's `update(verify: [.all])` would catch
        // this further downstream, but it surfaces a generic
        // string error; we want the caller to see the offending
        // key directly via the typed enum.
        let expectedKeys = Set(expectations.keys)
        for providedKey in stripped.keys where !expectedKeys.contains(
            providedKey
        ) {
            throw Qwen3ASRError.unexpectedTensor(providedKey)
        }

        // 3) Apply weights via the standard MLXNN path.
        let nested = ModuleParameters.unflattened(stripped)
        try encoder.update(parameters: nested, verify: [.all])
        eval(encoder)
    }

    /// Convenience: build an encoder from a config and immediately
    /// apply weights from a checkpoint at `url`. Performs config
    /// validation up front via the throwing initializer.
    public static func encoder(
        from url: URL, config: Qwen3ASRConfig
    ) throws -> Qwen3AudioEncoder {
        let model = try Qwen3AudioEncoder(checked: config)
        try loadAudioTower(from: url, into: model)
        return model
    }

    // MARK: - Helpers

    /// Snapshot of an encoder parameter for shape/dtype checking.
    private struct ParameterInfo {
        let shape: [Int]
        let dtype: DType
    }

    private static func expectedParameterMap(
        of encoder: Qwen3AudioEncoder
    ) -> [String: ParameterInfo] {
        var map: [String: ParameterInfo] = [:]
        for (name, value) in encoder.parameters().flattened() {
            map[name] = ParameterInfo(
                shape: value.shape, dtype: value.dtype
            )
        }
        return map
    }

    private static func isAcceptableFloatDType(_ dtype: DType) -> Bool {
        switch dtype {
        case .bfloat16, .float16, .float32:
            return true
        default:
            return false
        }
    }
}
