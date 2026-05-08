// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import os.log

#if canImport(Tokenizers)
import Tokenizers
#endif

#if canImport(Qwen3ASRMelFrontend)
import Qwen3ASRMelFrontend
#endif

private let logger = Logger(
    subsystem: "live.yooz.engine",
    category: "Qwen3ASRPipeline"
)

// MARK: - Public output types

/// Result of a single transcription pass.
public struct Qwen3ASRTranscription: Sendable, Equatable {
    /// Decoded text (without the `language X<asr_text>` prefix).
    public let text: String
    /// Auto-detected (or supplied) language label, e.g. `English`,
    /// `Persian`, `Arabic`. Lower-case names are normalized to the
    /// canonical model label when present in `supportLanguages`.
    public let language: String
    /// Generated token IDs (post the `<asr_text>` separator).
    public let generatedTokens: [Int]
    /// Number of audio tokens consumed from the encoder output.
    public let numAudioTokens: Int
    /// End-to-end wall time including encode + prefill + generation.
    public let totalSeconds: Double

    public init(
        text: String,
        language: String,
        generatedTokens: [Int],
        numAudioTokens: Int,
        totalSeconds: Double
    ) {
        self.text = text
        self.language = language
        self.generatedTokens = generatedTokens
        self.numAudioTokens = numAudioTokens
        self.totalSeconds = totalSeconds
    }
}

// MARK: - Pipeline

/// End-to-end Qwen3-ASR transcription pipeline in pure Swift / MLX.
///
/// Owns:
///   - the mel frontend
///   - the audio_tower encoder
///   - the Qwen3 text decoder
///   - the HuggingFace tokenizer (loaded via `swift-transformers`)
///
/// Hides every MLX detail behind two entry points:
///   - `load(from:)` — construct from a checkpoint directory
///   - `transcribe(pcm:language:maxNewTokens:)` — full forward pass
///
/// The engine HTTP wiring calls `transcribe(...)` from the
/// `/v1/stt/batch` handler without ever touching MLX directly.
///
/// `@MainActor` is *not* applied here — the pipeline is meant to live
/// inside a dedicated STT actor in the engine (the way `MLXLLMBackend`
/// does for the LLM module). Callers that need cross-thread access
/// must wrap the instance themselves.
public final class Qwen3ASRPipeline {
    public let config: Qwen3ASRFullConfig
    public let modelDirectory: URL

    private let melFrontend: MelFrontend
    private let encoder: Qwen3AudioEncoder
    private let decoder: Qwen3ASRTextDecoder

    /// Fully-qualified to disambiguate from the Parakeet module's
    /// `Tokenizer` struct (the YoozEngine target imports both).
    private let tokenizer: any Tokenizers.Tokenizer

    /// EOS token IDs the loop stops at. Hard-coded to match the
    /// upstream `mlx_audio` reference (`151645 = <|im_end|>`,
    /// `151643 = <|endoftext|>`).
    static let eosTokenIDs: Set<Int> = [151_645, 151_643]

    /// Convenience entry point: load every component from a single
    /// HuggingFace-style checkpoint directory containing
    /// `config.json`, `tokenizer*.json`, `model.safetensors`, and the
    /// preprocessor metadata.
    public static func load(
        from directory: URL
    ) async throws -> Qwen3ASRPipeline {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Qwen3ASRError.fileNotFound(configURL)
        }
        let fullConfig = try Qwen3ASRFullConfig.load(from: configURL)

        let weightsURL = directory.appendingPathComponent(
            "model.safetensors"
        )
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw Qwen3ASRError.fileNotFound(weightsURL)
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)

        // Mel frontend defaults to Whisper's 128-bin, 25 ms window /
        // 10 ms hop, identical to the published preprocessor_config.
        let mel = MelFrontend(config: .qwen3ASR)

        let pipeline = Qwen3ASRPipeline(
            config: fullConfig,
            modelDirectory: directory,
            melFrontend: mel,
            tokenizer: tokenizer
        )

        try pipeline.loadWeights(from: weightsURL)
        return pipeline
    }

    /// Designated initializer. Builds untrained encoder + decoder
    /// modules; weights are applied separately via `loadWeights`.
    public init(
        config: Qwen3ASRFullConfig,
        modelDirectory: URL,
        melFrontend: MelFrontend,
        tokenizer: any Tokenizers.Tokenizer
    ) {
        self.config = config
        self.modelDirectory = modelDirectory
        self.melFrontend = melFrontend
        self.encoder = Qwen3AudioEncoder(config.audio)
        self.decoder = Qwen3ASRTextDecoder(config.text)
        self.tokenizer = tokenizer
    }

    // MARK: - Weight loading

    /// Apply weights from a single safetensors file. Strips the
    /// `thinker.` prefix where present (some published checkpoints use
    /// it, others don't), routes `audio_tower.*` through the existing
    /// loader, and `model.*` into the decoder. Quantization is applied
    /// before the weight update so the decoder's Linear/Embedding
    /// layers are converted to their quantized variants in place.
    public func loadWeights(from url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Qwen3ASRError.fileNotFound(url)
        }

        let raw: [String: MLXArray]
        do {
            raw = try MLX.loadArrays(url: url)
        } catch {
            throw Qwen3ASRError.malformedSafetensors(
                url, String(describing: error)
            )
        }

        // 1) Strip optional `thinker.` prefix and split into audio /
        // text buckets.
        //
        // For tied-embedding checkpoints, `lm_head.*` is unused
        // (the decoder ties to `model.embed_tokens`). We feed those
        // keys into `textWeights` anyway and rely on the decoder's
        // `sanitize` step to drop them — keeping the bucket logic
        // straight rather than splitting a "drop-here vs drop-there"
        // decision across two places.
        var audioWeights: [String: MLXArray] = [:]
        var textWeights: [String: MLXArray] = [:]
        for (key, value) in raw {
            var k = key
            if k.hasPrefix("thinker.") {
                k = String(k.dropFirst("thinker.".count))
            }
            if k.hasPrefix("audio_tower.") {
                // Pass the full key through; the existing
                // `Qwen3SafetensorsLoader.applyAudioTowerWeights`
                // strips the `audio_tower.` prefix itself before the
                // parameter walk.
                audioWeights[k] = value
            } else if k.hasPrefix("model.") || k.hasPrefix("lm_head.") {
                // The decoder exposes `model` as a property; keeping
                // the original `model.*` (or `lm_head.*`) prefix lets
                // the MLXNN parameter walk line up with the
                // safetensors keys directly.
                textWeights[k] = value
            } else {
                // Unknown bucket — ignore but log so a future
                // checkpoint format change shows up in the engine log
                // rather than silently breaking parity.
                logger.notice(
                    "Qwen3ASRPipeline: ignoring unrecognized weight key \(k, privacy: .public)"
                )
            }
        }

        if audioWeights.isEmpty {
            throw Qwen3ASRError.noAudioTowerWeights(url)
        }

        // 2) Apply audio tower weights via the existing typed loader.
        try Qwen3SafetensorsLoader.applyAudioTowerWeights(
            audioWeights, to: encoder, sourceURL: url
        )

        // 3) Apply quantization to the text decoder before weight
        // update, mirroring MLXLMCommon.loadWeights. The filter only
        // returns a quantization tuple for layers that actually have
        // a `.scales` entry in the checkpoint — RMSNorms and other
        // non-quantizable modules return nil.
        let bits = config.quantBits
        let groupSize = config.quantGroupSize
        if let bits, let groupSize, bits > 0, groupSize > 0 {
            quantize(
                model: decoder,
                filter: { path, _ in
                    if textWeights["\(path).scales"] != nil {
                        return (
                            groupSize: groupSize,
                            bits: bits,
                            mode: QuantizationMode.affine
                        )
                    }
                    return nil
                }
            )
        }

        // 4) Drop tied lm_head weights, then apply.
        let sanitized = decoder.sanitize(weights: textWeights)
        let nested = ModuleParameters.unflattened(sanitized)
        try decoder.update(parameters: nested, verify: [.all])
        eval(decoder)
    }

    // MARK: - Public transcribe API

    /// Transcribe a 16 kHz mono PCM buffer.
    ///
    /// - Parameters:
    ///   - pcm: float samples in `[-1, 1]`. Must be 16 kHz mono.
    ///   - language: optional language hint (e.g. `English`, `Persian`,
    ///     `Arabic`). When `nil`, the model auto-detects via the
    ///     prefix `language X<asr_text>` it emits as the first tokens.
    ///   - maxNewTokens: cap on generated tokens (excludes the prompt).
    /// - Returns: text + language + token IDs.
    public func transcribe(
        pcm: [Float],
        language: String? = nil,
        maxNewTokens: Int = 8_192
    ) throws -> Qwen3ASRTranscription {
        let start = Date()

        guard !pcm.isEmpty else {
            throw Qwen3ASRError.invalidInput(
                "Qwen3ASRPipeline.transcribe: pcm buffer is empty"
            )
        }

        // 1) Mel features + attention mask -> audio_tower input.
        let mel = try melFrontend.computeFeatures(pcm: pcm, sampleRate: 16_000)
        let melArray = MLXArray(
            mel.features, [mel.numMelFilters, mel.numTotalFrames]
        )
        .expandedDimensions(axis: 0)  // (1, n_mels, n_frames)
        let attnMask = MLXArray(mel.attentionMask)
            .expandedDimensions(axis: 0)

        let audioFeatures = try encoder.forward(
            inputFeatures: melArray,
            featureAttentionMask: attnMask
        )
        eval(audioFeatures)

        let numAudioTokens = audioFeatures.dim(0)

        // 2) Build the prompt token ID sequence.
        let promptIDs = try buildPromptTokenIDs(
            numAudioTokens: numAudioTokens,
            language: language
        )

        // 3) Build the inputs_embeds tensor by looking up the prompt
        // and splicing the audio_tower hidden states into the
        // `<|audio_pad|>` slots.
        let promptArray = MLXArray(
            promptIDs.map { Int32($0) }
        ).expandedDimensions(axis: 0)
        let inputEmbeds = buildInputEmbeds(
            promptIDs: promptIDs,
            promptTensor: promptArray,
            audioFeatures: audioFeatures
        )

        // 4) Greedy decode loop. `newCache` comes from the
        // `KVCacheDimensionProvider` extension and lives in MLXLMCommon.
        let cache: [MLXLMCommon.KVCache] = decoder.newCache(parameters: nil)
        var logits = decoder.callAsEmbeddings(inputEmbeds, cache: cache)
        eval(logits)

        var generated: [Int] = []
        generated.reserveCapacity(min(maxNewTokens, 256))
        var nextToken = greedyArgmax(lastTokenLogits: logits)

        while generated.count < maxNewTokens {
            if Self.eosTokenIDs.contains(nextToken) {
                break
            }
            generated.append(nextToken)

            // Feed the sampled token back; embed it via the same
            // (possibly quantized) embedding table used for the prompt.
            let tokenArr = MLXArray([Int32(nextToken)])
                .expandedDimensions(axis: 0)
            let stepEmbed = decoder.model.embedTokens(tokenArr)
            logits = decoder.callAsEmbeddings(stepEmbed, cache: cache)
            eval(logits)
            nextToken = greedyArgmax(lastTokenLogits: logits)
        }

        // 5) Strip the language preamble: the model emits
        //    `language <Lang>` <|asr_text|> <real text...>`.
        let (resolvedLanguage, contentTokens) = extractLanguageAndContent(
            tokens: generated,
            providedLanguage: language
        )
        let text = tokenizer.decode(
            tokens: contentTokens,
            skipSpecialTokens: true
        )
        .trimmingCharacters(in: .whitespacesAndNewlines)

        let elapsed = Date().timeIntervalSince(start)
        let summary =
            "Qwen3ASRPipeline: \(pcm.count) samples, "
            + "\(String(format: "%.3f", elapsed))s, "
            + "\(generated.count) gen tokens, lang=\(resolvedLanguage)"
        logger.debug("\(summary, privacy: .public)")

        return Qwen3ASRTranscription(
            text: text,
            language: resolvedLanguage,
            generatedTokens: contentTokens,
            numAudioTokens: numAudioTokens,
            totalSeconds: elapsed
        )
    }

    // MARK: - Internals (visible to tests)

    /// Build the prompt token sequence the decoder expects:
    ///
    ///   `<|im_start|>system\n<|im_end|>\n`
    ///   `<|im_start|>user\n<|audio_start|>` [audio_pad × N] `<|audio_end|><|im_end|>\n`
    ///   `<|im_start|>assistant\n` [optional `language X<asr_text>` prefix]
    ///
    /// Returns the flat token ID list.
    func buildPromptTokenIDs(
        numAudioTokens: Int,
        language: String?
    ) throws -> [Int] {
        let assistantPrefix: String
        if let language {
            let canonical = canonicalLanguageName(language)
            assistantPrefix = "language \(canonical)<asr_text>"
        } else {
            assistantPrefix = ""
        }

        let audioPadRun = String(
            repeating: "<|audio_pad|>", count: numAudioTokens
        )
        // Match the Python reference byte-for-byte. The exact form is
        //   "<|im_start|>system\n<|im_end|>\n"
        //   "<|im_start|>user\n<|audio_start|>{pads}<|audio_end|><|im_end|>\n"
        //   "<|im_start|>assistant\n{prefix}"
        // with NO trailing newline after the prefix. Building it with
        // explicit `\n` interpolation avoids the trailing-newline trap
        // of multi-line Swift literals.
        let systemContent = ""  // matches reference: empty system
        let prompt =
            "<|im_start|>system\n\(systemContent)<|im_end|>\n"
            + "<|im_start|>user\n<|audio_start|>\(audioPadRun)"
            + "<|audio_end|><|im_end|>\n"
            + "<|im_start|>assistant\n\(assistantPrefix)"
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)

        // Sanity check the audio_pad count actually present.
        let padCount = tokens.filter { $0 == config.audioTokenId }.count
        guard padCount == numAudioTokens else {
            throw Qwen3ASRError.invalidInput(
                "Tokenizer emitted \(padCount) <|audio_pad|> tokens, "
                    + "expected \(numAudioTokens). Check the tokenizer "
                    + "config (special tokens must be registered)."
            )
        }
        return tokens
    }

    /// Splice audio_tower hidden states into the prompt embedding
    /// sequence at every `<|audio_pad|>` position. The audio features
    /// are cast to the embedding dtype before substitution to match
    /// the Python reference exactly (otherwise bf16 + f32 mixing
    /// produces a small parity drift in the first few tokens).
    func buildInputEmbeds(
        promptIDs: [Int],
        promptTensor: MLXArray,
        audioFeatures: MLXArray
    ) -> MLXArray {
        let baseEmbeds = decoder.model.embedTokens(promptTensor)
        let audioCast = audioFeatures.asType(baseEmbeds.dtype)

        // Find all audio_pad positions (Swift loop on host ints — the
        // prompt is short and this avoids a `.flatten().asType().asArray`
        // round-trip through MLX).
        let padIndices = promptIDs.enumerated().compactMap { idx, tok in
            tok == config.audioTokenId ? idx : nil
        }
        if padIndices.isEmpty {
            return baseEmbeds
        }

        // Replace each pad embedding by building a fresh tensor from
        // stacked rows. MLX slice-assign on `baseEmbeds` would mutate
        // the same buffer the embedding lookup returned; rebuilding
        // a fresh tensor keeps `baseEmbeds` immutable so tests that
        // snapshot intermediates see the unspliced source as-is.
        let seqLen = promptIDs.count
        let hiddenDim = baseEmbeds.dim(2)
        let flatBase = baseEmbeds.reshaped(seqLen, hiddenDim)

        var rows: [MLXArray] = []
        rows.reserveCapacity(seqLen)
        var audioCursor = 0
        for i in 0..<seqLen {
            if audioCursor < padIndices.count, padIndices[audioCursor] == i,
                audioCursor < audioCast.dim(0)
            {
                rows.append(audioCast[audioCursor])
                audioCursor += 1
            } else {
                rows.append(flatBase[i])
            }
        }
        let stacked = MLX.stacked(rows, axis: 0)
        return stacked.reshaped(1, seqLen, hiddenDim)
    }

    /// Pure host-side argmax over the last position of a logits
    /// tensor of shape `(batch, seqLen, vocab)`. The decoder's
    /// forward path always produces a non-empty logits tensor, so
    /// this never sees an empty argmax in practice; if it ever does,
    /// MLX's own `item(Int32.self)` precondition will surface a
    /// crash with the offending shape.
    func greedyArgmax(lastTokenLogits: MLXArray) -> Int {
        let lastSlice = lastTokenLogits[0, -1]  // (vocab,)
        let argmax = MLX.argMax(lastSlice, axis: -1)
        eval(argmax)
        return Int(argmax.item(Int32.self))
    }

    /// Strip the `language X<asr_text>` preamble Qwen3-ASR emits when
    /// `language: nil`. Returns `(language, contentTokens)`. When the
    /// caller supplied a language, we trust it and just slice off the
    /// `<asr_text>` separator if present.
    func extractLanguageAndContent(
        tokens: [Int],
        providedLanguage: String?
    ) -> (String, [Int]) {
        // Locate the <asr_text> separator (token id 151_704 in the
        // canonical tokenizer; resolved at runtime to stay correct
        // across tokenizer revisions).
        let asrTextId = tokenizer.convertTokenToId("<asr_text>")

        if let asrTextId, let cutPoint = tokens.firstIndex(of: asrTextId) {
            let preamble = Array(tokens[..<cutPoint])
            let body = Array(tokens[(cutPoint + 1)...])
            // Decode the preamble; extract the trailing language label
            // which sits after the literal "language ".
            let preambleText = tokenizer.decode(
                tokens: preamble, skipSpecialTokens: true
            )
            let language: String
            if let trimmed = preambleText
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .components(separatedBy: "language ")
                .last,
                !trimmed.isEmpty
            {
                language = canonicalLanguageName(trimmed)
            } else if let providedLanguage {
                language = canonicalLanguageName(providedLanguage)
            } else {
                language = "Unknown"
            }
            return (language, body)
        }

        // No <asr_text> seen — caller-supplied language wins;
        // otherwise fall back to "Unknown".
        let lang =
            providedLanguage.map { canonicalLanguageName($0) } ?? "Unknown"
        return (lang, tokens)
    }

    private func canonicalLanguageName(_ raw: String) -> String {
        let lowercased = raw.lowercased()
        for lang in config.supportLanguages where lang.lowercased() == lowercased
        {
            return lang
        }
        return raw
    }
}
