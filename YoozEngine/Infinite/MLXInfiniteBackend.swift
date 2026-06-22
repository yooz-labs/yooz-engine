// MLXInfiniteBackend.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import EngineCore
import Foundation
import os.log

#if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
import MLX
import MLXLMCommon
import MLXHuggingFace
import Tokenizers
import HuggingFace
#endif

private let mlxInfiniteLogger = Logger(
    subsystem: "live.yooz.engine",
    category: "MLXInfiniteBackend"
)

/// Outcome of one native-context generation.
public struct InfiniteGenerationResult: Sendable {
    public let text: String
    public let tokenCount: Int
    public let decodeTokensPerSecond: Double
    public let finishReason: String

    public init(
        text: String,
        tokenCount: Int,
        decodeTokensPerSecond: Double,
        finishReason: String
    ) {
        self.text = text
        self.tokenCount = tokenCount
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.finishReason = finishReason
    }
}

/// Real MLX-Swift backend for the InfiniteModule native-context path.
///
/// Loads a model whose architecture the `mlx-swift-lm` fork supports
/// (`qwen3_5_moe` and both Gemma4 rows — 26B-A4B #184 and the E4B OptiQ-4bit
/// build #186 — verified vs Python mlx-lm) and runs generation with a fresh
/// per-call KV cache.
/// Per-session KV reuse
/// (append-prefill / generate-decode) is a follow-up optimization within #182;
/// rebuilding the prefill per call is correct, just not the long-context
/// optimum. Reuses the proven load/generate pattern from `MLXLLMBackend`.
public actor MLXInfiniteBackend {
    public nonisolated let selection: InfiniteModelSelection

    #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
    private let container: ModelContainer

    private init(selection: InfiniteModelSelection, container: ModelContainer) {
        self.selection = selection
        self.container = container
    }
    #else
    private init(selection: InfiniteModelSelection) {
        self.selection = selection
    }
    #endif

    /// Load the model's weights (revision-pinned) into a `ModelContainer`.
    /// First-run downloads stream into `~/.cache/huggingface/hub/`.
    public static func load(
        _ descriptor: InfiniteBackendDescriptor
    ) async throws -> MLXInfiniteBackend {
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        guard let repository = descriptor.repository else {
            throw InfiniteError.modelSetFailed(
                "model \(descriptor.selection.rawValue) has no model repository to load"
            )
        }
        let repoRef = "\(repository.id)@\(repository.revision)"
        mlxInfiniteLogger.info(
            "Loading Infinite \(descriptor.selection.rawValue, privacy: .public) from \(repoRef, privacy: .public)"
        )
        do {
            let configuration = ModelConfiguration(
                id: repository.id,
                revision: repository.revision
            )
            let container = try await loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: { _ in }
            )
            return MLXInfiniteBackend(selection: descriptor.selection, container: container)
        } catch {
            throw InfiniteError.modelSetFailed(error.localizedDescription)
        }
        #else
        throw InfiniteError.modelSetFailed("MLX runtime is not linked into this build")
        #endif
    }

    /// Generate from the session's accumulated `context` plus a new `prompt`.
    /// Bounded to the model's native window; the 1M paging path is epic #180.
    public func generate(
        context: String,
        prompt: String,
        maxTokens: Int,
        nativeContextTokens: Int,
        temperature: Double = 0.7
    ) async throws -> InfiniteGenerationResult {
        #if canImport(MLXLMCommon) && canImport(MLXHuggingFace)
        let userText = context.isEmpty ? prompt : context + "\n\n" + prompt
        let userInput = UserInput(chat: [.user(userText)])
        let preparedInput = try await container.prepare(input: userInput)

        let promptTokens = preparedInput.text.tokens.size
        guard promptTokens <= nativeContextTokens else {
            throw InfiniteError.invalidSessionInput(
                "context (~\(promptTokens) tokens) exceeds the model's native window of \(nativeContextTokens); 1M paging is tracked in #180"
            )
        }

        // temperature 0 makes MLXLMCommon select argmax (greedy) and ignore
        // topP, giving deterministic output for the gemma4 parity test (#184).
        let params = GenerateParameters(
            maxTokens: maxTokens, temperature: Float(temperature), topP: 0.95
        )
        let collected = try await container.perform { ctx -> (String, GenerateCompletionInfo?) in
            let cache = ctx.model.newCache(parameters: params)
            let stream = try MLXLMCommon.generate(
                input: preparedInput,
                cache: cache,
                parameters: params,
                context: ctx
            )
            var text = ""
            var info: GenerateCompletionInfo?
            for await generation in stream {
                switch generation {
                case let .chunk(chunk, _):
                    text += chunk
                case let .info(completion):
                    info = completion
                default:
                    // Surface any future stream variant (e.g. a new error or
                    // tool-call frame) rather than silently dropping it into an
                    // empty/truncated result with finishReason "stop".
                    mlxInfiniteLogger.debug(
                        "MLXInfiniteBackend: unhandled generation stream variant, skipped"
                    )
                }
            }
            return (text, info)
        }
        // Prefer the engine's own decode-only stats (tokensPerSecond divides
        // generationTokenCount by generateTime, excluding prefill) and exact
        // stop reason over chunk-count proxies.
        let info = collected.1
        let finishReason: String
        switch info?.stopReason {
        case .length: finishReason = "length"
        case .cancelled: finishReason = "cancelled"
        case .stop, nil: finishReason = "stop"
        }
        return InfiniteGenerationResult(
            text: collected.0,
            tokenCount: info?.generationTokenCount ?? 0,
            decodeTokensPerSecond: info?.tokensPerSecond ?? 0,
            finishReason: finishReason
        )
        #else
        throw InfiniteError.generationUnavailable("MLX runtime is not linked into this build")
        #endif
    }
}
