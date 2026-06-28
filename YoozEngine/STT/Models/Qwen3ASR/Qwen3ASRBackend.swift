// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

/// Singleton actor wrapping the `Qwen3ASRPipeline` so the engine
/// can hold one pipeline instance alive across many HTTP requests.
/// The pipeline itself is a `final class` carrying MLX state that is
/// not `Sendable`; isolating it inside an actor is the standard
/// pattern for non-Sendable native model state.
///
/// Lifecycle:
///   1. `prepare(modelDir:)` — verify the model directory is ready
///      (files present, tokenizer prepped). No network IO.
///   2. `ensureLoaded(modelDir:)` — lazy-load the pipeline. Repeated
///      calls with the same directory are no-ops.
///   3. `transcribe(pcm:language:)` — batch transcribe.
///   4. `unload()` — drop the pipeline, freeing MLX memory.
public actor Qwen3ASRBackend {

    // MARK: - Singleton

    public static let shared = Qwen3ASRBackend()

    // MARK: - State

    private let logger = Logger(
        subsystem: "live.yooz.engine",
        category: "Qwen3ASRBackend"
    )

    private var pipeline: Qwen3ASRPipeline?
    private var loadedDirectory: URL?

    private init() {}

    // MARK: - Public surface

    /// Whether the backend has a pipeline loaded and ready to call.
    public var isLoaded: Bool {
        pipeline != nil
    }

    /// Directory the currently-loaded pipeline came from, if any.
    public var currentModelDirectory: URL? {
        loadedDirectory
    }

    /// Lazy-load the pipeline from `modelDir`. If a pipeline is
    /// already loaded from the same directory, returns immediately.
    /// If a pipeline is loaded from a different directory, the old
    /// one is unloaded first.
    public func ensureLoaded(modelDir: URL) async throws {
        if pipeline != nil,
           let loadedDirectory,
           loadedDirectory.standardizedFileURL == modelDir.standardizedFileURL
        {
            return
        }

        // Different directory or nothing loaded — drop old, load new.
        if pipeline != nil {
            await unload()
        }

        logger.info("Loading Qwen3-ASR pipeline from \(modelDir.path, privacy: .public)")
        let loaded = try await Qwen3ASRPipeline.load(from: modelDir)
        self.pipeline = loaded
        self.loadedDirectory = modelDir
        logger.info("Qwen3-ASR pipeline ready")
    }

    /// Run the batch transcribe path. Throws
    /// `Qwen3ASRError.pipelineNotLoaded` if `ensureLoaded` hasn't been
    /// called yet.
    public func transcribe(
        pcm: [Float],
        language: String? = nil,
        maxNewTokens: Int = 8_192
    ) throws -> Qwen3ASRTranscription {
        guard let pipeline else {
            throw Qwen3ASRError.pipelineNotLoaded
        }
        return try pipeline.transcribe(
            pcm: pcm,
            language: language,
            maxNewTokens: maxNewTokens
        )
    }

    /// Drop the pipeline and free MLX memory.
    public func unload() async {
        let hadPipeline = pipeline != nil
        if hadPipeline {
            logger.info("Unloading Qwen3-ASR pipeline")
        }
        pipeline = nil
        loadedDirectory = nil
        // Dropping the pipeline frees its weights; the buffers are returned to
        // the OS by `YoozSTTEngine`, which owns this backend's `.stt` MLX
        // residency and applies the cache-budget trim / flush AFTER awaiting
        // this `unload()` (see `YoozSTTEngine.setBackend`). This type can't call
        // `MLXResidency` itself — it is a separate SPM target with no EngineCore
        // dependency. The previous process-global `Memory.clearCache()` here
        // evicted a coexisting LLM's warm buffers; that cross-category stomp is
        // what the per-category residency budget removes.
    }
}
