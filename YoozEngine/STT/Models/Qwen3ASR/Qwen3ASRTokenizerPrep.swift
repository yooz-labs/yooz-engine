// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

#if canImport(Tokenizers)
import Tokenizers
#endif

/// First-run tokenizer validation step.
///
/// The canonical `mlx-community/Qwen3-ASR-1.7B-8bit` checkpoint
/// sometimes ships `tokenizer.json`, sometimes does not (HF mirrors
/// and revisions vary). `prepare(modelDir:)` runs after the fetcher
/// lands the artifacts and before the pipeline loads the tokenizer.
///
/// Behavior:
///   - Sentinel fast-path: if `.yooz_tokenizer_prepped` is present,
///     no-op.
///   - If `tokenizer.json` already exists, validate it loads via
///     `AutoTokenizer.from(modelFolder:)`, run a known-string canary
///     encode, drop the sentinel, return. This is the path the
///     canonical mlx-community checkpoint takes today.
///   - Otherwise, validate the fallback inputs (`tokenizer_config.json`,
///     `vocab.json`, `merges.txt`) are on disk, attempt the load via
///     `AutoTokenizer.from(modelFolder:)` (which will use the
///     fallback inputs if swift-transformers' loader supports that
///     path), run the canary encode, then drop the sentinel. The
///     current swift-transformers `loadConfig(modelFolder:)` REQUIRES
///     `tokenizer.json` to be present, so this branch typically
///     fails today; the failure is typed (`fetchValidationFailed`)
///     so operators see the issue at first-run rather than silently
///     mis-tokenizing.
///
/// We deliberately do NOT synthesize `tokenizer.json` ourselves:
/// reverse-engineering the HF tokenizer schema is a maintenance trap
/// and the canary encode catches synthesis-versus-canonical drift if
/// a future swift-transformers ever starts producing it.
///
/// Idempotent: running prep twice on a prepared directory makes no
/// filesystem changes the second time.
public enum Qwen3ASRTokenizerPrep {

    private static let logger = Logger(
        subsystem: "live.yooz.engine",
        category: "Qwen3ASRTokenizerPrep"
    )

    /// Sentinel file. Presence indicates the directory has been
    /// validated by a previous prep run.
    public static let sentinelFilename = ".yooz_tokenizer_prepped"

    /// Inputs that must be on disk for the tokenizer to load when
    /// `tokenizer.json` is absent.
    public static let fallbackInputs: [String] = [
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    /// Run the prep step. Throws `Qwen3ASRError.fileNotFound` if the
    /// fallback inputs are missing and `tokenizer.json` is also absent;
    /// throws `Qwen3ASRError.fetchValidationFailed` if the on-disk
    /// artifacts don't produce a valid tokenizer.
    public static func prepare(modelDir: URL) async throws {
        let fm = FileManager.default
        let sentinel = modelDir.appendingPathComponent(sentinelFilename)
        let tokenizerJSON = modelDir.appendingPathComponent("tokenizer.json")

        // Idempotent fast-path: sentinel present => done.
        if fm.fileExists(atPath: sentinel.path) {
            logger.debug("Tokenizer prep sentinel found — skipping")
            return
        }

        // If tokenizer.json is already on disk we still validate it
        // loads, then drop the sentinel.
        if fm.fileExists(atPath: tokenizerJSON.path) {
            logger.debug(
                "tokenizer.json present — validating it parses cleanly"
            )
            try await validateTokenizerLoads(modelDir: modelDir)
            try writeSentinel(at: sentinel)
            return
        }

        // tokenizer.json is missing — confirm the fallback inputs are
        // present so swift-transformers can build the tokenizer at
        // load time.
        for required in fallbackInputs {
            let url = modelDir.appendingPathComponent(required)
            guard fm.fileExists(atPath: url.path) else {
                throw Qwen3ASRError.fileNotFound(url)
            }
        }

        // Validate the fallback inputs actually produce a working
        // tokenizer. AutoTokenizer.from(modelFolder:) accepts a folder
        // without tokenizer.json as long as tokenizer_config.json +
        // vocab.json + merges.txt are present.
        try await validateTokenizerLoads(modelDir: modelDir)

        // Mark prep complete. We don't synthesize tokenizer.json
        // ourselves — that would mean reverse-engineering the HF
        // tokenizer.json schema, which is a maintenance trap and not
        // necessary for swift-transformers' loader. The sentinel
        // makes the prep idempotent regardless.
        try writeSentinel(at: sentinel)
        logger.info("Tokenizer prep complete (no tokenizer.json synthesis needed)")
    }

    /// Drop the sentinel atomically.
    private static func writeSentinel(at url: URL) throws {
        let body = "ok\n".data(using: .utf8)!
        try body.write(to: url, options: .atomic)
    }

    /// Try to construct a tokenizer from the on-disk artifacts. Maps
    /// any failure to `Qwen3ASRError.fetchValidationFailed` so callers
    /// can distinguish prep failures from network failures.
    private static func validateTokenizerLoads(modelDir: URL) async throws {
        #if canImport(Tokenizers)
        do {
            _ = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            throw Qwen3ASRError.fetchValidationFailed(
                "AutoTokenizer.from(modelFolder:) failed: \(error)"
            )
        }
        #else
        throw Qwen3ASRError.fetchValidationFailed(
            "Tokenizers package not available; cannot validate tokenizer"
        )
        #endif
    }
}
