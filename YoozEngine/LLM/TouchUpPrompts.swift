// TouchUpPrompts.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// System prompts for LLM touch-up processing.
///
/// Source of truth for training prompts:
/// `yooz-benchmark/finetune-pipeline/scripts/prepare_data.py`. The
/// canonical Swift mirrors live in `YoozPrompts.swift`; this file holds
/// validation/replacement prompts that have no training-time counterpart.
enum TouchUpPrompts {

    // MARK: - Light Model Prompts

    /// Alias of `YoozPrompts.lightStandard` (LIGHT_PROOFREAD). Kept as a
    /// stable name so existing callers (`TouchUpProcessor.process`
    /// default, `TouchUpEngine.selectPrompt` `.off` branch) can route
    /// through the canonical training-aligned prompt without API churn.
    /// Never duplicate the literal here — drift between the two would
    /// silently regress inference quality.
    static let proofread = YoozPrompts.lightStandard

    // MARK: - Quality Model Prompts

    // qualityStandard prompt lives in YoozPrompts.swift (the canonical version
    // used by TouchUpEngine.selectPrompt). Kept there to match fine-tuning data.

    // MARK: - Validation + Proofreading (Quality Model)

    /// Prompt for validation + proofreading with Yooz-Quality (Qwen3-1.7B)
    /// Used when vocabulary replacements need contextual validation
    static let validateAndProofread = """
        Do TWO tasks: (1) For each replacement, decide if it fits the context. (2) Fix grammar, capitalize, convert spoken numbers to digits.

        <examples>
        Input: {"text": "I talked to Claude about fifty dollars", "replacements": [{"orig": "cloud", "repl": "Claude"}]}
        {"result": "I talked to Claude about $50.", "keep": [true]}

        Input: {"text": "The Claude is fluffy", "replacements": [{"orig": "cloud", "repl": "Claude"}]}
        {"result": "The cloud is fluffy.", "keep": [false]}

        Input: {"text": "Ask Siri to set a timer for thirty minutes", "replacements": [{"orig": "series", "repl": "Siri"}]}
        {"result": "Ask Siri to set a timer for 30 minutes.", "keep": [true]}

        Input: {"text": "The Siri is on Netflix", "replacements": [{"orig": "series", "repl": "Siri"}]}
        {"result": "The series is on Netflix.", "keep": [false]}
        </examples>

        Always respond with ONLY a JSON object. Never include explanations.
        """

    // MARK: - Prompt Building

    /// Build a validation prompt with replacements
    /// - Parameters:
    ///   - text: The text with replacements already applied
    ///   - replacements: Array of (original, replacement) tuples
    /// - Returns: JSON prompt for the validation model
    static func buildValidationPrompt(
        text: String,
        replacements: [(original: String, replacement: String)]
    ) -> String {
        let replacementsJSON = replacements.map { r in
            "{\"orig\": \"\(escapeJSON(r.original))\", \"repl\": \"\(escapeJSON(r.replacement))\"}"
        }.joined(separator: ", ")

        return "{\"text\": \"\(escapeJSON(text))\", \"replacements\": [\(replacementsJSON)]}"
    }

    /// Escape special characters for JSON
    private static func escapeJSON(_ string: String) -> String {
        var result = string
        result = result.replacingOccurrences(of: "\\", with: "\\\\")
        result = result.replacingOccurrences(of: "\"", with: "\\\"")
        result = result.replacingOccurrences(of: "\n", with: "\\n")
        result = result.replacingOccurrences(of: "\r", with: "\\r")
        result = result.replacingOccurrences(of: "\t", with: "\\t")
        return result
    }
}
