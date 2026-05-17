// JSONParsing.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation
import os.log

private let logger = Logger(subsystem: "live.yooz.engine", category: "JSONParsing")

// MARK: - Response Structures

/// Response from proofreading (fast model)
struct ProofreadResponse: Codable, Sendable {
    let result: String

    init(result: String) {
        self.result = result
    }
}

/// Response from validation + proofreading (quality model)
struct ValidateResponse: Codable, Sendable {
    let result: String
    let keep: [Bool]

    init(result: String, keep: [Bool]) {
        self.result = result
        self.keep = keep
    }
}

// MARK: - Parsing Functions

/// Parse a proofreading response from the LLM
/// - Parameters:
///   - response: Raw LLM response text
///   - fallback: Fallback text if parsing fails
/// - Returns: Tuple of (result text, success flag)
func parseProofreadResponse(_ response: String, fallback: String) -> (text: String, success: Bool) {
    // Strip a leading/trailing ```json ... ``` fence if present, then trim.
    // Small models occasionally emit markdown-fenced output on long inputs
    // (engine #134) which defeats both the direct decode and the original
    // first-`{` / last-`}` heuristic when extra prose lives outside the fence.
    let trimmed = stripMarkdownFence(response)
        .trimmingCharacters(in: .whitespacesAndNewlines)

    // Try direct parse
    if let data = trimmed.data(using: .utf8),
       let decoded = try? JSONDecoder().decode(ProofreadResponse.self, from: data),
       !isPlaceholderEcho(decoded.result) {
        return (decoded.result, true)
    }

    // Try every balanced JSON object the model emitted, in order. This handles
    // preamble text ("Here is the corrected text: {...}"), trailing prose,
    // and the multi-object case where the model concatenates two outputs
    // ({"result":"A"}{"result":"B"}). We pick the first one that decodes to
    // a non-placeholder ProofreadResponse.
    for jsonText in extractJSONCandidates(from: trimmed) {
        if let data = jsonText.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(ProofreadResponse.self, from: data),
           !isPlaceholderEcho(decoded.result) {
            return (decoded.result, true)
        }
    }

    // Parsing failed (or model echoed the prompt template verbatim).
    // `.public` interpolation is intentional: without seeing the raw shape we
    // can't diagnose which malformed pattern the Light LLM is emitting on
    // long inputs (engine #134). The response is model output, not user PII.
    logger.warning("Failed to parse proofread response. Preview: \(trimmed.prefix(200), privacy: .public)")
    return (fallback, false)
}

/// Parse a validation response from the LLM
/// - Parameters:
///   - response: Raw LLM response text
///   - fallback: Fallback text if parsing fails
///   - numReplacements: Expected number of keep decisions
/// - Returns: Tuple of (result text, keep decisions, success flag)
func parseValidateResponse(
    _ response: String,
    fallback: String,
    numReplacements: Int
) -> (text: String, keepDecisions: [Bool], success: Bool) {
    // Default to reverting all replacements when parsing fails (safer)
    let defaultKeep = Array(repeating: false, count: numReplacements)
    // Same fence-strip + trim as the proofread path (engine #134).
    let trimmed = stripMarkdownFence(response)
        .trimmingCharacters(in: .whitespacesAndNewlines)

    // Try direct parse
    if let data = trimmed.data(using: .utf8),
       let decoded = try? JSONDecoder().decode(ValidateResponse.self, from: data),
       !isPlaceholderEcho(decoded.result) {
        let keep = normalizeKeepDecisions(decoded.keep, expected: numReplacements)
        return (decoded.result, keep, true)
    }

    // Try every balanced JSON object the model emitted; first decodable wins.
    for jsonText in extractJSONCandidates(from: trimmed) {
        if let data = jsonText.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(ValidateResponse.self, from: data),
           !isPlaceholderEcho(decoded.result) {
            let keep = normalizeKeepDecisions(decoded.keep, expected: numReplacements)
            return (decoded.result, keep, true)
        }
    }

    // Parsing failed (or model echoed the prompt template verbatim).
    // See proofread path for the `.public` rationale.
    logger.warning("Failed to parse validation response, reverting all. Preview: \(trimmed.prefix(200), privacy: .public)")
    return (fallback, defaultKeep, false)
}

/// Detect when a model has echoed the prompt template's placeholder string
/// verbatim instead of actually processing the input. The Light and Quality
/// prompts in `YoozPrompts` end with
/// `Always respond with {"result": "corrected text"}.` as a shape example,
/// and small models occasionally copy that literally
/// (engine #113 / yooz-whisper #182). Without this guard
/// `parseProofreadResponse` reports success and the user sees the
/// placeholder pasted.
private func isPlaceholderEcho(_ result: String) -> Bool {
    let normalized = result.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return normalized == YoozPrompts.resultPlaceholder.lowercased()
}

// MARK: - Helper Functions

/// Strip a single leading and trailing markdown code fence (```json ... ```
/// or ``` ... ```) if both are present. Models occasionally wrap their JSON
/// output in a fenced block on long inputs (engine #134); the bare-`{`
/// heuristic in `extractJSONCandidates` then trips on the backticks if extra
/// prose lives outside the fence. If no fully-paired fence is found the
/// original string is returned unchanged.
private func stripMarkdownFence(_ text: String) -> String {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix("```") else { return text }

    // Find the first newline (separates the opening fence's language tag from
    // the body) and the last ``` in the string (closing fence).
    guard let firstNewline = trimmed.firstIndex(of: "\n") else { return text }
    let afterOpen = trimmed.index(after: firstNewline)
    guard let closeRange = trimmed.range(of: "```", options: .backwards),
          closeRange.lowerBound > afterOpen else {
        return text
    }
    return String(trimmed[afterOpen..<closeRange.lowerBound])
}

/// Enumerate every balanced top-level `{...}` substring in the order it
/// appears. Brace counting is performed outside of JSON string literals (with
/// backslash escape awareness) so quoted braces in the model's text don't
/// throw off the depth count. Used by the proofread and validate paths to
/// recover from preamble text, trailing prose, and multi-object responses
/// (engine #134).
private func extractJSONCandidates(from text: String) -> [String] {
    var candidates: [String] = []
    let chars = Array(text)
    var i = 0
    while i < chars.count {
        if chars[i] == "{" {
            // Walk forward tracking brace depth; ignore braces inside strings.
            var depth = 0
            var inString = false
            var escaped = false
            var j = i
            while j < chars.count {
                let c = chars[j]
                if inString {
                    if escaped {
                        escaped = false
                    } else if c == "\\" {
                        escaped = true
                    } else if c == "\"" {
                        inString = false
                    }
                } else {
                    if c == "\"" {
                        inString = true
                    } else if c == "{" {
                        depth += 1
                    } else if c == "}" {
                        depth -= 1
                        if depth == 0 {
                            candidates.append(String(chars[i...j]))
                            i = j
                            break
                        }
                    }
                }
                j += 1
            }
            // Unterminated object: skip the unmatched `{` and continue.
            if depth != 0 {
                i += 1
                continue
            }
        }
        i += 1
    }
    return candidates
}

/// Normalize keep decisions to match expected count
private func normalizeKeepDecisions(_ keep: [Bool], expected: Int) -> [Bool] {
    if keep.count == expected {
        return keep
    } else if keep.count > expected {
        return Array(keep.prefix(expected))
    } else {
        // Pad with true (keep by default if not specified)
        return keep + Array(repeating: true, count: expected - keep.count)
    }
}
