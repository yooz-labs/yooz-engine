// GrammarMatch.swift
// GrammarModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// A single structured grammar correction, anchored to the original text.
///
/// Consumers (e.g. Yooz Crisp) use these to render targeted underlines with a
/// human-readable reason, instead of diffing original vs. corrected text
/// themselves. `offset` / `length` are **UTF-16 code units in the ORIGINAL
/// text** (NSString semantics), so macOS Accessibility consumers can map the
/// range directly onto an `NSRange`.
///
/// ## Position-recovery approach (important)
///
/// The grammar matcher is a compiled Rust library (`YoozTextCleanup`) exposed
/// over UniFFI. Its FFI surface returns only the fully corrected `String`; it
/// does **not** expose per-rule match metadata (rule id, category, message) or
/// per-match offsets. Surfacing those natively would require a Rust crate
/// change plus an XCFramework + UniFFI-binding regeneration (the bindings carry
/// hard-coded ABI checksums that the runtime validates at init), which is a
/// separate, build-tooling-gated change.
///
/// So this type is populated by a **token-aligned diff** between the original
/// and corrected text, computed entirely in Swift:
///
/// - `offset`, `length`, `original`, `replacement` are **exact and lossless**;
///   they are read directly from the original/corrected strings in UTF-16
///   coordinates, and each contiguous edit is reported as its own match (the
///   alignment does not merge non-adjacent edits the way a naive line/char
///   diff would).
/// - `ruleId`, `category`, `message` are **best-effort, diff-derived**. The
///   string-only FFI cannot attribute an edit to its originating LanguageTool
///   rule, so `ruleId` uses a synthetic, clearly-namespaced identifier
///   (`GRAMMAR_DIFF_*`), `category` defaults to `grammar`, and `message` is
///   synthesized from the edit ("Replace \"X\" with \"Y\"").
///
/// Recovering the true LanguageTool rule id / category / message is tracked as
/// the follow-up Rust-FFI enhancement; when that lands, this type's fields are
/// already shaped to carry it.
public struct GrammarMatch: Equatable, Sendable {
    /// Start of the matched range, in UTF-16 code units of the ORIGINAL text.
    public let offset: Int
    /// Length of the matched range, in UTF-16 code units of the ORIGINAL text.
    /// Zero for pure insertions.
    public let length: Int
    /// The matched substring of the original text. Empty for pure insertions.
    public let original: String
    /// The suggested replacement text. Empty for deletions.
    public let replacement: String
    /// Rule identifier. Diff-derived synthetic id under the current matcher.
    public let ruleId: String
    /// Rule category. Defaults to `grammar` under the current matcher.
    public let category: String
    /// Human-readable explanation of the suggestion.
    public let message: String
    /// Optional terse variant of `message`.
    public let shortMessage: String?

    public init(
        offset: Int,
        length: Int,
        original: String,
        replacement: String,
        ruleId: String,
        category: String,
        message: String,
        shortMessage: String? = nil
    ) {
        self.offset = offset
        self.length = length
        self.original = original
        self.replacement = replacement
        self.ruleId = ruleId
        self.category = category
        self.message = message
        self.shortMessage = shortMessage
    }
}

// MARK: - Match extraction

/// Computes structured `GrammarMatch` values from an original/corrected pair.
///
/// This is a pure, deterministic, dependency-free helper (no FFI, no NLTagger).
/// It diffs the two strings at word-token granularity and emits one match per
/// contiguous edit, with offsets/lengths in UTF-16 code units of the ORIGINAL
/// text.
enum GrammarMatchExtractor {

    /// Token-count ceiling for the O(n*m) LCS alignment. Inputs larger than
    /// this skip alignment and report a single whole-text block match.
    private static let maxTokensForAlignment = 4_000

    /// A word token plus its UTF-16 range within its source string.
    private struct Token {
        let text: String
        /// UTF-16 offset of the token start within its source string.
        let utf16Start: Int
        /// UTF-16 offset of the token end (exclusive) within its source string.
        let utf16End: Int
    }

    /// Build matches describing how `corrected` differs from `original`.
    ///
    /// - Returns: One match per contiguous edit. Empty when the strings are
    ///   equal.
    static func matches(original: String, corrected: String) -> [GrammarMatch] {
        guard original != corrected else { return [] }

        let origTokens = tokenize(original)
        let corrTokens = tokenize(corrected)

        let origNS = original as NSString
        let corrNS = corrected as NSString

        // The LCS table is O(n*m). Grammar checks run on short interactive
        // text; for a pathologically long input, skip the alignment and report
        // a single whole-text block match rather than risk a large allocation.
        guard origTokens.count <= maxTokensForAlignment,
              corrTokens.count <= maxTokensForAlignment else {
            let nsRange = NSRange(location: 0, length: origNS.length)
            return [makeMatch(
                offset: 0,
                length: origNS.length,
                original: origNS.substring(with: nsRange),
                replacement: corrected
            )]
        }

        // Longest-common-subsequence alignment over token TEXT, so a single
        // changed word in the middle of a sentence yields one targeted match
        // rather than dragging the whole tail into a diff.
        let lcs = lcsIndexPairs(
            origTokens.map(\.text),
            corrTokens.map(\.text)
        )

        var matches: [GrammarMatch] = []
        var oi = 0  // cursor into origTokens
        var ci = 0  // cursor into corrTokens

        // `lcs` holds (origIndex, corrIndex) anchor pairs that are equal and
        // strictly increasing. Between consecutive anchors, everything is an
        // edit. We also flush any trailing edit after the last anchor.
        func emitBlock(origRange: Range<Int>, corrRange: Range<Int>) {
            let (offset, length, originalText) = originalSpan(
                tokens: origTokens,
                range: origRange,
                source: origNS,
                // Anchor for a pure insertion: place it at the start of the
                // next original token, or at end of text if inserting at tail.
                insertionAnchorOrigIndex: origRange.lowerBound
            )
            let replacement = correctedSpan(
                tokens: corrTokens,
                range: corrRange,
                source: corrNS
            )
            matches.append(makeMatch(
                offset: offset,
                length: length,
                original: originalText,
                replacement: replacement
            ))
        }

        func emitEdit(origRange: Range<Int>, corrRange: Range<Int>) {
            // Skip no-op (both empty); shouldn't happen given guards, but be safe.
            if origRange.isEmpty && corrRange.isEmpty { return }

            // When the edit run has the same number of tokens on both sides,
            // it is almost always a set of independent single-word fixes
            // (e.g. "has a" -> "have an"). Splitting per-token keeps each fix
            // as its own targeted match instead of merging adjacent edits the
            // way a coarse block diff would. Token pairs that happen to be
            // equal are emitted as no-ops (skipped).
            if origRange.count == corrRange.count && origRange.count > 1 {
                for k in 0..<origRange.count {
                    let oIdx = origRange.lowerBound + k
                    let cIdx = corrRange.lowerBound + k
                    if origTokens[oIdx].text == corrTokens[cIdx].text { continue }
                    emitBlock(origRange: oIdx..<(oIdx + 1), corrRange: cIdx..<(cIdx + 1))
                }
                return
            }

            // Otherwise (insertion, deletion, or genuine many-to-one rewrite)
            // emit the whole contiguous block as a single match.
            emitBlock(origRange: origRange, corrRange: corrRange)
        }

        for pair in lcs {
            if oi < pair.orig || ci < pair.corr {
                emitEdit(origRange: oi..<pair.orig, corrRange: ci..<pair.corr)
            }
            oi = pair.orig + 1
            ci = pair.corr + 1
        }
        // Trailing edit after the final anchor.
        if oi < origTokens.count || ci < corrTokens.count {
            emitEdit(origRange: oi..<origTokens.count, corrRange: ci..<corrTokens.count)
        }

        return matches
    }

    // MARK: Tokenization

    /// Split into maximal non-whitespace runs ("words"), recording each token's
    /// UTF-16 range in `text`.
    private static func tokenize(_ text: String) -> [Token] {
        var tokens: [Token] = []
        let ns = text as NSString
        let length = ns.length
        var i = 0
        let whitespace = CharacterSet.whitespacesAndNewlines

        while i < length {
            // Skip whitespace (a UTF-16 unit is a whole BMP scalar; combined
            // emoji never start with a whitespace unit, so this is safe).
            while i < length, isWhitespaceUnit(ns.character(at: i), set: whitespace) {
                i += 1
            }
            guard i < length else { break }
            let start = i
            while i < length, !isWhitespaceUnit(ns.character(at: i), set: whitespace) {
                i += 1
            }
            let range = NSRange(location: start, length: i - start)
            tokens.append(Token(
                text: ns.substring(with: range),
                utf16Start: start,
                utf16End: i
            ))
        }
        return tokens
    }

    private static func isWhitespaceUnit(_ unit: unichar, set: CharacterSet) -> Bool {
        guard let scalar = Unicode.Scalar(unit) else { return false }
        return set.contains(scalar)
    }

    // MARK: Span construction

    /// UTF-16 offset/length and substring of an original-token range.
    private static func originalSpan(
        tokens: [Token],
        range: Range<Int>,
        source: NSString,
        insertionAnchorOrigIndex: Int
    ) -> (offset: Int, length: Int, text: String) {
        if range.isEmpty {
            // Pure insertion: zero-length anchor at the next token start, or at
            // end of text when inserting past the last token.
            let anchor: Int
            if insertionAnchorOrigIndex < tokens.count {
                anchor = tokens[insertionAnchorOrigIndex].utf16Start
            } else {
                anchor = source.length
            }
            return (anchor, 0, "")
        }
        let start = tokens[range.lowerBound].utf16Start
        let end = tokens[range.upperBound - 1].utf16End
        let nsRange = NSRange(location: start, length: end - start)
        return (start, end - start, source.substring(with: nsRange))
    }

    /// Substring of a corrected-token range (the replacement text).
    private static func correctedSpan(
        tokens: [Token],
        range: Range<Int>,
        source: NSString
    ) -> String {
        guard !range.isEmpty else { return "" }
        let start = tokens[range.lowerBound].utf16Start
        let end = tokens[range.upperBound - 1].utf16End
        return source.substring(with: NSRange(location: start, length: end - start))
    }

    // MARK: Match metadata (best-effort, diff-derived)

    private static func makeMatch(
        offset: Int,
        length: Int,
        original: String,
        replacement: String
    ) -> GrammarMatch {
        let ruleId: String
        let message: String
        if original.isEmpty {
            ruleId = "GRAMMAR_DIFF_INSERT"
            message = "Insert \"\(replacement)\""
        } else if replacement.isEmpty {
            ruleId = "GRAMMAR_DIFF_DELETE"
            message = "Remove \"\(original)\""
        } else {
            ruleId = "GRAMMAR_DIFF_REPLACE"
            message = "Replace \"\(original)\" with \"\(replacement)\""
        }
        return GrammarMatch(
            offset: offset,
            length: length,
            original: original,
            replacement: replacement,
            ruleId: ruleId,
            category: "grammar",
            message: message,
            shortMessage: nil
        )
    }

    // MARK: LCS

    private struct Pair { let orig: Int; let corr: Int }

    /// Classic dynamic-programming LCS, returning the aligned (orig, corr)
    /// index pairs of equal tokens.
    private static func lcsIndexPairs(_ a: [String], _ b: [String]) -> [Pair] {
        let n = a.count
        let m = b.count
        if n == 0 || m == 0 { return [] }

        // dp[i][j] = LCS length of a[i...] and b[j...].
        var dp = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in stride(from: m - 1, through: 0, by: -1) {
                if a[i] == b[j] {
                    dp[i][j] = dp[i + 1][j + 1] + 1
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                }
            }
        }

        var pairs: [Pair] = []
        var i = 0
        var j = 0
        while i < n && j < m {
            if a[i] == b[j] {
                pairs.append(Pair(orig: i, corr: j))
                i += 1
                j += 1
            } else if dp[i + 1][j] >= dp[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        return pairs
    }
}
