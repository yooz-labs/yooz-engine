// InfiniteTurnComposer.swift
// InfiniteModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Produces the plain-text chat-template fragments turn-commit needs to
/// frame one conversational turn (engine#267, epic#263 phase 4). Every
/// fragment here is a raw string; tokenization happens in the backend via
/// `tokenizer.encode(text:addSpecialTokens:false)`, never in a composer.
///
/// Turn-commit's durable-session commits always use `stableAssistantWrap`'s
/// STABLE historical rendering of a turn — the form the chat template gives
/// a turn once it is no longer last — never the momentary "still last turn"
/// rendering, so a committed turn's bytes never depend on whether another
/// turn ever actually follows it (ADR 0007 D1: sessions are append-only).
public protocol InfiniteTurnComposer: Sendable {
    /// Opens a user turn, before the user's own text.
    var userOpen: String { get }
    /// Closes a user turn, after the user's own text.
    var userClose: String { get }
    /// Opens the assistant's turn for a fresh generation. `thinking` selects
    /// whether the model is invited to reason (Qwen3.5's `<think>` opener);
    /// a composer with no per-turn think toggle ignores it.
    func generationPrompt(thinking: Bool) -> String
    /// The stable historical rendering of a completed assistant turn: no
    /// reasoning markers, matching how the chat template renders a turn
    /// once a later turn exists. Trims `answer` itself (mirrors the chat
    /// template trimming message content before rendering it).
    func stableAssistantWrap(answer: String) -> String
    /// Splits a branch's raw decoded text into `(reasoning, answer)`.
    func splitThinking(_ text: String) -> (reasoning: String, answer: String)
}

/// Qwen3.5 ChatML framing — verified byte-exact against the pinned
/// `mlx-community/Qwen3.5-0.8B-MLX-4bit`'s `chat_template.jinja`
/// (revision `5d894f8cc4ef3e6c88537bf3746ed262f549da6a`): a historical
/// (non-last) assistant turn renders `<|im_start|>assistant\n{content}
/// <|im_end|>\n` with no `<think>` wrapper — only the momentary-last turn
/// gets one (`loop.index0 > ns.last_query_index`), and that wrapper
/// disappears the instant a later user turn makes it non-last. `content`
/// is always `|trim`-ed by the template before either branch runs.
public struct Qwen35ChatMLComposer: InfiniteTurnComposer {
    public init() {}

    public var userOpen: String { "<|im_start|>user\n" }
    public var userClose: String { "<|im_end|>\n" }

    public func generationPrompt(thinking: Bool) -> String {
        thinking
            ? "<|im_start|>assistant\n<think>\n"
            : "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    public func stableAssistantWrap(answer: String) -> String {
        let trimmed = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        return "<|im_start|>assistant\n" + trimmed + "<|im_end|>\n"
    }

    /// Mirrors `d1_cache/turns.py`'s `split_thinking` exactly (infinite repo,
    /// `approaches/d1-infinite-cache/d1_cache/turns.py`):
    /// `reasoning = text.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")`,
    /// `answer = text.split("</think>")[-1].lstrip("\n")`. Reasoning splits
    /// on the FIRST `</think>`; answer splits on the LAST — with more than
    /// one think/answer cycle in one decode, anything between the first
    /// close and the last close is neither reasoning nor answer and is
    /// silently dropped, exactly like the Python reference. `rstrip`/`lstrip`
    /// strip only `"\n"` characters, not general whitespace — general
    /// whitespace trimming of the answer happens in `stableAssistantWrap`,
    /// matching the Python call site's separate `answer.strip()` (mirrors
    /// the chat template's own content trimming), not in this split.
    public func splitThinking(_ text: String) -> (reasoning: String, answer: String) {
        guard text.contains("</think>") else {
            return ("", text)
        }
        let closeParts = text.components(separatedBy: "</think>")
        let beforeFirstClose = closeParts.first ?? ""
        let afterLastClose = closeParts.last ?? ""

        let reasoningRStripped = Self.rstripNewlines(beforeFirstClose)
        let openParts = reasoningRStripped.components(separatedBy: "<think>")
        let reasoning = Self.lstripNewlines(openParts.last ?? reasoningRStripped)
        let answer = Self.lstripNewlines(afterLastClose)
        return (reasoning, answer)
    }

    private static func lstripNewlines(_ s: String) -> String {
        var result = Substring(s)
        while result.first == "\n" {
            result = result.dropFirst()
        }
        return String(result)
    }

    private static func rstripNewlines(_ s: String) -> String {
        var result = Substring(s)
        while result.last == "\n" {
            result = result.dropLast()
        }
        return String(result)
    }
}

/// Gemma4 framing — verified byte-exact against the pinned
/// `mlx-community/gemma-4-e4b-it-qat-OptiQ-4bit`'s `chat_template.jinja`
/// (revision `b4966f32e71f9f4976a78f74bc8944b1d064bcbf`).
///
/// DEVIATION from the classic Gemma2/3 `<start_of_turn>`/`<end_of_turn>`
/// scheme: Gemma4's `tokenizer_config.json` special tokens are
/// `sot_token: "<|turn>"` / `eot_token: "<turn|>"`, and a turn renders
/// `<|turn>{role}\n{content}<turn|>\n` where `role` is `"model"` for an
/// assistant message, not `"assistant"`. There is also no per-turn
/// `<think>` opener: `enable_thinking` only ever injects a `<|think|>`
/// marker once, in the very first system turn — the template has no
/// mechanism to reopen a think channel on a later generation prompt. So
/// `generationPrompt(thinking:)` ignores its argument, and `splitThinking`
/// is the identity split: Gemma4 sessions in this phase never quarantine
/// reasoning, they only ever commit the raw decoded text as the answer.
public struct Gemma4Composer: InfiniteTurnComposer {
    public init() {}

    public var userOpen: String { "<|turn>user\n" }
    public var userClose: String { "<turn|>\n" }

    public func generationPrompt(thinking: Bool) -> String {
        "<|turn>model\n"
    }

    public func stableAssistantWrap(answer: String) -> String {
        let trimmed = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        return "<|turn>model\n" + trimmed + "<turn|>\n"
    }

    public func splitThinking(_ text: String) -> (reasoning: String, answer: String) {
        ("", text)
    }
}
