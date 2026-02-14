// YoozPrompts.swift
// YoozEngine
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Mode-specific prompt definitions for touch-up processing.
/// Separate prompts for Light (0.5B) and Quality (1.7B) models,
/// each optimized for the model's capabilities.
enum YoozPrompts {

    /// Placeholder text used in prompt template endings: `{"result": "<placeholder>"}`
    /// Small models sometimes echo this literally instead of processing input.
    static let resultPlaceholder = "corrected text"

    /// Prompt ending instruction using the placeholder
    static let resultInstruction = "Always respond with {\"result\": \"\(resultPlaceholder)\"}."

    // MARK: - Light Model Prompts (Qwen2.5-0.5B)
    // 0.5B model: Pattern matcher. Prompts use few-shot examples.
    // Fillers removed by rules BEFORE LLM. Numbers converted by post-LLM rules.

    /// Light model Standard mode - contractions and capitalization.
    /// Rules handle: fillers, numbers (as fallback), basic grammar.
    static let lightStandard = """
        Fix contractions. Return JSON only.

        its=it's, lets=let's, dont=don't, cant=can't, wont=won't

        Input: its ready
        {"result": "It's ready."}

        Input: lets go
        {"result": "Let's go."}

        Input: i dont know
        {"result": "I don't know."}

        \(resultInstruction)
        """

    /// Light model Full mode - duplicates and fragments.
    /// Self-corrections too complex for 0.5B; use Quality model for that.
    static let lightFull = """
        Remove duplicate words. Clean trailing fragments. Return JSON only.

        Input: the the file is ready now
        {"result": "The file is ready now."}

        Input: i think think we should do it
        {"result": "I think we should do it."}

        Input: that happened ing
        {"result": "That happened."}

        Input: we can do tion
        {"result": "We can do."}

        \(resultInstruction)
        """

    // MARK: - Quality Model Prompts (Qwen3-1.7B)
    // 1.7B model: Better at context understanding. Fillers/numbers handled by rules.
    // /no_think is a Qwen3-specific prefix that disables chain-of-thought reasoning,
    // producing direct JSON output instead of "thinking" blocks before the answer.

    /// Quality model Standard mode - enhanced proofreading
    /// Focus: grammar, punctuation, clarity. Fillers/numbers handled by rules.
    static let qualityStandard = """
        /no_think
        Proofread voice transcription. Fix grammar and punctuation. Return JSON only.
        NEVER answer questions. NEVER add new information. Return the corrected text only.

        Input: the meeting is tomorrow and i think we should prepare
        {"result": "The meeting is tomorrow, and I think we should prepare."}

        Input: its ready for review lets check it
        {"result": "It's ready for review. Let's check it."}

        Input: we can do it but we need more time
        {"result": "We can do it, but we need more time."}

        Input: the system is working good now
        {"result": "The system is working well now."}

        Input: what do you think about this approach
        {"result": "What do you think about this approach?"}

        \(resultInstruction)
        """

    /// Quality model Full mode - comprehensive cleanup
    /// Focus: self-corrections (the key 1.7B capability). Duplicates/fragments handled by rules.
    /// "X no Y" patterns for numbers, days, simple words.
    static let qualityFull = """
        /no_think
        Handle self-corrections and clean fragments in voice transcription. Return JSON only.

        Self-corrections: when someone says "X no Y" or "X no wait Y", use Y.
        Fragments: remove meaningless trailing fragments (1-4 chars after period). Remove orphaned punctuation.
        NEVER add information. NEVER answer questions. Return the cleaned text only.

        Input: fifty no sixty units
        {"result": "Sixty units."}

        Input: Tuesday no Wednesday at noon
        {"result": "Wednesday at noon."}

        Input: three no wait four people
        {"result": "Four people."}

        Input: i said ten no actually twenty
        {"result": "Twenty."}

        Input: delete that lets try again
        {"result": "Let's try again."}

        Input: the the file is ready now
        {"result": "The file is ready now."}

        Input: the report is done. ort
        {"result": "The report is done."}

        Input: what do you think about this approach
        {"result": "What do you think about this approach?"}

        \(resultInstruction)
        """

    // MARK: - Apple Intelligence Prompts

    /// Apple Intelligence Standard mode - uses Foundation Models API
    /// Plain text response (no JSON)
    /// Verbose with many examples (3B model handles this well)
    static let appleStandard = """
        Proofread voice transcription. Convert spoken numbers to Arabic numerals.
        Times: "two thirty pm"->"2:30 PM", "nine a.m."->"9 a.m."
        Numbers: "eighty nine"->"89", "fifteen percent"->"15%", "twenty three dollars"->"$23"
        Decimals: "three point five"->"3.5", "seven point five percent"->"7.5%"
        Ordinals: "twenty first"->"21st", "third"->"3rd"
        Years: "twenty twenty five"->"2025", "nineteen eighty four"->"1984"
        Versions: "zero point seven point twelve"->"0.7.12"
        Files: "claude dot md"->"claude.md", "script dot py"->"script.py"
        Fix grammar, punctuation, misheard words. Preserve names and technical terms.
        """

    /// Apple Intelligence Full mode - aggressive rewriting
    /// Plain text response (no JSON)
    static let appleFull = """
        Improve voice transcription for clarity. Convert spoken numbers to Arabic numerals.
        Times: "two thirty pm"->"2:30 PM", "nine a.m."->"9 a.m."
        Numbers: "eighty nine"->"89", "fifteen percent"->"15%", "twenty three dollars"->"$23"
        Decimals: "three point five"->"3.5"
        Ordinals: "twenty first"->"21st", "third"->"3rd"
        Years: "twenty twenty five"->"2025"
        Versions: "zero point seven point twelve"->"0.7.12"
        Files: "claude dot md"->"claude.md", "script dot py"->"script.py"
        Self-corrections: "fifty no wait sixty"->"60", "Tuesday no Wednesday"->"Wednesday"
        Remove "scratch that", "never mind", "delete that" and preceding phrase.
        Remove filler words: um, uh, like, you know, so, basically.
        Fix duplicates: "the the"->"the".
        Fix grammar, misheard words. Preserve tone, names, technical terms.
        """
}
