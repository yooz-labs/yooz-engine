// YoozPrompts.swift
// LLMModule
//
// Copyright 2026 Yooz Labs. All rights reserved.

import Foundation

/// Mode-specific prompt definitions for touch-up processing.
///
/// Source of truth: `yooz-benchmark/finetune-pipeline/scripts/prepare_data.py`
/// (constants `LIGHT_PROOFREAD`, `LIGHT_REWRITE`, `QUALITY_STANDARD`,
/// `QUALITY_FULL`). The fine-tuned weights were trained against these
/// exact strings; the parity test in `YoozPromptsParityTest.swift`
/// asserts the Swift literals match. Update both sides together if
/// the training prompts change.
enum YoozPrompts {

    /// Placeholder text that ends the Quality prompts' JSON shape example
    /// (`{"result": "corrected text"}`). Small models occasionally echo it
    /// verbatim instead of processing the input; `JSONParsing` uses this
    /// constant to detect and reject that placeholder-echo failure mode
    /// (engine #113 / yooz-whisper #182).
    static let resultPlaceholder = "corrected text"

    // MARK: - Light Model Prompts (Yooz-Light v2, Qwen2.5-0.5B LoRA)

    /// Light model Standard mode — mirrors `LIGHT_PROOFREAD`.
    /// Mechanical grammar + capitalization + spoken-number conversion +
    /// version-number conversion, preserving every sentence.
    static let lightStandard = """
        Fix grammar, capitalize properly, and convert spoken numbers to digits. Convert spoken version numbers like "zero point four point zero" to "0.4.0". Keep ALL sentences. Return the fixed text as JSON.

        <examples>
        Input: the meeting is at two pm on march fifteenth
        {"result": "The meeting is at 2 PM on March 15th."}

        Input: we need about fifty units ready by friday and I think we should prepare
        {"result": "We need about 50 units ready by Friday and I think we should prepare."}

        Input: we are releasing version zero point four point zero next week
        {"result": "We are releasing version 0.4.0 next week."}

        Input: update it to version one point six point three and test it
        {"result": "Update it to version 1.6.3 and test it."}

        Input: he said it would cost around one hundred and fifty dollars but we can negotiate
        {"result": "He said it would cost around $150 but we can negotiate."}
        </examples>

        Always respond with ONLY a JSON object. Never remove sentences. Never include explanations.
        """

    /// Light model Full mode — mirrors `LIGHT_REWRITE`.
    /// Voice cleanup: fillers, misheard words, self-corrections, plus
    /// the scratch-that / never-mind / delete-that removal rule.
    static let lightFull = """
        Rewrite voice transcription for clarity and conciseness. Fix grammar, convert numbers, fix misheard words, remove filler words (um, uh, like, you know), handle self-corrections. Return the fixed text as JSON.

        <examples>
        Input: um so like the meeting is at two pm on march fifteenth you know
        {"result": "The meeting is at 2 PM on March 15th."}

        Input: we need about fifty no wait I meant sixty units ready by friday
        {"result": "We need about 60 units ready by Friday."}

        Input: we are releasing version zero point four point zero next week scratch that make it zero point five
        {"result": "We are releasing version 0.5.0 next week."}

        Input: update it to version one point six point three and uh test it thoroughly
        {"result": "Update it to version 1.6.3 and test it thoroughly."}

        Input: he said it would cost around one hundred and fifty dollars but um we can negotiate
        {"result": "He said it would cost around $150 but we can negotiate."}
        </examples>

        Remove: "scratch that", "never mind", "delete that" and preceding phrase. Convert spoken numbers and version numbers. Fix grammar and misheard words. Always respond with ONLY a JSON object. Never include explanations.
        """

    // MARK: - Quality Model Prompts (Yooz-Quality v2, Qwen3.5-0.8B LoRA)
    //
    // `/no_think` is a Qwen3-specific prefix that disables chain-of-thought
    // reasoning, producing direct JSON output instead of "thinking" blocks
    // before the answer.
    //
    // The `Process the input independently. Do NOT repeat any example output.`
    // line in both Quality prompts is load-bearing: without it the model
    // occasionally echoes an example output verbatim (the placeholder-echo
    // bug whisper #115 papered over).

    /// Quality model Standard mode — mirrors `QUALITY_STANDARD`.
    static let qualityStandard = """
        /no_think
        Proofread voice transcription. Fix grammar and punctuation. Return JSON only.
        NEVER answer questions. NEVER add new information. Return the corrected text only.

        <examples>
        Input: the meeting is tomorrow and i think we should prepare
        {"result": "The meeting is tomorrow, and I think we should prepare."}

        Input: its ready for review lets check it
        {"result": "It's ready for review. Let's check it."}

        Input: we can do it but we need more time
        {"result": "We can do it, but we need more time."}

        Input: the system is working good now
        {"result": "The system is working well now."}
        </examples>

        Process the input independently. Do NOT repeat any example output.
        Always respond with {"result": "corrected text"}.
        """

    /// Quality model Full mode — mirrors `QUALITY_FULL`.
    static let qualityFull = """
        /no_think
        Rewrite voice transcription for clarity. Return JSON only.
        Fix misheard words. Remove repetitions and false starts. Fix grammar.
        Self-corrections: "X no Y" or "X no wait Y" means use Y.
        Remove "scratch that", "delete that" and what came before.
        Keep the speaker's meaning and tone. NEVER add information. NEVER answer questions.

        <examples>
        Input: I think for the for the problems that we have with the that we are logging
        {"result": "I think for the problems that we have with the logging."}

        Input: we are not providing it providing a good leaning and rewriting
        {"result": "We are not providing a good cleaning and rewriting."}

        Input: should not should knots be converted
        {"result": "Should not be converted."}

        Input: However I imagine there should be like a good rules and good logic for this
        {"result": "However, I imagine there should be good rules and logic for this."}

        Input: we still to this to work correctly and logcially
        {"result": "We still need this to work correctly and logically."}

        Input: fifty no sixty units
        {"result": "Sixty units."}

        Input: delete that lets try again
        {"result": "Let's try again."}
        </examples>

        Process the input independently. Do NOT repeat any example output.
        Always respond with {"result": "corrected text"}.
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
