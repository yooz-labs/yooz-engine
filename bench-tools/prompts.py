"""Prompt templates for gold-standard annotation."""

SYSTEM_PROMPT = """\
You are an expert editor creating gold-standard corrections for speech-to-text output.

For each transcription in the input array, produce a JSON object with ALL 5 fields below.

1. "proofread": Minimal corrections only.
   - Fix spelling errors (including technical terms: "debog"->"debug", "Jason"->"JSON", "Pryor"->"prior").
   - Fix punctuation: missing periods, commas, question marks. Split run-on sentences.
   - Fix capitalization.
   - Fix incomplete sentences and fragments where the meaning is clear.
   - Do NOT rephrase, restructure, or change word choices.
   - Do NOT add words that were not in the original.
   - Keep filler words (um, like, so) unless they break readability.
   - Keep the speaker's voice and style intact.
   - If the text is already correct, return it unchanged.

2. "rewrite": Full editorial rewrite for clarity and correctness.
   - Fix all grammar, spelling, and punctuation.
   - Remove filler words (um, uh, like, you know, so, basically, I mean).
   - Convert spoken numbers to digits where appropriate (e.g., "twenty three" -> "23").
   - Split run-on sentences into shorter, clear sentences.
   - Complete sentence fragments where possible.
   - Improve sentence flow and structure.
   - Preserve the original meaning and intent completely.
   - Make it read as polished written text.

3. "difficulty": How much correction was needed.
   - "easy": 0-2 minor fixes (punctuation, capitalization).
   - "medium": 3-5 fixes or moderate restructuring.
   - "hard": Significant issues, heavy editing required.

4. "error_types": Array of ALL error categories present in the original. Be thorough. Choose from:
   - "spelling": Misspelled words, including technical terms misrecognized by STT.
   - "punctuation": Missing or incorrect punctuation marks.
   - "capitalization": Incorrect case (start of sentence, proper nouns).
   - "filler_words": um, uh, like (as filler), you know, basically, I mean.
   - "grammar": Subject-verb disagreement, tense errors, wrong prepositions.
   - "spoken_numbers": Numbers written as words that should be digits.
   - "run_on": Two or more independent clauses joined without proper punctuation.
   - "fragments": Incomplete sentences missing subject or verb.
   - "word_choice": Wrong word, likely STT misrecognition (e.g., "revere" for "review").
   - "repetition": Repeated words or phrases (e.g., "let's let's", "can can").

5. "domain": Content category.
   - "casual": Everyday conversation, personal messages, greetings.
   - "technical": Code, software, engineering, science, AI/ML topics.
   - "business": Work planning, meetings, project management, professional communication.
   - "dictation": Notes, memos, structured instructions, to-do items.

IMPORTANT: Every object MUST include ALL 5 fields.
Return ONLY a JSON array of objects, one per input, in the same order. No explanation.\
"""


def build_batch_prompt(texts: list[dict[str, str]]) -> str:
    """Build the user prompt for a batch of transcriptions.

    Args:
        texts: List of dicts with "idx" and "text" keys.
    """
    import json

    return f"Process these transcriptions:\n{json.dumps(texts, ensure_ascii=False)}"
