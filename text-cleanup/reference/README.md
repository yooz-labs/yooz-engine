# Reference Materials

This directory contains reference copies of external rule files used for research and implementation guidance.

## LanguageTool Rules

Downloaded from: https://github.com/languagetool-org/languagetool

### Files

| File | Description | Rules | Lines |
|------|-------------|-------|-------|
| `en-grammar.xml` | English grammar rules | ~1,771 | ~142K |
| `en-style.xml` | English style rules | TBD | TBD |

### Categories in grammar.xml

- APRIL (April Fools rules)
- CASING
- COLLOCATIONS
- COMPOUNDING
- CONFUSED_WORDS
- GRAMMAR
- MULTITOKEN_SPELLING
- NONSTANDARD_PHRASES
- PROPER_NOUNS
- PUNCTUATION
- SEMANTICS
- TYPOGRAPHY
- TYPOS

### License

LanguageTool is licensed under LGPL 2.1. These files are included for reference only. Our implementation uses a custom rule format inspired by, but not directly copying, LanguageTool rules.

### Usage

These files help us:
1. Understand rule structure and patterns
2. Identify relevant categories for spoken-to-written correction
3. Extract rule ideas (not copy verbatim)
4. Plan multi-language support

## Adding More Languages

To add another language's rules:

```bash
curl -sL "https://raw.githubusercontent.com/languagetool-org/languagetool/master/languagetool-language-modules/XX/src/main/resources/org/languagetool/rules/XX/grammar.xml" -o XX-grammar.xml
```

Where `XX` is the language code (de, fr, es, etc.)

---

**Last Updated**: December 27, 2025
