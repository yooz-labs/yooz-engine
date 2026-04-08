# LanguageTool Marker Behavior

## How Markers Work

The `<marker>` element in LanguageTool patterns defines which portion of matched text should be:
1. **Highlighted** as the error to the user
2. **Replaced** by the suggestion

### Key Rules

1. **If no marker exists**: The entire pattern is the error (all tokens get replaced)
2. **If marker wraps some tokens**: Only marked tokens get replaced, unmarked tokens are context

### Example: Partial Marker

```xml
<pattern>
  <marker>
    <token regexp="yes" case_sensitive="yes">Its|its</token>
  </marker>
  <token>still</token>
</pattern>
<suggestion>It's</suggestion>
```

**Behavior:**
- Input: "Its still sunny"
- Pattern matches: "Its still" (2 tokens)
- Marker covers: "Its" only (token index 0)
- Suggestion: "It's"
- Output: "It's still sunny" (only "Its" is replaced)

### Example: Full Pattern (No Marker)

```xml
<pattern>
  <token>gonna</token>
</pattern>
<suggestion>going to</suggestion>
```

**Behavior:**
- Input: "I'm gonna go"
- Pattern matches: "gonna" (1 token)
- No marker = entire pattern is error
- Output: "I'm going to go"

### Example: Multi-Token Marker

```xml
<pattern>
  <token>could</token>
  <marker>
    <token>of</token>
  </marker>
</pattern>
<suggestion>have</suggestion>
```

**Behavior:**
- Input: "could of known"
- Pattern matches: "could of" (2 tokens)
- Marker covers: "of" only (token index 1)
- Suggestion: "have"
- Output: "could have known"

## Implementation Strategy

### In Extraction Script

For each pattern, track:
- `marker_start`: Index of first token inside marker (0-based)
- `marker_end`: Index of last token inside marker (exclusive)
- If no marker: marker_start=0, marker_end=len(tokens) (entire pattern)

### In Rule Engine

When applying a rule:
1. Match the full pattern (including context tokens)
2. Replace only tokens[marker_start:marker_end] with suggestion
3. Keep tokens before marker_start and after marker_end

### Back-References with Markers

When suggestion uses `\1`, `\2`, etc.:
- Numbers refer to ALL pattern tokens, not just marked ones
- Example: `<suggestion>\1 have</suggestion>` with marker on token 2
  - `\1` refers to token 1 (even if it's context)

## Sources

- https://dev.languagetool.org/development-overview.html
- https://github.com/languagetool-org/languagetool/blob/master/languagetool-core/src/main/resources/org/languagetool/rules/rules.xsd
