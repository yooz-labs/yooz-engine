#!/usr/bin/env python3
"""
Extract LanguageTool rules that our engine can handle.

Filters out rules with unsupported features like:
- <match> with complex postag transformations
- <unify> elements
- Complex nested structures

Keeps rules with:
- Simple patterns
- Antipatterns
- Exceptions
- case_sensitive
- regexp
- Back-references (\1, \2)
- Simple <match no="N"/> converted to backref

Properly escapes XML entities.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from html import escape as html_escape


def escape_xml(text):
    """Properly escape XML special characters."""
    if not text:
        return ""
    # Use html.escape which handles &, <, > properly
    # But we need to be careful not to double-escape
    # First, unescape any existing entities, then re-escape
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )
    return html_escape(text, quote=False)


def convert_match_to_backref(suggestion_elem):
    r"""
    Convert <match no="N"/> elements to \N back-references.
    Returns (converted_text, success) where success=False if unconvertible match found.
    """
    # Get all text and child elements
    parts = []
    if suggestion_elem.text:
        parts.append(suggestion_elem.text)

    for child in suggestion_elem:
        if child.tag == "match":
            no = child.get("no")
            if no is None:
                return None, False

            # Check for complex match attributes we can't handle
            has_postag = child.get("postag") is not None
            has_regexp = child.get("regexp_match") is not None
            child.get("case_conversion") is not None

            # Skip rules with postag transformations or regexp replacements
            if has_postag or has_regexp:
                return None, False

            # We can handle simple matches and case conversion
            # (case conversion will be ignored for now but the rule still works)
            parts.append(f"\\{no}")

            if child.tail:
                parts.append(child.tail)
        else:
            # Other child elements - can't handle
            return None, False

    return "".join(parts), True


def can_handle_rule(rule_elem):
    """Check if we can handle this rule."""
    rule_xml = ET.tostring(rule_elem, encoding="unicode")

    # Skip rules with unsupported features
    if "<unify" in rule_xml:
        return False, "unify"
    if "<phraseref" in rule_xml:
        return False, "phraseref"
    if "<includephrases" in rule_xml:
        return False, "includephrases"

    # Must have a pattern
    pattern = rule_elem.find("pattern")
    if pattern is None:
        return False, "no_pattern"

    # Must have a suggestion (either direct or inside message)
    suggestions = rule_elem.findall(".//suggestion")
    if not suggestions:
        return False, "no_suggestion"

    # Check if suggestion has <match> elements - try to convert them
    if "<match " in rule_xml or "<match>" in rule_xml:
        for sugg in suggestions:
            _, success = convert_match_to_backref(sugg)
            if not success:
                return False, "complex_match"

    # Extract tokens with marker info - we now properly handle partial markers
    tokens, marker_start, marker_end = extract_pattern_with_marker(pattern)
    if not tokens:
        return False, "no_tokens"

    # Validate that tokens have content (text or regexp)
    # Skip rules that ONLY use postag (too broad without actual text to match)
    all_postag_only = True
    for token in tokens:
        text = (token.text or "").strip()
        has_regexp = token.get("regexp") == "yes"
        has_postag = token.get("postag") is not None

        if not text and not has_regexp and not has_postag:
            return False, "empty_token"

        # Check if this token has actual text content
        if text or has_regexp:
            all_postag_only = False

    # Skip rules where ALL tokens only have postag (no text) - too broad
    if all_postag_only:
        return False, "postag_only"

    return True, None


def extract_token(token_elem):
    """Extract token attributes."""
    text = token_elem.text.strip() if token_elem.text else ""

    attrs = {}
    if token_elem.get("regexp") == "yes":
        attrs["regexp"] = "yes"
    if token_elem.get("case_sensitive") == "yes":
        attrs["case_sensitive"] = "true"
    if token_elem.get("postag"):
        attrs["postag"] = token_elem.get("postag")
    if token_elem.get("postag_regexp") == "yes":
        attrs["postag_regexp"] = "yes"
    if token_elem.get("spacebefore"):
        attrs["spacebefore"] = token_elem.get("spacebefore")
    if token_elem.get("min"):
        attrs["min"] = token_elem.get("min")
    if token_elem.get("max"):
        attrs["max"] = token_elem.get("max")
    if token_elem.get("negate") == "yes":
        attrs["negate"] = "yes"
    if token_elem.get("inflected") == "yes":
        attrs["inflected"] = "yes"

    # Handle exceptions
    exceptions = []
    for exc in token_elem.findall("exception"):
        if exc.text:
            exc_text = exc.text.strip()
            exc_attrs = {}
            if exc.get("regexp") == "yes":
                exc_attrs["regexp"] = "yes"
            if exc.get("case_sensitive") == "yes":
                exc_attrs["case_sensitive"] = "true"
            exceptions.append((escape_xml(exc_text), exc_attrs))

    return escape_xml(text), attrs, exceptions


def format_token(text, attrs, exceptions):
    """Format token as XML."""
    # Escape attribute values
    safe_attrs = {k: escape_xml(v) for k, v in attrs.items()}
    attr_str = " ".join(f'{k}="{v}"' for k, v in safe_attrs.items())
    if attr_str:
        attr_str = " " + attr_str

    if exceptions:
        exc_lines = []
        for exc_text, exc_attrs in exceptions:
            exc_attr_str = " ".join(f'{k}="{v}"' for k, v in exc_attrs.items())
            if exc_attr_str:
                exc_attr_str = " " + exc_attr_str
            exc_lines.append(
                f"                            <exception{exc_attr_str}>{exc_text}</exception>"
            )
        exc_str = "\n".join(exc_lines)
        return f"""                        <token{attr_str}>
                            {text}
{exc_str}
                        </token>"""
    elif text:
        return f"                        <token{attr_str}>{text}</token>"
    else:
        return f"                        <token{attr_str} />"


def extract_pattern_with_marker(pattern_elem):
    """
    Extract pattern tokens and marker position.

    Returns:
        (tokens, marker_start, marker_end)
        - tokens: list of token elements
        - marker_start: index of first marked token (0-based)
        - marker_end: index after last marked token (exclusive)
        - If no marker, marker covers entire pattern (0, len(tokens))
    """
    tokens = []
    marker_start = None
    marker_end = None

    for child in pattern_elem:
        if child.tag == "token":
            tokens.append(child)
        elif child.tag == "marker":
            # Mark the start of marker region
            marker_start = len(tokens)
            for token in child.findall("token"):
                tokens.append(token)
            # Mark the end of marker region
            marker_end = len(tokens)

    # If no marker, entire pattern is the error
    if marker_start is None:
        marker_start = 0
        marker_end = len(tokens)

    return tokens, marker_start, marker_end


def format_pattern(pattern_elem, indent="                    "):
    """Format a pattern (or antipattern) as XML."""
    lines = [f"{indent}<pattern>"]

    # Extract tokens with marker info (we don't use marker info here, just tokens)
    all_tokens, _, _ = extract_pattern_with_marker(pattern_elem)

    for token in all_tokens:
        text, attrs, exceptions = extract_token(token)
        # Skip empty tokens without regexp or postag
        if not text and "regexp" not in attrs and "postag" not in attrs:
            continue
        lines.append(format_token(text, attrs, exceptions))

    lines.append(f"{indent}</pattern>")
    return "\n".join(lines)


def format_antipattern(antipattern_elem):
    """Format antipattern as XML."""
    lines = ["                    <antipattern>"]

    # Collect tokens (handle marker elements too)
    all_tokens = []
    for child in antipattern_elem:
        if child.tag == "token":
            all_tokens.append(child)
        elif child.tag == "marker":
            for token in child.findall("token"):
                all_tokens.append(token)

    for token in all_tokens:
        text, attrs, exceptions = extract_token(token)
        # Skip empty tokens without regexp or postag
        if not text and "regexp" not in attrs and "postag" not in attrs:
            continue
        lines.append(format_token(text, attrs, exceptions))

    lines.append("                    </antipattern>")
    return "\n".join(lines)


def extract_suggestion_text(suggestion_elem):
    """Extract suggestion text, converting <match> to back-references."""
    converted, success = convert_match_to_backref(suggestion_elem)
    if success and converted:
        return escape_xml(converted)
    elif suggestion_elem.text:
        return escape_xml(suggestion_elem.text.strip())
    return ""


def extract_message_text(message_elem):
    """Extract message text, handling embedded suggestions."""
    if message_elem is None:
        return "Grammar correction"

    # Get all text parts
    parts = []
    if message_elem.text:
        parts.append(message_elem.text)

    for child in message_elem:
        if child.tag == "suggestion":
            # Include suggestion text in message
            sugg_text = extract_suggestion_text(child)
            parts.append(f'"{sugg_text}"')
        elif child.tag == "match":
            # Convert match to backref representation
            no = child.get("no", "?")
            parts.append(f"[match {no}]")
        if child.tail:
            parts.append(child.tail)

    text = "".join(parts).strip()
    if not text:
        text = "Grammar correction"
    return escape_xml(text)


def extract_rule(rule_elem, category_id, rulegroup_id=None, rulegroup_name=None):
    """Extract and format a single rule."""
    # Use rule's own ID, or fall back to rulegroup ID with index
    rule_id = rule_elem.get("id")
    if not rule_id and rulegroup_id:
        rule_id = rulegroup_id
    elif not rule_id:
        rule_id = "UNKNOWN"

    rule_name = rule_elem.get("name")
    if not rule_name and rulegroup_name:
        rule_name = rulegroup_name
    elif not rule_name:
        rule_name = ""
    prio = rule_elem.get("prio", "")
    tags = rule_elem.get("tags", "")

    # Extract pattern with marker info
    pattern = rule_elem.find("pattern")
    tokens, marker_start, marker_end = extract_pattern_with_marker(pattern)

    # Build attributes (escape values)
    attrs = [f'id="{escape_xml(rule_id)}"', f'name="{escape_xml(rule_name)}"']
    if prio:
        attrs.append(f'prio="{escape_xml(prio)}"')
    if tags:
        attrs.append(f'tags="{escape_xml(tags)}"')
    # Add marker position - only if it's a partial marker (not covering entire pattern)
    if marker_start != 0 or marker_end != len(tokens):
        attrs.append(f'marker_start="{marker_start}"')
        attrs.append(f'marker_end="{marker_end}"')

    lines = [f"                <rule {' '.join(attrs)}>"]

    # Add antipatterns
    for antipattern in rule_elem.findall("antipattern"):
        lines.append(format_antipattern(antipattern))

    # Add pattern
    lines.append(format_pattern(pattern))

    # Add message
    message = rule_elem.find("message")
    msg_text = extract_message_text(message)
    lines.append(f"                    <message>{msg_text}</message>")

    # Add suggestion(s) - look for direct suggestions and those in message
    suggestions_found = set()
    for suggestion in rule_elem.findall(".//suggestion"):
        sugg_text = extract_suggestion_text(suggestion)
        if sugg_text and sugg_text not in suggestions_found:
            suggestions_found.add(sugg_text)
            lines.append(f"                    <suggestion>{sugg_text}</suggestion>")

    lines.append("                </rule>")

    return "\n".join(lines)


def process_file(input_path, output_path, max_rules=None):
    """Process LT XML file and extract compatible rules."""
    print(f"Processing {input_path}...")

    # Parse XML
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Collect rules by category
    categories = {}
    total_rules = 0
    extracted_rules = 0
    skip_reasons = {}

    for category in root.findall(".//category"):
        cat_id = category.get("id", "UNKNOWN")
        cat_name = category.get("name", cat_id)

        rules = []

        # Process direct rules
        for rule in category.findall("rule"):
            total_rules += 1
            can_handle, reason = can_handle_rule(rule)
            if can_handle:
                try:
                    rule_xml = extract_rule(rule, cat_id)
                    rules.append(rule_xml)
                    extracted_rules += 1
                except Exception as e:
                    print(f"  Error processing rule {rule.get('id', '?')}: {e}")
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        # Also check rulegroups
        for rulegroup in category.findall("rulegroup"):
            rg_id = rulegroup.get("id")
            rg_name = rulegroup.get("name")
            rule_index = 0
            for rule in rulegroup.findall("rule"):
                total_rules += 1
                can_handle, reason = can_handle_rule(rule)
                if can_handle:
                    try:
                        # Use rulegroup ID with index for rules without their own ID
                        effective_id = rule.get("id") or (
                            f"{rg_id}_{rule_index}" if rg_id else None
                        )
                        if not effective_id:
                            skip_reasons["no_id"] = skip_reasons.get("no_id", 0) + 1
                            continue
                        rule_xml = extract_rule(rule, cat_id, rg_id, rg_name)
                        rules.append(rule_xml)
                        extracted_rules += 1
                        rule_index += 1
                    except Exception as e:
                        print(f"  Error processing rule {rule.get('id', '?')}: {e}")
                else:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        if rules:
            categories[cat_id] = (cat_name, rules)

    print(f"  Total rules: {total_rules}, Extracted: {extracted_rules}")
    print(f"  Skip reasons: {skip_reasons}")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write("<rules>\n")

        for cat_id, (cat_name, rules) in sorted(categories.items()):
            if rules:
                f.write(
                    f'    <category id="{escape_xml(cat_id)}" name="{escape_xml(cat_name)}">\n'
                )
                for rule in rules:
                    f.write(rule + "\n")
                f.write("    </category>\n")

        f.write("</rules>\n")

    print(f"  Written to {output_path}")
    return extracted_rules


def main():
    script_dir = Path(__file__).parent.parent
    ref_dir = script_dir / "reference" / "languagetool"
    rules_dir = script_dir / "rules" / "en"

    # Process grammar rules
    grammar_count = process_file(
        ref_dir / "en-grammar.xml", rules_dir / "lt-grammar-full.xml"
    )

    # Process style rules
    style_count = process_file(
        ref_dir / "en-style.xml", rules_dir / "lt-style-full.xml"
    )

    print(f"\nTotal extracted: {grammar_count + style_count} rules")


if __name__ == "__main__":
    main()
