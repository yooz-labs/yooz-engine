#!/usr/bin/env python3
"""
Extract LanguageTool rules compatible with our simple tokenizer.
Filters out rules with POS tagging, complex patterns, and dynamic suggestions.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


def is_simple_rule(rule_element):
    """Check if rule is simple enough for our tokenizer."""
    xml_str = ET.tostring(rule_element, encoding="unicode")

    # Skip rules with POS tagging
    if "postag=" in xml_str or "postag_regexp=" in xml_str:
        return False, "postag"

    # Skip rules with antipatterns (negative matching)
    if "<antipattern>" in xml_str:
        return False, "antipattern"

    # Skip rules with dynamic suggestions (<match> elements)
    if "<match " in xml_str or "<match>" in xml_str:
        return False, "match"

    # Skip rules without proper suggestion
    suggestion = rule_element.find(".//suggestion")
    if suggestion is None or suggestion.text is None or not suggestion.text.strip():
        return False, "no_suggestion"

    # Skip rules with backreferences in suggestion (e.g., \1, \2)
    import re

    suggestion_text = suggestion.text if suggestion.text else ""
    if re.search(r"\\[0-9]", suggestion_text):
        return False, "backreference"

    # Skip rules without pattern
    pattern = rule_element.find(".//pattern")
    if pattern is None:
        return False, "no_pattern"

    # Skip rules with no tokens in pattern (including inside markers)
    tokens = pattern.findall(".//token")
    if not tokens:
        return False, "no_tokens"

    # Check all tokens have text content (allow regexp tokens without text)
    for token in tokens:
        if token.text is None or not token.text.strip():
            # Allow empty tokens only if they have regexp attribute
            if token.get("regexp") is None:
                return False, "empty_token"

    # Skip rules without id (anonymous rules)
    if rule_element.get("id") is None:
        return False, "no_id"

    # For rules with markers, check if we can build a full replacement
    # We handle simple cases where marker covers single word being replaced
    if "<marker>" in xml_str:
        marker = pattern.find(".//marker")
        if marker is not None:
            marker_tokens = marker.findall(".//token")
            # Only handle single-token markers for now
            if len(marker_tokens) != 1:
                return False, "complex_marker"

    return True, "ok"


def simplify_rule(rule_element):
    """Create a simplified version of the rule for our format."""
    # Create new rule element with only what we need
    new_rule = ET.Element("rule")
    new_rule.set("id", rule_element.get("id", "UNKNOWN"))
    new_rule.set("name", rule_element.get("name", rule_element.get("id", "Unknown")))

    # Copy pattern (handle markers by flattening)
    pattern = rule_element.find(".//pattern")
    new_pattern = ET.SubElement(new_rule, "pattern")

    # Track tokens and which one is the marked token
    all_tokens = []
    marked_index = -1

    # Iterate through pattern children to preserve order
    def process_element(elem):
        nonlocal marked_index
        for child in elem:
            if child.tag == "marker":
                marked_index = len(all_tokens)
                # Add tokens from inside marker
                for token in child.findall(".//token"):
                    all_tokens.append(token)
            elif child.tag == "token":
                all_tokens.append(child)

    process_element(pattern)

    # Add all tokens to new pattern
    for token in all_tokens:
        new_token = ET.SubElement(new_pattern, "token")
        if token.text:
            new_token.text = token.text
        # Copy supported attributes
        if token.get("regexp"):
            new_token.set("regexp", token.get("regexp"))
        if token.get("case_sensitive"):
            new_token.set("case_sensitive", token.get("case_sensitive"))

    # Copy message
    message = rule_element.find(".//message")
    new_message = ET.SubElement(new_rule, "message")
    if message is not None and message.text:
        new_message.text = message.text.strip()
    else:
        new_message.text = "Grammar correction"

    # Build suggestion (for marker rules, replace only the marked token)
    suggestion = rule_element.find(".//suggestion")
    suggestion_text = suggestion.text.strip() if suggestion.text else ""

    if marked_index >= 0 and len(all_tokens) > 1:
        # Build full replacement: tokens before marker + suggestion + tokens after marker
        parts = []
        for i, token in enumerate(all_tokens):
            if i == marked_index:
                parts.append(suggestion_text)
            else:
                parts.append(token.text if token.text else "")
        suggestion_text = " ".join(parts)

    new_suggestion = ET.SubElement(new_rule, "suggestion")
    new_suggestion.text = suggestion_text

    return new_rule


def extract_simple_rules(input_file, output_file):
    """Extract rules compatible with our simple tokenizer."""

    tree = ET.parse(input_file)
    root = tree.getroot()

    # Create new rules structure
    new_root = ET.Element("rules")

    stats = {"total": 0, "extracted": 0, "skipped": {}, "by_category": {}}

    for category in root.findall(".//category"):
        cat_id = category.get("id", "UNKNOWN")
        cat_name = category.get("name", cat_id)

        new_category = ET.SubElement(new_root, "category")
        new_category.set("id", cat_id)
        new_category.set("name", cat_name)

        rules_added = 0

        for rule in category.findall(".//rule"):
            stats["total"] += 1

            is_simple, reason = is_simple_rule(rule)
            if not is_simple:
                stats["skipped"][reason] = stats["skipped"].get(reason, 0) + 1
                continue

            # Simplify and add rule
            simplified = simplify_rule(rule)
            new_category.append(simplified)
            rules_added += 1
            stats["extracted"] += 1

        if rules_added > 0:
            stats["by_category"][cat_id] = rules_added
        else:
            # Remove empty category
            new_root.remove(new_category)

    # Write output
    tree = ET.ElementTree(new_root)
    ET.indent(tree, space="    ")
    tree.write(output_file, encoding="unicode", xml_declaration=True)

    return stats


def main():
    ref_dir = Path(__file__).parent.parent / "reference" / "languagetool"
    output_dir = Path(__file__).parent.parent / "rules" / "en"

    print("Extracting simple rules from LanguageTool...\n")

    # Extract grammar rules
    grammar_stats = extract_simple_rules(
        ref_dir / "en-grammar.xml", output_dir / "lt-grammar-simple.xml"
    )

    print("Grammar rules:")
    print(f"  Total: {grammar_stats['total']}")
    print(f"  Extracted: {grammar_stats['extracted']}")
    print("\n  Skipped by reason:")
    for reason, count in sorted(grammar_stats["skipped"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print("\n  By category:")
    for cat, count in sorted(grammar_stats["by_category"].items()):
        print(f"    {cat}: {count}")

    # Extract style rules
    style_stats = extract_simple_rules(
        ref_dir / "en-style.xml", output_dir / "lt-style-simple.xml"
    )

    print("\nStyle rules:")
    print(f"  Total: {style_stats['total']}")
    print(f"  Extracted: {style_stats['extracted']}")
    print("\n  Skipped by reason:")
    for reason, count in sorted(style_stats["skipped"].items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print("\n  By category:")
    for cat, count in sorted(style_stats["by_category"].items()):
        print(f"    {cat}: {count}")

    total_extracted = grammar_stats["extracted"] + style_stats["extracted"]
    total_skipped = sum(grammar_stats["skipped"].values()) + sum(
        style_stats["skipped"].values()
    )
    print(f"\n=== TOTAL EXTRACTED: {total_extracted} rules ===")
    print(f"=== TOTAL SKIPPED: {total_skipped} rules ===")
    print("\nOutput files:")
    print(f"  {output_dir / 'lt-grammar-simple.xml'}")
    print(f"  {output_dir / 'lt-style-simple.xml'}")


if __name__ == "__main__":
    main()
