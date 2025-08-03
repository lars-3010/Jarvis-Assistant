import pytest
from jarvis.services.graph.parser import MarkdownParser

def test_parse_frontmatter_only():
    content = """---
title: Test Note
tags: [tag1, tag2]
---
# Heading
Some content."""
    parser = MarkdownParser(content)
    parsed_data = parser.parse()
    assert parsed_data["frontmatter"] == {"title": "Test Note", "tags": ["tag1", "tag2"]}
    assert parser.content_without_frontmatter.strip().startswith("# Heading")

def test_parse_no_frontmatter():
    content = """# Heading
Some content.
[[Link Target|Link Alias]]"""
    parser = MarkdownParser(content)
    parsed_data = parser.parse()
    assert parsed_data["frontmatter"] == {}
    assert parser.content_without_frontmatter.strip().startswith("# Heading")

def test_extract_semantic_relationships():
    content = """---
    "up::":
      - Parent Note.md
    similar:
      - Related Note.md
      - Another Related.md
---
Content."""
    parser = MarkdownParser(content)
    parsed_data = parser.parse()
    relationships = parsed_data["relationships"]
    assert "up::" in relationships
    assert relationships["up::"][0]["target"] == "Parent Note.md"
    assert "similar" in relationships
    assert relationships["similar"][0]["target"] == "Related Note.md"
    assert relationships["similar"][1]["target"] == "Another Related.md"

def test_parse_internal_links():
    content = """Some text with [[Target Note]] and [[Another Note|Alias]]."""
    parser = MarkdownParser(content)
    parsed_data = parser.parse()
    links = parsed_data["links"]
    assert len(links) == 2
    assert links[0]["target"] == "Target Note.md"
    assert links[0]["alias"] is None
    assert links[1]["target"] == "Another Note.md"
    assert links[1]["alias"] == "Alias"

def test_parse_tags():
    content = """---
tags: [frontmatter_tag]
---
# Heading #content_tag
Some text #another_tag."""
    parser = MarkdownParser(content)
    parsed_data = parser.parse()
    tags = parsed_data["tags"]
    assert "frontmatter_tag" in tags
    assert "content_tag" in tags
    assert "another_tag" in tags
    assert len(tags) == 3

def test_parse_headings():
    content = """# Level 1
## Level 2
### Level 3"""
    parser = MarkdownParser(content)
    parsed_data = parser.parse()
    headings = parsed_data["headings"]
    assert len(headings) == 3
    assert headings[0]["text"] == "Level 1"
    assert headings[0]["level"] == 1
    assert headings[1]["text"] == "Level 2"
    assert headings[1]["level"] == 2

def test_normalize_targets():
    relationships = {
        "rel1": [{"target": "Note A"}],
        "rel2": [{"target": "Note B.md"}]
    }
    parser = MarkdownParser("") # Content doesn't matter for this test
    parser.frontmatter = {} # Mock frontmatter for _normalize_targets
    normalized = parser._normalize_targets(relationships)
    assert normalized["rel1"][0]["target"] == "Note A.md"
    assert normalized["rel2"][0]["target"] == "Note B.md"
