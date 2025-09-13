"""
Markdown parser for Obsidian notes with semantic relationships.

This module handles parsing of Obsidian markdown files, extracting frontmatter,
internal links, tags, headings, callouts, and semantic relationships.
"""

import re
from typing import Any

import yaml

from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)

# Regular expressions for Markdown elements
FRONTMATTER_PATTERN = r"^---\s*\n(.*?)\n---\s*\n"
INTERNAL_LINK_PATTERN = r"\[\[(.*?)(?:\|(.*?))?\]\]"
HEADING_PATTERN = r"^(#+)\s+(.*?)(?:\s*#+)?$"
TAG_PATTERN = r"(?:^|\s)#([a-zA-Z0-9_/-]+)(?:\s|$)"
CALLOUT_PATTERN = r"^\s*>\s*\[!([a-zA-Z]+)\](.*?)$"
BLOCK_REFERENCE_PATTERN = r"\^([a-zA-Z0-9_-]+)$"

# Semantic relationship types
SEMANTIC_RELATIONS = [
    "up", "similar", "leads_to", "contradicts", "extends", "implements",
    "see_also", "relates_to", "depends_on", "inspired_by"
]


class MarkdownParser:
    """Comprehensive Markdown parser for Obsidian notes."""

    def __init__(self, extract_semantic: bool = True, normalize_links: bool = True):
        """Initialize the parser.
        
        Args:
            extract_semantic: Whether to extract semantic relationships
            normalize_links: Whether to normalize link targets
        """
        self.extract_semantic = extract_semantic
        self.normalize_links = normalize_links

    def parse(self, content: str) -> dict[str, Any]:
        """Parse a Markdown file and extract all relevant information.
        
        Args:
            content: Markdown content
        
        Returns:
            Dictionary with extracted information
        """
        try:
            # Extract frontmatter
            frontmatter, content_without_frontmatter = self.parse_frontmatter(content)

            # Build result dictionary
            result = {
                "frontmatter": frontmatter or {},
                "content_without_frontmatter": content_without_frontmatter,
                "links": self.parse_internal_links(content_without_frontmatter),
                "tags": self.parse_tags(content_without_frontmatter, frontmatter),
                "headings": self.parse_headings(content_without_frontmatter),
                "callouts": self.parse_callouts(content_without_frontmatter),
                "block_references": self.parse_block_references(content_without_frontmatter),
                "word_count": len(content_without_frontmatter.split()),
                "character_count": len(content_without_frontmatter)
            }

            # Add semantic relationships if requested
            if self.extract_semantic and frontmatter:
                relationships = self.extract_semantic_relationships(frontmatter)
                if self.normalize_links:
                    relationships = self.normalize_targets(relationships)
                result["relationships"] = relationships

            return result

        except Exception as e:
            logger.error(f"Error parsing Markdown: {e}")
            return {"error": str(e), "frontmatter": {}, "links": [], "tags": []}

    def parse_frontmatter(self, content: str) -> tuple[dict[str, Any] | None, str]:
        """Extract YAML frontmatter from a Markdown file.
        
        Args:
            content: Markdown content
        
        Returns:
            Tuple of (frontmatter_dict, remaining_content)
        """
        match = re.search(FRONTMATTER_PATTERN, content, re.DOTALL)
        if not match:
            return None, content

        frontmatter_text = match.group(1)
        remaining_content = content[match.end():]

        try:
            frontmatter = yaml.safe_load(frontmatter_text)
            if not isinstance(frontmatter, dict):
                logger.warning(f"Frontmatter is not a dictionary: {frontmatter}")
                frontmatter = {}
            return frontmatter, remaining_content
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing frontmatter: {e}")
            return {}, remaining_content

    def extract_semantic_relationships(self, frontmatter: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
        """Extract semantic relationships from frontmatter.

        Args:
            frontmatter: Parsed frontmatter dictionary or None.

        Returns:
            Dictionary of relationship types to targets.
        """
        relationships: dict[str, list[dict[str, Any]]] = {}

        if not frontmatter:
            return relationships

        for rel_type in SEMANTIC_RELATIONS:
            if rel_type in frontmatter:
                rel_value = frontmatter[rel_type]

                # Skip processing if the value is None
                if rel_value is None:
                    logger.debug(f"Skipping relationship '{rel_type}' because its value is None.")
                    continue

                targets_list: list[dict[str, Any]] = []

                # Handle different formats
                if isinstance(rel_value, str):
                    cleaned_value = rel_value.strip()
                    if cleaned_value:
                        targets_list.append({"target": cleaned_value})
                    else:
                        logger.debug(f"Skipping empty string value for relationship '{rel_type}'.")

                elif isinstance(rel_value, list):
                    for item in rel_value:
                        target_str: str | None = None

                        # Item is a non-empty string
                        if isinstance(item, str) and item.strip():
                            target_str = item.strip()
                        # Item is a dict with target key
                        elif isinstance(item, dict) and "target" in item:
                            target_val = item["target"]
                            if isinstance(target_val, str) and target_val.strip():
                                target_str = target_val.strip()
                                # Preserve additional metadata from dict
                                target_dict = item.copy()
                                target_dict["target"] = target_str
                                targets_list.append(target_dict)
                                continue
                            else:
                                logger.warning(f"Invalid 'target' in item for '{rel_type}': {item}")
                        elif item is not None:
                            logger.warning(f"Unexpected item type '{type(item)}' in list for relationship '{rel_type}': {item}")

                        # Add simple string target
                        if target_str:
                            targets_list.append({"target": target_str})

                else:
                    logger.warning(f"Unexpected data type '{type(rel_value)}' for relationship '{rel_type}'. Expected string or list. Value: {rel_value}")

                # Only add the relationship type if we found valid targets
                if targets_list:
                    relationships[rel_type] = targets_list
                else:
                    logger.debug(f"No valid targets found for relationship '{rel_type}' after processing value: {rel_value}")

        return relationships

    def parse_internal_links(self, content: str) -> list[dict[str, Any]]:
        """Extract internal links from Markdown content.
        
        Args:
            content: Markdown content
        
        Returns:
            List of dictionaries with link targets and optional aliases
        """
        links = []
        for match in re.finditer(INTERNAL_LINK_PATTERN, content):
            full_match = match.group(0)
            link_text = match.group(1)

            # Handle optional alias with | syntax
            if "|" in link_text:
                parts = link_text.split("|", 1)
                target = parts[0].strip()
                alias = parts[1].strip()
            else:
                target = link_text.strip()
                alias = None

            # Handle section links (Note#Section)
            section = None
            if "#" in target:
                target_parts = target.split("#", 1)
                target = target_parts[0].strip()
                section = target_parts[1].strip()

            # Handle block references (Note^block)
            block_ref = None
            if "^" in target:
                target_parts = target.split("^", 1)
                target = target_parts[0].strip()
                block_ref = target_parts[1].strip()

            # Add .md extension if missing and target is not empty
            if target and not target.lower().endswith('.md'):
                target += '.md'

            link_info = {
                "target": target,
                "alias": alias,
                "section": section,
                "block_ref": block_ref,
                "full_match": full_match,
                "start": match.start(),
                "end": match.end()
            }

            links.append(link_info)

        return links

    def parse_tags(self, content: str, frontmatter: dict[str, Any] | None = None) -> list[str]:
        """Extract tags from frontmatter and markdown content.
        
        Args:
            content: Markdown content
            frontmatter: Already extracted frontmatter
        
        Returns:
            List of all found tags
        """
        tags = set()

        # Tags from frontmatter
        if frontmatter:
            # Check multiple possible frontmatter fields
            for field in ['tags', 'tag', 'category', 'categories']:
                if field in frontmatter:
                    fm_tags = frontmatter[field]
                    if isinstance(fm_tags, list):
                        for tag in fm_tags:
                            if tag:  # Skip empty tags
                                tags.add(str(tag).strip())
                    elif isinstance(fm_tags, str):
                        for tag in fm_tags.split(','):
                            tag = tag.strip()
                            if tag:  # Skip empty tags
                                tags.add(tag)

        # Tags from content (Markdown format: #tag)
        for match in re.finditer(TAG_PATTERN, content):
            tag = match.group(1).strip()
            if tag:  # Skip empty tags
                tags.add(tag)

        return sorted(list(tags))

    def parse_headings(self, content: str) -> list[dict[str, Any]]:
        """Extract headings from Markdown content.
        
        Args:
            content: Markdown content
        
        Returns:
            List of headings with level, text, and ID
        """
        headings = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines):
            match = re.match(HEADING_PATTERN, line)
            if match:
                level = len(match.group(1))  # Count number of # symbols
                text = match.group(2).strip()

                # Generate slug-style ID
                heading_id = self._generate_heading_id(text)

                headings.append({
                    "level": level,
                    "text": text,
                    "id": heading_id,
                    "line_number": line_num + 1
                })

        return headings

    def parse_callouts(self, content: str) -> list[dict[str, Any]]:
        """Extract callouts from Markdown content.
        
        Args:
            content: Markdown content
        
        Returns:
            List of callouts with type and content
        """
        callouts = []
        current_callout = None
        current_type = None
        current_content = []
        current_start_line = None

        lines = content.split('\n')

        for line_num, line in enumerate(lines):
            # New callout starts
            match = re.match(CALLOUT_PATTERN, line)
            if match:
                # Finish previous callout if exists
                if current_callout is not None:
                    callouts.append({
                        "type": current_type,
                        "content": '\n'.join(current_content).strip(),
                        "start_line": current_start_line,
                        "end_line": line_num
                    })

                # Start new callout
                current_type = match.group(1).lower()
                current_content = [match.group(2).strip()] if match.group(2).strip() else []
                current_callout = True
                current_start_line = line_num + 1

            # Continue current callout
            elif current_callout and line.strip().startswith('> '):
                current_content.append(line.strip()[2:])

            # Callout ends
            elif current_callout and not line.strip().startswith('>'):
                callouts.append({
                    "type": current_type,
                    "content": '\n'.join(current_content).strip(),
                    "start_line": current_start_line,
                    "end_line": line_num
                })
                current_callout = None
                current_type = None
                current_content = []
                current_start_line = None

        # Add the last callout if exists
        if current_callout:
            callouts.append({
                "type": current_type,
                "content": '\n'.join(current_content).strip(),
                "start_line": current_start_line,
                "end_line": len(lines)
            })

        return callouts

    def parse_block_references(self, content: str) -> list[dict[str, Any]]:
        """Extract block references from Markdown content.
        
        Args:
            content: Markdown content
        
        Returns:
            List of block references with IDs and line numbers
        """
        block_refs = []
        lines = content.split('\n')

        for line_num, line in enumerate(lines):
            match = re.search(BLOCK_REFERENCE_PATTERN, line.strip())
            if match:
                block_id = match.group(1)
                block_refs.append({
                    "id": block_id,
                    "line_number": line_num + 1,
                    "content": line.strip()
                })

        return block_refs

    def normalize_targets(self, relationships: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        """Normalize relationship targets by ensuring all have .md extension.
        
        Args:
            relationships: Dictionary of relationship type to targets
        
        Returns:
            Normalized relationships dictionary
        """
        normalized = {}

        for rel_type, targets in relationships.items():
            normalized_targets = []
            for target_info in targets:
                target = target_info["target"]
                # Add .md extension if missing
                if target and not target.lower().endswith('.md'):
                    target = f"{target}.md"

                # Create a new dict to avoid modifying the original
                normalized_target = target_info.copy()
                normalized_target["target"] = target
                normalized_targets.append(normalized_target)

            normalized[rel_type] = normalized_targets

        return normalized

    def extract_content_summary(self, content: str, max_length: int = 200) -> str:
        """Extract a summary from the content.
        
        Args:
            content: Markdown content
            max_length: Maximum length of summary
        
        Returns:
            Content summary
        """
        # Remove frontmatter if present
        _, content_without_frontmatter = self.parse_frontmatter(content)

        # Remove markdown formatting
        clean_content = re.sub(r'#+\s+', '', content_without_frontmatter)  # Headers
        clean_content = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_content)  # Bold
        clean_content = re.sub(r'\*(.*?)\*', r'\1', clean_content)  # Italic
        clean_content = re.sub(r'\[\[(.*?)\]\]', r'\1', clean_content)  # Links
        clean_content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_content)  # External links
        clean_content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', clean_content)  # Images
        clean_content = re.sub(r'`([^`]+)`', r'\1', clean_content)  # Inline code
        clean_content = re.sub(r'```.*?```', '', clean_content, flags=re.DOTALL)  # Code blocks

        # Clean up whitespace
        clean_content = ' '.join(clean_content.split())

        # Truncate to max length
        if len(clean_content) > max_length:
            clean_content = clean_content[:max_length].rsplit(' ', 1)[0] + '...'

        return clean_content.strip()

    def _generate_heading_id(self, text: str) -> str:
        """Generate a slug-style ID from heading text.
        
        Args:
            text: Heading text
        
        Returns:
            Slug-style ID
        """
        # Convert to lowercase and replace spaces with hyphens
        slug = text.lower()
        slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special characters
        slug = re.sub(r'[-\s]+', '-', slug)  # Replace spaces/hyphens with single hyphen
        slug = slug.strip('-')  # Remove leading/trailing hyphens

        return slug or 'heading'


# Convenience functions for backward compatibility
def parse_markdown(content: str, extract_semantic: bool = True) -> dict[str, Any]:
    """Parse a Markdown file and extract all relevant information.
    
    Args:
        content: Markdown content
        extract_semantic: Whether to extract semantic relationships
    
    Returns:
        Dictionary with extracted information
    """
    parser = MarkdownParser(extract_semantic=extract_semantic)
    return parser.parse(content)


def parse_frontmatter(content: str) -> tuple[dict[str, Any] | None, str]:
    """Extract YAML frontmatter from a Markdown file.
    
    Args:
        content: Markdown content
    
    Returns:
        Tuple of (frontmatter_dict, remaining_content)
    """
    parser = MarkdownParser()
    return parser.parse_frontmatter(content)


def parse_internal_links(content: str) -> list[dict[str, Any]]:
    """Extract internal links from Markdown content.
    
    Args:
        content: Markdown content
    
    Returns:
        List of dictionaries with link targets and optional aliases
    """
    parser = MarkdownParser()
    return parser.parse_internal_links(content)
