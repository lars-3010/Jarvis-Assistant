"""Markdown Parser for Obsidian Notes with Semantic Relationships"""
import re
import logging
import yaml
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regular Expressions for Markdown elements
FRONTMATTER_PATTERN = r"^---\s*\n(.*?)\n---\s*\n"
INTERNAL_LINK_PATTERN = r"\[\[([^\]]+?)(?:\|([^\]]+?))?\]\]"
HEADING_PATTERN = r"^(#+)\s+(.*?)(?:\s*#+)?$"
TAG_PATTERN = r"(?:^|\s)#([a-zA-Z0-9_-]+)"

# Semantic relationship types
SEMANTIC_RELATIONS = [
    "up::", "similar", "leads_to", "contradicts", "extends", "implements"
]


class MarkdownParser:
    """Parses a Markdown file and extracts all relevant information"""

    def __init__(self, content: str):
        self.content = content
        self.frontmatter: Optional[Dict[str, Any]] = None
        self.content_without_frontmatter: str = ""

    def parse(self) -> Dict[str, Any]:
        """
        Parses the Markdown content.

        Returns:
            Dictionary with extracted information
        """
        try:
            self._parse_frontmatter()

            result = {
                "frontmatter": self.frontmatter or {},
                "links": self._parse_internal_links(),
                "tags": self._parse_tags(),
                "headings": self._parse_headings(),
                "relationships": self._extract_semantic_relationships()
            }

            return result
        except Exception as e:
            logger.error(f"Error parsing Markdown: {e}")
            return {"error": str(e)}

    def _parse_frontmatter(self):
        """
        Extracts YAML frontmatter from the Markdown content.
        """
        match = re.search(FRONTMATTER_PATTERN, self.content, re.DOTALL)
        if not match:
            self.frontmatter = None
            self.content_without_frontmatter = self.content
            return

        frontmatter_text = match.group(1)
        self.content_without_frontmatter = self.content[match.end():]

        try:
            self.frontmatter = yaml.safe_load(frontmatter_text)
            if not isinstance(self.frontmatter, dict):
                logger.warning(f"Frontmatter is not a dictionary: {self.frontmatter}")
                self.frontmatter = {}
        except yaml.YAMLError as e:
            logger.warning(f"Error parsing frontmatter: {e}")
            self.frontmatter = {}

    def _extract_semantic_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extracts semantic relationships from frontmatter.

        Returns:
            Dictionary of relationship types to targets.
        """
        relationships: Dict[str, List[Dict[str, Any]]] = {}

        if not self.frontmatter:
            return relationships

        for rel_type in SEMANTIC_RELATIONS:
            if rel_type in self.frontmatter:
                rel_value = self.frontmatter[rel_type]

                if rel_value is None:
                    continue

                targets_list: List[Dict[str, Any]] = []

                if isinstance(rel_value, str):
                    cleaned_value = rel_value.strip()
                    if cleaned_value:
                        targets_list.append({"target": cleaned_value})

                elif isinstance(rel_value, list):
                    for item in rel_value:
                        target_str: Optional[str] = None
                        if isinstance(item, str) and item.strip():
                            target_str = item.strip()
                        elif isinstance(item, dict) and "target" in item:
                            target_val = item["target"]
                            if isinstance(target_val, str) and target_val.strip():
                                target_str = target_val.strip()

                        if target_str:
                            targets_list.append({"target": target_str})

                if targets_list:
                    relationships[rel_type] = targets_list

        return self._normalize_targets(relationships)

    def _parse_internal_links(self) -> List[Dict[str, str]]:
        """
        Extracts internal links from Markdown content.

        Returns:
            List of dictionaries with link targets and optional aliases.
        """
        links = []
        for match in re.finditer(INTERNAL_LINK_PATTERN, self.content_without_frontmatter):
            link_text = match.group(1)

            if match.group(2):
                target = match.group(1).strip()
                alias = match.group(2).strip()
            else:
                target = match.group(1).strip()
                alias = None

            if not target.lower().endswith('.md'):
                target += '.md'

            links.append({
                "target": target,
                "alias": alias
            })

        return links

    def _parse_tags(self) -> List[str]:
        """
        Extracts tags from frontmatter and markdown content.

        Returns:
            List of all found tags.
        """
        tags = set()

        if self.frontmatter and 'tags' in self.frontmatter:
            fm_tags = self.frontmatter['tags']
            if isinstance(fm_tags, list):
                for tag in fm_tags:
                    tags.add(str(tag))
            elif isinstance(fm_tags, str):
                for tag in fm_tags.split(','):
                    tags.add(tag.strip())

        for match in re.finditer(TAG_PATTERN, self.content_without_frontmatter):
            tag = match.group(1)
            tags.add(tag)

        return list(tags)

    def _parse_headings(self) -> List[Dict[str, Any]]:
        """
        Extracts headings from Markdown content.

        Returns:
            List of headings with level, text and ID.
        """
        headings = []
        for line in self.content_without_frontmatter.split('\n'):
            match = re.match(HEADING_PATTERN, line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append({
                    "level": level,
                    "text": text,
                    "id": text.lower().replace(' ', '-')
                })

        return headings

    def _normalize_targets(self, relationships: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Normalize relationship targets by ensuring all have .md extension.

        Args:
            relationships: Dictionary of relationship type to targets.

        Returns:
            Normalized relationships dictionary.
        """
        normalized = {}

        for rel_type, targets in relationships.items():
            normalized_targets = []
            for target_info in targets:
                target = target_info["target"]
                if not target.lower().endswith('.md'):
                    target = f"{target}.md"

                normalized_target = target_info.copy()
                normalized_target["target"] = target
                normalized_targets.append(normalized_target)

            normalized[rel_type] = normalized_targets

        return normalized
