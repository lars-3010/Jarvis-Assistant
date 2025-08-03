"""
Enhanced link extraction with comprehensive pattern matching.

This module provides robust link extraction from Obsidian vault content,
handling various link formats including wikilinks, markdown links, and embedded media.
The implementation fixes critical bugs in link extraction and provides comprehensive
validation and normalization of extracted links.
"""

import os
import re
from pathlib import Path
from re import Pattern

import networkx as nx

from jarvis.services.vault.reader import VaultReader
from jarvis.utils.logging import setup_logging

from ..models.data_models import Link, LinkStatistics, ValidationResult
from ..models.exceptions import LinkExtractionError

logger = setup_logging(__name__)


class LinkExtractor:
    """Enhanced link extraction with comprehensive pattern matching."""

    def __init__(self, vault_reader: VaultReader):
        """Initialize the link extractor.
        
        Args:
            vault_reader: VaultReader instance for file operations
        """
        self.vault_reader = vault_reader
        self.link_patterns = self._compile_link_patterns()
        self._file_cache: dict[str, set[str]] = {}
        self._broken_links_cache: set[str] = set()

    def _compile_link_patterns(self) -> dict[str, Pattern[str]]:
        """Compile regex patterns for different link types.
        
        Returns:
            Dictionary of compiled regex patterns
        """
        patterns = {
            # Wikilinks: [[Link]] or [[Link|Display Text]]
            'wikilink': re.compile(
                r'\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]',
                re.MULTILINE | re.DOTALL
            ),

            # Markdown links: [Display Text](Link)
            'markdown_link': re.compile(
                r'\[([^\]]+)\]\(([^)]+)\)',
                re.MULTILINE
            ),

            # Embedded links: ![[Link]] or ![[Link|Display Text]]
            'embedded_link': re.compile(
                r'!\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]',
                re.MULTILINE | re.DOTALL
            ),

            # Tag links: #tag or #nested/tag
            'tag_link': re.compile(
                r'(?:^|[\s\(])#([a-zA-Z0-9_/-]+)(?=[\s\)\.,!?;:]|$)',
                re.MULTILINE
            ),

            # Reference-style links: [Display Text][Reference]
            'reference_link': re.compile(
                r'\[([^\]]+)\]\[([^\]]+)\]',
                re.MULTILINE
            ),

            # URL links: http(s)://...
            'url_link': re.compile(
                r'https?://[^\s\)]+',
                re.MULTILINE
            )
        }

        logger.debug(f"Compiled {len(patterns)} link pattern types")
        return patterns

    def extract_all_links(self) -> tuple[nx.DiGraph, LinkStatistics]:
        """Extract all links and build comprehensive graph with robust error handling.
        
        Returns:
            Tuple of (directed graph, link statistics)
            
        Raises:
            LinkExtractionError: If critical extraction failure occurs
        """
        logger.info("Starting comprehensive link extraction with enhanced error handling")
        
        # Initialize tracking variables
        all_links: list[Link] = []
        link_counts = {}
        processed_files = 0
        failed_files = 0
        corrupted_files = []
        permission_errors = []
        encoding_errors = []

        try:
            # Get all markdown files with error handling
            try:
                markdown_files = self.vault_reader.get_markdown_files()
                logger.info(f"Found {len(markdown_files)} markdown files to process")
            except Exception as e:
                logger.error(f"Failed to get markdown files from vault: {e}")
                raise LinkExtractionError(f"Cannot access vault files: {e}") from e

            if not markdown_files:
                logger.warning("No markdown files found in vault")
                return nx.DiGraph(), LinkStatistics()

            # Build file existence cache with error handling
            try:
                self._build_file_cache(markdown_files)
            except Exception as e:
                logger.warning(f"Failed to build file cache: {e}, continuing without cache optimization")
                self._file_cache = {'all_files': set(str(f) for f in markdown_files)}

            # Extract links from all files with comprehensive error handling
            for file_index, file_path in enumerate(markdown_files):
                try:
                    # Read file with specific error handling
                    try:
                        content, metadata = self.vault_reader.read_file(str(file_path))
                    except PermissionError as e:
                        logger.warning(f"Permission denied reading {file_path}: {e}")
                        permission_errors.append(str(file_path))
                        failed_files += 1
                        continue
                    except UnicodeDecodeError as e:
                        logger.warning(f"Encoding error reading {file_path}: {e}")
                        encoding_errors.append(str(file_path))
                        failed_files += 1
                        continue
                    except FileNotFoundError as e:
                        logger.warning(f"File not found {file_path}: {e}")
                        failed_files += 1
                        continue

                    # Extract links with error handling
                    try:
                        file_links = self.extract_links_from_content(content, str(file_path))
                        all_links.extend(file_links)
                        link_counts[str(file_path)] = len(file_links)
                        processed_files += 1
                    except Exception as e:
                        logger.error(f"Link extraction failed for {file_path}: {e}")
                        corrupted_files.append(str(file_path))
                        failed_files += 1
                        # Continue processing other files
                        continue

                    # Progress logging
                    if processed_files % 100 == 0:
                        logger.debug(f"Processed {processed_files}/{len(markdown_files)} files "
                                   f"({failed_files} failed)")

                except Exception as e:
                    logger.error(f"Unexpected error processing file {file_path}: {e}")
                    failed_files += 1
                    continue

            # Log processing summary
            logger.info(f"Link extraction completed: {processed_files} processed, {failed_files} failed")
            logger.info(f"Extracted {len(all_links)} total links from {processed_files} files")
            
            if failed_files > 0:
                logger.warning(f"Failed to process {failed_files} files:")
                if permission_errors:
                    logger.warning(f"  - {len(permission_errors)} permission errors")
                if encoding_errors:
                    logger.warning(f"  - {len(encoding_errors)} encoding errors")
                if corrupted_files:
                    logger.warning(f"  - {len(corrupted_files)} corrupted/malformed files")

            # Check if we have enough data to continue
            if processed_files == 0:
                raise LinkExtractionError("No files could be processed successfully")
            
            if processed_files < len(markdown_files) * 0.5:
                logger.warning(f"Only {processed_files}/{len(markdown_files)} files processed successfully")

            # Build graph and compute statistics with error handling
            try:
                graph = self._build_link_graph(all_links, markdown_files)
                logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            except Exception as e:
                logger.error(f"Failed to build link graph: {e}")
                # Create empty graph as fallback
                graph = nx.DiGraph()
                for file_path in markdown_files:
                    graph.add_node(str(file_path))
                logger.warning("Created fallback graph with isolated nodes")

            try:
                statistics = self._compute_link_statistics(all_links, link_counts, len(markdown_files))
                # Add error statistics
                statistics.notes_failed = failed_files
                statistics.permission_errors = len(permission_errors)
                statistics.encoding_errors = len(encoding_errors)
                statistics.corrupted_files = len(corrupted_files)
            except Exception as e:
                logger.error(f"Failed to compute link statistics: {e}")
                # Create basic statistics as fallback
                statistics = LinkStatistics()
                statistics.total_links = len(all_links)
                statistics.notes_failed = failed_files

            return graph, statistics

        except LinkExtractionError:
            # Re-raise LinkExtractionError as-is
            raise
        except Exception as e:
            logger.error(f"Critical link extraction failure: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise LinkExtractionError(f"Critical link extraction failure: {e}") from e

    def extract_links_from_content(self, content: str, source_path: str) -> list[Link]:
        """Extract links from note content with comprehensive error handling.
        
        Args:
            content: Note content to extract links from
            source_path: Path of the source note
            
        Returns:
            List of extracted links (empty list if extraction fails)
        """
        if not content or not content.strip():
            logger.debug(f"Empty or whitespace-only content in {source_path}")
            return []

        links: list[Link] = []
        extraction_errors = []

        try:
            # Extract different types of links with individual error handling
            try:
                wikilinks = self._extract_wikilinks(content, source_path)
                links.extend(wikilinks)
                logger.debug(f"Extracted {len(wikilinks)} wikilinks from {source_path}")
            except Exception as e:
                extraction_errors.append(f"wikilinks: {e}")
                logger.warning(f"Failed to extract wikilinks from {source_path}: {e}")

            try:
                markdown_links = self._extract_markdown_links(content, source_path)
                links.extend(markdown_links)
                logger.debug(f"Extracted {len(markdown_links)} markdown links from {source_path}")
            except Exception as e:
                extraction_errors.append(f"markdown_links: {e}")
                logger.warning(f"Failed to extract markdown links from {source_path}: {e}")

            try:
                embedded_links = self._extract_embedded_links(content, source_path)
                links.extend(embedded_links)
                logger.debug(f"Extracted {len(embedded_links)} embedded links from {source_path}")
            except Exception as e:
                extraction_errors.append(f"embedded_links: {e}")
                logger.warning(f"Failed to extract embedded links from {source_path}: {e}")

            # Log extraction summary
            if extraction_errors:
                logger.warning(f"Partial link extraction from {source_path}: {len(extraction_errors)} errors")

            # Deduplicate links while preserving order with error handling
            unique_links = []
            seen = set()
            deduplication_errors = 0

            for link in links:
                try:
                    link_key = (link.source, link.target, link.link_type)
                    if link_key not in seen:
                        seen.add(link_key)
                        unique_links.append(link)
                except Exception as e:
                    logger.debug(f"Error during deduplication for link {link}: {e}")
                    deduplication_errors += 1

            if deduplication_errors > 0:
                logger.warning(f"Failed to deduplicate {deduplication_errors} links from {source_path}")

            # Validate and normalize links with comprehensive error handling
            validated_links = []
            normalization_errors = 0
            validation_errors = 0

            for link in unique_links:
                try:
                    # Normalize link path
                    try:
                        normalized_target = self._normalize_link_path(link.target, source_path)
                        if normalized_target:
                            link.target = normalized_target
                        else:
                            logger.debug(f"Link normalization returned None for {link.target} from {source_path}")
                            link.is_valid = False
                            validated_links.append(link)
                            continue
                    except Exception as e:
                        logger.debug(f"Failed to normalize link {link.target} from {source_path}: {e}")
                        normalization_errors += 1
                        link.is_valid = False
                        validated_links.append(link)
                        continue

                    # Validate link target
                    try:
                        link.is_valid = self._validate_link_target(normalized_target)
                        validated_links.append(link)
                    except Exception as e:
                        logger.debug(f"Failed to validate link {normalized_target} from {source_path}: {e}")
                        validation_errors += 1
                        link.is_valid = False
                        validated_links.append(link)

                except Exception as e:
                    logger.warning(f"Unexpected error processing link {link} from {source_path}: {e}")
                    # Add link with invalid status as fallback
                    link.is_valid = False
                    validated_links.append(link)

            # Log processing summary
            if normalization_errors > 0:
                logger.warning(f"Failed to normalize {normalization_errors} links from {source_path}")
            if validation_errors > 0:
                logger.warning(f"Failed to validate {validation_errors} links from {source_path}")

            logger.debug(f"Successfully processed {len(validated_links)} links from {source_path} "
                        f"({len([l for l in validated_links if l.is_valid])} valid)")

            return validated_links

        except Exception as e:
            logger.error(f"Critical error extracting links from {source_path}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            # Return empty list as fallback to allow processing to continue
            return []

    def _extract_wikilinks(self, content: str, source_path: str) -> list[Link]:
        """Extract wikilinks from content with error handling.
        
        Args:
            content: Content to extract from
            source_path: Source file path
            
        Returns:
            List of wikilink Link objects
        """
        links = []
        
        try:
            pattern = self.link_patterns['wikilink']
            match_count = 0
            error_count = 0

            for match in pattern.finditer(content):
                try:
                    match_count += 1
                    target = match.group(1).strip() if match.group(1) else ""
                    heading = match.group(2).strip() if match.group(2) else None
                    display_text = match.group(3).strip() if match.group(3) else None

                    if not target:
                        logger.debug(f"Empty wikilink target in {source_path} at match {match_count}")
                        continue

                    # Handle heading links
                    full_target = target
                    if heading:
                        full_target = f"{target}#{heading}"

                    # Calculate line number safely
                    try:
                        line_number = content[:match.start()].count('\n') + 1
                    except Exception as e:
                        logger.debug(f"Failed to calculate line number for wikilink in {source_path}: {e}")
                        line_number = None

                    link = Link(
                        source=source_path,
                        target=full_target,
                        link_type="wikilink",
                        display_text=display_text,
                        line_number=line_number
                    )
                    links.append(link)

                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error processing wikilink match {match_count} in {source_path}: {e}")
                    continue

            if error_count > 0:
                logger.warning(f"Failed to process {error_count}/{match_count} wikilink matches in {source_path}")

        except Exception as e:
            logger.error(f"Critical error extracting wikilinks from {source_path}: {e}")
            # Return partial results if any were extracted
            
        return links

    def _extract_markdown_links(self, content: str, source_path: str) -> list[Link]:
        """Extract markdown links from content with error handling.
        
        Args:
            content: Content to extract from
            source_path: Source file path
            
        Returns:
            List of markdown Link objects
        """
        links = []
        
        try:
            pattern = self.link_patterns['markdown_link']
            match_count = 0
            error_count = 0
            skipped_external = 0

            for match in pattern.finditer(content):
                try:
                    match_count += 1
                    display_text = match.group(1).strip() if match.group(1) else ""
                    target = match.group(2).strip() if match.group(2) else ""

                    if not target:
                        logger.debug(f"Empty markdown link target in {source_path} at match {match_count}")
                        continue

                    # Skip external URLs in markdown links for internal link analysis
                    if target.startswith(('http://', 'https://', 'ftp://', 'mailto:')):
                        skipped_external += 1
                        continue

                    # Calculate line number safely
                    try:
                        line_number = content[:match.start()].count('\n') + 1
                    except Exception as e:
                        logger.debug(f"Failed to calculate line number for markdown link in {source_path}: {e}")
                        line_number = None

                    link = Link(
                        source=source_path,
                        target=target,
                        link_type="markdown_link",
                        display_text=display_text,
                        line_number=line_number
                    )
                    links.append(link)

                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error processing markdown link match {match_count} in {source_path}: {e}")
                    continue

            if error_count > 0:
                logger.warning(f"Failed to process {error_count}/{match_count} markdown link matches in {source_path}")
            
            if skipped_external > 0:
                logger.debug(f"Skipped {skipped_external} external URLs in {source_path}")

        except Exception as e:
            logger.error(f"Critical error extracting markdown links from {source_path}: {e}")
            # Return partial results if any were extracted
            
        return links

    def _extract_embedded_links(self, content: str, source_path: str) -> list[Link]:
        """Extract embedded links (images, etc.) from content with error handling.
        
        Args:
            content: Content to extract from
            source_path: Source file path
            
        Returns:
            List of embedded Link objects
        """
        links = []
        
        try:
            pattern = self.link_patterns['embedded_link']
            match_count = 0
            error_count = 0

            for match in pattern.finditer(content):
                try:
                    match_count += 1
                    target = match.group(1).strip() if match.group(1) else ""
                    heading = match.group(2).strip() if match.group(2) else None
                    display_text = match.group(3).strip() if match.group(3) else None

                    if not target:
                        logger.debug(f"Empty embedded link target in {source_path} at match {match_count}")
                        continue

                    # Handle heading links
                    full_target = target
                    if heading:
                        full_target = f"{target}#{heading}"

                    # Calculate line number safely
                    try:
                        line_number = content[:match.start()].count('\n') + 1
                    except Exception as e:
                        logger.debug(f"Failed to calculate line number for embedded link in {source_path}: {e}")
                        line_number = None

                    link = Link(
                        source=source_path,
                        target=full_target,
                        link_type="embedded_link",
                        display_text=display_text,
                        line_number=line_number
                    )
                    links.append(link)

                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error processing embedded link match {match_count} in {source_path}: {e}")
                    continue

            if error_count > 0:
                logger.warning(f"Failed to process {error_count}/{match_count} embedded link matches in {source_path}")

        except Exception as e:
            logger.error(f"Critical error extracting embedded links from {source_path}: {e}")
            # Return partial results if any were extracted
            
        return links

    def _extract_tag_links(self, content: str, source_path: str) -> list[Link]:
        """Extract tag links from content.
        
        Args:
            content: Content to extract from
            source_path: Source file path
            
        Returns:
            List of tag Link objects
        """
        links = []
        pattern = self.link_patterns['tag_link']

        for match in pattern.finditer(content):
            tag = match.group(1).strip()

            if not tag:
                continue

            link = Link(
                source=source_path,
                target=f"#{tag}",
                link_type="tag_link",
                line_number=content[:match.start()].count('\n') + 1
            )
            links.append(link)

        return links

    def _normalize_link_path(self, link: str, source_path: str) -> str | None:
        """Normalize and resolve link paths to actual files.
        
        Args:
            link: Raw link text
            source_path: Path of the source file
            
        Returns:
            Normalized path or None if invalid
        """
        if not link or not link.strip():
            return None

        # Remove heading/anchor parts
        clean_link = link.split('#')[0].strip()
        if not clean_link:
            return None

        # Handle different link formats
        try:
            # Case 1: Absolute path (starts with /)
            if clean_link.startswith('/'):
                potential_path = clean_link[1:]  # Remove leading slash
            # Case 2: Relative path
            elif '/' in clean_link:
                source_dir = str(Path(source_path).parent)
                if source_dir == '.':
                    potential_path = clean_link
                else:
                    potential_path = os.path.normpath(os.path.join(source_dir, clean_link))
            # Case 3: Simple filename - search in vault
            else:
                potential_path = self._find_file_in_vault(clean_link)
                if potential_path is None:
                    # Try adding .md extension
                    potential_path = self._find_file_in_vault(f"{clean_link}.md")

            # Normalize path separators
            if potential_path:
                potential_path = potential_path.replace('\\', '/')
                # Remove leading slash if present
                potential_path = potential_path.lstrip('/')

            return potential_path

        except Exception as e:
            logger.debug(f"Error normalizing link path '{link}': {e}")
            return None

    def _find_file_in_vault(self, filename: str) -> str | None:
        """Find a file in the vault by name.
        
        Args:
            filename: Name of the file to find
            
        Returns:
            Relative path to the file or None if not found
        """
        if not filename:
            return None

        # Search in file cache
        filename_lower = filename.lower()
        for file_path in self._file_cache.get('all_files', set()):
            if Path(file_path).name.lower() == filename_lower:
                return file_path

        return None

    def _validate_link_target(self, target_path: str) -> bool:
        """Validate that link target exists in vault.
        
        Args:
            target_path: Target path to validate
            
        Returns:
            True if target exists, False otherwise
        """
        if not target_path:
            return False

        # Check cache first
        if target_path in self._broken_links_cache:
            return False

        # Check if file exists
        try:
            full_path = self.vault_reader.get_absolute_path(target_path)
            exists = full_path.exists()

            if not exists:
                self._broken_links_cache.add(target_path)

            return exists

        except Exception:
            self._broken_links_cache.add(target_path)
            return False

    def _build_file_cache(self, markdown_files: list[Path]) -> None:
        """Build cache of all files for faster lookups.
        
        Args:
            markdown_files: List of markdown files in vault
        """
        all_files = set()
        for file_path in markdown_files:
            all_files.add(str(file_path))

        self._file_cache['all_files'] = all_files
        logger.debug(f"Built file cache with {len(all_files)} files")

    def _build_link_graph(self, links: list[Link], all_files: list[Path]) -> nx.DiGraph:
        """Build directed graph from extracted links.
        
        Args:
            links: List of all extracted links
            all_files: List of all files in vault
            
        Returns:
            Directed graph representing note relationships
        """
        graph = nx.DiGraph()

        # Add all files as nodes
        for file_path in all_files:
            graph.add_node(str(file_path))

        # Add edges for valid links
        valid_links = 0
        broken_links = 0

        for link in links:
            if link.is_valid and link.source != link.target:
                # Avoid self-loops unless specifically needed
                graph.add_edge(link.source, link.target,
                             link_type=link.link_type,
                             display_text=link.display_text)
                valid_links += 1
            elif not link.is_valid:
                broken_links += 1

        logger.info(f"Graph built: {valid_links} valid links, {broken_links} broken links")

        return graph

    def _compute_link_statistics(self, links: list[Link], link_counts: dict[str, int],
                               total_files: int) -> LinkStatistics:
        """Compute comprehensive link statistics.
        
        Args:
            links: List of all extracted links
            link_counts: Dictionary of link counts per file
            total_files: Total number of files processed
            
        Returns:
            Link statistics
        """
        stats = LinkStatistics()

        if not links:
            return stats

        # Basic counts
        stats.total_links = len(links)
        unique_targets = set()
        broken_count = 0
        self_links = 0
        link_type_counts = {}

        for link in links:
            unique_targets.add(link.target)

            if not link.is_valid:
                broken_count += 1

            if link.source == link.target:
                self_links += 1

            link_type_counts[link.link_type] = link_type_counts.get(link.link_type, 0) + 1

        stats.unique_links = len(unique_targets)
        stats.broken_links = broken_count
        stats.self_links = self_links
        stats.link_types = link_type_counts

        # Compute derived statistics
        if link_counts:
            outgoing_counts = list(link_counts.values())
            stats.max_outgoing_links = max(outgoing_counts) if outgoing_counts else 0
            stats.avg_outgoing_links = sum(outgoing_counts) / len(outgoing_counts) if outgoing_counts else 0
            stats.notes_with_no_links = sum(1 for count in outgoing_counts if count == 0)

        # Bidirectional links computation would require graph analysis
        # This is a simplified version - actual implementation would analyze the graph
        stats.bidirectional_links = 0  # Placeholder - would need graph analysis

        return stats

    def validate_links(self, links: list[Link]) -> ValidationResult:
        """Validate a list of extracted links.
        
        Args:
            links: List of links to validate
            
        Returns:
            Validation result with detailed information
        """
        result = ValidationResult(valid=True)

        valid_links = 0
        broken_links = 0
        circular_refs = []

        for link in links:
            if link.is_valid:
                valid_links += 1
            else:
                broken_links += 1
                result.warnings.append(f"Broken link: {link.source} -> {link.target}")

            # Check for self-references (not necessarily errors)
            if link.source == link.target:
                result.warnings.append(f"Self-reference: {link.source}")

        result.links_extracted = len(links)
        result.links_broken = broken_links

        # Set validation status
        if broken_links > len(links) * 0.5:  # More than 50% broken
            result.valid = False
            result.errors.append(f"Too many broken links: {broken_links}/{len(links)}")

        if broken_links > 0:
            result.warnings.append(f"Found {broken_links} broken links out of {len(links)} total")

        return result
