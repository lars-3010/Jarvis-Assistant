"""
Unit tests for LinkExtractor class.

Tests various link format extraction including wikilinks and markdown links,
path normalization and link validation logic, and broken link handling.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import networkx as nx

from jarvis.services.vault.reader import VaultReader
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
from jarvis.tools.dataset_generation.models.data_models import Link, LinkStatistics
from jarvis.tools.dataset_generation.models.exceptions import LinkExtractionError


class TestLinkExtractor:
    """Test suite for LinkExtractor class."""

    @pytest.fixture
    def mock_vault_reader(self):
        """Create a mock VaultReader for testing."""
        mock_reader = Mock(spec=VaultReader)
        mock_reader.get_markdown_files.return_value = [
            Path("note1.md"),
            Path("note2.md"),
            Path("folder/note3.md")
        ]
        return mock_reader

    @pytest.fixture
    def link_extractor(self, mock_vault_reader):
        """Create a LinkExtractor instance for testing."""
        return LinkExtractor(mock_vault_reader)

    def test_compile_link_patterns(self, link_extractor):
        """Test that link patterns are compiled correctly."""
        patterns = link_extractor.link_patterns
        
        # Check that all expected patterns are present
        expected_patterns = [
            'wikilink', 'markdown_link', 'embedded_link', 
            'tag_link', 'reference_link', 'url_link'
        ]
        
        for pattern_name in expected_patterns:
            assert pattern_name in patterns
            assert hasattr(patterns[pattern_name], 'finditer')

    def test_extract_wikilinks_basic(self, link_extractor):
        """Test basic wikilink extraction."""
        content = "This is a [[Simple Link]] and [[Another Link|Display Text]]."
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        assert len(links) == 2
        
        # First link
        assert links[0].source == source_path
        assert links[0].target == "Simple Link"
        assert links[0].link_type == "wikilink"
        assert links[0].display_text is None
        
        # Second link with display text
        assert links[1].source == source_path
        assert links[1].target == "Another Link"
        assert links[1].link_type == "wikilink"
        assert links[1].display_text == "Display Text"

    def test_extract_wikilinks_with_headings(self, link_extractor):
        """Test wikilink extraction with heading references."""
        content = "Link to [[Note#Section]] and [[Another Note#Sub Section|Display]]."
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        assert len(links) == 2
        assert links[0].target == "Note#Section"
        assert links[1].target == "Another Note#Sub Section"
        assert links[1].display_text == "Display"

    def test_extract_wikilinks_multiline(self, link_extractor):
        """Test wikilink extraction across multiple lines."""
        content = """This is a [[Multi
Line Link]] that spans lines."""
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        assert len(links) == 1
        assert "Multi\nLine Link" in links[0].target

    def test_extract_wikilinks_empty_content(self, link_extractor):
        """Test wikilink extraction with empty content."""
        content = ""
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        assert len(links) == 0

    def test_extract_wikilinks_no_links(self, link_extractor):
        """Test wikilink extraction with no links present."""
        content = "This is just regular text with no links."
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        assert len(links) == 0

    def test_extract_markdown_links_basic(self, link_extractor):
        """Test basic markdown link extraction."""
        content = "Check out [Google](https://google.com) and [Local Note](note.md)."
        source_path = "test.md"
        
        links = link_extractor._extract_markdown_links(content, source_path)
        
        # Should only extract local note, not external URL
        assert len(links) == 1
        assert links[0].target == "note.md"
        assert links[0].display_text == "Local Note"
        assert links[0].link_type == "markdown_link"

    def test_extract_markdown_links_relative_paths(self, link_extractor):
        """Test markdown link extraction with relative paths."""
        content = "Links: [Relative](../folder/note.md) and [Same Dir](./note.md)."
        source_path = "test.md"
        
        links = link_extractor._extract_markdown_links(content, source_path)
        
        assert len(links) == 2
        assert links[0].target == "../folder/note.md"
        assert links[1].target == "./note.md"

    def test_extract_markdown_links_skip_external(self, link_extractor):
        """Test that external URLs are skipped in markdown links."""
        content = """
        [HTTP](http://example.com)
        [HTTPS](https://example.com)
        [FTP](ftp://example.com)
        [Email](mailto:test@example.com)
        [Local](local-note.md)
        """
        source_path = "test.md"
        
        links = link_extractor._extract_markdown_links(content, source_path)
        
        # Should only extract the local link
        assert len(links) == 1
        assert links[0].target == "local-note.md"

    def test_extract_embedded_links_basic(self, link_extractor):
        """Test basic embedded link extraction."""
        content = "Image: ![[image.png]] and ![[document.pdf|Document]]."
        source_path = "test.md"
        
        links = link_extractor._extract_embedded_links(content, source_path)
        
        assert len(links) == 2
        assert links[0].target == "image.png"
        assert links[0].link_type == "embedded_link"
        assert links[1].target == "document.pdf"
        assert links[1].display_text == "Document"

    def test_extract_embedded_links_with_headings(self, link_extractor):
        """Test embedded link extraction with heading references."""
        content = "Embed: ![[Note#Section]] and ![[Doc#Part|Display]]."
        source_path = "test.md"
        
        links = link_extractor._extract_embedded_links(content, source_path)
        
        assert len(links) == 2
        assert links[0].target == "Note#Section"
        assert links[1].target == "Doc#Part"
        assert links[1].display_text == "Display"

    def test_normalize_link_path_absolute(self, link_extractor):
        """Test link path normalization for absolute paths."""
        # Mock file cache
        link_extractor._file_cache = {
            'all_files': {'folder/note.md', 'other.md'}
        }
        
        result = link_extractor._normalize_link_path("/folder/note", "source.md")
        assert result == "folder/note"

    def test_normalize_link_path_relative(self, link_extractor):
        """Test link path normalization for relative paths."""
        # Mock file cache
        link_extractor._file_cache = {
            'all_files': {'folder/subfolder/note.md', 'folder/other.md'}
        }
        
        result = link_extractor._normalize_link_path("subfolder/note", "folder/source.md")
        assert result == "folder/subfolder/note"

    def test_normalize_link_path_simple_filename(self, link_extractor):
        """Test link path normalization for simple filenames."""
        # Mock file cache and _find_file_in_vault method
        link_extractor._file_cache = {
            'all_files': {'folder/target.md', 'other.md'}
        }
        
        with patch.object(link_extractor, '_find_file_in_vault') as mock_find:
            mock_find.return_value = "folder/target.md"
            
            result = link_extractor._normalize_link_path("target", "source.md")
            assert result == "folder/target.md"
            mock_find.assert_called_with("target")

    def test_normalize_link_path_with_extension(self, link_extractor):
        """Test link path normalization adds .md extension when needed."""
        link_extractor._file_cache = {
            'all_files': {'folder/target.md', 'other.md'}
        }
        
        with patch.object(link_extractor, '_find_file_in_vault') as mock_find:
            # First call returns None, second call (with .md) returns the file
            mock_find.side_effect = [None, "folder/target.md"]
            
            result = link_extractor._normalize_link_path("target", "source.md")
            assert result == "folder/target.md"
            assert mock_find.call_count == 2

    def test_normalize_link_path_with_heading(self, link_extractor):
        """Test link path normalization strips heading references."""
        link_extractor._file_cache = {
            'all_files': {'target.md'}
        }
        
        with patch.object(link_extractor, '_find_file_in_vault') as mock_find:
            mock_find.return_value = "target.md"
            
            result = link_extractor._normalize_link_path("target#heading", "source.md")
            assert result == "target.md"
            mock_find.assert_called_with("target")

    def test_normalize_link_path_empty_link(self, link_extractor):
        """Test link path normalization with empty or whitespace links."""
        assert link_extractor._normalize_link_path("", "source.md") is None
        assert link_extractor._normalize_link_path("   ", "source.md") is None
        assert link_extractor._normalize_link_path("#heading", "source.md") is None

    def test_find_file_in_vault_exact_match(self, link_extractor):
        """Test finding file in vault with exact name match."""
        link_extractor._file_cache = {
            'all_files': {'folder/target.md', 'other.md', 'TARGET.md'}
        }
        
        result = link_extractor._find_file_in_vault("target.md")
        assert result == "folder/target.md"

    def test_find_file_in_vault_case_insensitive(self, link_extractor):
        """Test finding file in vault with case-insensitive matching."""
        link_extractor._file_cache = {
            'all_files': {'folder/Target.md', 'other.md'}
        }
        
        result = link_extractor._find_file_in_vault("target.md")
        assert result == "folder/Target.md"

    def test_find_file_in_vault_not_found(self, link_extractor):
        """Test finding file in vault when file doesn't exist."""
        link_extractor._file_cache = {
            'all_files': {'folder/other.md', 'another.md'}
        }
        
        result = link_extractor._find_file_in_vault("nonexistent.md")
        assert result is None

    def test_find_file_in_vault_empty_filename(self, link_extractor):
        """Test finding file in vault with empty filename."""
        result = link_extractor._find_file_in_vault("")
        assert result is None

    def test_validate_link_target_valid(self, link_extractor):
        """Test link target validation for valid targets."""
        mock_path = Mock()
        mock_path.exists.return_value = True
        
        with patch.object(link_extractor.vault_reader, 'get_absolute_path') as mock_get_path:
            mock_get_path.return_value = mock_path
            
            result = link_extractor._validate_link_target("valid/path.md")
            assert result is True

    def test_validate_link_target_invalid(self, link_extractor):
        """Test link target validation for invalid targets."""
        mock_path = Mock()
        mock_path.exists.return_value = False
        
        with patch.object(link_extractor.vault_reader, 'get_absolute_path') as mock_get_path:
            mock_get_path.return_value = mock_path
            
            result = link_extractor._validate_link_target("invalid/path.md")
            assert result is False
            
            # Should be cached in broken links
            assert "invalid/path.md" in link_extractor._broken_links_cache

    def test_validate_link_target_cached_broken(self, link_extractor):
        """Test link target validation uses broken links cache."""
        link_extractor._broken_links_cache.add("cached/broken.md")
        
        result = link_extractor._validate_link_target("cached/broken.md")
        assert result is False

    def test_validate_link_target_exception(self, link_extractor):
        """Test link target validation handles exceptions."""
        with patch.object(link_extractor.vault_reader, 'get_absolute_path') as mock_get_path:
            mock_get_path.side_effect = Exception("File system error")
            
            result = link_extractor._validate_link_target("error/path.md")
            assert result is False
            assert "error/path.md" in link_extractor._broken_links_cache

    def test_validate_link_target_empty(self, link_extractor):
        """Test link target validation with empty target."""
        result = link_extractor._validate_link_target("")
        assert result is False

    def test_build_file_cache(self, link_extractor):
        """Test building file cache from markdown files."""
        markdown_files = [Path("note1.md"), Path("folder/note2.md"), Path("note3.md")]
        
        link_extractor._build_file_cache(markdown_files)
        
        expected_files = {"note1.md", "folder/note2.md", "note3.md"}
        assert link_extractor._file_cache['all_files'] == expected_files

    def test_build_link_graph_basic(self, link_extractor):
        """Test building link graph from extracted links."""
        all_files = [Path("note1.md"), Path("note2.md"), Path("note3.md")]
        links = [
            Link("note1.md", "note2.md", "wikilink", is_valid=True),
            Link("note2.md", "note3.md", "wikilink", is_valid=True),
            Link("note1.md", "nonexistent.md", "wikilink", is_valid=False)
        ]
        
        graph = link_extractor._build_link_graph(links, all_files)
        
        # Check nodes
        assert graph.number_of_nodes() == 3
        assert "note1.md" in graph.nodes()
        assert "note2.md" in graph.nodes()
        assert "note3.md" in graph.nodes()
        
        # Check edges (only valid links)
        assert graph.number_of_edges() == 2
        assert graph.has_edge("note1.md", "note2.md")
        assert graph.has_edge("note2.md", "note3.md")
        assert not graph.has_edge("note1.md", "nonexistent.md")

    def test_build_link_graph_self_links(self, link_extractor):
        """Test building link graph excludes self-links."""
        all_files = [Path("note1.md"), Path("note2.md")]
        links = [
            Link("note1.md", "note1.md", "wikilink", is_valid=True),  # Self-link
            Link("note1.md", "note2.md", "wikilink", is_valid=True)
        ]
        
        graph = link_extractor._build_link_graph(links, all_files)
        
        # Should exclude self-link
        assert graph.number_of_edges() == 1
        assert not graph.has_edge("note1.md", "note1.md")
        assert graph.has_edge("note1.md", "note2.md")

    def test_build_link_graph_edge_attributes(self, link_extractor):
        """Test building link graph preserves edge attributes."""
        all_files = [Path("note1.md"), Path("note2.md")]
        links = [
            Link("note1.md", "note2.md", "wikilink", display_text="Display", is_valid=True)
        ]
        
        graph = link_extractor._build_link_graph(links, all_files)
        
        edge_data = graph.get_edge_data("note1.md", "note2.md")
        assert edge_data['link_type'] == "wikilink"
        assert edge_data['display_text'] == "Display"

    def test_compute_link_statistics_basic(self, link_extractor):
        """Test computing basic link statistics."""
        links = [
            Link("note1.md", "note2.md", "wikilink", is_valid=True),
            Link("note1.md", "note3.md", "markdown_link", is_valid=True),
            Link("note2.md", "nonexistent.md", "wikilink", is_valid=False),
            Link("note3.md", "note3.md", "wikilink", is_valid=True)  # Self-link
        ]
        link_counts = {"note1.md": 2, "note2.md": 1, "note3.md": 1}
        total_files = 3
        
        stats = link_extractor._compute_link_statistics(links, link_counts, total_files)
        
        assert stats.total_links == 4
        assert stats.unique_links == 3  # note2.md, note3.md, nonexistent.md
        assert stats.broken_links == 1
        assert stats.self_links == 1
        assert stats.link_types == {"wikilink": 3, "markdown_link": 1}
        assert stats.max_outgoing_links == 2
        assert stats.avg_outgoing_links == 4/3  # 4 links / 3 files

    def test_compute_link_statistics_empty(self, link_extractor):
        """Test computing link statistics with no links."""
        links = []
        link_counts = {}
        total_files = 3
        
        stats = link_extractor._compute_link_statistics(links, link_counts, total_files)
        
        assert stats.total_links == 0
        assert stats.unique_links == 0
        assert stats.broken_links == 0
        assert stats.self_links == 0
        assert stats.link_types == {}

    def test_extract_links_from_content_comprehensive(self, link_extractor):
        """Test comprehensive link extraction from content."""
        content = """
        # Test Note
        
        This note contains [[Wikilink]] and [Markdown](local.md).
        Also has ![[embedded.png]] and [[Link|Display Text]].
        """
        source_path = "test.md"
        
        # Mock the individual extraction methods
        with patch.object(link_extractor, '_extract_wikilinks') as mock_wiki, \
             patch.object(link_extractor, '_extract_markdown_links') as mock_md, \
             patch.object(link_extractor, '_extract_embedded_links') as mock_embed, \
             patch.object(link_extractor, '_normalize_link_path') as mock_normalize, \
             patch.object(link_extractor, '_validate_link_target') as mock_validate:
            
            # Setup mock returns
            mock_wiki.return_value = [Link("test.md", "Wikilink", "wikilink")]
            mock_md.return_value = [Link("test.md", "local.md", "markdown_link")]
            mock_embed.return_value = [Link("test.md", "embedded.png", "embedded_link")]
            mock_normalize.return_value = "normalized_path.md"
            mock_validate.return_value = True
            
            links = link_extractor.extract_links_from_content(content, source_path)
            
            # Should call all extraction methods
            mock_wiki.assert_called_once_with(content, source_path)
            mock_md.assert_called_once_with(content, source_path)
            mock_embed.assert_called_once_with(content, source_path)
            
            # Should have 3 links total
            assert len(links) == 3

    def test_extract_links_from_content_empty(self, link_extractor):
        """Test link extraction from empty content."""
        result = link_extractor.extract_links_from_content("", "test.md")
        assert result == []
        
        result = link_extractor.extract_links_from_content("   ", "test.md")
        assert result == []

    def test_extract_links_from_content_error_handling(self, link_extractor):
        """Test link extraction handles errors gracefully."""
        content = "Some content with [[links]]"
        source_path = "test.md"
        
        # Mock extraction methods to raise exceptions
        with patch.object(link_extractor, '_extract_wikilinks') as mock_wiki, \
             patch.object(link_extractor, '_extract_markdown_links') as mock_md, \
             patch.object(link_extractor, '_extract_embedded_links') as mock_embed:
            
            mock_wiki.side_effect = Exception("Wikilink extraction failed")
            mock_md.return_value = []
            mock_embed.return_value = []
            
            # Should not raise exception, should return empty list
            links = link_extractor.extract_links_from_content(content, source_path)
            assert links == []

    def test_extract_all_links_success(self, link_extractor):
        """Test successful extraction of all links."""
        # Mock vault reader methods
        markdown_files = [Path("note1.md"), Path("note2.md")]
        link_extractor.vault_reader.get_markdown_files.return_value = markdown_files
        link_extractor.vault_reader.read_file.side_effect = [
            ("Content with [[link1]]", {"created": 1234567890}),
            ("Content with [[link2]]", {"created": 1234567891})
        ]
        
        # Mock link extraction
        with patch.object(link_extractor, 'extract_links_from_content') as mock_extract, \
             patch.object(link_extractor, '_build_file_cache') as mock_cache, \
             patch.object(link_extractor, '_build_link_graph') as mock_graph, \
             patch.object(link_extractor, '_compute_link_statistics') as mock_stats:
            
            mock_extract.side_effect = [
                [Link("note1.md", "link1", "wikilink", is_valid=True)],
                [Link("note2.md", "link2", "wikilink", is_valid=True)]
            ]
            mock_graph.return_value = nx.DiGraph()
            mock_stats.return_value = LinkStatistics()
            
            graph, stats = link_extractor.extract_all_links()
            
            # Verify methods were called
            mock_cache.assert_called_once()
            mock_graph.assert_called_once()
            mock_stats.assert_called_once()
            
            assert isinstance(graph, nx.DiGraph)
            assert isinstance(stats, LinkStatistics)

    def test_extract_all_links_no_files(self, link_extractor):
        """Test extraction when no markdown files are found."""
        link_extractor.vault_reader.get_markdown_files.return_value = []
        
        graph, stats = link_extractor.extract_all_links()
        
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 0
        assert isinstance(stats, LinkStatistics)

    def test_extract_all_links_vault_access_error(self, link_extractor):
        """Test extraction handles vault access errors."""
        link_extractor.vault_reader.get_markdown_files.side_effect = Exception("Vault access failed")
        
        with pytest.raises(LinkExtractionError) as exc_info:
            link_extractor.extract_all_links()
        
        assert "Cannot access vault files" in str(exc_info.value)

    def test_extract_all_links_file_read_errors(self, link_extractor):
        """Test extraction handles individual file read errors."""
        markdown_files = [Path("note1.md"), Path("note2.md"), Path("note3.md")]
        link_extractor.vault_reader.get_markdown_files.return_value = markdown_files
        
        # Mock file reading with various errors
        link_extractor.vault_reader.read_file.side_effect = [
            ("Good content", {}),  # Success
            PermissionError("Permission denied"),  # Permission error
            UnicodeDecodeError("utf-8", b"", 0, 1, "Invalid encoding")  # Encoding error
        ]
        
        with patch.object(link_extractor, 'extract_links_from_content') as mock_extract, \
             patch.object(link_extractor, '_build_file_cache') as mock_cache, \
             patch.object(link_extractor, '_build_link_graph') as mock_graph, \
             patch.object(link_extractor, '_compute_link_statistics') as mock_stats:
            
            mock_extract.return_value = []
            mock_graph.return_value = nx.DiGraph()
            mock_stats.return_value = LinkStatistics()
            
            graph, stats = link_extractor.extract_all_links()
            
            # Should process successfully despite errors
            assert isinstance(graph, nx.DiGraph)
            assert isinstance(stats, LinkStatistics)

    def test_extract_all_links_all_files_fail(self, link_extractor):
        """Test extraction when all files fail to process."""
        markdown_files = [Path("note1.md"), Path("note2.md")]
        link_extractor.vault_reader.get_markdown_files.return_value = markdown_files
        link_extractor.vault_reader.read_file.side_effect = [
            PermissionError("Permission denied"),
            FileNotFoundError("File not found")
        ]
        
        with patch.object(link_extractor, '_build_file_cache'):
            with pytest.raises(LinkExtractionError) as exc_info:
                link_extractor.extract_all_links()
            
            assert "No files could be processed successfully" in str(exc_info.value)

    def test_extract_all_links_progress_logging(self, link_extractor):
        """Test that progress is logged during extraction."""
        # Create many files to trigger progress logging
        markdown_files = [Path(f"note{i}.md") for i in range(150)]
        link_extractor.vault_reader.get_markdown_files.return_value = markdown_files
        link_extractor.vault_reader.read_file.return_value = ("Content", {})
        
        with patch.object(link_extractor, 'extract_links_from_content') as mock_extract, \
             patch.object(link_extractor, '_build_file_cache') as mock_cache, \
             patch.object(link_extractor, '_build_link_graph') as mock_graph, \
             patch.object(link_extractor, '_compute_link_statistics') as mock_stats, \
             patch('jarvis.tools.dataset_generation.extractors.link_extractor.logger') as mock_logger:
            
            mock_extract.return_value = []
            mock_graph.return_value = nx.DiGraph()
            mock_stats.return_value = LinkStatistics()
            
            link_extractor.extract_all_links()
            
            # Should log progress at 100-file intervals
            progress_calls = [call for call in mock_logger.debug.call_args_list 
                            if "Processed" in str(call)]
            assert len(progress_calls) >= 1

    def test_line_number_calculation(self, link_extractor):
        """Test that line numbers are calculated correctly for links."""
        content = """Line 1
Line 2 with [[Link1]]
Line 3
Line 4 with [[Link2]]"""
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        assert len(links) == 2
        assert links[0].line_number == 2  # Link1 on line 2
        assert links[1].line_number == 4  # Link2 on line 4

    def test_deduplication(self, link_extractor):
        """Test that duplicate links are removed."""
        content = """
        [[Same Link]] and [[Same Link]] again.
        Also [Same](same.md) and [Same](same.md).
        """
        source_path = "test.md"
        
        with patch.object(link_extractor, '_normalize_link_path') as mock_normalize, \
             patch.object(link_extractor, '_validate_link_target') as mock_validate:
            
            mock_normalize.return_value = "normalized.md"
            mock_validate.return_value = True
            
            links = link_extractor.extract_links_from_content(content, source_path)
            
            # Should deduplicate based on (source, target, link_type)
            wikilinks = [l for l in links if l.link_type == "wikilink"]
            markdown_links = [l for l in links if l.link_type == "markdown_link"]
            
            assert len(wikilinks) == 1  # Deduplicated wikilinks
            assert len(markdown_links) == 1  # Deduplicated markdown links


class TestLinkExtractionErrorHandling:
    """Test suite for error handling in link extraction."""

    @pytest.fixture
    def mock_vault_reader(self):
        """Create a mock VaultReader for testing."""
        return Mock(spec=VaultReader)

    @pytest.fixture
    def link_extractor(self, mock_vault_reader):
        """Create a LinkExtractor instance for testing."""
        return LinkExtractor(mock_vault_reader)

    def test_broken_link_detection(self, link_extractor):
        """Test detection and handling of broken links."""
        content = "Link to [[Nonexistent Note]] and [[Valid Note]]."
        source_path = "test.md"
        
        with patch.object(link_extractor, '_normalize_link_path') as mock_normalize, \
             patch.object(link_extractor, '_validate_link_target') as mock_validate:
            
            mock_normalize.side_effect = ["nonexistent.md", "valid.md"]
            mock_validate.side_effect = [False, True]  # First invalid, second valid
            
            links = link_extractor.extract_links_from_content(content, source_path)
            
            assert len(links) == 2
            assert not links[0].is_valid  # Broken link
            assert links[1].is_valid     # Valid link

    def test_circular_reference_handling(self, link_extractor):
        """Test handling of circular references."""
        # This is more of an integration test, but we can test the basic case
        all_files = [Path("note1.md"), Path("note2.md")]
        links = [
            Link("note1.md", "note2.md", "wikilink", is_valid=True),
            Link("note2.md", "note1.md", "wikilink", is_valid=True)  # Circular
        ]
        
        graph = link_extractor._build_link_graph(links, all_files)
        
        # NetworkX should handle circular references fine
        assert graph.has_edge("note1.md", "note2.md")
        assert graph.has_edge("note2.md", "note1.md")
        
        # Check for cycles
        cycles = list(nx.simple_cycles(graph))
        assert len(cycles) >= 1  # Should detect the cycle

    def test_malformed_link_handling(self, link_extractor):
        """Test handling of malformed links."""
        content = """
        [[]] - Empty wikilink
        [[   ]] - Whitespace wikilink
        []() - Empty markdown link
        [Text]() - Markdown link with empty URL
        """
        source_path = "test.md"
        
        links = link_extractor.extract_links_from_content(content, source_path)
        
        # Should handle malformed links gracefully (likely return empty list or skip them)
        # The exact behavior depends on implementation, but should not crash
        assert isinstance(links, list)

    def test_special_characters_in_links(self, link_extractor):
        """Test handling of special characters in links."""
        content = """
        [[Note with spaces]]
        [[Note-with-dashes]]
        [[Note_with_underscores]]
        [[Note (with parentheses)]]
        [[Note [with brackets]]]
        [[Note "with quotes"]]
        """
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        # Should extract all links with special characters
        assert len(links) >= 5  # At least the valid ones
        
        targets = [link.target for link in links]
        assert "Note with spaces" in targets
        assert "Note-with-dashes" in targets
        assert "Note_with_underscores" in targets

    def test_unicode_handling(self, link_extractor):
        """Test handling of Unicode characters in links."""
        content = """
        [[Café Notes]]
        [[数学笔记]]
        [[Émile's Notes]]
        [[🚀 Rocket Notes]]
        """
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        # Should handle Unicode characters properly
        assert len(links) == 4
        
        targets = [link.target for link in links]
        assert "Café Notes" in targets
        assert "数学笔记" in targets
        assert "Émile's Notes" in targets
        assert "🚀 Rocket Notes" in targets

    def test_very_long_links(self, link_extractor):
        """Test handling of very long links."""
        long_target = "A" * 1000  # Very long link target
        content = f"[[{long_target}]]"
        source_path = "test.md"
        
        links = link_extractor._extract_wikilinks(content, source_path)
        
        # Should handle long links without issues
        assert len(links) == 1
        assert links[0].target == long_target

    def test_nested_brackets_handling(self, link_extractor):
        """Test handling of nested brackets in links."""
        content = """
        [[Note [with nested] brackets]]
        [Text [with nested] brackets](note.md)
        """
        source_path = "test.md"
        
        # This tests the regex patterns' ability to handle nested structures
        wikilinks = link_extractor._extract_wikilinks(content, source_path)
        markdown_links = link_extractor._extract_markdown_links(content, source_path)
        
        # Behavior depends on regex implementation
        # At minimum, should not crash
        assert isinstance(wikilinks, list)
        assert isinstance(markdown_links, list)