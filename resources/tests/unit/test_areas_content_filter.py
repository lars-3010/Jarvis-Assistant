"""
Unit tests for AreasContentFilter class.

Tests Areas/ content identification and filtering logic, validation of Areas/
folder structure and content, and error handling for missing or empty Areas/ folders.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from jarvis.tools.dataset_generation.filters.areas_filter import AreasContentFilter
from jarvis.tools.dataset_generation.models.exceptions import (
    VaultValidationError,
    AreasNotFoundError,
    InsufficientAreasContentError
)
from jarvis.utils.config import get_settings


class TestAreasContentFilter:
    """Test suite for AreasContentFilter class."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault structure for testing."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir)
        
        # Create Areas folder structure
        areas_path = vault_path / "Areas"
        areas_path.mkdir()
        
        # Create subdirectories in Areas
        (areas_path / "Computer Science").mkdir()
        (areas_path / "Natural Science").mkdir()
        (areas_path / "Business").mkdir()
        
        # Create markdown files in Areas
        (areas_path / "Computer Science" / "algorithms.md").write_text("# Algorithms\nContent about algorithms")
        (areas_path / "Computer Science" / "data-structures.md").write_text("# Data Structures\nContent about data structures")
        (areas_path / "Natural Science" / "physics.md").write_text("# Physics\nContent about physics")
        (areas_path / "Natural Science" / "chemistry.md").write_text("# Chemistry\nContent about chemistry")
        (areas_path / "Business" / "strategy.md").write_text("# Strategy\nContent about business strategy")
        
        # Create non-Areas folders and files (should be excluded)
        journal_path = vault_path / "Journal"
        journal_path.mkdir()
        (journal_path / "2024-01-01.md").write_text("# Daily Journal\nPersonal content")
        (journal_path / "2024-01-02.md").write_text("# Daily Journal\nMore personal content")
        
        inbox_path = vault_path / "Inbox"
        inbox_path.mkdir()
        (inbox_path / "quick-note.md").write_text("# Quick Note\nTemporary content")
        
        people_path = vault_path / "People"
        people_path.mkdir()
        (people_path / "john-doe.md").write_text("# John Doe\nPersonal contact info")
        
        # Create some hidden files (should be ignored)
        (areas_path / ".hidden-file.md").write_text("Hidden content")
        (areas_path / "Computer Science" / ".DS_Store").write_text("System file")
        
        yield vault_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def empty_vault(self):
        """Create a temporary vault with no Areas folder."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir)
        
        # Create other folders but no Areas
        journal_path = vault_path / "Journal"
        journal_path.mkdir()
        (journal_path / "2024-01-01.md").write_text("# Daily Journal\nPersonal content")
        
        yield vault_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def areas_with_insufficient_content(self):
        """Create a temporary vault with Areas folder but insufficient content."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir)
        
        # Create Areas folder with minimal content
        areas_path = vault_path / "Areas"
        areas_path.mkdir()
        (areas_path / "Computer Science").mkdir()
        (areas_path / "Computer Science" / "single-note.md").write_text("# Single Note\nMinimal content")
        
        yield vault_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_init_default_settings(self, temp_vault):
        """Test AreasContentFilter initialization with default settings."""
        with patch('jarvis.tools.dataset_generation.filters.areas_filter.get_settings') as mock_settings:
            mock_settings.return_value.dataset_areas_folder_name = "Areas"
            
            filter_instance = AreasContentFilter(str(temp_vault))
            
            assert filter_instance.vault_path == temp_vault.resolve()
            assert filter_instance.areas_folder_name == "Areas"
            assert filter_instance.areas_folder_path == temp_vault.resolve() / "Areas"
            assert filter_instance.min_content_threshold == 5

    def test_init_custom_areas_folder_name(self, temp_vault):
        """Test AreasContentFilter initialization with custom areas folder name."""
        filter_instance = AreasContentFilter(str(temp_vault), areas_folder_name="Knowledge")
        
        assert filter_instance.areas_folder_name == "Knowledge"
        assert filter_instance.areas_folder_path == temp_vault.resolve() / "Knowledge"

    def test_init_custom_min_content_threshold(self, temp_vault):
        """Test AreasContentFilter initialization with custom minimum content threshold."""
        filter_instance = AreasContentFilter(str(temp_vault), min_content_threshold=10)
        
        assert filter_instance.min_content_threshold == 10

    def test_validate_areas_folder_success(self, temp_vault):
        """Test successful validation of Areas folder with sufficient content."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        result = filter_instance.validate_areas_folder()
        
        assert result["areas_folder_exists"] is True
        assert result["areas_folder_name"] == "Areas"
        assert result["markdown_file_count"] == 5  # 5 markdown files in Areas
        assert result["subdirectory_count"] == 3   # 3 subdirectories in Areas
        assert result["total_size_bytes"] > 0
        assert result["validation_passed"] is True
        assert result["error_message"] is None

    def test_validate_areas_folder_vault_not_found(self):
        """Test validation when vault path doesn't exist."""
        nonexistent_path = "/nonexistent/vault/path"
        filter_instance = AreasContentFilter(nonexistent_path)
        
        with pytest.raises(VaultValidationError) as exc_info:
            filter_instance.validate_areas_folder()
        
        assert "Vault path does not exist" in str(exc_info.value)
        assert exc_info.value.vault_path == nonexistent_path
        assert exc_info.value.validation_type == "vault_not_found"

    def test_validate_areas_folder_areas_not_found(self, empty_vault):
        """Test validation when Areas folder doesn't exist."""
        filter_instance = AreasContentFilter(str(empty_vault))
        
        with pytest.raises(AreasNotFoundError) as exc_info:
            filter_instance.validate_areas_folder()
        
        assert "Areas/ folder not found in vault" in str(exc_info.value)
        assert exc_info.value.vault_path == str(empty_vault.resolve())
        assert exc_info.value.areas_folder_name == "Areas"

    def test_validate_areas_folder_areas_is_file(self, temp_vault):
        """Test validation when Areas path exists but is a file, not a directory."""
        # Create a file named "Areas" instead of a directory
        areas_file = temp_vault / "Areas"
        import shutil
        shutil.rmtree(areas_file)  # Remove the directory
        areas_file.write_text("This is a file, not a directory")
        
        filter_instance = AreasContentFilter(str(temp_vault))
        
        with pytest.raises(AreasNotFoundError) as exc_info:
            filter_instance.validate_areas_folder()
        
        assert "Areas/ folder not found in vault" in str(exc_info.value)

    def test_validate_areas_folder_insufficient_content(self, areas_with_insufficient_content):
        """Test validation when Areas folder has insufficient content."""
        filter_instance = AreasContentFilter(str(areas_with_insufficient_content))
        
        with pytest.raises(InsufficientAreasContentError) as exc_info:
            filter_instance.validate_areas_folder()
        
        assert "Insufficient notes in Areas/ folder" in str(exc_info.value)
        assert exc_info.value.areas_count == 1
        assert exc_info.value.required_minimum == 5

    def test_validate_areas_folder_custom_threshold(self, areas_with_insufficient_content):
        """Test validation with custom minimum content threshold."""
        filter_instance = AreasContentFilter(str(areas_with_insufficient_content), min_content_threshold=1)
        
        result = filter_instance.validate_areas_folder()
        
        assert result["validation_passed"] is True
        assert result["markdown_file_count"] == 1

    def test_validate_areas_folder_ignores_hidden_files(self, temp_vault):
        """Test that validation ignores hidden files and system files."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        result = filter_instance.validate_areas_folder()
        
        # Should not count .hidden-file.md or .DS_Store
        assert result["markdown_file_count"] == 5  # Only visible .md files

    def test_validate_areas_folder_handles_file_access_errors(self, temp_vault):
        """Test validation handles file access errors gracefully."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Mock Path.stat to raise an error for one file
        original_stat = Path.stat
        def mock_stat(self, **kwargs):
            if "algorithms.md" in str(self):
                raise OSError("Permission denied")
            return original_stat(self, **kwargs)
        
        with patch.object(Path, 'stat', mock_stat):
            result = filter_instance.validate_areas_folder()
            
            # Should still pass validation, just skip the problematic file
            assert result["validation_passed"] is True
            # Total size might be less due to skipped file, but validation should succeed

    def test_is_areas_content_basic_cases(self, temp_vault):
        """Test basic Areas content identification."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Areas content should return True
        assert filter_instance.is_areas_content("Areas/Computer Science/algorithms.md") is True
        assert filter_instance.is_areas_content("Areas/Natural Science/physics.md") is True
        assert filter_instance.is_areas_content("Areas/Business/strategy.md") is True
        
        # Non-Areas content should return False
        assert filter_instance.is_areas_content("Journal/2024-01-01.md") is False
        assert filter_instance.is_areas_content("Inbox/quick-note.md") is False
        assert filter_instance.is_areas_content("People/john-doe.md") is False

    def test_is_areas_content_edge_cases(self, temp_vault):
        """Test Areas content identification edge cases."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Root Areas folder itself
        assert filter_instance.is_areas_content("Areas") is True
        
        # Areas with different path separators
        assert filter_instance.is_areas_content("Areas\\Computer Science\\algorithms.md") is True
        
        # Areas with leading/trailing slashes
        assert filter_instance.is_areas_content("/Areas/Computer Science/algorithms.md") is True
        assert filter_instance.is_areas_content("Areas/Computer Science/algorithms.md/") is True
        
        # Similar but not Areas folder
        assert filter_instance.is_areas_content("AreasBackup/file.md") is False
        assert filter_instance.is_areas_content("MyAreas/file.md") is False

    def test_is_areas_content_custom_folder_name(self, temp_vault):
        """Test Areas content identification with custom folder name."""
        filter_instance = AreasContentFilter(str(temp_vault), areas_folder_name="Knowledge")
        
        # Should use custom folder name
        assert filter_instance.is_areas_content("Knowledge/Computer Science/algorithms.md") is True
        assert filter_instance.is_areas_content("Areas/Computer Science/algorithms.md") is False

    def test_filter_file_paths_basic(self, temp_vault):
        """Test basic file path filtering."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        all_paths = [
            Path("Areas/Computer Science/algorithms.md"),
            Path("Areas/Natural Science/physics.md"),
            Path("Journal/2024-01-01.md"),
            Path("Inbox/quick-note.md"),
            Path("Areas/Business/strategy.md"),
            Path("People/john-doe.md")
        ]
        
        filtered_paths = filter_instance.filter_file_paths(all_paths)
        
        assert len(filtered_paths) == 3
        expected_paths = {
            Path("Areas/Computer Science/algorithms.md"),
            Path("Areas/Natural Science/physics.md"),
            Path("Areas/Business/strategy.md")
        }
        assert set(filtered_paths) == expected_paths

    def test_filter_file_paths_empty_list(self, temp_vault):
        """Test filtering empty file path list."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        filtered_paths = filter_instance.filter_file_paths([])
        
        assert filtered_paths == []

    def test_filter_file_paths_no_areas_content(self, temp_vault):
        """Test filtering when no Areas content is present."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        all_paths = [
            Path("Journal/2024-01-01.md"),
            Path("Inbox/quick-note.md"),
            Path("People/john-doe.md")
        ]
        
        filtered_paths = filter_instance.filter_file_paths(all_paths)
        
        assert filtered_paths == []

    def test_filter_file_paths_all_areas_content(self, temp_vault):
        """Test filtering when all content is Areas content."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        all_paths = [
            Path("Areas/Computer Science/algorithms.md"),
            Path("Areas/Natural Science/physics.md"),
            Path("Areas/Business/strategy.md")
        ]
        
        filtered_paths = filter_instance.filter_file_paths(all_paths)
        
        assert len(filtered_paths) == 3
        assert set(filtered_paths) == set(all_paths)

    def test_get_areas_structure_success(self, temp_vault):
        """Test getting Areas folder structure information."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        structure = filter_instance.get_areas_structure()
        
        assert structure["exists"] is True
        assert structure["name"] == "Areas"
        assert structure["path"] == str(temp_vault.resolve() / "Areas")
        assert len(structure["subdirectories"]) == 3
        assert len(structure["markdown_files"]) == 5
        assert structure["total_files"] == 5
        assert structure["total_size_bytes"] > 0
        
        # Check that subdirectories are correct
        expected_subdirs = {
            "Areas/Computer Science",
            "Areas/Natural Science", 
            "Areas/Business"
        }
        assert set(structure["subdirectories"]) == expected_subdirs
        
        # Check that markdown files are included
        assert "Areas/Computer Science/algorithms.md" in structure["markdown_files"]
        assert "Areas/Natural Science/physics.md" in structure["markdown_files"]

    def test_get_areas_structure_areas_not_found(self, empty_vault):
        """Test getting Areas structure when Areas folder doesn't exist."""
        filter_instance = AreasContentFilter(str(empty_vault))
        
        structure = filter_instance.get_areas_structure()
        
        assert structure["exists"] is False
        assert "Areas folder not found" in structure["error"]

    def test_get_areas_structure_handles_errors(self, temp_vault):
        """Test that get_areas_structure handles errors gracefully."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Mock iterdir to raise an exception
        with patch.object(Path, 'iterdir', side_effect=PermissionError("Access denied")):
            structure = filter_instance.get_areas_structure()
            
            assert structure["exists"] is True  # Path exists check happens first
            assert "Error analyzing Areas structure" in structure["error"]

    def test_get_areas_structure_ignores_hidden_files(self, temp_vault):
        """Test that get_areas_structure ignores hidden files."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        structure = filter_instance.get_areas_structure()
        
        # Should not include .hidden-file.md
        hidden_files = [f for f in structure["markdown_files"] if f.startswith(".")]
        assert len(hidden_files) == 0

    def test_get_exclusion_summary_success(self, temp_vault):
        """Test getting exclusion summary information."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        summary = filter_instance.get_exclusion_summary()
        
        assert summary["filtering_enabled"] is True
        assert summary["areas_folder_name"] == "Areas"
        assert summary["areas_folder_path"] == str(temp_vault.resolve() / "Areas")
        assert "Only content from the Areas folder will be included" in summary["privacy_note"]
        
        # Check excluded folders
        expected_excluded = {"Journal", "Inbox", "People"}
        assert set(summary["excluded_folders"]) == expected_excluded
        assert summary["excluded_folder_count"] == 3

    def test_get_exclusion_summary_no_excluded_folders(self, temp_vault):
        """Test exclusion summary when only Areas folder exists."""
        # Remove non-Areas folders
        import shutil
        shutil.rmtree(temp_vault / "Journal")
        shutil.rmtree(temp_vault / "Inbox") 
        shutil.rmtree(temp_vault / "People")
        
        filter_instance = AreasContentFilter(str(temp_vault))
        
        summary = filter_instance.get_exclusion_summary()
        
        assert summary["excluded_folders"] == []
        assert summary["excluded_folder_count"] == 0

    def test_get_exclusion_summary_handles_errors(self, temp_vault):
        """Test that get_exclusion_summary handles errors gracefully."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Mock iterdir to raise an exception
        with patch.object(Path, 'iterdir', side_effect=OSError("Disk error")):
            summary = filter_instance.get_exclusion_summary()
            
            assert summary["filtering_enabled"] is True
            assert "Error analyzing exclusions" in summary["error"]

    def test_get_exclusion_summary_ignores_hidden_folders(self, temp_vault):
        """Test that exclusion summary ignores hidden folders."""
        # Create a hidden folder
        hidden_folder = temp_vault / ".hidden_folder"
        hidden_folder.mkdir()
        (hidden_folder / "file.md").write_text("Hidden content")
        
        filter_instance = AreasContentFilter(str(temp_vault))
        
        summary = filter_instance.get_exclusion_summary()
        
        # Should not include .hidden_folder in excluded folders
        assert ".hidden_folder" not in summary["excluded_folders"]

    def test_str_representation(self, temp_vault):
        """Test string representation of AreasContentFilter."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        str_repr = str(filter_instance)
        
        assert "AreasContentFilter" in str_repr
        assert str(temp_vault) in str_repr
        assert "Areas" in str_repr

    def test_repr_representation(self, temp_vault):
        """Test detailed string representation of AreasContentFilter."""
        filter_instance = AreasContentFilter(str(temp_vault), min_content_threshold=10)
        
        repr_str = repr(filter_instance)
        
        assert "AreasContentFilter" in repr_str
        assert f"vault_path='{temp_vault.resolve()}'" in repr_str
        assert "areas_folder_name='Areas'" in repr_str
        assert "min_content_threshold=10" in repr_str


class TestAreasContentFilterIntegration:
    """Integration tests for AreasContentFilter with realistic vault structures."""

    @pytest.fixture
    def realistic_vault(self):
        """Create a realistic vault structure for integration testing."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir)
        
        # Create a realistic Obsidian vault structure
        areas_path = vault_path / "Areas"
        areas_path.mkdir()
        
        # Computer Science area with nested structure
        cs_path = areas_path / "Computer Science"
        cs_path.mkdir()
        (cs_path / "Programming Languages").mkdir()
        (cs_path / "Algorithms").mkdir()
        (cs_path / "Data Structures").mkdir()
        
        # Create files with realistic content
        (cs_path / "Programming Languages" / "Python.md").write_text("""# Python Programming

Python is a high-level programming language.

## Features
- Dynamic typing
- Interpreted
- Object-oriented

## Links
- [[Data Structures]]
- [[Algorithms]]
""")
        
        (cs_path / "Algorithms" / "Sorting.md").write_text("""# Sorting Algorithms

Various algorithms for sorting data.

## Types
- Bubble Sort
- Quick Sort
- Merge Sort

## Related
- [[Data Structures]]
""")
        
        (cs_path / "Data Structures" / "Arrays.md").write_text("""# Arrays

Arrays are fundamental data structures.

## Properties
- Fixed size
- Contiguous memory
- Random access

## See Also
- [[Python]]
- [[Sorting]]
""")
        
        # Natural Science area
        ns_path = areas_path / "Natural Science"
        ns_path.mkdir()
        (ns_path / "Physics").mkdir()
        (ns_path / "Chemistry").mkdir()
        
        (ns_path / "Physics" / "Quantum Mechanics.md").write_text("""# Quantum Mechanics

The branch of physics dealing with quantum phenomena.

## Principles
- Wave-particle duality
- Uncertainty principle
- Superposition
""")
        
        (ns_path / "Chemistry" / "Organic Chemistry.md").write_text("""# Organic Chemistry

The study of carbon-based compounds.

## Topics
- Hydrocarbons
- Functional groups
- Reactions
""")
        
        # Business area
        business_path = areas_path / "Business"
        business_path.mkdir()
        (business_path / "Strategy.md").write_text("""# Business Strategy

Strategic planning and execution.

## Components
- Vision and mission
- SWOT analysis
- Competitive advantage
""")
        
        # Personal folders (should be excluded)
        journal_path = vault_path / "Journal"
        journal_path.mkdir()
        for i in range(1, 8):  # Week of journal entries
            (journal_path / f"2024-01-0{i}.md").write_text(f"""# Daily Journal - 2024-01-0{i}

Personal thoughts and activities for today.

## Mood
Happy

## Activities
- Work
- Exercise
- Reading

## Reflections
Personal reflections about the day.
""")
        
        inbox_path = vault_path / "Inbox"
        inbox_path.mkdir()
        (inbox_path / "Random Thought.md").write_text("# Random Thought\n\nSomething I need to organize later.")
        (inbox_path / "Meeting Notes.md").write_text("# Meeting Notes\n\nNotes from today's meeting.")
        
        people_path = vault_path / "People"
        people_path.mkdir()
        (people_path / "Alice Smith.md").write_text("# Alice Smith\n\nContact: alice@example.com\nRole: Colleague")
        (people_path / "Bob Johnson.md").write_text("# Bob Johnson\n\nContact: bob@example.com\nRole: Friend")
        
        # Templates folder
        templates_path = vault_path / "Templates"
        templates_path.mkdir()
        (templates_path / "Daily Note Template.md").write_text("# {{date}}\n\n## Tasks\n\n## Notes\n")
        
        yield vault_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_realistic_vault_validation(self, realistic_vault):
        """Test validation with a realistic vault structure."""
        filter_instance = AreasContentFilter(str(realistic_vault))
        
        result = filter_instance.validate_areas_folder()
        
        assert result["validation_passed"] is True
        assert result["markdown_file_count"] == 6  # 6 files in Areas
        assert result["subdirectory_count"] == 3   # CS, NS, Business
        assert result["total_size_bytes"] > 800   # Substantial content

    def test_realistic_vault_filtering(self, realistic_vault):
        """Test filtering with a realistic vault structure."""
        filter_instance = AreasContentFilter(str(realistic_vault))
        
        # Simulate all files in vault
        all_files = []
        for root, dirs, files in os.walk(realistic_vault):
            for file in files:
                if file.endswith('.md'):
                    rel_path = Path(root).relative_to(realistic_vault) / file
                    all_files.append(rel_path)
        
        filtered_files = filter_instance.filter_file_paths(all_files)
        
        # Should only include Areas files
        assert len(filtered_files) == 6  # Only Areas files
        
        # Verify all filtered files are in Areas
        for file_path in filtered_files:
            assert str(file_path).startswith("Areas/")
        
        # Verify specific files are included
        areas_files = [str(f) for f in filtered_files]
        assert "Areas/Computer Science/Programming Languages/Python.md" in areas_files
        assert "Areas/Natural Science/Physics/Quantum Mechanics.md" in areas_files
        assert "Areas/Business/Strategy.md" in areas_files

    def test_realistic_vault_exclusion_summary(self, realistic_vault):
        """Test exclusion summary with a realistic vault structure."""
        filter_instance = AreasContentFilter(str(realistic_vault))
        
        summary = filter_instance.get_exclusion_summary()
        
        expected_excluded = {"Journal", "Inbox", "People", "Templates"}
        assert set(summary["excluded_folders"]) == expected_excluded
        assert summary["excluded_folder_count"] == 4

    def test_realistic_vault_areas_structure(self, realistic_vault):
        """Test Areas structure analysis with a realistic vault."""
        filter_instance = AreasContentFilter(str(realistic_vault))
        
        structure = filter_instance.get_areas_structure()
        
        assert structure["exists"] is True
        assert structure["total_files"] == 6
        assert len(structure["subdirectories"]) == 3
        
        # Check that nested structure is captured
        subdirs = structure["subdirectories"]
        assert "Areas/Computer Science" in subdirs
        assert "Areas/Natural Science" in subdirs
        assert "Areas/Business" in subdirs

    def test_performance_with_large_file_list(self, realistic_vault):
        """Test performance with a large number of files."""
        filter_instance = AreasContentFilter(str(realistic_vault))
        
        # Create a large list of file paths
        large_file_list = []
        for i in range(1000):
            if i % 10 == 0:  # 10% Areas content
                large_file_list.append(Path(f"Areas/Computer Science/file_{i}.md"))
            else:  # 90% non-Areas content
                large_file_list.append(Path(f"Journal/file_{i}.md"))
        
        import time
        start_time = time.time()
        filtered_files = filter_instance.filter_file_paths(large_file_list)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 1.0  # Less than 1 second
        assert len(filtered_files) == 100  # 10% of 1000 files


class TestAreasContentFilterErrorHandling:
    """Test suite for error handling in AreasContentFilter."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault structure for testing."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir)
        
        # Create Areas folder structure
        areas_path = vault_path / "Areas"
        areas_path.mkdir()
        
        # Create subdirectories in Areas
        (areas_path / "Computer Science").mkdir()
        (areas_path / "Natural Science").mkdir()
        (areas_path / "Business").mkdir()
        
        # Create markdown files in Areas
        (areas_path / "Computer Science" / "algorithms.md").write_text("# Algorithms\nContent about algorithms")
        (areas_path / "Computer Science" / "data-structures.md").write_text("# Data Structures\nContent about data structures")
        (areas_path / "Natural Science" / "physics.md").write_text("# Physics\nContent about physics")
        (areas_path / "Natural Science" / "chemistry.md").write_text("# Chemistry\nContent about chemistry")
        (areas_path / "Business" / "strategy.md").write_text("# Strategy\nContent about business strategy")
        
        yield vault_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

    def test_vault_path_resolution_error(self):
        """Test handling of vault path resolution errors."""
        # Use a path that might cause resolution issues
        problematic_path = "~/nonexistent/vault/with/tildes"
        
        # Should not raise exception during initialization
        filter_instance = AreasContentFilter(problematic_path)
        
        # Error should be caught during validation
        with pytest.raises(VaultValidationError):
            filter_instance.validate_areas_folder()

    def test_permission_errors_during_validation(self, temp_vault):
        """Test handling of permission errors during validation."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Mock Path.exists to raise PermissionError
        with patch.object(Path, 'exists', side_effect=PermissionError("Permission denied")):
            # The PermissionError should propagate up since it's not caught in the current implementation
            with pytest.raises(PermissionError):
                filter_instance.validate_areas_folder()

    def test_unicode_handling_in_paths(self, temp_vault):
        """Test handling of Unicode characters in file paths."""
        # Create files with Unicode names
        areas_path = temp_vault / "Areas"
        unicode_folder = areas_path / "Ñatural Sciençe"
        unicode_folder.mkdir()
        unicode_file = unicode_folder / "Phýsics.md"
        unicode_file.write_text("# Phýsics\nContent with Unicode: αβγδε")
        
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Should handle Unicode paths correctly
        result = filter_instance.validate_areas_folder()
        assert result["validation_passed"] is True
        assert result["markdown_file_count"] == 6  # Original 5 + 1 Unicode file
        
        # Should identify Unicode paths as Areas content
        unicode_path = "Areas/Ñatural Sciençe/Phýsics.md"
        assert filter_instance.is_areas_content(unicode_path) is True

    def test_very_long_paths(self, temp_vault):
        """Test handling of very long file paths."""
        areas_path = temp_vault / "Areas"
        
        # Create a deeply nested structure
        deep_path = areas_path
        for i in range(10):  # Create 10 levels of nesting
            deep_path = deep_path / f"level_{i}_with_a_very_long_name_that_might_cause_issues"
            deep_path.mkdir()
        
        long_file = deep_path / "file_with_extremely_long_name_that_tests_path_length_limits.md"
        long_file.write_text("# Long Path Test\nContent in deeply nested file")
        
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Should handle long paths correctly
        result = filter_instance.validate_areas_folder()
        assert result["validation_passed"] is True
        
        # Should identify long paths as Areas content
        long_path_str = str(long_file.relative_to(temp_vault))
        assert filter_instance.is_areas_content(long_path_str) is True

    def test_concurrent_access_simulation(self, temp_vault):
        """Test behavior under simulated concurrent access."""
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Simulate concurrent validation calls
        results = []
        for _ in range(5):
            try:
                result = filter_instance.validate_areas_folder()
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        # All calls should succeed with consistent results
        assert len(results) == 5
        for result in results:
            assert "error" not in result
            assert result["validation_passed"] is True

    def test_memory_efficiency_with_large_structure(self, temp_vault):
        """Test memory efficiency with large vault structures."""
        areas_path = temp_vault / "Areas"
        
        # Create many files to test memory usage
        for i in range(100):
            folder = areas_path / f"Domain_{i}"
            folder.mkdir()
            for j in range(10):
                file_path = folder / f"note_{j}.md"
                file_path.write_text(f"# Note {j} in Domain {i}\nContent for note {j}")
        
        filter_instance = AreasContentFilter(str(temp_vault))
        
        # Should handle large structures without issues
        result = filter_instance.validate_areas_folder()
        structure = filter_instance.get_areas_structure()
        
        # Verify the operations completed successfully
        assert result["validation_passed"] is True
        assert structure["total_files"] == 1005  # Original 5 + 1000 new files
        assert structure["exists"] is True