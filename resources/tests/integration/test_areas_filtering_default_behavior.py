"""
Integration tests for Areas/ filtering default behavior.

Tests that the new default configuration correctly filters to Areas/ content only,
and that existing functionality works with the new defaults.
"""

import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.utils.config import get_settings


class TestAreasFilteringDefaultBehavior:
    """Test Areas/ filtering default behavior and configuration."""

    @pytest.fixture
    def temp_vault_with_areas(self):
        """Create a temporary vault with Areas/ and non-Areas/ content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir) / "test_vault"
            vault_path.mkdir()
            
            # Create .obsidian directory
            obsidian_dir = vault_path / ".obsidian"
            obsidian_dir.mkdir()
            
            # Create Areas/ folder structure (should be included by default)
            areas_dir = vault_path / "Areas"
            areas_dir.mkdir()
            
            areas_cs_dir = areas_dir / "Computer Science"
            areas_cs_dir.mkdir()
            
            areas_data_dir = areas_dir / "Data Analysis"
            areas_data_dir.mkdir()
            
            # Create Areas/ content (need at least 5 notes for validation)
            areas_notes = [
                (areas_cs_dir / "machine_learning.md", """---
title: Machine Learning
tags: [ml, ai]
domains: [artificial-intelligence]
---

# Machine Learning

Core concepts in machine learning.

## Algorithms
- Supervised learning
- Unsupervised learning
- [[deep_learning.md|Deep Learning]]

## Applications
Used in [[data_science.md|Data Science]] projects.
"""),
                (areas_cs_dir / "deep_learning.md", """---
title: Deep Learning
tags: [dl, neural-networks]
domains: [artificial-intelligence]
---

# Deep Learning

Advanced machine learning using neural networks.

## Connection
Built on [[machine_learning.md]] principles.
"""),
                (areas_data_dir / "data_science.md", """---
title: Data Science
tags: [data, analytics]
domains: [data-analysis]
---

# Data Science

Data science methodology and tools.

## ML Connection
Uses [[machine_learning.md]] techniques.
"""),
                (areas_cs_dir / "algorithms.md", """---
title: Algorithms
tags: [algorithms, cs]
domains: [computer-science]
---

# Algorithms

Fundamental algorithms and data structures.

## Types
- Sorting algorithms
- Search algorithms
- Graph algorithms

## Connection
Used in [[machine_learning.md]] implementations.
"""),
                (areas_data_dir / "statistics.md", """---
title: Statistics
tags: [stats, math]
domains: [mathematics]
---

# Statistics

Statistical methods and analysis.

## Applications
- Hypothesis testing
- Regression analysis
- Used in [[data_science.md]] workflows

## Connection
Foundation for [[machine_learning.md]] models.
"""),
            ]
            
            # Create non-Areas/ content (should be excluded by default)
            journal_dir = vault_path / "Journal"
            journal_dir.mkdir()
            
            inbox_dir = vault_path / "Inbox"
            inbox_dir.mkdir()
            
            projects_dir = vault_path / "Projects"
            projects_dir.mkdir()
            
            non_areas_notes = [
                (journal_dir / "2024-01-01.md", """---
title: Daily Journal
tags: [journal, personal]
---

# Daily Journal Entry

Personal thoughts and reflections.

## Work Notes
Worked on [[machine_learning.md]] project today.
"""),
                (inbox_dir / "random_idea.md", """---
title: Random Idea
tags: [inbox, temporary]
---

# Random Idea

Quick note to process later.

## Related
Might connect to [[data_science.md]].
"""),
                (projects_dir / "ml_project.md", """---
title: ML Project Alpha
tags: [project, active]
---

# ML Project Alpha

Active machine learning project.

## Foundation
Based on [[machine_learning.md]] concepts.
"""),
            ]
            
            # Write all notes
            for note_path, content in areas_notes + non_areas_notes:
                note_path.write_text(content, encoding='utf-8')
            
            yield vault_path

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create mock vector encoder for consistent testing."""
        mock_encoder = Mock()
        
        def encode_documents(documents):
            # Create deterministic embeddings
            import numpy as np
            embeddings = []
            for i, doc in enumerate(documents):
                # Use index for deterministic embeddings
                np.random.seed(i)
                embedding = np.random.rand(384)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        mock_encoder.encode_documents.side_effect = encode_documents
        mock_encoder.encode_batch.side_effect = encode_documents
        mock_encoder.get_embedding_dimension.return_value = 384
        
        return mock_encoder

    def test_default_areas_filtering_enabled(self, temp_vault_with_areas, temp_output_dir, mock_vector_encoder):
        """Test that Areas/ filtering is enabled by default."""
        # Create generator with default settings (should enable Areas/ filtering)
        generator = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            skip_validation=True  # Skip validation for testing
        )
        
        # Verify that Areas/ filtering is enabled by default
        assert generator.areas_only is True, "Areas/ filtering should be enabled by default"
        
        # Mock vector encoder
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            # Generate datasets
            result = generator.generate_datasets(
                notes_filename="areas_only_notes.csv",
                pairs_filename="areas_only_pairs.csv"
            )
            
            assert result.success is True, f"Dataset generation should succeed: {result.error_message}"
            
            # Load and verify datasets
            notes_df = pd.read_csv(temp_output_dir / "areas_only_notes.csv")
            pairs_df = pd.read_csv(temp_output_dir / "areas_only_pairs.csv")
            
            # Should only have 5 notes from Areas/ folder
            assert len(notes_df) == 5, f"Should have 5 Areas/ notes, got {len(notes_df)}"
            
            # Verify all notes are from Areas/ folder
            for note_path in notes_df['note_path']:
                assert "Areas" in str(note_path), f"Note path should contain 'Areas': {note_path}"
            
            # Verify no Journal/, Inbox/, or Projects/ content
            for note_path in notes_df['note_path']:
                assert "Journal" not in str(note_path), f"Should not include Journal content: {note_path}"
                assert "Inbox" not in str(note_path), f"Should not include Inbox content: {note_path}"
                assert "Projects" not in str(note_path), f"Should not include Projects content: {note_path}"
            
            # Verify pairs dataset only references Areas/ notes
            all_pair_paths = set(pairs_df['note_a_path']) | set(pairs_df['note_b_path'])
            for pair_path in all_pair_paths:
                assert "Areas" in str(pair_path), f"Pair path should contain 'Areas': {pair_path}"
            
            print(f"✓ Default Areas/ filtering working correctly:")
            print(f"  - Processed {len(notes_df)} notes from Areas/ folder")
            print(f"  - Generated {len(pairs_df)} pairs")
            print(f"  - Excluded Journal/, Inbox/, and Projects/ content")

    def test_explicit_areas_filtering_disabled(self, temp_vault_with_areas, temp_output_dir, mock_vector_encoder):
        """Test that Areas/ filtering can be explicitly disabled."""
        # Create generator with Areas/ filtering explicitly disabled
        generator = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            areas_only=False,  # Explicitly disable
            skip_validation=True  # Skip validation for testing
        )
        
        # Verify that Areas/ filtering is disabled
        assert generator.areas_only is False, "Areas/ filtering should be disabled when explicitly set"
        
        # Mock vector encoder
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            # Generate datasets
            result = generator.generate_datasets(
                notes_filename="full_vault_notes.csv",
                pairs_filename="full_vault_pairs.csv"
            )
            
            assert result.success is True, f"Dataset generation should succeed: {result.error_message}"
            
            # Load and verify datasets
            notes_df = pd.read_csv(temp_output_dir / "full_vault_notes.csv")
            pairs_df = pd.read_csv(temp_output_dir / "full_vault_pairs.csv")
            
            # Should have all 8 notes (5 Areas/ + 3 non-Areas/)
            assert len(notes_df) == 8, f"Should have 8 total notes, got {len(notes_df)}"
            
            # Verify we have both Areas/ and non-Areas/ content
            areas_notes = [path for path in notes_df['note_path'] if "Areas" in str(path)]
            non_areas_notes = [path for path in notes_df['note_path'] if "Areas" not in str(path)]
            
            assert len(areas_notes) == 5, f"Should have 5 Areas/ notes, got {len(areas_notes)}"
            assert len(non_areas_notes) == 3, f"Should have 3 non-Areas/ notes, got {len(non_areas_notes)}"
            
            # Verify we have Journal/, Inbox/, and Projects/ content
            journal_notes = [path for path in notes_df['note_path'] if "Journal" in str(path)]
            inbox_notes = [path for path in notes_df['note_path'] if "Inbox" in str(path)]
            projects_notes = [path for path in notes_df['note_path'] if "Projects" in str(path)]
            
            assert len(journal_notes) == 1, "Should have 1 Journal note"
            assert len(inbox_notes) == 1, "Should have 1 Inbox note"
            assert len(projects_notes) == 1, "Should have 1 Projects note"
            
            print(f"✓ Explicit Areas/ filtering disable working correctly:")
            print(f"  - Processed {len(notes_df)} notes from entire vault")
            print(f"  - Generated {len(pairs_df)} pairs")
            print(f"  - Included all content types")

    def test_default_output_directory_configuration(self, temp_vault_with_areas):
        """Test that the new default output directory is configured correctly."""
        # Test default output directory from configuration
        settings = get_settings()
        
        # The actual default might be overridden by environment variables
        # Let's test that the configuration system works and path expansion works
        expanded_path = settings.get_dataset_output_path()
        assert expanded_path.is_absolute(), "Expanded path should be absolute"
        assert "datasets" in str(expanded_path), "Expanded path should contain 'datasets'"
        
        # Test that the configuration field exists and has a reasonable value
        assert settings.dataset_output_dir is not None, "Dataset output directory should be configured"
        assert len(settings.dataset_output_dir) > 0, "Dataset output directory should not be empty"
        
        print(f"✓ Default output directory configuration working correctly:")
        print(f"  - Configured: {settings.dataset_output_dir}")
        print(f"  - Expanded: {expanded_path}")

    def test_areas_folder_name_configuration(self, temp_vault_with_areas, temp_output_dir, mock_vector_encoder):
        """Test that the Areas/ folder name can be configured."""
        # Test default Areas folder name
        settings = get_settings()
        assert settings.dataset_areas_folder_name == "Areas", \
            f"Default Areas folder name should be 'Areas', got {settings.dataset_areas_folder_name}"
        
        # Create generator with default settings
        generator = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            skip_validation=True  # Skip validation for testing
        )
        
        # Verify the Areas filter is using the correct folder name
        assert generator.areas_filter.areas_folder_name == "Areas", \
            "Areas filter should use configured folder name"
        
        print(f"✓ Areas folder name configuration working correctly:")
        print(f"  - Configured name: {settings.dataset_areas_folder_name}")

    def test_areas_filtering_validation(self, temp_vault_with_areas, temp_output_dir):
        """Test that Areas/ filtering validation works correctly."""
        # Create generator with default settings
        generator = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            skip_validation=True  # Skip validation for testing
        )
        
        # Test validation with Areas/ folder present
        validation_result = generator.validate_vault()
        assert validation_result.valid is True, "Validation should pass with Areas/ folder present"
        assert validation_result.areas_folder_exists is True, "Should detect Areas/ folder exists"
        assert validation_result.areas_notes_count == 5, f"Should count 5 Areas/ notes, got {validation_result.areas_notes_count}"
        assert validation_result.filtering_mode == "areas_only", "Should report areas_only filtering mode"
        
        print(f"✓ Areas/ filtering validation working correctly:")
        print(f"  - Areas folder exists: {validation_result.areas_folder_exists}")
        print(f"  - Areas notes count: {validation_result.areas_notes_count}")
        print(f"  - Filtering mode: {validation_result.filtering_mode}")

    def test_areas_filtering_with_missing_areas_folder(self, temp_output_dir):
        """Test behavior when Areas/ folder is missing."""
        # Create vault without Areas/ folder
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir) / "vault_no_areas"
            vault_path.mkdir()
            
            # Create .obsidian directory
            obsidian_dir = vault_path / ".obsidian"
            obsidian_dir.mkdir()
            
            # Create only non-Areas/ content
            journal_dir = vault_path / "Journal"
            journal_dir.mkdir()
            
            journal_note = journal_dir / "2024-01-01.md"
            journal_note.write_text("""---
title: Daily Journal
---

# Daily Journal Entry

Personal thoughts.
""")
            
            # Create generator with default settings (Areas/ filtering enabled)
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                skip_validation=True  # Skip validation for testing
            )
            
            # Test validation should fail
            validation_result = generator.validate_vault()
            assert validation_result.valid is False, "Validation should fail without Areas/ folder"
            assert validation_result.areas_folder_exists is False, "Should detect Areas/ folder missing"
            assert "Areas/ folder not found" in str(validation_result.errors), "Should report missing Areas/ folder"
            
            print(f"✓ Missing Areas/ folder handling working correctly:")
            print(f"  - Validation failed as expected")
            print(f"  - Error message: {validation_result.errors}")

    def test_areas_filtering_summary_metadata(self, temp_vault_with_areas, temp_output_dir, mock_vector_encoder):
        """Test that filtering metadata is included in generation summary."""
        # Create generator with default settings
        generator = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            skip_validation=True  # Skip validation for testing
        )
        
        # Mock vector encoder
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            # Generate datasets
            result = generator.generate_datasets()
            
            assert result.success is True, "Dataset generation should succeed"
            
            # Verify filtering metadata in summary
            summary = result.summary
            assert summary.filtering_enabled is True, "Summary should indicate filtering is enabled"
            assert summary.areas_folder_path is not None, "Summary should include Areas/ folder path"
            assert "Areas" in summary.areas_folder_path, "Areas folder path should contain 'Areas'"
            assert summary.privacy_mode is True, "Summary should indicate privacy mode is active"
            
            # Verify excluded folders are reported
            assert len(summary.excluded_folders) > 0, "Summary should report excluded folders"
            excluded_folder_names = [folder.lower() for folder in summary.excluded_folders]
            assert any("journal" in name for name in excluded_folder_names), "Should report Journal as excluded"
            assert any("inbox" in name for name in excluded_folder_names), "Should report Inbox as excluded"
            assert any("projects" in name for name in excluded_folder_names), "Should report Projects as excluded"
            
            print(f"✓ Filtering metadata in summary working correctly:")
            print(f"  - Filtering enabled: {summary.filtering_enabled}")
            print(f"  - Areas folder path: {summary.areas_folder_path}")
            print(f"  - Privacy mode: {summary.privacy_mode}")
            print(f"  - Excluded folders: {summary.excluded_folders}")

    def test_backward_compatibility_with_explicit_settings(self, temp_vault_with_areas, temp_output_dir, mock_vector_encoder):
        """Test backward compatibility when explicitly setting areas_only=False."""
        # Create generator with explicit areas_only=False (old behavior)
        generator = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            areas_only=False,  # Explicit old behavior
            skip_validation=True  # Skip validation for testing
        )
        
        # Mock vector encoder
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            # Generate datasets
            result = generator.generate_datasets()
            
            assert result.success is True, "Dataset generation should succeed"
            
            # Load datasets
            notes_df = pd.read_csv(result.notes_dataset_path)
            
            # Should process all notes (backward compatibility)
            assert len(notes_df) == 8, "Should process all notes when areas_only=False"
            
            # Verify filtering metadata reflects disabled state
            summary = result.summary
            assert summary.filtering_enabled is False, "Summary should indicate filtering is disabled"
            assert summary.privacy_mode is False, "Summary should indicate privacy mode is inactive"
            
            print(f"✓ Backward compatibility working correctly:")
            print(f"  - Processed all {len(notes_df)} notes")
            print(f"  - Filtering disabled as requested")

    def test_configuration_override_behavior(self, temp_vault_with_areas, temp_output_dir):
        """Test that constructor parameters override configuration defaults."""
        # Test that explicit constructor parameter overrides config default
        generator_disabled = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            areas_only=False,  # Override default True
            skip_validation=True  # Skip validation for testing
        )
        
        assert generator_disabled.areas_only is False, "Constructor parameter should override config default"
        
        # Test that explicit constructor parameter can also reinforce default
        generator_enabled = DatasetGenerator(
            vault_path=temp_vault_with_areas,
            output_dir=temp_output_dir,
            areas_only=True,  # Explicit True (same as default)
            skip_validation=True  # Skip validation for testing
        )
        
        assert generator_enabled.areas_only is True, "Constructor parameter should work when matching default"
        
        print(f"✓ Configuration override behavior working correctly:")
        print(f"  - Constructor can override defaults")
        print(f"  - Constructor can reinforce defaults")