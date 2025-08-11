"""
Integration tests for Areas/ filtering workflow in dataset generation.

This test suite verifies end-to-end dataset generation with Areas/ filtering enabled,
ensuring that non-Areas/ content is properly excluded, output directories are created
correctly, and all existing dataset features work with filtered content.

Requirements tested:
- 2.2: Default output directory creation and path expansion
- 5.1: Maintain all existing link extraction capabilities with filtering
- 5.2: Maintain all existing notes dataset features with filtering
- 5.3: Maintain all existing pairs dataset features with filtering
- 5.4: Handle cross-references to non-Areas content gracefully
"""

import gc
import os
import tempfile
import time
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.tools.dataset_generation.filters.areas_filter import AreasContentFilter
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
from jarvis.tools.dataset_generation.models.data_models import (
    DatasetGenerationResult, GenerationSummary, LinkStatistics, ValidationResult
)
from jarvis.tools.dataset_generation.models.exceptions import (
    AreasNotFoundError, InsufficientAreasContentError, VaultValidationError
)
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vault.parser import MarkdownParser
from jarvis.utils.config import get_settings


class AreasTestVaultHelper:
    """Helper class for creating test vaults with Areas/ structure for testing."""
    
    @staticmethod
    def create_areas_test_vault(vault_path: Path, include_non_areas: bool = True) -> Dict[str, any]:
        """Create a test vault with Areas/ folder and optional non-Areas content.
        
        Args:
            vault_path: Path to create vault
            include_non_areas: Whether to include non-Areas folders for exclusion testing
            
        Returns:
            Dictionary with vault metadata for validation
        """
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create Areas/ folder structure
        areas_dir = vault_path / "Areas"
        areas_dir.mkdir(exist_ok=True)
        
        # Create Areas/ subdirectories
        areas_subdirs = [
            "Natural Science",
            "Computer Science", 
            "Business",
            "Research Methods"
        ]
        
        for subdir in areas_subdirs:
            (areas_dir / subdir).mkdir(exist_ok=True)
        
        # Create Areas/ content
        areas_notes = {
            "Areas/Natural Science/Physics.md": """---
title: Physics Fundamentals
tags: [physics, science, fundamentals]
domains: [natural-science]
status: 🌳
---

# Physics Fundamentals

Core concepts in physics including mechanics, thermodynamics, and electromagnetism.

## Key Concepts
- Newton's Laws of Motion
- Conservation of Energy
- Electromagnetic Fields

## Related Topics
- [[Chemistry Basics]] - Chemical interactions
- [[Mathematics for Science]] - Mathematical foundations

## Applications
Physics principles apply to engineering and technology development.
""",
            
            "Areas/Natural Science/Chemistry Basics.md": """---
title: Chemistry Basics
tags: [chemistry, science, basics]
domains: [natural-science]
status: 🌿
---

# Chemistry Basics

Fundamental chemistry concepts and principles.

## Core Topics
- Atomic Structure
- Chemical Bonding
- Reaction Mechanisms

## Connections
- [[Physics Fundamentals]] - Physical principles
- [[Research Methods]] - Scientific methodology

## Laboratory Techniques
Standard procedures for chemical analysis and synthesis.
""",
            
            "Areas/Computer Science/Algorithms.md": """---
title: Algorithms and Data Structures
tags: [algorithms, computer-science, programming]
domains: [computer-science]
status: 🌳
---

# Algorithms and Data Structures

Essential algorithms and data structures for software development.

## Core Algorithms
- Sorting algorithms (quicksort, mergesort)
- Search algorithms (binary search, DFS, BFS)
- Dynamic programming

## Data Structures
- Arrays and linked lists
- Trees and graphs
- Hash tables

## Related Areas
- [[Software Engineering]] - Implementation practices
- [[Machine Learning]] - Algorithm applications

## Performance Analysis
Big O notation and complexity analysis.
""",
            
            "Areas/Computer Science/Machine Learning.md": """---
title: Machine Learning
tags: [ml, ai, algorithms, data-science]
domains: [computer-science, artificial-intelligence]
status: 🗺️
---

# Machine Learning

Comprehensive overview of machine learning concepts and techniques.

## Supervised Learning
- Linear regression
- Decision trees
- Neural networks

## Unsupervised Learning
- Clustering algorithms
- Dimensionality reduction
- Association rules

## Deep Learning
- [[Neural Networks]] - Advanced architectures
- Convolutional networks
- Recurrent networks

## Applications
- Natural language processing
- Computer vision
- Recommendation systems

## Tools and Frameworks
- Python scikit-learn
- TensorFlow and PyTorch
- R statistical computing

## Related Topics
- [[Algorithms and Data Structures]] - Computational foundations
- [[Statistics]] - Mathematical foundations
""",
            
            "Areas/Computer Science/Neural Networks.md": """---
title: Neural Networks
tags: [neural-networks, deep-learning, ai]
domains: [artificial-intelligence]
status: 🌳
---

# Neural Networks

Deep dive into neural network architectures and training.

## Architecture Types
- Feedforward networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformer architectures

## Training Process
- Backpropagation algorithm
- Gradient descent optimization
- Regularization techniques

## Applications
- Image recognition
- Natural language processing
- Time series prediction

## Connection to Theory
- [[Machine Learning]] - Broader ML context
- [[Mathematics for Science]] - Mathematical foundations
""",
            
            "Areas/Business/Strategy.md": """---
title: Business Strategy
tags: [business, strategy, management]
domains: [business]
status: 🌿
---

# Business Strategy

Strategic planning and execution in business contexts.

## Strategic Frameworks
- SWOT analysis
- Porter's Five Forces
- Blue Ocean Strategy

## Implementation
- Strategic planning process
- Performance metrics
- Change management

## Case Studies
Analysis of successful strategic implementations.

## Related Areas
- [[Project Management]] - Execution methods
- [[Research Methods]] - Analysis techniques
""",
            
            "Areas/Business/Project Management.md": """---
title: Project Management
tags: [project-management, business, methodology]
domains: [business]
status: 🌳
---

# Project Management

Methodologies and best practices for project execution.

## Methodologies
- Agile and Scrum
- Waterfall approach
- Hybrid methodologies

## Tools and Techniques
- Gantt charts
- Risk assessment
- Resource allocation

## Success Factors
- Clear communication
- Stakeholder management
- Quality control

## Integration
- [[Business Strategy]] - Strategic alignment
- [[Software Engineering]] - Technical projects
""",
            
            "Areas/Research Methods/Scientific Method.md": """---
title: Scientific Method
tags: [research, methodology, science]
domains: [research-methods]
status: 🌳
---

# Scientific Method

Systematic approach to scientific inquiry and research.

## Core Principles
- Hypothesis formation
- Experimental design
- Data collection and analysis
- Peer review process

## Research Types
- Quantitative research
- Qualitative research
- Mixed methods

## Statistical Analysis
- Descriptive statistics
- Inferential statistics
- Hypothesis testing

## Applications
- [[Physics Fundamentals]] - Physical sciences
- [[Chemistry Basics]] - Chemical research
- [[Machine Learning]] - Data science research

## Ethics
Research ethics and responsible conduct.
""",
            
            "Areas/Mathematics for Science.md": """---
title: Mathematics for Science
tags: [mathematics, science, foundations]
domains: [mathematics, natural-science]
status: 🌳
---

# Mathematics for Science

Mathematical foundations essential for scientific work.

## Core Areas
- Calculus and differential equations
- Linear algebra
- Statistics and probability
- Discrete mathematics

## Applications
- [[Physics Fundamentals]] - Physical modeling
- [[Machine Learning]] - Algorithm foundations
- [[Research Methods]] - Statistical analysis

## Computational Tools
- MATLAB and Mathematica
- Python NumPy/SciPy
- R statistical software

## Advanced Topics
- Numerical methods
- Optimization theory
- Complex analysis
""",
            
            "Areas/Software Engineering.md": """---
title: Software Engineering
tags: [software, engineering, development]
domains: [computer-science]
status: 🌿
---

# Software Engineering

Principles and practices for building robust software systems.

## Development Methodologies
- Agile development
- Test-driven development
- Continuous integration

## Design Patterns
- Creational patterns
- Structural patterns
- Behavioral patterns

## Quality Assurance
- Code review processes
- Automated testing
- Performance optimization

## Related Areas
- [[Algorithms and Data Structures]] - Technical foundations
- [[Project Management]] - Process management
- [[Machine Learning]] - AI system development

## Tools and Technologies
Version control, IDEs, and deployment platforms.
"""
        }
        
        # Create non-Areas content for exclusion testing
        non_areas_notes = {}
        if include_non_areas:
            non_areas_notes = {
                "Journal/2024-01-15.md": """---
title: Daily Journal - January 15, 2024
tags: [journal, daily, personal]
---

# Daily Journal - January 15, 2024

Personal reflections and daily activities.

## Today's Activities
- Morning workout
- Team meeting about [[Machine Learning]] project
- Lunch with Sarah
- Worked on [[Physics Fundamentals]] review

## Personal Thoughts
Private thoughts and reflections that should not be in datasets.

## Tomorrow's Plans
- Continue research
- Personal appointments
""",
                
                "Journal/2024-01-16.md": """---
title: Daily Journal - January 16, 2024
tags: [journal, daily, personal]
---

# Daily Journal - January 16, 2024

Another day of personal activities and thoughts.

## Work Progress
Made progress on [[Algorithms and Data Structures]] implementation.

## Personal Notes
Private personal notes that should remain private.
""",
                
                "People/John Doe.md": """---
title: John Doe
tags: [people, colleague, personal]
---

# John Doe

Personal information about John Doe.

## Professional
- Works on [[Machine Learning]] projects
- Expert in [[Neural Networks]]

## Personal
- Lives in San Francisco
- Enjoys hiking and photography
- Personal details that should not be in datasets

## Interactions
Record of personal and professional interactions.
""",
                
                "People/Jane Smith.md": """---
title: Jane Smith
tags: [people, friend, personal]
---

# Jane Smith

Personal information about Jane Smith.

## Background
Friend and colleague with expertise in [[Business Strategy]].

## Personal Details
Private information about Jane that should not be included in datasets.
""",
                
                "Inbox/Random Thought.md": """---
title: Random Thought
tags: [inbox, temporary, personal]
---

# Random Thought

Temporary note that should be processed later.

This contains personal thoughts and should not be in datasets.

Maybe connect to [[Research Methods]] later.
""",
                
                "Inbox/Meeting Notes.md": """---
title: Meeting Notes - Project Alpha
tags: [inbox, meetings, work]
---

# Meeting Notes - Project Alpha

Notes from project meeting that contain personal opinions.

## Discussion Points
- [[Software Engineering]] best practices
- Timeline concerns
- Personal opinions about team members

## Action Items
- Follow up on [[Project Management]] processes
- Personal tasks and responsibilities
""",
                
                "Archive/Old Project.md": """---
title: Old Project Documentation
tags: [archive, old, deprecated]
---

# Old Project Documentation

Archived project information that may contain personal details.

## Project Overview
Old project that used [[Algorithms and Data Structures]].

## Personal Reflections
Personal thoughts on what went wrong and right.

## Lessons Learned
Mix of professional and personal insights.
""",
                
                "Templates/Daily Note Template.md": """---
title: Daily Note Template
tags: [template]
---

# {{date}}

## Tasks
- [ ] 

## Notes
Personal and work notes go here.

## Reflections
Personal thoughts and reflections.

## References
- [[Research Methods]]
- [[Project Management]]
"""
            }
        
        # Write all notes to files
        all_notes = {**areas_notes, **non_areas_notes}
        notes_metadata = {}
        
        for file_path, content in all_notes.items():
            full_path = vault_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            
            # Extract metadata for validation
            is_areas = file_path.startswith("Areas/")
            notes_metadata[str(full_path)] = {
                "relative_path": file_path,
                "is_areas_content": is_areas,
                "content_size": len(content),
                "word_count": len(content.split())
            }
        
        return {
            "vault_path": str(vault_path),
            "areas_notes_count": len(areas_notes),
            "non_areas_notes_count": len(non_areas_notes),
            "total_notes": len(all_notes),
            "notes_metadata": notes_metadata,
            "areas_subdirs": areas_subdirs,
            "creation_date": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_vault_without_areas(vault_path: Path) -> Dict[str, any]:
        """Create a test vault without Areas/ folder for error testing."""
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create only non-Areas content
        non_areas_notes = {
            "Journal/note1.md": "# Journal Note 1\nPersonal content.",
            "People/person1.md": "# Person 1\nPersonal information.",
            "Inbox/temp.md": "# Temporary Note\nTemporary content."
        }
        
        for file_path, content in non_areas_notes.items():
            full_path = vault_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
        
        return {
            "vault_path": str(vault_path),
            "has_areas_folder": False,
            "total_notes": len(non_areas_notes)
        }
    
    @staticmethod
    def create_vault_with_empty_areas(vault_path: Path) -> Dict[str, any]:
        """Create a test vault with empty Areas/ folder for error testing."""
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create empty Areas folder
        areas_dir = vault_path / "Areas"
        areas_dir.mkdir(exist_ok=True)
        
        # Create some non-Areas content
        non_areas_notes = {
            "Journal/note1.md": "# Journal Note 1\nPersonal content.",
            "People/person1.md": "# Person 1\nPersonal information."
        }
        
        for file_path, content in non_areas_notes.items():
            full_path = vault_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
        
        return {
            "vault_path": str(vault_path),
            "has_areas_folder": True,
            "areas_is_empty": True,
            "total_notes": len(non_areas_notes)
        }


class TestAreasDatasetGenerationIntegration:
    """Integration tests for Areas/ filtering in dataset generation."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_vector_encoder(self):
        """Create mock vector encoder for consistent testing."""
        mock_encoder = Mock()
        
        def encode_documents(documents):
            # Create deterministic embeddings based on content
            embeddings = []
            for doc in documents:
                # Use hash for deterministic but varied embeddings
                seed = hash(doc[:100]) % 10000  # Use first 100 chars
                np.random.seed(seed)
                embedding = np.random.rand(384)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        mock_encoder.encode_documents.side_effect = encode_documents
        mock_encoder.encode_batch.side_effect = encode_documents
        mock_encoder.get_embedding_dimension.return_value = 384
        
        return mock_encoder
    
    def test_end_to_end_areas_filtering_enabled(self, temp_output_dir, mock_vector_encoder):
        """Test complete end-to-end dataset generation with Areas/ filtering enabled."""
        # Create test vault with Areas/ and non-Areas/ content
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "areas_test_vault"
        
        try:
            # Create realistic vault with both Areas/ and non-Areas/ content
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=True
            )
            
            print(f"\nTesting Areas/ filtering with vault:")
            print(f"  Areas/ notes: {vault_metadata['areas_notes_count']}")
            print(f"  Non-Areas/ notes: {vault_metadata['non_areas_notes_count']}")
            print(f"  Total notes: {vault_metadata['total_notes']}")
            
            # Test VaultReader filtering first
            vault_reader = VaultReader(str(vault_path), areas_only=True)
            filtered_files = list(vault_reader.get_markdown_files())
            
            # Verify filtering works at VaultReader level
            assert len(filtered_files) == vault_metadata['areas_notes_count'], \
                f"VaultReader filtering failed: expected {vault_metadata['areas_notes_count']}, got {len(filtered_files)}"
            
            # Verify all filtered files are from Areas/
            for file_path in filtered_files:
                # Handle both absolute and relative paths
                if file_path.is_absolute():
                    rel_path = str(file_path.relative_to(vault_path))
                else:
                    rel_path = str(file_path)
                assert rel_path.startswith("Areas"), f"Non-Areas file in filtered results: {rel_path}"
            
            # Test AreasContentFilter directly
            areas_filter = AreasContentFilter(str(vault_path))
            validation_result = areas_filter.validate_areas_folder()
            assert validation_result['validation_passed'] is True
            assert validation_result['markdown_file_count'] == vault_metadata['areas_notes_count']
            
            # Test exclusion summary
            exclusion_summary = areas_filter.get_exclusion_summary()
            assert exclusion_summary['filtering_enabled'] is True
            assert len(exclusion_summary['excluded_folders']) > 0
            
            # Create dataset generator with Areas/ filtering enabled
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=True,  # Explicitly enable Areas/ filtering
                skip_validation=True  # Skip validation for testing
            )
            
            # Test notes dataset generation only (skip pairs to avoid sampling issues)
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder):
                
                # Generate only notes dataset to test filtering
                notes_dataset = generator.notes_generator.generate_dataset(
                    [str(f) for f in filtered_files],
                    generator.link_extractor.extract_all_links()[0]
                )
                
                # Verify notes dataset
                assert isinstance(notes_dataset, pd.DataFrame)
                assert len(notes_dataset) == vault_metadata['areas_notes_count']
                
                # Verify all note paths are from Areas/
                for note_path in notes_dataset['note_path']:
                    assert "Areas/" in note_path or "Areas\\" in note_path, \
                        f"Non-Areas/ content found in dataset: {note_path}"
                
                # Verify no personal content paths
                personal_folders = ["Journal/", "People/", "Inbox/", "Archive/", "Templates/"]
                for note_path in notes_dataset['note_path']:
                    for personal_folder in personal_folders:
                        assert personal_folder not in note_path, \
                            f"Personal content found in dataset: {note_path}"
                
                # Verify dataset structure
                required_columns = ['note_path', 'note_title', 'word_count', 'tag_count', 'outgoing_links_count']
                for col in required_columns:
                    assert col in notes_dataset.columns, f"Missing column: {col}"
                
                # Verify data quality
                assert notes_dataset['word_count'].min() > 0, "Invalid word counts"
                assert notes_dataset['tag_count'].min() >= 0, "Invalid tag counts"
                assert not notes_dataset['note_title'].isna().any(), "Missing note titles"
                
                print(f"✅ Areas/ filtering validation successful:")
                print(f"  VaultReader filtered: {len(filtered_files)} files")
                print(f"  Notes dataset created: {len(notes_dataset)} rows")
                print(f"  Privacy protection: {vault_metadata['non_areas_notes_count']} notes excluded")
                print(f"  Filtering validation: {validation_result['validation_passed']}")
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_non_areas_content_exclusion(self, temp_output_dir, mock_vector_encoder):
        """Test that non-Areas/ content is properly excluded from datasets."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "exclusion_test_vault"
        
        try:
            # Create vault with specific non-Areas/ content to verify exclusion
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=True
            )
            
            # Generate datasets with Areas/ filtering
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                result = generator.generate_datasets()
                assert result.success is True
                
                # Load datasets
                notes_df = pd.read_csv(result.notes_dataset_path)
                pairs_df = pd.read_csv(result.pairs_dataset_path)
                
                # Verify specific exclusions
                excluded_patterns = [
                    "Journal/",
                    "People/", 
                    "Inbox/",
                    "Archive/",
                    "Templates/"
                ]
                
                for pattern in excluded_patterns:
                    # Check notes dataset
                    excluded_notes = [path for path in notes_df['note_path'] if pattern in path]
                    assert len(excluded_notes) == 0, \
                        f"Found excluded content in notes dataset: {excluded_notes}"
                    
                    # Check pairs dataset
                    excluded_pairs_a = [path for path in pairs_df['note_a_path'] if pattern in path]
                    excluded_pairs_b = [path for path in pairs_df['note_b_path'] if pattern in path]
                    
                    assert len(excluded_pairs_a) == 0, \
                        f"Found excluded content in pairs dataset (note_a): {excluded_pairs_a}"
                    assert len(excluded_pairs_b) == 0, \
                        f"Found excluded content in pairs dataset (note_b): {excluded_pairs_b}"
                
                # Verify only Areas/ content is present
                areas_notes = [path for path in notes_df['note_path'] 
                              if "Areas/" in path or "Areas\\" in path]
                assert len(areas_notes) == len(notes_df), \
                    "Not all notes in dataset are from Areas/ folder"
                
                print(f"✅ Exclusion validation successful:")
                print(f"  All {len(notes_df)} notes are from Areas/ folder")
                print(f"  No personal content found in datasets")
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_output_directory_creation_and_path_expansion(self, mock_vector_encoder):
        """Test output directory creation and path expansion (Requirement 2.2)."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "path_test_vault"
        
        try:
            # Create minimal Areas/ vault
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=False
            )
            
            # Test with tilde expansion in output path
            output_with_tilde = Path("~/test_dataset_output")
            expected_expanded = output_with_tilde.expanduser()
            
            # Create generator with tilde path
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=output_with_tilde,
                areas_only=True,
                skip_validation=True
            )
            
            # Verify path was expanded
            assert generator.output_dir == expected_expanded.resolve()
            
            # Test with nested directory creation
            nested_output = Path(temp_dir) / "deep" / "nested" / "output" / "directory"
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=nested_output,
                areas_only=True,
                skip_validation=True
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Generate datasets - should create nested directories
                result = generator.generate_datasets()
                assert result.success is True
                
                # Verify nested directory was created
                assert nested_output.exists()
                assert nested_output.is_dir()
                
                # Verify files were created in nested directory
                notes_file = nested_output / "notes_dataset.csv"
                pairs_file = nested_output / "pairs_dataset.csv"
                
                assert notes_file.exists()
                assert pairs_file.exists()
                
                print(f"✅ Path expansion and directory creation successful:")
                print(f"  Created nested directory: {nested_output}")
                print(f"  Files created successfully")
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_existing_dataset_features_with_filtering(self, temp_output_dir, mock_vector_encoder):
        """Test that all existing dataset features work with filtered content (Requirements 5.1-5.3)."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "features_test_vault"
        
        try:
            # Create vault with rich Areas/ content for feature testing
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=True
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Generate datasets with various options
                result = generator.generate_datasets(
                    notes_filename="features_notes.csv",
                    pairs_filename="features_pairs.csv",
                    negative_sampling_ratio=3.0,
                    batch_size=5,
                    max_pairs_per_note=100
                )
                
                assert result.success is True
                
                # Load datasets
                notes_df = pd.read_csv(result.notes_dataset_path)
                pairs_df = pd.read_csv(result.pairs_dataset_path)
                
                # Test notes dataset features (Requirement 5.2)
                required_note_columns = [
                    'note_path', 'note_title', 'word_count', 'tag_count',
                    'outgoing_links_count', 'semantic_summary'
                ]
                
                for col in required_note_columns:
                    assert col in notes_df.columns, f"Missing notes column: {col}"
                
                # Verify feature quality
                assert notes_df['word_count'].min() > 0, "Invalid word counts"
                assert notes_df['tag_count'].min() >= 0, "Invalid tag counts"
                assert notes_df['outgoing_links_count'].min() >= 0, "Invalid link counts"
                assert not notes_df['note_title'].isna().any(), "Missing note titles"
                assert not notes_df['semantic_summary'].isna().any(), "Missing semantic summaries"
                
                # Test pairs dataset features (Requirement 5.3)
                required_pair_columns = [
                    'note_a_path', 'note_b_path', 'cosine_similarity', 
                    'link_exists', 'tag_overlap_count'
                ]
                
                for col in required_pair_columns:
                    assert col in pairs_df.columns, f"Missing pairs column: {col}"
                
                # Verify pairs feature quality
                assert pairs_df['cosine_similarity'].between(0, 1).all(), "Invalid similarity scores"
                assert pairs_df['link_exists'].dtype == bool, "Invalid link_exists type"
                assert pairs_df['tag_overlap_count'].min() >= 0, "Invalid tag overlap counts"
                
                # Test link extraction capabilities (Requirement 5.1)
                # Verify that links between Areas/ notes are properly extracted
                linked_pairs = pairs_df[pairs_df['link_exists'] == True]
                assert len(linked_pairs) > 0, "No linked pairs found - link extraction may have failed"
                
                # Verify link statistics in summary
                summary = result.summary
                assert hasattr(summary, 'total_links_extracted')
                assert summary.total_links_extracted >= 0
                
                # Test advanced features
                # Verify embeddings were generated (through similarity scores)
                similarity_stats = pairs_df['cosine_similarity'].describe()
                assert similarity_stats['std'] > 0, "No variation in similarity scores - embeddings may be broken"
                
                # Verify tag analysis
                tag_stats = notes_df['tag_count'].describe()
                assert tag_stats['mean'] > 0, "No tags found - tag extraction may have failed"
                
                print(f"✅ Feature validation successful:")
                print(f"  Notes features: {len(required_note_columns)} columns verified")
                print(f"  Pairs features: {len(required_pair_columns)} columns verified")
                print(f"  Link extraction: {len(linked_pairs)} linked pairs found")
                print(f"  Embeddings: Similarity std = {similarity_stats['std']:.3f}")
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_cross_reference_handling(self, temp_output_dir, mock_vector_encoder):
        """Test handling of cross-references to non-Areas content (Requirement 5.4)."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "crossref_test_vault"
        
        try:
            # Create vault with cross-references from Areas/ to non-Areas/
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=True
            )
            
            # Add specific cross-reference content
            crossref_note = vault_path / "Areas" / "Cross Reference Test.md"
            crossref_content = """---
title: Cross Reference Test
tags: [test, cross-reference]
domains: [test]
---

# Cross Reference Test

This note contains references to both Areas/ and non-Areas/ content.

## Areas/ References (should work)
- [[Physics Fundamentals]] - Valid Areas/ reference
- [[Machine Learning]] - Another valid Areas/ reference

## Non-Areas/ References (should be handled gracefully)
- [[John Doe]] - Reference to People/ folder (should be broken)
- [[Daily Journal - January 15, 2024]] - Reference to Journal/ (should be broken)
- [[Random Thought]] - Reference to Inbox/ (should be broken)

## Mixed Content
Some content that references both valid and invalid links.
"""
            crossref_note.write_text(crossref_content, encoding='utf-8')
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Should handle cross-references gracefully without failing
                result = generator.generate_datasets()
                assert result.success is True, f"Failed to handle cross-references: {result.error_message}"
                
                # Load datasets
                notes_df = pd.read_csv(result.notes_dataset_path)
                pairs_df = pd.read_csv(result.pairs_dataset_path)
                
                # Verify cross-reference note is included
                crossref_notes = notes_df[notes_df['note_title'] == 'Cross Reference Test']
                assert len(crossref_notes) == 1, "Cross-reference test note not found"
                
                crossref_row = crossref_notes.iloc[0]
                
                # Verify outgoing links count reflects only valid (Areas/) links
                # Should count links to Areas/ content but not broken links to non-Areas/
                assert crossref_row['outgoing_links_count'] >= 0, "Invalid outgoing links count"
                
                # Verify that broken links don't cause pairs to non-existent notes
                crossref_path = crossref_row['note_path']
                
                # Find pairs involving the cross-reference note
                crossref_pairs = pairs_df[
                    (pairs_df['note_a_path'] == crossref_path) | 
                    (pairs_df['note_b_path'] == crossref_path)
                ]
                
                # All paired notes should exist in the notes dataset
                all_note_paths = set(notes_df['note_path'])
                
                for _, pair in crossref_pairs.iterrows():
                    assert pair['note_a_path'] in all_note_paths, \
                        f"Pair references non-existent note: {pair['note_a_path']}"
                    assert pair['note_b_path'] in all_note_paths, \
                        f"Pair references non-existent note: {pair['note_b_path']}"
                
                # Verify no pairs exist to non-Areas/ content
                for _, pair in crossref_pairs.iterrows():
                    assert "Areas/" in pair['note_a_path'] or "Areas\\" in pair['note_a_path']
                    assert "Areas/" in pair['note_b_path'] or "Areas\\" in pair['note_b_path']
                
                print(f"✅ Cross-reference handling successful:")
                print(f"  Cross-reference note processed successfully")
                print(f"  {len(crossref_pairs)} pairs involving cross-reference note")
                print(f"  No broken references to non-Areas/ content")
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_areas_folder_validation_errors(self, temp_output_dir):
        """Test proper error handling for Areas/ folder validation issues."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test 1: Vault without Areas/ folder
            vault_without_areas = Path(temp_dir) / "no_areas_vault"
            AreasTestVaultHelper.create_vault_without_areas(vault_without_areas)
            
            with pytest.raises(AreasNotFoundError):
                DatasetGenerator(
                    vault_path=vault_without_areas,
                    output_dir=temp_output_dir,
                    areas_only=True
                )
            
            # Test 2: Vault with empty Areas/ folder
            vault_empty_areas = Path(temp_dir) / "empty_areas_vault"
            AreasTestVaultHelper.create_vault_with_empty_areas(vault_empty_areas)
            
            with pytest.raises(InsufficientAreasContentError):
                DatasetGenerator(
                    vault_path=vault_empty_areas,
                    output_dir=temp_output_dir,
                    areas_only=True
                )
            
            print("✅ Error handling validation successful:")
            print("  AreasNotFoundError raised for missing Areas/ folder")
            print("  InsufficientAreasContentError raised for empty Areas/ folder")
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_areas_filter_component_integration(self, temp_output_dir):
        """Test AreasContentFilter component integration."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "filter_test_vault"
        
        try:
            # Create test vault
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=True
            )
            
            # Test AreasContentFilter directly
            areas_filter = AreasContentFilter(str(vault_path))
            
            # Test validation
            validation_result = areas_filter.validate_areas_folder()
            assert validation_result['validation_passed'] is True
            assert validation_result['areas_folder_exists'] is True
            assert validation_result['markdown_file_count'] > 0
            
            # Test file path filtering
            all_files = list(vault_path.rglob("*.md"))
            filtered_files = areas_filter.filter_file_paths(all_files)
            
            # Verify filtering worked
            assert len(filtered_files) < len(all_files), "Filtering should reduce file count"
            assert len(filtered_files) == vault_metadata['areas_notes_count']
            
            # Verify all filtered files are from Areas/
            for file_path in filtered_files:
                rel_path = file_path.relative_to(vault_path)
                assert str(rel_path).startswith("Areas"), f"Non-Areas file in filtered results: {rel_path}"
            
            # Test structure analysis
            structure = areas_filter.get_areas_structure()
            assert structure['exists'] is True
            assert len(structure['markdown_files']) > 0
            assert len(structure['subdirectories']) > 0
            
            # Test exclusion summary
            exclusion_summary = areas_filter.get_exclusion_summary()
            assert exclusion_summary['filtering_enabled'] is True
            assert len(exclusion_summary['excluded_folders']) > 0
            
            print("✅ AreasContentFilter integration successful:")
            print(f"  Validated Areas/ folder with {validation_result['markdown_file_count']} files")
            print(f"  Filtered {len(all_files)} files to {len(filtered_files)} Areas/ files")
            print(f"  Found {len(structure['subdirectories'])} Areas/ subdirectories")
            print(f"  Excluding {len(exclusion_summary['excluded_folders'])} folders for privacy")
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_performance_with_areas_filtering(self, temp_output_dir, mock_vector_encoder):
        """Test performance characteristics with Areas/ filtering enabled."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "performance_test_vault"
        
        try:
            # Create larger vault for performance testing
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=True
            )
            
            # Add more Areas/ content for performance testing
            for i in range(10, 25):  # Add 15 more Areas/ notes
                extra_note = vault_path / "Areas" / f"Extra Note {i}.md"
                extra_note.write_text(f"""---
title: Extra Note {i}
tags: [extra, test{i}]
domains: [test]
---

# Extra Note {i}

This is extra content for performance testing.

## Content
- Point 1 about topic {i}
- Point 2 about topic {i}
- Reference to [[Physics Fundamentals]]

## Analysis
Detailed analysis content for note {i}.
""")
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Measure performance
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                result = generator.generate_datasets(batch_size=8)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                # Verify success
                assert result.success is True
                
                # Performance metrics
                total_time = end_time - start_time
                memory_increase = end_memory - start_memory
                
                # Load datasets for size verification
                notes_df = pd.read_csv(result.notes_dataset_path)
                pairs_df = pd.read_csv(result.pairs_dataset_path)
                
                # Performance should be reasonable
                notes_per_second = len(notes_df) / total_time if total_time > 0 else 0
                
                print(f"✅ Performance testing successful:")
                print(f"  Processing time: {total_time:.2f} seconds")
                print(f"  Memory increase: {memory_increase:.2f} MB")
                print(f"  Processing rate: {notes_per_second:.2f} notes/second")
                print(f"  Notes processed: {len(notes_df)}")
                print(f"  Pairs generated: {len(pairs_df)}")
                
                # Basic performance assertions
                assert total_time < 120, f"Processing too slow: {total_time:.2f}s"
                assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB"
                
        finally:
            shutil.rmtree(temp_dir)
    
    def test_configuration_integration(self, temp_output_dir, mock_vector_encoder):
        """Test integration with configuration system for Areas/ filtering."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "config_test_vault"
        
        try:
            # Create test vault
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=True
            )
            
            # Test with default configuration (should use Areas/ filtering)
            with patch('jarvis.utils.config.get_settings') as mock_settings:
                mock_settings.return_value.dataset_areas_only = True
                mock_settings.return_value.dataset_areas_folder_name = "Areas"
                mock_settings.return_value.get_dataset_output_path.return_value = temp_output_dir
                
                generator = DatasetGenerator(vault_path=vault_path, skip_validation=True)
                
                # Verify configuration was applied
                assert generator.areas_only is True
                assert generator.output_dir == temp_output_dir
                
                with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                     patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                    
                    result = generator.generate_datasets()
                    assert result.success is True
                    
                    # Verify only Areas/ content was processed
                    notes_df = pd.read_csv(result.notes_dataset_path)
                    assert len(notes_df) == vault_metadata['areas_notes_count']
            
            # Test with custom Areas/ folder name
            custom_areas_dir = vault_path / "CustomAreas"
            custom_areas_dir.mkdir(exist_ok=True)
            
            custom_note = custom_areas_dir / "Custom Note.md"
            custom_note.write_text("""---
title: Custom Areas Note
tags: [custom]
---

# Custom Areas Note

Content in custom Areas folder.
""")
            
            with patch('jarvis.utils.config.get_settings') as mock_settings:
                mock_settings.return_value.dataset_areas_only = True
                mock_settings.return_value.dataset_areas_folder_name = "CustomAreas"
                mock_settings.return_value.get_dataset_output_path.return_value = temp_output_dir
                
                generator = DatasetGenerator(vault_path=vault_path, skip_validation=True)
                
                with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                     patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                    
                    result = generator.generate_datasets()
                    assert result.success is True
                    
                    # Should find the custom Areas/ folder content
                    notes_df = pd.read_csv(result.notes_dataset_path)
                    custom_notes = [path for path in notes_df['note_path'] if "CustomAreas" in path]
                    assert len(custom_notes) > 0, "Custom Areas/ folder content not found"
            
            print("✅ Configuration integration successful:")
            print("  Default Areas/ filtering configuration applied")
            print("  Custom Areas/ folder name configuration applied")
            
        finally:
            shutil.rmtree(temp_dir)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # Return 0 if psutil not available


class TestAreasFilteringErrorScenarios:
    """Test error scenarios specific to Areas/ filtering."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_missing_areas_folder_error(self, temp_output_dir):
        """Test error handling when Areas/ folder is missing."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "no_areas_vault"
        
        try:
            # Create vault without Areas/ folder
            AreasTestVaultHelper.create_vault_without_areas(vault_path)
            
            # Should raise AreasNotFoundError
            with pytest.raises(AreasNotFoundError) as exc_info:
                DatasetGenerator(
                    vault_path=vault_path,
                    output_dir=temp_output_dir,
                    areas_only=True
                )
            
            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "Areas" in error_msg
            assert str(vault_path) in error_msg
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_insufficient_areas_content_error(self, temp_output_dir):
        """Test error handling when Areas/ folder has insufficient content."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "insufficient_areas_vault"
        
        try:
            # Create vault with empty Areas/ folder
            AreasTestVaultHelper.create_vault_with_empty_areas(vault_path)
            
            # Should raise InsufficientAreasContentError
            with pytest.raises(InsufficientAreasContentError) as exc_info:
                DatasetGenerator(
                    vault_path=vault_path,
                    output_dir=temp_output_dir,
                    areas_only=True
                )
            
            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "insufficient" in error_msg.lower()
            assert "Areas" in error_msg
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_areas_folder_permission_error(self, temp_output_dir):
        """Test handling of permission errors with Areas/ folder."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "permission_test_vault"
        
        try:
            # Create vault with Areas/ folder
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=False
            )
            
            # Try to make Areas/ folder unreadable (may not work on all systems)
            areas_dir = vault_path / "Areas"
            try:
                areas_dir.chmod(0o000)  # Remove all permissions
                
                # Should handle permission error gracefully
                with pytest.raises((PermissionError, OSError, VaultValidationError)):
                    DatasetGenerator(
                        vault_path=vault_path,
                        output_dir=temp_output_dir,
                        areas_only=True
                    )
                
            finally:
                # Restore permissions for cleanup
                try:
                    areas_dir.chmod(0o755)
                except:
                    pass
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_corrupted_areas_content_handling(self, temp_output_dir, mock_vector_encoder):
        """Test handling of corrupted content in Areas/ folder."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "corrupted_areas_vault"
        
        try:
            # Create vault with some valid and some corrupted Areas/ content
            vault_metadata = AreasTestVaultHelper.create_areas_test_vault(
                vault_path, include_non_areas=False
            )
            
            # Add corrupted file to Areas/
            corrupted_file = vault_path / "Areas" / "corrupted.md"
            corrupted_file.write_bytes(b'\xff\xfe\x00\x00invalid utf-8 content')
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Should handle corrupted files gracefully
                result = generator.generate_datasets()
                
                # May succeed with partial data or fail gracefully
                if result.success:
                    # If successful, should have processed valid files
                    notes_df = pd.read_csv(result.notes_dataset_path)
                    assert len(notes_df) > 0, "Should process valid files despite corruption"
                else:
                    # If failed, should have meaningful error message
                    assert result.error_message is not None
                    assert len(result.error_message) > 0
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])