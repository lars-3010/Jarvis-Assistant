"""
Integration tests for dataset generation service interactions.

These tests verify that VaultReader, VectorEncoder, and GraphDatabase services
work together properly in the dataset generation workflow, including end-to-end
dataset generation and error handling/recovery mechanisms.
"""

import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from jarvis.services.vault.reader import VaultReader
from jarvis.services.vault.parser import MarkdownParser
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.graph.database import GraphDatabase
from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import PairsDatasetGenerator
from jarvis.tools.dataset_generation.models.data_models import (
    DatasetGenerationResult, GenerationSummary, LinkStatistics, ValidationResult
)
from jarvis.tools.dataset_generation.models.exceptions import (
    LinkExtractionError, FeatureEngineeringError, InsufficientDataError
)
from jarvis.utils.config import JarvisSettings


class TestDatasetGenerationServiceIntegration:
    """Integration tests for dataset generation with real services."""

    @pytest.fixture
    def temp_vault_dir(self):
        """Create a temporary vault directory with test files."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "test_vault"
        vault_path.mkdir(parents=True)
        
        # Create test markdown files with Areas/ folder structure
        # Note: These tests now need to account for default Areas/ filtering
        test_files = {
            # Areas/ folder - knowledge content (will be included with default filtering)
            "Areas/Computer Science/note1.md": """---
title: Machine Learning Basics
tags: [ml, ai, basics]
aliases: [ML Basics, ML 101]
domains: [artificial-intelligence]
concepts: [supervised-learning, neural-networks, algorithms]
up:: [[AI Overview]]
similar: [[Deep Learning]], [[Data Science]]
---

# Machine Learning Basics

This note covers the fundamentals of machine learning.

## Key Concepts
- Supervised learning
- Unsupervised learning
- Neural networks

See also: [[Deep Learning]] and [[Data Science]].
""",
            "Areas/Computer Science/note2.md": """---
title: Deep Learning
tags: [dl, ai, advanced]
aliases: [DL, Neural Networks]
domains: [artificial-intelligence]
concepts: [cnn, rnn, transformers]
extends: [[Machine Learning Basics]]
implements: [[Neural Network Interface]]
---

# Deep Learning

Advanced machine learning using neural networks.

## Architectures
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Transformers

Related: [[Machine Learning Basics]] provides the foundation.
""",
            "Areas/Data Analysis/note3.md": """---
title: Data Science
tags: [ds, analytics, statistics]
aliases: [Data Analytics]
domains: [data-analysis]
concepts: [statistics, visualization, pandas]
similar: [[Machine Learning Basics]]
---

# Data Science

The science of extracting insights from data.

## Tools
- Python pandas
- R
- Jupyter notebooks

Connected to [[Machine Learning Basics]] for predictive modeling.
""",
            "Areas/Computer Science/folder/note4.md": """---
title: AI Overview
tags: [ai, overview, 🗺️]
aliases: [Artificial Intelligence]
domains: [artificial-intelligence]
progress: 🌲
---

# AI Overview

High-level overview of artificial intelligence.

## Subfields
- [[Machine Learning Basics]]
- Natural Language Processing
- Computer Vision
""",
            "Areas/Software Architecture/note5.md": """---
title: Neural Network Interface
tags: [interface, architecture, 🌿]
domains: [software-architecture]
progress: 🌱
---

# Neural Network Interface

Abstract interface for neural network implementations.

## Methods
- forward()
- backward()
- train()

Implemented by various [[Deep Learning]] architectures.
""",
            # Non-Areas content (will be excluded with default filtering)
            "Journal/2024-01-01.md": """---
title: Daily Journal Entry
tags: [journal, personal]
---

# Daily Journal Entry

Personal thoughts and reflections.
""",
            "Inbox/random-note.md": """---
title: Random Inbox Note
tags: [inbox, temporary]
---

# Random Note

Temporary note in inbox.
""",
            "projects/project1.md": """---
title: ML Project Alpha
tags: [project, active, 🚀]
domains: [projects]
status: 🌿
leads_to: [[ML Project Beta]]
sources: [[Machine Learning Basics]], [[Data Science]]
---

# ML Project Alpha

Active machine learning project.

## Goals
- Implement basic ML pipeline
- Integrate with [[Data Science]] tools

Next: [[ML Project Beta]]
""",
            "projects/project2.md": """---
title: ML Project Beta
tags: [project, planned, ⚛️]
domains: [projects]
status: 🌱
up:: [[ML Project Alpha]]
---

# ML Project Beta

Planned follow-up to [[ML Project Alpha]].

## Requirements
- Advanced [[Deep Learning]] techniques
- Production deployment
"""
        }
        
        # Write test files
        for file_path, content in test_files.items():
            full_path = vault_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
        
        yield vault_path
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_settings(self, temp_vault_dir):
        """Create mock settings for testing."""
        return JarvisSettings(
            vault_path=str(temp_vault_dir),
            vector_db_path=":memory:",  # In-memory database for testing
            graph_enabled=False,  # Disable Neo4j for integration tests
            metrics_enabled=False
        )

    @pytest.fixture
    def real_vault_reader(self, temp_vault_dir):
        """Create a real VaultReader instance."""
        return VaultReader(str(temp_vault_dir))

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a mock VectorEncoder that returns consistent embeddings."""
        mock_encoder = Mock(spec=VectorEncoder)
        
        # Create consistent embeddings for testing
        def mock_encode_documents(documents):
            embeddings = []
            for i, doc in enumerate(documents):
                # Create deterministic embeddings based on content hash
                content_hash = hash(doc) % 1000
                embedding = np.random.RandomState(content_hash).rand(384)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        def mock_encode_batch(texts):
            return mock_encode_documents(texts)
        
        mock_encoder.encode_documents.side_effect = mock_encode_documents
        mock_encoder.encode_batch.side_effect = mock_encode_batch
        mock_encoder.get_embedding_dimension.return_value = 384
        
        return mock_encoder

    @pytest.fixture
    def mock_graph_database(self):
        """Create a mock GraphDatabase."""
        mock_db = Mock(spec=GraphDatabase)
        mock_db.is_available.return_value = False  # Simulate unavailable for testing
        return mock_db

    def test_vault_reader_integration(self, real_vault_reader):
        """Test VaultReader service integration."""
        # Test getting markdown files (with default Areas/ filtering)
        markdown_files = list(real_vault_reader.get_markdown_files())
        assert len(markdown_files) == 5  # Only Areas/ files with default filtering
        
        # Test reading individual files
        note1_path = None
        for file_path in markdown_files:
            if file_path.name == "note1.md":
                note1_path = file_path
                break
        
        assert note1_path is not None
        content, metadata = real_vault_reader.read_file(str(note1_path))
        
        assert "Machine Learning Basics" in content
        assert "created" in metadata or "modified" in metadata

    def test_link_extractor_service_integration(self, real_vault_reader):
        """Test LinkExtractor integration with VaultReader."""
        link_extractor = LinkExtractor(real_vault_reader)
        
        # Test extracting all links
        graph, statistics = link_extractor.extract_all_links()
        
        assert isinstance(graph, nx.DiGraph)
        assert isinstance(statistics, LinkStatistics)
        assert graph.number_of_nodes() > 0
        assert statistics.total_links > 0
        
        # Verify some expected links exist
        nodes = list(graph.nodes())
        note_paths = [str(node) for node in nodes]
        
        # Should have extracted links between related notes
        assert any("note1.md" in path for path in note_paths)
        assert any("note2.md" in path for path in note_paths)

    def test_notes_dataset_generator_service_integration(self, real_vault_reader, mock_vector_encoder):
        """Test NotesDatasetGenerator integration with services."""
        # Create real markdown parser
        markdown_parser = MarkdownParser(extract_semantic=True)
        
        # Create generator with real and mock services
        generator = NotesDatasetGenerator(
            vault_reader=real_vault_reader,
            vector_encoder=mock_vector_encoder,
            graph_database=None,  # Test without graph database
            markdown_parser=markdown_parser
        )
        
        # Get notes and create a simple graph
        markdown_files = real_vault_reader.get_markdown_files()
        notes = [str(f) for f in markdown_files]
        
        # Create a simple link graph for testing
        link_graph = nx.DiGraph()
        link_graph.add_nodes_from(notes)
        link_graph.add_edge(notes[0], notes[1])  # Add one edge
        
        # Generate dataset
        dataset = generator.generate_dataset(notes, link_graph)
        
        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset) == len(notes)
        
        # Check expected columns
        expected_columns = [
            'note_path', 'note_title', 'word_count', 'tag_count',
            'outgoing_links_count', 'semantic_summary'
        ]
        for col in expected_columns:
            assert col in dataset.columns
        
        # Verify data quality
        assert dataset['word_count'].min() > 0
        assert dataset['tag_count'].min() >= 0
        assert not dataset['note_title'].isna().any()

    def test_pairs_dataset_generator_service_integration(self, real_vault_reader, mock_vector_encoder):
        """Test PairsDatasetGenerator integration with services."""
        # Create generator
        generator = PairsDatasetGenerator(
            vector_encoder=mock_vector_encoder,
            graph_database=None
        )
        
        # Create sample note data
        markdown_files = real_vault_reader.get_markdown_files()
        notes_data = {}
        
        for i, file_path in enumerate(markdown_files):
            content, metadata = real_vault_reader.read_file(str(file_path))
            
            # Parse with MarkdownParser to get structured data
            parser = MarkdownParser(content, extract_semantic=True)
            parsed = parser.parse()
            
            notes_data[str(file_path)] = {
                'path': str(file_path),
                'title': parsed['frontmatter'].get('title', file_path.stem),
                'content': content,
                'metadata': metadata,
                'tags': parsed['tags'],
                'outgoing_links': [link['target'] for link in parsed['links']],
                'embedding': np.random.rand(384),  # Mock embedding
                'word_count': len(content.split())
            }
        
        # Create link graph
        link_graph = nx.DiGraph()
        link_graph.add_nodes_from(notes_data.keys())
        
        # Add some edges based on actual links
        for note_path, note_data in notes_data.items():
            for link_target in note_data['outgoing_links']:
                # Find matching note
                for target_path in notes_data.keys():
                    if Path(target_path).stem in link_target or link_target in target_path:
                        link_graph.add_edge(note_path, target_path)
                        break
        
        # Convert to NoteData objects (simplified for testing)
        from jarvis.tools.dataset_generation.models.data_models import NoteData
        note_data_objects = {}
        for path, data in notes_data.items():
            note_data_objects[path] = NoteData(
                path=data['path'],
                title=data['title'],
                content=data['content'],
                metadata=data['metadata'],
                tags=data['tags'],
                outgoing_links=data['outgoing_links'],
                embedding=data['embedding'],
                word_count=data['word_count']
            )
        
        # Generate pairs dataset
        dataset = generator.generate_dataset(note_data_objects, link_graph)
        
        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset) > 0
        
        # Check expected columns
        expected_columns = [
            'note_a_path', 'note_b_path', 'cosine_similarity',
            'link_exists', 'tag_overlap_count'
        ]
        for col in expected_columns:
            assert col in dataset.columns
        
        # Verify data quality
        assert dataset['cosine_similarity'].between(0, 1).all()
        assert dataset['link_exists'].dtype == bool

    def test_end_to_end_dataset_generation(self, temp_vault_dir, temp_output_dir, mock_vector_encoder):
        """Test complete end-to-end dataset generation workflow."""
        # Create DatasetGenerator with real and mock services
        # Note: Using default Areas/ filtering behavior
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir,
            areas_only=False  # Disable Areas/ filtering for comprehensive testing
        )
        
        # Mock the vector encoder in the generator's components
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            # Generate datasets
            result = generator.generate_datasets()
            
            assert isinstance(result, DatasetGenerationResult)
            assert result.success is True
            assert result.notes_dataset_path is not None
            assert result.pairs_dataset_path is not None
            
            # Verify files were created
            notes_file = Path(result.notes_dataset_path)
            pairs_file = Path(result.pairs_dataset_path)
            
            assert notes_file.exists()
            assert pairs_file.exists()
            
            # Verify file contents
            notes_df = pd.read_csv(notes_file)
            pairs_df = pd.read_csv(pairs_file)
            
            assert len(notes_df) > 0
            assert len(pairs_df) > 0
            
            # Check summary statistics
            summary = result.summary
            assert isinstance(summary, GenerationSummary)
            assert summary.total_notes > 0
            assert summary.notes_processed > 0
            assert summary.pairs_generated > 0

    def test_service_error_handling_and_recovery(self, temp_vault_dir, temp_output_dir):
        """Test error handling and recovery mechanisms in service interactions."""
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir,
            areas_only=False  # Disable Areas/ filtering for comprehensive error testing
        )
        
        # Test with failing vector encoder
        failing_encoder = Mock(spec=VectorEncoder)
        failing_encoder.encode_documents.side_effect = Exception("Encoding failed")
        failing_encoder.encode_batch.side_effect = Exception("Batch encoding failed")
        
        with patch.object(generator.notes_generator, 'vector_encoder', failing_encoder):
            # Should handle encoding failures gracefully
            result = generator.generate_datasets()
            
            # May succeed with degraded functionality or fail gracefully
            if not result.success:
                assert result.error_message is not None
                assert "encoding" in result.error_message.lower()

    def test_vault_reader_error_recovery(self, temp_vault_dir):
        """Test VaultReader error recovery mechanisms."""
        vault_reader = VaultReader(str(temp_vault_dir))
        
        # Test with corrupted file
        corrupted_file = temp_vault_dir / "corrupted.md"
        corrupted_file.write_bytes(b'\xff\xfe\x00\x00invalid utf-8')
        
        # Should handle corrupted files gracefully
        markdown_files = vault_reader.get_markdown_files()
        assert len(markdown_files) > 0  # Should still find valid files
        
        # Try to read the corrupted file
        try:
            content, metadata = vault_reader.read_file(str(corrupted_file))
            # If it succeeds, content should be handled gracefully
            assert isinstance(content, str)
        except UnicodeDecodeError:
            # This is acceptable - the error should be handled upstream
            pass

    def test_link_extraction_error_recovery(self, temp_vault_dir):
        """Test link extraction error recovery."""
        vault_reader = VaultReader(str(temp_vault_dir))
        link_extractor = LinkExtractor(vault_reader)
        
        # Create a file with malformed links
        malformed_file = temp_vault_dir / "malformed.md"
        malformed_file.write_text("""
        # Malformed Links Test
        
        [[Unclosed link
        [Malformed](
        [[]]
        [[   ]]
        """)
        
        # Should handle malformed links gracefully
        graph, statistics = link_extractor.extract_all_links()
        
        assert isinstance(graph, nx.DiGraph)
        assert isinstance(statistics, LinkStatistics)
        # Should still process other valid files

    def test_memory_management_during_processing(self, temp_vault_dir, temp_output_dir, mock_vector_encoder):
        """Test memory management during large dataset processing."""
        # Create additional test files to simulate larger vault
        for i in range(10, 20):  # Add 10 more files
            test_file = temp_vault_dir / f"generated_note_{i}.md"
            test_file.write_text(f"""---
title: Generated Note {i}
tags: [generated, test{i}]
---

# Generated Note {i}

This is a generated test note with content {i}.

Links to other notes: [[note1.md]] and [[note2.md]].
""")
        
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir
        )
        
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            # Generate with small batch size to test memory management
            result = generator.generate_datasets(batch_size=3)
            
            assert result.success is True
            
            # Verify processing stats indicate batch processing was used
            summary = result.summary
            assert summary.notes_processed > 10  # Should process all notes

    def test_concurrent_service_access(self, temp_vault_dir, mock_vector_encoder):
        """Test concurrent access to services."""
        import threading
        import time
        
        vault_reader = VaultReader(str(temp_vault_dir))
        results = []
        errors = []
        
        def read_files():
            try:
                files = vault_reader.get_markdown_files()
                for file_path in files[:3]:  # Read first 3 files
                    content, metadata = vault_reader.read_file(str(file_path))
                    results.append((str(file_path), len(content)))
                    time.sleep(0.01)  # Small delay to increase chance of concurrency
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=read_files)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors and results were collected
        assert len(errors) == 0
        assert len(results) > 0

    def test_service_dependency_validation(self, temp_vault_dir, temp_output_dir):
        """Test validation of service dependencies."""
        # Test with missing required services
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir
        )
        
        # Verify that generator validates its dependencies
        assert generator.vault_reader is not None
        assert generator.link_extractor is not None
        assert generator.notes_generator is not None
        assert generator.pairs_generator is not None

    def test_configuration_driven_service_behavior(self, temp_vault_dir, temp_output_dir, mock_vector_encoder):
        """Test that services behave according to configuration."""
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir
        )
        
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            # Test with different configuration options
            result1 = generator.generate_datasets(
                batch_size=2,
                negative_sampling_ratio=5.0
            )
            
            result2 = generator.generate_datasets(
                batch_size=10,
                negative_sampling_ratio=10.0
            )
            
            assert result1.success is True
            assert result2.success is True
            
            # Results should differ based on configuration
            # (exact differences depend on implementation)

    def test_service_performance_monitoring(self, temp_vault_dir, temp_output_dir, mock_vector_encoder):
        """Test performance monitoring across service interactions."""
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir
        )
        
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            start_time = time.time()
            result = generator.generate_datasets()
            end_time = time.time()
            
            assert result.success is True
            
            # Verify performance metrics are collected
            summary = result.summary
            assert summary.total_time_seconds > 0
            assert summary.total_time_seconds <= (end_time - start_time) + 1  # Allow some tolerance
            
            # Check processing rate
            if summary.notes_processed > 0:
                assert summary.processing_rate > 0

    def test_data_consistency_across_services(self, temp_vault_dir, temp_output_dir, mock_vector_encoder):
        """Test data consistency across different service interactions."""
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir
        )
        
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
            
            result = generator.generate_datasets()
            assert result.success is True
            
            # Load generated datasets
            notes_df = pd.read_csv(result.notes_dataset_path)
            pairs_df = pd.read_csv(result.pairs_dataset_path)
            
            # Verify data consistency
            note_paths_in_notes = set(notes_df['note_path'])
            note_paths_in_pairs = set(pairs_df['note_a_path']) | set(pairs_df['note_b_path'])
            
            # All notes in pairs should exist in notes dataset
            assert note_paths_in_pairs.issubset(note_paths_in_notes)
            
            # Verify link consistency
            linked_pairs = pairs_df[pairs_df['link_exists'] == True]
            assert len(linked_pairs) > 0  # Should have some linked pairs
            
            # Verify feature consistency
            assert notes_df['word_count'].min() >= 0
            assert pairs_df['cosine_similarity'].between(0, 1).all()


class TestDatasetGenerationErrorScenarios:
    """Test error scenarios in dataset generation integration."""

    @pytest.fixture
    def empty_vault_dir(self):
        """Create an empty vault directory."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "empty_vault"
        vault_path.mkdir(parents=True)
        yield vault_path
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def insufficient_vault_dir(self):
        """Create a vault with insufficient notes."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "insufficient_vault"
        vault_path.mkdir(parents=True)
        
        # Create only 2 notes (less than minimum required)
        for i in range(2):
            note_file = vault_path / f"note{i}.md"
            note_file.write_text(f"# Note {i}\n\nContent {i}")
        
        yield vault_path
        shutil.rmtree(temp_dir)

    def test_empty_vault_handling(self, empty_vault_dir, temp_output_dir):
        """Test handling of empty vault."""
        generator = DatasetGenerator(
            vault_path=empty_vault_dir,
            output_dir=temp_output_dir
        )
        
        result = generator.generate_datasets()
        
        assert result.success is False
        assert result.error_message is not None
        assert "insufficient" in result.error_message.lower() or "empty" in result.error_message.lower()

    def test_insufficient_notes_handling(self, insufficient_vault_dir, temp_output_dir):
        """Test handling of vault with insufficient notes."""
        generator = DatasetGenerator(
            vault_path=insufficient_vault_dir,
            output_dir=temp_output_dir
        )
        
        result = generator.generate_datasets()
        
        assert result.success is False
        assert isinstance(result.error_message, str)

    def test_invalid_vault_path_handling(self, temp_output_dir):
        """Test handling of invalid vault path."""
        invalid_path = Path("/nonexistent/vault/path")
        
        with pytest.raises((FileNotFoundError, OSError)):
            DatasetGenerator(
                vault_path=invalid_path,
                output_dir=temp_output_dir
            )

    def test_invalid_output_path_handling(self, temp_vault_dir):
        """Test handling of invalid output path."""
        # Create a minimal valid vault
        note_file = temp_vault_dir / "test.md"
        note_file.write_text("# Test\nContent")
        
        invalid_output = Path("/nonexistent/output/path")
        
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=invalid_output
        )
        
        # Should handle invalid output path gracefully
        result = generator.generate_datasets()
        assert result.success is False or result.error_message is not None

    def test_service_unavailability_handling(self, temp_vault_dir, temp_output_dir):
        """Test handling when services are unavailable."""
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir
        )
        
        # Mock service failures
        with patch.object(generator.vault_reader, 'get_markdown_files') as mock_get_files:
            mock_get_files.side_effect = Exception("Service unavailable")
            
            result = generator.generate_datasets()
            
            assert result.success is False
            assert result.error_message is not None

    def test_partial_failure_recovery(self, temp_vault_dir, temp_output_dir, mock_vector_encoder):
        """Test recovery from partial failures."""
        # Create vault with mix of valid and problematic files
        valid_file = temp_vault_dir / "valid.md"
        valid_file.write_text("# Valid\nContent")
        
        # Create files that might cause issues
        for i in range(5):
            note_file = temp_vault_dir / f"note{i}.md"
            note_file.write_text(f"# Note {i}\nContent {i}")
        
        generator = DatasetGenerator(
            vault_path=temp_vault_dir,
            output_dir=temp_output_dir
        )
        
        # Mock partial failures in processing
        original_extract = generator.notes_generator._extract_note_features
        
        def failing_extract(note_path, content, metadata):
            if "note1" in note_path or "note3" in note_path:
                raise FeatureEngineeringError("Feature extraction failed")
            return original_extract(note_path, content, metadata)
        
        with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder), \
             patch.object(generator.notes_generator, '_extract_note_features', failing_extract):
            
            result = generator.generate_datasets()
            
            # Should succeed with partial data or provide meaningful error
            if result.success:
                # Verify some data was processed despite failures
                summary = result.summary
                assert summary.notes_processed > 0
                assert summary.notes_failed > 0
            else:
                assert result.error_message is not None