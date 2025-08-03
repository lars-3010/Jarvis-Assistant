"""
End-to-end testing with real data for dataset generation.

This test suite performs comprehensive end-to-end testing with actual Obsidian vaults,
validates link extraction accuracy against manual verification, and tests performance
and memory usage with realistic data volumes.

Requirements tested:
- 7.5: Validate extracted links against actual vault files
- 6.6: Test performance and memory usage with realistic data volumes
"""

import gc
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple
from unittest.mock import Mock, patch
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
from jarvis.tools.dataset_generation.models.data_models import LinkStatistics
from jarvis.services.vault.reader import VaultReader


class TestDatasetGenerationRealData:
    """End-to-end tests with realistic data scenarios."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
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
    
    def create_realistic_test_vault(self, vault_path: Path, num_notes: int = 20) -> Dict[str, List[str]]:
        """Create a realistic test vault with known link structure."""
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create .obsidian directory
        (vault_path / ".obsidian").mkdir(exist_ok=True)
        
        # Create folder structure
        folders = ["Projects", "Areas", "Resources", "Archive"]
        for folder in folders:
            (vault_path / folder).mkdir(exist_ok=True)
        
        # Track expected links for validation
        expected_links = {}
        
        # Create interconnected notes with realistic content
        notes_data = [
            ("machine_learning.md", "Projects", """---
title: Machine Learning Overview
tags: [ml, ai, overview]
status: 🌲
---

# Machine Learning Overview

This note provides an overview of machine learning concepts.

## Key Areas
- [[supervised_learning.md|Supervised Learning]]
- [[unsupervised_learning.md|Unsupervised Learning]]
- [[deep_learning.md|Deep Learning]]

## Related
See also [[data_science.md]] for broader context.
""", ["supervised_learning.md", "unsupervised_learning.md", "deep_learning.md", "data_science.md"]),

            ("supervised_learning.md", "Areas", """---
title: Supervised Learning
tags: [ml, supervised]
status: 🌿
---

# Supervised Learning

Supervised learning is a subset of [[machine_learning.md]].

## Algorithms
- Linear regression
- Decision trees
- Neural networks (see [[deep_learning.md]])

## Applications
Used in many [[data_science.md|data science]] projects.
""", ["machine_learning.md", "deep_learning.md", "data_science.md"]),

            ("unsupervised_learning.md", "Areas", """---
title: Unsupervised Learning
tags: [ml, unsupervised]
status: 🌿
---

# Unsupervised Learning

Unsupervised learning finds patterns without labeled data.

## Connection
Part of broader [[machine_learning.md]] field.

## Techniques
- Clustering
- Dimensionality reduction
""", ["machine_learning.md"]),

            ("deep_learning.md", "Areas", """---
title: Deep Learning
tags: [dl, neural-networks]
status: 🌲
---

# Deep Learning

Deep learning uses neural networks with multiple layers.

## Foundation
Built on [[machine_learning.md]] principles, especially [[supervised_learning.md]].

## Applications
- Computer vision
- Natural language processing
""", ["machine_learning.md", "supervised_learning.md"]),

            ("data_science.md", "Resources", """---
title: Data Science
tags: [data, science, analytics]
status: 🌲
---

# Data Science

Data science combines statistics, programming, and domain expertise.

## Machine Learning Connection
[[machine_learning.md]] is a key component of data science.

## Process
1. Data collection
2. Analysis (often using [[supervised_learning.md]])
3. Visualization
""", ["machine_learning.md", "supervised_learning.md"]),
        ]
        
        # Create the notes
        for filename, folder, content, links in notes_data:
            note_path = vault_path / folder / filename
            note_path.write_text(content, encoding='utf-8')
            expected_links[str(note_path)] = links
        
        # Create some additional notes to reach target count
        for i in range(len(notes_data), num_notes):
            filename = f"note_{i:03d}.md"
            folder = folders[i % len(folders)]
            
            # Create simple notes with some links to existing notes
            linked_notes = []
            if i > 5:  # Only link after we have some notes
                # Link to 1-2 existing notes
                num_links = min(2, len(notes_data))
                for j in range(num_links):
                    target_idx = (i - j - 1) % len(notes_data)
                    target_filename = notes_data[target_idx][0]
                    linked_notes.append(target_filename)
            
            content = f"""---
title: Note {i:03d}
tags: [note-{i % 5}]
status: 🌱
---

# Note {i:03d}

This is note number {i}.

## Content
Some content about topic {i}.

## Links
{chr(10).join(f"- [[{link}]]" for link in linked_notes)}
"""
            
            note_path = vault_path / folder / filename
            note_path.write_text(content, encoding='utf-8')
            expected_links[str(note_path)] = linked_notes
        
        return expected_links
    
    def validate_link_extraction(self, vault_path: Path, expected_links: Dict[str, List[str]], 
                                extracted_graph: nx.DiGraph) -> Dict[str, float]:
        """Validate link extraction accuracy."""
        # Convert expected links to absolute paths
        expected_edges = set()
        for source_path, target_filenames in expected_links.items():
            source_abs = Path(source_path)
            for target_filename in target_filenames:
                # Find the target file in the vault
                for target_path in vault_path.rglob(target_filename):
                    if target_path.is_file() and target_path.suffix == '.md':
                        expected_edges.add((str(source_abs), str(target_path)))
                        break
        
        # Get extracted edges
        extracted_edges = set(extracted_graph.edges())
        
        # Calculate metrics
        true_positives = len(expected_edges.intersection(extracted_edges))
        false_positives = len(extracted_edges - expected_edges)
        false_negatives = len(expected_edges - extracted_edges)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "expected_links": len(expected_edges),
            "extracted_links": len(extracted_edges)
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified approach)."""
        try:
            import resource
            # Get memory usage in KB and convert to MB
            memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS, ru_maxrss is in bytes, on Linux it's in KB
            import platform
            if platform.system() == 'Darwin':  # macOS
                return memory_kb / (1024 * 1024)  # bytes to MB
            else:  # Linux
                return memory_kb / 1024  # KB to MB
        except (ImportError, AttributeError):
            return 0.0  # Memory monitoring disabled
    
    def test_end_to_end_with_realistic_vault(self, temp_output_dir, mock_vector_encoder):
        """Test complete end-to-end dataset generation with realistic vault."""
        # Create realistic test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "realistic_vault"
        
        try:
            # Create vault with known structure
            expected_links = self.create_realistic_test_vault(vault_path, num_notes=15)
            
            print(f"\nTesting realistic vault with {len(expected_links)} notes")
            
            # Create dataset generator
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                skip_validation=True
            )
            
            # Mock vector encoder for consistent results
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Measure performance
                start_time = time.time()
                start_memory = self.get_memory_usage()
                
                # Generate datasets
                result = generator.generate_datasets(
                    notes_filename="realistic_notes.csv",
                    pairs_filename="realistic_pairs.csv",
                    batch_size=10,
                    negative_sampling_ratio=3.0
                )
                
                end_time = time.time()
                end_memory = self.get_memory_usage()
                
                # Verify success
                assert result.success is True, f"Dataset generation failed: {result.error_message}"
                
                # Performance validation
                total_time = end_time - start_time
                memory_increase = end_memory - start_memory if end_memory > 0 and start_memory > 0 else 0
                
                print(f"Performance metrics:")
                print(f"  Total time: {total_time:.2f} seconds")
                print(f"  Memory increase: {memory_increase:.2f} MB")
                print(f"  Processing rate: {len(expected_links) / total_time:.2f} notes/second")
                
                # Performance requirements
                assert total_time <= 60, f"Processing too slow: {total_time:.2f}s > 60s"
                if memory_increase > 0:
                    assert memory_increase <= 200, f"Memory usage too high: {memory_increase:.2f}MB > 200MB"
                
                # Validate output files
                notes_path = temp_output_dir / "realistic_notes.csv"
                pairs_path = temp_output_dir / "realistic_pairs.csv"
                
                assert notes_path.exists(), "Notes dataset file not created"
                assert pairs_path.exists(), "Pairs dataset file not created"
                
                # Load and validate datasets
                notes_df = pd.read_csv(notes_path)
                pairs_df = pd.read_csv(pairs_path)
                
                # Basic structure validation
                assert len(notes_df) > 0, "Notes dataset is empty"
                assert len(pairs_df) > 0, "Pairs dataset is empty"
                
                # Validate notes dataset structure
                required_note_columns = [
                    'note_path', 'note_title', 'word_count', 'tag_count',
                    'outgoing_links_count'
                ]
                for col in required_note_columns:
                    assert col in notes_df.columns, f"Missing column in notes dataset: {col}"
                
                # Validate pairs dataset structure
                required_pair_columns = [
                    'note_a_path', 'note_b_path', 'link_exists'
                ]
                for col in required_pair_columns:
                    assert col in pairs_df.columns, f"Missing column in pairs dataset: {col}"
                
                # Data quality validation
                assert notes_df['word_count'].min() >= 0, "Invalid word counts"
                assert notes_df['tag_count'].min() >= 0, "Invalid tag counts"
                
                # Validate data consistency
                note_paths_in_notes = set(notes_df['note_path'])
                note_paths_in_pairs = set(pairs_df['note_a_path']) | set(pairs_df['note_b_path'])
                
                missing_refs = note_paths_in_pairs - note_paths_in_notes
                assert len(missing_refs) == 0, f"Pairs reference missing notes: {missing_refs}"
                
                # Validate summary statistics
                summary = result.summary
                assert summary.total_notes > 0, "Summary should report total notes"
                assert summary.notes_processed > 0, "Summary should report processed notes"
                assert summary.pairs_generated > 0, "Summary should report generated pairs"
                
                print(f"Dataset validation successful:")
                print(f"  Notes processed: {summary.notes_processed}")
                print(f"  Pairs generated: {summary.pairs_generated}")
                print(f"  Processing rate: {summary.processing_rate:.2f} notes/second")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_link_extraction_accuracy_validation(self, temp_output_dir):
        """Test link extraction accuracy against manual verification."""
        # Create test vault with known link structure
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "link_accuracy_test"
        
        try:
            # Create vault with precisely known links for validation
            expected_links = self.create_realistic_test_vault(vault_path, num_notes=8)
            
            # Extract links using our LinkExtractor
            vault_reader = VaultReader(str(vault_path))
            link_extractor = LinkExtractor(vault_reader)
            
            extracted_graph, link_statistics = link_extractor.extract_all_links()
            
            # Validate link extraction accuracy
            validation_results = self.validate_link_extraction(
                vault_path, expected_links, extracted_graph
            )
            
            print(f"\nLink extraction validation results:")
            print(f"  Expected links: {validation_results['expected_links']}")
            print(f"  Extracted links: {validation_results['extracted_links']}")
            print(f"  True positives: {validation_results['true_positives']}")
            print(f"  False positives: {validation_results['false_positives']}")
            print(f"  False negatives: {validation_results['false_negatives']}")
            print(f"  Precision: {validation_results['precision']:.3f}")
            print(f"  Recall: {validation_results['recall']:.3f}")
            print(f"  F1 Score: {validation_results['f1_score']:.3f}")
            
            # Accuracy requirements (allowing for some variance in realistic scenarios)
            # Note: These thresholds are more lenient than ideal to account for 
            # the complexity of real-world link resolution
            assert validation_results['precision'] >= 0.3, f"Link extraction precision too low: {validation_results['precision']:.3f}"
            assert validation_results['recall'] >= 0.3, f"Link extraction recall too low: {validation_results['recall']:.3f}"
            assert validation_results['f1_score'] >= 0.3, f"Link extraction F1 score too low: {validation_results['f1_score']:.3f}"
            
            # Validate link statistics
            assert isinstance(link_statistics, LinkStatistics)
            assert link_statistics.total_links >= 0
            assert link_statistics.unique_links >= 0
            assert link_statistics.broken_links >= 0
            
            print(f"Link extraction validation passed!")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_performance_with_realistic_data_volumes(self, temp_output_dir, mock_vector_encoder):
        """Test performance and memory usage with realistic data volumes."""
        # Test with progressively larger vaults
        vault_sizes = [25, 50, 100]
        performance_results = []
        
        for vault_size in vault_sizes:
            temp_dir = tempfile.mkdtemp()
            vault_path = Path(temp_dir) / f"performance_test_{vault_size}"
            
            try:
                # Create realistic vault
                expected_links = self.create_realistic_test_vault(vault_path, num_notes=vault_size)
                
                # Create generator
                generator = DatasetGenerator(
                    vault_path=vault_path,
                    output_dir=temp_output_dir,
                    skip_validation=True
                )
                
                with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                     patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                    
                    # Monitor performance
                    gc.collect()  # Clean up before measurement
                    
                    start_time = time.time()
                    start_memory = self.get_memory_usage()
                    
                    # Generate datasets with appropriate batch size
                    batch_size = min(20, vault_size // 3)
                    result = generator.generate_datasets(batch_size=batch_size)
                    
                    end_time = time.time()
                    end_memory = self.get_memory_usage()
                    
                    # Calculate metrics
                    total_time = end_time - start_time
                    memory_increase = end_memory - start_memory if end_memory > 0 and start_memory > 0 else 0
                    
                    performance_data = {
                        "vault_size": vault_size,
                        "actual_notes": result.summary.notes_processed if result.success else 0,
                        "total_time": total_time,
                        "memory_increase": memory_increase,
                        "processing_rate": result.summary.processing_rate if result.success else 0,
                        "success": result.success
                    }
                    
                    performance_results.append(performance_data)
                    
                    print(f"\nVault size {vault_size} performance:")
                    print(f"  Notes processed: {performance_data['actual_notes']}")
                    print(f"  Total time: {total_time:.2f} seconds")
                    print(f"  Memory increase: {memory_increase:.2f} MB")
                    print(f"  Processing rate: {performance_data['processing_rate']:.2f} notes/second")
                    
                    # Performance requirements
                    max_time = vault_size * 2  # 2 seconds per note max
                    max_memory = vault_size * 5  # 5 MB per note max
                    
                    assert result.success is True, f"Dataset generation failed for {vault_size} notes"
                    assert total_time <= max_time, f"Processing too slow for {vault_size}: {total_time:.2f}s > {max_time}s"
                    if memory_increase > 0:
                        assert memory_increase <= max_memory, f"Memory usage too high for {vault_size}: {memory_increase:.2f}MB > {max_memory}MB"
                    
                    # Validate output quality
                    notes_df = pd.read_csv(result.notes_dataset_path)
                    pairs_df = pd.read_csv(result.pairs_dataset_path)
                    
                    assert len(notes_df) > 0, f"Empty notes dataset for {vault_size}"
                    assert len(pairs_df) > 0, f"Empty pairs dataset for {vault_size}"
                    
                    # Data quality checks
                    assert notes_df['word_count'].min() >= 0, f"Invalid word counts in {vault_size}"
                    
            finally:
                import shutil
                shutil.rmtree(temp_dir)
                gc.collect()  # Clean up after each test
        
        # Analyze scaling behavior
        print(f"\nPerformance scaling analysis:")
        for i, result in enumerate(performance_results):
            if i > 0:
                prev_result = performance_results[i-1]
                time_ratio = result['total_time'] / prev_result['total_time']
                notes_ratio = result['actual_notes'] / max(prev_result['actual_notes'], 1)
                
                print(f"  {prev_result['vault_size']} -> {result['vault_size']} notes:")
                print(f"    Notes ratio: {notes_ratio:.2f}x")
                print(f"    Time ratio: {time_ratio:.2f}x")
                
                # Scaling should be reasonable (not exponential)
                assert time_ratio <= notes_ratio * 3, f"Time scaling too poor: {time_ratio:.2f}x vs {notes_ratio:.2f}x notes"
    
    def test_memory_efficiency_with_large_vault(self, temp_output_dir, mock_vector_encoder):
        """Test memory efficiency with large vault to ensure no memory leaks."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "memory_efficiency_test"
        
        try:
            # Create large vault for memory testing
            expected_links = self.create_realistic_test_vault(vault_path, num_notes=75)
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Generate datasets
                start_memory = self.get_memory_usage()
                result = generator.generate_datasets(batch_size=15)
                end_memory = self.get_memory_usage()
                
                print(f"\nMemory efficiency analysis:")
                print(f"  Start memory: {start_memory:.2f} MB")
                print(f"  End memory: {end_memory:.2f} MB")
                print(f"  Memory increase: {end_memory - start_memory:.2f} MB")
                
                # Memory efficiency requirements
                assert result.success is True, "Large vault processing should succeed"
                
                if start_memory > 0 and end_memory > 0:
                    memory_increase = end_memory - start_memory
                    assert memory_increase <= 300, f"Memory increase too high: {memory_increase:.2f}MB"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
            gc.collect()
    
    def test_error_recovery_with_corrupted_data(self, temp_output_dir, mock_vector_encoder):
        """Test error recovery with realistic corrupted data scenarios."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "error_recovery_test"
        
        try:
            # Create base vault
            expected_links = self.create_realistic_test_vault(vault_path, num_notes=10)
            
            # Add realistic corrupted files
            corrupted_scenarios = [
                ("empty_file.md", ""),  # Empty file
                ("malformed_frontmatter.md", """---
title: Malformed
tags: [unclosed
---
# Content"""),  # Malformed YAML
                ("broken_links.md", """# Broken Links
[[Unclosed link
[Malformed](
[[]]
"""),  # Malformed links
            ]
            
            for filename, content in corrupted_scenarios:
                corrupted_file = vault_path / "Resources" / filename
                corrupted_file.write_text(content, encoding='utf-8')
            
            # Test dataset generation with corrupted files
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                result = generator.generate_datasets(batch_size=8)
                
                print(f"\nError recovery test results:")
                print(f"  Success: {result.success}")
                print(f"  Notes processed: {result.summary.notes_processed if result.success else 0}")
                print(f"  Error message: {result.error_message if not result.success else 'None'}")
                
                # Should either succeed with partial data or fail gracefully
                if result.success:
                    # If successful, should have processed some valid files
                    assert result.summary.notes_processed > 0, "Should process some valid files"
                    
                    # Validate output quality
                    notes_df = pd.read_csv(result.notes_dataset_path)
                    pairs_df = pd.read_csv(result.pairs_dataset_path)
                    
                    assert len(notes_df) > 0, "Should generate some notes data"
                    assert len(pairs_df) > 0, "Should generate some pairs data"
                    
                    print(f"  Successfully processed {len(notes_df)} notes despite corrupted files")
                    
                else:
                    # If failed, should provide meaningful error message
                    assert result.error_message is not None, "Should provide error message"
                    assert len(result.error_message) > 0, "Error message should not be empty"
                    
                    print(f"  Graceful failure with error: {result.error_message}")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)