"""
Performance and scalability tests for dataset generation.

Tests dataset generation with vaults of different sizes, memory usage and
processing time for large datasets, and batch processing efficiency.
"""

import gc
import os
import psutil
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import PairsDatasetGenerator
from jarvis.tools.dataset_generation.utils.memory_monitor import MemoryMonitor
from jarvis.tools.dataset_generation.utils.parallel_processor import ParallelBatchProcessor
from jarvis.tools.dataset_generation.models.data_models import NoteData
from jarvis.services.vault.reader import VaultReader


class PerformanceTestHelper:
    """Helper class for performance testing utilities."""
    
    @staticmethod
    def create_test_vault(vault_path: Path, num_notes: int, avg_content_size: int = 500,
                         link_density: float = 0.1) -> None:
        """Create a test vault with specified characteristics.
        
        Args:
            vault_path: Path to create vault in
            num_notes: Number of notes to create
            avg_content_size: Average content size in characters
            link_density: Probability of links between notes
        """
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Generate note content templates
        content_templates = [
            "# {title}\n\nThis note discusses {topic}.\n\n## Key Points\n{points}\n\n## Related\n{links}",
            "# {title}\n\n{topic} is an important concept.\n\n### Details\n{details}\n\n### See Also\n{links}",
            "# {title}\n\nOverview of {topic}.\n\n## Background\n{background}\n\n## References\n{links}"
        ]
        
        topics = [
            "machine learning", "data science", "artificial intelligence", "neural networks",
            "deep learning", "computer vision", "natural language processing", "robotics",
            "algorithms", "data structures", "software engineering", "system design",
            "databases", "web development", "mobile development", "cloud computing"
        ]
        
        # Create notes with realistic content and links
        note_paths = []
        for i in range(num_notes):
            note_name = f"note_{i:04d}.md"
            note_path = vault_path / note_name
            note_paths.append(note_name)
            
            # Generate content
            title = f"Note {i}: {np.random.choice(topics).title()}"
            topic = np.random.choice(topics)
            
            # Generate variable-length content
            content_multiplier = np.random.uniform(0.5, 2.0)
            target_size = int(avg_content_size * content_multiplier)
            
            points = []
            details = []
            background = []
            
            # Fill content to reach target size
            current_size = 0
            while current_size < target_size:
                if len(points) < 5:
                    point = f"- Important aspect of {topic} #{len(points) + 1}"
                    points.append(point)
                    current_size += len(point)
                
                if len(details) < 3:
                    detail = f"Detailed explanation of {topic} concept {len(details) + 1}. " * 3
                    details.append(detail)
                    current_size += len(detail)
                
                if len(background) < 2:
                    bg = f"Background information about {topic} and its applications. " * 2
                    background.append(bg)
                    current_size += len(bg)
                
                if len(points) >= 5 and len(details) >= 3 and len(background) >= 2:
                    break
            
            # Generate links based on density
            links = []
            for j in range(i):  # Only link to previous notes
                if np.random.random() < link_density:
                    target_note = f"note_{j:04d}"
                    links.append(f"[[{target_note}]]")
            
            # Add some frontmatter
            frontmatter = f"""---
title: {title}
tags: [{topic.replace(' ', '-')}, note-{i % 10}]
created: {int(time.time()) + i}
---

"""
            
            # Format content
            template = np.random.choice(content_templates)
            content = frontmatter + template.format(
                title=title,
                topic=topic,
                points='\n'.join(points),
                details='\n'.join(details),
                background='\n'.join(background),
                links='\n'.join(links) if links else "No related notes yet."
            )
            
            note_path.write_text(content, encoding='utf-8')
    
    @staticmethod
    def measure_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def measure_cpu_usage():
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)


class TestDatasetGenerationPerformance:
    """Performance tests for dataset generation with different vault sizes."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a fast mock vector encoder for performance testing."""
        mock_encoder = Mock()
        
        def fast_encode_documents(documents):
            # Simulate fast encoding with random embeddings
            return np.random.rand(len(documents), 384)
        
        def fast_encode_batch(texts):
            return fast_encode_documents(texts)
        
        mock_encoder.encode_documents.side_effect = fast_encode_documents
        mock_encoder.encode_batch.side_effect = fast_encode_batch
        mock_encoder.get_embedding_dimension.return_value = 384
        
        return mock_encoder
    
    @pytest.mark.parametrize("vault_size,expected_max_time", [
        (50, 30),    # Small vault: 50 notes, max 30 seconds
        (100, 60),   # Medium vault: 100 notes, max 60 seconds
        (200, 120),  # Large vault: 200 notes, max 120 seconds
    ])
    def test_dataset_generation_performance_by_size(self, vault_size, expected_max_time,
                                                   temp_output_dir, mock_vector_encoder):
        """Test dataset generation performance with different vault sizes."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / f"test_vault_{vault_size}"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, vault_size, avg_content_size=300, link_density=0.05
            )
            
            # Create generator with Areas/ filtering disabled for performance testing
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=False  # Disable Areas/ filtering for comprehensive performance testing
            )
            
            # Mock vector encoder for consistent performance
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Measure performance
                start_time = time.time()
                start_memory = PerformanceTestHelper.measure_memory_usage()
                
                result = generator.generate_datasets(batch_size=20)
                
                end_time = time.time()
                end_memory = PerformanceTestHelper.measure_memory_usage()
                
                # Verify success
                assert result.success is True
                
                # Check performance metrics
                total_time = end_time - start_time
                memory_increase = end_memory - start_memory
                
                print(f"\nVault size: {vault_size} notes")
                print(f"Total time: {total_time:.2f} seconds")
                print(f"Memory increase: {memory_increase:.2f} MB")
                print(f"Processing rate: {vault_size / total_time:.2f} notes/second")
                
                # Performance assertions
                assert total_time <= expected_max_time, f"Processing took {total_time:.2f}s, expected <= {expected_max_time}s"
                assert memory_increase <= 500, f"Memory increase {memory_increase:.2f}MB too high"
                
                # Verify output quality
                summary = result.summary
                assert summary.notes_processed == vault_size
                assert summary.processing_rate > 0
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_memory_usage_scaling(self, temp_output_dir, mock_vector_encoder):
        """Test memory usage scaling with vault size."""
        vault_sizes = [25, 50, 100]
        memory_usage = []
        
        for size in vault_sizes:
            # Create test vault
            temp_dir = tempfile.mkdtemp()
            vault_path = Path(temp_dir) / f"memory_test_{size}"
            
            try:
                PerformanceTestHelper.create_test_vault(
                    vault_path, size, avg_content_size=200, link_density=0.03
                )
                
                # Force garbage collection before measurement
                gc.collect()
                start_memory = PerformanceTestHelper.measure_memory_usage()
                
                generator = DatasetGenerator(
                    vault_path=vault_path,
                    output_dir=temp_output_dir
                )
                
                with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                     patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                    
                    result = generator.generate_datasets(batch_size=10)
                    
                    peak_memory = PerformanceTestHelper.measure_memory_usage()
                    memory_increase = peak_memory - start_memory
                    memory_usage.append((size, memory_increase))
                    
                    print(f"Vault size: {size}, Memory increase: {memory_increase:.2f} MB")
                    
                    assert result.success is True
                    
                    # Memory should not grow excessively
                    assert memory_increase <= size * 2, f"Memory usage too high: {memory_increase:.2f}MB for {size} notes"
                
            finally:
                import shutil
                shutil.rmtree(temp_dir)
                gc.collect()  # Clean up after each test
        
        # Check that memory usage scales reasonably
        memory_per_note = [mem / size for size, mem in memory_usage]
        avg_memory_per_note = sum(memory_per_note) / len(memory_per_note)
        
        print(f"Average memory per note: {avg_memory_per_note:.3f} MB")
        assert avg_memory_per_note <= 5.0, "Memory usage per note too high"
    
    def test_batch_processing_efficiency(self, temp_output_dir, mock_vector_encoder):
        """Test efficiency of different batch sizes."""
        # Create medium-sized test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "batch_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 80, avg_content_size=250, link_density=0.04
            )
            
            batch_sizes = [5, 10, 20, 40]
            performance_results = []
            
            for batch_size in batch_sizes:
                generator = DatasetGenerator(
                    vault_path=vault_path,
                    output_dir=temp_output_dir
                )
                
                with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                     patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                    
                    start_time = time.time()
                    start_memory = PerformanceTestHelper.measure_memory_usage()
                    
                    result = generator.generate_datasets(batch_size=batch_size)
                    
                    end_time = time.time()
                    end_memory = PerformanceTestHelper.measure_memory_usage()
                    
                    total_time = end_time - start_time
                    memory_increase = end_memory - start_memory
                    
                    performance_results.append({
                        'batch_size': batch_size,
                        'time': total_time,
                        'memory': memory_increase,
                        'rate': 80 / total_time
                    })
                    
                    print(f"Batch size: {batch_size}, Time: {total_time:.2f}s, Memory: {memory_increase:.2f}MB")
                    
                    assert result.success is True
            
            # Analyze batch size efficiency
            # Larger batch sizes should generally be more efficient (up to a point)
            times = [r['time'] for r in performance_results]
            
            # At least one batch size should be reasonably fast
            assert min(times) <= 60, "All batch sizes too slow"
            
            # Memory usage should be reasonable for all batch sizes
            memories = [r['memory'] for r in performance_results]
            assert max(memories) <= 300, "Memory usage too high for some batch sizes"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_parallel_processing_performance(self, temp_output_dir, mock_vector_encoder):
        """Test parallel processing performance improvements."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "parallel_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 60, avg_content_size=300, link_density=0.05
            )
            
            # Test sequential vs parallel processing
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Sequential processing (batch_size=1 simulates sequential)
                start_time = time.time()
                result_sequential = generator.generate_datasets(batch_size=1)
                sequential_time = time.time() - start_time
                
                # Parallel processing (larger batch size)
                start_time = time.time()
                result_parallel = generator.generate_datasets(batch_size=15)
                parallel_time = time.time() - start_time
                
                print(f"Sequential time: {sequential_time:.2f}s")
                print(f"Parallel time: {parallel_time:.2f}s")
                print(f"Speedup: {sequential_time / parallel_time:.2f}x")
                
                assert result_sequential.success is True
                assert result_parallel.success is True
                
                # Parallel should be faster (allowing some variance)
                assert parallel_time <= sequential_time * 1.2, "Parallel processing not providing expected speedup"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_memory_monitoring_effectiveness(self, temp_output_dir, mock_vector_encoder):
        """Test effectiveness of memory monitoring and adaptive batch sizing."""
        # Create test vault that might stress memory
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "memory_monitor_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 100, avg_content_size=800, link_density=0.08
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            # Mock memory monitor to simulate memory pressure
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder), \
                 patch('jarvis.tools.dataset_generation.utils.memory_monitor.MemoryMonitor') as mock_monitor_class:
                
                # Create mock monitor that reports memory pressure after some processing
                mock_monitor = Mock()
                mock_monitor.should_pause.side_effect = [False] * 5 + [True] * 2 + [False] * 10
                mock_monitor.current_usage.return_value = 800  # MB
                mock_monitor.force_cleanup.return_value = None
                mock_monitor_class.return_value = mock_monitor
                
                start_time = time.time()
                result = generator.generate_datasets(batch_size=20)
                total_time = time.time() - start_time
                
                assert result.success is True
                
                # Verify memory monitor was used
                assert mock_monitor.should_pause.call_count > 0
                
                # Processing should complete despite memory pressure
                summary = result.summary
                assert summary.notes_processed > 0
                
                print(f"Processing with memory monitoring: {total_time:.2f}s")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestDatasetGenerationScalability:
    """Scalability tests for dataset generation."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_vector_encoder(self):
        """Create mock vector encoder for scalability testing."""
        mock_encoder = Mock()
        
        def encode_documents(documents):
            # Simulate realistic encoding time
            time.sleep(len(documents) * 0.001)  # 1ms per document
            return np.random.rand(len(documents), 384)
        
        mock_encoder.encode_documents.side_effect = encode_documents
        mock_encoder.encode_batch.side_effect = encode_documents
        mock_encoder.get_embedding_dimension.return_value = 384
        
        return mock_encoder
    
    def test_large_vault_processing(self, temp_output_dir, mock_vector_encoder):
        """Test processing of large vaults (stress test)."""
        # Create large test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "large_vault"
        
        try:
            # Create a larger vault for stress testing
            vault_size = 300
            PerformanceTestHelper.create_test_vault(
                vault_path, vault_size, avg_content_size=400, link_density=0.03
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                start_time = time.time()
                start_memory = PerformanceTestHelper.measure_memory_usage()
                
                # Use adaptive batch sizing for large vault
                result = generator.generate_datasets(batch_size=25)
                
                end_time = time.time()
                end_memory = PerformanceTestHelper.measure_memory_usage()
                
                total_time = end_time - start_time
                memory_increase = end_memory - start_memory
                
                print(f"\nLarge vault processing:")
                print(f"Vault size: {vault_size} notes")
                print(f"Total time: {total_time:.2f} seconds")
                print(f"Memory increase: {memory_increase:.2f} MB")
                print(f"Processing rate: {vault_size / total_time:.2f} notes/second")
                
                assert result.success is True
                
                # Performance requirements for large vaults
                assert total_time <= 300, f"Large vault processing too slow: {total_time:.2f}s"
                assert memory_increase <= 1000, f"Memory usage too high: {memory_increase:.2f}MB"
                
                # Verify data quality
                summary = result.summary
                assert summary.notes_processed == vault_size
                assert summary.pairs_generated > 0
                
                # Check output files exist and have reasonable size
                notes_df = pd.read_csv(result.notes_dataset_path)
                pairs_df = pd.read_csv(result.pairs_dataset_path)
                
                assert len(notes_df) == vault_size
                assert len(pairs_df) > 0
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_concurrent_processing_scalability(self, temp_output_dir, mock_vector_encoder):
        """Test scalability under concurrent processing scenarios."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "concurrent_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 80, avg_content_size=300, link_density=0.04
            )
            
            # Test multiple concurrent dataset generations
            results = []
            errors = []
            threads = []
            
            def generate_dataset(thread_id):
                try:
                    thread_output_dir = temp_output_dir / f"thread_{thread_id}"
                    thread_output_dir.mkdir(exist_ok=True)
                    
                    generator = DatasetGenerator(
                        vault_path=vault_path,
                        output_dir=thread_output_dir
                    )
                    
                    with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                         patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                        
                        start_time = time.time()
                        result = generator.generate_datasets(batch_size=10)
                        end_time = time.time()
                        
                        results.append({
                            'thread_id': thread_id,
                            'success': result.success,
                            'time': end_time - start_time,
                            'notes_processed': result.summary.notes_processed if result.success else 0
                        })
                        
                except Exception as e:
                    errors.append((thread_id, str(e)))
            
            # Start multiple threads
            num_threads = 3
            for i in range(num_threads):
                thread = threading.Thread(target=generate_dataset, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Analyze results
            assert len(errors) == 0, f"Concurrent processing errors: {errors}"
            assert len(results) == num_threads
            
            successful_results = [r for r in results if r['success']]
            assert len(successful_results) == num_threads, "Not all concurrent processes succeeded"
            
            # Check performance consistency
            times = [r['time'] for r in successful_results]
            avg_time = sum(times) / len(times)
            max_time = max(times)
            
            print(f"Concurrent processing - Average time: {avg_time:.2f}s, Max time: {max_time:.2f}s")
            
            # Times should be reasonably consistent (within 50% of average)
            assert max_time <= avg_time * 1.5, "Concurrent processing times too inconsistent"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_streaming_processing_scalability(self, temp_output_dir, mock_vector_encoder):
        """Test streaming processing for very large datasets."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "streaming_test"
        
        try:
            # Create vault that would be challenging to process in memory
            vault_size = 150
            PerformanceTestHelper.create_test_vault(
                vault_path, vault_size, avg_content_size=600, link_density=0.06
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Monitor memory usage during streaming processing
                memory_samples = []
                
                def monitor_memory():
                    for _ in range(30):  # Monitor for 30 seconds
                        memory_samples.append(PerformanceTestHelper.measure_memory_usage())
                        time.sleep(1)
                
                # Start memory monitoring
                monitor_thread = threading.Thread(target=monitor_memory)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                start_time = time.time()
                result = generator.generate_datasets(batch_size=15)
                end_time = time.time()
                
                total_time = end_time - start_time
                
                assert result.success is True
                
                # Analyze memory usage pattern
                if memory_samples:
                    max_memory = max(memory_samples)
                    min_memory = min(memory_samples)
                    memory_variance = max_memory - min_memory
                    
                    print(f"Streaming processing memory usage:")
                    print(f"Min: {min_memory:.2f}MB, Max: {max_memory:.2f}MB, Variance: {memory_variance:.2f}MB")
                    
                    # Memory variance should be reasonable (not constantly growing)
                    assert memory_variance <= 400, f"Memory variance too high: {memory_variance:.2f}MB"
                
                # Performance should be reasonable
                processing_rate = vault_size / total_time
                print(f"Streaming processing rate: {processing_rate:.2f} notes/second")
                
                assert processing_rate >= 0.5, "Streaming processing too slow"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_error_recovery_scalability(self, temp_output_dir, mock_vector_encoder):
        """Test error recovery mechanisms under scale."""
        # Create test vault with some problematic files
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "error_recovery_test"
        
        try:
            # Create normal files
            PerformanceTestHelper.create_test_vault(
                vault_path, 60, avg_content_size=300, link_density=0.04
            )
            
            # Add some problematic files
            problematic_files = [
                ("empty.md", ""),  # Empty file
                ("malformed.md", "[[Unclosed link\n[Malformed]("),  # Malformed links
                ("large.md", "# Large\n" + "Content " * 10000),  # Very large file
            ]
            
            for filename, content in problematic_files:
                (vault_path / filename).write_text(content)
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            # Mock some processing failures
            original_extract = generator.notes_generator._extract_note_features
            failure_count = 0
            
            def failing_extract(note_path, content, metadata):
                nonlocal failure_count
                # Fail on some files to test error recovery
                if "note_0010" in note_path or "note_0025" in note_path:
                    failure_count += 1
                    raise Exception(f"Simulated processing failure for {note_path}")
                return original_extract(note_path, content, metadata)
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.notes_generator, '_extract_note_features', failing_extract):
                
                start_time = time.time()
                result = generator.generate_datasets(batch_size=12)
                end_time = time.time()
                
                total_time = end_time - start_time
                
                # Should handle errors gracefully
                if result.success:
                    # Partial success is acceptable
                    summary = result.summary
                    assert summary.notes_processed > 0
                    assert summary.notes_failed > 0
                    
                    print(f"Error recovery test:")
                    print(f"Processed: {summary.notes_processed}, Failed: {summary.notes_failed}")
                    print(f"Success rate: {summary.success_rate:.2f}")
                    
                    # Should process most files despite some failures
                    assert summary.success_rate >= 0.8, "Too many processing failures"
                    
                else:
                    # Complete failure is also acceptable if error handling is working
                    assert result.error_message is not None
                
                # Should complete in reasonable time even with errors
                assert total_time <= 120, f"Error recovery took too long: {total_time:.2f}s"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestDatasetGenerationResourceUsage:
    """Tests for resource usage optimization and limits."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_cpu_usage_optimization(self, temp_output_dir):
        """Test CPU usage optimization during processing."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "cpu_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 50, avg_content_size=400, link_density=0.05
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            # Monitor CPU usage during processing
            cpu_samples = []
            
            def monitor_cpu():
                for _ in range(20):  # Monitor for 20 seconds
                    cpu_samples.append(PerformanceTestHelper.measure_cpu_usage())
                    time.sleep(1)
            
            # Start CPU monitoring
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Mock vector encoder for consistent behavior
            mock_encoder = Mock()
            mock_encoder.encode_documents.return_value = np.random.rand(10, 384)
            mock_encoder.encode_batch.return_value = np.random.rand(10, 384)
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_encoder):
                
                result = generator.generate_datasets(batch_size=10)
                
                assert result.success is True
                
                # Analyze CPU usage
                if cpu_samples:
                    avg_cpu = sum(cpu_samples) / len(cpu_samples)
                    max_cpu = max(cpu_samples)
                    
                    print(f"CPU usage - Average: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
                    
                    # CPU usage should be reasonable (not constantly at 100%)
                    assert avg_cpu <= 80, f"Average CPU usage too high: {avg_cpu:.1f}%"
                    assert max_cpu <= 95, f"Peak CPU usage too high: {max_cpu:.1f}%"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_disk_io_optimization(self, temp_output_dir):
        """Test disk I/O optimization during processing."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "disk_io_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 40, avg_content_size=500, link_density=0.04
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            # Monitor disk I/O
            process = psutil.Process(os.getpid())
            io_start = process.io_counters()
            
            # Mock vector encoder
            mock_encoder = Mock()
            mock_encoder.encode_documents.return_value = np.random.rand(10, 384)
            mock_encoder.encode_batch.return_value = np.random.rand(10, 384)
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_encoder):
                
                start_time = time.time()
                result = generator.generate_datasets(batch_size=8)
                end_time = time.time()
                
                io_end = process.io_counters()
                
                assert result.success is True
                
                # Calculate I/O metrics
                read_bytes = io_end.read_bytes - io_start.read_bytes
                write_bytes = io_end.write_bytes - io_start.write_bytes
                total_time = end_time - start_time
                
                read_rate = read_bytes / (1024 * 1024) / total_time  # MB/s
                write_rate = write_bytes / (1024 * 1024) / total_time  # MB/s
                
                print(f"Disk I/O - Read: {read_rate:.2f} MB/s, Write: {write_rate:.2f} MB/s")
                
                # I/O rates should be reasonable
                assert read_rate <= 100, f"Read rate too high: {read_rate:.2f} MB/s"
                assert write_rate <= 50, f"Write rate too high: {write_rate:.2f} MB/s"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_memory_leak_detection(self, temp_output_dir):
        """Test for memory leaks during repeated processing."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "memory_leak_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 30, avg_content_size=300, link_density=0.03
            )
            
            # Mock vector encoder
            mock_encoder = Mock()
            mock_encoder.encode_documents.return_value = np.random.rand(10, 384)
            mock_encoder.encode_batch.return_value = np.random.rand(10, 384)
            
            memory_measurements = []
            
            # Run multiple iterations to detect memory leaks
            for iteration in range(5):
                gc.collect()  # Force garbage collection
                start_memory = PerformanceTestHelper.measure_memory_usage()
                
                generator = DatasetGenerator(
                    vault_path=vault_path,
                    output_dir=temp_output_dir / f"iteration_{iteration}"
                )
                
                with patch.object(generator.notes_generator, 'vector_encoder', mock_encoder), \
                     patch.object(generator.pairs_generator, 'vector_encoder', mock_encoder):
                    
                    result = generator.generate_datasets(batch_size=6)
                    assert result.success is True
                
                # Clean up generator
                del generator
                gc.collect()
                
                end_memory = PerformanceTestHelper.measure_memory_usage()
                memory_measurements.append(end_memory - start_memory)
                
                print(f"Iteration {iteration + 1}: Memory increase {memory_measurements[-1]:.2f} MB")
            
            # Check for memory leaks
            # Memory usage should not consistently increase across iterations
            if len(memory_measurements) >= 3:
                # Check if memory usage is consistently increasing
                increasing_trend = all(
                    memory_measurements[i] <= memory_measurements[i + 1] * 1.2
                    for i in range(len(memory_measurements) - 1)
                )
                
                avg_memory = sum(memory_measurements) / len(memory_measurements)
                max_memory = max(memory_measurements)
                
                print(f"Memory usage - Average: {avg_memory:.2f} MB, Max: {max_memory:.2f} MB")
                
                # Memory usage should be reasonable and not show severe leaks
                assert max_memory <= avg_memory * 2, "Possible memory leak detected"
                assert avg_memory <= 100, f"Average memory usage too high: {avg_memory:.2f} MB"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    @pytest.mark.parametrize("resource_limit", [
        {"memory_mb": 200, "time_seconds": 60},
        {"memory_mb": 500, "time_seconds": 120},
    ])
    def test_resource_constrained_processing(self, resource_limit, temp_output_dir):
        """Test processing under resource constraints."""
        # Create test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "resource_constrained_test"
        
        try:
            PerformanceTestHelper.create_test_vault(
                vault_path, 60, avg_content_size=350, link_density=0.04
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            # Mock vector encoder with controlled resource usage
            mock_encoder = Mock()
            
            def controlled_encode(documents):
                # Simulate controlled resource usage
                time.sleep(len(documents) * 0.002)  # 2ms per document
                return np.random.rand(len(documents), 384)
            
            mock_encoder.encode_documents.side_effect = controlled_encode
            mock_encoder.encode_batch.side_effect = controlled_encode
            
            # Monitor resource usage
            start_time = time.time()
            start_memory = PerformanceTestHelper.measure_memory_usage()
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_encoder):
                
                # Use smaller batch size for resource-constrained processing
                result = generator.generate_datasets(batch_size=5)
                
                end_time = time.time()
                end_memory = PerformanceTestHelper.measure_memory_usage()
                
                total_time = end_time - start_time
                memory_increase = end_memory - start_memory
                
                print(f"Resource constrained processing:")
                print(f"Time: {total_time:.2f}s (limit: {resource_limit['time_seconds']}s)")
                print(f"Memory: {memory_increase:.2f}MB (limit: {resource_limit['memory_mb']}MB)")
                
                # Should complete within resource limits
                assert total_time <= resource_limit['time_seconds'], f"Exceeded time limit"
                assert memory_increase <= resource_limit['memory_mb'], f"Exceeded memory limit"
                
                # Should still produce valid results
                if result.success:
                    summary = result.summary
                    assert summary.notes_processed > 0
                    assert summary.pairs_generated > 0
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)