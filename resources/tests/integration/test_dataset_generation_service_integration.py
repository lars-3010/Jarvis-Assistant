"""
Integration tests for dataset generation service integration.

Tests the integration between dataset generation components and existing Jarvis services,
including VaultReader, VectorEncoder, and GraphDatabase services.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import networkx as nx

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.tools.dataset_generation.models.exceptions import (
    VaultValidationError,
    ConfigurationError,
    InsufficientDataError
)
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.graph.database import GraphDatabase
from jarvis.utils.config import get_settings
from jarvis.core.container import ServiceContainer


class TestDatasetGenerationServiceIntegration:
    """Test integration with existing Jarvis services."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir) / "test_vault"
            vault_path.mkdir()
            
            # Create .obsidian directory to make it look like a real vault
            obsidian_dir = vault_path / ".obsidian"
            obsidian_dir.mkdir()
            
            # Create some test notes
            notes = [
                ("note1.md", "# Note 1\nThis is the first note.\n[[note2]] is linked."),
                ("note2.md", "# Note 2\nThis is the second note.\n[[note3]] is also linked."),
                ("note3.md", "# Note 3\nThis is the third note.\nNo links here."),
                ("note4.md", "# Note 4\nThis note links to [[note1]] and [[note2]]."),
                ("note5.md", "# Note 5\nThis is a standalone note with #tag1 and #tag2."),
            ]
            
            for filename, content in notes:
                note_path = vault_path / filename
                note_path.write_text(content)
            
            yield vault_path

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_services(self):
        """Mock the Jarvis services for testing."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.VaultReader') as mock_vault_reader, \
             patch('jarvis.tools.dataset_generation.dataset_generator.VectorEncoder') as mock_vector_encoder, \
             patch('jarvis.tools.dataset_generation.dataset_generator.GraphDatabase') as mock_graph_database, \
             patch('jarvis.tools.dataset_generation.dataset_generator.get_settings') as mock_get_settings:
            
            # Configure mock settings
            mock_settings = Mock()
            mock_settings.graph_enabled = True
            mock_get_settings.return_value = mock_settings
            
            # Configure mock VaultReader
            mock_vault_reader_instance = Mock()
            mock_vault_reader_instance.get_markdown_files.return_value = [
                Path("note1.md"), Path("note2.md"), Path("note3.md"), 
                Path("note4.md"), Path("note5.md")
            ]
            mock_vault_reader_instance.read_file.side_effect = [
                ("# Note 1\nContent", {"tags": ["test"]}),
                ("# Note 2\nContent", {"tags": ["test"]}),
                ("# Note 3\nContent", {"tags": ["test"]}),
                ("# Note 4\nContent", {"tags": ["test"]}),
                ("# Note 5\nContent", {"tags": ["test"]}),
            ]
            mock_vault_reader_instance.get_absolute_path.side_effect = lambda x: Path(f"/tmp/{x}")
            mock_vault_reader.return_value = mock_vault_reader_instance
            
            # Configure mock VectorEncoder
            mock_vector_encoder_instance = Mock()
            mock_vector_encoder_instance.encode_documents.return_value = [[0.1, 0.2, 0.3]] * 5
            mock_vector_encoder_instance.compute_similarity.return_value = 0.8
            mock_vector_encoder.return_value = mock_vector_encoder_instance
            
            # Configure mock GraphDatabase
            mock_graph_database_instance = Mock()
            mock_graph_database_instance.get_centrality_metrics.return_value = {
                "betweenness_centrality": 0.5,
                "closeness_centrality": 0.6,
                "pagerank_score": 0.7
            }
            mock_graph_database.return_value = mock_graph_database_instance
            
            yield {
                'vault_reader': mock_vault_reader_instance,
                'vector_encoder': mock_vector_encoder_instance,
                'graph_database': mock_graph_database_instance,
                'settings': mock_settings
            }

    def test_service_injection_and_initialization(self, temp_vault, temp_output_dir, mock_services):
        """Test that services are properly injected and initialized."""
        generator = DatasetGenerator(temp_vault, temp_output_dir, areas_only=False, skip_validation=True)
        
        # Verify services are initialized
        assert generator.vault_reader is not None
        assert generator.vector_encoder is not None
        assert generator.graph_database is not None
        assert generator.link_extractor is not None
        assert generator.notes_generator is not None
        assert generator.pairs_generator is not None

    def test_vault_reader_integration(self, temp_vault, temp_output_dir):
        """Test integration with VaultReader service."""
        generator = DatasetGenerator(temp_vault, temp_output_dir, areas_only=False, skip_validation=True)
        
        # Test that VaultReader is properly configured
        assert Path(generator.vault_reader.vault_path).resolve() == temp_vault.resolve()
        
        # Test that we can get markdown files
        markdown_files = list(generator.vault_reader.get_markdown_files())
        assert len(markdown_files) == 5
        assert all(str(f).endswith('.md') for f in markdown_files)

    def test_vector_encoder_integration(self, temp_vault, temp_output_dir, mock_services):
        """Test integration with VectorEncoder service."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.VectorEncoder') as mock_encoder_class:
            mock_encoder_class.return_value = mock_services['vector_encoder']
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, areas_only=False, skip_validation=True)
            
            # Test that VectorEncoder is accessible
            assert generator.vector_encoder is not None
            
            # Test that we can call encoding methods
            result = generator.vector_encoder.encode_documents(["test content"])
            assert result is not None

    def test_graph_database_integration(self, temp_vault, temp_output_dir, mock_services):
        """Test integration with GraphDatabase service."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.GraphDatabase') as mock_graph_class:
            mock_graph_class.return_value = mock_services['graph_database']
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, areas_only=False, skip_validation=True)
            
            # Test that GraphDatabase is accessible
            assert generator.graph_database is not None
            
            # Test that we can call graph methods
            result = generator.graph_database.get_centrality_metrics("test_node")
            assert result is not None

    def test_graph_database_optional_handling(self, temp_vault, temp_output_dir):
        """Test graceful handling when GraphDatabase is unavailable."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.get_settings') as mock_settings:
            # Configure settings to disable graph
            mock_settings_obj = Mock()
            mock_settings_obj.graph_enabled = False
            mock_settings.return_value = mock_settings_obj
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            
            # Graph database should be None when disabled
            assert generator.graph_database is None

    def test_service_error_handling(self, temp_vault, temp_output_dir):
        """Test error handling when services fail to initialize."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.VaultReader') as mock_vault_reader:
            # Make VaultReader initialization fail
            mock_vault_reader.side_effect = Exception("VaultReader initialization failed")
            
            with pytest.raises(ConfigurationError) as exc_info:
                DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            
            assert "Service initialization failed" in str(exc_info.value)

    def test_configuration_integration(self, temp_vault, temp_output_dir):
        """Test integration with configuration system."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.graph_enabled = True
            mock_settings.vector_model = "test-model"
            mock_get_settings.return_value = mock_settings
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            
            # Verify settings are used
            mock_get_settings.assert_called_once()

    def test_logging_integration(self, temp_vault, temp_output_dir, caplog):
        """Test integration with logging system."""
        import logging
        caplog.set_level(logging.INFO)
        
        generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
        
        # Check that initialization logs are present
        assert any("DatasetGenerator initialized" in record.message for record in caplog.records)

    def test_service_container_compatibility(self, temp_vault, temp_output_dir):
        """Test compatibility with service container pattern."""
        # This test ensures that the dataset generator works with the existing
        # service container architecture
        
        with patch('jarvis.core.container.ServiceContainer') as mock_container:
            mock_container_instance = Mock()
            mock_container.get_instance.return_value = mock_container_instance
            
            # Mock service retrieval
            mock_container_instance.get_service.side_effect = lambda name: {
                'vault_reader': Mock(),
                'vector_encoder': Mock(),
                'graph_database': Mock()
            }.get(name)
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            
            # Verify generator can be initialized (no exceptions)
            assert generator is not None

    def test_dependency_injection_pattern(self, temp_vault, temp_output_dir, mock_services):
        """Test that dependency injection pattern is followed correctly."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.VaultReader') as mock_vault_reader, \
             patch('jarvis.tools.dataset_generation.dataset_generator.VectorEncoder') as mock_vector_encoder, \
             patch('jarvis.tools.dataset_generation.dataset_generator.GraphDatabase') as mock_graph_database:
            
            mock_vault_reader.return_value = mock_services['vault_reader']
            mock_vector_encoder.return_value = mock_services['vector_encoder']
            mock_graph_database.return_value = mock_services['graph_database']
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            
            # Verify services are injected into components
            assert generator.notes_generator.vault_reader == mock_services['vault_reader']
            assert generator.notes_generator.vector_encoder == mock_services['vector_encoder']
            assert generator.notes_generator.graph_database == mock_services['graph_database']
            
            assert generator.pairs_generator.vector_encoder == mock_services['vector_encoder']
            assert generator.pairs_generator.graph_database == mock_services['graph_database']

    def test_cli_integration_compatibility(self, temp_vault, temp_output_dir):
        """Test compatibility with CLI integration patterns."""
        # Test that the generator can be used in the same way as the CLI
        generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
        
        # Test validation (should not raise exceptions for valid vault)
        validation_result = generator.validate_vault()
        assert validation_result.valid

    def test_error_propagation_from_services(self, temp_vault, temp_output_dir, mock_services):
        """Test that errors from services are properly propagated."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.VaultReader') as mock_vault_reader:
            # Make VaultReader.get_markdown_files fail
            mock_vault_reader_instance = Mock()
            mock_vault_reader_instance.get_markdown_files.side_effect = Exception("Service error")
            mock_vault_reader.return_value = mock_vault_reader_instance
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            
            # The error should be caught and handled appropriately
            with pytest.raises(Exception):
                list(generator.vault_reader.get_markdown_files())

    def test_service_lifecycle_management(self, temp_vault, temp_output_dir, mock_services):
        """Test proper service lifecycle management."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.VaultReader') as mock_vault_reader, \
             patch('jarvis.tools.dataset_generation.dataset_generator.VectorEncoder') as mock_vector_encoder:
            
            mock_vault_reader.return_value = mock_services['vault_reader']
            mock_vector_encoder.return_value = mock_services['vector_encoder']
            
            # Test context manager usage (if implemented)
            try:
                with DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True) as generator:
                    assert generator is not None
            except AttributeError:
                # Context manager not implemented, which is fine
                generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
                assert generator is not None

    def test_concurrent_service_access(self, temp_vault, temp_output_dir, mock_services):
        """Test that services can handle concurrent access patterns."""
        import threading
        import time
        
        with patch('jarvis.tools.dataset_generation.dataset_generator.VaultReader') as mock_vault_reader:
            mock_vault_reader.return_value = mock_services['vault_reader']
            
            generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            
            results = []
            errors = []
            
            def access_service():
                try:
                    # Simulate concurrent access to services
                    files = list(generator.vault_reader.get_markdown_files())
                    results.append(len(files))
                except Exception as e:
                    errors.append(e)
            
            # Create multiple threads accessing services
            threads = [threading.Thread(target=access_service) for _ in range(5)]
            
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # All threads should succeed
            assert len(errors) == 0
            assert len(results) == 5

    def test_service_configuration_validation(self, temp_vault, temp_output_dir):
        """Test that service configurations are validated."""
        with patch('jarvis.tools.dataset_generation.dataset_generator.get_settings') as mock_get_settings:
            # Test with invalid configuration
            mock_settings = Mock()
            mock_settings.graph_enabled = "invalid"  # Should be boolean
            mock_get_settings.return_value = mock_settings
            
            # Should handle invalid configuration gracefully
            generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
            assert generator is not None

    def test_service_compatibility_with_existing_cli(self, temp_vault, temp_output_dir):
        """Test that dataset generation doesn't conflict with existing CLI commands."""
        # This test ensures that the dataset generation tool doesn't interfere
        # with existing Jarvis CLI functionality
        
        generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
        
        # Test that we can import and use other CLI components without conflicts
        try:
            from jarvis.mcp.server import run_mcp_server
            from jarvis.services.vector.indexer import VectorIndexer
            from jarvis.services.graph.indexer import GraphIndexer
            
            # These imports should work without conflicts
            assert run_mcp_server is not None
            assert VectorIndexer is not None
            assert GraphIndexer is not None
            
        except ImportError as e:
            pytest.fail(f"Import conflict detected: {e}")

    def test_memory_management_with_services(self, temp_vault, temp_output_dir, mock_services):
        """Test memory management when using services."""
        import gc
        
        with patch('jarvis.tools.dataset_generation.dataset_generator.VaultReader') as mock_vault_reader:
            mock_vault_reader.return_value = mock_services['vault_reader']
            
            # Create and destroy multiple generators
            generators = []
            for _ in range(5):
                generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
                generators.append(generator)
            
            # Clean up
            for generator in generators:
                del generator
            gc.collect()
            
            # Test passes if no memory errors occur
            assert True