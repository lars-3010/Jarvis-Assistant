"""
Unit tests for semantic similarity fix in dataset generation.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.tools.dataset_generation.models.data_models import NoteData


class TestSemanticSimilarityFix:
    """Test that semantic similarity computation works correctly."""
    
    def create_test_vault(self):
        """Create a minimal test vault."""
        vault_dir = Path(tempfile.mkdtemp())
        areas_dir = vault_dir / "Areas" / "Test"
        areas_dir.mkdir(parents=True)
        
        # Create test notes with different content
        test_notes = [
            ("ai.md", "# Artificial Intelligence\nMachine learning and neural networks are key technologies."),
            ("data.md", "# Data Science\nStatistics and Python programming for data analysis."),
            ("weather.md", "# Weather Report\nToday is sunny and warm outside.")
        ]
        
        for filename, content in test_notes:
            (areas_dir / filename).write_text(content)
        
        return vault_dir
    
    def test_embeddings_generated_in_pairs_data_loading(self):
        """Test that embeddings are generated when loading notes data for pairs."""
        vault_path = self.create_test_vault()
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize dataset generator
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            # Get note paths
            all_notes = [str(path) for path in generator.vault_reader.get_markdown_files()]
            assert len(all_notes) == 3
            
            # Load notes data (this should generate embeddings)
            notes_data = generator._load_notes_data_for_pairs(all_notes)
            
            # Verify all notes have embeddings
            assert len(notes_data) == 3
            for note_path, note_data in notes_data.items():
                assert note_data.embedding is not None, f"No embedding for {note_path}"
                assert isinstance(note_data.embedding, np.ndarray), f"Invalid embedding type for {note_path}"
                assert note_data.embedding.shape[0] > 0, f"Empty embedding for {note_path}"
                assert not np.all(note_data.embedding == 0), f"Zero embedding for {note_path}"
                
        finally:
            import shutil
            shutil.rmtree(vault_path)
            shutil.rmtree(output_dir)
    
    def test_semantic_similarity_computation(self):
        """Test that semantic similarity returns non-zero values."""
        vault_path = self.create_test_vault()
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize dataset generator
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            # Load notes data
            all_notes = [str(path) for path in generator.vault_reader.get_markdown_files()]
            notes_data = generator._load_notes_data_for_pairs(all_notes)
            
            # Test similarity computation
            from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import PairsDatasetGenerator
            pairs_gen = PairsDatasetGenerator(generator.vector_encoder, None)
            
            note_paths = list(notes_data.keys())
            
            # Test similarity between AI and Data Science notes (should be > 0)
            ai_note = notes_data[note_paths[0]]
            data_note = notes_data[note_paths[1]]
            similarity_related = pairs_gen._compute_semantic_similarity(ai_note, data_note)
            
            # Test similarity between AI and Weather notes (should be lower but > 0)
            weather_note = notes_data[note_paths[2]]
            similarity_unrelated = pairs_gen._compute_semantic_similarity(ai_note, weather_note)
            
            # Assertions
            assert similarity_related != 0.0, "Similarity between related notes should not be 0"
            assert similarity_unrelated != 0.0, "Similarity between unrelated notes should not be 0"
            assert -1.0 <= similarity_related <= 1.0, "Similarity should be in valid range"
            assert -1.0 <= similarity_unrelated <= 1.0, "Similarity should be in valid range"
            
            # Both similarities should be reasonable values (the exact ordering may vary)
            # The key point is that they're not zero and are in valid ranges
            print(f"Similarity AI-Data: {similarity_related:.6f}")
            print(f"Similarity AI-Weather: {similarity_unrelated:.6f}")
            
        finally:
            import shutil
            shutil.rmtree(vault_path)
            shutil.rmtree(output_dir)
    
    def test_embedding_batch_generation_efficiency(self):
        """Test that embeddings are generated in batch for efficiency."""
        vault_path = self.create_test_vault()
        output_dir = Path(tempfile.mkdtemp())
        
        try:
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=output_dir,
                areas_only=True,
                skip_validation=True
            )
            
            # Mock the vector encoder to track calls
            with patch.object(generator.vector_encoder, 'encode_documents') as mock_encode:
                # Set up mock to return valid embeddings
                mock_encode.return_value = np.random.rand(3, 384).astype(np.float32)
                
                all_notes = [str(path) for path in generator.vault_reader.get_markdown_files()]
                notes_data = generator._load_notes_data_for_pairs(all_notes)
                
                # Should be called once for batch processing, not once per note
                assert mock_encode.call_count == 1, f"Expected 1 batch call, got {mock_encode.call_count}"
                
                # Verify the call was made with all contents
                call_args = mock_encode.call_args[0][0]  # First positional argument
                assert len(call_args) == 3, "Should process all 3 notes in one batch"
                
        finally:
            import shutil
            shutil.rmtree(vault_path)
            shutil.rmtree(output_dir)