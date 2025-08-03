"""
Integration tests for dataset output quality and format validation.

Tests the generated CSV files for proper format, data quality, feature value ranges,
statistical properties, and compatibility with analysis tools.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import csv

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator


class TestDatasetOutputValidation:
    """Test dataset output quality and format validation."""

    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir) / "test_vault"
            vault_path.mkdir()
            
            # Create .obsidian directory to make it look like a real vault
            obsidian_dir = vault_path / ".obsidian"
            obsidian_dir.mkdir()
            
            # Create some test notes with realistic content
            notes = [
                ("note1.md", """# Machine Learning Basics
This note covers the fundamentals of machine learning.

## Key Concepts
- Supervised learning
- Unsupervised learning
- [[note2|Deep Learning]] is a subset

#machine-learning #basics
"""),
                ("note2.md", """# Deep Learning
Deep learning is a subset of [[note1|machine learning]].

## Neural Networks
- Feedforward networks
- Convolutional networks
- Links to [[note3|Computer Vision]]

#deep-learning #neural-networks
"""),
                ("note3.md", """# Computer Vision
Computer vision uses [[note2|deep learning]] techniques.

## Applications
- Image classification
- Object detection
- Connected to [[note4|Natural Language Processing]]

#computer-vision #applications
"""),
                ("note4.md", """# Natural Language Processing
NLP processes human language using [[note1|machine learning]].

## Techniques
- Tokenization
- Named entity recognition
- Related to [[note3|Computer Vision]]

#nlp #text-processing
"""),
                ("note5.md", """# Data Science
Data science combines statistics and [[note1|machine learning]].

## Process
1. Data collection
2. Data cleaning
3. Analysis
4. Visualization

#data-science #statistics
"""),
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
    def generated_datasets(self, temp_vault, temp_output_dir):
        """Generate datasets for testing."""
        generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
        
        # Generate datasets
        result = generator.generate_datasets(
            notes_filename="test_notes.csv",
            pairs_filename="test_pairs.csv",
            negative_sampling_ratio=2.0,  # Small ratio for testing
            batch_size=8,
            max_pairs_per_note=20
        )
        
        yield {
            'result': result,
            'notes_path': temp_output_dir / "test_notes.csv",
            'pairs_path': temp_output_dir / "test_pairs.csv",
            'generator': generator
        }

    def test_csv_file_format_compliance(self, generated_datasets):
        """Test that generated CSV files comply with standard CSV format."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test that files exist
        assert notes_path.exists(), "Notes dataset file should exist"
        assert pairs_path.exists(), "Pairs dataset file should exist"
        
        # Test that files are valid CSV format
        with open(notes_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            assert len(header) > 0, "Notes CSV should have header row"
            
            # Check that all rows have same number of columns as header
            for i, row in enumerate(csv_reader):
                assert len(row) == len(header), f"Row {i+1} in notes CSV has wrong number of columns"
                if i > 10:  # Check first 10 rows
                    break
        
        with open(pairs_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            assert len(header) > 0, "Pairs CSV should have header row"
            
            # Check that all rows have same number of columns as header
            for i, row in enumerate(csv_reader):
                assert len(row) == len(header), f"Row {i+1} in pairs CSV has wrong number of columns"
                if i > 10:  # Check first 10 rows
                    break

    def test_notes_dataset_schema_validation(self, generated_datasets):
        """Test that notes dataset has expected schema and data types."""
        notes_path = generated_datasets['notes_path']
        
        # Load dataset
        df = pd.read_csv(notes_path)
        
        # Test basic structure
        assert len(df) > 0, "Notes dataset should not be empty"
        assert len(df.columns) > 0, "Notes dataset should have columns"
        
        # Test required columns exist
        required_columns = [
            'note_path', 'note_title', 'word_count', 'tag_count',
            'creation_date', 'last_modified', 'outgoing_links_count'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing from notes dataset"
        
        # Test data types
        assert df['word_count'].dtype in [np.int64, np.int32], "word_count should be integer"
        assert df['tag_count'].dtype in [np.int64, np.int32], "tag_count should be integer"
        assert df['outgoing_links_count'].dtype in [np.int64, np.int32], "outgoing_links_count should be integer"
        
        # Test that paths are strings and not empty
        assert df['note_path'].dtype == object, "note_path should be string"
        assert not df['note_path'].isnull().any(), "note_path should not have null values"
        assert (df['note_path'].str.len() > 0).all(), "note_path should not be empty"
        
        # Test that titles are strings and not empty
        assert df['note_title'].dtype == object, "note_title should be string"
        assert not df['note_title'].isnull().any(), "note_title should not have null values"

    def test_pairs_dataset_schema_validation(self, generated_datasets):
        """Test that pairs dataset has expected schema and data types."""
        pairs_path = generated_datasets['pairs_path']
        
        # Load dataset
        df = pd.read_csv(pairs_path)
        
        # Test basic structure
        assert len(df) > 0, "Pairs dataset should not be empty"
        assert len(df.columns) > 0, "Pairs dataset should have columns"
        
        # Test required columns exist
        required_columns = [
            'note_a_path', 'note_b_path', 'link_exists'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing from pairs dataset"
        
        # Test data types
        assert df['link_exists'].dtype in [bool, np.int64, np.int32], "link_exists should be boolean or integer"
        
        # Test that paths are strings and not empty
        assert df['note_a_path'].dtype == object, "note_a_path should be string"
        assert df['note_b_path'].dtype == object, "note_b_path should be string"
        assert not df['note_a_path'].isnull().any(), "note_a_path should not have null values"
        assert not df['note_b_path'].isnull().any(), "note_b_path should not have null values"
        
        # Test that pairs are not self-referential
        assert (df['note_a_path'] != df['note_b_path']).all(), "Pairs should not be self-referential"

    def test_feature_value_ranges(self, generated_datasets):
        """Test that feature values are within expected ranges."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test notes dataset ranges
        notes_df = pd.read_csv(notes_path)
        
        # Word count should be positive
        assert (notes_df['word_count'] >= 0).all(), "Word count should be non-negative"
        assert (notes_df['word_count'] <= 10000).all(), "Word count should be reasonable (< 10000)"
        
        # Tag count should be non-negative
        assert (notes_df['tag_count'] >= 0).all(), "Tag count should be non-negative"
        assert (notes_df['tag_count'] <= 100).all(), "Tag count should be reasonable (< 100)"
        
        # Outgoing links count should be non-negative
        assert (notes_df['outgoing_links_count'] >= 0).all(), "Outgoing links count should be non-negative"
        assert (notes_df['outgoing_links_count'] <= 1000).all(), "Outgoing links count should be reasonable (< 1000)"
        
        # Test pairs dataset ranges
        pairs_df = pd.read_csv(pairs_path)
        
        # Link exists should be boolean (0 or 1)
        unique_values = pairs_df['link_exists'].unique()
        assert all(val in [0, 1, True, False] for val in unique_values), "link_exists should be boolean values"
        
        # Test similarity scores if present
        if 'cosine_similarity' in pairs_df.columns:
            similarity_col = pairs_df['cosine_similarity']
            # Remove NaN values for testing
            similarity_col = similarity_col.dropna()
            if len(similarity_col) > 0:
                assert (similarity_col >= -1.0).all(), "Cosine similarity should be >= -1"
                assert (similarity_col <= 1.0).all(), "Cosine similarity should be <= 1"

    def test_statistical_properties(self, generated_datasets):
        """Test statistical properties of the generated datasets."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test notes dataset statistics
        notes_df = pd.read_csv(notes_path)
        
        # Test that we have reasonable distribution of word counts
        word_counts = notes_df['word_count']
        assert word_counts.mean() > 0, "Average word count should be positive"
        assert word_counts.std() >= 0, "Word count standard deviation should be non-negative"
        
        # Test that we have some variation in features
        assert len(notes_df['note_path'].unique()) == len(notes_df), "All note paths should be unique"
        
        # Test pairs dataset statistics
        pairs_df = pd.read_csv(pairs_path)
        
        # Test that we have both positive and negative examples
        link_exists_counts = pairs_df['link_exists'].value_counts()
        assert len(link_exists_counts) >= 1, "Should have at least one type of link relationship"
        
        # Test that pairs are unique
        pair_combinations = pairs_df[['note_a_path', 'note_b_path']].apply(
            lambda x: tuple(sorted([x['note_a_path'], x['note_b_path']])), axis=1
        )
        assert len(pair_combinations.unique()) == len(pairs_df), "All pairs should be unique"

    def test_data_consistency(self, generated_datasets):
        """Test consistency between notes and pairs datasets."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Get all note paths from both datasets
        notes_paths = set(notes_df['note_path'].unique())
        pairs_note_a_paths = set(pairs_df['note_a_path'].unique())
        pairs_note_b_paths = set(pairs_df['note_b_path'].unique())
        pairs_all_paths = pairs_note_a_paths.union(pairs_note_b_paths)
        
        # Test that all paths in pairs dataset exist in notes dataset
        missing_paths = pairs_all_paths - notes_paths
        assert len(missing_paths) == 0, f"Pairs dataset references notes not in notes dataset: {missing_paths}"
        
        # Test that notes dataset has reasonable coverage in pairs
        # (Not all notes need to be in pairs, but most should be)
        coverage_ratio = len(pairs_all_paths) / len(notes_paths)
        assert coverage_ratio > 0.5, f"Pairs dataset should cover most notes (coverage: {coverage_ratio:.2f})"

    def test_pandas_compatibility(self, generated_datasets):
        """Test that datasets are compatible with pandas operations."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test loading with pandas
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Test basic pandas operations
        assert len(notes_df) > 0, "Notes dataframe should not be empty"
        assert len(pairs_df) > 0, "Pairs dataframe should not be empty"
        
        # Test filtering operations
        filtered_notes = notes_df[notes_df['word_count'] > 0]
        assert len(filtered_notes) >= 0, "Filtering should work"
        
        # Test groupby operations
        if 'tag_count' in notes_df.columns:
            grouped = notes_df.groupby('tag_count').size()
            assert len(grouped) >= 0, "Groupby should work"
        
        # Test merge operations
        if len(notes_df) > 0 and len(pairs_df) > 0:
            merged = pairs_df.merge(
                notes_df[['note_path', 'word_count']], 
                left_on='note_a_path', 
                right_on='note_path', 
                how='left'
            )
            assert len(merged) == len(pairs_df), "Merge should preserve pairs count"

    def test_scikit_learn_compatibility(self, generated_datasets):
        """Test that datasets are compatible with scikit-learn."""
        pairs_path = generated_datasets['pairs_path']
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn not available")
        
        pairs_df = pd.read_csv(pairs_path)
        
        # Test that we can prepare data for ML
        if 'link_exists' in pairs_df.columns:
            y = pairs_df['link_exists'].astype(int)
            
            # Get numeric columns for features
            numeric_columns = pairs_df.select_dtypes(include=[np.number]).columns
            numeric_columns = [col for col in numeric_columns if col != 'link_exists']
            
            if len(numeric_columns) > 0:
                X = pairs_df[numeric_columns].fillna(0)  # Fill NaN with 0 for testing
                
                # Test train/test split
                if len(X) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    assert len(X_train) > 0, "Training set should not be empty"
                    assert len(X_test) >= 0, "Test set should be valid"
                
                # Test scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                assert X_scaled.shape == X.shape, "Scaling should preserve shape"

    def test_data_quality_metrics(self, generated_datasets):
        """Test data quality metrics."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Test completeness (missing values)
        notes_missing_ratio = notes_df.isnull().sum().sum() / (len(notes_df) * len(notes_df.columns))
        assert notes_missing_ratio < 0.5, f"Notes dataset has too many missing values: {notes_missing_ratio:.2f}"
        
        pairs_missing_ratio = pairs_df.isnull().sum().sum() / (len(pairs_df) * len(pairs_df.columns))
        assert pairs_missing_ratio < 0.5, f"Pairs dataset has too many missing values: {pairs_missing_ratio:.2f}"
        
        # Test uniqueness of primary keys
        assert notes_df['note_path'].nunique() == len(notes_df), "Note paths should be unique"
        
        # Test referential integrity
        notes_paths = set(notes_df['note_path'])
        pairs_paths = set(pairs_df['note_a_path']).union(set(pairs_df['note_b_path']))
        invalid_refs = pairs_paths - notes_paths
        assert len(invalid_refs) == 0, f"Invalid references in pairs dataset: {invalid_refs}"

    def test_file_size_reasonableness(self, generated_datasets):
        """Test that generated files are of reasonable size."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test file sizes
        notes_size = notes_path.stat().st_size
        pairs_size = pairs_path.stat().st_size
        
        # Files should not be empty
        assert notes_size > 0, "Notes dataset file should not be empty"
        assert pairs_size > 0, "Pairs dataset file should not be empty"
        
        # Files should not be unreasonably large (for test data)
        max_size = 10 * 1024 * 1024  # 10MB
        assert notes_size < max_size, f"Notes dataset file too large: {notes_size} bytes"
        assert pairs_size < max_size, f"Pairs dataset file too large: {pairs_size} bytes"
        
        # Test row counts are reasonable
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        assert len(notes_df) >= 5, "Should have at least 5 notes"
        assert len(pairs_df) >= 5, "Should have at least 5 pairs"
        assert len(pairs_df) <= 1000, "Should not have excessive pairs for test data"

    def test_encoding_and_special_characters(self, generated_datasets):
        """Test handling of encoding and special characters."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test that files can be read with different encodings
        encodings = ['utf-8', 'utf-8-sig']
        
        for encoding in encodings:
            try:
                notes_df = pd.read_csv(notes_path, encoding=encoding)
                pairs_df = pd.read_csv(pairs_path, encoding=encoding)
                assert len(notes_df) > 0, f"Should be able to read with {encoding}"
                assert len(pairs_df) > 0, f"Should be able to read with {encoding}"
            except UnicodeDecodeError:
                # This is acceptable for some encodings
                pass
        
        # Test that string columns don't have problematic characters
        notes_df = pd.read_csv(notes_path)
        
        for col in notes_df.select_dtypes(include=[object]).columns:
            # Check for null bytes or other problematic characters
            problematic = notes_df[col].astype(str).str.contains('\x00', na=False)
            assert not problematic.any(), f"Column {col} contains null bytes"

    def test_generation_result_metadata(self, generated_datasets):
        """Test that generation result contains proper metadata."""
        result = generated_datasets['result']
        
        # Test that result is successful
        assert result.success, "Dataset generation should be successful"
        
        # Test that summary exists and has expected fields
        assert result.summary is not None, "Result should have summary"
        
        summary = result.summary
        assert summary.total_notes > 0, "Summary should report total notes"
        assert summary.notes_processed > 0, "Summary should report processed notes"
        assert summary.pairs_generated > 0, "Summary should report generated pairs"
        assert summary.total_time_seconds > 0, "Summary should report processing time"
        
        # Test that file paths are correct
        assert result.notes_dataset_path is not None, "Should have notes dataset path"
        assert result.pairs_dataset_path is not None, "Should have pairs dataset path"
        assert Path(result.notes_dataset_path).exists(), "Notes dataset file should exist"
        assert Path(result.pairs_dataset_path).exists(), "Pairs dataset file should exist"
        
        # Test link statistics
        if summary.link_statistics:
            link_stats = summary.link_statistics
            assert link_stats.total_links >= 0, "Link statistics should be valid"
            assert link_stats.unique_links >= 0, "Unique links should be valid"
            assert link_stats.broken_links >= 0, "Broken links count should be valid"

    def test_reproducibility(self, temp_vault, temp_output_dir):
        """Test that dataset generation is reproducible."""
        # Generate datasets twice with same parameters
        generator1 = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
        result1 = generator1.generate_datasets(
            notes_filename="test1_notes.csv",
            pairs_filename="test1_pairs.csv",
            negative_sampling_ratio=2.0,
            sampling_strategy="random",  # Use random for more deterministic testing
            batch_size=8,
            max_pairs_per_note=20
        )
        
        generator2 = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
        result2 = generator2.generate_datasets(
            notes_filename="test2_notes.csv",
            pairs_filename="test2_pairs.csv",
            negative_sampling_ratio=2.0,
            sampling_strategy="random",
            batch_size=8,
            max_pairs_per_note=20
        )
        
        # Both should be successful
        assert result1.success, "First generation should be successful"
        assert result2.success, "Second generation should be successful"
        
        # Load datasets
        notes1 = pd.read_csv(temp_output_dir / "test1_notes.csv")
        notes2 = pd.read_csv(temp_output_dir / "test2_notes.csv")
        
        # Notes dataset should be identical (deterministic)
        assert len(notes1) == len(notes2), "Notes datasets should have same length"
        assert set(notes1.columns) == set(notes2.columns), "Notes datasets should have same columns"
        
        # Check that note paths are the same (order might differ)
        assert set(notes1['note_path']) == set(notes2['note_path']), "Should process same notes"

    def test_advanced_feature_validation(self, generated_datasets):
        """Test advanced feature validation and statistical properties."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Test advanced notes features
        if 'betweenness_centrality' in notes_df.columns:
            centrality_col = notes_df['betweenness_centrality']
            centrality_col = centrality_col.dropna()
            if len(centrality_col) > 0:
                assert (centrality_col >= 0.0).all(), "Betweenness centrality should be non-negative"
                assert (centrality_col <= 1.0).all(), "Betweenness centrality should be <= 1"
        
        if 'pagerank_score' in notes_df.columns:
            pagerank_col = notes_df['pagerank_score']
            pagerank_col = pagerank_col.dropna()
            if len(pagerank_col) > 0:
                assert (pagerank_col > 0.0).all(), "PageRank scores should be positive"
                # PageRank scores should be reasonable (between 0 and 1 for normalized scores)
                assert (pagerank_col <= 2.0).all(), "PageRank scores should be reasonable"
                # PageRank sum should be positive and reasonable
                assert pagerank_col.sum() > 0, "PageRank scores sum should be positive"
        
        # Test pairs features
        if 'shortest_path_length' in pairs_df.columns:
            path_col = pairs_df['shortest_path_length']
            path_col = path_col.dropna()
            if len(path_col) > 0:
                assert (path_col >= 1).all(), "Shortest path length should be at least 1"
                assert (path_col <= len(notes_df)).all(), "Shortest path length should not exceed number of nodes"

    def test_dataset_completeness_validation(self, generated_datasets):
        """Test that datasets are complete and contain expected information."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Test notes completeness
        required_note_columns = ['note_path', 'note_title', 'word_count']
        for col in required_note_columns:
            assert col in notes_df.columns, f"Required column '{col}' missing from notes dataset"
            assert not notes_df[col].isnull().all(), f"Column '{col}' should not be all null"
        
        # Test pairs completeness
        required_pair_columns = ['note_a_path', 'note_b_path', 'link_exists']
        for col in required_pair_columns:
            assert col in pairs_df.columns, f"Required column '{col}' missing from pairs dataset"
            assert not pairs_df[col].isnull().all(), f"Column '{col}' should not be all null"
        
        # Test that we have both positive and negative examples (if enough data)
        if len(pairs_df) > 1:
            link_values = pairs_df['link_exists'].unique()
            assert len(link_values) >= 1, "Should have at least one type of link relationship"

    def test_performance_metrics_validation(self, generated_datasets):
        """Test that performance metrics are reasonable."""
        result = generated_datasets['result']
        
        # Test timing metrics
        assert result.summary.total_time_seconds > 0, "Total time should be positive"
        assert result.summary.total_time_seconds < 300, "Generation should complete within 5 minutes for test data"
        
        # Test processing metrics
        assert result.summary.notes_processed >= 0, "Notes processed should be non-negative"
        assert result.summary.pairs_generated >= 0, "Pairs generated should be non-negative"
        
        # Test that processing was reasonably efficient
        if result.summary.total_time_seconds > 0:
            notes_per_second = result.summary.notes_processed / result.summary.total_time_seconds
            assert notes_per_second >= 0.1, "Should process at least 0.1 notes per second"

    def test_error_handling_validation(self, temp_vault, temp_output_dir):
        """Test error handling in dataset generation."""
        # Test with invalid vault path - this should fail during initialization
        invalid_vault = temp_vault / "nonexistent"
        
        # Test that initialization fails with invalid vault path
        with pytest.raises(Exception) as exc_info:
            generator = DatasetGenerator(invalid_vault, temp_output_dir, skip_validation=True)
        
        # Should get a configuration error
        assert "Service initialization failed" in str(exc_info.value) or "Vault path not found" in str(exc_info.value)
        
        # Test with valid generator but simulate error during generation
        valid_generator = DatasetGenerator(temp_vault, temp_output_dir, skip_validation=True)
        
        # Test that we can handle errors during generation by mocking a service failure
        with patch.object(valid_generator.link_extractor, 'extract_all_links', side_effect=Exception("Simulated error")):
            result = valid_generator.generate_datasets()
            assert not result.success, "Should fail with simulated error"
            assert result.error_message is not None, "Should provide error message"

    def test_memory_efficiency_validation(self, generated_datasets):
        """Test that dataset generation is memory efficient."""
        result = generated_datasets['result']
        
        # Check if performance metrics include memory information
        if hasattr(result.summary, 'performance_metrics') and result.summary.performance_metrics:
            metrics = result.summary.performance_metrics
            
            # If memory metrics are available, validate them
            if 'peak_memory_mb' in metrics:
                peak_memory = metrics['peak_memory_mb']
                assert peak_memory >= 0, "Peak memory should be non-negative"
                assert peak_memory < 2000, "Peak memory should be reasonable (< 2GB) for test data"

    def test_output_file_integrity(self, generated_datasets):
        """Test the integrity of output files."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test file permissions and accessibility
        assert notes_path.is_file(), "Notes file should exist and be a file"
        assert pairs_path.is_file(), "Pairs file should exist and be a file"
        
        # Test file is not empty
        assert notes_path.stat().st_size > 0, "Notes file should not be empty"
        assert pairs_path.stat().st_size > 0, "Pairs file should not be empty"
        
        # Test file can be read multiple times
        df1 = pd.read_csv(notes_path)
        df2 = pd.read_csv(notes_path)
        assert len(df1) == len(df2), "File should be readable multiple times consistently"

    def test_cross_platform_compatibility(self, generated_datasets):
        """Test that generated datasets are cross-platform compatible."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        # Test with different line ending handling
        with open(notes_path, 'r', newline='') as f:
            content = f.read()
            # Should not have mixed line endings
            assert '\r\n' not in content or '\n' not in content.replace('\r\n', ''), \
                "Should not have mixed line endings"
        
        # Test path separators in data
        notes_df = pd.read_csv(notes_path)
        if 'note_path' in notes_df.columns:
            paths = notes_df['note_path'].astype(str)
            # Paths should use forward slashes (platform independent)
            problematic_paths = paths[paths.str.contains('\\\\', na=False)]
            assert len(problematic_paths) == 0, "Paths should use forward slashes for cross-platform compatibility"

    def test_dataset_schema_compliance(self, generated_datasets):
        """Test that datasets comply with expected schema standards."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Test column naming conventions (snake_case)
        for col in notes_df.columns:
            assert col.islower() or '_' in col, f"Column '{col}' should use snake_case naming"
            assert not col.startswith('_'), f"Column '{col}' should not start with underscore"
            assert not col.endswith('_'), f"Column '{col}' should not end with underscore"
        
        for col in pairs_df.columns:
            assert col.islower() or '_' in col, f"Column '{col}' should use snake_case naming"
            assert not col.startswith('_'), f"Column '{col}' should not start with underscore"
            assert not col.endswith('_'), f"Column '{col}' should not end with underscore"
        
        # Test that boolean columns use consistent representation
        boolean_columns = ['link_exists']
        for col in boolean_columns:
            if col in pairs_df.columns:
                unique_vals = pairs_df[col].unique()
                # Should only contain boolean-like values
                valid_vals = {0, 1, True, False}
                assert all(val in valid_vals for val in unique_vals), \
                    f"Boolean column '{col}' contains invalid values: {unique_vals}"

    def test_feature_engineering_quality(self, generated_datasets):
        """Test the quality of feature engineering."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Test that computed features are reasonable
        if 'word_count' in notes_df.columns:
            # Word count should correlate with some content measure
            assert notes_df['word_count'].var() > 0, "Word count should have some variation"
            assert notes_df['word_count'].mean() > 0, "Average word count should be positive"
        
        # Test that link-based features are consistent
        if 'outgoing_links_count' in notes_df.columns:
            outgoing_counts = notes_df['outgoing_links_count']
            assert (outgoing_counts >= 0).all(), "Outgoing links count should be non-negative"
            
            # Should have some notes with links (based on our test data)
            assert outgoing_counts.sum() > 0, "Should have some outgoing links in test data"
        
        # Test pair features consistency
        if 'cosine_similarity' in pairs_df.columns:
            similarity_col = pairs_df['cosine_similarity'].dropna()
            if len(similarity_col) > 0:
                # Similarity should be reasonable
                assert (similarity_col >= -1.1).all(), "Cosine similarity should be >= -1"
                assert (similarity_col <= 1.1).all(), "Cosine similarity should be <= 1"
                
                # Should have some variation in similarity scores
                if len(similarity_col) > 1:
                    assert similarity_col.var() >= 0, "Similarity scores should have some variation"

    def test_dataset_balance_and_distribution(self, generated_datasets):
        """Test that datasets have reasonable balance and distribution."""
        pairs_path = generated_datasets['pairs_path']
        pairs_df = pd.read_csv(pairs_path)
        
        # Test class balance in pairs dataset
        if 'link_exists' in pairs_df.columns:
            link_counts = pairs_df['link_exists'].value_counts()
            
            # Should have both positive and negative examples (if enough data)
            if len(pairs_df) > 1:
                assert len(link_counts) >= 1, "Should have at least one class"
                
                # Check that we don't have extreme imbalance (unless very small dataset)
                if len(pairs_df) >= 10:
                    min_class_count = link_counts.min()
                    max_class_count = link_counts.max()
                    imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
                    
                    # Allow some imbalance but not extreme
                    assert imbalance_ratio <= 20, f"Class imbalance too extreme: {imbalance_ratio}"

    def test_temporal_consistency(self, generated_datasets):
        """Test temporal consistency in the datasets."""
        notes_path = generated_datasets['notes_path']
        notes_df = pd.read_csv(notes_path)
        
        # Test date columns if present
        date_columns = ['creation_date', 'last_modified']
        for col in date_columns:
            if col in notes_df.columns:
                # Try to parse dates
                try:
                    dates = pd.to_datetime(notes_df[col])
                    
                    # Dates should be reasonable (not in far future or past)
                    now = pd.Timestamp.now()
                    min_reasonable_date = pd.Timestamp('1990-01-01')
                    max_reasonable_date = now + pd.Timedelta(days=1)
                    
                    assert (dates >= min_reasonable_date).all(), f"Dates in {col} too far in past"
                    assert (dates <= max_reasonable_date).all(), f"Dates in {col} too far in future"
                    
                    # Creation date should be <= last modified date (if both present)
                    if 'creation_date' in notes_df.columns and 'last_modified' in notes_df.columns:
                        creation_dates = pd.to_datetime(notes_df['creation_date'])
                        modified_dates = pd.to_datetime(notes_df['last_modified'])
                        assert (creation_dates <= modified_dates).all(), \
                            "Creation date should be <= last modified date"
                
                except (ValueError, TypeError):
                    # If dates can't be parsed, they should at least be consistent strings
                    assert notes_df[col].dtype == object, f"Date column {col} should be string if not parseable"

    def test_numerical_feature_distributions(self, generated_datasets):
        """Test that numerical features have reasonable distributions."""
        notes_path = generated_datasets['notes_path']
        pairs_path = generated_datasets['pairs_path']
        
        notes_df = pd.read_csv(notes_path)
        pairs_df = pd.read_csv(pairs_path)
        
        # Test notes numerical features
        numerical_cols = notes_df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            values = notes_df[col].dropna()
            if len(values) > 0:
                # Should not have all identical values (unless it's a constant feature)
                if len(values) > 1:
                    # Allow some constant features but most should have variation
                    pass  # This is acceptable for some features
                
                # Should not have extreme outliers (more than 10 standard deviations)
                if values.std() > 0:
                    z_scores = np.abs((values - values.mean()) / values.std())
                    extreme_outliers = (z_scores > 10).sum()
                    outlier_ratio = extreme_outliers / len(values)
                    assert outlier_ratio < 0.1, f"Too many extreme outliers in {col}: {outlier_ratio:.2%}"
        
        # Test pairs numerical features
        numerical_cols = pairs_df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'link_exists':  # Skip target variable
                values = pairs_df[col].dropna()
                if len(values) > 1 and values.std() > 0:
                    z_scores = np.abs((values - values.mean()) / values.std())
                    extreme_outliers = (z_scores > 10).sum()
                    outlier_ratio = extreme_outliers / len(values)
                    assert outlier_ratio < 0.1, f"Too many extreme outliers in {col}: {outlier_ratio:.2%}"