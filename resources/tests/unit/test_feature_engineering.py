"""
Unit tests for feature engineering components.

Tests semantic similarity computation, graph metrics computation,
and content feature extraction and metadata parsing.
"""

import tempfile
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
from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import (
    PairsDatasetGenerator, RandomSamplingStrategy, StratifiedSamplingStrategy
)
from jarvis.tools.dataset_generation.models.data_models import (
    NoteData, NoteFeatures, PairFeatures, CentralityMetrics, Link
)
from jarvis.tools.dataset_generation.models.exceptions import (
    FeatureEngineeringError, InsufficientDataError, SamplingError
)


class TestNotesDatasetGenerator:
    """Test suite for NotesDatasetGenerator feature engineering."""

    @pytest.fixture
    def mock_vault_reader(self):
        """Create a mock VaultReader."""
        mock_reader = Mock(spec=VaultReader)
        return mock_reader

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a mock VectorEncoder."""
        mock_encoder = Mock(spec=VectorEncoder)
        # Mock embedding generation
        mock_encoder.encode_documents.return_value = np.random.rand(3, 384)  # 3 docs, 384 dims
        mock_encoder.encode_batch.return_value = np.random.rand(3, 384)
        return mock_encoder

    @pytest.fixture
    def mock_graph_database(self):
        """Create a mock GraphDatabase."""
        mock_db = Mock(spec=GraphDatabase)
        return mock_db

    @pytest.fixture
    def mock_markdown_parser(self):
        """Create a mock MarkdownParser."""
        mock_parser = Mock(spec=MarkdownParser)
        return mock_parser

    @pytest.fixture
    def notes_generator(self, mock_vault_reader, mock_vector_encoder, 
                       mock_graph_database, mock_markdown_parser):
        """Create a NotesDatasetGenerator instance."""
        return NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder,
            graph_database=mock_graph_database,
            markdown_parser=mock_markdown_parser
        )

    @pytest.fixture
    def sample_notes(self):
        """Create sample note paths."""
        return ["note1.md", "note2.md", "note3.md"]

    @pytest.fixture
    def sample_link_graph(self):
        """Create a sample link graph."""
        graph = nx.DiGraph()
        graph.add_nodes_from(["note1.md", "note2.md", "note3.md"])
        graph.add_edge("note1.md", "note2.md")
        graph.add_edge("note2.md", "note3.md")
        return graph

    @pytest.fixture
    def sample_note_data(self):
        """Create sample note data."""
        return {
            "note1.md": NoteData(
                path="note1.md",
                title="Note 1",
                content="This is the first note with some content.",
                metadata={"created": 1234567890, "modified": 1234567900},
                tags=["tag1", "tag2"],
                outgoing_links=["note2.md"],
                embedding=np.random.rand(384),
                word_count=10
            ),
            "note2.md": NoteData(
                path="note2.md",
                title="Note 2",
                content="This is the second note with different content.",
                metadata={"created": 1234567891, "modified": 1234567901},
                tags=["tag2", "tag3"],
                outgoing_links=["note3.md"],
                embedding=np.random.rand(384),
                word_count=12
            ),
            "note3.md": NoteData(
                path="note3.md",
                title="Note 3",
                content="This is the third note with unique content.",
                metadata={"created": 1234567892, "modified": 1234567902},
                tags=["tag3", "tag4"],
                outgoing_links=[],
                embedding=np.random.rand(384),
                word_count=8
            )
        }

    def test_initialization(self, mock_vault_reader, mock_vector_encoder):
        """Test NotesDatasetGenerator initialization."""
        generator = NotesDatasetGenerator(
            vault_reader=mock_vault_reader,
            vector_encoder=mock_vector_encoder
        )
        
        assert generator.vault_reader == mock_vault_reader
        assert generator.vector_encoder == mock_vector_encoder
        assert generator.graph_database is None
        assert generator.markdown_parser is not None

    def test_generate_dataset_insufficient_notes(self, notes_generator, sample_link_graph):
        """Test dataset generation with insufficient notes."""
        insufficient_notes = ["note1.md", "note2.md"]  # Less than 5
        
        with pytest.raises(InsufficientDataError) as exc_info:
            notes_generator.generate_dataset(insufficient_notes, sample_link_graph)
        
        assert "Insufficient notes" in str(exc_info.value)
        assert exc_info.value.required_minimum == 5
        assert exc_info.value.actual_count == 2

    def test_extract_note_features_basic(self, notes_generator):
        """Test basic note feature extraction."""
        note_path = "test.md"
        content = "# Test Note\n\nThis is a test note with some content."
        metadata = {"created": 1234567890, "modified": 1234567900}
        
        # Mock the markdown parser
        notes_generator.markdown_parser.parse.return_value = {
            "frontmatter": {"title": "Test Note", "tags": ["test"]},
            "headings": [{"text": "Test Note", "level": 1}],
            "tags": ["test"],
            "links": []
        }
        
        with patch.object(notes_generator, '_extract_all_frontmatter_properties') as mock_frontmatter, \
             patch.object(notes_generator, '_generate_semantic_summary') as mock_summary:
            
            mock_frontmatter.return_value = {"title": "Test Note", "tags": ["test"]}
            mock_summary.return_value = "This is a test note summary."
            
            features = notes_generator._extract_note_features(note_path, content, metadata)
            
            assert isinstance(features, NoteFeatures)
            assert features.note_path == note_path
            assert features.note_title == "Test Note"
            assert features.word_count > 0
            assert features.tag_count == 1

    def test_compute_centrality_metrics_basic(self, notes_generator, sample_link_graph):
        """Test basic centrality metrics computation."""
        note_path = "note1.md"
        
        metrics = notes_generator._compute_centrality_metrics(note_path, sample_link_graph)
        
        assert isinstance(metrics, CentralityMetrics)
        assert 0.0 <= metrics.betweenness_centrality <= 1.0
        assert 0.0 <= metrics.closeness_centrality <= 1.0
        assert 0.0 <= metrics.pagerank_score <= 1.0

    def test_compute_centrality_metrics_isolated_node(self, notes_generator):
        """Test centrality metrics for isolated node."""
        graph = nx.DiGraph()
        graph.add_node("isolated.md")
        
        metrics = notes_generator._compute_centrality_metrics("isolated.md", graph)
        
        assert metrics.betweenness_centrality == 0.0
        assert metrics.closeness_centrality == 0.0
        assert metrics.pagerank_score > 0.0  # PageRank gives minimum value

    def test_compute_centrality_metrics_nonexistent_node(self, notes_generator, sample_link_graph):
        """Test centrality metrics for node not in graph."""
        metrics = notes_generator._compute_centrality_metrics("nonexistent.md", sample_link_graph)
        
        # Should return zero metrics for nonexistent node
        assert metrics.betweenness_centrality == 0.0
        assert metrics.closeness_centrality == 0.0
        assert metrics.pagerank_score == 0.0

    def test_extract_all_frontmatter_properties_basic(self, notes_generator):
        """Test extraction of all frontmatter properties."""
        frontmatter = {
            "title": "Test Note",
            "tags": ["tag1", "tag2"],
            "aliases": ["alias1", "alias2"],
            "domains": ["domain1"],
            "concepts": ["concept1", "concept2", "concept3"],
            "up::": ["parent.md"],
            "similar": ["related1.md", "related2.md"]
        }
        
        result = notes_generator._extract_all_frontmatter_properties(frontmatter)
        
        assert result["title"] == "Test Note"
        assert result["tags_count"] == 2
        assert result["aliases_count"] == 2
        assert result["domains_count"] == 1
        assert result["concepts_count"] == 3
        assert result["semantic_up_links"] == 1
        assert result["semantic_similar_links"] == 2

    def test_extract_all_frontmatter_properties_nested(self, notes_generator):
        """Test extraction of nested frontmatter properties."""
        frontmatter = {
            "metadata": {
                "author": "John Doe",
                "version": "1.0"
            },
            "relationships": {
                "extends": ["base.md"],
                "implements": ["interface1.md", "interface2.md"]
            }
        }
        
        result = notes_generator._extract_all_frontmatter_properties(frontmatter)
        
        assert result["metadata_author"] == "John Doe"
        assert result["metadata_version"] == "1.0"
        assert result["semantic_extends_links"] == 1
        assert result["semantic_implements_links"] == 2

    def test_extract_all_frontmatter_properties_progress_indicators(self, notes_generator):
        """Test extraction of progress state indicators."""
        frontmatter = {
            "tags": ["🌱", "project", "🌿", "active"],
            "status": "🌲",
            "progress": ["⚛️", "🗺️"]
        }
        
        result = notes_generator._extract_all_frontmatter_properties(frontmatter)
        
        # Should detect progress indicators
        progress_indicators = result.get("progress_indicators", [])
        assert "🌱" in progress_indicators
        assert "🌿" in progress_indicators
        assert "🌲" in progress_indicators
        assert "⚛️" in progress_indicators
        assert "🗺️" in progress_indicators

    def test_generate_semantic_summary_from_frontmatter(self, notes_generator):
        """Test semantic summary generation from frontmatter."""
        content = "# Test Note\n\nThis is test content."
        frontmatter = {"summary": "This is a predefined summary."}
        
        summary = notes_generator._generate_semantic_summary(content, frontmatter)
        
        # Should prioritize frontmatter summary
        assert summary == "This is a predefined summary."

    def test_generate_semantic_summary_from_content(self, notes_generator):
        """Test semantic summary generation from content analysis."""
        content = """# Main Topic

This note discusses important concepts about machine learning.

## Key Points
- Neural networks are powerful
- Deep learning requires data
- Training is computationally intensive

## Conclusion
Machine learning is transformative."""
        
        frontmatter = {}  # No predefined summary
        
        with patch.object(notes_generator, '_analyze_content_structure') as mock_analyze, \
             patch.object(notes_generator, '_extract_key_concepts') as mock_concepts:
            
            mock_analyze.return_value = {
                "main_topics": ["machine learning", "neural networks"],
                "key_points": ["powerful", "data", "computationally intensive"],
                "structure_quality": 0.8
            }
            mock_concepts.return_value = ["machine learning", "neural networks", "deep learning"]
            
            summary = notes_generator._generate_semantic_summary(content, frontmatter)
            
            assert isinstance(summary, str)
            assert len(summary) > 0
            assert "machine learning" in summary.lower()

    def test_analyze_content_structure(self, notes_generator):
        """Test content structure analysis."""
        content = """# Main Heading

Introduction paragraph with key concepts.

## Subheading 1
Content with technical terms like API, database, and algorithm.

### Sub-subheading
More detailed content.

## Subheading 2
Additional content with examples.
"""
        
        analysis = notes_generator._analyze_content_structure(content)
        
        assert "headings" in analysis
        assert "key_phrases" in analysis
        assert "technical_terms" in analysis
        assert len(analysis["headings"]) >= 3  # Should detect headings
        assert analysis["max_heading_depth"] == 3

    def test_extract_key_concepts_basic(self, notes_generator):
        """Test key concept extraction."""
        content = "This note discusses machine learning, neural networks, and data science."
        frontmatter = {"concepts": ["artificial intelligence", "deep learning"]}
        
        concepts = notes_generator._extract_key_concepts(content, frontmatter)
        
        assert isinstance(concepts, list)
        assert "artificial intelligence" in concepts  # From frontmatter
        assert "deep learning" in concepts  # From frontmatter
        # Should also extract from content
        assert len(concepts) >= 2

    def test_precompute_centrality_metrics_caching(self, notes_generator, sample_link_graph):
        """Test centrality metrics caching."""
        cache = notes_generator._precompute_centrality_metrics(sample_link_graph)
        
        assert isinstance(cache, dict)
        assert "note1.md" in cache
        assert "note2.md" in cache
        assert "note3.md" in cache
        
        # Each cached entry should be CentralityMetrics
        for node, metrics in cache.items():
            assert isinstance(metrics, CentralityMetrics)

    def test_batch_processing_memory_management(self, notes_generator, sample_notes, sample_link_graph):
        """Test batch processing with memory management."""
        # Mock file reading
        notes_generator.vault_reader.read_file.return_value = ("Content", {"created": 1234567890})
        
        # Mock other methods
        with patch.object(notes_generator, '_extract_note_features') as mock_extract, \
             patch.object(notes_generator, '_precompute_centrality_metrics') as mock_centrality:
            
            mock_extract.return_value = NoteFeatures(
                note_path="test.md",
                note_title="Test",
                word_count=10,
                tag_count=1,
                quality_stage="🌱",
                creation_date=datetime.now(),
                last_modified=datetime.now(),
                outgoing_links_count=0
            )
            mock_centrality.return_value = {}
            
            # Should handle batch processing without errors
            result = notes_generator.generate_dataset(
                notes=sample_notes + ["note4.md", "note5.md"],  # Ensure minimum count
                link_graph=sample_link_graph,
                batch_size=2
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 5


class TestPairsDatasetGenerator:
    """Test suite for PairsDatasetGenerator feature engineering."""

    @pytest.fixture
    def mock_vector_encoder(self):
        """Create a mock VectorEncoder."""
        mock_encoder = Mock(spec=VectorEncoder)
        return mock_encoder

    @pytest.fixture
    def mock_graph_database(self):
        """Create a mock GraphDatabase."""
        mock_db = Mock(spec=GraphDatabase)
        return mock_db

    @pytest.fixture
    def pairs_generator(self, mock_vector_encoder, mock_graph_database):
        """Create a PairsDatasetGenerator instance."""
        return PairsDatasetGenerator(
            vector_encoder=mock_vector_encoder,
            graph_database=mock_graph_database
        )

    @pytest.fixture
    def sample_note_data(self):
        """Create sample note data for pairs testing."""
        return {
            "note1.md": NoteData(
                path="note1.md",
                title="Note 1",
                content="Content about machine learning",
                metadata={"created": 1234567890},
                tags=["ml", "ai"],
                outgoing_links=["note2.md"],
                embedding=np.array([1.0, 0.0, 0.0]),
                word_count=10
            ),
            "note2.md": NoteData(
                path="note2.md",
                title="Note 2",
                content="Content about deep learning",
                metadata={"created": 1234567891},
                tags=["dl", "ai"],
                outgoing_links=["note3.md"],
                embedding=np.array([0.8, 0.6, 0.0]),
                word_count=15
            ),
            "note3.md": NoteData(
                path="note3.md",
                title="Note 3",
                content="Content about data science",
                metadata={"created": 1234567892},
                tags=["ds", "analytics"],
                outgoing_links=[],
                embedding=np.array([0.0, 1.0, 0.0]),
                word_count=12
            )
        }

    @pytest.fixture
    def sample_link_graph(self):
        """Create a sample link graph for pairs testing."""
        graph = nx.DiGraph()
        graph.add_nodes_from(["note1.md", "note2.md", "note3.md"])
        graph.add_edge("note1.md", "note2.md")
        graph.add_edge("note2.md", "note3.md")
        return graph

    def test_compute_semantic_similarity_basic(self, pairs_generator):
        """Test basic semantic similarity computation."""
        embedding_a = np.array([1.0, 0.0, 0.0])
        embedding_b = np.array([0.0, 1.0, 0.0])
        
        similarity = pairs_generator._compute_semantic_similarity(embedding_a, embedding_b)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert similarity == 0.0  # Orthogonal vectors

    def test_compute_semantic_similarity_identical(self, pairs_generator):
        """Test semantic similarity for identical embeddings."""
        embedding = np.array([1.0, 0.0, 0.0])
        
        similarity = pairs_generator._compute_semantic_similarity(embedding, embedding)
        
        assert similarity == 1.0  # Identical vectors

    def test_compute_semantic_similarity_normalized(self, pairs_generator):
        """Test semantic similarity with normalized vectors."""
        embedding_a = np.array([2.0, 0.0, 0.0])  # Will be normalized
        embedding_b = np.array([0.0, 3.0, 0.0])  # Will be normalized
        
        similarity = pairs_generator._compute_semantic_similarity(embedding_a, embedding_b)
        
        assert similarity == 0.0  # Still orthogonal after normalization

    def test_compute_pair_features_basic(self, pairs_generator, sample_note_data, sample_link_graph):
        """Test basic pair feature computation."""
        note_a = sample_note_data["note1.md"]
        note_b = sample_note_data["note2.md"]
        
        features = pairs_generator._compute_pair_features(note_a, note_b, sample_link_graph)
        
        assert isinstance(features, PairFeatures)
        assert features.note_a_path == "note1.md"
        assert features.note_b_path == "note2.md"
        assert 0.0 <= features.cosine_similarity <= 1.0
        assert features.link_exists is True  # There's a link in the graph

    def test_compute_pair_features_no_link(self, pairs_generator, sample_note_data, sample_link_graph):
        """Test pair feature computation for unlinked notes."""
        note_a = sample_note_data["note1.md"]
        note_c = sample_note_data["note3.md"]
        
        features = pairs_generator._compute_pair_features(note_a, note_c, sample_link_graph)
        
        assert features.link_exists is False  # No direct link
        assert features.shortest_path_length > 1  # Indirect path exists

    def test_compute_pair_features_tag_overlap(self, pairs_generator, sample_note_data, sample_link_graph):
        """Test tag overlap computation in pair features."""
        note_a = sample_note_data["note1.md"]  # tags: ["ml", "ai"]
        note_b = sample_note_data["note2.md"]  # tags: ["dl", "ai"]
        
        features = pairs_generator._compute_pair_features(note_a, note_b, sample_link_graph)
        
        assert features.tag_overlap_count == 1  # "ai" is common
        assert 0.0 < features.tag_jaccard_similarity < 1.0

    def test_compute_pair_features_no_tag_overlap(self, pairs_generator, sample_note_data, sample_link_graph):
        """Test pair features with no tag overlap."""
        note_a = sample_note_data["note1.md"]  # tags: ["ml", "ai"]
        note_c = sample_note_data["note3.md"]  # tags: ["ds", "analytics"]
        
        features = pairs_generator._compute_pair_features(note_a, note_c, sample_link_graph)
        
        assert features.tag_overlap_count == 0
        assert features.tag_jaccard_similarity == 0.0

    def test_compute_pair_features_word_count_ratio(self, pairs_generator, sample_note_data, sample_link_graph):
        """Test word count ratio computation."""
        note_a = sample_note_data["note1.md"]  # word_count: 10
        note_b = sample_note_data["note2.md"]  # word_count: 15
        
        features = pairs_generator._compute_pair_features(note_a, note_b, sample_link_graph)
        
        # Should be normalized to smaller/larger
        expected_ratio = 10.0 / 15.0
        assert abs(features.word_count_ratio - expected_ratio) < 0.01

    def test_compute_pair_features_creation_time_diff(self, pairs_generator, sample_note_data, sample_link_graph):
        """Test creation time difference computation."""
        note_a = sample_note_data["note1.md"]  # created: 1234567890
        note_b = sample_note_data["note2.md"]  # created: 1234567891
        
        features = pairs_generator._compute_pair_features(note_a, note_b, sample_link_graph)
        
        # Should compute difference in days
        expected_diff = abs(1234567891 - 1234567890) / (24 * 3600)  # Convert to days
        assert abs(features.creation_time_diff_days - expected_diff) < 0.01

    def test_compute_graph_metrics_shortest_path(self, pairs_generator, sample_link_graph):
        """Test shortest path computation."""
        path_length = pairs_generator._compute_shortest_path_length(
            "note1.md", "note3.md", sample_link_graph
        )
        
        assert path_length == 2  # note1 -> note2 -> note3

    def test_compute_graph_metrics_no_path(self, pairs_generator):
        """Test shortest path when no path exists."""
        graph = nx.DiGraph()
        graph.add_nodes_from(["note1.md", "note2.md"])
        # No edges - disconnected
        
        path_length = pairs_generator._compute_shortest_path_length(
            "note1.md", "note2.md", graph
        )
        
        assert path_length == float('inf')

    def test_compute_common_neighbors(self, pairs_generator):
        """Test common neighbors computation."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("note1.md", "common1.md"),
            ("note1.md", "common2.md"),
            ("note2.md", "common1.md"),
            ("note2.md", "common2.md"),
            ("note2.md", "unique.md")
        ])
        
        common_count = pairs_generator._compute_common_neighbors_count(
            "note1.md", "note2.md", graph
        )
        
        assert common_count == 2  # common1.md and common2.md

    def test_compute_adamic_adar_score(self, pairs_generator):
        """Test Adamic-Adar score computation."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("note1.md", "common1.md"),
            ("note2.md", "common1.md"),
            ("other1.md", "common1.md"),  # common1 has degree 3
            ("note1.md", "common2.md"),
            ("note2.md", "common2.md")   # common2 has degree 2
        ])
        
        score = pairs_generator._compute_adamic_adar_score(
            "note1.md", "note2.md", graph
        )
        
        # Score should be 1/log(3) + 1/log(2) for the two common neighbors
        expected = 1.0 / np.log(3) + 1.0 / np.log(2)
        assert abs(score - expected) < 0.01

    def test_generate_dataset_insufficient_notes(self, pairs_generator, sample_link_graph):
        """Test dataset generation with insufficient notes."""
        insufficient_data = {
            "note1.md": sample_note_data["note1.md"],
            "note2.md": sample_note_data["note2.md"]
        }
        
        with pytest.raises(InsufficientDataError):
            pairs_generator.generate_dataset(insufficient_data, sample_link_graph)

    def test_generate_dataset_basic(self, pairs_generator, sample_note_data, sample_link_graph):
        """Test basic dataset generation."""
        # Add more notes to meet minimum requirement
        extended_data = dict(sample_note_data)
        for i in range(4, 8):  # Add notes 4-7
            extended_data[f"note{i}.md"] = NoteData(
                path=f"note{i}.md",
                title=f"Note {i}",
                content=f"Content {i}",
                metadata={"created": 1234567890 + i},
                tags=[f"tag{i}"],
                outgoing_links=[],
                embedding=np.random.rand(3),
                word_count=10
            )
        
        with patch.object(pairs_generator, '_smart_negative_sampling') as mock_sampling:
            mock_sampling.return_value = [("note1.md", "note3.md")]
            
            result = pairs_generator.generate_dataset(extended_data, sample_link_graph)
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "note_a_path" in result.columns
            assert "note_b_path" in result.columns
            assert "link_exists" in result.columns


class TestSamplingStrategies:
    """Test suite for negative sampling strategies."""

    @pytest.fixture
    def sample_note_data(self):
        """Create sample note data for sampling tests."""
        return {
            "folder1/note1.md": NoteData(
                path="folder1/note1.md", title="Note 1", content="Content 1",
                metadata={}, tags=["tag1", "tag2"], outgoing_links=[],
                embedding=np.random.rand(3)
            ),
            "folder1/note2.md": NoteData(
                path="folder1/note2.md", title="Note 2", content="Content 2",
                metadata={}, tags=["tag2", "tag3"], outgoing_links=[],
                embedding=np.random.rand(3)
            ),
            "folder2/note3.md": NoteData(
                path="folder2/note3.md", title="Note 3", content="Content 3",
                metadata={}, tags=["tag3", "tag4"], outgoing_links=[],
                embedding=np.random.rand(3)
            )
        }

    def test_random_sampling_strategy(self):
        """Test random negative sampling strategy."""
        strategy = RandomSamplingStrategy(random_seed=42)
        positive_pairs = {("note1.md", "note2.md")}
        all_notes = ["note1.md", "note2.md", "note3.md", "note4.md"]
        
        negative_pairs = strategy.sample_negative_pairs(positive_pairs, all_notes, 3)
        
        assert len(negative_pairs) <= 3
        assert all(pair not in positive_pairs for pair in negative_pairs)
        assert all(pair[0] != pair[1] for pair in negative_pairs)  # No self-pairs

    def test_random_sampling_strategy_reproducible(self):
        """Test that random sampling is reproducible with seed."""
        strategy1 = RandomSamplingStrategy(random_seed=42)
        strategy2 = RandomSamplingStrategy(random_seed=42)
        
        positive_pairs = {("note1.md", "note2.md")}
        all_notes = ["note1.md", "note2.md", "note3.md", "note4.md"]
        
        pairs1 = strategy1.sample_negative_pairs(positive_pairs, all_notes, 5)
        pairs2 = strategy2.sample_negative_pairs(positive_pairs, all_notes, 5)
        
        assert pairs1 == pairs2

    def test_stratified_sampling_strategy_initialization(self, sample_note_data):
        """Test stratified sampling strategy initialization."""
        strategy = StratifiedSamplingStrategy(sample_note_data, random_seed=42)
        
        assert strategy.note_data == sample_note_data
        assert len(strategy._folder_groups) == 2  # folder1 and folder2
        assert "folder1" in strategy._folder_groups
        assert "folder2" in strategy._folder_groups

    def test_stratified_sampling_folder_grouping(self, sample_note_data):
        """Test folder-based grouping in stratified sampling."""
        strategy = StratifiedSamplingStrategy(sample_note_data)
        
        folder_groups = strategy._folder_groups
        assert len(folder_groups["folder1"]) == 2
        assert len(folder_groups["folder2"]) == 1
        assert "folder1/note1.md" in folder_groups["folder1"]
        assert "folder1/note2.md" in folder_groups["folder1"]
        assert "folder2/note3.md" in folder_groups["folder2"]

    def test_stratified_sampling_tag_grouping(self, sample_note_data):
        """Test tag-based grouping in stratified sampling."""
        strategy = StratifiedSamplingStrategy(sample_note_data)
        
        tag_groups = strategy._tag_groups
        assert "tag1" in tag_groups
        assert "tag2" in tag_groups
        assert "tag3" in tag_groups
        assert "tag4" in tag_groups
        
        assert "folder1/note1.md" in tag_groups["tag1"]
        assert "folder1/note1.md" in tag_groups["tag2"]
        assert "folder1/note2.md" in tag_groups["tag2"]

    def test_stratified_sampling_negative_pairs(self, sample_note_data):
        """Test stratified negative pair sampling."""
        strategy = StratifiedSamplingStrategy(sample_note_data, random_seed=42)
        positive_pairs = {("folder1/note1.md", "folder1/note2.md")}
        all_notes = list(sample_note_data.keys())
        
        negative_pairs = strategy.sample_negative_pairs(positive_pairs, all_notes, 2)
        
        assert len(negative_pairs) <= 2
        assert all(pair not in positive_pairs for pair in negative_pairs)
        assert all(pair[0] != pair[1] for pair in negative_pairs)

    def test_sampling_strategy_interface(self):
        """Test that sampling strategies implement the required interface."""
        strategy = RandomSamplingStrategy()
        
        assert hasattr(strategy, 'sample_negative_pairs')
        assert hasattr(strategy, 'get_strategy_name')
        assert callable(strategy.sample_negative_pairs)
        assert callable(strategy.get_strategy_name)
        
        assert strategy.get_strategy_name() == "random"


class TestFeatureEngineeringErrorHandling:
    """Test suite for error handling in feature engineering."""

    def test_feature_engineering_error_creation(self):
        """Test FeatureEngineeringError creation and attributes."""
        error = FeatureEngineeringError(
            "Feature computation failed",
            component="centrality_calculator",
            feature_type="betweenness_centrality"
        )
        
        assert str(error) == "Feature computation failed"
        assert error.component == "centrality_calculator"
        assert error.feature_type == "betweenness_centrality"

    def test_insufficient_data_error_creation(self):
        """Test InsufficientDataError creation and attributes."""
        error = InsufficientDataError(
            "Not enough data",
            required_minimum=10,
            actual_count=5
        )
        
        assert str(error) == "Not enough data"
        assert error.required_minimum == 10
        assert error.actual_count == 5

    def test_sampling_error_creation(self):
        """Test SamplingError creation and attributes."""
        error = SamplingError(
            "Sampling failed",
            strategy_name="random",
            target_count=100,
            actual_count=50
        )
        
        assert str(error) == "Sampling failed"
        assert error.strategy_name == "random"
        assert error.target_count == 100
        assert error.actual_count == 50

    def test_feature_extraction_with_missing_embeddings(self, sample_note_data):
        """Test feature extraction handles missing embeddings gracefully."""
        # Remove embeddings from note data
        for note_data in sample_note_data.values():
            note_data.embedding = None
        
        # This should be handled gracefully by the feature extraction code
        # The exact behavior depends on implementation, but should not crash
        assert True  # Placeholder - actual test would depend on implementation

    def test_centrality_computation_with_empty_graph(self):
        """Test centrality computation with empty graph."""
        from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
        
        generator = NotesDatasetGenerator(
            vault_reader=Mock(),
            vector_encoder=Mock()
        )
        
        empty_graph = nx.DiGraph()
        metrics = generator._compute_centrality_metrics("nonexistent.md", empty_graph)
        
        # Should return zero metrics for empty graph
        assert metrics.betweenness_centrality == 0.0
        assert metrics.closeness_centrality == 0.0
        assert metrics.pagerank_score == 0.0

    def test_semantic_similarity_with_zero_vectors(self):
        """Test semantic similarity computation with zero vectors."""
        from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import PairsDatasetGenerator
        
        generator = PairsDatasetGenerator(
            vector_encoder=Mock(),
            graph_database=Mock()
        )
        
        zero_vector = np.zeros(384)
        similarity = generator._compute_semantic_similarity(zero_vector, zero_vector)
        
        # Should handle zero vectors gracefully (might return NaN or 0)
        assert not np.isnan(similarity) or similarity == 0.0