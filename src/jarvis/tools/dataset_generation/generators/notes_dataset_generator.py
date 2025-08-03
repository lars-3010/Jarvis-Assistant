"""
Notes dataset generator for individual note analysis.

This module creates a comprehensive dataset containing properties and features
of individual notes in an Obsidian vault, including semantic embeddings,
centrality metrics, and content-based features.
"""

import gc
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from jarvis.services.graph.database import GraphDatabase
from jarvis.services.vault.parser import MarkdownParser
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.logging import setup_logging

from ..models.data_models import (
    CentralityMetrics,
    NoteData,
    NoteFeatures,
    ProcessingStats,
    ValidationResult,
)
from ..models.exceptions import FeatureEngineeringError, InsufficientDataError

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = setup_logging(__name__)


class NotesDatasetGenerator:
    """Generate individual note properties dataset."""

    def __init__(self, vault_reader: VaultReader, vector_encoder: VectorEncoder,
                 graph_database: GraphDatabase | None = None, markdown_parser: MarkdownParser | None = None):
        """Initialize the notes dataset generator.
        
        Args:
            vault_reader: VaultReader service for file operations
            vector_encoder: VectorEncoder service for embeddings
            graph_database: Optional GraphDatabase service for centrality metrics
            markdown_parser: Optional MarkdownParser service for frontmatter extraction
        """
        self.vault_reader = vault_reader
        self.vector_encoder = vector_encoder
        self.graph_database = graph_database
        self.markdown_parser = markdown_parser or MarkdownParser(extract_semantic=True)
        self._processing_stats = None

    def generate_dataset(self, notes: list[str], link_graph: nx.DiGraph,
                        batch_size: int = 32, progress_callback=None) -> pd.DataFrame:
        """Generate comprehensive notes dataset with optimized batch processing.
        
        Args:
            notes: List of note file paths
            link_graph: NetworkX graph of note relationships
            batch_size: Initial batch size for processing (will be adapted based on memory)
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame containing note features
            
        Raises:
            InsufficientDataError: If not enough notes for meaningful dataset
            FeatureEngineeringError: If feature extraction fails
        """
        logger.info(f"Starting optimized notes dataset generation for {len(notes)} notes")

        if len(notes) < 5:
            raise InsufficientDataError(
                f"Insufficient notes for dataset generation: {len(notes)} < 5",
                required_minimum=5,
                actual_count=len(notes)
            )

        # Initialize processing stats with memory monitoring
        self._processing_stats = ProcessingStats(start_time=datetime.now())
        memory_monitor = self._create_memory_monitor()

        try:
            # Load and process notes in adaptive batches
            note_features_list = []
            processed_count = 0
            failed_count = 0
            current_batch_size = batch_size

            # Pre-compute centrality metrics for all nodes
            centrality_cache = self._precompute_centrality_metrics(link_graph)

            # Process notes in adaptive batches
            i = 0
            while i < len(notes):
                # Adjust batch size based on memory usage
                current_batch_size = self._adjust_batch_size(
                    current_batch_size, memory_monitor, processed_count
                )
                
                batch = notes[i:i + current_batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.debug(f"Processing batch {batch_num}: {len(batch)} notes (adaptive size: {current_batch_size})")

                # Monitor memory before processing
                memory_before = memory_monitor.get_current_usage()

                # Process batch with memory monitoring
                batch_features = self._process_note_batch_optimized(
                    batch, link_graph, centrality_cache, memory_monitor
                )

                # Update counters
                for features in batch_features:
                    if features is not None:
                        note_features_list.append(features)
                        processed_count += 1
                    else:
                        failed_count += 1

                # Memory cleanup after batch
                memory_after = memory_monitor.get_current_usage()
                if memory_after > memory_before * 1.5:  # Memory increased significantly
                    logger.debug("Performing garbage collection due to memory increase")
                    gc.collect()

                # Update progress
                total_processed = processed_count + failed_count
                if progress_callback:
                    progress_callback(total_processed, len(notes))

                # Log progress with memory info
                if batch_num % 10 == 0:
                    memory_mb = memory_monitor.get_current_usage()
                    logger.info(f"Processed {total_processed}/{len(notes)} notes "
                               f"(Memory: {memory_mb:.1f}MB, Batch size: {current_batch_size})")

                i += current_batch_size

            # Update processing stats
            self._processing_stats.end_time = datetime.now()
            self._processing_stats.items_processed = processed_count
            self._processing_stats.items_failed = failed_count
            self._processing_stats.memory_usage_mb = memory_monitor.get_current_usage()

            logger.info(f"Notes processing complete: {processed_count} successful, {failed_count} failed")
            logger.info(f"Peak memory usage: {memory_monitor.get_peak_usage():.1f}MB")

            if not note_features_list:
                raise FeatureEngineeringError("No note features could be extracted")

            # Convert to DataFrame with memory optimization
            dataset = self._create_dataframe_optimized(note_features_list, memory_monitor)

            logger.info(f"Notes dataset created: {len(dataset)} rows, {len(dataset.columns)} columns")
            return dataset

        except Exception as e:
            logger.error(f"Notes dataset generation failed: {e}")
            if isinstance(e, (InsufficientDataError, FeatureEngineeringError)):
                raise
            raise FeatureEngineeringError(f"Failed to generate notes dataset: {e}") from e

    def _process_note_batch(self, batch_notes: list[str], link_graph: nx.DiGraph,
                          centrality_cache: dict[str, CentralityMetrics]) -> list[NoteFeatures | None]:
        """Process a batch of notes to extract features.
        
        Args:
            batch_notes: List of note paths in this batch
            link_graph: NetworkX graph for centrality calculations
            centrality_cache: Pre-computed centrality metrics
            
        Returns:
            List of NoteFeatures objects (None for failed notes)
        """
        batch_features = []

        # Load note data for batch
        note_data_batch = []
        for note_path in batch_notes:
            try:
                note_data = self._load_note_data(note_path)
                note_data_batch.append(note_data)
            except Exception as e:
                logger.warning(f"Failed to load note data for {note_path}: {e}")
                note_data_batch.append(None)

        # Generate embeddings for valid notes in batch
        valid_notes = [data for data in note_data_batch if data is not None]
        if valid_notes:
            try:
                contents = [data.content for data in valid_notes]
                embeddings = self.vector_encoder.encode_documents(contents)

                # Assign embeddings back to note data
                embedding_idx = 0
                for i, data in enumerate(note_data_batch):
                    if data is not None:
                        data.embedding = embeddings[embedding_idx]
                        embedding_idx += 1

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                # Set embeddings to None for error handling
                for data in note_data_batch:
                    if data is not None:
                        data.embedding = None

        # Extract features for each note
        for note_data in note_data_batch:
            if note_data is None:
                batch_features.append(None)
                continue

            try:
                features = self._extract_note_features(note_data, link_graph, centrality_cache)
                batch_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for {note_data.path}: {e}")
                batch_features.append(None)

        return batch_features

    def _load_note_data(self, note_path: str) -> NoteData:
        """Load complete note data from file.
        
        Args:
            note_path: Path to the note file
            
        Returns:
            NoteData object with loaded information
        """
        try:
            # Read file content and metadata
            content, metadata = self.vault_reader.read_file(note_path)

            # Parse markdown content with comprehensive frontmatter extraction
            parsed_content = self.markdown_parser.parse(content)
            frontmatter = parsed_content.get('frontmatter', {})
            content_without_frontmatter = parsed_content.get('content_without_frontmatter', content)

            # Extract basic information
            path_obj = Path(note_path)
            title = path_obj.stem

            # Extract tags using the parser (handles both frontmatter and content tags)
            tags = parsed_content.get('tags', [])

            # Extract outgoing links using the parser
            links_data = parsed_content.get('links', [])
            outgoing_links = [link['target'] for link in links_data if link.get('target')]

            # Get file statistics
            full_path = self.vault_reader.get_absolute_path(note_path)
            file_stat = full_path.stat()

            # Extract comprehensive frontmatter properties
            all_frontmatter_properties = self._extract_all_frontmatter_properties(frontmatter)
            
            # Extract semantic relationships
            semantic_relationships = parsed_content.get('relationships', {})
            
            # Extract progress indicators from tags and frontmatter
            progress_indicators = self._extract_progress_indicators(tags, frontmatter)

            # Create NoteData object with enhanced properties
            note_data = NoteData(
                path=note_path,
                title=title,
                content=content_without_frontmatter,
                metadata=metadata,
                tags=tags,
                outgoing_links=outgoing_links,
                word_count=len(content_without_frontmatter.split()) if content_without_frontmatter else 0,
                creation_date=datetime.fromtimestamp(file_stat.st_ctime),
                last_modified=datetime.fromtimestamp(file_stat.st_mtime),
                all_frontmatter_properties=all_frontmatter_properties,
                semantic_relationships=semantic_relationships,
                progress_indicators=progress_indicators
            )

            # Extract quality stage from metadata or frontmatter
            note_data.quality_stage = (
                frontmatter.get('quality_stage') or 
                metadata.get('quality_stage') or 
                self._infer_quality_stage_from_progress(progress_indicators) or
                'unknown'
            )

            return note_data

        except Exception as e:
            logger.error(f"Failed to load note data for {note_path}: {e}")
            raise FeatureEngineeringError(f"Failed to load note data: {e}") from e

    def _extract_tags(self, content: str, metadata: dict) -> list[str]:
        """Extract tags from content and metadata.
        
        Args:
            content: Note content
            metadata: Note metadata
            
        Returns:
            List of tags
        """
        tags = set()

        # Extract from metadata
        if 'tags' in metadata:
            meta_tags = metadata['tags']
            if isinstance(meta_tags, list):
                tags.update(meta_tags)
            elif isinstance(meta_tags, str):
                tags.add(meta_tags)

        # Extract hashtags from content
        import re
        hashtag_pattern = re.compile(r'#([a-zA-Z0-9_/-]+)')
        content_tags = hashtag_pattern.findall(content or '')
        tags.update(content_tags)

        return list(tags)

    def _extract_outgoing_links(self, content: str) -> list[str]:
        """Extract outgoing links from content (simplified version).
        
        Args:
            content: Note content
            
        Returns:
            List of link targets
        """
        if not content:
            return []

        links = []

        # Simple regex for wikilinks [[link]]
        import re
        wikilink_pattern = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
        wikilinks = wikilink_pattern.findall(content)
        links.extend(wikilinks)

        return links

    def _precompute_centrality_metrics(self, graph: nx.DiGraph) -> dict[str, CentralityMetrics]:
        """Pre-compute centrality metrics for all nodes in the graph.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary mapping node IDs to centrality metrics
        """
        logger.info("Pre-computing centrality metrics for all nodes")
        centrality_cache = {}

        if graph.number_of_nodes() == 0:
            return centrality_cache

        try:
            # Compute all centrality measures at once for efficiency
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            pagerank = nx.pagerank(graph, max_iter=1000)
            degree = nx.degree_centrality(graph)

            # Clustering coefficient (on undirected version)
            undirected = graph.to_undirected()
            clustering = nx.clustering(undirected)

            # Eigenvector centrality (may fail for some graphs)
            eigenvector = {}
            try:
                eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            except (nx.NetworkXError, np.linalg.LinAlgError):
                logger.warning("Eigenvector centrality computation failed, using zeros")
                eigenvector = dict.fromkeys(graph.nodes(), 0.0)

            # Create CentralityMetrics for each node
            for node in graph.nodes():
                metrics = CentralityMetrics(
                    betweenness_centrality=betweenness.get(node, 0.0),
                    closeness_centrality=closeness.get(node, 0.0),
                    pagerank_score=pagerank.get(node, 0.0),
                    degree_centrality=degree.get(node, 0.0),
                    eigenvector_centrality=eigenvector.get(node, 0.0),
                    clustering_coefficient=clustering.get(node, 0.0)
                )
                centrality_cache[node] = metrics

            logger.info(f"Centrality metrics computed for {len(centrality_cache)} nodes")

        except Exception as e:
            logger.warning(f"Failed to compute centrality metrics: {e}")
            # Return empty cache - features will use default values

        return centrality_cache

    def _extract_note_features(self, note_data: NoteData, graph: nx.DiGraph,
                             centrality_cache: dict[str, CentralityMetrics]) -> NoteFeatures:
        """Extract all features for a single note with comprehensive error handling.
        
        Args:
            note_data: Complete note information
            graph: NetworkX graph for centrality calculations
            centrality_cache: Pre-computed centrality metrics
            
        Returns:
            NoteFeatures object with all extracted features (with defaults for failed extractions)
        """
        feature_errors = []
        
        try:
            # Get centrality metrics from cache with error handling
            try:
                centrality_metrics = centrality_cache.get(note_data.path, CentralityMetrics())
            except Exception as e:
                logger.warning(f"Failed to get centrality metrics for {note_data.path}: {e}")
                centrality_metrics = CentralityMetrics()
                feature_errors.append("centrality_metrics")

            # Count incoming links with error handling
            incoming_links_count = 0
            try:
                if graph and graph.has_node(note_data.path):
                    incoming_links_count = graph.in_degree(note_data.path)
            except Exception as e:
                logger.warning(f"Failed to count incoming links for {note_data.path}: {e}")
                feature_errors.append("incoming_links")

            # Generate semantic summary with error handling
            semantic_summary = ""
            try:
                semantic_summary = self._generate_semantic_summary(
                    note_data.content, note_data.all_frontmatter_properties
                )
            except Exception as e:
                logger.warning(f"Failed to generate semantic summary for {note_data.path}: {e}")
                feature_errors.append("semantic_summary")
                # Fallback to truncated content
                if note_data.content:
                    semantic_summary = note_data.content[:200] + "..." if len(note_data.content) > 200 else note_data.content

            # Calculate file size with error handling
            file_size = 0
            try:
                full_path = self.vault_reader.get_absolute_path(note_data.path)
                file_size = full_path.stat().st_size if full_path.exists() else 0
            except Exception as e:
                logger.debug(f"Failed to get file size for {note_data.path}: {e}")
                feature_errors.append("file_size")

            # Analyze content structure with error handling
            content_analysis = {
                'heading_count': 0,
                'max_heading_depth': 0,
                'technical_terms': [],
                'concept_density': 0.0
            }
            try:
                content_analysis = self._analyze_content_structure(note_data.content)
            except Exception as e:
                logger.warning(f"Failed to analyze content structure for {note_data.path}: {e}")
                feature_errors.append("content_analysis")

            # Extract frontmatter features with error handling
            frontmatter_features = {
                'aliases_count': 0,
                'domains_count': 0,
                'concepts_count': 0,
                'sources_count': 0,
                'has_summary_field': False,
                'progress_state': 'unknown'
            }
            try:
                frontmatter_features = self._compute_frontmatter_features(note_data)
            except Exception as e:
                logger.warning(f"Failed to compute frontmatter features for {note_data.path}: {e}")
                feature_errors.append("frontmatter_features")

            # Extract semantic relationship counts with error handling
            semantic_counts = {
                'up': 0,
                'similar': 0,
                'leads_to': 0,
                'extends': 0,
                'implements': 0
            }
            try:
                semantic_counts = self._compute_semantic_relationship_counts(note_data.semantic_relationships)
            except Exception as e:
                logger.warning(f"Failed to compute semantic relationships for {note_data.path}: {e}")
                feature_errors.append("semantic_relationships")

            # Calculate technical term density with error handling
            technical_term_density = 0.0
            try:
                technical_term_density = len(content_analysis['technical_terms']) / max(note_data.word_count, 1)
            except Exception as e:
                logger.debug(f"Failed to calculate technical term density for {note_data.path}: {e}")
                feature_errors.append("technical_term_density")

            # Create NoteFeatures object with all features (using defaults for failed extractions)
            try:
                features = NoteFeatures(
                    note_path=note_data.path,
                    note_title=note_data.title or Path(note_data.path).stem,
                    word_count=note_data.word_count or 0,
                    tag_count=len(note_data.tags) if note_data.tags else 0,
                    quality_stage=note_data.quality_stage or 'unknown',
                    creation_date=note_data.creation_date or datetime.now(),
                    last_modified=note_data.last_modified or datetime.now(),
                    outgoing_links_count=len(note_data.outgoing_links) if note_data.outgoing_links else 0,
                    incoming_links_count=incoming_links_count,
                    betweenness_centrality=centrality_metrics.betweenness_centrality,
                    closeness_centrality=centrality_metrics.closeness_centrality,
                    pagerank_score=centrality_metrics.pagerank_score,
                    clustering_coefficient=centrality_metrics.clustering_coefficient,
                    semantic_cluster_id=-1,  # Would require clustering analysis
                    semantic_summary=semantic_summary,
                    file_size=file_size,
                    # Enhanced frontmatter-derived features
                    aliases_count=frontmatter_features['aliases_count'],
                    domains_count=frontmatter_features['domains_count'],
                    concepts_count=frontmatter_features['concepts_count'],
                    sources_count=frontmatter_features['sources_count'],
                    has_summary_field=frontmatter_features['has_summary_field'],
                    progress_state=frontmatter_features['progress_state'],
                    # Semantic relationship counts
                    semantic_up_links=semantic_counts['up'],
                    semantic_similar_links=semantic_counts['similar'],
                    semantic_leads_to_links=semantic_counts['leads_to'],
                    semantic_extends_links=semantic_counts['extends'],
                    semantic_implements_links=semantic_counts['implements'],
                    # Content analysis features
                    heading_count=content_analysis['heading_count'],
                    max_heading_depth=content_analysis['max_heading_depth'],
                    technical_term_density=technical_term_density,
                    concept_density_score=content_analysis['concept_density']
                )

                # Log feature extraction summary
                if feature_errors:
                    logger.info(f"Feature extraction for {note_data.path} completed with {len(feature_errors)} errors: {feature_errors}")
                else:
                    logger.debug(f"Feature extraction for {note_data.path} completed successfully")

                return features

            except Exception as e:
                logger.error(f"Failed to create NoteFeatures object for {note_data.path}: {e}")
                raise FeatureEngineeringError(f"Failed to create features object: {e}") from e

        except FeatureEngineeringError:
            # Re-raise FeatureEngineeringError as-is
            raise
        except Exception as e:
            logger.error(f"Critical error extracting features for {note_data.path}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            raise FeatureEngineeringError(f"Critical feature extraction failure for {note_data.path}: {e}") from e

    def _compute_frontmatter_features(self, note_data: NoteData) -> dict[str, any]:
        """Compute features derived from frontmatter properties.
        
        Args:
            note_data: Complete note information
            
        Returns:
            Dictionary of frontmatter-derived features
        """
        features = {
            'aliases_count': 0,
            'domains_count': 0,
            'concepts_count': 0,
            'sources_count': 0,
            'has_summary_field': False,
            'progress_state': 'unknown'
        }
        
        try:
            frontmatter = note_data.all_frontmatter_properties
            
            # Count aliases
            if 'aliases_count' in frontmatter:
                features['aliases_count'] = frontmatter['aliases_count']
            elif 'aliases' in frontmatter:
                aliases = frontmatter['aliases']
                if isinstance(aliases, list):
                    features['aliases_count'] = len(aliases)
                elif isinstance(aliases, str):
                    features['aliases_count'] = len([a.strip() for a in aliases.split(',') if a.strip()])
            
            # Count domains
            if 'domains_count' in frontmatter:
                features['domains_count'] = frontmatter['domains_count']
            elif 'domains' in frontmatter:
                domains = frontmatter['domains']
                if isinstance(domains, list):
                    features['domains_count'] = len(domains)
                elif isinstance(domains, str):
                    features['domains_count'] = len([d.strip() for d in domains.split(',') if d.strip()])
            
            # Count concepts
            if 'concepts_count' in frontmatter:
                features['concepts_count'] = frontmatter['concepts_count']
            elif 'concepts' in frontmatter:
                concepts = frontmatter['concepts']
                if isinstance(concepts, list):
                    features['concepts_count'] = len(concepts)
                elif isinstance(concepts, str):
                    features['concepts_count'] = len([c.strip() for c in concepts.split(',') if c.strip()])
            
            # Count sources
            if 'sources_count' in frontmatter:
                features['sources_count'] = frontmatter['sources_count']
            elif 'sources' in frontmatter:
                sources = frontmatter['sources']
                if isinstance(sources, list):
                    features['sources_count'] = len(sources)
                elif isinstance(sources, str):
                    features['sources_count'] = len([s.strip() for s in sources.split(',') if s.strip()])
            
            # Check for summary field
            features['has_summary_field'] = 'summary' in frontmatter and bool(frontmatter.get('summary', '').strip())
            
            # Determine progress state from indicators
            if note_data.progress_indicators:
                features['progress_state'] = note_data.progress_indicators[0]  # Use first indicator
            
        except Exception as e:
            logger.warning(f"Error computing frontmatter features: {e}")
            
        return features

    def _compute_semantic_relationship_counts(self, semantic_relationships: dict[str, list[str]]) -> dict[str, int]:
        """Compute counts of different semantic relationship types.
        
        Args:
            semantic_relationships: Dictionary of relationship types to targets
            
        Returns:
            Dictionary of relationship type counts
        """
        counts = {
            'up': 0,
            'similar': 0,
            'leads_to': 0,
            'extends': 0,
            'implements': 0
        }
        
        try:
            # Map relationship field names to count keys
            relationship_mapping = {
                'up': 'up',
                'up::': 'up',
                'similar': 'similar',
                'leads_to': 'leads_to',
                'leads to': 'leads_to',
                'extends': 'extends',
                'implements': 'implements'
            }
            
            for rel_type, targets in semantic_relationships.items():
                mapped_type = relationship_mapping.get(rel_type.lower())
                if mapped_type and isinstance(targets, list):
                    counts[mapped_type] = len(targets)
                    
        except Exception as e:
            logger.warning(f"Error computing semantic relationship counts: {e}")
            
        return counts

    def _extract_all_frontmatter_properties(self, frontmatter: dict[str, any]) -> dict[str, any]:
        """Extract and flatten all frontmatter properties dynamically.
        
        Args:
            frontmatter: Parsed frontmatter dictionary
            
        Returns:
            Flattened dictionary of all frontmatter properties
        """
        if not frontmatter:
            return {}

        flattened_properties = {}
        
        try:
            # Process all frontmatter properties dynamically
            for key, value in frontmatter.items():
                # Handle different data types appropriately
                if isinstance(value, list):
                    # Convert lists to comma-separated strings for analysis
                    flattened_properties[f"{key}_list"] = value
                    flattened_properties[f"{key}_count"] = len(value)
                    flattened_properties[f"{key}_str"] = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    # Flatten nested dictionaries with prefixed keys
                    for nested_key, nested_value in value.items():
                        flattened_key = f"{key}_{nested_key}"
                        flattened_properties[flattened_key] = nested_value
                elif isinstance(value, (str, int, float, bool)):
                    # Store primitive types directly
                    flattened_properties[key] = value
                else:
                    # Convert other types to string representation
                    flattened_properties[key] = str(value)

            logger.debug(f"Extracted {len(flattened_properties)} frontmatter properties")
            
        except Exception as e:
            logger.warning(f"Error extracting frontmatter properties: {e}")
            
        return flattened_properties

    def _extract_progress_indicators(self, tags: list[str], frontmatter: dict[str, any]) -> list[str]:
        """Extract progress state indicators from tags and frontmatter.
        
        Args:
            tags: List of tags from the note
            frontmatter: Frontmatter dictionary
            
        Returns:
            List of progress indicators found
        """
        progress_indicators = []
        
        # Define progress state emojis and their meanings
        progress_states = {
            '🌱': 'seedling',
            '🌿': 'fledgling', 
            '🌲': 'forest',
            '⚛️': 'atomic',
            '🗺️': 'map',
            '⧉': 'squares',
            '🎓': 'graduating'
        }
        
        try:
            # Check tags for progress indicators
            for tag in tags:
                if tag in progress_states:
                    progress_indicators.append(tag)
                    
            # Check frontmatter for progress-related fields
            progress_fields = ['progress', 'state', 'status', 'stage', 'maturity']
            for field in progress_fields:
                if field in frontmatter:
                    value = frontmatter[field]
                    if isinstance(value, str) and value in progress_states:
                        progress_indicators.append(value)
                        
        except Exception as e:
            logger.warning(f"Error extracting progress indicators: {e}")
            
        return list(set(progress_indicators))  # Remove duplicates

    def _infer_quality_stage_from_progress(self, progress_indicators: list[str]) -> str | None:
        """Infer quality stage from progress indicators.
        
        Args:
            progress_indicators: List of progress indicators
            
        Returns:
            Inferred quality stage or None
        """
        if not progress_indicators:
            return None
            
        # Map progress indicators to quality stages
        quality_mapping = {
            '🌱': 'seedling',
            '🌿': 'developing',
            '🌲': 'mature',
            '⚛️': 'atomic',
            '🗺️': 'comprehensive',
            '⧉': 'project',
            '🎓': 'educational'
        }
        
        # Return the first mapped quality stage found
        for indicator in progress_indicators:
            if indicator in quality_mapping:
                return quality_mapping[indicator]
                
        return None

    def _generate_semantic_summary(self, content: str, frontmatter: dict[str, any] = None) -> str:
        """Generate intelligent semantic summary prioritizing frontmatter summary.
        
        Args:
            content: Note content
            frontmatter: Frontmatter dictionary (optional)
            
        Returns:
            Semantic summary string
        """
        # Priority 1: Use existing summary from frontmatter
        if frontmatter and 'summary' in frontmatter:
            summary = frontmatter['summary']
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        
        # Priority 2: Generate intelligent summary from content
        if not content or len(content.strip()) < 50:
            return "Insufficient content for summary"

        try:
            # Analyze content structure for better summarization
            content_analysis = self._analyze_content_structure(content)
            
            # Extract key concepts from content and frontmatter
            key_concepts = self._extract_key_concepts(content, frontmatter or {})
            
            # Generate summary based on content type and structure
            if content_analysis['content_type'] == 'technical':
                return self._generate_technical_summary(content, key_concepts)
            elif content_analysis['content_type'] == 'conceptual':
                return self._generate_conceptual_summary(content, key_concepts)
            else:
                return self._generate_general_summary(content, content_analysis)
                
        except Exception as e:
            logger.warning(f"Error generating intelligent summary: {e}")
            # Fallback to simple extractive summary
            return self._generate_simple_summary(content)

    def _analyze_content_structure(self, content: str) -> dict[str, any]:
        """Analyze content structure for intelligent summarization.
        
        Args:
            content: Note content
            
        Returns:
            Dictionary with content analysis results
        """
        analysis = {
            'heading_count': 0,
            'max_heading_depth': 0,
            'key_phrases': [],
            'technical_terms': [],
            'main_topics': [],
            'content_type': 'general',
            'concept_density': 0.0
        }
        
        try:
            lines = content.split('\n')
            
            # Analyze headings
            heading_pattern = re.compile(r'^(#+)\s+(.+)$')
            for line in lines:
                match = heading_pattern.match(line.strip())
                if match:
                    analysis['heading_count'] += 1
                    depth = len(match.group(1))
                    analysis['max_heading_depth'] = max(analysis['max_heading_depth'], depth)
                    analysis['main_topics'].append(match.group(2).strip())
            
            # Identify content type based on patterns
            content_lower = content.lower()
            if any(term in content_lower for term in ['algorithm', 'function', 'class', 'method', 'implementation']):
                analysis['content_type'] = 'technical'
            elif any(term in content_lower for term in ['concept', 'theory', 'principle', 'framework']):
                analysis['content_type'] = 'conceptual'
            elif any(term in content_lower for term in ['project', 'task', 'goal', 'objective']):
                analysis['content_type'] = 'project'
            
            # Extract technical terms (capitalized words, acronyms)
            import re
            tech_pattern = re.compile(r'\b[A-Z][A-Za-z]*(?:[A-Z][a-z]*)*\b|\b[A-Z]{2,}\b')
            analysis['technical_terms'] = list(set(tech_pattern.findall(content)))
            
            # Calculate concept density (ratio of technical terms to total words)
            word_count = len(content.split())
            if word_count > 0:
                analysis['concept_density'] = len(analysis['technical_terms']) / word_count
                
        except Exception as e:
            logger.warning(f"Error analyzing content structure: {e}")
            
        return analysis

    def _extract_key_concepts(self, content: str, frontmatter: dict[str, any]) -> list[str]:
        """Extract key concepts and terminology from content and metadata.
        
        Args:
            content: Note content
            frontmatter: Frontmatter dictionary
            
        Returns:
            List of key concepts
        """
        key_concepts = []
        
        try:
            # Extract from frontmatter concept fields
            concept_fields = ['concepts', 'keywords', 'terms', 'topics']
            for field in concept_fields:
                if field in frontmatter:
                    value = frontmatter[field]
                    if isinstance(value, list):
                        key_concepts.extend(value)
                    elif isinstance(value, str):
                        key_concepts.extend([term.strip() for term in value.split(',')])
            
            # Extract from content (simple approach - could be enhanced with NLP)
            # Look for terms in bold or emphasized text
            import re
            bold_pattern = re.compile(r'\*\*([^*]+)\*\*|\*([^*]+)\*')
            bold_matches = bold_pattern.findall(content)
            for match in bold_matches:
                concept = match[0] or match[1]
                if concept and len(concept.split()) <= 3:  # Limit to short phrases
                    key_concepts.append(concept.strip())
            
            # Remove duplicates and filter
            key_concepts = list(set([concept for concept in key_concepts if concept and len(concept) > 2]))
            
        except Exception as e:
            logger.warning(f"Error extracting key concepts: {e}")
            
        return key_concepts[:10]  # Limit to top 10 concepts

    def _generate_technical_summary(self, content: str, key_concepts: list[str]) -> str:
        """Generate summary for technical content.
        
        Args:
            content: Note content
            key_concepts: List of key concepts
            
        Returns:
            Technical summary
        """
        # Find first paragraph that mentions key concepts
        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) > 50 and any(concept.lower() in paragraph.lower() for concept in key_concepts):
                # Clean up the paragraph
                clean_paragraph = re.sub(r'#+\s*', '', paragraph)  # Remove headers
                clean_paragraph = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_paragraph)  # Remove bold
                clean_paragraph = ' '.join(clean_paragraph.split())  # Normalize whitespace
                
                if len(clean_paragraph) > 200:
                    return clean_paragraph[:200] + "..."
                return clean_paragraph
        
        return self._generate_simple_summary(content)

    def _generate_conceptual_summary(self, content: str, key_concepts: list[str]) -> str:
        """Generate summary for conceptual content.
        
        Args:
            content: Note content
            key_concepts: List of key concepts
            
        Returns:
            Conceptual summary
        """
        # Look for definition-like sentences
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 30 and 
                any(word in sentence.lower() for word in ['is', 'are', 'refers to', 'means', 'defined as']) and
                any(concept.lower() in sentence.lower() for concept in key_concepts)):
                
                clean_sentence = re.sub(r'#+\s*', '', sentence)
                clean_sentence = ' '.join(clean_sentence.split())
                
                if len(clean_sentence) > 200:
                    return clean_sentence[:200] + "..."
                return clean_sentence
        
        return self._generate_simple_summary(content)

    def _generate_general_summary(self, content: str, content_analysis: dict[str, any]) -> str:
        """Generate summary for general content.
        
        Args:
            content: Note content
            content_analysis: Content analysis results
            
        Returns:
            General summary
        """
        # Use main topics from headings if available
        if content_analysis['main_topics']:
            topics = content_analysis['main_topics'][:3]  # First 3 topics
            return f"Note covering: {', '.join(topics)}"
        
        return self._generate_simple_summary(content)

    def _generate_simple_summary(self, content: str) -> str:
        """Generate simple extractive summary as fallback.
        
        Args:
            content: Note content
            
        Returns:
            Simple summary
        """
        # Simple extractive summary - take first meaningful sentence
        sentences = content.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.startswith('#'):
                return sentence[:200] + "..." if len(sentence) > 200 else sentence

        # Fallback to first 200 characters
        return content[:200].strip() + "..." if len(content) > 200 else content.strip()

    def _create_dataframe(self, features_list: list[NoteFeatures]) -> pd.DataFrame:
        """Create pandas DataFrame from list of NoteFeatures.
        
        Args:
            features_list: List of NoteFeatures objects
            
        Returns:
            DataFrame with note features
        """
        data = []

        for features in features_list:
            row = {
                # Basic note information
                'note_path': features.note_path,
                'note_title': features.note_title,
                'word_count': features.word_count,
                'tag_count': features.tag_count,
                'quality_stage': features.quality_stage,
                'creation_date': features.creation_date,
                'last_modified': features.last_modified,
                'file_size': features.file_size,
                'reading_time_minutes': features.reading_time_minutes,
                
                # Link and graph features
                'outgoing_links_count': features.outgoing_links_count,
                'incoming_links_count': features.incoming_links_count,
                'betweenness_centrality': features.betweenness_centrality,
                'closeness_centrality': features.closeness_centrality,
                'pagerank_score': features.pagerank_score,
                'clustering_coefficient': features.clustering_coefficient,
                'semantic_cluster_id': features.semantic_cluster_id,
                
                # Semantic features
                'semantic_summary': features.semantic_summary,
                
                # Enhanced frontmatter-derived features
                'aliases_count': features.aliases_count,
                'domains_count': features.domains_count,
                'concepts_count': features.concepts_count,
                'sources_count': features.sources_count,
                'has_summary_field': features.has_summary_field,
                'progress_state': features.progress_state,
                
                # Semantic relationship counts
                'semantic_up_links': features.semantic_up_links,
                'semantic_similar_links': features.semantic_similar_links,
                'semantic_leads_to_links': features.semantic_leads_to_links,
                'semantic_extends_links': features.semantic_extends_links,
                'semantic_implements_links': features.semantic_implements_links,
                
                # Content analysis features
                'heading_count': features.heading_count,
                'max_heading_depth': features.max_heading_depth,
                'technical_term_density': features.technical_term_density,
                'concept_density_score': features.concept_density_score
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Ensure proper data types
        numeric_columns = [
            'word_count', 'tag_count', 'outgoing_links_count', 'incoming_links_count',
            'betweenness_centrality', 'closeness_centrality', 'pagerank_score',
            'clustering_coefficient', 'semantic_cluster_id', 'file_size', 'reading_time_minutes',
            'aliases_count', 'domains_count', 'concepts_count', 'sources_count',
            'semantic_up_links', 'semantic_similar_links', 'semantic_leads_to_links',
            'semantic_extends_links', 'semantic_implements_links',
            'heading_count', 'max_heading_depth', 'technical_term_density', 'concept_density_score'
        ]

        boolean_columns = ['has_summary_field']

        # Convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Convert boolean columns
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        return df

    def validate_inputs(self, notes: list[str], link_graph: nx.DiGraph) -> ValidationResult:
        """Validate input parameters and data quality.
        
        Args:
            notes: List of note paths
            link_graph: NetworkX graph
            
        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(valid=True)

        # Check minimum notes requirement
        if len(notes) < 5:
            result.valid = False
            result.errors.append(f"Insufficient notes: {len(notes)} < 5 required")

        # Check if notes are accessible
        accessible_notes = 0
        for note_path in notes[:10]:  # Check first 10 for performance
            try:
                self.vault_reader.read_file(note_path)
                accessible_notes += 1
            except Exception:
                result.warnings.append(f"Cannot access note: {note_path}")

        if accessible_notes == 0:
            result.valid = False
            result.errors.append("No accessible notes found")

        # Check graph validity
        if not isinstance(link_graph, nx.DiGraph):
            result.valid = False
            result.errors.append("Invalid graph type")

        result.notes_processed = len(notes)
        return result

    def get_feature_descriptions(self) -> dict[str, str]:
        """Get descriptions of all generated features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            # Basic note information
            'note_path': 'Relative path to the note file',
            'note_title': 'Title of the note (filename without extension)',
            'word_count': 'Number of words in the note content',
            'tag_count': 'Number of tags associated with the note',
            'quality_stage': 'Quality stage of the note (from metadata or progress indicators)',
            'creation_date': 'Date when the note was created',
            'last_modified': 'Date when the note was last modified',
            'file_size': 'Size of the note file in bytes',
            'reading_time_minutes': 'Estimated reading time in minutes',
            
            # Link and graph features
            'outgoing_links_count': 'Number of outgoing links from this note',
            'incoming_links_count': 'Number of incoming links to this note',
            'betweenness_centrality': 'Betweenness centrality in the link graph',
            'closeness_centrality': 'Closeness centrality in the link graph',
            'pagerank_score': 'PageRank score in the link graph',
            'clustering_coefficient': 'Clustering coefficient in the link graph',
            'semantic_cluster_id': 'ID of semantic cluster (-1 if not clustered)',
            
            # Semantic features
            'semantic_summary': 'Intelligent semantic summary prioritizing frontmatter summary field',
            
            # Enhanced frontmatter-derived features
            'aliases_count': 'Number of aliases defined in frontmatter',
            'domains_count': 'Number of domains/categories defined in frontmatter',
            'concepts_count': 'Number of concepts defined in frontmatter',
            'sources_count': 'Number of sources/references defined in frontmatter',
            'has_summary_field': 'Whether the note has a summary field in frontmatter',
            'progress_state': 'Progress state indicator (🌱, 🌿, 🌲, ⚛️, 🗺️, ⧉, 🎓)',
            
            # Semantic relationship counts
            'semantic_up_links': 'Number of "up::" hierarchical relationship links',
            'semantic_similar_links': 'Number of "similar:" relationship links',
            'semantic_leads_to_links': 'Number of "leads to:" relationship links',
            'semantic_extends_links': 'Number of "extends:" relationship links',
            'semantic_implements_links': 'Number of "implements:" relationship links',
            
            # Content analysis features
            'heading_count': 'Number of headings in the note content',
            'max_heading_depth': 'Maximum heading depth (number of # symbols)',
            'technical_term_density': 'Density of technical terms (capitalized words/acronyms)',
            'concept_density_score': 'Conceptual density score based on technical terminology'
        }

    def get_processing_stats(self) -> ProcessingStats | None:
        """Get processing statistics from the last generation run.
        
        Returns:
            ProcessingStats object or None if no processing has occurred
        """
        return self._processing_stats

    def _create_memory_monitor(self):
        """Create a memory monitor for tracking memory usage."""
        return MemoryMonitor()

    def _adjust_batch_size(self, current_batch_size: int, memory_monitor, processed_count: int) -> int:
        """Adjust batch size based on memory usage and performance.
        
        Args:
            current_batch_size: Current batch size
            memory_monitor: Memory monitoring instance
            processed_count: Number of items processed so far
            
        Returns:
            Adjusted batch size
        """
        try:
            current_memory = memory_monitor.get_current_usage()
            
            # If memory usage is high, reduce batch size
            if current_memory > 1500:  # 1.5GB threshold
                new_size = max(8, current_batch_size // 2)
                logger.debug(f"Reducing batch size from {current_batch_size} to {new_size} due to high memory usage ({current_memory:.1f}MB)")
                return new_size
            
            # If memory usage is low and we're processing efficiently, increase batch size
            elif current_memory < 500 and processed_count > 50:  # 500MB threshold, after some processing
                new_size = min(64, current_batch_size * 2)
                logger.debug(f"Increasing batch size from {current_batch_size} to {new_size} due to low memory usage ({current_memory:.1f}MB)")
                return new_size
            
            return current_batch_size
            
        except Exception as e:
            logger.debug(f"Failed to adjust batch size: {e}")
            return current_batch_size

    def _process_note_batch_optimized(self, batch_notes: list[str], link_graph: nx.DiGraph,
                                    centrality_cache: dict[str, CentralityMetrics], 
                                    memory_monitor) -> list[NoteFeatures | None]:
        """Process a batch of notes with memory optimization.
        
        Args:
            batch_notes: List of note paths in this batch
            link_graph: NetworkX graph for centrality calculations
            centrality_cache: Pre-computed centrality metrics
            memory_monitor: Memory monitoring instance
            
        Returns:
            List of NoteFeatures objects (None for failed notes)
        """
        batch_features = []
        
        try:
            # Load note data for batch with memory monitoring
            note_data_batch = []
            for note_path in batch_notes:
                try:
                    note_data = self._load_note_data(note_path)
                    note_data_batch.append(note_data)
                    
                    # Check memory after loading each note
                    if memory_monitor.should_pause():
                        logger.debug("Pausing for memory management during note loading")
                        gc.collect()
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.warning(f"Failed to load note data for {note_path}: {e}")
                    note_data_batch.append(None)

            # Generate embeddings for valid notes in batch with chunking
            valid_notes = [data for data in note_data_batch if data is not None]
            if valid_notes:
                try:
                    # Process embeddings in smaller chunks to manage memory
                    embedding_chunk_size = min(16, len(valid_notes))
                    all_embeddings = []
                    
                    for i in range(0, len(valid_notes), embedding_chunk_size):
                        chunk = valid_notes[i:i + embedding_chunk_size]
                        contents = [data.content for data in chunk]
                        
                        chunk_embeddings = self.vector_encoder.encode_documents(contents)
                        all_embeddings.extend(chunk_embeddings)
                        
                        # Memory cleanup after each chunk
                        if memory_monitor.should_pause():
                            gc.collect()

                    # Assign embeddings back to note data
                    embedding_idx = 0
                    for data in note_data_batch:
                        if data is not None:
                            data.embedding = all_embeddings[embedding_idx]
                            embedding_idx += 1

                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch: {e}")
                    # Set embeddings to None for error handling
                    for data in note_data_batch:
                        if data is not None:
                            data.embedding = None

            # Extract features for each note with memory monitoring
            for note_data in note_data_batch:
                if note_data is None:
                    batch_features.append(None)
                    continue

                try:
                    features = self._extract_note_features(note_data, link_graph, centrality_cache)
                    batch_features.append(features)
                    
                    # Periodic memory check during feature extraction
                    if len(batch_features) % 8 == 0 and memory_monitor.should_pause():
                        gc.collect()
                        
                except Exception as e:
                    logger.warning(f"Failed to extract features for {note_data.path}: {e}")
                    batch_features.append(None)

            return batch_features
            
        except Exception as e:
            logger.error(f"Optimized batch processing failed: {e}")
            # Fallback to regular batch processing
            return self._process_note_batch(batch_notes, link_graph, centrality_cache)

    def _create_dataframe_optimized(self, note_features_list: list[NoteFeatures], 
                                  memory_monitor) -> pd.DataFrame:
        """Create DataFrame with memory optimization.
        
        Args:
            note_features_list: List of NoteFeatures objects
            memory_monitor: Memory monitoring instance
            
        Returns:
            Optimized DataFrame
        """
        try:
            logger.info("Creating optimized DataFrame from note features")
            
            # Convert features to dictionaries in chunks to manage memory
            chunk_size = 1000
            all_data = []
            
            for i in range(0, len(note_features_list), chunk_size):
                chunk = note_features_list[i:i + chunk_size]
                chunk_data = []
                
                for features in chunk:
                    if features is not None:
                        # Convert NoteFeatures to dictionary
                        feature_dict = {
                            'note_path': features.note_path,
                            'note_title': features.note_title,
                            'word_count': features.word_count,
                            'tag_count': features.tag_count,
                            'quality_stage': features.quality_stage,
                            'creation_date': features.creation_date,
                            'last_modified': features.last_modified,
                            'outgoing_links_count': features.outgoing_links_count,
                            'incoming_links_count': features.incoming_links_count,
                            'betweenness_centrality': features.betweenness_centrality,
                            'closeness_centrality': features.closeness_centrality,
                            'pagerank_score': features.pagerank_score,
                            'clustering_coefficient': features.clustering_coefficient,
                            'semantic_cluster_id': features.semantic_cluster_id,
                            'semantic_summary': features.semantic_summary,
                            'file_size': features.file_size,
                            'reading_time_minutes': features.reading_time_minutes,
                            'aliases_count': features.aliases_count,
                            'domains_count': features.domains_count,
                            'concepts_count': features.concepts_count,
                            'sources_count': features.sources_count,
                            'has_summary_field': features.has_summary_field,
                            'progress_state': features.progress_state,
                            'semantic_up_links': features.semantic_up_links,
                            'semantic_similar_links': features.semantic_similar_links,
                            'semantic_leads_to_links': features.semantic_leads_to_links,
                            'semantic_extends_links': features.semantic_extends_links,
                            'semantic_implements_links': features.semantic_implements_links,
                            'heading_count': features.heading_count,
                            'max_heading_depth': features.max_heading_depth,
                            'technical_term_density': features.technical_term_density,
                            'concept_density_score': features.concept_density_score
                        }
                        chunk_data.append(feature_dict)
                
                all_data.extend(chunk_data)
                
                # Memory cleanup after each chunk
                if memory_monitor.should_pause():
                    gc.collect()
                    
            # Create DataFrame with memory-efficient dtypes
            df = pd.DataFrame(all_data)
            
            # Optimize data types to reduce memory usage
            df = self._optimize_dataframe_dtypes(df)
            
            logger.info(f"DataFrame created with {len(df)} rows and optimized dtypes")
            return df
            
        except Exception as e:
            logger.error(f"Optimized DataFrame creation failed: {e}")
            # Fallback to simple DataFrame creation
            return self._create_dataframe(note_features_list)

    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types to reduce memory usage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized dtypes
        """
        try:
            # Convert integer columns to smaller dtypes where possible
            int_columns = ['word_count', 'tag_count', 'outgoing_links_count', 'incoming_links_count',
                          'semantic_cluster_id', 'file_size', 'aliases_count', 'domains_count',
                          'concepts_count', 'sources_count', 'semantic_up_links', 'semantic_similar_links',
                          'semantic_leads_to_links', 'semantic_extends_links', 'semantic_implements_links',
                          'heading_count', 'max_heading_depth']
            
            for col in int_columns:
                if col in df.columns:
                    # Use smallest possible integer type
                    max_val = df[col].max()
                    if max_val <= 127:
                        df[col] = df[col].astype('int8')
                    elif max_val <= 32767:
                        df[col] = df[col].astype('int16')
                    elif max_val <= 2147483647:
                        df[col] = df[col].astype('int32')
            
            # Convert float columns to float32 where precision allows
            float_columns = ['betweenness_centrality', 'closeness_centrality', 'pagerank_score',
                           'clustering_coefficient', 'reading_time_minutes', 'technical_term_density',
                           'concept_density_score']
            
            for col in float_columns:
                if col in df.columns:
                    df[col] = df[col].astype('float32')
            
            # Convert boolean columns
            bool_columns = ['has_summary_field']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype('bool')
            
            # Convert categorical columns
            categorical_columns = ['quality_stage', 'progress_state']
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].astype('category')
            
            logger.debug("DataFrame dtypes optimized for memory efficiency")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to optimize DataFrame dtypes: {e}")
            return df

    def _create_dataframe(self, note_features_list: list[NoteFeatures]) -> pd.DataFrame:
        """Create DataFrame from note features (fallback method).
        
        Args:
            note_features_list: List of NoteFeatures objects
            
        Returns:
            DataFrame containing note features
        """
        data = []
        for features in note_features_list:
            if features is not None:
                data.append({
                    'note_path': features.note_path,
                    'note_title': features.note_title,
                    'word_count': features.word_count,
                    'tag_count': features.tag_count,
                    'quality_stage': features.quality_stage,
                    'creation_date': features.creation_date,
                    'last_modified': features.last_modified,
                    'outgoing_links_count': features.outgoing_links_count,
                    'incoming_links_count': features.incoming_links_count,
                    'betweenness_centrality': features.betweenness_centrality,
                    'closeness_centrality': features.closeness_centrality,
                    'pagerank_score': features.pagerank_score,
                    'clustering_coefficient': features.clustering_coefficient,
                    'semantic_cluster_id': features.semantic_cluster_id,
                    'semantic_summary': features.semantic_summary,
                    'file_size': features.file_size,
                    'reading_time_minutes': features.reading_time_minutes,
                    'aliases_count': features.aliases_count,
                    'domains_count': features.domains_count,
                    'concepts_count': features.concepts_count,
                    'sources_count': features.sources_count,
                    'has_summary_field': features.has_summary_field,
                    'progress_state': features.progress_state,
                    'semantic_up_links': features.semantic_up_links,
                    'semantic_similar_links': features.semantic_similar_links,
                    'semantic_leads_to_links': features.semantic_leads_to_links,
                    'semantic_extends_links': features.semantic_extends_links,
                    'semantic_implements_links': features.semantic_implements_links,
                    'heading_count': features.heading_count,
                    'max_heading_depth': features.max_heading_depth,
                    'technical_term_density': features.technical_term_density,
                    'concept_density_score': features.concept_density_score
                })
        
        return pd.DataFrame(data)


class MemoryMonitor:
    """Monitor memory usage and provide memory management utilities."""
    
    def __init__(self, threshold_mb: int = 1000):
        """Initialize memory monitor.
        
        Args:
            threshold_mb: Memory threshold in MB for triggering pauses
        """
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.peak_usage = 0
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
    
    def get_current_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Current memory usage in MB
        """
        if self.process:
            try:
                usage_bytes = self.process.memory_info().rss
                usage_mb = usage_bytes / (1024 * 1024)
                self.peak_usage = max(self.peak_usage, usage_mb)
                return usage_mb
            except Exception:
                return 0.0
        return 0.0
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB.
        
        Returns:
            Peak memory usage in MB
        """
        return self.peak_usage
    
    def should_pause(self) -> bool:
        """Check if processing should pause for memory management.
        
        Returns:
            True if memory usage is above threshold
        """
        if self.process:
            try:
                current_bytes = self.process.memory_info().rss
                return current_bytes > self.threshold_bytes
            except Exception:
                return False
        return False
    
    def get_memory_info(self) -> dict[str, float]:
        """Get detailed memory information.
        
        Returns:
            Dictionary with memory statistics
        """
        if self.process:
            try:
                memory_info = self.process.memory_info()
                return {
                    'rss_mb': memory_info.rss / (1024 * 1024),
                    'vms_mb': memory_info.vms / (1024 * 1024),
                    'peak_mb': self.peak_usage,
                    'threshold_mb': self.threshold_bytes / (1024 * 1024)
                }
            except Exception:
                pass
        
        return {
            'rss_mb': 0.0,
            'vms_mb': 0.0,
            'peak_mb': self.peak_usage,
            'threshold_mb': self.threshold_bytes / (1024 * 1024)
        }