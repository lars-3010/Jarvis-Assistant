"""
Streaming processing utilities for large dataset generation.

This module provides streaming processing capabilities to handle large datasets
without loading all data into memory at once, preventing memory overflow.
"""

import csv
import tempfile
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional
import pandas as pd

from jarvis.utils.logging import setup_logging
from ..models.data_models import PairFeatures, NoteFeatures

logger = setup_logging(__name__)


class StreamingPairGenerator:
    """Generate note pairs in streaming fashion to avoid memory overflow."""

    def __init__(self, max_memory_pairs: int = 10000):
        """Initialize streaming pair generator.
        
        Args:
            max_memory_pairs: Maximum pairs to keep in memory at once
        """
        self.max_memory_pairs = max_memory_pairs
        self.temp_files = []

    def stream_pairs(self, all_notes: List[str], positive_pairs: set[tuple[str, str]], 
                    negative_pairs: List[tuple[str, str]]) -> Iterator[tuple[str, str, bool]]:
        """Stream all pairs without loading them all into memory.
        
        Args:
            all_notes: List of all note paths
            positive_pairs: Set of positive pairs
            negative_pairs: List of negative pairs
            
        Yields:
            Tuples of (note_a, note_b, is_positive)
        """
        logger.info(f"Streaming {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
        
        # Stream positive pairs first
        for pair in positive_pairs:
            yield (pair[0], pair[1], True)
        
        # Stream negative pairs
        for pair in negative_pairs:
            yield (pair[0], pair[1], False)

    def stream_pair_combinations(self, all_notes: List[str], 
                               positive_pairs: set[tuple[str, str]],
                               max_pairs: Optional[int] = None) -> Iterator[tuple[str, str, bool]]:
        """Stream all possible pair combinations, marking positive/negative.
        
        Args:
            all_notes: List of all note paths
            positive_pairs: Set of known positive pairs
            max_pairs: Maximum number of pairs to generate (None for all)
            
        Yields:
            Tuples of (note_a, note_b, is_positive)
        """
        logger.info(f"Streaming pair combinations from {len(all_notes)} notes")
        
        pair_count = 0
        for i, note_a in enumerate(all_notes):
            for j, note_b in enumerate(all_notes[i+1:], i+1):
                if max_pairs and pair_count >= max_pairs:
                    return
                
                pair = tuple(sorted([note_a, note_b]))
                is_positive = pair in positive_pairs
                
                yield (note_a, note_b, is_positive)
                pair_count += 1
                
                if pair_count % 10000 == 0:
                    logger.debug(f"Streamed {pair_count} pairs")

    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files.clear()


class StreamingDatasetWriter:
    """Write dataset results in streaming fashion to avoid memory buildup."""

    def __init__(self, output_path: Path, buffer_size: int = 1000):
        """Initialize streaming dataset writer.
        
        Args:
            output_path: Path to output CSV file
            buffer_size: Number of rows to buffer before writing
        """
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.buffer = []
        self.csv_writer = None
        self.csv_file = None
        self.headers_written = False

    def __enter__(self):
        """Enter context manager."""
        self.csv_file = open(self.output_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.buffer:
            self._flush_buffer()
        if self.csv_file:
            self.csv_file.close()

    def write_features(self, features: PairFeatures | NoteFeatures):
        """Write features to the streaming output.
        
        Args:
            features: Features object to write
        """
        # Convert features to dictionary
        if isinstance(features, PairFeatures):
            feature_dict = self._pair_features_to_dict(features)
        elif isinstance(features, NoteFeatures):
            feature_dict = self._note_features_to_dict(features)
        else:
            logger.warning(f"Unknown features type: {type(features)}")
            return

        # Write headers if not done yet
        if not self.headers_written:
            self.csv_writer.writerow(feature_dict.keys())
            self.headers_written = True

        # Add to buffer
        self.buffer.append(list(feature_dict.values()))

        # Flush buffer if full
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush the current buffer to file."""
        if self.buffer and self.csv_writer:
            self.csv_writer.writerows(self.buffer)
            self.csv_file.flush()
            self.buffer.clear()

    def _pair_features_to_dict(self, features: PairFeatures) -> Dict[str, Any]:
        """Convert PairFeatures to dictionary."""
        return {
            'note_a_path': features.note_a_path,
            'note_b_path': features.note_b_path,
            'cosine_similarity': features.cosine_similarity,
            'tfidf_similarity': features.tfidf_similarity,
            'combined_similarity': features.combined_similarity,
            'semantic_cluster_match': features.semantic_cluster_match,
            'tag_overlap_count': features.tag_overlap_count,
            'tag_jaccard_similarity': features.tag_jaccard_similarity,
            'vault_path_distance': features.vault_path_distance,
            'shortest_path_length': features.shortest_path_length,
            'common_neighbors_count': features.common_neighbors_count,
            'adamic_adar_score': features.adamic_adar_score,
            'word_count_ratio': features.word_count_ratio,
            'creation_time_diff_days': features.creation_time_diff_days,
            'quality_stage_compatibility': features.quality_stage_compatibility,
            'source_centrality': features.source_centrality,
            'target_centrality': features.target_centrality,
            'clustering_coefficient': features.clustering_coefficient,
            'link_exists': features.link_exists,
            'title_similarity': getattr(features, 'title_similarity', 0.0)
        }

    def _note_features_to_dict(self, features: NoteFeatures) -> Dict[str, Any]:
        """Convert NoteFeatures to dictionary."""
        return {
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


class DiskBasedCache:
    """Disk-based cache for intermediate results to prevent memory overflow."""

    def __init__(self, cache_dir: Optional[Path] = None, max_memory_items: int = 1000):
        """Initialize disk-based cache.
        
        Args:
            cache_dir: Directory for cache files (None for temp directory)
            max_memory_items: Maximum items to keep in memory cache
        """
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp(prefix="jarvis_cache_"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_items = max_memory_items
        self.memory_cache = {}
        self.disk_files = {}
        
        logger.debug(f"Initialized disk cache at: {self.cache_dir}")

    def put(self, key: str, value: Any):
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If memory cache is full, move oldest items to disk
        if len(self.memory_cache) >= self.max_memory_items:
            self._move_to_disk()
        
        self.memory_cache[key] = value

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        if key in self.disk_files:
            try:
                disk_file = self.disk_files[key]
                with open(disk_file, 'rb') as f:
                    import pickle
                    value = pickle.load(f)
                
                # Move back to memory cache
                self.memory_cache[key] = value
                return value
                
            except Exception as e:
                logger.warning(f"Failed to load from disk cache {key}: {e}")
                # Remove invalid disk file
                if key in self.disk_files:
                    try:
                        self.disk_files[key].unlink()
                    except:
                        pass
                    del self.disk_files[key]
        
        return None

    def _move_to_disk(self):
        """Move half of memory cache items to disk."""
        items_to_move = len(self.memory_cache) // 2
        
        for i, (key, value) in enumerate(list(self.memory_cache.items())):
            if i >= items_to_move:
                break
            
            try:
                # Create disk file
                disk_file = self.cache_dir / f"cache_{hash(key) % 10000}.pkl"
                with open(disk_file, 'wb') as f:
                    import pickle
                    pickle.dump(value, f)
                
                # Update tracking
                self.disk_files[key] = disk_file
                del self.memory_cache[key]
                
            except Exception as e:
                logger.warning(f"Failed to move {key} to disk cache: {e}")

    def clear(self):
        """Clear all cache data."""
        self.memory_cache.clear()
        
        # Remove disk files
        for disk_file in self.disk_files.values():
            try:
                if disk_file.exists():
                    disk_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove disk cache file {disk_file}: {e}")
        
        self.disk_files.clear()
        
        # Remove cache directory if empty
        try:
            if self.cache_dir.exists() and not any(self.cache_dir.iterdir()):
                self.cache_dir.rmdir()
        except Exception as e:
            logger.warning(f"Failed to remove cache directory {self.cache_dir}: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.clear()
        except:
            pass


class StreamingProcessor:
    """Main streaming processor for large dataset generation."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize streaming processor.
        
        Args:
            cache_dir: Directory for temporary cache files
        """
        self.cache_dir = cache_dir
        self.disk_cache = DiskBasedCache(cache_dir)
        self.temp_files = []

    def process_large_dataset(self, notes_data: Dict[str, Any], 
                            output_path: Path,
                            processing_func,
                            chunk_size: int = 1000) -> int:
        """Process large dataset in streaming fashion.
        
        Args:
            notes_data: Dictionary of note data
            output_path: Path for output file
            processing_func: Function to process each chunk
            chunk_size: Size of each processing chunk
            
        Returns:
            Number of items processed
        """
        logger.info(f"Starting streaming processing with chunk size {chunk_size}")
        
        total_processed = 0
        
        try:
            with StreamingDatasetWriter(output_path) as writer:
                # Process data in chunks
                notes_list = list(notes_data.keys())
                
                for i in range(0, len(notes_list), chunk_size):
                    chunk = notes_list[i:i + chunk_size]
                    chunk_data = {k: notes_data[k] for k in chunk}
                    
                    logger.debug(f"Processing chunk {i//chunk_size + 1}: {len(chunk)} items")
                    
                    # Process chunk
                    try:
                        results = processing_func(chunk_data)
                        
                        # Write results
                        for result in results:
                            if result is not None:
                                writer.write_features(result)
                                total_processed += 1
                                
                    except Exception as e:
                        logger.error(f"Failed to process chunk {i//chunk_size + 1}: {e}")
                        continue
                    
                    # Log progress
                    if (i // chunk_size + 1) % 10 == 0:
                        logger.info(f"Processed {total_processed} items in {i//chunk_size + 1} chunks")
            
            logger.info(f"Streaming processing completed: {total_processed} items processed")
            return total_processed
            
        except Exception as e:
            logger.error(f"Streaming processing failed: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up temporary resources."""
        try:
            self.disk_cache.clear()
            
            for temp_file in self.temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
            
            self.temp_files.clear()
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass