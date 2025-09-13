"""
Document indexing service for vector database.

This module handles the indexing of documents into the vector database,
including text extraction, embedding generation, and storage.
"""

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.config import get_settings
from jarvis.utils.errors import ConfigurationError, ServiceError
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)


class IndexingStats:
    """Statistics for indexing operations."""

    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.total_time = 0.0
        self.embedding_time = 0.0
        self.storage_time = 0.0
        self.start_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'skipped_files': self.skipped_files,
            'failed_files': self.failed_files,
            'total_time': self.total_time,
            'embedding_time': self.embedding_time,
            'storage_time': self.storage_time,
            'files_per_second': self.processed_files / max(self.total_time, 0.001)
        }


class VectorIndexer:
    """Document indexing service for vector search."""

    def __init__(
        self,
        database: VectorDatabase,
        encoder: VectorEncoder,
        vaults: dict[str, Path],
        batch_size: int | None = None
    ):
        """Initialize the indexer.
        
        Args:
            database: Vector database instance
            encoder: Vector encoder instance
            vaults: Dictionary mapping vault names to paths
            batch_size: Batch size for processing (from settings if None)
        """
        self.database = database
        self.encoder = encoder
        self.vaults = vaults

        # Get batch size from settings if not provided
        if batch_size is None:
            settings = get_settings()
            batch_size = getattr(settings, 'index_batch_size', 32)

        self.batch_size = batch_size
        logger.info(f"Initialized indexer with batch size: {self.batch_size}")

    def index_files(
        self,
        vault_name_paths: Sequence[tuple[str, Path]],
        force_reindex: bool = False,
        progress_callback: callable = None
    ) -> IndexingStats:
        """Index a collection of files.
        
        Args:
            vault_name_paths: Sequence of (vault_name, file_path) tuples
            force_reindex: Whether to reindex files even if they haven't changed
            progress_callback: Optional callback for progress updates
            
        Returns:
            IndexingStats object with processing statistics
        """
        stats = IndexingStats()
        stats.total_files = len(vault_name_paths)

        if not vault_name_paths:
            logger.info("No files to index")
            return stats

        logger.info(f"Starting indexing of {stats.total_files} files")

        # Filter files that need indexing
        files_to_process = []
        for vault_name, path in vault_name_paths:
            try:
                if self._should_index_file(vault_name, path, force_reindex):
                    files_to_process.append((vault_name, path))
                else:
                    stats.skipped_files += 1
            except Exception as e:
                logger.error(f"Error checking file {path}: {e}")
                stats.failed_files += 1
                raise ServiceError(f"Error checking file {path}: {e}") from e

        logger.info(f"Processing {len(files_to_process)} files, skipping {stats.skipped_files}")

        # Process files in batches
        for i in range(0, len(files_to_process), self.batch_size):
            batch = files_to_process[i:i + self.batch_size]
            batch_stats = self._process_batch(batch)

            # Update overall stats
            stats.processed_files += batch_stats.processed_files
            stats.failed_files += batch_stats.failed_files
            stats.embedding_time += batch_stats.embedding_time
            stats.storage_time += batch_stats.storage_time

            # Progress callback
            if progress_callback:
                progress_callback(stats.processed_files, stats.total_files)

            logger.debug(f"Batch {i//self.batch_size + 1}: processed {batch_stats.processed_files}, failed {batch_stats.failed_files}")

        stats.total_time = time.time() - stats.start_time

        logger.info(
            f"Indexing complete: {stats.processed_files} processed, "
            f"{stats.skipped_files} skipped, {stats.failed_files} failed "
            f"in {stats.total_time:.2f}s"
        )

        return stats

    def _process_batch(self, batch: list[tuple[str, Path]]) -> IndexingStats:
        """Process a batch of files.
        
        Args:
            batch: List of (vault_name, file_path) tuples
            
        Returns:
            IndexingStats for this batch
        """
        stats = IndexingStats()

        if not batch:
            return stats

        # Load texts
        texts = []
        valid_files = []

        for vault_name, path in batch:
            try:
                text = self._load_file_text(path)
                if text:
                    texts.append(text)
                    valid_files.append((vault_name, path))
                else:
                    logger.warning(f"Empty file content: {path}")
                    stats.skipped_files += 1
            except Exception as e:
                logger.error(f"Failed to load file {path}: {e}")
                stats.failed_files += 1

        if not texts:
            return stats

        # Generate embeddings
        try:
            embedding_start = time.time()
            embeddings = self.encoder.encode_documents(texts, batch_size=self.batch_size)
            stats.embedding_time = time.time() - embedding_start

            logger.debug(f"Generated embeddings for {len(texts)} files in {stats.embedding_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch: {e}")
            stats.failed_files += len(valid_files)
            raise ServiceError(f"Failed to generate embeddings for batch: {e}") from e

        # Store in database
        storage_start = time.time()
        for (vault_name, path), embedding in zip(valid_files, embeddings, strict=False):
            try:
                # Calculate relative path
                vault_path = self.vaults[vault_name]
                relative_path = path.relative_to(vault_path)

                # Get file metadata
                stat = path.stat()
                last_modified = stat.st_mtime

                # Calculate checksum for change detection
                checksum = self._calculate_checksum(path)

                # Store in database
                success = self.database.store_note(
                    path=relative_path,
                    vault_name=vault_name,
                    last_modified=last_modified,
                    embedding=embedding.tolist(),
                    checksum=checksum
                )

                if success:
                    stats.processed_files += 1
                else:
                    stats.failed_files += 1

            except Exception as e:
                logger.error(f"Failed to store file {path}: {e}")
                stats.failed_files += 1
                raise ServiceError(f"Failed to store file {path}: {e}") from e

        stats.storage_time = time.time() - storage_start

        return stats

    def _should_index_file(self, vault_name: str, path: Path, force_reindex: bool) -> bool:
        """Determine if a file should be indexed.
        
        Args:
            vault_name: Name of the vault
            path: Path to the file
            force_reindex: Whether to force reindexing
            
        Returns:
            True if file should be indexed
        """
        if force_reindex:
            return True

        # Check if file exists in database
        try:
            vault_path = self.vaults[vault_name]
            relative_path = path.relative_to(vault_path)

            existing_note = self.database.get_note_by_path(vault_name, relative_path)

            if not existing_note:
                return True  # New file

            # Check if file has been modified
            current_mtime = path.stat().st_mtime
            stored_mtime = existing_note.get('last_modified', 0)

            if current_mtime > stored_mtime:
                return True  # File modified

            # Optionally check checksum for more accurate change detection
            if 'checksum' in existing_note:
                current_checksum = self._calculate_checksum(path)
                if current_checksum != existing_note['checksum']:
                    return True  # Content changed

            return False  # No changes detected

        except Exception as e:
            logger.warning(f"Error checking file status {path}: {e}")
            raise ServiceError(f"Error checking file status {path}: {e}") from e

    def _load_file_text(self, path: Path) -> str:
        """Load text content from a file.
        
        Args:
            path: Path to the file
            
        Returns:
            File content as string
        """
        try:
            # Try UTF-8 first
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                return path.read_text(encoding='latin-1')
            except Exception as e:
                logger.error(f"Failed to read file {path} with fallback encoding: {e}")
                raise ServiceError(f"Failed to read file {path} with fallback encoding: {e}") from e
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return ""

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate MD5 checksum of file content.
        
        Args:
            path: Path to the file
            
        Returns:
            MD5 checksum as hex string
        """
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {path}: {e}")
            raise ServiceError(f"Failed to calculate checksum for {path}: {e}") from e

    def index_vault(
        self,
        vault_name: str,
        file_patterns: list[str] | None = None,
        force_reindex: bool = False,
        progress_callback: callable = None
    ) -> IndexingStats:
        """Index all files in a vault.
        
        Args:
            vault_name: Name of the vault to index
            file_patterns: File patterns to include (defaults to *.md)
            force_reindex: Whether to force reindexing of all files
            progress_callback: Optional callback for progress updates
            
        Returns:
            IndexingStats object with processing statistics
        """
        if vault_name not in self.vaults:
            raise ConfigurationError(f"Unknown vault: {vault_name}")

        vault_path = self.vaults[vault_name]
        if not vault_path.exists():
            raise ConfigurationError(f"Vault path does not exist: {vault_path}")

        # Default to markdown files
        if file_patterns is None:
            file_patterns = ['*.md']

        # Find all matching files
        files_to_index = []
        for pattern in file_patterns:
            files_to_index.extend([
                (vault_name, path) for path in vault_path.rglob(pattern)
                if path.is_file()
            ])

        logger.info(f"Found {len(files_to_index)} files in vault '{vault_name}' matching patterns: {file_patterns}")

        return self.index_files(files_to_index, force_reindex, progress_callback)

    def get_indexing_stats(self) -> dict[str, Any]:
        """Get general indexing statistics.
        
        Returns:
            Dictionary with indexing statistics
        """
        total_notes = self.database.num_notes()
        vault_stats = {}

        for vault_name in self.vaults.keys():
            vault_stats[vault_name] = self.database.get_vault_stats(vault_name)

        return {
            'total_notes': total_notes,
            'vault_stats': vault_stats,
            'configured_vaults': list(self.vaults.keys()),
            'batch_size': self.batch_size
        }
