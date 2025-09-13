"""
ChromaDB Vector Database Adapter.

This module provides a ChromaDB implementation of the IVectorDatabase interface,
allowing the system to use ChromaDB as an alternative vector database backend.
"""

from collections.abc import Sequence
from pathlib import Path

import torch

from jarvis.core.interfaces import IVectorDatabase
from jarvis.utils.errors import JarvisError, ServiceError
import logging

logger = logging.getLogger(__name__)


class ChromaVectorDatabase(IVectorDatabase):
    """ChromaDB-based vector database for document embeddings."""

    def __init__(self, collection_name: str = "jarvis-embeddings",
                 persist_directory: str | None = None,
                 host: str | None = None,
                 port: int | None = None):
        """Initialize the ChromaDB vector database.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the database (for local mode)
            host: ChromaDB server host (for server mode)
            port: ChromaDB server port (for server mode)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ServiceError("ChromaDB is not installed. Install with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.host = host
        self.port = port

        try:
            if host and port:
                # Server mode
                self.client = chromadb.HttpClient(host=host, port=port)
                logger.info(f"Connected to ChromaDB server at {host}:{port}")
            # Local mode with persistence
            elif persist_directory:
                Path(persist_directory).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
                logger.info(f"Connected to persistent ChromaDB at {persist_directory}")
            else:
                self.client = chromadb.Client()
                logger.info("Connected to in-memory ChromaDB")

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

        except Exception as e:
            raise ServiceError(f"Failed to connect to ChromaDB: {e}") from e

    @classmethod
    def from_config(cls, config) -> "ChromaVectorDatabase":
        """Create ChromaVectorDatabase instance from DatabaseConfig.
        
        Args:
            config: DatabaseConfig instance with ChromaDB configuration
            
        Returns:
            ChromaVectorDatabase instance
        """
        return cls(
            collection_name=config.get('collection_name', 'jarvis-embeddings'),
            persist_directory=config.get('persist_directory'),
            host=config.get('host'),
            port=config.get('port')
        )

    def close(self) -> None:
        """Close the database connection."""
        # ChromaDB client doesn't require explicit closing
        logger.info("ChromaDB connection closed")

    def num_notes(self) -> int:
        """Get the total number of notes in the database."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count notes: {e}")
            raise JarvisError(f"Failed to count notes: {e}") from e

    def get_most_recent_seen_timestamp(self, vault_name: str) -> float | None:
        """Get the most recent seen timestamp for a vault."""
        try:
            # Query all documents for this vault
            results = self.collection.get(
                where={"vault_name": vault_name}
            )

            if not results['metadatas']:
                return None

            # Find the most recent timestamp
            timestamps = [
                metadata.get('last_modified', 0)
                for metadata in results['metadatas']
                if metadata.get('last_modified')
            ]

            return max(timestamps) if timestamps else None

        except Exception as e:
            logger.error(f"Failed to get recent timestamp for vault {vault_name}: {e}")
            raise JarvisError(f"Failed to get recent timestamp for vault {vault_name}: {e}") from e

    def store_note(
        self,
        path: Path,
        vault_name: str,
        last_modified: float,
        embedding: list[float],
        checksum: str | None = None
    ) -> bool:
        """Store a note in the database.
        
        Args:
            path: Path to the note (relative to vault root)
            vault_name: Name of the vault containing the note
            last_modified: Last modification timestamp
            embedding: 384-dimensional embedding vector
            checksum: Optional content checksum for change detection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            document_id = f"{vault_name}::{path!s}"

            # Check if document already exists
            existing = self.collection.get(ids=[document_id])

            metadata = {
                "vault_name": vault_name,
                "path": str(path),
                "last_modified": last_modified,
                "checksum": checksum
            }

            if existing['ids']:
                # Update existing document
                self.collection.update(
                    ids=[document_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
            else:
                # Add new document
                self.collection.add(
                    ids=[document_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )

            logger.debug(f"Stored note: {vault_name}/{path}")
            return True

        except Exception as e:
            logger.error(f"Failed to store note {vault_name}/{path}: {e}")
            raise JarvisError(f"Failed to store note {vault_name}/{path}: {e}") from e

    def search(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 10,
        vault_name: str | None = None
    ) -> Sequence[tuple[str, Path, float]]:
        """Search for notes similar to a query embedding.
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Maximum number of results to return
            vault_name: Optional vault name filter
            
        Returns:
            List of (vault_name, path, similarity_score) tuples
        """
        try:
            # Prepare query filter
            where_filter = None
            if vault_name:
                where_filter = {"vault_name": vault_name}

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_filter
            )

            # Format results
            search_results = []
            if results['metadatas'] and results['distances']:
                for metadata, distance in zip(results['metadatas'][0], results['distances'][0], strict=False):
                    vault = metadata['vault_name']
                    path = Path(metadata['path'])
                    # Convert distance to similarity score (ChromaDB returns distances)
                    similarity = 1.0 - distance
                    search_results.append((vault, path, similarity))

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise JarvisError(f"Search failed: {e}") from e

    def get_note_by_path(self, vault_name: str, path: Path) -> dict | None:
        """Get a specific note by vault name and path."""
        try:
            document_id = f"{vault_name}::{path!s}"
            results = self.collection.get(
                ids=[document_id],
                include=['metadatas', 'embeddings']
            )

            if results['ids']:
                metadata = results['metadatas'][0]
                embedding = results['embeddings'][0] if results['embeddings'] else None

                return {
                    'path': metadata['path'],
                    'vault_name': metadata['vault_name'],
                    'last_modified': metadata['last_modified'],
                    'embedding': embedding,
                    'checksum': metadata.get('checksum')
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get note {vault_name}/{path}: {e}")
            raise JarvisError(f"Failed to get note {vault_name}/{path}: {e}") from e

    def delete_note(self, vault_name: str, path: Path) -> bool:
        """Delete a note from the database."""
        try:
            document_id = f"{vault_name}::{path!s}"
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted note: {vault_name}/{path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete note {vault_name}/{path}: {e}")
            raise JarvisError(f"Failed to delete note {vault_name}/{path}: {e}") from e

    def get_vault_stats(self, vault_name: str) -> dict:
        """Get statistics for a specific vault."""
        try:
            # Get all documents for this vault
            results = self.collection.get(
                where={"vault_name": vault_name},
                include=['metadatas']
            )

            if not results['metadatas']:
                return {
                    'vault_name': vault_name,
                    'note_count': 0,
                    'latest_modified': None,
                    'earliest_modified': None
                }

            # Calculate statistics
            timestamps = [
                metadata.get('last_modified', 0)
                for metadata in results['metadatas']
                if metadata.get('last_modified')
            ]

            return {
                'vault_name': vault_name,
                'note_count': len(results['ids']),
                'latest_modified': max(timestamps) if timestamps else None,
                'earliest_modified': min(timestamps) if timestamps else None
            }

        except Exception as e:
            logger.error(f"Failed to get vault stats for {vault_name}: {e}")
            raise JarvisError(f"Failed to get vault stats for {vault_name}: {e}") from e

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Register with the factory
try:
    from jarvis.database.factory import DatabaseFactory
    DatabaseFactory.register_vector_backend('chroma', ChromaVectorDatabase)
    logger.info("Registered ChromaDB vector backend")
except ImportError:
    # Factory not yet available during import
    pass
