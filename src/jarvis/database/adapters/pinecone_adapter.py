"""
Pinecone Vector Database Adapter.

This module provides a Pinecone implementation of the IVectorDatabase interface,
allowing the system to use Pinecone as a cloud-based vector database backend.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import time

import torch

from jarvis.core.interfaces import IVectorDatabase
from jarvis.utils.logging import setup_logging
from jarvis.utils.errors import JarvisError, ServiceError

logger = setup_logging(__name__)


class PineconeVectorDatabase(IVectorDatabase):
    """Pinecone-based vector database for document embeddings."""
    
    def __init__(self, api_key: str, environment: str, index_name: str = "jarvis-embeddings"):
        """Initialize the Pinecone vector database.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-west1-gcp')
            index_name: Name of the Pinecone index
        """
        try:
            import pinecone
        except ImportError:
            raise ServiceError("Pinecone is not installed. Install with: pip install pinecone-client")
        
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        try:
            # Initialize Pinecone
            pinecone.init(api_key=api_key, environment=environment)
            
            # Connect to index
            if index_name not in pinecone.list_indexes():
                # Create index if it doesn't exist
                logger.info(f"Creating Pinecone index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=384,  # MiniLM-L6-v2 embedding dimension
                    metric="cosine"
                )
                # Wait for index to be ready
                while not pinecone.describe_index(index_name).status['ready']:
                    time.sleep(1)
            
            self.index = pinecone.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            raise ServiceError(f"Failed to connect to Pinecone: {e}") from e

    @classmethod
    def from_config(cls, config) -> "PineconeVectorDatabase":
        """Create PineconeVectorDatabase instance from DatabaseConfig.
        
        Args:
            config: DatabaseConfig instance with Pinecone configuration
            
        Returns:
            PineconeVectorDatabase instance
        """
        api_key = config.get('api_key')
        environment = config.get('environment')
        index_name = config.get('index_name', 'jarvis-embeddings')
        
        if not api_key:
            raise ServiceError("Pinecone API key is required")
        if not environment:
            raise ServiceError("Pinecone environment is required")
        
        return cls(api_key=api_key, environment=environment, index_name=index_name)

    def close(self) -> None:
        """Close the database connection."""
        # Pinecone client doesn't require explicit closing
        logger.info("Pinecone connection closed")

    def num_notes(self) -> int:
        """Get the total number of notes in the database."""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            logger.error(f"Failed to count notes: {e}")
            raise JarvisError(f"Failed to count notes: {e}") from e

    def get_most_recent_seen_timestamp(self, vault_name: str) -> Optional[float]:
        """Get the most recent seen timestamp for a vault."""
        try:
            # Query all vectors for this vault and find max timestamp
            # Note: This is expensive for large datasets, consider caching
            query_response = self.index.query(
                vector=[0.0] * 384,  # Dummy vector
                filter={"vault_name": vault_name},
                top_k=10000,  # Large number to get all results
                include_metadata=True
            )
            
            timestamps = [
                match.metadata.get('last_modified', 0) 
                for match in query_response.matches
                if match.metadata.get('last_modified')
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
        embedding: List[float],
        checksum: Optional[str] = None
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
            vector_id = f"{vault_name}::{str(path)}"
            
            metadata = {
                "vault_name": vault_name,
                "path": str(path),
                "last_modified": last_modified,
                "checksum": checksum or ""
            }
            
            # Upsert vector (insert or update)
            self.index.upsert(
                vectors=[(vector_id, embedding, metadata)]
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
        vault_name: Optional[str] = None
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
            filter_dict = None
            if vault_name:
                filter_dict = {"vault_name": vault_name}
            
            # Perform similarity search
            query_response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Format results
            search_results = []
            for match in query_response.matches:
                vault = match.metadata['vault_name']
                path = Path(match.metadata['path'])
                similarity = match.score  # Pinecone returns similarity scores
                search_results.append((vault, path, similarity))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise JarvisError(f"Search failed: {e}") from e

    def get_note_by_path(self, vault_name: str, path: Path) -> Optional[dict]:
        """Get a specific note by vault name and path."""
        try:
            vector_id = f"{vault_name}::{str(path)}"
            
            fetch_response = self.index.fetch(ids=[vector_id])
            
            if vector_id in fetch_response.vectors:
                vector_data = fetch_response.vectors[vector_id]
                metadata = vector_data.metadata
                
                return {
                    'path': metadata['path'],
                    'vault_name': metadata['vault_name'],
                    'last_modified': metadata['last_modified'],
                    'embedding': vector_data.values,
                    'checksum': metadata.get('checksum')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get note {vault_name}/{path}: {e}")
            raise JarvisError(f"Failed to get note {vault_name}/{path}: {e}") from e

    def delete_note(self, vault_name: str, path: Path) -> bool:
        """Delete a note from the database."""
        try:
            vector_id = f"{vault_name}::{str(path)}"
            self.index.delete(ids=[vector_id])
            logger.debug(f"Deleted note: {vault_name}/{path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete note {vault_name}/{path}: {e}")
            raise JarvisError(f"Failed to delete note {vault_name}/{path}: {e}") from e

    def get_vault_stats(self, vault_name: str) -> dict:
        """Get statistics for a specific vault."""
        try:
            # Get index stats with filter
            stats = self.index.describe_index_stats(filter={"vault_name": vault_name})
            
            # Query for timestamp statistics
            query_response = self.index.query(
                vector=[0.0] * 384,  # Dummy vector
                filter={"vault_name": vault_name},
                top_k=10000,  # Large number to get all results
                include_metadata=True
            )
            
            timestamps = [
                match.metadata.get('last_modified', 0) 
                for match in query_response.matches
                if match.metadata.get('last_modified')
            ]
            
            return {
                'vault_name': vault_name,
                'note_count': len(query_response.matches),
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
    DatabaseFactory.register_vector_backend('pinecone', PineconeVectorDatabase)
    logger.info("Registered Pinecone vector backend")
except ImportError:
    # Factory not yet available during import
    pass