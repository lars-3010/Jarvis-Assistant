"""
Vector database implementation using DuckDB.

This module provides the core database interface for storing and searching
document embeddings using DuckDB's vector similarity capabilities.
"""

from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent

import duckdb
import torch

from jarvis.core.interfaces import IVectorDatabase
from jarvis.utils.errors import JarvisError, ServiceError
import logging

logger = logging.getLogger(__name__)


class VectorDatabase(IVectorDatabase):
    """DuckDB-based vector database for document embeddings."""

    def __init__(self, database_path: Path, read_only: bool = False, create_if_missing: bool = False):
        """Initialize the vector database.
        
        Args:
            database_path: Path to the DuckDB database file
            read_only: Whether to open in read-only mode
            create_if_missing: Whether to create database if it doesn't exist
        """
        self.database_path = database_path
        self.read_only = read_only
        self.create_if_missing = create_if_missing

        # Ensure database directory exists
        database_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle missing database file gracefully
        if not database_path.exists() and not create_if_missing:
            if read_only:
                raise ServiceError(
                    f"Database file does not exist at {database_path} and cannot be created in read-only mode. "
                    f"Please ensure the database file exists or set create_if_missing=True."
                )
            else:
                raise ServiceError(
                    f"Database file does not exist at {database_path}. "
                    f"Set create_if_missing=True to automatically create the database, "
                    f"or use DatabaseInitializer to create it manually."
                )

        try:
            # If database doesn't exist and create_if_missing is True, DuckDB will create it
            self.ddb_connection = duckdb.connect(str(database_path), read_only=read_only)
            logger.info(f"Connected to vector database: {database_path}")
        except Exception as e:
            # Provide more specific error messages for common issues
            if "database does not exist" in str(e).lower():
                raise ServiceError(
                    f"Database file does not exist at {database_path}. "
                    f"Use DatabaseInitializer.ensure_database_exists() to create it, "
                    f"or set create_if_missing=True when initializing VectorDatabase."
                ) from e
            elif "permission" in str(e).lower():
                raise ServiceError(
                    f"Permission denied accessing database at {database_path}. "
                    f"Please check file permissions and ensure the directory is writable."
                ) from e
            else:
                raise ServiceError(f"Failed to connect to DuckDB database at {database_path}: {e}") from e

        if not self.read_only:
            self.initialize()

    @classmethod
    def ensure_exists(cls, database_path: Path) -> bool:
        """Check if database exists and is accessible before opening.
        
        Args:
            database_path: Path to the database file
            
        Returns:
            True if database exists and is accessible, False otherwise
        """
        try:
            if not database_path.exists():
                logger.debug(f"Database file does not exist: {database_path}")
                return False

            # Check if file is readable
            if not database_path.is_file():
                logger.error(f"Database path exists but is not a file: {database_path}")
                return False

            # Try to open database in read-only mode to test accessibility
            try:
                with duckdb.connect(str(database_path), read_only=True) as conn:
                    # Test basic connectivity
                    conn.execute("SELECT 1").fetchone()
                    logger.debug(f"Database accessibility confirmed: {database_path}")
                    return True
            except Exception as e:
                logger.error(f"Database exists but is not accessible: {database_path}, error: {e}")
                return False

        except Exception as e:
            logger.error(f"Error checking database existence: {database_path}, error: {e}")
            return False

    @classmethod
    def from_config(cls, config) -> "VectorDatabase":
        """Create VectorDatabase instance from DatabaseConfig.
        
        Args:
            config: DatabaseConfig instance with DuckDB configuration
            
        Returns:
            VectorDatabase instance
        """
        database_path = config.get('database_path')
        read_only = config.get('read_only', False)
        create_if_missing = config.get('create_if_missing', False)

        if not isinstance(database_path, Path):
            database_path = Path(database_path)

        return cls(database_path=database_path, read_only=read_only, create_if_missing=create_if_missing)

    def initialize(self) -> None:
        """Initialize the database schema."""
        try:
            self.ddb_connection.execute(
                dedent("""
                CREATE TABLE IF NOT EXISTS notes (
                    path STRING PRIMARY KEY,
                    vault_name STRING,
                    last_modified FLOAT,
                    emb_minilm_l6_v2 FLOAT[384],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum STRING DEFAULT NULL
                )
            """)
            )
            logger.info("Vector database schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise JarvisError(f"Failed to initialize database schema: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, 'ddb_connection'):
            self.ddb_connection.close()
            logger.info("Vector database connection closed")

    def is_healthy(self) -> bool:
        """Check if the database connection is healthy.
        
        Returns:
            True if database is accessible and functional, False otherwise
        """
        try:
            # Try to execute a simple query to test connection
            self.ddb_connection.execute("SELECT 1").fetchone()

            # Check if required tables exist using DuckDB system tables
            try:
                # Try to query the notes table directly
                self.ddb_connection.execute("SELECT COUNT(*) FROM notes").fetchone()
                return True
            except Exception:
                # If notes table doesn't exist, that's still "healthy" for a new database
                # The table will be created when initialize() is called
                logger.debug("Notes table not found, but database connection is healthy")
                return True

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def num_notes(self) -> int:
        """Get the total number of notes in the database."""
        try:
            result = self.ddb_connection.execute("SELECT COUNT(*) FROM notes").fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count notes: {e}")
            raise JarvisError(f"Failed to count notes: {e}") from e

    def get_most_recent_seen_timestamp(self, vault_name: str) -> float | None:
        """Get the most recent seen timestamp for a vault.
        
        Args:
            vault_name: Name of the vault
            
        Returns:
            Most recent timestamp or None if no notes found
        """
        try:
            result = self.ddb_connection.execute(
                "SELECT max(last_modified) FROM notes WHERE vault_name = ?",
                (vault_name,)
            ).fetchone()
            return result[0] if result and result[0] is not None else None
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
        if self.read_only:
            logger.warning("Cannot store note in read-only database")
            return False

        try:
            # DuckDB array update limitation: delete first, then insert
            self.ddb_connection.execute(
                "DELETE FROM notes WHERE vault_name = ? AND path = ?",
                (vault_name, str(path))
            )

            self.ddb_connection.execute(
                "INSERT INTO notes (path, vault_name, last_modified, emb_minilm_l6_v2, checksum) VALUES (?, ?, ?, ?, ?)",
                (str(path), vault_name, last_modified, embedding, checksum),
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
            if vault_name:
                query = """
                    SELECT vault_name, path, (-(emb_minilm_l6_v2 <-> ?)) as similarity
                    FROM notes 
                    WHERE vault_name = ?
                    ORDER BY emb_minilm_l6_v2 <-> ? 
                    LIMIT ?
                """
                params = (query_embedding.tolist(), vault_name, query_embedding.tolist(), top_k)
            else:
                query = """
                    SELECT vault_name, path, (-(emb_minilm_l6_v2 <-> ?)) as similarity
                    FROM notes 
                    ORDER BY emb_minilm_l6_v2 <-> ? 
                    LIMIT ?
                """
                params = (query_embedding.tolist(), query_embedding.tolist(), top_k)

            results = self.ddb_connection.execute(query, params)
            return [(r[0], Path(r[1]), r[2]) for r in results.fetchall()]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise JarvisError(f"Search failed: {e}") from e

    def get_note_by_path(self, vault_name: str, path: Path) -> dict | None:
        """Get a specific note by vault name and path.
        
        Args:
            vault_name: Name of the vault
            path: Path to the note
            
        Returns:
            Note data dictionary or None if not found
        """
        try:
            result = self.ddb_connection.execute(
                "SELECT * FROM notes WHERE vault_name = ? AND path = ?",
                (vault_name, str(path))
            ).fetchone()

            if result:
                return {
                    'path': result[0],
                    'vault_name': result[1],
                    'last_modified': result[2],
                    'embedding': result[3],
                    'created_at': result[4] if len(result) > 4 else None,
                    'checksum': result[5] if len(result) > 5 else None
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get note {vault_name}/{path}: {e}")
            raise JarvisError(f"Failed to get note {vault_name}/{path}: {e}") from e

    def delete_note(self, vault_name: str, path: Path) -> bool:
        """Delete a note from the database.
        
        Args:
            vault_name: Name of the vault
            path: Path to the note
            
        Returns:
            True if successful, False otherwise
        """
        if self.read_only:
            logger.warning("Cannot delete note in read-only database")
            return False

        try:
            self.ddb_connection.execute(
                "DELETE FROM notes WHERE vault_name = ? AND path = ?",
                (vault_name, str(path))
            )
            logger.debug(f"Deleted note: {vault_name}/{path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete note {vault_name}/{path}: {e}")
            raise JarvisError(f"Failed to delete note {vault_name}/{path}: {e}") from e

    def get_vault_stats(self, vault_name: str) -> dict:
        """Get statistics for a specific vault.
        
        Args:
            vault_name: Name of the vault
            
        Returns:
            Dictionary with vault statistics
        """
        try:
            result = self.ddb_connection.execute(
                """
                SELECT 
                    COUNT(*) as note_count,
                    MAX(last_modified) as latest_modified,
                    MIN(last_modified) as earliest_modified
                FROM notes 
                WHERE vault_name = ?
                """,
                (vault_name,)
            ).fetchone()

            return {
                'vault_name': vault_name,
                'note_count': result[0] if result else 0,
                'latest_modified': result[1] if result else None,
                'earliest_modified': result[2] if result else None
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
