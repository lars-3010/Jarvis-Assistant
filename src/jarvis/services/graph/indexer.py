"""
Document indexing service for the graph database.

This module handles the indexing of documents into the graph database,
including text extraction, relationship parsing, and storage.
"""
import logging
from pathlib import Path

from jarvis.services.graph.database import GraphDatabase
from jarvis.services.graph.parser import MarkdownParser
from jarvis.services.vault.reader import VaultReader
from jarvis.utils.errors import ConfigurationError, ServiceError

logger = logging.getLogger(__name__)

class GraphIndexer:
    """Document indexing service for graph search."""

    def __init__(
        self,
        database: GraphDatabase,
        vaults: dict[str, Path],
    ):
        """
        Initialize the indexer.

        Args:
            database: Graph database instance
            vaults: Dictionary mapping vault names to paths
        """
        self.database = database
        self.vaults = vaults
        self.vault_readers = {name: VaultReader(str(path)) for name, path in vaults.items()}

    def index_vault(self, vault_name: str, file_patterns: list[str] | None = None):
        """
        Index all files in a vault.

        Args:
            vault_name: Name of the vault to index
            file_patterns: File patterns to include (defaults to *.md)
        """
        if vault_name not in self.vaults:
            raise ConfigurationError(f"Unknown vault: {vault_name}")

        vault_path = self.vaults[vault_name]
        if not vault_path.exists():
            raise ConfigurationError(f"Vault path does not exist: {vault_path}")

        if file_patterns is None:
            file_patterns = ["*.md"]

        files_to_index = []
        for pattern in file_patterns:
            files_to_index.extend([
                path for path in vault_path.rglob(pattern)
                if path.is_file()
            ])

        logger.info(f"Found {len(files_to_index)} files in vault '{vault_name}' matching patterns: {file_patterns}")
        logger.info(f"Starting graph indexing of {len(files_to_index)} files...")

        indexed_count = 0
        failed_count = 0

        for i, path in enumerate(files_to_index, 1):
            try:
                self.index_file(vault_name, path)
                indexed_count += 1
                if i % 10 == 0:  # Log progress every 10 files
                    logger.info(f"Graph indexing progress: {i}/{len(files_to_index)} files processed")
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to index file {path}: {e}")
                # Continue processing other files instead of stopping

        logger.info(f"Graph indexing completed: {indexed_count} files indexed successfully, {failed_count} failed")

    def index_file(self, vault_name: str, path: Path):
        """
        Index a single file.

        Args:
            vault_name: The name of the vault.
            path: The path to the file.
        """
        try:
            relative_path = str(path.relative_to(self.vaults[vault_name]))
            logger.debug(f"Starting indexing of file: {relative_path}")

            content, _ = self.vault_readers[vault_name].read_file(relative_path)
            logger.debug(f"Read content length: {len(content)} characters from {relative_path}")

            parser = MarkdownParser(content)
            parsed_data = parser.parse()
            logger.debug(f"Parsed data for {relative_path}: tags={len(parsed_data.get('tags', []))}, links={len(parsed_data.get('links', []))}, relationships={len(parsed_data.get('relationships', {}))}")

            note_data = {
                "path": relative_path,
                "title": path.stem,
                "name": path.name,  # Add full filename
                "content": content,  # Add full content
                "tags": parsed_data.get("tags", []),
                "links": parsed_data.get("links", []),
                "relationships": parsed_data.get("relationships", {}),
            }

            logger.debug(f"Created note_data for {relative_path}: title='{note_data['title']}', name='{note_data['name']}', content_length={len(note_data['content'])}")

            result = self.database.create_or_update_note(note_data)
            logger.info(f"Successfully indexed file: {relative_path} - {result.get('operation', 'unknown')}")

        except ServiceError as e:
            logger.error(f"Failed to index file {path}: {e}")
            raise ServiceError(f"Failed to index file {path}: {e}") from e
