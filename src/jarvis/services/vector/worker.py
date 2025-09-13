"""
Background worker for vector indexing and file watching.

This module provides background processing capabilities for indexing documents
and monitoring file system changes in real-time.
"""

import asyncio as _asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

from jarvis.core.event_integration import EventTypes
from jarvis.core.events import publish_event_threadsafe
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vector.indexer import VectorIndexer
from jarvis.services.vector.searcher import SearchResult, VectorSearcher
from jarvis.utils.config import get_settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class IndexMessage:
    """Message for indexing a file."""
    vault_name: str
    path: Path
    operation: str = "index"  # index, delete, move


@dataclass
class SearchRequest:
    """Search request message."""
    query: str
    top_k: int = 10
    vault_name: str | None = None


@dataclass
class SearchResponse:
    """Search response message."""
    results: list[SearchResult]
    query: str
    processing_time: float


class VectorWorker:
    """Background worker for vector operations."""

    def __init__(
        self,
        database: VectorDatabase,
        encoder: VectorEncoder,
        vaults: dict[str, Path],
        batch_size: int | None = None,
        enable_watching: bool = True,
        auto_index: bool = False
    ):
        """Initialize the worker.
        
        Args:
            database: Vector database instance
            encoder: Vector encoder instance
            vaults: Dictionary mapping vault names to paths
            batch_size: Batch size for processing
            enable_watching: Whether to enable file system watching
            auto_index: Whether to automatically index all files on startup
        """
        self.database = database
        self.encoder = encoder
        self.vaults = vaults

        # Get batch size from settings if not provided
        if batch_size is None:
            settings = get_settings()
            batch_size = getattr(settings, 'index_batch_size', 32)

        self.batch_size = batch_size
        self.enable_watching = enable_watching
        self.auto_index = auto_index

        # Initialize components
        self.indexer = VectorIndexer(database, encoder, vaults, batch_size)
        self.searcher = VectorSearcher(database, encoder, vaults)

        # Background processing
        self.index_queue: Queue = Queue()
        self.running = False
        self.worker_thread: threading.Thread | None = None
        self.watchers: list[DirectoryWatcher] = []

        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'searches_performed': 0,
            'uptime_start': time.time()
        }

        logger.info(f"Initialized worker with {len(vaults)} vaults, watching: {enable_watching}")

    def start(self) -> None:
        """Start the background worker."""
        if self.running:
            logger.warning("⚠️ Worker already running")
            return

        self.running = True
        logger.info("🚀 Starting vector worker...")
        logger.debug(f"📁 Worker configuration: batch_size={self.batch_size}, watching={self.enable_watching}, auto_index={self.auto_index}")
        logger.debug(f"📁 Vaults: {[(name, str(path)) for name, path in self.vaults.items()]}")

        # Start background processing thread
        logger.info("🧵 Starting background processing thread")
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.debug(f"🧵 Background thread started: {self.worker_thread.name}")

        # Auto-index if requested
        if self.auto_index:
            logger.info("📚 Auto-indexing requested")
            self._enqueue_all_vaults()

        # Start file system watchers
        if self.enable_watching:
            logger.info("🔍 Starting file system watchers")
            self._start_watchers()

        logger.info("✅ Vector worker started successfully")

        # Log initial stats
        stats = self.get_stats()
        logger.info(f"📊 Initial worker stats: {stats['watchers_active']} watchers, queue size: {stats['queue_size']}")

    def stop(self) -> None:
        """Stop the background worker."""
        if not self.running:
            logger.debug("🚫 Worker not running, nothing to stop")
            return

        logger.info("🛑 Stopping vector worker...")

        # Log final stats before stopping
        try:
            stats = self.get_stats()
            logger.info(f"📊 Final worker stats: processed={stats['files_processed']}, failed={stats['files_failed']}, searches={stats['searches_performed']}, uptime={stats['uptime']:.2f}s")
        except Exception as e:
            logger.error(f"💥 Error getting final stats: {e}")

        self.running = False

        # Stop file system watchers
        logger.info("🔍 Stopping file system watchers")
        self._stop_watchers()

        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            logger.info("⏳ Waiting for worker thread to finish (timeout: 10s)")
            self.worker_thread.join(timeout=10.0)

            if self.worker_thread.is_alive():
                logger.error("⚠️ Worker thread did not finish within timeout")
            else:
                logger.debug("✅ Worker thread finished cleanly")

        logger.info("✅ Vector worker stopped")

    def enqueue_file(self, vault_name: str, path: Path, operation: str = "index") -> None:
        """Enqueue a file for processing.
        
        Args:
            vault_name: Name of the vault
            path: Path to the file
            operation: Operation to perform (index, delete, move)
        """
        if not self.running:
            logger.warning("⚠️ Worker not running, cannot enqueue file")
            return

        # Additional validation for iCloud files
        if "iCloud" in str(path):
            logger.debug(f"📱 iCloud file event: {operation} {vault_name}/{path}")
            # Skip .icloud placeholder files
            if path.suffix == ".icloud":
                logger.debug(f"📱 Skipping .icloud placeholder file: {path}")
                return

        message = IndexMessage(vault_name, path, operation)
        self.index_queue.put(message)
        logger.debug(f"➡️ Enqueued {operation} operation for {vault_name}/{path}")

    def enqueue_vault(self, vault_name: str, file_patterns: list[str] | None = None) -> None:
        """Enqueue all files in a vault for indexing.
        
        Args:
            vault_name: Name of the vault
            file_patterns: File patterns to include (defaults to *.md)
        """
        if vault_name not in self.vaults:
            logger.error(f"Unknown vault: {vault_name}")
            return

        vault_path = self.vaults[vault_name]
        if not vault_path.exists():
            logger.error(f"Vault path does not exist: {vault_path}")
            return

        # Default to markdown files
        if file_patterns is None:
            file_patterns = ['*.md']

        # Find and enqueue all matching files
        for pattern in file_patterns:
            for path in vault_path.rglob(pattern):
                if path.is_file():
                    self.enqueue_file(vault_name, path)

        logger.info(f"Enqueued all files in vault '{vault_name}' for indexing")

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform a search operation.
        
        Args:
            request: Search request
            
        Returns:
            Search response with results
        """
        start_time = time.time()

        try:
            results = self.searcher.search(
                query=request.query,
                top_k=request.top_k,
                vault_name=request.vault_name
            )

            self.stats['searches_performed'] += 1
            processing_time = time.time() - start_time

            logger.debug(f"Search for '{request.query[:50]}...' returned {len(results)} results in {processing_time:.3f}s")

            return SearchResponse(
                results=results,
                query=request.query,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResponse(
                results=[],
                query=request.query,
                processing_time=time.time() - start_time
            )

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics.
        
        Returns:
            Dictionary with worker statistics
        """
        uptime = time.time() - self.stats['uptime_start']

        return {
            'running': self.running,
            'uptime': uptime,
            'queue_size': self.index_queue.qsize(),
            'files_processed': self.stats['files_processed'],
            'files_failed': self.stats['files_failed'],
            'searches_performed': self.stats['searches_performed'],
            'watchers_active': len(self.watchers),
            'database_stats': self.searcher.get_vault_stats(),
            'indexer_stats': self.indexer.get_indexing_stats()
        }

    def _worker_loop(self) -> None:
        """Main worker loop for background processing."""
        logger.debug("🔄 Worker loop started")
        loop_count = 0
        last_stats_log = time.time()

        while self.running:
            try:
                # Process a batch of indexing requests
                self._process_index_batch()

                loop_count += 1

                # Log stats periodically (every 100 iterations or 60 seconds)
                current_time = time.time()
                if loop_count % 100 == 0 or (current_time - last_stats_log) > 60:
                    queue_size = self.index_queue.qsize()
                    if queue_size > 0:
                        logger.debug(f"📊 Worker loop stats: iteration={loop_count}, queue_size={queue_size}")
                    last_stats_log = current_time

                # Small sleep to prevent busy waiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"💥 Error in worker loop: {e}")
                logger.error(f"🔍 Exception type: {type(e).__name__}")
                time.sleep(1.0)  # Longer sleep on error

        logger.debug("✅ Worker loop ended")

    def _process_index_batch(self) -> None:
        """Process a batch of indexing messages."""
        batch = []

        # Collect a batch of messages
        for _ in range(self.batch_size):
            try:
                message = self.index_queue.get_nowait()
                batch.append(message)
            except Empty:
                break

        if not batch:
            return

        logger.debug(f"📋 Processing batch of {len(batch)} messages")

        # Group by operation type
        files_to_index = []
        files_to_delete = []

        for message in batch:
            if message.operation == "index":
                files_to_index.append((message.vault_name, message.path))
                logger.debug(f"📝 Queued for indexing: {message.vault_name}/{message.path}")
            elif message.operation == "delete":
                files_to_delete.append((message.vault_name, message.path))
                logger.debug(f"🗑️ Queued for deletion: {message.vault_name}/{message.path}")

        # Process indexing
        if files_to_index:
            try:
                logger.debug(f"📝 Starting batch indexing of {len(files_to_index)} files")
                stats = self.indexer.index_files(files_to_index)
                self.stats['files_processed'] += stats.processed_files
                self.stats['files_failed'] += stats.failed_files
                logger.debug(f"✅ Batch indexing completed: processed={stats.processed_files}, failed={stats.failed_files}")

                # Publish vault updated events (best-effort)
                for vault_name, path in files_to_index:
                    try:
                        publish_event_threadsafe(
                            EventTypes.DOCUMENT_UPDATED,
                            {
                                'vault_name': vault_name,
                                'path': str(path),
                                'operation': 'index',
                                'timestamp': time.time(),
                            },
                            source="vector_worker",
                        )
                    except Exception as e:
                        logger.debug(f"Event publish (index) failed for {vault_name}:{path}: {e}")
            except Exception as e:
                logger.error(f"💥 Batch indexing failed: {e}")
                logger.error(f"🔍 Exception type: {type(e).__name__}")
                self.stats['files_failed'] += len(files_to_index)

        # Process deletions
        for vault_name, path in files_to_delete:
            try:
                vault_path = self.vaults[vault_name]
                relative_path = path.relative_to(vault_path)
                logger.debug(f"🗑️ Deleting from index: {vault_name}/{relative_path}")
                success = self.database.delete_note(vault_name, relative_path)
                if success:
                    logger.debug(f"✅ Deleted {vault_name}/{relative_path} from index")
                    try:
                        publish_event_threadsafe(
                            EventTypes.DOCUMENT_DELETED,
                            {
                                'vault_name': vault_name,
                                'path': str(path),
                                'operation': 'delete',
                                'timestamp': time.time(),
                            },
                            source="vector_worker",
                        )
                    except Exception as e:
                        logger.debug(f"Event publish (delete) failed for {vault_name}:{path}: {e}")
                else:
                    logger.warning(f"⚠️ Failed to delete {vault_name}/{relative_path} from index")
                    self.stats['files_failed'] += 1
            except Exception as e:
                logger.error(f"💥 Failed to delete {vault_name}/{path}: {e}")
                logger.error(f"🔍 Exception type: {type(e).__name__}")
                self.stats['files_failed'] += 1

    def _enqueue_all_vaults(self) -> None:
        """Enqueue all files from all vaults for indexing."""
        logger.info("Auto-indexing all vaults...")
        for vault_name in self.vaults.keys():
            self.enqueue_vault(vault_name)

    def _start_watchers(self) -> None:
        """Start file system watchers for all vaults."""
        self.watchers = []

        for vault_name, vault_path in self.vaults.items():
            try:
                logger.debug(f"🔍 Creating watcher for vault '{vault_name}' at {vault_path}")

                # Additional validation for iCloud paths
                if "iCloud" in str(vault_path):
                    logger.info(f"📱 iCloud vault detected: {vault_name}")
                    logger.info(f"📱 Path: {vault_path}")
                    logger.info(f"📱 Exists: {vault_path.exists()}")
                    logger.info(f"📱 Is dir: {vault_path.is_dir()}")
                    logger.info(f"📱 Readable: {vault_path.is_dir() and vault_path.exists()}")

                    # Check for .icloud files that might indicate sync issues
                    if vault_path.exists():
                        icloud_files = list(vault_path.glob("**/*.icloud"))
                        if icloud_files:
                            logger.warning(f"📱 Found {len(icloud_files)} .icloud files - some files may be syncing")
                            logger.debug(f"📱 Sample .icloud files: {[str(f) for f in icloud_files[:3]]}")

                watcher = DirectoryWatcher(
                    worker=self,
                    vault_name=vault_name,
                    directory=vault_path,
                    recursive=True
                )

                logger.debug(f"🔍 Starting watcher for '{vault_name}'")
                watcher.start()
                self.watchers.append(watcher)
                logger.info(f"✅ Started watcher for vault '{vault_name}' at {vault_path}")

            except Exception as e:
                logger.error(f"💥 Failed to start watcher for {vault_name}: {e}")
                logger.error(f"🔍 Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")

        logger.info(f"📊 Started {len(self.watchers)} out of {len(self.vaults)} watchers")

    def _stop_watchers(self) -> None:
        """Stop all file system watchers."""
        logger.debug(f"🛑 Stopping {len(self.watchers)} watchers")

        for i, watcher in enumerate(self.watchers):
            try:
                logger.debug(f"🛑 Stopping watcher {i+1}/{len(self.watchers)}")
                watcher.stop()
                logger.debug(f"✅ Stopped watcher {i+1}")
            except Exception as e:
                logger.error(f"💥 Error stopping watcher {i+1}: {e}")
                logger.error(f"🔍 Exception type: {type(e).__name__}")

        self.watchers.clear()
        logger.debug("✅ All watchers stopped and cleared")


class DirectoryWatcher:
    """File system watcher for a directory."""

    def __init__(self, worker: VectorWorker, vault_name: str, directory: Path, recursive: bool = True):
        """Initialize the directory watcher.
        
        Args:
            worker: Vector worker instance
            vault_name: Name of the vault
            directory: Directory to watch
            recursive: Whether to watch subdirectories
        """
        self.worker = worker
        self.vault_name = vault_name
        self.directory = directory
        self.recursive = recursive
        self.observer = Observer()

        logger.debug(f"🔍 DirectoryWatcher initialized for {vault_name}: {directory}")

    def start(self) -> None:
        """Start watching the directory."""
        logger.debug(f"🔍 Starting directory watcher for {self.vault_name}: {self.directory}")

        # Enhanced patterns for iCloud environments
        patterns = ["*.md"]
        ignore_patterns = ["*.icloud", "*.tmp", "*.temp", "*~", "*.DS_Store"]

        event_handler = MarkdownFileEventHandler(
            worker=self.worker,
            vault_name=self.vault_name,
            patterns=patterns,
            ignore_patterns=ignore_patterns,
            ignore_directories=True,
            case_sensitive=False,
        )

        logger.debug(f"🔍 Scheduling observer for {self.directory} (recursive={self.recursive})")
        self.observer.schedule(
            event_handler,
            str(self.directory),
            recursive=self.recursive
        )

        logger.debug(f"🔍 Starting observer for {self.directory}")
        self.observer.start()
        logger.debug(f"✅ Started watching {self.directory}")

    def stop(self) -> None:
        """Stop watching the directory."""
        logger.debug(f"🛑 Stopping directory watcher for {self.vault_name}: {self.directory}")

        try:
            self.observer.stop()
            logger.debug(f"🛑 Observer stopped for {self.directory}")

            # Join with timeout to prevent hanging
            self.observer.join(timeout=5.0)

            if self.observer.is_alive():
                logger.warning(f"⚠️ Observer thread for {self.directory} did not finish within timeout")
            else:
                logger.debug(f"✅ Observer thread joined for {self.directory}")

        except Exception as e:
            logger.error(f"💥 Error stopping directory watcher for {self.directory}: {e}")
            logger.error(f"🔍 Exception type: {type(e).__name__}")

        logger.debug(f"✅ Stopped watching {self.directory}")


class MarkdownFileEventHandler(PatternMatchingEventHandler):
    """Event handler for markdown file changes."""

    def __init__(self, worker: VectorWorker, vault_name: str, **kwargs):
        """Initialize the event handler.
        
        Args:
            worker: Vector worker instance
            vault_name: Name of the vault
            **kwargs: Additional arguments for PatternMatchingEventHandler
        """
        super().__init__(**kwargs)
        self.worker = worker
        self.vault_name = vault_name

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if not event.is_directory:
            path = Path(event.src_path)
            logger.debug(f"🆕 File created: {path}")
            self.worker.enqueue_file(self.vault_name, path, "index")

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if not event.is_directory:
            path = Path(event.src_path)
            logger.debug(f"✏️ File modified: {path}")
            self.worker.enqueue_file(self.vault_name, path, "index")

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion events."""
        if not event.is_directory:
            path = Path(event.src_path)
            logger.debug(f"🗑️ File deleted: {path}")
            self.worker.enqueue_file(self.vault_name, path, "delete")

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events."""
        if not event.is_directory:
            src_path = Path(event.src_path)
            dest_path = Path(event.dest_path)
            logger.debug(f"📬 File moved: {src_path} -> {dest_path}")

            # Delete old location and index new location
            self.worker.enqueue_file(self.vault_name, src_path, "delete")
            self.worker.enqueue_file(self.vault_name, dest_path, "index")
