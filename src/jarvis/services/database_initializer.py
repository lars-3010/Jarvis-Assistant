"""
Database initialization service for Jarvis Assistant.

This module provides robust database initialization logic that can create
and initialize database files when they're missing, while maintaining
backward compatibility with existing installations.
"""

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from jarvis.utils.config import JarvisSettings
from jarvis.utils.database_errors import (
    DatabaseErrorHandler,
)
from jarvis.utils.errors import ServiceError
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseState:
    """Information about database state."""
    exists: bool
    path: Path
    size_bytes: int
    created_at: datetime | None
    last_modified: datetime | None
    schema_version: str | None
    is_healthy: bool
    error_message: str | None

    # Statistics
    table_count: int
    note_count: int
    embedding_count: int


@dataclass
class InitializationResult:
    """Result of database initialization attempt."""
    success: bool
    action_taken: str  # "created", "validated", "migrated", "failed"
    database_state: DatabaseState
    error_message: str | None
    warnings: list[str]
    duration_ms: float


class DatabaseRecoveryStrategy:
    """Strategies for handling database issues."""

    def __init__(self, database_path: Path, settings: JarvisSettings):
        self.database_path = database_path
        self.settings = settings
        self.error_handler = DatabaseErrorHandler(database_path)

    def handle_missing_file(self) -> InitializationResult:
        """Handle missing database file with enhanced error handling."""
        start_time = time.time()
        warnings = []

        try:
            logger.info(f"📁 Database file not found at {self.database_path}, creating new database")

            # Check disk space before creating database
            try:
                disk_usage = shutil.disk_usage(self.database_path.parent)
                available_mb = disk_usage.free / (1024 * 1024)
                required_mb = 10  # Minimum 10MB required for database creation

                if available_mb < required_mb:
                    disk_error = self.error_handler.handle_disk_space_error(required_mb * 1024 * 1024)
                    duration_ms = (time.time() - start_time) * 1000

                    database_state = DatabaseState(
                        exists=False,
                        path=self.database_path,
                        size_bytes=0,
                        created_at=None,
                        last_modified=None,
                        schema_version=None,
                        is_healthy=False,
                        error_message=str(disk_error),
                        table_count=0,
                        note_count=0,
                        embedding_count=0
                    )

                    return InitializationResult(
                        success=False,
                        action_taken="failed",
                        database_state=database_state,
                        error_message=str(disk_error),
                        warnings=disk_error.suggestions,
                        duration_ms=duration_ms
                    )

                logger.debug(f"💾 Disk space check passed: {available_mb:.1f} MB available")

            except Exception as disk_check_error:
                logger.warning(f"⚠️ Could not check disk space: {disk_check_error}")
                warnings.append("Could not verify available disk space")

            # Ensure parent directory exists
            try:
                self.database_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"📂 Created directory structure: {self.database_path.parent}")
            except PermissionError as perm_error:
                perm_error_obj = self.error_handler.handle_permission_error(perm_error, "create directory")
                duration_ms = (time.time() - start_time) * 1000

                database_state = DatabaseState(
                    exists=False,
                    path=self.database_path,
                    size_bytes=0,
                    created_at=None,
                    last_modified=None,
                    schema_version=None,
                    is_healthy=False,
                    error_message=str(perm_error_obj),
                    table_count=0,
                    note_count=0,
                    embedding_count=0
                )

                return InitializationResult(
                    success=False,
                    action_taken="failed",
                    database_state=database_state,
                    error_message=str(perm_error_obj),
                    warnings=perm_error_obj.suggestions,
                    duration_ms=duration_ms
                )

            # Create new database with schema
            logger.debug(f"🔧 Creating database schema at {self.database_path}")
            with duckdb.connect(str(self.database_path)) as conn:
                # Initialize schema
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notes (
                        path STRING PRIMARY KEY,
                        vault_name STRING,
                        last_modified FLOAT,
                        emb_minilm_l6_v2 FLOAT[384],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        checksum STRING DEFAULT NULL
                    )
                """)

                # Add metadata table for schema versioning
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS database_metadata (
                        key STRING PRIMARY KEY,
                        value STRING,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Insert schema version
                conn.execute("""
                    INSERT OR REPLACE INTO database_metadata (key, value, updated_at)
                    VALUES ('schema_version', '1.0.0', CURRENT_TIMESTAMP)
                """)

                # Insert creation timestamp
                conn.execute("""
                    INSERT OR REPLACE INTO database_metadata (key, value, updated_at)
                    VALUES ('created_at', ?, CURRENT_TIMESTAMP)
                """, (datetime.now().isoformat(),))

            # Get database state after creation
            database_state = self._get_database_state()
            duration_ms = (time.time() - start_time) * 1000

            logger.info(f"✅ Database created successfully at {self.database_path} in {duration_ms:.2f}ms")

            return InitializationResult(
                success=True,
                action_taken="created",
                database_state=database_state,
                error_message=None,
                warnings=warnings,
                duration_ms=duration_ms
            )

        except PermissionError as perm_error:
            perm_error_obj = self.error_handler.handle_permission_error(perm_error, "create database")
            duration_ms = (time.time() - start_time) * 1000

            logger.error(f"🚫 {perm_error_obj}")
            for suggestion in perm_error_obj.suggestions:
                logger.error(f"   💡 {suggestion}")

            database_state = DatabaseState(
                exists=False,
                path=self.database_path,
                size_bytes=0,
                created_at=None,
                last_modified=None,
                schema_version=None,
                is_healthy=False,
                error_message=str(perm_error_obj),
                table_count=0,
                note_count=0,
                embedding_count=0
            )

            return InitializationResult(
                success=False,
                action_taken="failed",
                database_state=database_state,
                error_message=str(perm_error_obj),
                warnings=perm_error_obj.suggestions,
                duration_ms=duration_ms
            )

        except Exception as e:
            # Handle other database creation errors
            duration_ms = (time.time() - start_time) * 1000

            # Check if it's a disk space issue
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['no space', 'disk full', 'insufficient space']):
                disk_error = self.error_handler.handle_disk_space_error()
                logger.error(f"💾 {disk_error}")

                database_state = DatabaseState(
                    exists=False,
                    path=self.database_path,
                    size_bytes=0,
                    created_at=None,
                    last_modified=None,
                    schema_version=None,
                    is_healthy=False,
                    error_message=str(disk_error),
                    table_count=0,
                    note_count=0,
                    embedding_count=0
                )

                return InitializationResult(
                    success=False,
                    action_taken="failed",
                    database_state=database_state,
                    error_message=str(disk_error),
                    warnings=disk_error.suggestions,
                    duration_ms=duration_ms
                )

            # Generic database creation error
            db_error = self.error_handler.handle_generic_database_error(e, "create database")
            logger.error(f"💥 {db_error}")
            for suggestion in db_error.suggestions:
                logger.error(f"   💡 {suggestion}")

            database_state = DatabaseState(
                exists=False,
                path=self.database_path,
                size_bytes=0,
                created_at=None,
                last_modified=None,
                schema_version=None,
                is_healthy=False,
                error_message=str(db_error),
                table_count=0,
                note_count=0,
                embedding_count=0
            )

            return InitializationResult(
                success=False,
                action_taken="failed",
                database_state=database_state,
                error_message=str(db_error),
                warnings=db_error.suggestions,
                duration_ms=duration_ms
            )

    def handle_permission_error(self, error: Exception) -> InitializationResult:
        """Handle permission-related errors with enhanced guidance."""
        start_time = time.time()

        # Use enhanced error handler
        perm_error = self.error_handler.handle_permission_error(error, "access database")

        logger.error(f"🚫 {perm_error}")
        for suggestion in perm_error.suggestions:
            logger.error(f"   💡 {suggestion}")

        database_state = DatabaseState(
            exists=self.database_path.exists(),
            path=self.database_path,
            size_bytes=0,
            created_at=None,
            last_modified=None,
            schema_version=None,
            is_healthy=False,
            error_message=str(perm_error),
            table_count=0,
            note_count=0,
            embedding_count=0
        )

        duration_ms = (time.time() - start_time) * 1000

        return InitializationResult(
            success=False,
            action_taken="failed",
            database_state=database_state,
            error_message=str(perm_error),
            warnings=perm_error.suggestions,
            duration_ms=duration_ms
        )

    def handle_corruption(self) -> InitializationResult:
        """Handle database corruption with enhanced error handling and recovery guidance."""
        start_time = time.time()
        warnings = []
        backup_path = None
        backup_created = False

        try:
            logger.warning(f"🔧 Database corruption detected, attempting recovery for {self.database_path}")

            # Create backup if enabled
            if getattr(self.settings, 'database_backup_on_corruption', True):
                backup_path = self.database_path.with_suffix(f'.backup.{int(time.time())}.duckdb')
                try:
                    logger.info(f"💾 Creating backup of corrupted database: {backup_path}")
                    shutil.copy2(self.database_path, backup_path)
                    backup_created = True
                    warnings.append(f"Backup created at {backup_path}")
                    logger.info("✅ Backup created successfully")
                except Exception as backup_error:
                    logger.error(f"⚠️ Failed to create backup: {backup_error}")
                    warnings.append(f"Failed to create backup: {backup_error}")
            else:
                logger.info("📝 Backup creation disabled in settings")
                warnings.append("Backup creation was disabled - data may be lost")

            # Remove corrupted database
            try:
                logger.warning(f"🗑️ Removing corrupted database: {self.database_path}")
                self.database_path.unlink()
                logger.info("✅ Corrupted database removed")
            except PermissionError as perm_error:
                perm_error_obj = self.error_handler.handle_permission_error(perm_error, "remove corrupted database")
                duration_ms = (time.time() - start_time) * 1000

                database_state = DatabaseState(
                    exists=self.database_path.exists(),
                    path=self.database_path,
                    size_bytes=0,
                    created_at=None,
                    last_modified=None,
                    schema_version=None,
                    is_healthy=False,
                    error_message=str(perm_error_obj),
                    table_count=0,
                    note_count=0,
                    embedding_count=0
                )

                return InitializationResult(
                    success=False,
                    action_taken="failed",
                    database_state=database_state,
                    error_message=str(perm_error_obj),
                    warnings=warnings + perm_error_obj.suggestions,
                    duration_ms=duration_ms
                )

            # Create new database
            logger.info("🔧 Creating new database to replace corrupted one")
            result = self.handle_missing_file()
            result.action_taken = "recreated_after_corruption"
            result.warnings.extend(warnings)

            if result.success:
                logger.info("✅ Database recovery completed successfully")

            return result

        except Exception as e:
            # Use enhanced error handler for corruption recovery failure
            corruption_error = self.error_handler.handle_corruption_error(e, backup_created, backup_path)
            duration_ms = (time.time() - start_time) * 1000

            logger.error(f"💥 {corruption_error}")
            for suggestion in corruption_error.suggestions:
                logger.error(f"   💡 {suggestion}")

            database_state = DatabaseState(
                exists=self.database_path.exists(),
                path=self.database_path,
                size_bytes=0,
                created_at=None,
                last_modified=None,
                schema_version=None,
                is_healthy=False,
                error_message=str(corruption_error),
                table_count=0,
                note_count=0,
                embedding_count=0
            )

            return InitializationResult(
                success=False,
                action_taken="failed",
                database_state=database_state,
                error_message=str(corruption_error),
                warnings=warnings + corruption_error.suggestions,
                duration_ms=duration_ms
            )

    def _get_database_state(self) -> DatabaseState:
        """Get current database state information."""
        try:
            if not self.database_path.exists():
                return DatabaseState(
                    exists=False,
                    path=self.database_path,
                    size_bytes=0,
                    created_at=None,
                    last_modified=None,
                    schema_version=None,
                    is_healthy=False,
                    error_message="Database file does not exist",
                    table_count=0,
                    note_count=0,
                    embedding_count=0
                )

            # Get file stats
            stat = self.database_path.stat()
            size_bytes = stat.st_size
            last_modified = datetime.fromtimestamp(stat.st_mtime)

            # Try to connect and get database info
            with duckdb.connect(str(self.database_path), read_only=True) as conn:
                # Get schema version
                schema_version = None
                try:
                    result = conn.execute(
                        "SELECT value FROM database_metadata WHERE key = 'schema_version'"
                    ).fetchone()
                    schema_version = result[0] if result else None
                except Exception:
                    # Metadata table might not exist in older databases
                    schema_version = "unknown"

                # Get creation time
                created_at = None
                try:
                    result = conn.execute(
                        "SELECT value FROM database_metadata WHERE key = 'created_at'"
                    ).fetchone()
                    if result:
                        created_at = datetime.fromisoformat(result[0])
                except Exception:
                    # Use file creation time as fallback
                    created_at = datetime.fromtimestamp(stat.st_ctime)

                # Get table count
                table_count = 0
                try:
                    result = conn.execute(
                        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main'"
                    ).fetchone()
                    table_count = result[0] if result else 0
                except Exception:
                    pass

                # Get note count
                note_count = 0
                try:
                    result = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
                    note_count = result[0] if result else 0
                except Exception:
                    pass

                # Get embedding count (notes with embeddings)
                embedding_count = 0
                try:
                    result = conn.execute(
                        "SELECT COUNT(*) FROM notes WHERE emb_minilm_l6_v2 IS NOT NULL"
                    ).fetchone()
                    embedding_count = result[0] if result else 0
                except Exception:
                    pass

            return DatabaseState(
                exists=True,
                path=self.database_path,
                size_bytes=size_bytes,
                created_at=created_at,
                last_modified=last_modified,
                schema_version=schema_version,
                is_healthy=True,
                error_message=None,
                table_count=table_count,
                note_count=note_count,
                embedding_count=embedding_count
            )

        except Exception as e:
            return DatabaseState(
                exists=self.database_path.exists(),
                path=self.database_path,
                size_bytes=0,
                created_at=None,
                last_modified=None,
                schema_version=None,
                is_healthy=False,
                error_message=str(e),
                table_count=0,
                note_count=0,
                embedding_count=0
            )


class DatabaseInitializer:
    """Handles database creation and initialization logic."""

    def __init__(self, database_path: Path, settings: JarvisSettings):
        """Initialize the database initializer.
        
        Args:
            database_path: Path to the database file
            settings: Jarvis settings instance
        """
        self.database_path = database_path
        self.settings = settings
        self.recovery_strategy = DatabaseRecoveryStrategy(database_path, settings)

    def ensure_database_exists(self) -> bool:
        """Ensure database exists and is properly initialized.
        
        Returns:
            True if database is ready for use, False otherwise
        """
        try:
            result = self._initialize_database()

            if result.success:
                logger.info(f"Database initialization successful: {result.action_taken}")
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(warning)
                return True
            else:
                logger.error(f"Database initialization failed: {result.error_message}")
                return False

        except Exception as e:
            logger.error(f"Unexpected error during database initialization: {e}")
            return False

    def create_database(self) -> None:
        """Create a new database with proper schema."""
        if self.database_path.exists():
            raise ServiceError(f"Database already exists at {self.database_path}")

        result = self.recovery_strategy.handle_missing_file()
        if not result.success:
            raise ServiceError(f"Failed to create database: {result.error_message}")

        logger.info(f"Database created successfully at {self.database_path}")

    def validate_database(self) -> bool:
        """Validate existing database health and schema.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            if not self.database_path.exists():
                logger.error(f"Database file does not exist: {self.database_path}")
                return False

            # Try to connect and perform basic operations
            with duckdb.connect(str(self.database_path), read_only=True) as conn:
                # Test basic connectivity
                conn.execute("SELECT 1").fetchone()

                # Check if required tables exist
                try:
                    conn.execute("SELECT COUNT(*) FROM notes").fetchone()
                except Exception as e:
                    logger.error(f"Notes table is missing or corrupted: {e}")
                    return False

                # Check schema version if metadata table exists
                try:
                    result = conn.execute(
                        "SELECT value FROM database_metadata WHERE key = 'schema_version'"
                    ).fetchone()
                    if result:
                        schema_version = result[0]
                        expected_version = getattr(self.settings, 'database_schema_version', '1.0.0')
                        if schema_version != expected_version:
                            logger.warning(f"Schema version mismatch: found {schema_version}, expected {expected_version}")
                            # For now, we'll consider this a warning, not a failure
                except Exception:
                    # Metadata table might not exist in older databases
                    logger.debug("Database metadata table not found (older database)")

            logger.debug(f"Database validation successful: {self.database_path}")
            return True

        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False

    def get_database_info(self) -> dict[str, Any]:
        """Get information about the database state.
        
        Returns:
            Dictionary with database information
        """
        state = self.recovery_strategy._get_database_state()

        return {
            'exists': state.exists,
            'path': str(state.path),
            'size_bytes': state.size_bytes,
            'size_mb': round(state.size_bytes / (1024 * 1024), 2) if state.size_bytes > 0 else 0,
            'created_at': state.created_at.isoformat() if state.created_at else None,
            'last_modified': state.last_modified.isoformat() if state.last_modified else None,
            'schema_version': state.schema_version,
            'is_healthy': state.is_healthy,
            'error_message': state.error_message,
            'table_count': state.table_count,
            'note_count': state.note_count,
            'embedding_count': state.embedding_count,
            'has_embeddings': state.embedding_count > 0
        }

    def _initialize_database(self) -> InitializationResult:
        """Internal method to initialize database with comprehensive error handling."""
        start_time = time.time()

        try:
            # Check if database file exists
            if not self.database_path.exists():
                logger.info("Database file not found, creating new database")
                return self.recovery_strategy.handle_missing_file()

            # Check if database directory is accessible
            if not os.access(self.database_path.parent, os.R_OK | os.W_OK):
                error = PermissionError(f"Cannot access database directory: {self.database_path.parent}")
                return self.recovery_strategy.handle_permission_error(error)

            # Check if database file is accessible
            if not os.access(self.database_path, os.R_OK):
                error = PermissionError(f"Cannot read database file: {self.database_path}")
                return self.recovery_strategy.handle_permission_error(error)

            # Try to validate existing database
            if self.validate_database():
                # Database is healthy
                database_state = self.recovery_strategy._get_database_state()
                duration_ms = (time.time() - start_time) * 1000

                logger.debug(f"Database validation successful in {duration_ms:.2f}ms")

                return InitializationResult(
                    success=True,
                    action_taken="validated",
                    database_state=database_state,
                    error_message=None,
                    warnings=[],
                    duration_ms=duration_ms
                )
            else:
                # Database exists but is not healthy - might be corrupted
                logger.warning("Database exists but failed validation, attempting recovery")
                return self.recovery_strategy.handle_corruption()

        except PermissionError as e:
            return self.recovery_strategy.handle_permission_error(e)
        except Exception as e:
            # Check if it's a corruption-related error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['corrupt', 'malformed', 'damaged', 'invalid']):
                logger.warning(f"Database corruption detected: {e}")
                return self.recovery_strategy.handle_corruption()
            else:
                # Generic error handling
                duration_ms = (time.time() - start_time) * 1000
                error_msg = f"Database initialization failed: {e}"
                logger.error(error_msg)

                database_state = DatabaseState(
                    exists=self.database_path.exists(),
                    path=self.database_path,
                    size_bytes=0,
                    created_at=None,
                    last_modified=None,
                    schema_version=None,
                    is_healthy=False,
                    error_message=error_msg,
                    table_count=0,
                    note_count=0,
                    embedding_count=0
                )

                return InitializationResult(
                    success=False,
                    action_taken="failed",
                    database_state=database_state,
                    error_message=error_msg,
                    warnings=[],
                    duration_ms=duration_ms
                )
