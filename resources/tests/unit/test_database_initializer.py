"""
Unit tests for DatabaseInitializer and related database initialization functionality.

This module tests the comprehensive database initialization logic including
error handling, recovery strategies, and VectorDatabase enhancements.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest
import duckdb

from jarvis.services.database_initializer import (
    DatabaseInitializer, 
    DatabaseRecoveryStrategy, 
    DatabaseState, 
    InitializationResult
)
from jarvis.services.vector.database import VectorDatabase
from jarvis.utils.config import JarvisSettings
from jarvis.utils.errors import ServiceError


class TestDatabaseState:
    """Test DatabaseState dataclass."""
    
    def test_database_state_creation(self):
        """Test creating DatabaseState with all fields."""
        state = DatabaseState(
            exists=True,
            path=Path("/test/db.duckdb"),
            size_bytes=1024,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            schema_version="1.0.0",
            is_healthy=True,
            error_message=None,
            table_count=2,
            note_count=10,
            embedding_count=5
        )
        
        assert state.exists is True
        assert state.path == Path("/test/db.duckdb")
        assert state.size_bytes == 1024
        assert state.schema_version == "1.0.0"
        assert state.is_healthy is True
        assert state.error_message is None
        assert state.table_count == 2
        assert state.note_count == 10
        assert state.embedding_count == 5
    
    def test_database_state_with_error(self):
        """Test DatabaseState with error condition."""
        state = DatabaseState(
            exists=False,
            path=Path("/test/missing.duckdb"),
            size_bytes=0,
            created_at=None,
            last_modified=None,
            schema_version=None,
            is_healthy=False,
            error_message="Database file not found",
            table_count=0,
            note_count=0,
            embedding_count=0
        )
        
        assert state.exists is False
        assert state.is_healthy is False
        assert state.error_message == "Database file not found"


class TestInitializationResult:
    """Test InitializationResult dataclass."""
    
    def test_successful_initialization_result(self):
        """Test successful initialization result."""
        state = DatabaseState(
            exists=True, path=Path("/test"), size_bytes=1024,
            created_at=None, last_modified=None, schema_version="1.0.0",
            is_healthy=True, error_message=None,
            table_count=1, note_count=0, embedding_count=0
        )
        
        result = InitializationResult(
            success=True,
            action_taken="created",
            database_state=state,
            error_message=None,
            warnings=[],
            duration_ms=150.0
        )
        
        assert result.success is True
        assert result.action_taken == "created"
        assert result.error_message is None
        assert result.warnings == []
        assert result.duration_ms == 150.0
    
    def test_failed_initialization_result(self):
        """Test failed initialization result."""
        state = DatabaseState(
            exists=False, path=Path("/test"), size_bytes=0,
            created_at=None, last_modified=None, schema_version=None,
            is_healthy=False, error_message="Permission denied",
            table_count=0, note_count=0, embedding_count=0
        )
        
        result = InitializationResult(
            success=False,
            action_taken="failed",
            database_state=state,
            error_message="Permission denied",
            warnings=["Check file permissions"],
            duration_ms=50.0
        )
        
        assert result.success is False
        assert result.action_taken == "failed"
        assert result.error_message == "Permission denied"
        assert "Check file permissions" in result.warnings


class TestDatabaseRecoveryStrategy:
    """Test DatabaseRecoveryStrategy class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test.duckdb"
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=JarvisSettings)
        settings.database_backup_on_corruption = True
        settings.database_schema_version = "1.0.0"
        return settings
    
    def test_handle_missing_file_success(self, temp_db_path, mock_settings):
        """Test successful handling of missing database file."""
        strategy = DatabaseRecoveryStrategy(temp_db_path, mock_settings)
        
        result = strategy.handle_missing_file()
        
        assert result.success is True
        assert result.action_taken == "created"
        assert temp_db_path.exists()
        assert result.database_state.exists is True
        assert result.database_state.is_healthy is True
        assert result.error_message is None
    
    def test_handle_missing_file_permission_error(self, mock_settings):
        """Test handling missing file with permission error."""
        # Use a path that should cause permission issues
        restricted_path = Path("/root/restricted/test.duckdb")
        strategy = DatabaseRecoveryStrategy(restricted_path, mock_settings)
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            result = strategy.handle_missing_file()
            
            assert result.success is False
            assert result.action_taken == "failed"
            assert "Failed to create database" in result.error_message
    
    def test_handle_permission_error(self, temp_db_path, mock_settings):
        """Test handling permission errors."""
        strategy = DatabaseRecoveryStrategy(temp_db_path, mock_settings)
        error = PermissionError("Access denied")
        
        result = strategy.handle_permission_error(error)
        
        assert result.success is False
        assert result.action_taken == "failed"
        assert "Permission denied" in result.error_message
        assert len(result.warnings) > 0
        assert any("Check file permissions" in warning for warning in result.warnings)
    
    def test_handle_corruption_with_backup(self, temp_db_path, mock_settings):
        """Test handling corruption with backup creation."""
        # Create a corrupted database file
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_db_path.write_text("corrupted data")
        
        strategy = DatabaseRecoveryStrategy(temp_db_path, mock_settings)
        
        result = strategy.handle_corruption()
        
        assert result.success is True
        assert result.action_taken == "recreated_after_corruption"
        assert temp_db_path.exists()
        assert len(result.warnings) > 0
        assert any("Backup created" in warning for warning in result.warnings)
    
    def test_handle_corruption_without_backup(self, temp_db_path, mock_settings):
        """Test handling corruption without backup."""
        mock_settings.database_backup_on_corruption = False
        
        # Create a corrupted database file
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_db_path.write_text("corrupted data")
        
        strategy = DatabaseRecoveryStrategy(temp_db_path, mock_settings)
        
        result = strategy.handle_corruption()
        
        assert result.success is True
        assert result.action_taken == "recreated_after_corruption"
        assert temp_db_path.exists()
        # Should not have backup warnings
        assert not any("Backup created" in warning for warning in result.warnings)
    
    def test_get_database_state_missing_file(self, temp_db_path, mock_settings):
        """Test getting database state for missing file."""
        strategy = DatabaseRecoveryStrategy(temp_db_path, mock_settings)
        
        state = strategy._get_database_state()
        
        assert state.exists is False
        assert state.is_healthy is False
        assert state.error_message == "Database file does not exist"
        assert state.note_count == 0
        assert state.embedding_count == 0
    
    def test_get_database_state_healthy_database(self, temp_db_path, mock_settings):
        """Test getting database state for healthy database."""
        # Create a real database
        with duckdb.connect(str(temp_db_path)) as conn:
            conn.execute("""
                CREATE TABLE notes (
                    path STRING PRIMARY KEY,
                    vault_name STRING,
                    last_modified FLOAT,
                    emb_minilm_l6_v2 FLOAT[384],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum STRING DEFAULT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE database_metadata (
                    key STRING PRIMARY KEY,
                    value STRING,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT INTO database_metadata (key, value) 
                VALUES ('schema_version', '1.0.0')
            """)
        
        strategy = DatabaseRecoveryStrategy(temp_db_path, mock_settings)
        
        state = strategy._get_database_state()
        
        assert state.exists is True
        assert state.is_healthy is True
        assert state.schema_version == "1.0.0"
        assert state.error_message is None
        assert state.table_count >= 2  # notes and database_metadata tables


class TestDatabaseInitializer:
    """Test DatabaseInitializer class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test.duckdb"
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=JarvisSettings)
        settings.database_backup_on_corruption = True
        settings.database_schema_version = "1.0.0"
        return settings
    
    def test_ensure_database_exists_missing_file(self, temp_db_path, mock_settings):
        """Test ensuring database exists when file is missing."""
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        result = initializer.ensure_database_exists()
        
        assert result is True
        assert temp_db_path.exists()
    
    def test_ensure_database_exists_healthy_database(self, temp_db_path, mock_settings):
        """Test ensuring database exists when database is already healthy."""
        # Create a healthy database first
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        initializer.ensure_database_exists()
        
        # Test with existing healthy database
        result = initializer.ensure_database_exists()
        
        assert result is True
        assert temp_db_path.exists()
    
    def test_create_database_success(self, temp_db_path, mock_settings):
        """Test successful database creation."""
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        initializer.create_database()
        
        assert temp_db_path.exists()
        # Verify database is functional
        with duckdb.connect(str(temp_db_path), read_only=True) as conn:
            result = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
            assert result[0] == 0
    
    def test_create_database_already_exists(self, temp_db_path, mock_settings):
        """Test creating database when it already exists."""
        # Create database first
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_db_path.touch()
        
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        with pytest.raises(ServiceError, match="Database already exists"):
            initializer.create_database()
    
    def test_validate_database_healthy(self, temp_db_path, mock_settings):
        """Test validating a healthy database."""
        # Create a healthy database
        with duckdb.connect(str(temp_db_path)) as conn:
            conn.execute("""
                CREATE TABLE notes (
                    path STRING PRIMARY KEY,
                    vault_name STRING,
                    last_modified FLOAT,
                    emb_minilm_l6_v2 FLOAT[384],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum STRING DEFAULT NULL
                )
            """)
        
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        result = initializer.validate_database()
        
        assert result is True
    
    def test_validate_database_missing_file(self, temp_db_path, mock_settings):
        """Test validating missing database file."""
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        result = initializer.validate_database()
        
        assert result is False
    
    def test_validate_database_missing_table(self, temp_db_path, mock_settings):
        """Test validating database with missing notes table."""
        # Create database without notes table
        with duckdb.connect(str(temp_db_path)) as conn:
            conn.execute("CREATE TABLE dummy (id INTEGER)")
        
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        result = initializer.validate_database()
        
        assert result is False
    
    def test_get_database_info(self, temp_db_path, mock_settings):
        """Test getting database information."""
        # Create database with some data
        with duckdb.connect(str(temp_db_path)) as conn:
            conn.execute("""
                CREATE TABLE notes (
                    path STRING PRIMARY KEY,
                    vault_name STRING,
                    last_modified FLOAT,
                    emb_minilm_l6_v2 FLOAT[384],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum STRING DEFAULT NULL
                )
            """)
            # Create a proper 384-dimensional embedding array
            embedding_array = [0.1] * 384  # Create array with 384 elements
            conn.execute("""
                INSERT INTO notes (path, vault_name, last_modified, emb_minilm_l6_v2) 
                VALUES ('test.md', 'test_vault', 1234567890, ?)
            """, (embedding_array,))
        
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        info = initializer.get_database_info()
        
        assert info['exists'] is True
        assert info['note_count'] == 1
        assert info['embedding_count'] == 1
        assert info['has_embeddings'] is True
        assert info['size_mb'] > 0
        assert 'path' in info
        assert 'is_healthy' in info
    
    def test_initialize_database_permission_error(self, temp_db_path, mock_settings):
        """Test database initialization with permission error."""
        # Test the permission error handling by directly calling the recovery strategy
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        
        # Test the handle_permission_error method directly
        error = PermissionError("Permission denied")
        result = initializer.recovery_strategy.handle_permission_error(error)
        
        assert result.success is False
        assert result.action_taken == "failed"
        assert "Permission denied" in result.error_message
        assert len(result.warnings) > 0


class TestVectorDatabaseEnhancements:
    """Test VectorDatabase enhancements for initialization support."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "vector_test.duckdb"
    
    def test_ensure_exists_missing_database(self, temp_db_path):
        """Test ensure_exists with missing database."""
        result = VectorDatabase.ensure_exists(temp_db_path)
        
        assert result is False
    
    def test_ensure_exists_existing_database(self, temp_db_path):
        """Test ensure_exists with existing database."""
        # Create database first
        with VectorDatabase(temp_db_path, create_if_missing=True):
            pass
        
        result = VectorDatabase.ensure_exists(temp_db_path)
        
        assert result is True
    
    def test_ensure_exists_corrupted_database(self, temp_db_path):
        """Test ensure_exists with corrupted database."""
        # Create corrupted file
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_db_path.write_text("corrupted data")
        
        result = VectorDatabase.ensure_exists(temp_db_path)
        
        assert result is False
    
    def test_create_if_missing_true(self, temp_db_path):
        """Test VectorDatabase with create_if_missing=True."""
        with VectorDatabase(temp_db_path, create_if_missing=True) as db:
            assert temp_db_path.exists()
            assert db.num_notes() == 0
    
    def test_create_if_missing_false(self, temp_db_path):
        """Test VectorDatabase with create_if_missing=False (default)."""
        with pytest.raises(ServiceError, match="does not exist"):
            VectorDatabase(temp_db_path, create_if_missing=False)
    
    def test_read_only_with_missing_database(self, temp_db_path):
        """Test read-only mode with missing database."""
        with pytest.raises(ServiceError, match="read-only mode"):
            VectorDatabase(temp_db_path, read_only=True, create_if_missing=False)
    
    def test_from_config_with_create_if_missing(self, temp_db_path):
        """Test from_config method with create_if_missing parameter."""
        config = {
            'database_path': temp_db_path,
            'read_only': False,
            'create_if_missing': True
        }
        
        with VectorDatabase.from_config(config) as db:
            assert temp_db_path.exists()
            assert db.num_notes() == 0
    
    def test_backward_compatibility_two_params(self, temp_db_path):
        """Test backward compatibility with two-parameter constructor."""
        # Create database first
        with VectorDatabase(temp_db_path, create_if_missing=True):
            pass
        
        # Test old-style constructor
        with VectorDatabase(temp_db_path, read_only=True) as db:
            assert db.num_notes() == 0
    
    def test_backward_compatibility_one_param(self, temp_db_path):
        """Test backward compatibility with one-parameter constructor."""
        # Create database first
        with VectorDatabase(temp_db_path, create_if_missing=True):
            pass
        
        # Test old-style constructor
        with VectorDatabase(temp_db_path) as db:
            assert db.num_notes() == 0
    
    def test_error_messages_quality(self, temp_db_path):
        """Test that error messages are helpful and specific."""
        with pytest.raises(ServiceError) as exc_info:
            VectorDatabase(temp_db_path, create_if_missing=False)
        
        error_msg = str(exc_info.value)
        assert "DatabaseInitializer" in error_msg or "create_if_missing=True" in error_msg
    
    def test_permission_error_handling(self):
        """Test permission error handling."""
        # Test that VectorDatabase properly handles and wraps permission errors
        with tempfile.TemporaryDirectory() as temp_dir:
            restricted_path = Path(temp_dir) / "restricted" / "test.duckdb"
            
            # Mock duckdb.connect to raise a permission error
            with patch('duckdb.connect', side_effect=PermissionError("Permission denied")):
                with pytest.raises(ServiceError) as exc_info:
                    VectorDatabase(restricted_path, create_if_missing=True)
                
                # The error should be caught and wrapped in a ServiceError
                error_msg = str(exc_info.value)
                assert ("Permission denied" in error_msg or 
                        "Failed to connect" in error_msg)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "integration_test.duckdb"
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=JarvisSettings)
        settings.database_backup_on_corruption = True
        settings.database_schema_version = "1.0.0"
        return settings
    
    def test_full_initialization_workflow(self, temp_db_path, mock_settings):
        """Test complete initialization workflow."""
        # Step 1: Initialize database
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        success = initializer.ensure_database_exists()
        assert success is True
        
        # Step 2: Verify database info
        db_info = initializer.get_database_info()
        assert db_info['exists'] is True
        assert db_info['is_healthy'] is True
        assert db_info['note_count'] == 0
        
        # Step 3: Use VectorDatabase with existing database
        with VectorDatabase(temp_db_path, read_only=True) as db:
            assert db.num_notes() == 0
            assert db.is_healthy() is True
    
    def test_recovery_from_corruption(self, temp_db_path, mock_settings):
        """Test recovery from database corruption."""
        # Step 1: Create corrupted database
        temp_db_path.parent.mkdir(parents=True, exist_ok=True)
        temp_db_path.write_text("corrupted data")
        
        # Step 2: Initialize should recover
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        success = initializer.ensure_database_exists()
        assert success is True
        
        # Step 3: Database should be functional
        with VectorDatabase(temp_db_path, read_only=True) as db:
            assert db.is_healthy() is True
    
    def test_concurrent_access_simulation(self, temp_db_path, mock_settings):
        """Test simulation of concurrent access scenarios."""
        # Create database
        initializer = DatabaseInitializer(temp_db_path, mock_settings)
        initializer.ensure_database_exists()
        
        # Simulate multiple connections (sequential for testing)
        connections = []
        try:
            for i in range(3):
                db = VectorDatabase(temp_db_path, read_only=True)
                connections.append(db)
                assert db.is_healthy() is True
        finally:
            # Clean up connections
            for db in connections:
                db.close()
    
    def test_missing_directory_creation(self, mock_settings):
        """Test database creation with missing parent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use nested path that doesn't exist
            nested_path = Path(temp_dir) / "nested" / "deep" / "test.duckdb"
            
            initializer = DatabaseInitializer(nested_path, mock_settings)
            success = initializer.ensure_database_exists()
            
            assert success is True
            assert nested_path.exists()
            assert nested_path.parent.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])