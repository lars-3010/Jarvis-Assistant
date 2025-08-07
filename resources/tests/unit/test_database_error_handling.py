"""
Tests for enhanced database error handling and user messaging.

This module tests the comprehensive error handling improvements for database
initialization with user-friendly messages and troubleshooting guidance.
"""

import os
import shutil
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from jarvis.utils.database_errors import (
    DatabaseErrorHandler,
    DatabaseInitializationError,
    DatabasePermissionError,
    DatabaseCorruptionError,
    DatabaseConnectionError,
    DiskSpaceError
)
from jarvis.services.database_initializer import DatabaseInitializer, DatabaseRecoveryStrategy
from jarvis.utils.config import JarvisSettings


class TestDatabaseErrorHandler:
    """Test the DatabaseErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.database_path = self.temp_dir / "test.duckdb"
        self.error_handler = DatabaseErrorHandler(self.database_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_handle_missing_database_error(self):
        """Test handling of missing database file error."""
        error = self.error_handler.handle_missing_database_error()
        
        assert isinstance(error, DatabaseInitializationError)
        assert error.error_code == "DATABASE_NOT_FOUND"
        assert "not found" in str(error).lower()
        assert len(error.suggestions) > 0
        assert "automatically" in error.suggestions[0].lower()
        
        # Check context information
        assert error.context["database_path"] == str(self.database_path)
        assert error.context["parent_directory_exists"] == self.temp_dir.exists()
    
    def test_handle_permission_error(self):
        """Test handling of permission errors."""
        original_error = PermissionError("Access denied")
        error = self.error_handler.handle_permission_error(original_error, "create")
        
        assert isinstance(error, DatabasePermissionError)
        assert error.error_code == "DATABASE_PERMISSION_DENIED"
        assert "permission denied" in str(error).lower()
        assert "create" in str(error)
        assert len(error.suggestions) > 0
        
        # Check for specific permission suggestions
        suggestions_text = " ".join(error.suggestions).lower()
        assert "chmod" in suggestions_text or "permission" in suggestions_text
    
    def test_handle_corruption_error(self):
        """Test handling of database corruption errors."""
        original_error = Exception("Database is corrupted")
        backup_path = self.temp_dir / "backup.duckdb"
        
        error = self.error_handler.handle_corruption_error(
            original_error, 
            backup_created=True, 
            backup_path=backup_path
        )
        
        assert isinstance(error, DatabaseCorruptionError)
        assert error.error_code == "DATABASE_CORRUPTED"
        assert "corruption" in str(error).lower()
        assert len(error.suggestions) > 0
        
        # Check backup information in context
        assert error.context["backup_created"] is True
        assert error.context["backup_path"] == str(backup_path)
    
    def test_handle_disk_space_error(self):
        """Test handling of disk space errors."""
        required_space = 100 * 1024 * 1024  # 100MB
        error = self.error_handler.handle_disk_space_error(required_space)
        
        assert isinstance(error, DiskSpaceError)
        assert error.error_code == "INSUFFICIENT_DISK_SPACE"
        assert "disk space" in str(error).lower()
        assert len(error.suggestions) > 0
        
        # Check context information
        assert error.context["required_space_bytes"] == required_space
        assert "disk_space_info" in error.context
    
    def test_handle_connection_error(self):
        """Test handling of database connection errors."""
        original_error = Exception("Connection failed")
        error = self.error_handler.handle_connection_error(original_error, "connect")
        
        assert isinstance(error, DatabaseConnectionError)
        assert error.error_code == "DATABASE_CONNECTION_FAILED"
        assert "connect" in str(error).lower()
        assert len(error.suggestions) > 0
        
        # Check context information
        assert error.context["operation"] == "connect"
        assert error.context["original_error"] == str(original_error)
    
    def test_format_error_for_user(self):
        """Test user-friendly error formatting."""
        error = DatabaseInitializationError(
            message="Test error",
            error_code="TEST_ERROR",
            suggestions=["Try this", "Try that"],
            context={"database_path": str(self.database_path)}
        )
        
        formatted = self.error_handler.format_error_for_user(error)
        
        assert "❌" in formatted
        assert "💡" in formatted
        assert "Test error" in formatted
        assert "Try this" in formatted
        assert "Try that" in formatted
        assert str(self.database_path) in formatted
    
    def test_format_error_for_mcp(self):
        """Test MCP-formatted error response."""
        error = DatabaseInitializationError(
            message="Test error",
            error_code="TEST_ERROR",
            suggestions=["Try this"],
            context={"database_path": str(self.database_path)}
        )
        
        formatted = self.error_handler.format_error_for_mcp(error)
        
        assert formatted["success"] is False
        assert formatted["error"]["code"] == "TEST_ERROR"
        assert formatted["error"]["message"] == "Test error"
        assert "Try this" in formatted["error"]["suggestions"]
        assert "troubleshooting" in formatted
        assert "common_causes" in formatted["troubleshooting"]
    
    @patch('os.access')
    def test_permissions_info_gathering(self, mock_access):
        """Test gathering of detailed permissions information."""
        # Mock permission checks
        mock_access.side_effect = lambda path, mode: True
        
        # Create the parent directory
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        permissions_info = self.error_handler._get_permissions_info()
        
        assert "parent_exists" in permissions_info
        assert "parent_readable" in permissions_info
        assert "parent_writable" in permissions_info
        assert "file_exists" in permissions_info
        assert "file_readable" in permissions_info
        assert "file_writable" in permissions_info
    
    @patch('shutil.disk_usage')
    def test_disk_space_info_gathering(self, mock_disk_usage):
        """Test gathering of disk space information."""
        # Mock disk usage
        mock_usage = Mock()
        mock_usage.total = 1000000000  # 1GB
        mock_usage.used = 500000000    # 500MB
        mock_usage.free = 500000000    # 500MB
        mock_disk_usage.return_value = mock_usage
        
        disk_info = self.error_handler._get_disk_space_info()
        
        assert disk_info["total_bytes"] == 1000000000
        assert disk_info["used_bytes"] == 500000000
        assert disk_info["available_bytes"] == 500000000


class TestDatabaseRecoveryStrategy:
    """Test the enhanced DatabaseRecoveryStrategy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.database_path = self.temp_dir / "test.duckdb"
        self.settings = Mock(spec=JarvisSettings)
        self.settings.database_backup_on_corruption = True
        self.recovery_strategy = DatabaseRecoveryStrategy(self.database_path, self.settings)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('duckdb.connect')
    def test_handle_missing_file_success(self, mock_connect):
        """Test successful handling of missing database file."""
        # Mock successful database creation
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Mock the _get_database_state method to return a healthy state
        with patch.object(self.recovery_strategy, '_get_database_state') as mock_get_state:
            mock_state = Mock()
            mock_state.exists = True
            mock_state.path = self.database_path
            mock_state.size_bytes = 1024
            mock_state.created_at = None
            mock_state.last_modified = None
            mock_state.schema_version = "1.0.0"
            mock_state.is_healthy = True
            mock_state.error_message = None
            mock_state.table_count = 2
            mock_state.note_count = 0
            mock_state.embedding_count = 0
            mock_get_state.return_value = mock_state
            
            result = self.recovery_strategy.handle_missing_file()
        
        assert result.success is True
        assert result.action_taken == "created"
        assert result.database_state.exists is True
        assert len(result.warnings) >= 0
    
    @patch('duckdb.connect')
    def test_handle_missing_file_permission_error(self, mock_connect):
        """Test handling of permission error during database creation."""
        # Mock permission error
        mock_connect.side_effect = PermissionError("Access denied")
        
        result = self.recovery_strategy.handle_missing_file()
        
        assert result.success is False
        assert result.action_taken == "failed"
        assert "permission" in result.error_message.lower()
        assert len(result.warnings) > 0
    
    @patch('shutil.disk_usage')
    @patch('duckdb.connect')
    def test_handle_missing_file_disk_space_error(self, mock_connect, mock_disk_usage):
        """Test handling of insufficient disk space during database creation."""
        # Mock insufficient disk space
        mock_usage = Mock()
        mock_usage.free = 1024 * 1024  # 1MB (less than required 10MB)
        mock_disk_usage.return_value = mock_usage
        
        result = self.recovery_strategy.handle_missing_file()
        
        assert result.success is False
        assert result.action_taken == "failed"
        assert "disk space" in result.error_message.lower()
        assert len(result.warnings) > 0
    
    def test_handle_permission_error(self):
        """Test handling of permission errors with enhanced messaging."""
        original_error = PermissionError("Access denied")
        
        result = self.recovery_strategy.handle_permission_error(original_error)
        
        assert result.success is False
        assert result.action_taken == "failed"
        assert "permission" in result.error_message.lower()
        assert len(result.warnings) > 0
        
        # Check that suggestions are provided
        suggestions_text = " ".join(result.warnings).lower()
        assert "permission" in suggestions_text or "chmod" in suggestions_text
    
    @patch('shutil.copy2')
    @patch('pathlib.Path.unlink')
    def test_handle_corruption_with_backup(self, mock_unlink, mock_copy):
        """Test handling of corruption with backup creation."""
        # Create a fake corrupted database file
        self.database_path.touch()
        
        # Mock successful backup and removal
        mock_copy.return_value = None
        mock_unlink.return_value = None
        
        # Mock the handle_missing_file call that happens after corruption cleanup
        with patch.object(self.recovery_strategy, 'handle_missing_file') as mock_handle_missing:
            mock_result = Mock()
            mock_result.success = True
            mock_result.action_taken = "created"
            mock_result.warnings = []
            mock_handle_missing.return_value = mock_result
            
            result = self.recovery_strategy.handle_corruption()
        
        assert result.success is True
        assert result.action_taken == "recreated_after_corruption"
        assert any("backup" in warning.lower() for warning in result.warnings)
    
    def test_handle_corruption_backup_disabled(self):
        """Test handling of corruption with backup disabled."""
        # Disable backup in settings
        self.settings.database_backup_on_corruption = False
        
        # Create a fake corrupted database file
        self.database_path.touch()
        
        # Mock the handle_missing_file call
        with patch.object(self.recovery_strategy, 'handle_missing_file') as mock_handle_missing:
            mock_result = Mock()
            mock_result.success = True
            mock_result.action_taken = "created"
            mock_result.warnings = []
            mock_handle_missing.return_value = mock_result
            
            with patch('pathlib.Path.unlink'):
                result = self.recovery_strategy.handle_corruption()
        
        assert result.success is True
        assert any("backup creation was disabled" in warning.lower() for warning in result.warnings)


class TestDatabaseInitializer:
    """Test the enhanced DatabaseInitializer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.database_path = self.temp_dir / "test.duckdb"
        self.settings = Mock(spec=JarvisSettings)
        self.settings.database_backup_on_corruption = True
        self.settings.database_schema_version = "1.0.0"
        self.initializer = DatabaseInitializer(self.database_path, self.settings)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('duckdb.connect')
    def test_ensure_database_exists_success(self, mock_connect):
        """Test successful database initialization."""
        # Mock successful database operations
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = None
        
        # Mock the recovery strategy
        with patch.object(self.initializer.recovery_strategy, '_get_database_state') as mock_get_state:
            mock_state = Mock()
            mock_state.exists = True
            mock_state.is_healthy = True
            mock_get_state.return_value = mock_state
            
            result = self.initializer.ensure_database_exists()
        
        assert result is True
    
    def test_ensure_database_exists_failure(self):
        """Test database initialization failure with enhanced error handling."""
        # Mock initialization failure
        with patch.object(self.initializer, '_initialize_database') as mock_init:
            mock_init.side_effect = Exception("Initialization failed")
            
            result = self.initializer.ensure_database_exists()
        
        assert result is False
    
    @patch('duckdb.connect')
    def test_validate_database_success(self, mock_connect):
        """Test successful database validation."""
        # Create a fake database file
        self.database_path.touch()
        
        # Mock successful database connection and queries
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchone.return_value = (1,)  # Mock successful query
        
        result = self.initializer.validate_database()
        
        assert result is True
    
    def test_validate_database_missing_file(self):
        """Test database validation with missing file."""
        # Don't create the database file
        result = self.initializer.validate_database()
        
        assert result is False
    
    @patch('duckdb.connect')
    def test_validate_database_connection_error(self, mock_connect):
        """Test database validation with connection error."""
        # Create a fake database file
        self.database_path.touch()
        
        # Mock connection error
        mock_connect.side_effect = Exception("Connection failed")
        
        result = self.initializer.validate_database()
        
        assert result is False
    
    def test_get_database_info(self):
        """Test getting database information."""
        # Mock the recovery strategy's _get_database_state method
        with patch.object(self.initializer.recovery_strategy, '_get_database_state') as mock_get_state:
            mock_state = Mock()
            mock_state.exists = True
            mock_state.path = self.database_path
            mock_state.size_bytes = 1024
            mock_state.created_at = None
            mock_state.last_modified = None
            mock_state.schema_version = "1.0.0"
            mock_state.is_healthy = True
            mock_state.error_message = None
            mock_state.table_count = 2
            mock_state.note_count = 10
            mock_state.embedding_count = 5
            mock_get_state.return_value = mock_state
            
            info = self.initializer.get_database_info()
        
        assert info["exists"] is True
        assert info["path"] == str(self.database_path)
        assert info["size_bytes"] == 1024
        assert info["size_mb"] == 0.0  # 1024 bytes = 0.0 MB (rounded)
        assert info["schema_version"] == "1.0.0"
        assert info["is_healthy"] is True
        assert info["table_count"] == 2
        assert info["note_count"] == 10
        assert info["embedding_count"] == 5
        assert info["has_embeddings"] is True


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.database_path = self.temp_dir / "integration_test.duckdb"
        self.settings = Mock(spec=JarvisSettings)
        self.settings.database_backup_on_corruption = True
        self.settings.database_schema_version = "1.0.0"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_error_handling(self):
        """Test end-to-end error handling flow."""
        # Create an initializer
        initializer = DatabaseInitializer(self.database_path, self.settings)
        error_handler = DatabaseErrorHandler(self.database_path)
        
        # Test missing database error handling
        missing_error = error_handler.handle_missing_database_error()
        assert isinstance(missing_error, DatabaseInitializationError)
        
        # Test MCP formatting
        mcp_formatted = error_handler.format_error_for_mcp(missing_error)
        assert mcp_formatted["success"] is False
        assert "troubleshooting" in mcp_formatted
        
        # Test user formatting
        user_formatted = error_handler.format_error_for_user(missing_error)
        assert "❌" in user_formatted
        assert "💡" in user_formatted
    
    def test_permission_error_flow(self):
        """Test permission error handling flow."""
        error_handler = DatabaseErrorHandler(self.database_path)
        
        # Simulate permission error
        perm_error = PermissionError("Access denied")
        handled_error = error_handler.handle_permission_error(perm_error, "create database")
        
        # Verify error properties
        assert isinstance(handled_error, DatabasePermissionError)
        assert handled_error.error_code == "DATABASE_PERMISSION_DENIED"
        assert len(handled_error.suggestions) > 0
        
        # Test formatting
        user_message = error_handler.format_error_for_user(handled_error)
        assert "permission" in user_message.lower()
        assert "chmod" in user_message.lower() or "access" in user_message.lower()
    
    def test_corruption_recovery_flow(self):
        """Test corruption recovery error handling flow."""
        error_handler = DatabaseErrorHandler(self.database_path)
        
        # Simulate corruption error
        corruption_error = Exception("Database file is corrupted")
        backup_path = self.database_path.with_suffix(".backup.duckdb")
        
        handled_error = error_handler.handle_corruption_error(
            corruption_error, 
            backup_created=True, 
            backup_path=backup_path
        )
        
        # Verify error properties
        assert isinstance(handled_error, DatabaseCorruptionError)
        assert handled_error.error_code == "DATABASE_CORRUPTED"
        assert len(handled_error.suggestions) > 0
        
        # Check backup information
        assert handled_error.context["backup_created"] is True
        assert handled_error.context["backup_path"] == str(backup_path)
        
        # Test formatting
        user_message = error_handler.format_error_for_user(handled_error)
        assert "corruption" in user_message.lower()
        assert "backup" in user_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])