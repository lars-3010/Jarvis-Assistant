"""
Enhanced error handling and user messaging for database initialization.

This module provides comprehensive error handling with user-friendly messages,
specific guidance for common issues, and troubleshooting hints.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from jarvis.utils.errors import (
    DatabaseError,
    DatabaseInitializationError,
    DatabasePermissionError,
    DatabaseCorruptionError,
    DatabaseConnectionError,
    DiskSpaceError
)


@dataclass
class ErrorContext:
    """Context information for database errors."""
    database_path: Path
    operation: str
    timestamp: float
    system_info: Dict[str, Any]
    additional_context: Dict[str, Any]


class DatabaseErrorHandler:
    """Enhanced error handler for database operations with user-friendly messaging."""
    
    def __init__(self, database_path: Path):
        """Initialize error handler.
        
        Args:
            database_path: Path to the database file
        """
        self.database_path = database_path
    
    def handle_missing_database_error(self, context: Optional[Dict[str, Any]] = None) -> DatabaseInitializationError:
        """Handle missing database file error with comprehensive guidance.
        
        Args:
            context: Additional context information
            
        Returns:
            DatabaseInitializationError with detailed guidance
        """
        message = f"Database file not found at '{self.database_path}'"
        
        suggestions = [
            "The database will be created automatically on first startup",
            f"Ensure the directory '{self.database_path.parent}' is writable",
            "Check that the database path in your configuration is correct",
            "If you moved the database file, update your configuration or move it back"
        ]
        
        # Add directory-specific suggestions
        if not self.database_path.parent.exists():
            suggestions.insert(1, f"Create the directory: mkdir -p '{self.database_path.parent}'")
        
        error_context = {
            "database_path": str(self.database_path),
            "parent_directory_exists": self.database_path.parent.exists(),
            "parent_directory_writable": self._check_directory_writable(self.database_path.parent),
            "suggested_action": "automatic_creation"
        }
        
        if context:
            error_context.update(context)
        
        return DatabaseInitializationError(
            message=message,
            error_code="DATABASE_NOT_FOUND",
            suggestions=suggestions,
            context=error_context
        )
    
    def handle_permission_error(self, original_error: Exception, operation: str = "access") -> DatabasePermissionError:
        """Handle permission-related errors with specific guidance.
        
        Args:
            original_error: The original permission error
            operation: The operation that failed (access, create, write, etc.)
            
        Returns:
            DatabasePermissionError with detailed guidance
        """
        message = f"Permission denied: Cannot {operation} database at '{self.database_path}'"
        
        # Get current permissions
        permissions_info = self._get_permissions_info()
        
        suggestions = []
        
        # Directory-specific suggestions
        if not permissions_info["parent_readable"]:
            suggestions.append(f"Grant read access to directory: chmod +r '{self.database_path.parent}'")
        
        if not permissions_info["parent_writable"]:
            suggestions.append(f"Grant write access to directory: chmod +w '{self.database_path.parent}'")
        
        # File-specific suggestions
        if self.database_path.exists():
            if not permissions_info["file_readable"]:
                suggestions.append(f"Grant read access to file: chmod +r '{self.database_path}'")
            
            if not permissions_info["file_writable"]:
                suggestions.append(f"Grant write access to file: chmod +w '{self.database_path}'")
        
        # General suggestions
        suggestions.extend([
            "Check if another process is using the database file",
            "Ensure you have the necessary user permissions",
            "Try running with appropriate user privileges",
            f"Verify ownership: ls -la '{self.database_path.parent}'"
        ])
        
        # Add macOS-specific suggestions if applicable
        if self._is_macos():
            suggestions.extend([
                "On macOS, check System Preferences > Security & Privacy > Privacy > Full Disk Access",
                "Ensure your terminal/application has necessary permissions"
            ])
        
        error_context = {
            "database_path": str(self.database_path),
            "operation": operation,
            "original_error": str(original_error),
            "permissions_info": permissions_info,
            "system_platform": os.name
        }
        
        return DatabasePermissionError(
            message=message,
            error_code="DATABASE_PERMISSION_DENIED",
            suggestions=suggestions,
            context=error_context
        )
    
    def handle_corruption_error(self, original_error: Exception, backup_created: bool = False, backup_path: Optional[Path] = None) -> DatabaseCorruptionError:
        """Handle database corruption with recovery guidance.
        
        Args:
            original_error: The original corruption error
            backup_created: Whether a backup was created
            backup_path: Path to the backup file if created
            
        Returns:
            DatabaseCorruptionError with recovery guidance
        """
        message = f"Database corruption detected at '{self.database_path}'"
        
        suggestions = [
            "A new database will be created automatically",
            "Your data may be recoverable from the backup"
        ]
        
        if backup_created and backup_path:
            suggestions.extend([
                f"Backup created at: {backup_path}",
                "You can attempt manual recovery from the backup if needed",
                f"To restore backup: cp '{backup_path}' '{self.database_path}'"
            ])
        else:
            suggestions.append("No backup was created - data may be lost")
        
        suggestions.extend([
            "Check disk space and file system health",
            "Run disk utility to check for file system errors",
            "Consider backing up your vault files separately"
        ])
        
        # Add file system check suggestions
        if self._is_macos():
            suggestions.append("Run Disk Utility or 'diskutil verifyVolume /' to check file system")
        elif self._is_linux():
            suggestions.append("Run 'fsck' to check file system integrity")
        
        error_context = {
            "database_path": str(self.database_path),
            "original_error": str(original_error),
            "backup_created": backup_created,
            "backup_path": str(backup_path) if backup_path else None,
            "file_size": self._get_file_size(),
            "disk_space": self._get_disk_space_info()
        }
        
        return DatabaseCorruptionError(
            message=message,
            error_code="DATABASE_CORRUPTED",
            suggestions=suggestions,
            context=error_context
        )
    
    def handle_disk_space_error(self, required_space: Optional[int] = None) -> DiskSpaceError:
        """Handle insufficient disk space error.
        
        Args:
            required_space: Required space in bytes
            
        Returns:
            DiskSpaceError with space management guidance
        """
        disk_info = self._get_disk_space_info()
        available_mb = disk_info["available_bytes"] / (1024 * 1024)
        
        message = f"Insufficient disk space for database at '{self.database_path}'"
        
        suggestions = [
            f"Available space: {available_mb:.1f} MB",
            "Free up disk space by removing unnecessary files",
            "Consider moving the database to a location with more space",
            "Empty trash/recycle bin to free up space"
        ]
        
        if required_space:
            required_mb = required_space / (1024 * 1024)
            suggestions.insert(1, f"Required space: {required_mb:.1f} MB")
        
        # Add platform-specific cleanup suggestions
        if self._is_macos():
            suggestions.extend([
                "Use 'About This Mac > Storage > Manage' for cleanup recommendations",
                "Clear Downloads folder and empty Trash",
                "Use 'sudo du -sh /* | sort -hr' to find large directories"
            ])
        elif self._is_linux():
            suggestions.extend([
                "Use 'df -h' to check disk usage",
                "Use 'du -sh /* | sort -hr' to find large directories",
                "Clear package caches: 'sudo apt clean' or equivalent"
            ])
        
        error_context = {
            "database_path": str(self.database_path),
            "disk_space_info": disk_info,
            "required_space_bytes": required_space
        }
        
        return DiskSpaceError(
            message=message,
            error_code="INSUFFICIENT_DISK_SPACE",
            suggestions=suggestions,
            context=error_context
        )
    
    def handle_connection_error(self, original_error: Exception, operation: str = "connect") -> DatabaseConnectionError:
        """Handle database connection errors.
        
        Args:
            original_error: The original connection error
            operation: The operation that failed
            
        Returns:
            DatabaseConnectionError with connection guidance
        """
        message = f"Failed to {operation} to database at '{self.database_path}'"
        
        suggestions = [
            "Check if the database file is accessible",
            "Ensure no other process is using the database exclusively",
            "Verify the database file is not corrupted",
            "Check file permissions and ownership"
        ]
        
        # Add specific suggestions based on error type
        error_str = str(original_error).lower()
        
        if "locked" in error_str or "busy" in error_str:
            suggestions.extend([
                "Another process may be using the database",
                "Wait a moment and try again",
                "Check for running Jarvis processes: ps aux | grep jarvis"
            ])
        
        if "permission" in error_str:
            suggestions.extend([
                "Check file permissions with: ls -la '{}'".format(self.database_path),
                "Ensure you have read/write access to the database file"
            ])
        
        error_context = {
            "database_path": str(self.database_path),
            "operation": operation,
            "original_error": str(original_error),
            "file_exists": self.database_path.exists(),
            "file_size": self._get_file_size(),
            "permissions_info": self._get_permissions_info()
        }
        
        return DatabaseConnectionError(
            message=message,
            error_code="DATABASE_CONNECTION_FAILED",
            suggestions=suggestions,
            context=error_context
        )
    
    def handle_generic_database_error(self, original_error: Exception, operation: str) -> DatabaseError:
        """Handle generic database errors with basic guidance.
        
        Args:
            original_error: The original error
            operation: The operation that failed
            
        Returns:
            DatabaseError with basic guidance
        """
        message = f"Database error during {operation}: {original_error}"
        
        suggestions = [
            "Check the database file and directory permissions",
            "Ensure sufficient disk space is available",
            "Verify the database path is correct",
            "Try restarting the application",
            "Check system logs for additional error details"
        ]
        
        error_context = {
            "database_path": str(self.database_path),
            "operation": operation,
            "original_error": str(original_error),
            "error_type": type(original_error).__name__,
            "system_info": self._get_system_info()
        }
        
        return DatabaseError(
            message=message,
            error_code="DATABASE_ERROR",
            suggestions=suggestions,
            context=error_context
        )
    
    def format_error_for_user(self, error: DatabaseError) -> str:
        """Format error for user-friendly display.
        
        Args:
            error: The database error to format
            
        Returns:
            Formatted error message with suggestions
        """
        lines = [
            f"❌ {error}",
            "",
            "💡 Troubleshooting Steps:"
        ]
        
        for i, suggestion in enumerate(error.suggestions, 1):
            lines.append(f"   {i}. {suggestion}")
        
        if error.context.get("database_path"):
            lines.extend([
                "",
                "📍 Database Information:",
                f"   Path: {error.context['database_path']}"
            ])
        
        return "\n".join(lines)
    
    def format_error_for_mcp(self, error: DatabaseError) -> Dict[str, Any]:
        """Format error for MCP server response.
        
        Args:
            error: The database error to format
            
        Returns:
            Structured error response for MCP
        """
        return {
            "success": False,
            "error": {
                "code": error.error_code,
                "message": str(error),
                "details": "Database initialization failed",
                "suggestions": error.suggestions,
                "context": {
                    "database_path": error.context.get("database_path"),
                    "operation": error.context.get("operation"),
                    "timestamp": error.context.get("timestamp")
                }
            },
            "troubleshooting": {
                "common_causes": self._get_common_causes(error.error_code),
                "next_steps": self._get_next_steps(error.error_code),
                "documentation_links": self._get_documentation_links(error.error_code)
            }
        }
    
    def _check_directory_writable(self, directory: Path) -> bool:
        """Check if directory is writable."""
        try:
            if not directory.exists():
                return False
            return os.access(directory, os.W_OK)
        except Exception:
            return False
    
    def _get_permissions_info(self) -> Dict[str, bool]:
        """Get detailed permissions information."""
        info = {
            "parent_exists": self.database_path.parent.exists(),
            "parent_readable": False,
            "parent_writable": False,
            "file_exists": self.database_path.exists(),
            "file_readable": False,
            "file_writable": False
        }
        
        try:
            if info["parent_exists"]:
                info["parent_readable"] = os.access(self.database_path.parent, os.R_OK)
                info["parent_writable"] = os.access(self.database_path.parent, os.W_OK)
            
            if info["file_exists"]:
                info["file_readable"] = os.access(self.database_path, os.R_OK)
                info["file_writable"] = os.access(self.database_path, os.W_OK)
        except Exception:
            pass
        
        return info
    
    def _get_file_size(self) -> Optional[int]:
        """Get database file size in bytes."""
        try:
            if self.database_path.exists():
                return self.database_path.stat().st_size
        except Exception:
            pass
        return None
    
    def _get_disk_space_info(self) -> Dict[str, int]:
        """Get disk space information."""
        try:
            if self.database_path.parent.exists():
                usage = shutil.disk_usage(self.database_path.parent)
                return {
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "available_bytes": usage.free
                }
        except Exception:
            pass
        
        return {
            "total_bytes": 0,
            "used_bytes": 0,
            "available_bytes": 0
        }
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information."""
        return {
            "platform": os.name,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "working_directory": str(Path.cwd())
        }
    
    def _is_macos(self) -> bool:
        """Check if running on macOS."""
        return os.name == 'posix' and os.uname().sysname == 'Darwin'
    
    def _is_linux(self) -> bool:
        """Check if running on Linux."""
        return os.name == 'posix' and os.uname().sysname == 'Linux'
    
    def _get_common_causes(self, error_code: str) -> List[str]:
        """Get common causes for specific error codes."""
        causes_map = {
            "DATABASE_NOT_FOUND": [
                "First time running Jarvis Assistant",
                "Database file was moved or deleted",
                "Incorrect database path in configuration"
            ],
            "DATABASE_PERMISSION_DENIED": [
                "Insufficient file system permissions",
                "Database file is owned by another user",
                "Directory is not writable",
                "Security software blocking access"
            ],
            "DATABASE_CORRUPTED": [
                "Unexpected shutdown during database write",
                "Disk space exhaustion during operation",
                "File system errors or hardware issues",
                "Concurrent access without proper locking"
            ],
            "INSUFFICIENT_DISK_SPACE": [
                "Disk is full or nearly full",
                "Large temporary files consuming space",
                "Database growth exceeded available space"
            ],
            "DATABASE_CONNECTION_FAILED": [
                "Database file is locked by another process",
                "File permissions prevent access",
                "Database file is corrupted",
                "Network storage connectivity issues"
            ]
        }
        
        return causes_map.get(error_code, ["Unknown error cause"])
    
    def _get_next_steps(self, error_code: str) -> List[str]:
        """Get recommended next steps for specific error codes."""
        steps_map = {
            "DATABASE_NOT_FOUND": [
                "Allow automatic database creation",
                "Verify configuration settings",
                "Check directory permissions"
            ],
            "DATABASE_PERMISSION_DENIED": [
                "Fix file permissions",
                "Run with appropriate user privileges",
                "Check security software settings"
            ],
            "DATABASE_CORRUPTED": [
                "Allow automatic recovery",
                "Restore from backup if available",
                "Check file system integrity"
            ],
            "INSUFFICIENT_DISK_SPACE": [
                "Free up disk space",
                "Move database to larger storage",
                "Clean up temporary files"
            ],
            "DATABASE_CONNECTION_FAILED": [
                "Check for running processes",
                "Verify file permissions",
                "Restart the application"
            ]
        }
        
        return steps_map.get(error_code, ["Contact support for assistance"])
    
    def _get_documentation_links(self, error_code: str) -> List[str]:
        """Get relevant documentation links for error codes."""
        # In a real implementation, these would be actual documentation URLs
        return [
            "https://docs.jarvis-assistant.dev/troubleshooting/database-issues",
            "https://docs.jarvis-assistant.dev/installation/setup-guide",
            "https://docs.jarvis-assistant.dev/configuration/database-config"
        ]