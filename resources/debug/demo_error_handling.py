#!/usr/bin/env python3
"""
Demonstration script for enhanced database error handling.

This script shows the improved error messages and troubleshooting guidance
provided by the enhanced database initialization error handling.
"""

import tempfile
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, '../../src')

from jarvis.utils.database_errors import DatabaseErrorHandler
from jarvis.services.database_initializer import DatabaseInitializer
from jarvis.utils.config import JarvisSettings


def demo_missing_database_error():
    """Demonstrate missing database error handling."""
    print("=" * 60)
    print("DEMO: Missing Database Error Handling")
    print("=" * 60)
    
    # Create a temporary path that doesn't exist
    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "nonexistent" / "database.duckdb"
    
    error_handler = DatabaseErrorHandler(database_path)
    error = error_handler.handle_missing_database_error()
    
    print("User-friendly error message:")
    print(error_handler.format_error_for_user(error))
    
    print("\nMCP-formatted error response:")
    mcp_response = error_handler.format_error_for_mcp(error)
    print(f"Success: {mcp_response['success']}")
    print(f"Error Code: {mcp_response['error']['code']}")
    print(f"Message: {mcp_response['error']['message']}")
    print("Suggestions:")
    for i, suggestion in enumerate(mcp_response['error']['suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_permission_error():
    """Demonstrate permission error handling."""
    print("\n" + "=" * 60)
    print("DEMO: Permission Error Handling")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "database.duckdb"
    
    error_handler = DatabaseErrorHandler(database_path)
    original_error = PermissionError("Access denied to database file")
    error = error_handler.handle_permission_error(original_error, "create database")
    
    print("User-friendly error message:")
    print(error_handler.format_error_for_user(error))
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_corruption_error():
    """Demonstrate corruption error handling."""
    print("\n" + "=" * 60)
    print("DEMO: Database Corruption Error Handling")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "database.duckdb"
    backup_path = temp_dir / "database.backup.duckdb"
    
    error_handler = DatabaseErrorHandler(database_path)
    original_error = Exception("Database file is corrupted and cannot be read")
    error = error_handler.handle_corruption_error(
        original_error, 
        backup_created=True, 
        backup_path=backup_path
    )
    
    print("User-friendly error message:")
    print(error_handler.format_error_for_user(error))
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_disk_space_error():
    """Demonstrate disk space error handling."""
    print("\n" + "=" * 60)
    print("DEMO: Disk Space Error Handling")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "database.duckdb"
    
    error_handler = DatabaseErrorHandler(database_path)
    required_space = 100 * 1024 * 1024  # 100MB
    error = error_handler.handle_disk_space_error(required_space)
    
    print("User-friendly error message:")
    print(error_handler.format_error_for_user(error))
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


def demo_database_initializer_integration():
    """Demonstrate database initializer with enhanced error handling."""
    print("\n" + "=" * 60)
    print("DEMO: Database Initializer Integration")
    print("=" * 60)
    
    # Create a path in a non-existent directory to trigger error handling
    temp_dir = Path(tempfile.mkdtemp())
    database_path = temp_dir / "test_database.duckdb"
    
    # Create mock settings
    class MockSettings:
        database_backup_on_corruption = True
        database_schema_version = "1.0.0"
    
    settings = MockSettings()
    initializer = DatabaseInitializer(database_path, settings)
    
    print(f"Attempting to initialize database at: {database_path}")
    
    # This should succeed and create the database
    success = initializer.ensure_database_exists()
    
    if success:
        print("✅ Database initialization successful!")
        
        # Get database information
        db_info = initializer.get_database_info()
        print(f"Database info:")
        print(f"  - Exists: {db_info['exists']}")
        print(f"  - Size: {db_info['size_mb']} MB")
        print(f"  - Schema version: {db_info['schema_version']}")
        print(f"  - Table count: {db_info['table_count']}")
        print(f"  - Note count: {db_info['note_count']}")
        print(f"  - Has embeddings: {db_info['has_embeddings']}")
    else:
        print("❌ Database initialization failed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Enhanced Database Error Handling Demonstration")
    print("=" * 60)
    print("This script demonstrates the improved error messages and")
    print("troubleshooting guidance for database initialization issues.")
    print()
    
    try:
        demo_missing_database_error()
        demo_permission_error()
        demo_corruption_error()
        demo_disk_space_error()
        demo_database_initializer_integration()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The enhanced error handling provides:")
        print("✅ User-friendly error messages with emojis")
        print("✅ Specific troubleshooting steps for each error type")
        print("✅ Platform-specific guidance (macOS, Linux)")
        print("✅ Structured error responses for MCP integration")
        print("✅ Comprehensive context information for debugging")
        print("✅ Graceful degradation and recovery strategies")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()