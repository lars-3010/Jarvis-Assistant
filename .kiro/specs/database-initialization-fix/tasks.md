# Implementation Plan

- [x] 1. Create DatabaseInitializer class with core functionality
  - Create new file `src/jarvis/services/database_initializer.py` with `DatabaseInitializer` class
  - Implement `ensure_database_exists()` method to check if database file exists and create if missing
  - Implement `create_database()` method to create new database with proper schema initialization
  - Implement `validate_database()` method to check existing database health
  - Add comprehensive error handling for file system operations and database creation
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3_

- [x] 2. Enhance VectorDatabase class with initialization support
  - Modify `VectorDatabase.__init__()` to accept `create_if_missing` parameter
  - Add `@classmethod ensure_exists()` method to check database existence before opening
  - Update database connection logic to handle missing database files gracefully
  - Ensure backward compatibility with existing VectorDatabase usage
  - Add proper error messages for database initialization failures
  - _Requirements: 1.1, 1.4, 3.1, 3.3_

- [x] 3. Create data models for database state tracking
  - Create `DatabaseState` dataclass in `src/jarvis/models/database.py` to track database information
  - Create `InitializationResult` dataclass to capture initialization attempt results
  - Add validation methods to ensure data consistency
  - Include comprehensive metadata fields for troubleshooting
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 4. Implement database recovery strategies
  - Create `DatabaseRecoveryStrategy` class to handle various database issues
  - Implement `handle_missing_file()` method for missing database scenarios
  - Implement `handle_permission_error()` method with clear user guidance
  - Implement `handle_corruption()` method with backup and recovery options
  - Add comprehensive logging for all recovery operations
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5. Update MCP server startup logic
  - Modify `run_mcp_server()` function in `src/jarvis/mcp/server.py` to use DatabaseInitializer
  - Replace direct database health check with initialization-aware logic
  - Add proper error handling and user-friendly error messages
  - Ensure graceful fallback when database initialization fails
  - Update logging to provide clear feedback about database operations
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3_

- [ ] 6. Add configuration options for database initialization
  - Add `database_auto_create`, `database_backup_on_corruption`, and `database_schema_version` fields to `JarvisSettings`
  - Update environment variable handling for new database configuration options
  - Add validation for new configuration parameters
  - Update default configuration values to enable auto-creation by default
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 7. Create CLI command for manual database management
  - Add new `init-database` command to main CLI interface
  - Implement database initialization with `--force` option for recreation
  - Add database status reporting functionality
  - Include comprehensive error handling and user feedback
  - Add confirmation prompts for destructive operations
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 8. Write comprehensive unit tests for database initialization
  - Create `test_database_initializer.py` with tests for all DatabaseInitializer methods
  - Test database creation with various path scenarios (missing directory, existing file, etc.)
  - Test permission handling and error recovery strategies
  - Test VectorDatabase enhancements with `create_if_missing` parameter
  - Mock file system operations for reliable testing
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3_

- [ ] 9. Write integration tests for server startup scenarios
  - Create `test_mcp_server_initialization.py` for end-to-end startup testing
  - Test MCP server startup with missing database file
  - Test server startup with existing healthy database
  - Test server startup with corrupted database scenarios
  - Test server startup with permission issues
  - Verify proper error messages and recovery behavior
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3_

- [ ] 10. Add performance monitoring and metrics
  - Add database initialization duration metrics to monitoring system
  - Implement counters for initialization attempts and results
  - Add structured logging for database operations with proper context
  - Include database state information in health check responses
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 11. Update error handling and user messaging
  - Enhance error messages throughout the database initialization flow
  - Add specific guidance for common issues (permissions, disk space, corruption)
  - Implement user-friendly error reporting in MCP server responses
  - Add troubleshooting hints to error messages
  - _Requirements: 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

- [ ] 12. Test with real-world scenarios and edge cases
  - Test initialization with the actual configured database path from .env file
  - Verify behavior when database directory exists but file is missing
  - Test concurrent access scenarios during initialization
  - Validate proper cleanup of temporary files and connections
  - Test with various file system permissions and disk space conditions
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3_