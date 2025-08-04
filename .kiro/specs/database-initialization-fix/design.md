# Design Document

## Overview

This design addresses the database initialization issue in the Jarvis MCP server where the system fails to start when the configured database file doesn't exist. The solution involves implementing robust database initialization logic that can create and initialize the database file when it's missing, while maintaining backward compatibility with existing installations.

## Architecture

### Current Problem

The current architecture has a critical flaw in the startup sequence:

1. `run_mcp_server()` is called with a database path
2. The function immediately tries to test database accessibility by opening it in read-only mode
3. If the database file doesn't exist, DuckDB throws an error: "database does not exist"
4. The MCP server fails to start

### Proposed Solution Architecture

The solution implements a three-phase database initialization approach:

1. **Pre-flight Check Phase**: Check if database file exists and is accessible
2. **Initialization Phase**: Create and initialize database if missing
3. **Validation Phase**: Verify database health before proceeding with server startup

## Components and Interfaces

### 1. Database Initialization Service

```python
class DatabaseInitializer:
    """Handles database creation and initialization logic."""
    
    def __init__(self, database_path: Path, settings: JarvisSettings):
        self.database_path = database_path
        self.settings = settings
    
    def ensure_database_exists(self) -> bool:
        """Ensure database exists and is properly initialized."""
        
    def create_database(self) -> None:
        """Create a new database with proper schema."""
        
    def validate_database(self) -> bool:
        """Validate existing database health and schema."""
        
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database state."""
```

### 2. Enhanced VectorDatabase Class

Modify the existing `VectorDatabase` class to support initialization mode:

```python
class VectorDatabase(IVectorDatabase):
    def __init__(self, database_path: Path, read_only: bool = False, create_if_missing: bool = False):
        """Initialize with option to create database if missing."""
        
    @classmethod
    def ensure_exists(cls, database_path: Path) -> bool:
        """Class method to ensure database exists before opening."""
```

### 3. Modified Server Startup Logic

Update `run_mcp_server()` to use the new initialization logic:

```python
async def run_mcp_server(
    vaults: Dict[str, Path],
    database_path: Path,
    settings: Optional[JarvisSettings] = None,
    watch: bool = False
) -> None:
    """Enhanced server startup with database initialization."""
    
    # Phase 1: Pre-flight check
    initializer = DatabaseInitializer(database_path, settings)
    
    # Phase 2: Ensure database exists
    if not initializer.ensure_database_exists():
        raise ServiceError("Failed to initialize database")
    
    # Phase 3: Proceed with normal startup
    # ... existing server creation logic
```

### 4. CLI Command for Manual Database Management

Add a new CLI command for manual database operations:

```python
@click.command()
@click.option('--database-path', help='Database file path')
@click.option('--force', is_flag=True, help='Force recreation of existing database')
def init_database(database_path: Optional[str], force: bool):
    """Initialize or reset the Jarvis database."""
```

## Data Models

### Database State Information

```python
@dataclass
class DatabaseState:
    """Information about database state."""
    exists: bool
    path: Path
    size_bytes: int
    created_at: Optional[datetime]
    last_modified: Optional[datetime]
    schema_version: Optional[str]
    is_healthy: bool
    error_message: Optional[str]
    
    # Statistics
    table_count: int
    note_count: int
    embedding_count: int
```

### Initialization Result

```python
@dataclass
class InitializationResult:
    """Result of database initialization attempt."""
    success: bool
    action_taken: str  # "created", "validated", "migrated", "failed"
    database_state: DatabaseState
    error_message: Optional[str]
    warnings: List[str]
    duration_ms: float
```

## Error Handling

### Error Categories and Responses

1. **Missing Database File**
   - Action: Create new database with schema
   - Log: "Database file not found, creating new database at {path}"
   - User feedback: Clear success message with database location

2. **Missing Database Directory**
   - Action: Create directory structure, then create database
   - Log: "Database directory not found, creating {directory}"
   - User feedback: Directory and database creation confirmation

3. **Permission Issues**
   - Action: Provide clear error message with remediation steps
   - Log: "Permission denied accessing database at {path}"
   - User feedback: Specific instructions for fixing permissions

4. **Corrupted Database**
   - Action: Attempt backup and recreation (with user confirmation)
   - Log: "Database appears corrupted, attempting recovery"
   - User feedback: Recovery options and backup information

5. **Schema Version Mismatch**
   - Action: Attempt migration or provide upgrade guidance
   - Log: "Database schema version mismatch, migration required"
   - User feedback: Migration status and any manual steps needed

### Error Recovery Strategies

```python
class DatabaseRecoveryStrategy:
    """Strategies for handling database issues."""
    
    def handle_missing_file(self, path: Path) -> InitializationResult:
        """Handle missing database file."""
        
    def handle_permission_error(self, path: Path, error: Exception) -> InitializationResult:
        """Handle permission-related errors."""
        
    def handle_corruption(self, path: Path) -> InitializationResult:
        """Handle database corruption."""
        
    def handle_schema_mismatch(self, path: Path, current_version: str, expected_version: str) -> InitializationResult:
        """Handle schema version mismatches."""
```

## Testing Strategy

### Unit Tests

1. **DatabaseInitializer Tests**
   - Test database creation with various path scenarios
   - Test permission handling
   - Test schema initialization
   - Test validation logic

2. **VectorDatabase Enhancement Tests**
   - Test `create_if_missing` parameter
   - Test `ensure_exists` class method
   - Test error handling for various failure modes

3. **Server Startup Tests**
   - Test startup with missing database
   - Test startup with existing database
   - Test startup with corrupted database
   - Test startup with permission issues

### Integration Tests

1. **End-to-End Initialization Tests**
   - Test complete MCP server startup with missing database
   - Test server startup with existing database
   - Test recovery scenarios

2. **CLI Command Tests**
   - Test manual database initialization
   - Test force recreation
   - Test error scenarios

### Test Data Scenarios

```python
# Test scenarios to cover
TEST_SCENARIOS = [
    "missing_database_file",
    "missing_database_directory", 
    "existing_healthy_database",
    "existing_corrupted_database",
    "permission_denied_directory",
    "permission_denied_file",
    "disk_space_insufficient",
    "schema_version_mismatch",
    "concurrent_access_conflict"
]
```

## Implementation Plan

### Phase 1: Core Database Initialization

1. Create `DatabaseInitializer` class with basic functionality
2. Enhance `VectorDatabase` class with `create_if_missing` option
3. Add database existence checking logic
4. Implement basic schema creation

### Phase 2: Enhanced Error Handling

1. Implement comprehensive error detection
2. Add recovery strategies for common issues
3. Improve error messages and user guidance
4. Add logging for troubleshooting

### Phase 3: Server Integration

1. Modify `run_mcp_server()` to use new initialization logic
2. Update MCP main entry point
3. Add proper error propagation
4. Test with various startup scenarios

### Phase 4: CLI Tools and Documentation

1. Add CLI command for manual database management
2. Create troubleshooting documentation
3. Add configuration examples
4. Update installation guides

## Configuration Changes

### New Configuration Options

```python
# Add to JarvisSettings
database_auto_create: bool = Field(
    default=True,
    env="JARVIS_DATABASE_AUTO_CREATE",
    description="Automatically create database if it doesn't exist"
)

database_backup_on_corruption: bool = Field(
    default=True,
    env="JARVIS_DATABASE_BACKUP_ON_CORRUPTION", 
    description="Create backup before attempting corruption recovery"
)

database_schema_version: str = Field(
    default="1.0.0",
    env="JARVIS_DATABASE_SCHEMA_VERSION",
    description="Expected database schema version"
)
```

### Environment Variable Updates

```bash
# New environment variables
JARVIS_DATABASE_AUTO_CREATE=true
JARVIS_DATABASE_BACKUP_ON_CORRUPTION=true
JARVIS_DATABASE_SCHEMA_VERSION=1.0.0
```

## Backward Compatibility

### Existing Installation Support

1. **Existing Databases**: No changes to existing database files
2. **Configuration**: All existing configuration options remain valid
3. **API**: No breaking changes to public APIs
4. **Behavior**: Default behavior creates database if missing (opt-out available)

### Migration Path

1. **Automatic**: Most users will see automatic database creation
2. **Manual**: Users can disable auto-creation and use CLI tools
3. **Validation**: Existing databases are validated but not modified
4. **Rollback**: Users can revert to previous behavior via configuration

## Performance Considerations

### Initialization Performance

- Database creation: < 1 second for empty database
- Schema validation: < 100ms for existing database
- Directory creation: < 50ms
- Error detection: < 200ms

### Memory Usage

- Initialization process: < 10MB additional memory
- No persistent memory overhead after initialization
- Temporary database connections closed promptly

### Disk Usage

- Empty database: ~100KB initial size
- Schema overhead: ~10KB
- Backup files: Same size as original (only when needed)

## Security Considerations

### File System Security

1. **Permissions**: Respect existing file system permissions
2. **Directory Creation**: Use secure directory creation (755 permissions)
3. **File Creation**: Use secure file creation (644 permissions)
4. **Path Validation**: Validate database paths to prevent directory traversal

### Data Security

1. **Backup Handling**: Secure backup file creation and cleanup
2. **Error Messages**: Avoid exposing sensitive path information in logs
3. **Recovery**: Secure handling of corrupted database data
4. **Cleanup**: Proper cleanup of temporary files

## Monitoring and Observability

### Metrics

```python
# New metrics to track
database_initialization_duration_seconds = Histogram(
    'jarvis_database_initialization_duration_seconds',
    'Time taken to initialize database'
)

database_initialization_attempts_total = Counter(
    'jarvis_database_initialization_attempts_total',
    'Total database initialization attempts',
    ['result']  # success, failure, skipped
)

database_health_check_duration_seconds = Histogram(
    'jarvis_database_health_check_duration_seconds', 
    'Time taken for database health checks'
)
```

### Logging

```python
# Structured logging for database operations
logger.info("Database initialization started", extra={
    "database_path": str(database_path),
    "auto_create": settings.database_auto_create,
    "operation": "initialize"
})

logger.info("Database initialization completed", extra={
    "database_path": str(database_path),
    "action_taken": result.action_taken,
    "duration_ms": result.duration_ms,
    "operation": "initialize"
})
```

## Documentation Updates

### User Documentation

1. **Installation Guide**: Add database initialization steps
2. **Troubleshooting**: Add database-related troubleshooting section
3. **Configuration**: Document new database configuration options
4. **CLI Reference**: Document new database management commands

### Developer Documentation

1. **Architecture**: Update architecture documentation
2. **API Reference**: Document new classes and methods
3. **Testing**: Document testing strategies and scenarios
4. **Contributing**: Update contribution guidelines for database changes