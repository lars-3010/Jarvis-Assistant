# Error Handling Architecture

## Comprehensive Error Management and User Guidance System

## Overview

Jarvis Assistant implements a sophisticated error handling architecture that provides comprehensive error recovery, user-friendly messaging, and actionable guidance. The system transforms technical errors into meaningful user experiences while maintaining system reliability.

## Core Components

### DatabaseErrorHandler Class

The central component for database-related error management:

```python
class DatabaseErrorHandler:
    def __init__(self, database_path: Path)
    def handle_missing_database_error(self, context: Optional[Dict[str, Any]] = None) -> DatabaseInitializationError
    def handle_permission_error(self, original_error: Exception, operation: str = "access") -> DatabasePermissionError
    def handle_corruption_error(self, original_error: Exception, backup_created: bool = False, backup_path: Optional[Path] = None) -> DatabaseCorruptionError
    def handle_disk_space_error(self, required_space: Optional[int] = None) -> DiskSpaceError
    def handle_connection_error(self, original_error: Exception, operation: str = "connect") -> DatabaseConnectionError
```

### Enhanced Error Types

**Hierarchical Error Structure**:
```
JarvisError (Base)
├── DatabaseError
│   ├── DatabaseInitializationError
│   ├── DatabasePermissionError
│   ├── DatabaseCorruptionError
│   ├── DatabaseConnectionError
│   └── DiskSpaceError
├── ServiceError
│   ├── ServiceUnavailableError
│   └── ValidationError
└── ToolExecutionError
```

### Error Context System

Each error includes comprehensive context information:

```python
@dataclass
class ErrorContext:
    database_path: Path
    operation: str
    timestamp: float
    system_info: Dict[str, Any]
    additional_context: Dict[str, Any]
```

## Error Handling Patterns

### 1. Database Initialization Errors

**Missing Database File**:
```python
def handle_missing_database_error(self, context: Optional[Dict[str, Any]] = None) -> DatabaseInitializationError:
    message = f"Database file not found at '{self.database_path}'"
    
    suggestions = [
        "The database will be created automatically on first startup",
        f"Ensure the directory '{self.database_path.parent}' is writable",
        "Check that the database path in your configuration is correct"
    ]
    
    return DatabaseInitializationError(
        message=message,
        error_code="DATABASE_NOT_FOUND",
        suggestions=suggestions,
        context=error_context
    )
```

**Permission Errors**:
- Platform-specific guidance (macOS, Linux, Windows)
- Specific permission commands
- Security software considerations
- File ownership verification

**Corruption Recovery**:
- Automatic backup creation
- Recovery guidance
- File system health checks
- Manual recovery options

### 2. Service Degradation Patterns

**Graph Service Fallback**:
```python
# In MCP server implementation
if not context.graph_database or not context.graph_database.is_healthy:
    logger.warning("Graph search unavailable, falling back to semantic search.")
    fallback_results = await _handle_semantic_search(context, fallback_args)
    
    fallback_message = types.TextContent(
        type="text",
        text="**Graph search is unavailable. Falling back to semantic search.**\n\n"
    )
    return [fallback_message] + fallback_results
```

### 3. MCP Tool Error Integration

**Enhanced Error Responses**:
```python
except DatabaseError as db_error:
    logger.error(f"Database error in tool {name}: {db_error}")
    
    # Format enhanced database error for MCP response
    error_handler = DatabaseErrorHandler(context.database_path)
    formatted_error = error_handler.format_error_for_user(db_error)
    
    return [
        types.TextContent(
            type="text",
            text=f"Database Error in {name}:\n\n{formatted_error}"
        )
    ]
```

## Error Response Formats

### User-Friendly Format

```
❌ Database corruption detected at '/path/to/database.db'

💡 Troubleshooting Steps:
   1. A new database will be created automatically
   2. Your data may be recoverable from the backup
   3. Backup created at: /path/to/database.db.backup.2024-01-15
   4. Check disk space and file system health

📍 Database Information:
   Path: /path/to/database.db
```

### MCP Protocol Format

```json
{
    "success": false,
    "error": {
        "code": "DATABASE_CORRUPTED",
        "message": "Database corruption detected",
        "details": "Database initialization failed",
        "suggestions": [
            "A new database will be created automatically",
            "Your data may be recoverable from the backup"
        ]
    },
    "troubleshooting": {
        "common_causes": [
            "Unexpected shutdown during database write",
            "Disk space exhaustion during operation"
        ],
        "next_steps": [
            "Allow automatic recovery",
            "Restore from backup if available"
        ]
    }
}
```

## Platform-Specific Guidance

### macOS-Specific Handling

```python
def _is_macos(self) -> bool:
    return os.name == 'posix' and os.uname().sysname == 'Darwin'

# macOS-specific suggestions
if self._is_macos():
    suggestions.extend([
        "On macOS, check System Preferences > Security & Privacy > Privacy > Full Disk Access",
        "Ensure your terminal/application has necessary permissions",
        "Run Disk Utility or 'diskutil verifyVolume /' to check file system"
    ])
```

### Linux-Specific Handling

```python
elif self._is_linux():
    suggestions.extend([
        "Use 'df -h' to check disk usage",
        "Use 'du -sh /* | sort -hr' to find large directories",
        "Run 'fsck' to check file system integrity"
    ])
```

## Error Recovery Strategies

### Automatic Recovery

| Error Type | Recovery Action | User Notification |
|------------|----------------|-------------------|
| **Missing Database** | Create new database with schema | "Database created automatically" |
| **Corruption** | Backup corrupted file, create new | "Database recovered, backup preserved" |
| **Schema Mismatch** | Log warning, continue operation | "Database version compatibility noted" |

### User-Guided Recovery

| Error Type | User Action Required | Guidance Provided |
|------------|---------------------|-------------------|
| **Permission Denied** | Fix file permissions | Specific chmod/chown commands |
| **Disk Space** | Free up space | Platform-specific cleanup suggestions |
| **Service Unavailable** | Check service status | Service-specific troubleshooting |

## Integration Points

### MCP Server Integration

The error handling system is fully integrated into the MCP server:

```python
# From src/jarvis/mcp/server.py
try:
    results = []
    if name == "search-semantic":
        results = await _handle_semantic_search(context, arguments or {})
    # ... other tools
    
except DatabaseError as db_error:
    error_handler = DatabaseErrorHandler(context.database_path)
    formatted_error = error_handler.format_error_for_user(db_error)
    return [types.TextContent(type="text", text=formatted_error)]

except JarvisError as e:
    error_text = f"❌ Internal error in {name}: {str(e)}"
    if hasattr(e, 'suggestions') and e.suggestions:
        error_text += "\n\n💡 Troubleshooting Steps:"
        for i, suggestion in enumerate(e.suggestions, 1):
            error_text += f"\n   {i}. {suggestion}"
    return [types.TextContent(type="text", text=error_text)]
```

### Database Initialization Integration

```python
# From src/jarvis/services/database_initializer.py
try:
    self._create_database_file()
except PermissionError as e:
    error_handler = DatabaseErrorHandler(self.database_path)
    raise error_handler.handle_permission_error(e, "create database")
except OSError as e:
    if "No space left on device" in str(e):
        error_handler = DatabaseErrorHandler(self.database_path)
        raise error_handler.handle_disk_space_error()
    else:
        raise
```

## Testing Strategy

### Error Scenario Testing

```python
def test_database_permission_error_handling():
    """Test permission error handling with specific guidance."""
    handler = DatabaseErrorHandler(Path("/restricted/database.db"))
    
    permission_error = PermissionError("Permission denied")
    result = handler.handle_permission_error(permission_error, "create")
    
    assert result.error_code == "DATABASE_PERMISSION_DENIED"
    assert "chmod" in " ".join(result.suggestions)
    assert result.context["operation"] == "create"

def test_mcp_tool_error_integration():
    """Test MCP tool error integration."""
    # Simulate database error in MCP tool
    with patch('jarvis.mcp.server.context.database') as mock_db:
        mock_db.side_effect = DatabaseCorruptionError("Database corrupted")
        
        response = await handle_call_tool("search-semantic", {"query": "test"})
        
        assert "Database Error" in response[0].text
        assert "💡 Troubleshooting Steps:" in response[0].text
```

### Error Recovery Testing

```python
def test_automatic_database_recovery():
    """Test automatic database recovery flow."""
    corrupted_db_path = create_corrupted_database()
    
    initializer = DatabaseInitializer(corrupted_db_path, settings)
    result = initializer.ensure_database_exists()
    
    assert result is True
    assert corrupted_db_path.with_suffix('.backup').exists()
    assert is_valid_database(corrupted_db_path)
```

## Monitoring and Metrics

### Error Tracking

```python
class ErrorMetrics:
    def record_error(self, error_type: str, error_code: str, recovery_successful: bool):
        self.error_counter.labels(
            error_type=error_type,
            error_code=error_code,
            recovered=str(recovery_successful)
        ).inc()
    
    def record_recovery_time(self, error_type: str, duration_seconds: float):
        self.recovery_time_histogram.labels(error_type=error_type).observe(duration_seconds)
```

### Health Check Integration

```python
def get_error_health_status(self) -> Dict[str, Any]:
    return {
        "error_handling_enabled": True,
        "recent_errors": self.get_recent_errors(hours=24),
        "recovery_success_rate": self.calculate_recovery_success_rate(),
        "common_error_types": self.get_common_error_types()
    }
```

## Future Enhancements

### Planned Improvements

1. **Predictive Error Prevention**: Analyze patterns to prevent errors before they occur
2. **Self-Healing Capabilities**: Automatic system repair for common issues
3. **Enhanced User Guidance**: Interactive troubleshooting workflows
4. **Error Analytics**: Detailed error pattern analysis and reporting

### Extension Points

- **Custom Error Handlers**: Plugin system for domain-specific error handling
- **External Monitoring**: Integration with monitoring systems
- **User Feedback Loop**: Error resolution feedback collection
- **Automated Reporting**: Error trend analysis and reporting

---

*This error handling architecture ensures robust system operation while providing excellent user experience during error conditions.*