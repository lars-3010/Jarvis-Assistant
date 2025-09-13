# Error Codes

Comprehensive reference for all error codes, messages, and resolution strategies in Jarvis Assistant. This guide helps diagnose and resolve issues quickly.

## Quick Navigation

- [Error Code Format](#error-code-format)
- [MCP Tool Errors](#mcp-tool-errors)
- [Database Errors](#database-errors)
- [Configuration Errors](#configuration-errors)
- [System Errors](#system-errors)
- [Resolution Strategies](#resolution-strategies)

---

## Error Code Format

### Error Message Structure

All errors follow a consistent format for easy identification and resolution:

```
Error: {error_message}
```

For detailed errors:
```
{Tool/Service} error: {specific_error_message}
```

For system errors:
```
{Component} failed: {reason} - {details}
```

### Custom Exception Hierarchy

Jarvis Assistant uses a custom exception hierarchy for more precise error handling and reporting:

-   **`JarvisError`**: Base exception for all Jarvis Assistant errors. All other custom exceptions inherit from this.
-   **`ConfigurationError`**: Raised when there is an issue with the application configuration (e.g., missing required settings, invalid paths).
-   **`ServiceError`**: Base exception for errors occurring within service layers (e.g., `VectorDatabase`, `VaultReader`).
-   **`ServiceUnavailableError`**: A specific `ServiceError` raised when a required external service (e.g., Neo4j) is unavailable or cannot be connected to.
-   **`ValidationError`**: Raised when input data or internal data fails validation checks.
-   **`ToolExecutionError`**: Raised when an MCP tool encounters an error during its execution.

### Error Categories

| Category | Prefix | Severity | Description |
|----------|--------|----------|-------------|
| **Input Validation** | `Error:` | Low | Invalid user input parameters |
| **Resource Access** | `Error:` | Medium | File/vault access issues |
| **Service Errors** | `{Service} error:` | High | Database or service failures |
| **System Errors** | `{Component} failed:` | Critical | System-level failures |

---

## MCP Tool Errors

### search-semantic

#### SEARCH_001: Empty Query
```
Error: Query parameter is required
```
**Cause**: Query parameter is missing or empty string  
**Resolution**: Provide a non-empty query string  
**Example**: `{"query": "machine learning"}`

#### SEARCH_002: Invalid Limit
```
Error: Limit must be between 1 and 50
```
**Cause**: Limit parameter outside valid range  
**Resolution**: Use limit between 1 and 50  
**Example**: `{"query": "test", "limit": 10}`

#### SEARCH_003: Invalid Similarity Threshold
```
Error: Similarity threshold must be between 0.0 and 1.0
```
**Cause**: Similarity threshold outside valid range  
**Resolution**: Use threshold between 0.0 and 1.0  
**Example**: `{"query": "test", "similarity_threshold": 0.8}`

#### SEARCH_004: Unknown Vault
```
Error: Unknown vault 'vault_name'
```
**Cause**: Specified vault name doesn't exist  
**Resolution**: Check available vaults with `list-vaults` tool  
**Example**: Use one of the configured vault names

#### SEARCH_005: Search Operation Failed
```
Search error: Database connection failed
```
**Cause**: Vector database unavailable or corrupted  
**Resolution**: Check database file and re-index if necessary  
**Commands**: 
```bash
uv run jarvis index --vault /path/to/vault
```

#### SEARCH_006: Encoding Failed
```
Search error: Failed to encode query
```
**Cause**: Text encoding model unavailable or failed  
**Resolution**: Check model installation and device settings  
**Commands**:
```bash
# Check model availability
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### SEARCH_007: No Results
```
No results found for query: 'query_text'
```
**Cause**: Query didn't match any indexed documents  
**Resolution**: Try different search terms or check if vault is indexed  
**Note**: This is informational, not an error

### read-note

#### READ_001: Missing Path
```
Error: Path parameter is required
```
**Cause**: Path parameter is missing  
**Resolution**: Provide a valid file path  
**Example**: `{"path": "notes/example.md"}`

#### READ_002: Invalid Vault
```
Error: Unknown vault 'vault_name'
```
**Cause**: Specified vault doesn't exist  
**Resolution**: Check available vaults or omit vault parameter  
**Commands**: Use `list-vaults` to see available vaults

#### READ_003: File Not Found
```
Error reading note: File not found
```
**Cause**: Specified file doesn't exist in vault  
**Resolution**: Check file path and ensure file exists  
**Commands**: Use `search-vault` to find available files

#### READ_004: No Vaults Available
```
Error: No vaults available
```
**Cause**: No vaults configured or accessible  
**Resolution**: Configure vault in environment or settings  
**Commands**:
```bash
export JARVIS_VAULT_PATH="/path/to/vault"
```

#### READ_005: File Access Error
```
Error reading note: Permission denied
```
**Cause**: Insufficient permissions to read file  
**Resolution**: Check file permissions  
**Commands**:
```bash
chmod 644 /path/to/file.md
```

#### READ_006: File Too Large
```
Error reading note: File size exceeds limit
```
**Cause**: File larger than maximum allowed size  
**Resolution**: File size limit or split large files  
**Default Limit**: 10MB

### search-vault

#### VAULT_001: Empty Query
```
Error: Query parameter is required
```
**Cause**: Query parameter missing or empty  
**Resolution**: Provide search query  
**Example**: `{"query": "meeting"}`

#### VAULT_002: Invalid Vault
```
Error: Unknown vault 'vault_name'
```
**Cause**: Specified vault doesn't exist  
**Resolution**: Use existing vault name or omit parameter  

#### VAULT_003: Search Failed
```
Error searching vault: I/O operation failed
```
**Cause**: File system error during search  
**Resolution**: Check vault path and permissions  
**Commands**:
```bash
ls -la "$JARVIS_VAULT_PATH"
```

#### VAULT_004: No Results
```
No results found in filenames for query: 'query_text'
No results found in content and filenames for query: 'query_text'
```
**Cause**: No files match search criteria  
**Resolution**: Try different search terms or check vault contents  
**Note**: Informational message, not an error

### search-graph

#### GRAPH_001: Missing Path
```
Error: query_note_path parameter is required
```
**Cause**: Query note path parameter missing  
**Resolution**: Provide path to center note  
**Example**: `{"query_note_path": "concepts/ai.md"}`

#### GRAPH_002: Graph Service Unavailable
```
Graph search is currently unavailable: Neo4j connection failed
```
**Cause**: Neo4j database not running or misconfigured  
**Resolution**: Start Neo4j and check connection settings  
**Commands**:
```bash
# Check Neo4j status
neo4j status

# Start Neo4j
neo4j start

# Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p password "RETURN 1"
```

#### GRAPH_003: Connection Timeout
```
Graph search failed due to database connection issues: Connection timeout
```
**Cause**: Neo4j database unresponsive  
**Resolution**: Check Neo4j performance and configuration  
**Commands**:
```bash
# Check Neo4j logs
tail -f /var/log/neo4j/neo4j.log
```

#### GRAPH_004: Authentication Failed
```
Graph search is currently unavailable: Authentication failed
```
**Cause**: Invalid Neo4j credentials  
**Resolution**: Check NEO4J_USER and NEO4J_PASSWORD  
**Commands**:
```bash
echo $NEO4J_USER
echo $NEO4J_PASSWORD
```

#### GRAPH_005: No Results
```
No results found for query: 'note_path'
```
**Cause**: Note not found in graph or no connections  
**Resolution**: Check if note exists and is indexed in graph  
**Commands**:
```bash
uv run jarvis graph-index --vault /path/to/vault
```

#### GRAPH_006: Query Failed
```
Graph search failed: Cypher query execution error
```
**Cause**: Graph database query error  
**Resolution**: Check graph database integrity  
**Commands**:
```bash
# Test basic query
cypher-shell -a bolt://localhost:7687 -u neo4j -p password "MATCH (n) RETURN count(n)"
```

### list-vaults

#### LIST_001: Service Error
```
Error listing vaults: Database initialization failed
```
**Cause**: Cannot access vault statistics from database  
**Resolution**: Check database file and permissions  
**Commands**:
```bash
ls -la ~/.jarvis/jarvis.duckdb
```

---

## Database Errors

### Vector Database (DuckDB)

#### DB_001: Connection Failed
```
VectorDatabase error: Unable to connect to database
```
**Cause**: Database file corrupted or inaccessible  
**Resolution**: Re-create database or fix permissions  
**Commands**:
```bash
# Remove corrupted database
rm ~/.jarvis/jarvis.duckdb

# Re-index vault
uv run jarvis index --vault /path/to/vault
```

#### DB_002: Table Not Found
```
VectorDatabase error: Table 'documents' does not exist
```
**Cause**: Database not properly initialized  
**Resolution**: Run indexing to create tables  
**Commands**:
```bash
uv run jarvis index --vault /path/to/vault
```

#### DB_003: Disk Space Error
```
VectorDatabase error: No space left on device
```
**Cause**: Insufficient disk space for database operations  
**Resolution**: Free up disk space or change database location  
**Commands**:
```bash
df -h ~/.jarvis/
export JARVIS_DB_PATH="/new/location/jarvis.duckdb"
```

#### DB_004: Permission Denied
```
VectorDatabase error: Permission denied
```
**Cause**: Insufficient permissions for database file  
**Resolution**: Fix file permissions  
**Commands**:
```bash
chmod 644 ~/.jarvis/jarvis.duckdb
chmod 755 ~/.jarvis/
```

#### DB_005: Lock Error
```
VectorDatabase error: Database is locked
```
**Cause**: Another process accessing database  
**Resolution**: Close other processes or wait for completion  
**Commands**:
```bash
# Find processes using database
lsof ~/.jarvis/jarvis.duckdb

# Kill if necessary
kill -9 <PID>
```

### Graph Database (Neo4j)

#### NEO4J_001: Service Unavailable
```
Neo4j connection failed: ServiceUnavailable
```
**Cause**: Neo4j service not running  
**Resolution**: Start Neo4j service  
**Commands**:
```bash
# Linux/macOS
sudo systemctl start neo4j
# or
neo4j start

# Check status
neo4j status
```

#### NEO4J_002: Authentication Error
```
Neo4j connection failed: AuthError
```
**Cause**: Invalid username or password  
**Resolution**: Check credentials and reset if necessary  
**Commands**:
```bash
# Reset password
neo4j-admin set-initial-password new_password

# Update environment
export NEO4J_PASSWORD=new_password
```

#### NEO4J_003: Connection Timeout
```
Neo4j connection failed: ConnectionTimeout
```
**Cause**: Network or performance issues  
**Resolution**: Check network connectivity and Neo4j performance  
**Commands**:
```bash
# Test connection
telnet localhost 7687

# Check Neo4j logs
tail -f /var/log/neo4j/neo4j.log
```

#### NEO4J_004: Memory Error
```
Neo4j connection failed: OutOfMemoryError
```
**Cause**: Insufficient memory for Neo4j operations  
**Resolution**: Increase Neo4j memory allocation  
**Configuration**:
```
# neo4j.conf
dbms.memory.heap.initial_size=1G
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G
```

#### NEO4J_005: Transaction Failed
```
Neo4j query failed: TransactionError
```
**Cause**: Query execution failed  
**Resolution**: Check query syntax and database state  
**Commands**:
```bash
# Test basic connectivity
cypher-shell -a bolt://localhost:7687 -u neo4j -p password "RETURN 1"
```

---

## Configuration Errors

### Environment Configuration

#### CONFIG_001: Missing Required Variable
```
Configuration error: NEO4J_PASSWORD environment variable is required
```
**Cause**: Required environment variable not set  
**Resolution**: Set the missing environment variable  
**Commands**:
```bash
export NEO4J_PASSWORD=your_password
```

#### CONFIG_002: Invalid Path
```
Configuration error: JARVIS_VAULT_PATH does not exist
```
**Cause**: Vault path points to non-existent directory  
**Resolution**: Create directory or update path  
**Commands**:
```bash
mkdir -p /path/to/vault
export JARVIS_VAULT_PATH=/path/to/vault
```

#### CONFIG_003: Invalid Format
```
Configuration error: JARVIS_LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
```
**Cause**: Invalid log level specification  
**Resolution**: Use valid log level  
**Commands**:
```bash
export JARVIS_LOG_LEVEL=INFO
```

### Model Configuration

#### MODEL_001: Model Download Failed
```
Model error: Failed to download sentence transformer model
```
**Cause**: Network issues or invalid model name  
**Resolution**: Check network and model name  
**Commands**:
```bash
# Test model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

#### MODEL_002: Device Not Available
```
Model error: CUDA device not available
```
**Cause**: CUDA requested but not available  
**Resolution**: Use CPU or install CUDA  
**Commands**:
```bash
export JARVIS_DEVICE=cpu
```

#### MODEL_003: Insufficient Memory
```
Model error: OutOfMemoryError during encoding
```
**Cause**: Model too large for available memory  
**Resolution**: Use smaller model or reduce batch size  
**Configuration**:
```bash
export JARVIS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
export JARVIS_BATCH_SIZE=8
```

---

## System Errors

### File System Errors

#### FS_001: Permission Denied
```
File system error: Permission denied accessing vault
```
**Cause**: Insufficient permissions for vault directory  
**Resolution**: Fix directory permissions  
**Commands**:
```bash
chmod -R 755 /path/to/vault
```

#### FS_002: Disk Space
```
File system error: No space left on device
```
**Cause**: Insufficient disk space  
**Resolution**: Free up space or change location  
**Commands**:
```bash
df -h
# Clean up unnecessary files
```

#### FS_003: Path Too Long
```
File system error: File name too long
```
**Cause**: File path exceeds system limits  
**Resolution**: Shorten file names or path structure  

### Memory Errors

#### MEM_001: Out of Memory
```
System error: OutOfMemoryError
```
**Cause**: Insufficient system memory  
**Resolution**: Reduce batch size or close other applications  
**Configuration**:
```bash
export JARVIS_BATCH_SIZE=8
export JARVIS_CACHE_SIZE=32
```

#### MEM_002: Memory Leak
```
System warning: High memory usage detected
```
**Cause**: Potential memory leak  
**Resolution**: Restart service and monitor usage  
**Commands**:
```bash
# Monitor memory usage
top -p $(pgrep -f jarvis)
```

### Network Errors

#### NET_001: Connection Refused
```
Network error: Connection refused
```
**Cause**: Service not listening on specified port  
**Resolution**: Check service status and port configuration  
**Commands**:
```bash
netstat -tlnp | grep 7687
```

#### NET_002: DNS Resolution Failed
```
Network error: Name resolution failed
```
**Cause**: Cannot resolve hostname  
**Resolution**: Check DNS settings or use IP address  
**Commands**:
```bash
nslookup localhost
```

---

## Resolution Strategies

### Quick Diagnostics

#### Check System Status
```bash
# Check all services
uv run jarvis-mcp-stdio --help
uv run jarvis list-vaults

# Check database
ls -la ~/.jarvis/
file ~/.jarvis/jarvis.duckdb

# Check Neo4j
neo4j status
cypher-shell -a bolt://localhost:7687 -u neo4j -p password "RETURN 1"
```

#### Environment Validation
```bash
# Check environment variables
env | grep JARVIS
env | grep NEO4J

# Validate paths
ls -la "$JARVIS_VAULT_PATH"
ls -la "$JARVIS_DB_PATH"
```

#### Test Components
```bash
# Test vector search
uv run python -c "
from jarvis.services.vector.database import VectorDatabase
db = VectorDatabase('$JARVIS_DB_PATH')
print('Vector database OK')
"

# Test graph search
uv run python -c "
from jarvis.services.graph.database import GraphDatabase
import os
db = GraphDatabase(os.getenv('NEO4J_URI'), os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
db.check_connection()
print('Graph database OK')
"
```

### Common Resolution Steps

#### 1. Restart Services
```bash
# Restart Neo4j
neo4j restart

# Re-index vault
uv run jarvis index --vault "$JARVIS_VAULT_PATH"
```

#### 2. Reset Configuration
```bash
# Reset to defaults
unset JARVIS_DB_PATH
unset JARVIS_MODEL_NAME
unset JARVIS_DEVICE

# Use minimal configuration
export JARVIS_VAULT_PATH=/path/to/vault
export NEO4J_PASSWORD=password
```

#### 3. Clean Reinstall
```bash
# Remove data
rm -rf ~/.jarvis/

# Reinstall dependencies
uv sync

# Re-index from scratch
uv run jarvis index --vault "$JARVIS_VAULT_PATH"
uv run jarvis graph-index --vault "$JARVIS_VAULT_PATH"
```

### Error Prevention

#### Regular Maintenance
```bash
# Check disk space
df -h ~/.jarvis/

# Check log files
tail -f ~/.jarvis/logs/jarvis.log

# Verify database integrity
uv run python -c "
import duckdb
conn = duckdb.connect('~/.jarvis/jarvis.duckdb')
print(conn.execute('PRAGMA database_size').fetchone())
"
```

#### Monitoring
```bash
# Monitor system resources
top -p $(pgrep -f jarvis)

# Check file descriptors
lsof -p $(pgrep -f jarvis)

# Monitor network connections
netstat -tulpn | grep jarvis
```

---

## Error Reporting

### Information to Include

When reporting errors, include:

1. **Error Message**: Complete error text
2. **Environment**: OS, Python version, Jarvis version
3. **Configuration**: Relevant environment variables
4. **Steps to Reproduce**: What actions led to the error
5. **Logs**: Recent log entries (sanitized)

### Log Collection
```bash
# Enable debug logging
export JARVIS_LOG_LEVEL=DEBUG

# Run failing command
uv run jarvis mcp --vault "$JARVIS_VAULT_PATH" 2>&1 | tee error.log

# Sanitize and share error.log
```

### Bug Report Template
```markdown
## Error Description
[Brief description of the error]

## Error Message
```
[Exact error message here]
```

## Environment
- OS: [e.g., macOS 14.0, Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- Jarvis Assistant: [e.g., 0.2.0]
- Neo4j: [e.g., 5.0.0]

## Configuration
```bash
JARVIS_VAULT_PATH=/path/to/vault
NEO4J_URI=bolt://localhost:7687
# (exclude passwords)
```

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [Error occurs]

## Expected Behavior
[What should have happened]

## Logs
```
[Relevant log entries - remove sensitive information]
```
```

---

## Next Steps

- [API Reference](api-reference.md) - Complete API documentation
- [Configuration Reference](configuration-reference.md) - Setup and configuration
- [Troubleshooting](../07-maintenance/troubleshooting.md) - Detailed problem resolution
- [Performance Tuning](../07-maintenance/performance-tuning.md) - Optimization techniques
