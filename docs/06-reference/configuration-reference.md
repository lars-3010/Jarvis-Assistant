# Configuration Reference

Complete reference for all configuration options, environment variables, and settings for Jarvis Assistant. This guide covers system configuration, performance tuning, and deployment options.

## Quick Navigation

- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Database Configuration](#database-configuration)
- [MCP Server Configuration](#mcp-server-configuration)
- [Performance Settings](#performance-settings)
- [Security Configuration](#security-configuration)
- [Logging Configuration](#logging-configuration)

---

## Environment Variables

### Core Configuration

## Environment Variables

### Application Settings

#### JARVIS_DEBUG
- **Type**: `boolean`
- **Description**: Enable debug mode for verbose logging and detailed error messages.
- **Default**: `False`

#### JARVIS_BACKEND_PORT
- **Type**: `integer`
- **Description**: Port for the backend API server (if enabled).
- **Default**: `8000`

#### JARVIS_CORS_ORIGINS
- **Type**: `list of strings`
- **Description**: Comma-separated list of allowed origins for CORS (Cross-Origin Resource Sharing).
- **Default**: `*` (allows all origins)

### LLM Settings

#### GOOGLE_API_KEY
- **Type**: `string`
- **Description**: API key for Google LLM services.
- **Default**: `None`

#### GEMINI_MODEL_ID
- **Type**: `string`
- **Description**: Identifier for the Gemini model to be used.
- **Default**: `gemini-1.5-flash-latest`

### Vault Settings

#### JARVIS_VAULT_PATH
- **Type**: `string` (file path)
- **Description**: Absolute path to the primary Obsidian vault.
- **Default**: `""` (empty string, must be configured)

#### JARVIS_VAULT_WATCH
- **Type**: `boolean`
- **Description**: Enable file system watching for automatic re-indexing of vault changes.
- **Default**: `True`

#### JARVIS_EXCLUDED_FOLDERS
- **Type**: `list of strings`
- **Description**: Comma-separated list of folder names to exclude from indexing and querying.
- **Example**: `Journaling,Atlas/People`
- **Default**: `Journaling,Atlas/People,Atlas/work People`

### Vector Database Settings (DuckDB)

#### JARVIS_VECTOR_DB_PATH
- **Type**: `string` (file path)
- **Description**: Path to the DuckDB database file used for vector storage.
- **Default**: `data/jarvis-vector.duckdb`

#### JARVIS_VECTOR_DB_READ_ONLY
- **Type**: `boolean`
- **Description**: Open the vector database in read-only mode. Useful for multiple consumers.
- **Default**: `False`

### Graph Database Settings (Neo4j)

#### JARVIS_GRAPH_ENABLED
- **Type**: `boolean`
- **Description**: Enable or disable Neo4j graph database integration. If disabled, graph search will fall back to semantic search.
- **Default**: `True`

#### JARVIS_NEO4J_URI
- **Type**: `string` (URI)
- **Description**: Connection URI for the Neo4j database.
- **Example**: `bolt://localhost:7687`
- **Default**: `bolt://localhost:7687`

#### JARVIS_NEO4J_USER
- **Type**: `string`
- **Description**: Username for Neo4j database authentication.
- **Default**: `neo4j`

#### JARVIS_NEO4J_PASSWORD
- **Type**: `string`
- **Description**: Password for Neo4j database authentication.
- **Default**: `password`

### Embedding Settings

#### JARVIS_EMBEDDING_MODEL_NAME
- **Type**: `string`
- **Description**: Name of the sentence transformer model used for generating embeddings.
- **Default**: `paraphrase-MiniLM-L6-v2`

#### JARVIS_EMBEDDING_DEVICE
- **Type**: `string`
- **Description**: PyTorch device to use for embedding generation (e.g., `cpu`, `cuda`, `mps`).
- **Default**: `mps` (Apple Silicon optimized)

#### JARVIS_EMBEDDING_BATCH_SIZE
- **Type**: `integer`
- **Description**: Batch size for processing documents during embedding generation.
- **Default**: `32`

### MCP Server Settings

#### JARVIS_MCP_SERVER_NAME
- **Type**: `string`
- **Description**: Name of the MCP server reported to clients.
- **Default**: `jarvis-assistant`

#### JARVIS_MCP_SERVER_VERSION
- **Type**: `string`
- **Description**: Version of the MCP server reported to clients.
- **Default**: `0.2.0`

#### JARVIS_MCP_CACHE_SIZE
- **Type**: `integer`
- **Description**: Maximum number of cached MCP tool call results.
- **Default**: `100`

#### JARVIS_MCP_CACHE_TTL
- **Type**: `integer`
- **Description**: Time to live for cached MCP entries in seconds.
- **Default**: `300`

### Indexing Settings

#### JARVIS_INDEX_BATCH_SIZE
- **Type**: `integer`
- **Description**: Batch size for processing documents during indexing.
- **Default**: `32`

#### JARVIS_INDEX_ENQUEUE_ALL
- **Type**: `boolean`
- **Description**: Force re-indexing of all documents on startup, regardless of changes.
- **Default**: `False`

### Search Settings

#### JARVIS_SEARCH_DEFAULT_LIMIT
- **Type**: `integer`
- **Description**: Default maximum number of results to return for search queries.
- **Default**: `10`

#### JARVIS_SEARCH_SIMILARITY_THRESHOLD
- **Type**: `float`
- **Description**: Minimum similarity score for search results. Lower values return more results.
- **Default**: `-10.0` (effectively no threshold for cosine distance)

### Logging Settings

#### JARVIS_LOG_LEVEL
- **Type**: `string`
- **Description**: Minimum logging level for application output.
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Default**: `INFO`

#### JARVIS_LOG_FORMAT
- **Type**: `string`
- **Description**: Python logging format string for log messages.
- **Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

## Configuration Files

### Settings File

Jarvis Assistant uses Pydantic settings for configuration. Create a `jarvis_settings.json` file:

```json
{
  "vault_paths": {
    "personal": "/Users/username/Documents/PersonalVault",
    "research": "/Users/username/Documents/ResearchVault"
  },
  "database_path": "/Users/username/.jarvis/jarvis.duckdb",
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_user": "neo4j",
  "neo4j_password": "your_password",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "device": "cpu",
  "log_level": "INFO",
  "batch_size": 32,
  "cache_size": 256,
  "similarity_threshold": 0.0,
  "max_workers": 4
}
```

### Claude Desktop Configuration

For Claude Desktop integration, configure the MCP server in Claude's settings:

```json
{
  "mcpServers": {
    "jarvis-assistant": {
      "command": "uv",
      "args": ["run", "jarvis", "mcp", "--vault", "/path/to/your/vault"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_password",
        "JARVIS_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Development Configuration

For development, create a `.env` file in the project root:

```env
# Core Configuration
JARVIS_VAULT_PATH=./resources/test_vault
JARVIS_DB_PATH=./data/jarvis_dev.duckdb
JARVIS_LOG_LEVEL=DEBUG

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=development_password

# Model Configuration
JARVIS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
JARVIS_DEVICE=cpu
JARVIS_BATCH_SIZE=16

# Performance Configuration
JARVIS_CACHE_SIZE=64
JARVIS_MAX_WORKERS=2
```

---

## Database Configuration

### DuckDB Configuration

#### Connection Settings

```python
# Database connection parameters
{
    "database_path": "/path/to/jarvis.duckdb",
    "read_only": False,  # Set to True for MCP server
    "access_mode": "automatic",
    "threads": 4,
    "memory_limit": "1GB",
    "temp_directory": "/tmp"
}
```

#### Performance Tuning

```sql
-- DuckDB performance settings
PRAGMA threads=4;
PRAGMA memory_limit='2GB';
PRAGMA temp_directory='/tmp/duckdb';
PRAGMA enable_progress_bar=true;
PRAGMA enable_profiling=true;
```

#### Vector Extension Configuration

```python
# Vector search configuration
{
    "vector_dimension": 384,  # Model-dependent
    "index_type": "hnsw",     # Hierarchical Navigable Small World
    "ef_construction": 200,   # Index construction parameter
    "m": 16,                  # Index connectivity parameter
    "distance_metric": "cosine"
}
```

### Neo4j Configuration

#### Connection Pool Settings

```python
# Neo4j driver configuration
{
    "uri": "bolt://localhost:7687",
    "auth": ("neo4j", "password"),
    "max_connection_lifetime": 3600,  # seconds
    "max_connection_pool_size": 50,
    "connection_acquisition_timeout": 60,  # seconds
    "encrypted": False,  # Set to True for production
    "trust": "TRUST_ALL_CERTIFICATES"
}
```

#### Database Settings

```cypher
// Neo4j database configuration
CALL dbms.setConfigValue('dbms.memory.heap.initial_size', '1G');
CALL dbms.setConfigValue('dbms.memory.heap.max_size', '2G');
CALL dbms.setConfigValue('dbms.memory.pagecache.size', '1G');
CALL dbms.setConfigValue('dbms.logs.query.enabled', 'true');
```

#### Index Configuration

```cypher
// Create indexes for better performance
CREATE INDEX note_path_index IF NOT EXISTS FOR (n:Note) ON (n.path);
CREATE INDEX note_title_index IF NOT EXISTS FOR (n:Note) ON (n.title);
CREATE INDEX note_tags_index IF NOT EXISTS FOR (n:Note) ON (n.tags);
CREATE FULLTEXT INDEX note_content_index IF NOT EXISTS FOR (n:Note) ON EACH [n.content];
```

---

## MCP Server Configuration

### Server Settings

```python
# MCP server configuration
{
    "server_name": "jarvis-assistant",
    "server_version": "0.2.0",
    "timeout": 30,  # seconds
    "max_message_size": 1048576,  # 1MB
    "enable_debug": False,
    "log_requests": True,
    "log_responses": False  # Set to True for debugging
}
```

### Tool Configuration

```python
# Individual tool settings
{
    "search_semantic": {
        "enabled": True,
        "max_results": 50,
        "default_results": 10,
        "cache_enabled": True,
        "cache_ttl": 300  # seconds
    },
    "search_vault": {
        "enabled": True,
        "max_results": 100,
        "default_results": 20,
        "content_search_enabled": True
    },
    "search_graph": {
        "enabled": True,
        "max_depth": 5,
        "default_depth": 1,
        "timeout": 30  # seconds
    },
    "read_note": {
        "enabled": True,
        "max_file_size": 10485760,  # 10MB
        "encoding": "utf-8"
    },
    "list_vaults": {
        "enabled": True,
        "include_stats": True,
        "include_model_info": True
    }
}
```

### Transport Configuration

#### Stdio Transport (Default)

```json
{
  "transport": "stdio",
  "stdio": {
    "buffer_size": 8192,
    "line_buffered": true,
    "encoding": "utf-8"
  }
}
```

#### TCP Transport (Optional)

```json
{
  "transport": "tcp",
  "tcp": {
    "host": "localhost",
    "port": 8080,
    "backlog": 5,
    "nodelay": true,
    "keepalive": true
  }
}
```

---

## Performance Settings

### Memory Configuration

#### System Memory

```python
# Memory limits and optimization
{
    "max_memory_usage": "2GB",
    "enable_memory_monitoring": True,
    "memory_warning_threshold": 0.8,  # 80%
    "gc_enabled": True,
    "gc_threshold": 1000  # objects
}
```

#### Cache Configuration

```python
# Caching settings
{
    "query_cache": {
        "enabled": True,
        "max_size": 256,
        "ttl": 300  # seconds
    },
    "encoding_cache": {
        "enabled": True,
        "max_size": 128,
        "ttl": 3600  # seconds
    },
    "file_cache": {
        "enabled": False,  # File content caching
        "max_size": 64,
        "ttl": 60  # seconds
    }
}
```

### Concurrency Settings

```python
# Threading and async configuration
{
    "max_concurrent_requests": 10,
    "request_timeout": 30,  # seconds
    "worker_threads": 4,
    "async_enabled": True,
    "connection_pool_size": 20
}
```

### Search Performance

```python
# Search optimization settings
{
    "semantic_search": {
        "batch_encoding": True,
        "batch_size": 32,
        "precompute_norms": True,
        "use_gpu": False
    },
    "graph_search": {
        "query_timeout": 30,  # seconds
        "max_traversal_depth": 5,
        "result_limit": 1000
    },
    "vault_search": {
        "file_type_filters": [".md", ".txt", ".org"],
        "max_file_size": 10485760,  # 10MB
        "content_preview_length": 200
    }
}
```

---

## Security Configuration

### Authentication Settings

```python
# Security configuration
{
    "require_authentication": False,  # For local use
    "api_key_required": False,
    "allowed_origins": ["*"],  # CORS settings
    "max_request_size": 1048576,  # 1MB
    "rate_limiting": {
        "enabled": False,
        "requests_per_minute": 60
    }
}
```

### File Access Security

```python
# File system security
{
    "vault_access": {
        "restrict_to_vault": True,
        "allow_symlinks": False,
        "max_path_depth": 20,
        "forbidden_patterns": [
            "*.exe", "*.bat", "*.sh", "*.py",
            "__pycache__", ".git", ".env"
        ]
    },
    "file_validation": {
        "check_file_type": True,
        "allowed_extensions": [".md", ".txt", ".org", ".rst"],
        "max_file_size": 10485760,  # 10MB
        "scan_for_malware": False
    }
}
```

### Network Security

```python
# Network security settings
{
    "ssl_enabled": False,  # For local development
    "ssl_cert_path": None,
    "ssl_key_path": None,
    "cipher_suites": ["ECDHE+AESGCM", "ECDHE+CHACHA20"],
    "min_tls_version": "1.2"
}
```

---

## Logging Configuration

### Log Levels and Output

```python
# Logging configuration
{
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": None,  # None for stdout/stderr
    "log_rotation": {
        "enabled": False,
        "max_size": "10MB",
        "backup_count": 5
    }
}
```

### Component-Specific Logging

```python
# Per-component log levels
{
    "loggers": {
        "jarvis.services.vector": "DEBUG",
        "jarvis.services.graph": "INFO", 
        "jarvis.mcp.server": "INFO",
        "jarvis.services.vault": "WARNING",
        "neo4j": "WARNING",
        "sentence_transformers": "ERROR"
    }
}
```

### Structured Logging

```python
# Structured logging format
{
    "structured_logging": {
        "enabled": True,
        "format": "json",
        "include_fields": [
            "timestamp", "level", "component",
            "operation", "duration", "error"
        ]
    }
}
```

### Log Output Examples

#### Standard Format
```
2024-01-15 14:30:22,123 - jarvis.services.vector.searcher - INFO - Search completed: query='machine learning', results=5, duration=0.234s
```

#### JSON Format
```json
{
  "timestamp": "2024-01-15T14:30:22.123Z",
  "level": "INFO",
  "component": "vector.searcher",
  "operation": "semantic_search",
  "query_length": 17,
  "results_count": 5,
  "duration_ms": 234,
  "similarity_scores": [0.95, 0.87, 0.82, 0.78, 0.73]
}
```

---

## Platform-Specific Configuration

### macOS Configuration

```bash
# macOS-specific settings
export JARVIS_VAULT_PATH="/Users/$USER/Documents/Obsidian Vault"
export JARVIS_DB_PATH="/Users/$USER/.jarvis/jarvis.duckdb"
export JARVIS_DEVICE="mps"  # For Apple Silicon

# Claude Desktop config location
~/.config/claude-desktop/claude_desktop_config.json
```

### Linux Configuration

```bash
# Linux-specific settings
export JARVIS_VAULT_PATH="/home/$USER/Documents/ObsidianVault"
export JARVIS_DB_PATH="/home/$USER/.local/share/jarvis/jarvis.duckdb"
export JARVIS_DEVICE="cpu"

# XDG Base Directory compliance
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_CACHE_HOME="$HOME/.cache"
```

### Windows Configuration

```cmd
REM Windows-specific settings
set JARVIS_VAULT_PATH=C:\Users\%USERNAME%\Documents\ObsidianVault
set JARVIS_DB_PATH=C:\Users\%USERNAME%\AppData\Local\jarvis\jarvis.duckdb
set JARVIS_DEVICE=cpu

REM Claude Desktop config location
%APPDATA%\Claude\claude_desktop_config.json
```

---

## Docker Configuration

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  jarvis-assistant:
    build: .
    environment:
      - JARVIS_VAULT_PATH=/vault
      - JARVIS_DB_PATH=/data/jarvis.duckdb
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - JARVIS_LOG_LEVEL=INFO
    volumes:
      - ./vault:/vault:ro
      - jarvis-data:/data
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:5.0
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j-data:/data
    ports:
      - "7687:7687"
      - "7474:7474"

volumes:
  jarvis-data:
  neo4j-data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

# Install UV
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN uv sync --frozen

# Set default environment variables
ENV JARVIS_LOG_LEVEL=INFO
ENV JARVIS_DEVICE=cpu

# Expose MCP port (if using TCP)
EXPOSE 8080

# Run MCP server
CMD ["uv", "run", "jarvis", "mcp", "--vault", "/vault"]
```

---

## Troubleshooting Configuration

### Common Issues

#### Environment Variables Not Found
```bash
# Check environment variables
env | grep JARVIS
echo $JARVIS_VAULT_PATH
echo $NEO4J_PASSWORD
```

#### Permission Issues
```bash
# Check file permissions
ls -la ~/.jarvis/
ls -la "$JARVIS_VAULT_PATH"

# Fix permissions if needed
chmod 755 ~/.jarvis/
chmod 644 ~/.jarvis/jarvis.duckdb
```

#### Database Connection Issues
```bash
# Test Neo4j connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p password "RETURN 1"

# Test DuckDB access
python -c "import duckdb; print(duckdb.connect('$JARVIS_DB_PATH').execute('SELECT 1').fetchone())"
```

### Configuration Validation

```python
# Validate configuration
uv run python -c "
from jarvis.utils.config import get_settings
settings = get_settings()
print('Configuration valid!')
print(f'Vault path: {settings.vault_paths}')
print(f'Database: {settings.database_path}')
print(f'Neo4j: {settings.neo4j_uri}')
"
```

## Extension System Configuration

*Added in Phase 0 - Extension Foundation*

### Extension System Settings

#### JARVIS_EXTENSIONS_ENABLED
- **Type**: `boolean`
- **Description**: Enable the extension system for plugin architecture.
- **Default**: `False`
- **Phase**: Phase 0+

#### JARVIS_EXTENSIONS_AUTO_LOAD
- **Type**: `list of strings`
- **Description**: Comma-separated list of extensions to automatically load on startup.
- **Default**: `[]` (empty)
- **Example**: `ai,custom-tools`

#### JARVIS_EXTENSIONS_DIRECTORY
- **Type**: `string`
- **Description**: Directory containing extensions.
- **Default**: `src/jarvis/extensions`

### AI Extension Settings

*Available when AI extension is enabled*

#### JARVIS_AI_EXTENSION_ENABLED
- **Type**: `boolean`
- **Description**: Enable AI extension with LLM capabilities.
- **Default**: `False`
- **Phase**: Phase 1+

#### JARVIS_AI_LLM_PROVIDER
- **Type**: `string`
- **Description**: LLM provider to use.
- **Options**: `ollama`, `huggingface`
- **Default**: `ollama`

#### JARVIS_AI_LLM_MODELS
- **Type**: `list of strings`
- **Description**: Available LLM models for the AI extension.
- **Default**: `["llama2:7b"]`
- **Example**: `llama2:7b,mistral:7b,codellama:13b`

#### JARVIS_AI_MAX_MEMORY_GB
- **Type**: `integer`
- **Description**: Maximum memory usage for AI operations (GB).
- **Default**: `8`
- **Minimum**: `1`

#### JARVIS_AI_TIMEOUT_SECONDS
- **Type**: `integer`
- **Description**: Timeout for AI operations (seconds).
- **Default**: `30`
- **Minimum**: `5`

#### JARVIS_AI_GRAPHRAG_ENABLED
- **Type**: `boolean`
- **Description**: Enable GraphRAG capabilities (Phase 2).
- **Default**: `False`
- **Phase**: Phase 2+

#### JARVIS_AI_WORKFLOWS_ENABLED
- **Type**: `boolean`
- **Description**: Enable workflow orchestration (Phase 3).
- **Default**: `False`
- **Phase**: Phase 3+

### Extension Configuration Example

```bash
# Enable extension system
export JARVIS_EXTENSIONS_ENABLED=true
export JARVIS_EXTENSIONS_AUTO_LOAD=ai

# Configure AI extension
export JARVIS_AI_EXTENSION_ENABLED=true
export JARVIS_AI_LLM_PROVIDER=ollama
export JARVIS_AI_LLM_MODELS="llama2:7b,mistral:7b"
export JARVIS_AI_MAX_MEMORY_GB=16
export JARVIS_AI_TIMEOUT_SECONDS=45
```

### Extension Health Check

```python
# Check extension system status
uv run python -c "
from jarvis.extensions import ExtensionManager
from jarvis.core.container import ServiceContainer
from jarvis.utils.config import get_settings

settings = get_settings()
container = ServiceContainer(settings)
manager = ExtensionManager(settings, container)

# Check if extensions are enabled
print(f'Extensions enabled: {settings.extensions_enabled}')
print(f'AI extension enabled: {settings.ai_extension_enabled}')
print(f'Auto-load: {settings.extensions_auto_load}')
"
```

---

## Next Steps

- [Error Codes](error-codes.md) - Complete error reference
- [API Reference](api-reference.md) - Complete API documentation
- [Performance Tuning](../07-maintenance/performance-tuning.md) - Optimization guide
- [Troubleshooting](../07-maintenance/troubleshooting.md) - Problem resolution