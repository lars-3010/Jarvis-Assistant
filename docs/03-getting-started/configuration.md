# Configuration Guide

*Understanding and customizing Jarvis Assistant settings*

> Note (Updated): In addition to env vars and CLI flags, Jarvis supports YAML-based user configuration via `config/base.yaml` and `config/local.yaml` (merged). See "User YAML Configuration" below.

## Configuration Overview

Jarvis Assistant uses a layered configuration system that allows customization through environment variables, configuration files, and command-line arguments.

## Configuration Priority

Settings are applied in this order (highest to lowest priority):

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Configuration file** (config/.env)
4. **Default values** (lowest priority)

## Core Configuration

### Basic Settings

#### Vault Configuration
```bash
# Primary vault path
JARVIS_VAULT_PATH=/path/to/your/obsidian/vault

# Database storage location
JARVIS_DB_PATH=/path/to/database/directory

# Default: ~/.jarvis/data/
```

#### Embedding Model Settings
```bash
# Model name (from sentence-transformers)
EMBEDDING_MODEL_NAME=paraphrase-MiniLM-L6-v2

# Device for inference (auto, cpu, cuda, mps)
EMBEDDING_DEVICE=auto

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE=32

# Model cache directory
EMBEDDING_CACHE_DIR=/path/to/cache/embeddings
```

#### Search Configuration
```bash
# Default similarity threshold for semantic search
DEFAULT_SIMILARITY_THRESHOLD=0.7

# Maximum results per search
MAX_SEARCH_RESULTS=50

# Search result caching duration (seconds)
SEARCH_CACHE_TTL=300
```

### Performance Settings

#### Processing Configuration
```bash
# Number of worker processes
MAX_WORKERS=4

# Chunk size for document processing
CHUNK_SIZE=256

# Chunk overlap for context preservation
CHUNK_OVERLAP=50

# Maximum file size to process (MB)
MAX_FILE_SIZE_MB=10
```

#### Memory Management
```bash
# Maximum memory usage for caching (MB)
MAX_CACHE_SIZE_MB=1024

# LRU cache sizes
EMBEDDING_CACHE_SIZE=10000
SEARCH_CACHE_SIZE=1000
FILE_CACHE_SIZE=500
```

### Database Configuration

#### DuckDB Settings
```bash
# DuckDB database file path
DUCKDB_PATH=/path/to/jarvis.duckdb

# Memory limit for DuckDB (MB)
DUCKDB_MEMORY_LIMIT=1024

# Number of threads for DuckDB
DUCKDB_THREADS=4
```

#### Neo4j Settings (Optional)
```bash
# Neo4j connection URI
NEO4J_URI=bolt://localhost:7687

# Neo4j authentication
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Connection pool settings
NEO4J_MAX_CONNECTIONS=10
NEO4J_CONNECTION_TIMEOUT=30
```

## Environment File Setup

### Creating Configuration File

#### Step 1: Copy Template
```bash
# Copy example configuration
cp config/.env.example config/.env

# Edit configuration
nano config/.env
```

#### Step 2: Example .env File
```bash
# Jarvis Assistant Configuration

# Core Settings
JARVIS_VAULT_PATH=/Users/username/Documents/ObsidianVault
JARVIS_DB_PATH=/Users/username/.jarvis/data

# Embedding Settings
EMBEDDING_MODEL_NAME=paraphrase-MiniLM-L6-v2
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=32

# Search Settings
DEFAULT_SIMILARITY_THRESHOLD=0.7
MAX_SEARCH_RESULTS=50
SEARCH_CACHE_TTL=300

# Performance Settings
MAX_WORKERS=4
CHUNK_SIZE=256
CHUNK_OVERLAP=50
MAX_FILE_SIZE_MB=10

# Memory Settings
MAX_CACHE_SIZE_MB=1024
EMBEDDING_CACHE_SIZE=10000
SEARCH_CACHE_SIZE=1000

# Database Settings
DUCKDB_MEMORY_LIMIT=1024
DUCKDB_THREADS=4

# Neo4j Settings (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Logging
LOG_LEVEL=INFO
LOG_FILE=/Users/username/.jarvis/logs/jarvis.log
```

## Advanced Configuration

### Embedding Model Selection

#### Available Models
```python
# Fast, small models (recommended for most users)
EMBEDDING_MODEL_NAME=paraphrase-MiniLM-L6-v2         # 384 dim, 90MB
EMBEDDING_MODEL_NAME=paraphrase-MiniLM-L12-v2        # 384 dim, 120MB

# Higher quality models (slower, more memory)
EMBEDDING_MODEL_NAME=paraphrase-mpnet-base-v2        # 768 dim, 420MB
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2                # 384 dim, 90MB

# Multilingual models
EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2  # 384 dim, 470MB
```

#### Model Comparison
| Model | Dimensions | Size | Speed | Quality | Use Case |
|-------|------------|------|--------|---------|----------|
| paraphrase-MiniLM-L6-v2 | 384 | 90MB | Fast | Good | General purpose |
| paraphrase-mpnet-base-v2 | 768 | 420MB | Slow | Excellent | High quality |
| all-MiniLM-L6-v2 | 384 | 90MB | Fast | Good | General purpose |
| multilingual-MiniLM-L12-v2 | 384 | 470MB | Medium | Good | Multilingual |

### Device Configuration

#### CPU Configuration
```bash
# Force CPU usage
EMBEDDING_DEVICE=cpu

# CPU-specific optimizations
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

#### GPU Configuration
```bash
# NVIDIA GPU (CUDA)
EMBEDDING_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# Apple Silicon (MPS)
EMBEDDING_DEVICE=mps

# Check GPU availability
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
```

### Chunking Strategy

#### Chunk Size Optimization
```bash
# For short documents (notes, articles)
CHUNK_SIZE=256
CHUNK_OVERLAP=50

# For long documents (books, papers)
CHUNK_SIZE=512
CHUNK_OVERLAP=100

# For very long documents
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
```

#### Chunking Behavior
```bash
# Chunking method (sentence, paragraph, fixed)
CHUNKING_METHOD=sentence

# Minimum chunk size (characters)
MIN_CHUNK_SIZE=50

# Maximum chunk size (characters)
MAX_CHUNK_SIZE=2000
```

## Command-Line Configuration

### Override Environment Variables

#### Basic Usage
```bash
# Override vault path
uv run jarvis --vault /different/path mcp

# Override similarity threshold
uv run jarvis search --similarity-threshold 0.8 --query "test"

# Override embedding model
uv run jarvis --embedding-model all-MiniLM-L6-v2 index
```

#### Advanced Options
```bash
# Multiple overrides
uv run jarvis \
  --vault /path/to/vault \
  --db-path /path/to/db \
  --embedding-model paraphrase-mpnet-base-v2 \
  --batch-size 16 \
  --max-workers 2 \
  mcp --watch
```

### Verbose and Debug Modes

#### Logging Levels
```bash
# Quiet mode (errors only)
uv run jarvis --quiet mcp

# Normal mode (default)
uv run jarvis mcp

# Verbose mode (info + debug)
uv run jarvis --verbose mcp

# Debug mode (all logs)
uv run jarvis --debug mcp
```

## MCP Server Configuration

### Claude Desktop Integration

#### Basic MCP Configuration
```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio",
      "args": ["/path/to/your/obsidian/vault"],
      "type": "stdio",
      "cwd": "/path/to/jarvis-assistant"
    }
  }
}
```

#### Advanced MCP Configuration
```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio",
      "args": [
        "/path/to/your/obsidian/vault",
        "--similarity-threshold", "0.8",
        "--max-results", "20",
        "--embedding-model", "paraphrase-mpnet-base-v2"
      ],
      "type": "stdio",
      "cwd": "/path/to/jarvis-assistant",
      "env": {
        "PYTHONUNBUFFERED": "1",
        "LOG_LEVEL": "INFO",
        "EMBEDDING_DEVICE": "cuda",
        "MAX_WORKERS": "4"
      }
    }
  }
}
```

### MCP Tool Parameters

#### Default Tool Configuration
```bash
# Semantic search defaults
SEMANTIC_SEARCH_THRESHOLD=0.7
SEMANTIC_SEARCH_LIMIT=10

# Vault search defaults
VAULT_SEARCH_LIMIT=20
VAULT_SEARCH_CONTENT=true

# Graph search defaults
GRAPH_SEARCH_DEPTH=2
GRAPH_SEARCH_MAX_NODES=50

# File read defaults
READ_NOTE_INCLUDE_METADATA=true
READ_NOTE_MAX_SIZE=1048576  # 1MB
```

## Monitoring and Logging

### Log Configuration

#### Log Levels
```bash
# Log level options
LOG_LEVEL=DEBUG    # All messages
LOG_LEVEL=INFO     # Info and above (default)
LOG_LEVEL=WARNING  # Warnings and errors
LOG_LEVEL=ERROR    # Errors only
LOG_LEVEL=CRITICAL # Critical errors only
```

#### Log Output
```bash
# Log to file
LOG_FILE=/path/to/logs/jarvis.log

# Log to console (default)
LOG_FILE=

# Log to both file and console
LOG_FILE=/path/to/logs/jarvis.log
LOG_CONSOLE=true
```

#### Log Format
```bash
# Log format options
LOG_FORMAT=detailed  # Timestamp, level, module, message
LOG_FORMAT=simple    # Level and message only
LOG_FORMAT=json      # JSON structured logging
```

### Performance Monitoring

#### Metrics Collection
```bash
# Enable metrics collection
ENABLE_METRICS=true

# Metrics output file
METRICS_FILE=/path/to/metrics/jarvis_metrics.json

# Metrics collection interval (seconds)
METRICS_INTERVAL=60
```

#### Health Checks
```bash
# Enable health check endpoint
ENABLE_HEALTH_CHECK=true

# Health check port
HEALTH_CHECK_PORT=8080

# Health check path
HEALTH_CHECK_PATH=/health
```

## Security Configuration

### Access Control

#### File System Access
```bash
# Allowed vault paths (comma-separated)
ALLOWED_VAULT_PATHS=/path/to/vault1,/path/to/vault2

# Restricted file patterns
RESTRICTED_PATTERNS=*.key,*.secret,*.env

# Maximum file size for reading
MAX_READ_FILE_SIZE=10485760  # 10MB
```

#### Network Security
```bash
# Neo4j SSL settings
NEO4J_ENCRYPTED=true
NEO4J_TRUST_STORE=/path/to/trust/store

# API rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60  # seconds
```

## Validation and Testing

### Configuration Validation

#### Check Configuration
```bash
# Validate current configuration
uv run jarvis --validate-config

# Check specific setting
uv run jarvis --check-setting EMBEDDING_MODEL_NAME

# Test configuration with dry run
uv run jarvis --dry-run mcp
```

#### Configuration Diagnostics
```bash
# Show all configuration values
uv run jarvis --show-config

# Show configuration sources
uv run jarvis --show-config-sources

# Export configuration to file
uv run jarvis --export-config > config_backup.env
```

### Performance Testing

#### Benchmark Configuration
```bash
# Test embedding performance
uv run jarvis --benchmark-embeddings

# Test search performance
uv run jarvis --benchmark-search --vault /path/to/vault

# Test different configurations
uv run jarvis --benchmark-config \
  --embedding-model paraphrase-MiniLM-L6-v2 \
  --batch-size 32 \
  --max-workers 4
```

## Troubleshooting Configuration

### Common Issues

#### Environment Variable Not Working
```bash
# Check if variable is set
echo $EMBEDDING_MODEL_NAME

# Check if variable is being used
uv run jarvis --show-config | grep EMBEDDING_MODEL_NAME
```

#### Model Download Issues
```bash
# Pre-download models
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"

# Check model cache
ls -la ~/.cache/torch/sentence_transformers/
```

#### Database Connection Issues
```bash
# Test DuckDB connection
uv run python -c "import duckdb; print(duckdb.connect(':memory:').execute('SELECT 1').fetchone())"

# Test Neo4j connection
uv run python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); print(driver.verify_connectivity())"
```

### Reset Configuration

#### Reset to Defaults
```bash
# Remove configuration file
rm config/.env

# Clear cache
rm -rf ~/.cache/jarvis/

# Reinitialize
uv run jarvis --init-config
```

## For More Information

- **[First Queries](first-queries.md)** - Test your configuration
- **[Performance Tuning](../07-maintenance/performance-tuning.md)** - Optimize settings
- **[Troubleshooting](../07-maintenance/troubleshooting.md)** - Common issues
- **[API Reference](../06-reference/configuration-reference.md)** - All configuration options
