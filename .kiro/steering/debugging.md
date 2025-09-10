# Debugging & Troubleshooting Guide

## Quick Problem-Solving Workflows

### MCP Tool Issues

#### "Tool not responding" or "Tool times out"
```bash
# 1. Check if MCP server is running
ps aux | grep "jarvis mcp"

# 2. Check logs for errors
tail -f ~/.jarvis/logs/mcp.log

# 3. Test tool directly
uv run python -c "
from jarvis.mcp.server import MCPServer
server = MCPServer()
result = server.search_semantic('test query')
print(result)
"

# 4. Check service health
uv run jarvis health-check
```

#### "Vault not found" errors
```bash
# 1. Verify vault path exists
ls -la /path/to/vault

# 2. Check vault permissions
ls -la /path/to/vault/*.md

# 3. Test vault discovery
uv run python -c "
from jarvis.services.vault import VaultService
vault = VaultService()
notes = vault.discover_notes('/path/to/vault')
print(f'Found {len(notes)} notes')
"
```

### Search Issues

#### "No search results" when results expected
```bash
# 1. Check if vault is indexed
uv run jarvis stats --vault /path/to/vault

# 2. Re-index vault
uv run jarvis index --vault /path/to/vault --force

# 3. Test different search types
uv run jarvis search --vault /path/to/vault --query "test" --type keyword
uv run jarvis search --vault /path/to/vault --query "test" --type semantic

# 4. Check database contents
uv run python -c "
import duckdb
conn = duckdb.connect('data/jarvis.db')
result = conn.execute('SELECT COUNT(*) FROM documents').fetchone()
print(f'Documents in DB: {result[0]}')
"
```

#### "Search too slow" performance issues
```bash
# 1. Check memory usage
uv run python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f}MB')
"

# 2. Check database indexes
uv run python -c "
import duckdb
conn = duckdb.connect('data/jarvis.db')
indexes = conn.execute('SHOW TABLES').fetchall()
print('Database tables:', indexes)
"

# 3. Enable performance logging
export JARVIS_LOG_LEVEL=DEBUG
uv run jarvis search --vault /path/to/vault --query "test"
```

### Database Issues

#### "Database connection failed"
```bash
# 1. Check if database file exists
ls -la data/jarvis.db

# 2. Check database permissions
ls -la data/

# 3. Test database connection
uv run python -c "
import duckdb
try:
    conn = duckdb.connect('data/jarvis.db')
    print('Database connection successful')
    conn.close()
except Exception as e:
    print(f'Database error: {e}')
"

# 4. Recreate database if corrupted
mv data/jarvis.db data/jarvis.db.backup
uv run jarvis index --vault /path/to/vault
```

#### "Embedding generation fails"
```bash
# 1. Check sentence-transformers installation
uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model loaded successfully')
"

# 2. Test embedding generation
uv run python -c "
from jarvis.services.vector import VectorService
vector_service = VectorService()
embedding = vector_service.encode('test text')
print(f'Embedding shape: {len(embedding)}')
"

# 3. Check available memory
free -h
```

### Service Issues

#### "Service unavailable" errors
```bash
# 1. Check service registration
uv run python -c "
from jarvis.core.container import container
services = container.list_services()
print('Registered services:', services)
"

# 2. Test service initialization
uv run python -c "
from jarvis.core.container import get_service
try:
    service = get_service('vector_service')
    print('Vector service available')
except Exception as e:
    print(f'Vector service error: {e}')
"

# 3. Check service dependencies
uv run python -c "
import duckdb
import sentence_transformers
print('Dependencies available')
"
```

### Common Error Messages & Solutions

#### `ModuleNotFoundError: No module named 'jarvis'`
```bash
# Solution: Install in development mode
uv sync
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### `FileNotFoundError: [Errno 2] No such file or directory: 'data/jarvis.db'`
```bash
# Solution: Create data directory and initialize database
mkdir -p data
uv run jarvis index --vault /path/to/vault
```

#### `RuntimeError: CUDA out of memory`
```bash
# Solution: Use CPU-only mode
export JARVIS_DEVICE=cpu
uv run jarvis index --vault /path/to/vault
```

#### `PermissionError: [Errno 13] Permission denied`
```bash
# Solution: Fix permissions
chmod -R 755 data/
chmod -R 644 data/*.db
```

### Performance Debugging

#### Memory Usage Investigation
```python
# Add to problematic code
import psutil
import gc

def log_memory_usage(label: str):
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"{label}: {memory_mb:.1f}MB")

# Usage
log_memory_usage("Before operation")
# ... your code ...
log_memory_usage("After operation")
gc.collect()
log_memory_usage("After garbage collection")
```

#### Response Time Investigation
```python
# Add timing to MCP tools
import time

def timed_operation(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

# Usage
@timed_operation
def search_semantic(query: str):
    # ... implementation ...
```

### Log Analysis

#### Enable Debug Logging
```bash
# Set environment variable
export JARVIS_LOG_LEVEL=DEBUG

# Or in code
import logging
logging.getLogger('jarvis').setLevel(logging.DEBUG)
```

#### Key Log Patterns to Look For
- `"Database connection failed"` → Database issues
- `"Service unavailable"` → Service registration problems
- `"Memory usage high"` → Performance issues
- `"Timeout exceeded"` → Performance or hanging operations
- `"Validation error"` → Input parameter issues

### Testing & Validation

#### Quick Health Check
```bash
# Run comprehensive health check
uv run python -c "
from jarvis.monitoring.health import system_health_check
health = system_health_check()
for component, status in health.items():
    print(f'{component}: {status}')
"
```

#### Validate Configuration
```bash
# Check configuration
uv run python -c "
from jarvis.utils.config import get_config
config = get_config()
print('Configuration loaded successfully')
print(f'Database path: {config.database_path}')
print(f'Log level: {config.log_level}')
"
```

### Emergency Recovery

#### Reset Everything
```bash
# 1. Stop all processes
pkill -f "jarvis mcp"

# 2. Clear caches and databases
rm -rf data/jarvis.db
rm -rf ~/.jarvis/cache/

# 3. Reinstall dependencies
uv sync --reinstall

# 4. Re-index vault
uv run jarvis index --vault /path/to/vault

# 5. Test basic functionality
uv run jarvis search --vault /path/to/vault --query "test"
```

#### Backup & Restore
```bash
# Backup current state
tar -czf jarvis-backup-$(date +%Y%m%d).tar.gz data/ ~/.jarvis/

# Restore from backup
tar -xzf jarvis-backup-20241201.tar.gz
```

## Debugging Checklist

When something breaks, check in this order:

1. **Environment**: Are dependencies installed? (`uv sync`)
2. **Permissions**: Can we read/write files? (`ls -la data/`)
3. **Services**: Are services registered? (Check container)
4. **Database**: Is database accessible? (Test connection)
5. **Memory**: Are we running out of memory? (`free -h`)
6. **Logs**: What do the logs say? (`tail -f logs/`)
7. **Configuration**: Is config valid? (Test config loading)

## Getting Help

- **Check logs first**: `~/.jarvis/logs/` or console output
- **Test components individually**: Use Python REPL to test services
- **Enable debug logging**: `export JARVIS_LOG_LEVEL=DEBUG`
- **Check system resources**: Memory, disk space, permissions
- **Try minimal reproduction**: Isolate the problem to smallest case