# Common Patterns & Solutions

## Recurring Solutions for Typical Problems

### Error Handling Patterns

#### MCP Tool Error Response
```python
# Standard error response pattern
def handle_mcp_error(error_code: str, message: str, suggestions: List[str] = None):
    return {
        "success": False,
        "error": {
            "code": error_code,
            "message": message,
            "suggestions": suggestions or []
        }
    }

# Usage
if not vault_exists(vault_path):
    return handle_mcp_error(
        "VAULT_NOT_FOUND",
        f"Vault not found at {vault_path}",
        ["Check vault path", "Verify permissions"]
    )
```

#### Service Exception Handling
```python
# Graceful service degradation
try:
    results = vector_service.search(query)
except DatabaseConnectionError:
    logger.warning("Vector search unavailable, falling back to keyword search")
    results = keyword_service.search(query)
except Exception as e:
    logger.error(f"Search failed: {e}")
    return handle_mcp_error("SEARCH_ERROR", "Search operation failed")
```

### Caching Patterns

#### Simple Result Caching
```python
from functools import lru_cache
from typing import Optional

@lru_cache(maxsize=1000)
def get_cached_embeddings(file_path: str, content_hash: str):
    """Cache embeddings with content hash for invalidation"""
    return embedding_service.encode(content)

# Usage with cache invalidation
def update_file_embeddings(file_path: str):
    content = read_file(file_path)
    content_hash = hashlib.md5(content.encode()).hexdigest()
    
    # Clear old cache entry
    get_cached_embeddings.cache_clear()
    
    # Generate new embeddings
    return get_cached_embeddings(file_path, content_hash)
```

#### Time-Based Caching
```python
import time
from typing import Dict, Tuple, Any

class TTLCache:
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())

# Usage
search_cache = TTLCache(ttl_seconds=300)  # 5 minute cache

def cached_search(query: str):
    cached_result = search_cache.get(query)
    if cached_result:
        return cached_result
    
    result = perform_search(query)
    search_cache.set(query, result)
    return result
```

### Service Integration Patterns

#### Dependency Injection Usage
```python
# Service registration (in container setup)
container.register("vector_service", VectorService)
container.register("graph_service", GraphService)

# Service access in MCP tools
def search_semantic(query: str, limit: int = 10):
    vector_service = get_service("vector_service")
    
    try:
        results = vector_service.search(query, limit)
        return success_response(results)
    except ServiceUnavailableError:
        return handle_mcp_error("SERVICE_UNAVAILABLE", "Vector search temporarily unavailable")
```

#### Service Health Checking
```python
def check_service_health(service_name: str) -> bool:
    """Check if a service is healthy and available"""
    try:
        service = get_service(service_name)
        return service.health_check()
    except Exception:
        return False

# Usage in MCP tools
def search_combined(query: str):
    available_services = []
    
    if check_service_health("vector_service"):
        available_services.append("semantic")
    if check_service_health("keyword_service"):
        available_services.append("keyword")
    
    if not available_services:
        return handle_mcp_error("NO_SERVICES", "No search services available")
    
    # Proceed with available services
```

### Data Processing Patterns

#### Batch Processing
```python
def process_vault_in_batches(vault_path: str, batch_size: int = 50):
    """Process large vaults in manageable batches"""
    notes = discover_notes(vault_path)
    
    for i in range(0, len(notes), batch_size):
        batch = notes[i:i + batch_size]
        
        # Process batch
        documents = [parse_document(note) for note in batch]
        embeddings = embedding_service.encode_batch([d.content for d in documents])
        
        # Store batch
        database.store_batch(documents, embeddings)
        
        # Memory management
        if i % (batch_size * 10) == 0:  # Every 10 batches
            gc.collect()
```

#### Progress Tracking
```python
def index_vault_with_progress(vault_path: str):
    """Index vault with progress reporting for long operations"""
    notes = discover_notes(vault_path)
    total_notes = len(notes)
    
    for i, note in enumerate(notes):
        # Process note
        process_note(note)
        
        # Report progress every 10%
        if i % (total_notes // 10) == 0:
            progress = (i / total_notes) * 100
            logger.info(f"Indexing progress: {progress:.1f}% ({i}/{total_notes})")
    
    logger.info("Indexing complete")
```

### Validation Patterns

#### Parameter Validation
```python
from pydantic import BaseModel, validator

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    vault: Optional[str] = None
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('limit')
    def limit_reasonable(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Limit must be between 1 and 100')
        return v

# Usage in MCP tools
def search_semantic_validated(query: str, limit: int = 10, vault: str = None):
    try:
        request = SearchRequest(query=query, limit=limit, vault=vault)
    except ValidationError as e:
        return handle_mcp_error("VALIDATION_ERROR", str(e))
    
    # Proceed with validated parameters
    return perform_search(request.query, request.limit, request.vault)
```

### Performance Patterns

#### Memory-Aware Processing
```python
import psutil

def memory_aware_processing(items: List[Any], memory_threshold_mb: int = 1000):
    """Process items with memory monitoring"""
    process = psutil.Process()
    
    for item in items:
        # Check memory usage
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > memory_threshold_mb:
            logger.warning(f"Memory usage high: {memory_mb:.1f}MB, pausing")
            gc.collect()
            time.sleep(0.1)
        
        # Process item
        process_item(item)
```

#### Timeout Handling
```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds: int):
    """Context manager for operation timeouts"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Usage
def search_with_timeout(query: str):
    try:
        with timeout(15):  # 15 second timeout
            return perform_search(query)
    except TimeoutError:
        return handle_mcp_error("TIMEOUT", "Search operation timed out")
```

### Testing Patterns

#### Test Data Creation
```python
def create_test_vault(notes: List[Tuple[str, str]]) -> str:
    """Create temporary vault for testing"""
    vault_path = tempfile.mkdtemp()
    
    for filename, content in notes:
        file_path = os.path.join(vault_path, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    return vault_path

# Usage
def test_search():
    vault = create_test_vault([
        ("ai.md", "# AI\nArtificial intelligence content"),
        ("python.md", "# Python\nPython programming content")
    ])
    
    results = search_service.search("artificial intelligence")
    assert len(results) > 0
    assert "ai.md" in results[0].path
```

#### Mock Service Dependencies
```python
from unittest.mock import Mock, patch

def test_mcp_tool_with_service_failure():
    """Test MCP tool handles service failures gracefully"""
    
    with patch('jarvis.core.container.get_service') as mock_get_service:
        # Mock service failure
        mock_service = Mock()
        mock_service.search.side_effect = DatabaseConnectionError("DB down")
        mock_get_service.return_value = mock_service
        
        # Test tool handles failure
        result = search_semantic("test query")
        
        assert result["success"] is False
        assert "SERVICE_UNAVAILABLE" in result["error"]["code"]
```

## When to Use These Patterns

- **Error Handling**: Every MCP tool and service method
- **Caching**: Expensive operations (embeddings, search results)
- **Batch Processing**: Large vault operations (>1000 files)
- **Validation**: All user inputs and MCP parameters
- **Timeouts**: Operations that might hang (database, file I/O)
- **Memory Monitoring**: Large data processing operations