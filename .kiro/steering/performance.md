# Performance & Monitoring Standards

## Performance Philosophy
- **Local-First Performance**: Optimize for local machine capabilities, not cloud scale
- **Graceful Degradation**: Maintain core functionality even under resource constraints
- **Predictable Performance**: Consistent response times across different vault sizes
- **Resource Efficiency**: Minimize memory and CPU usage for background operations

## Performance Targets

### Response Time Requirements
```python
# MCP Tool Response Times
PERFORMANCE_TARGETS = {
    "health_status": 0.5,      # 500ms
    "list_vaults": 1.0,        # 1 second
    "search_vault": 3.0,       # 3 seconds
    "search_semantic": 5.0,    # 5 seconds
    "search_combined": 8.0,    # 8 seconds
    "vault_analytics": 15.0,   # 15 seconds
}
```

### Scalability Targets
- **Small Vaults** (< 1,000 notes): Sub-second search responses
- **Medium Vaults** (1,000 - 10,000 notes): < 5 second search responses
- **Large Vaults** (> 10,000 notes): < 15 second search responses with progress indicators

### Resource Usage Limits
- **Memory**: < 1GB for typical operations, < 2GB for large vault indexing
- **CPU**: < 80% sustained usage during indexing
- **Disk I/O**: Minimize file system operations through caching
- **Database**: Connection pooling with max 10 concurrent connections

## Caching Strategy

### Multi-Layer Caching
```python
# Cache hierarchy (fastest to slowest)
1. In-memory LRU cache (query results)
2. DuckDB result cache (computed embeddings)
3. File system cache (parsed documents)
4. Source files (vault notes)
```

### Cache Configuration
```python
CACHE_SETTINGS = {
    "query_cache_size": 1000,           # Number of cached queries
    "embedding_cache_ttl": 3600,        # 1 hour TTL for embeddings
    "document_cache_size": 5000,        # Parsed documents in memory
    "result_cache_ttl": 300,            # 5 minutes for search results
}
```

### Cache Invalidation
- **File Changes**: Invalidate document and embedding caches
- **Vault Structure Changes**: Clear all caches for affected vault
- **Time-based**: TTL expiration for computed results
- **Manual**: Admin tools for cache clearing

## Performance Monitoring

### Key Metrics Collection
```python
# Response time metrics
response_time_histogram = Histogram(
    'jarvis_response_time_seconds',
    'Response time for MCP tools',
    ['tool_name', 'vault_id']
)

# Resource usage metrics
memory_usage_gauge = Gauge(
    'jarvis_memory_usage_bytes',
    'Current memory usage'
)

# Cache performance metrics
cache_hit_ratio = Gauge(
    'jarvis_cache_hit_ratio',
    'Cache hit ratio by cache type',
    ['cache_type']
)
```

### Performance Logging
```python
import time
from jarvis.monitoring.metrics import performance_timer

@performance_timer('semantic_search')
def semantic_search(query: str, limit: int = 10):
    start_time = time.time()
    try:
        results = perform_search(query, limit)
        return results
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Semantic search completed", extra={
            "query_length": len(query),
            "result_count": len(results),
            "elapsed_seconds": elapsed,
            "performance_target": PERFORMANCE_TARGETS["search_semantic"]
        })
```

## Database Performance

### DuckDB Optimization
```sql
-- Optimize vector similarity queries
CREATE INDEX idx_embeddings_vector ON embeddings USING HNSW(embedding);

-- Optimize text search queries
CREATE INDEX idx_documents_content ON documents USING FTS(content);

-- Optimize metadata queries
CREATE INDEX idx_documents_path ON documents(vault_id, file_path);
CREATE INDEX idx_documents_modified ON documents(modified_time);
```

### Query Optimization Patterns
```python
# Use prepared statements for repeated queries
SEARCH_QUERY = """
    SELECT d.file_path, d.content, v.similarity_score
    FROM documents d
    JOIN (
        SELECT file_path, 
               array_cosine_similarity(embedding, ?) as similarity_score
        FROM embeddings 
        WHERE similarity_score > ?
        ORDER BY similarity_score DESC
        LIMIT ?
    ) v ON d.file_path = v.file_path
"""

# Batch operations for better performance
def batch_insert_embeddings(embeddings: List[Embedding]):
    with connection_pool.get_connection() as conn:
        conn.executemany(INSERT_EMBEDDING_QUERY, embeddings)
```

### Connection Management
```python
# Connection pooling configuration
CONNECTION_POOL_CONFIG = {
    "min_connections": 2,
    "max_connections": 10,
    "connection_timeout": 30,
    "idle_timeout": 300,
    "retry_attempts": 3
}
```

## Memory Management

### Memory Usage Patterns
```python
# Use generators for large result sets
def stream_search_results(query: str) -> Iterator[SearchResult]:
    for batch in get_results_in_batches(query, batch_size=100):
        for result in batch:
            yield result

# Implement memory-aware processing
def process_large_vault(vault_path: str):
    memory_monitor = MemoryMonitor(threshold_mb=1500)
    
    for note_batch in get_notes_in_batches(vault_path):
        if memory_monitor.should_pause():
            gc.collect()  # Force garbage collection
            time.sleep(0.1)  # Brief pause
        
        process_note_batch(note_batch)
```

### Memory Monitoring
```python
import psutil
import gc

class MemoryMonitor:
    def __init__(self, threshold_mb: int = 1000):
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.process = psutil.Process()
    
    def current_usage(self) -> int:
        return self.process.memory_info().rss
    
    def should_pause(self) -> bool:
        return self.current_usage() > self.threshold_bytes
    
    def force_cleanup(self):
        gc.collect()
        # Clear internal caches if memory pressure is high
        if self.current_usage() > self.threshold_bytes * 1.5:
            clear_internal_caches()
```

## Performance Testing

### Load Testing
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def load_test_search(concurrent_requests: int = 10):
    """Test search performance under concurrent load"""
    queries = generate_test_queries(concurrent_requests)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [
            executor.submit(search_service.search, query)
            for query in queries
        ]
        
        results = [future.result() for future in futures]
    
    elapsed = time.time() - start_time
    
    assert elapsed < 30.0  # All requests should complete within 30 seconds
    assert all(len(r.results) > 0 for r in results)  # All should return results
```

### Performance Regression Testing
```python
def test_search_performance_regression():
    """Ensure search performance doesn't degrade over time"""
    baseline_times = load_performance_baseline()
    
    current_times = measure_current_performance()
    
    for operation, current_time in current_times.items():
        baseline_time = baseline_times.get(operation, float('inf'))
        regression_threshold = baseline_time * 1.2  # 20% regression tolerance
        
        assert current_time < regression_threshold, (
            f"Performance regression detected for {operation}: "
            f"{current_time:.2f}s > {regression_threshold:.2f}s"
        )
```

## Optimization Strategies

### Lazy Loading
```python
class VaultIndex:
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self._embeddings = None  # Lazy-loaded
        self._graph_data = None  # Lazy-loaded
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = load_embeddings(self.vault_path)
        return self._embeddings
```

### Batch Processing
```python
def index_vault_efficiently(vault_path: str):
    """Index vault using batch processing for optimal performance"""
    notes = discover_notes(vault_path)
    
    # Process in batches to manage memory
    for batch in chunked(notes, batch_size=50):
        # Parse documents in batch
        documents = [parse_document(note) for note in batch]
        
        # Generate embeddings in batch (more efficient)
        embeddings = embedding_service.encode_batch([d.content for d in documents])
        
        # Store in database as batch
        database.store_batch(documents, embeddings)
```

### Incremental Updates
```python
def incremental_vault_update(vault_path: str):
    """Update only changed files for better performance"""
    last_update = get_last_update_time(vault_path)
    changed_files = find_changed_files(vault_path, since=last_update)
    
    if not changed_files:
        return  # No updates needed
    
    # Only process changed files
    for file_path in changed_files:
        update_file_index(file_path)
    
    update_last_update_time(vault_path)
```

## Monitoring and Alerting

### Health Checks
```python
def system_health_check() -> Dict[str, Any]:
    """Comprehensive system health check"""
    return {
        "database_connection": check_database_health(),
        "memory_usage": get_memory_usage_percent(),
        "cache_hit_ratio": get_cache_hit_ratio(),
        "average_response_time": get_average_response_time(),
        "error_rate": get_error_rate_last_hour(),
        "disk_space": get_available_disk_space()
    }
```

### Performance Alerts
```python
# Alert thresholds
ALERT_THRESHOLDS = {
    "response_time_p95": 10.0,     # 95th percentile response time
    "memory_usage_percent": 85,     # Memory usage percentage
    "error_rate_percent": 5,        # Error rate percentage
    "cache_hit_ratio": 0.7,        # Minimum cache hit ratio
}

def check_performance_alerts():
    """Check if any performance metrics exceed thresholds"""
    metrics = collect_current_metrics()
    
    alerts = []
    for metric, threshold in ALERT_THRESHOLDS.items():
        if metrics[metric] > threshold:
            alerts.append(f"{metric} exceeded threshold: {metrics[metric]} > {threshold}")
    
    return alerts
```