# Performance Tuning

Comprehensive guide for optimizing Jarvis Assistant performance across all components. This guide covers database optimization, search performance, memory management, and system-level tuning.

## Quick Navigation

- [Performance Overview](#performance-overview)
- [Vector Search Optimization](#vector-search-optimization)
- [Database Performance](#database-performance)
- [Graph Search Tuning](#graph-search-tuning)
- [Memory Management](#memory-management)
- [System-Level Optimization](#system-level-optimization)
- [Monitoring and Profiling](#monitoring-and-profiling)

---

## Performance Overview

### Performance Metrics

Key performance indicators for Jarvis Assistant:

| Metric | Target | Good | Needs Improvement |
|--------|--------|------|-------------------|
| **Search Response Time** | < 500ms | < 1s | > 2s |
| **Indexing Speed** | > 100 files/min | > 50 files/min | < 20 files/min |
| **Memory Usage** | < 1GB | < 2GB | > 4GB |
| **Database Size** | Efficient | Reasonable | Bloated |
| **Graph Query Time** | < 200ms | < 500ms | > 1s |

### Performance Bottlenecks

Common performance bottlenecks and their symptoms:

1. **Model Loading**: Slow first search (~5-30 seconds)
2. **Vector Encoding**: High CPU usage during search
3. **Database I/O**: Slow searches with disk activity
4. **Memory Pressure**: System slowdown, swapping
5. **Graph Queries**: Slow relationship traversal
6. **Network Latency**: Neo4j connection delays

---

## Vector Search Optimization

### Model Selection

Choose the optimal model for your use case:

```python
# Performance vs Accuracy trade-offs
MODELS = {
    "fastest": "sentence-transformers/all-MiniLM-L6-v2",      # 384 dims, ~80MB
    "balanced": "sentence-transformers/all-mpnet-base-v2",     # 768 dims, ~420MB
    "accurate": "sentence-transformers/all-distilroberta-v1",  # 768 dims, ~290MB
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

# Configuration for different scenarios
export JARVIS_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"  # Default: Fast
```

### Encoding Optimization

#### Batch Processing
```bash
# Optimize batch size based on available memory
export JARVIS_BATCH_SIZE=32   # High-memory systems (16GB+)
export JARVIS_BATCH_SIZE=16   # Medium-memory systems (8-16GB)
export JARVIS_BATCH_SIZE=8    # Low-memory systems (<8GB)
export JARVIS_BATCH_SIZE=4    # Minimal systems (<4GB)
```

#### Device Selection
```bash
# GPU acceleration (if available)
export JARVIS_DEVICE=cuda     # NVIDIA GPUs
export JARVIS_DEVICE=mps      # Apple Silicon (M1/M2)
export JARVIS_DEVICE=cpu      # CPU-only (default)

# Verify GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
"
```

#### Caching Strategy
```python
# Enhanced caching configuration
CACHE_CONFIG = {
    "query_cache": {
        "enabled": True,
        "max_size": 256,        # Number of cached queries
        "ttl": 300             # Cache lifetime (seconds)
    },
    "encoding_cache": {
        "enabled": True,
        "max_size": 128,        # Number of cached encodings
        "ttl": 3600            # Longer lifetime for encodings
    },
    "model_cache": {
        "preload": True,        # Preload model at startup
        "keep_alive": True      # Keep model in memory
    }
}
```

### Search Optimization

#### Similarity Thresholds
```bash
# Use appropriate thresholds to reduce result sets
export JARVIS_SIMILARITY_THRESHOLD=0.7   # Focused results
export JARVIS_SIMILARITY_THRESHOLD=0.5   # Balanced results
export JARVIS_SIMILARITY_THRESHOLD=0.0   # All results (default)

# Dynamic threshold adjustment
python -c "
# Higher thresholds for better performance
import os
if 'precision' in os.getenv('SEARCH_MODE', ''):
    os.environ['JARVIS_SIMILARITY_THRESHOLD'] = '0.8'
elif 'balanced' in os.getenv('SEARCH_MODE', ''):
    os.environ['JARVIS_SIMILARITY_THRESHOLD'] = '0.6'
"
```

#### Result Limiting
```bash
# Limit result counts for better performance
export JARVIS_MAX_SEARCH_RESULTS=20      # API maximum
export JARVIS_DEFAULT_SEARCH_RESULTS=10  # Default for tools

# Progressive loading for large result sets
export JARVIS_ENABLE_PAGINATION=true
```

### Pre-computation Strategies

#### Model Warmup
```python
# Warm up model at startup
import time
from sentence_transformers import SentenceTransformer

def warmup_model(model_name: str) -> None:
    """Warm up the model to reduce first-query latency."""
    print(f"Warming up model: {model_name}")
    start_time = time.time()
    
    model = SentenceTransformer(model_name)
    
    # Warm up with sample queries
    warmup_queries = [
        "machine learning artificial intelligence",
        "project management productivity",
        "data science analytics",
    ]
    
    for query in warmup_queries:
        model.encode(query)
    
    warmup_time = time.time() - start_time
    print(f"Model warmup completed in {warmup_time:.2f}s")

# Usage in startup script
warmup_model(os.getenv('JARVIS_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'))
```

#### Background Processing
```python
# Background indexing strategy
import asyncio
from pathlib import Path

async def background_indexer(vault_path: Path, batch_size: int = 50):
    """Index files in background with rate limiting."""
    
    files = list(vault_path.glob("**/*.md"))
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        
        # Process batch
        await process_batch(batch)
        
        # Rate limiting to avoid overwhelming system
        await asyncio.sleep(0.1)
        
        if i % 500 == 0:
            print(f"Indexed {i}/{len(files)} files")
```

---

## Database Performance

### DuckDB Optimization

#### Configuration Tuning
```sql
-- Performance settings for DuckDB
PRAGMA threads=4;                    -- Match CPU cores
PRAGMA memory_limit='2GB';           -- Set memory limit
PRAGMA temp_directory='/tmp/duckdb'; -- Fast temporary storage

-- Enable query profiling
PRAGMA enable_profiling=true;
PRAGMA enable_progress_bar=true;

-- Optimize for read-heavy workloads
PRAGMA checkpoint_threshold='1GB';
PRAGMA wal_autocheckpoint=1000;
```

#### Index Optimization
```python
# Vector index configuration
VECTOR_INDEX_CONFIG = {
    "index_type": "hnsw",           # Hierarchical Navigable Small World
    "distance_metric": "cosine",    # Cosine similarity
    "ef_construction": 200,         # Build-time parameter
    "m": 16,                       # Connectivity parameter
    "ef_search": 100,              # Search-time parameter
    "max_connections": 64           # Maximum connections per node
}

# Create optimized indexes
CREATE_INDEX_SQL = """
-- Primary key index
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_id ON documents(id);

-- Path lookup index
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);

-- Vault filtering index
CREATE INDEX IF NOT EXISTS idx_documents_vault ON documents(vault_name);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_documents_content_fts 
ON documents USING FTS(content);

-- Vector similarity index (if supported)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw 
ON documents USING hnsw(embedding) 
WITH (ef_construction=200, m=16);
"""
```

#### Query Optimization
```sql
-- Optimized similarity search query
WITH ranked_results AS (
    SELECT 
        id,
        path,
        vault_name,
        array_cosine_similarity(embedding, $1) as similarity_score,
        content
    FROM documents 
    WHERE vault_name = COALESCE($2, vault_name)  -- Optional vault filter
    AND array_cosine_similarity(embedding, $1) > COALESCE($3, 0.0)  -- Threshold filter
)
SELECT * FROM ranked_results 
ORDER BY similarity_score DESC 
LIMIT $4;

-- Use query plans to optimize
EXPLAIN ANALYZE SELECT ...;
```

#### Storage Optimization
```python
# Database maintenance procedures
def optimize_database(db_path: str) -> None:
    """Optimize database for better performance."""
    
    import duckdb
    
    conn = duckdb.connect(db_path)
    
    try:
        # Analyze table statistics
        conn.execute("ANALYZE documents;")
        
        # Vacuum to reclaim space
        conn.execute("VACUUM;")
        
        # Update table statistics
        conn.execute("UPDATE STATISTICS documents;")
        
        # Checkpoint WAL
        conn.execute("CHECKPOINT;")
        
        print("Database optimization completed")
        
    finally:
        conn.close()

# Run optimization weekly
optimize_database("~/.jarvis/jarvis.duckdb")
```

### Storage Configuration

#### File System Optimization
```bash
# Use SSD for database storage
export JARVIS_DB_PATH="/fast-ssd/jarvis.duckdb"

# Enable SSD optimizations
sudo mount -o remount,discard /fast-ssd/

# Set I/O scheduler for SSDs
echo "noop" | sudo tee /sys/block/nvme0n1/queue/scheduler

# For HDDs, use deadline scheduler
echo "deadline" | sudo tee /sys/block/sda/queue/scheduler
```

#### Memory-Mapped Files
```python
# DuckDB memory mapping configuration
DATABASE_CONFIG = {
    "access_mode": "automatic",
    "checkpoint_threshold": "1GB",
    "force_checkpoint": False,
    "use_temporary_directory": True,
    "temporary_directory": "/tmp/duckdb",
    "max_memory": "2GB",
    "threads": 4
}
```

---

## Graph Search Tuning

### Neo4j Performance

#### Memory Configuration
```bash
# Neo4j memory settings (neo4j.conf)
dbms.memory.heap.initial_size=1G
dbms.memory.heap.max_size=2G
dbms.memory.pagecache.size=1G

# Transaction state memory
dbms.tx_state.memory_allocation=ON_HEAP
dbms.tx_state.max_off_heap_memory=2G
```

#### Query Optimization
```cypher
-- Create performance indexes
CREATE INDEX note_path_index IF NOT EXISTS FOR (n:Note) ON (n.path);
CREATE INDEX note_title_index IF NOT EXISTS FOR (n:Note) ON (n.title);
CREATE INDEX note_tags_index IF NOT EXISTS FOR (n:Note) ON (n.tags);

-- Full-text search index
CREATE FULLTEXT INDEX note_content_fulltext IF NOT EXISTS 
FOR (n:Note) ON EACH [n.content, n.title];

-- Relationship type index
CREATE INDEX rel_type_index IF NOT EXISTS FOR ()-[r:LINKS_TO]-() ON (r.type);
```

#### Connection Pool Tuning
```python
# Neo4j driver configuration
NEO4J_CONFIG = {
    "max_connection_lifetime": 3600,     # 1 hour
    "max_connection_pool_size": 50,      # Connection pool size
    "connection_acquisition_timeout": 60, # Acquisition timeout
    "connection_timeout": 30,            # Connection timeout
    "max_retry_time": 30,               # Retry timeout
    "initial_retry_delay": 1.0,         # Initial retry delay
    "retry_delay_multiplier": 2.0,      # Delay multiplier
    "retry_delay_jitter_factor": 0.2    # Jitter factor
}
```

### Graph Query Optimization

#### Efficient Traversal Patterns
```cypher
-- Optimized graph traversal with limits
MATCH (center:Note {path: $notePath})
MATCH (center)-[r*1..$depth]-(connected:Note)
WITH center, connected, r
WHERE connected <> center
RETURN 
    center.path as center_path,
    collect(DISTINCT {
        path: connected.path,
        title: connected.title,
        tags: connected.tags,
        distance: length(r)
    }) as connections
ORDER BY center_path
LIMIT 1000;

-- Use relationship direction for better performance
MATCH (center:Note {path: $notePath})
MATCH (center)-[r:LINKS_TO*1..$depth]->(connected:Note)
RETURN center, connected, r
LIMIT 500;
```

#### Query Caching
```python
# Query result caching
from functools import lru_cache
import time

class GraphSearchCache:
    def __init__(self, max_size: int = 128, ttl: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get_cache_key(self, note_path: str, depth: int) -> str:
        return f"{note_path}:{depth}"
    
    def get(self, note_path: str, depth: int):
        key = self.get_cache_key(note_path, depth)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None
    
    def put(self, note_path: str, depth: int, result):
        if len(self.cache) >= self.max_size:
            # Remove oldest entries
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = self.get_cache_key(note_path, depth)
        self.cache[key] = (result, time.time())
```

---

## Memory Management

### Memory Optimization

#### Memory Profiling
```python
import psutil
import gc
from typing import Dict, Any

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "gc_count": len(gc.get_objects())
    }

def monitor_memory():
    """Monitor memory usage continuously."""
    import time
    
    while True:
        stats = get_memory_stats()
        print(f"Memory: {stats['rss_mb']:.1f}MB ({stats['percent']:.1f}%)")
        
        if stats['percent'] > 80:
            print("Warning: High memory usage detected")
            gc.collect()  # Force garbage collection
        
        time.sleep(10)
```

#### Garbage Collection Tuning
```python
import gc

# Optimize garbage collection
gc.set_threshold(700, 10, 10)  # More aggressive collection

# Custom memory management
class MemoryManager:
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.max_memory_mb:
            print(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
            gc.collect()
            return False
        return True
    
    def cleanup_if_needed(self):
        """Cleanup memory if needed."""
        if not self.check_memory_usage():
            # Clear caches
            self.clear_caches()
            gc.collect()
    
    def clear_caches(self):
        """Clear various caches to free memory."""
        # Clear query cache
        # Clear encoding cache
        # Clear model cache if needed
        pass
```

#### Model Memory Management
```python
# Lazy model loading
class LazyModelLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def unload_model(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

# Usage
model_loader = LazyModelLoader('sentence-transformers/all-MiniLM-L6-v2')

# Model is loaded on first use
embeddings = model_loader.model.encode(["test query"])

# Unload when memory is tight
model_loader.unload_model()
```

---

## System-Level Optimization

### Operating System Tuning

#### Linux Optimization
```bash
# Kernel parameters for better performance
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max=134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' | sudo tee -a /etc/sysctl.conf

# Apply settings
sudo sysctl -p

# Huge pages for large datasets
echo 'vm.nr_hugepages=1024' | sudo tee -a /etc/sysctl.conf

# I/O scheduler optimization
echo 'noop' | sudo tee /sys/block/*/queue/scheduler  # For SSDs
echo 'deadline' | sudo tee /sys/block/*/queue/scheduler  # For HDDs

# File descriptor limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf
```

#### macOS Optimization
```bash
# Increase file descriptor limits
launchctl limit maxfiles 65536 65536

# Memory pressure relief
sudo purge  # Clear system caches

# Disable unnecessary services
sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.metadata.mds.plist

# SSD optimization
sudo trimforce enable
```

#### Process Priority
```bash
# Run with higher priority
nice -n -10 uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"

# Or use ionice for I/O priority
ionice -c 1 -n 4 uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"

# Combined CPU and I/O priority
nice -n -10 ionice -c 1 -n 4 uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"
```

### Resource Allocation

#### CPU Optimization
```python
# CPU affinity settings
import os
import psutil

def set_cpu_affinity(cores: list = None):
    """Set CPU affinity for the process."""
    if cores is None:
        # Use all available cores
        cores = list(range(psutil.cpu_count()))
    
    try:
        process = psutil.Process()
        process.cpu_affinity(cores)
        print(f"CPU affinity set to cores: {cores}")
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")

# Use specific cores for compute-intensive tasks
set_cpu_affinity([0, 1, 2, 3])  # Use first 4 cores
```

#### Thread Pool Configuration
```python
# Optimized thread pool settings
import concurrent.futures
import os

# Calculate optimal thread count
CPU_COUNT = os.cpu_count()
OPTIMAL_THREADS = min(CPU_COUNT * 2, 8)  # Cap at 8 threads

# Configure thread pools
THREAD_POOL_CONFIG = {
    "search_threads": min(4, CPU_COUNT),
    "index_threads": min(2, CPU_COUNT),
    "io_threads": min(8, CPU_COUNT * 2),
    "graph_threads": min(2, CPU_COUNT)
}

class OptimizedThreadPool:
    def __init__(self):
        self.search_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=THREAD_POOL_CONFIG["search_threads"],
            thread_name_prefix="search"
        )
        self.index_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=THREAD_POOL_CONFIG["index_threads"],
            thread_name_prefix="index"
        )
    
    def shutdown(self):
        self.search_pool.shutdown(wait=True)
        self.index_pool.shutdown(wait=True)
```

---

## Monitoring and Profiling

### Performance Monitoring

#### Real-time Monitoring
```python
import time
import threading
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    timestamp: float
    search_time: float
    memory_usage: float
    cpu_usage: float
    active_connections: int

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring in background."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metric = self._collect_metrics()
                self.metrics.append(metric)
                
                # Keep only last 1000 metrics
                if len(self.metrics) > 1000:
                    self.metrics = self.metrics[-1000:]
                
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        import psutil
        
        process = psutil.Process()
        
        return PerformanceMetrics(
            timestamp=time.time(),
            search_time=0.0,  # To be filled by search operations
            memory_usage=process.memory_percent(),
            cpu_usage=process.cpu_percent(),
            active_connections=len(process.connections())
        )
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.metrics:
            return {}
        
        recent_metrics = self.metrics[-60:]  # Last 5 minutes
        
        return {
            "avg_memory": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "max_memory": max(m.memory_usage for m in recent_metrics),
            "avg_cpu": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "max_cpu": max(m.cpu_usage for m in recent_metrics),
            "sample_count": len(recent_metrics)
        }
```

#### Profiling Tools
```python
# CPU profiling
import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def profile_cpu():
    """Context manager for CPU profiling."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()

# Usage
def search_function(query: str):
    # Your search implementation
    pass

with profile_cpu() as profiler:
    search_function("test query")

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

#### Memory Profiling
```python
# Memory profiling with memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    """Function to profile memory usage."""
    # Your memory-intensive code here
    pass

# Line-by-line memory profiling
# Install: pip install memory_profiler
# Run: python -m memory_profiler script.py
```

### Alerting and Thresholds

#### Performance Alerts
```python
class PerformanceAlerting:
    def __init__(self):
        self.thresholds = {
            "memory_percent": 80.0,
            "cpu_percent": 90.0,
            "search_time_ms": 2000,
            "error_rate": 0.05
        }
        self.alert_handlers = []
    
    def add_alert_handler(self, handler):
        self.alert_handlers.append(handler)
    
    def check_thresholds(self, metrics: PerformanceMetrics):
        """Check if any thresholds are exceeded."""
        alerts = []
        
        if metrics.memory_usage > self.thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.cpu_usage > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.search_time > self.thresholds["search_time_ms"]:
            alerts.append(f"Slow search: {metrics.search_time:.0f}ms")
        
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, message: str):
        """Send alert to all registered handlers."""
        for handler in self.alert_handlers:
            handler(message)

# Example alert handler
def log_alert(message: str):
    print(f"ALERT: {message}")

alerting = PerformanceAlerting()
alerting.add_alert_handler(log_alert)
```

---

## Performance Testing

### Benchmark Suite

```python
import time
import statistics
from typing import List, Dict, Any

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_search(self, searcher, queries: List[str], iterations: int = 10) -> Dict[str, float]:
        """Benchmark search performance."""
        times = []
        
        for query in queries:
            for _ in range(iterations):
                start_time = time.time()
                try:
                    results = searcher.search(query, top_k=10)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                except Exception as e:
                    print(f"Search failed for '{query}': {e}")
        
        return {
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def benchmark_indexing(self, indexer, file_paths: List[str]) -> Dict[str, float]:
        """Benchmark indexing performance."""
        start_time = time.time()
        
        try:
            indexer.index_files(file_paths)
            end_time = time.time()
            
            total_time = end_time - start_time
            files_per_second = len(file_paths) / total_time
            
            return {
                "total_time_s": total_time,
                "files_per_second": files_per_second,
                "avg_time_per_file_ms": (total_time / len(file_paths)) * 1000
            }
        except Exception as e:
            print(f"Indexing benchmark failed: {e}")
            return {}
    
    def run_full_benchmark(self):
        """Run complete performance benchmark suite."""
        print("=== Jarvis Assistant Performance Benchmark ===")
        
        # Benchmark search
        search_queries = [
            "machine learning artificial intelligence",
            "project management productivity",
            "data science analytics",
            "programming software development",
            "research methodology analysis"
        ]
        
        search_results = self.benchmark_search(searcher, search_queries)
        print(f"Search Performance:")
        print(f"  Mean: {search_results['mean_ms']:.1f}ms")
        print(f"  Median: {search_results['median_ms']:.1f}ms")
        print(f"  Range: {search_results['min_ms']:.1f}ms - {search_results['max_ms']:.1f}ms")
        
        # Add more benchmarks as needed
        
        return {
            "search": search_results,
            # "indexing": indexing_results,
            # "graph": graph_results
        }

# Usage
benchmark = PerformanceBenchmark()
results = benchmark.run_full_benchmark()
```

### Load Testing

```python
import concurrent.futures
import random
import time
from typing import List

class LoadTester:
    def __init__(self, searcher):
        self.searcher = searcher
        self.queries = [
            "artificial intelligence machine learning",
            "data science analytics",
            "project management",
            "software development",
            "research methodology"
        ]
    
    def simulate_user(self, user_id: int, duration_seconds: int) -> Dict[str, Any]:
        """Simulate a single user's search behavior."""
        search_count = 0
        error_count = 0
        response_times = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            query = random.choice(self.queries)
            
            try:
                search_start = time.time()
                results = self.searcher.search(query, top_k=10)
                response_time = (time.time() - search_start) * 1000
                
                response_times.append(response_time)
                search_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"User {user_id} search error: {e}")
            
            # Random delay between searches
            time.sleep(random.uniform(1, 5))
        
        return {
            "user_id": user_id,
            "search_count": search_count,
            "error_count": error_count,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0
        }
    
    def run_load_test(self, concurrent_users: int, duration_seconds: int) -> Dict[str, Any]:
        """Run load test with specified concurrent users."""
        print(f"Starting load test: {concurrent_users} users for {duration_seconds}s")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(self.simulate_user, user_id, duration_seconds)
                for user_id in range(concurrent_users)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Aggregate results
        total_searches = sum(r["search_count"] for r in results)
        total_errors = sum(r["error_count"] for r in results)
        avg_response_times = [r["avg_response_time"] for r in results if r["avg_response_time"] > 0]
        
        return {
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "total_searches": total_searches,
            "total_errors": total_errors,
            "error_rate": total_errors / total_searches if total_searches > 0 else 0,
            "searches_per_second": total_searches / duration_seconds,
            "avg_response_time": statistics.mean(avg_response_times) if avg_response_times else 0,
            "user_results": results
        }

# Usage
load_tester = LoadTester(searcher)
load_results = load_tester.run_load_test(concurrent_users=5, duration_seconds=60)
print(f"Load test results: {load_results}")
```

---

## Best Practices Summary

### Configuration Checklist

- [ ] **Model Selection**: Choose appropriate model for use case
- [ ] **Batch Size**: Optimize based on available memory
- [ ] **Caching**: Enable query and encoding caches
- [ ] **Thresholds**: Set similarity thresholds for performance
- [ ] **Database**: Optimize DuckDB configuration
- [ ] **Neo4j**: Tune memory and connection settings
- [ ] **System**: Configure OS-level optimizations

### Monitoring Checklist

- [ ] **Memory Usage**: Monitor and set alerts
- [ ] **Response Times**: Track search performance
- [ ] **Error Rates**: Monitor for failures
- [ ] **Resource Usage**: Track CPU, disk, network
- [ ] **Database Performance**: Monitor query execution
- [ ] **Graph Performance**: Track traversal times

### Maintenance Checklist

- [ ] **Database Vacuum**: Regular database optimization
- [ ] **Index Rebuilding**: Periodic index maintenance
- [ ] **Cache Clearing**: Clear caches when needed
- [ ] **Log Rotation**: Manage log file sizes
- [ ] **Performance Review**: Regular performance analysis
- [ ] **Threshold Tuning**: Adjust based on usage patterns

---

## Next Steps

- [Backup Recovery](backup-recovery.md) - Data protection strategies
- [Updates Migration](updates-migration.md) - Handling system changes
- [Troubleshooting](troubleshooting.md) - Problem resolution guide
- [Configuration Reference](../06-reference/configuration-reference.md) - Complete config options