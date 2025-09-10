# Performance Standards

## Core Performance Philosophy
- **Local-First**: Optimize for user's machine, not cloud scale
- **Graceful Degradation**: Core features work even under resource constraints
- **Predictable**: Consistent response times across vault sizes

## Key Performance Targets
- **Search Operations**: < 5 seconds for typical vaults
- **MCP Tools**: < 15 seconds maximum response time
- **Memory Usage**: < 1GB for normal operations
- **Large Vaults**: Use progress indicators for operations > 10 seconds

## When Performance Matters
- **User-Facing Operations**: Search, vault listing, health checks
- **Background Indexing**: Should not block user interactions
- **Memory Pressure**: Monitor and degrade gracefully when approaching limits

## Caching Strategy
- **Multi-Layer**: In-memory → DuckDB → File system → Source files
- **Cache Invalidation**: File changes clear related caches automatically
- **TTL**: 5 minutes for search results, 1 hour for embeddings

## Performance Monitoring
- **Key Metrics**: Response times, memory usage, cache hit ratios
- **Logging**: Include timing info in structured logs for MCP tools
- **Alerting**: Monitor for operations exceeding target times

## Database Performance
- **Indexing**: Use HNSW for vector similarity, FTS for text search
- **Batch Operations**: Process embeddings in batches, not individually
- **Connection Pooling**: Max 10 connections, 30s timeout

## Memory Management
- **Use Generators**: Stream large result sets instead of loading all into memory
- **Batch Processing**: Process vault notes in batches of 50-100
- **Memory Monitoring**: Pause and garbage collect when approaching 1.5GB usage
- **Lazy Loading**: Load embeddings and graph data only when needed

## Optimization Strategies
- **Lazy Loading**: Load embeddings and graph data only when needed
- **Batch Processing**: Process vault notes in batches of 50-100
- **Incremental Updates**: Only reprocess changed files
- **Connection Pooling**: Reuse database connections

## Performance Testing (Only When Needed)
- **Load Testing**: Test concurrent search requests (10+ simultaneous)
- **Regression Testing**: Ensure performance doesn't degrade over time
- **Memory Testing**: Verify operations stay under memory limits

## Health Monitoring
- **Key Health Checks**: Database connection, memory usage, cache hit ratio
- **Alert Thresholds**: >10s response time, >85% memory usage, <70% cache hit ratio