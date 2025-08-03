# MCP Protocol Standards & Best Practices

## Core MCP Principles
- **Structured Responses**: All MCP tools should return structured JSON when possible, with fallback to human-readable text
- **Error Handling**: Use proper MCP error codes and provide actionable error messages
- **Performance**: Tools should respond within 15 seconds for typical operations
- **Graceful Degradation**: Tools should work even when optional services are unavailable

## Response Format Standards

### Structured JSON Responses
```python
{
    "success": true,
    "data": {
        # Core response data
    },
    "metadata": {
        "timestamp": "2024-01-15T10:30:00Z",
        "processing_time_ms": 150,
        "confidence_score": 0.85,
        "data_freshness": "current"
    },
    "analytics": {
        # Optional analytical insights
    }
}
```

### Error Response Format
```python
{
    "success": false,
    "error": {
        "code": "VAULT_NOT_FOUND",
        "message": "Vault not found at specified path",
        "details": "Path '/invalid/path' does not exist or is not accessible",
        "suggestions": ["Check vault path", "Verify permissions"]
    }
}
```

## Tool Implementation Patterns

### Service Integration
- Always use dependency injection container for service access
- Handle service unavailability gracefully
- Log service interactions for debugging

### Caching Strategy
- Cache expensive operations (embeddings, graph queries)
- Respect cache TTL and invalidation patterns
- Provide cache status in metadata

### Input Validation
- Validate all parameters using Pydantic models
- Provide clear validation error messages
- Support both required and optional parameters

## Error Handling Standards

### Error Categories
- `VAULT_ERROR`: Vault access or parsing issues
- `SEARCH_ERROR`: Search operation failures
- `SYSTEM_ERROR`: System resource or configuration issues
- `VALIDATION_ERROR`: Input parameter validation failures

### Error Response Guidelines
- Include specific error codes for programmatic handling
- Provide human-readable error messages
- Suggest concrete remediation steps
- Include relevant context (file paths, parameter values)

## Performance Requirements

### Response Time Targets
- Simple queries (health, stats): < 1 second
- Search operations: < 5 seconds
- Complex analytics: < 15 seconds
- Bulk operations: Progress indicators required

### Resource Management
- Limit memory usage per operation
- Implement timeouts for long-running operations
- Use connection pooling for database operations
- Clean up resources in finally blocks

## Testing Standards

### MCP Tool Testing
- Test both success and error scenarios
- Mock external dependencies (databases, file system)
- Validate response schema compliance
- Test parameter validation edge cases

### Integration Testing
- Test with real vault data
- Verify service integration points
- Test concurrent operation handling
- Validate caching behavior

## Documentation Requirements

### Tool Documentation
- Clear parameter descriptions with examples
- Response format documentation with schemas
- Error condition documentation
- Performance characteristics and limitations

### Code Documentation
- Docstrings for all public methods
- Type hints for all parameters and returns
- Inline comments for complex logic
- Architecture decision records for design choices

## Monitoring & Observability

### Metrics Collection
- Response times per tool
- Error rates and types
- Cache hit/miss ratios
- Resource usage patterns

### Logging Standards
- Structured logging with JSON format
- Include request IDs for tracing
- Log performance metrics
- Separate debug and production log levels

## Backward Compatibility

### Version Management
- Maintain API compatibility across versions
- Deprecate features gracefully with warnings
- Provide migration guides for breaking changes
- Support multiple response formats during transitions

### Client Support
- Support both text and structured responses
- Provide feature detection capabilities
- Handle client capability negotiation
- Maintain fallback modes for older clients