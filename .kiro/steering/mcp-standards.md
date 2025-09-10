# MCP Protocol Standards

## Core Principles
- **Structured JSON**: Return structured data when possible, fallback to text
- **15 Second Limit**: All MCP tools must respond within 15 seconds
- **Graceful Errors**: Provide actionable error messages with suggestions
- **Service Independence**: Tools work even when optional services are down

## Response Formats

### Success Response
```json
{
    "success": true,
    "data": { /* core response data */ },
    "metadata": {
        "processing_time_ms": 150,
        "confidence_score": 0.85
    }
}
```

### Error Response
```json
{
    "success": false,
    "error": {
        "code": "VAULT_NOT_FOUND",
        "message": "Vault not found at specified path",
        "suggestions": ["Check vault path", "Verify permissions"]
    }
}
```

## Implementation Essentials

### Service Integration
- Use dependency injection for service access
- Handle service unavailability gracefully
- Cache expensive operations (embeddings, graph queries)

### Input Validation
- Validate parameters using Pydantic models
- Provide clear validation error messages

### Error Categories
- `VAULT_ERROR`: Vault access issues
- `SEARCH_ERROR`: Search operation failures  
- `SYSTEM_ERROR`: Resource/configuration issues
- `VALIDATION_ERROR`: Parameter validation failures

### Performance Requirements
- Simple queries: < 1 second
- Search operations: < 5 seconds
- Complex analytics: < 15 seconds
- Use timeouts and clean up resources

## Testing & Documentation
- Test success and error scenarios for each MCP tool
- Document parameters, response formats, and error conditions
- Use structured logging with JSON format
- Include performance characteristics in documentation

## Backward Compatibility
- Maintain API compatibility across versions
- Support both text and structured responses
- Deprecate features gracefully with warnings