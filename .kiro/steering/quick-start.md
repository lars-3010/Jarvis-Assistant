# Quick Start Guide

## Get Productive in 15 Minutes

### 1. Essential Commands (2 minutes)
```bash
# Setup
uv sync                                    # Install dependencies
uv run ruff format src/                   # Format code
uv run mypy src/                          # Type check

# Development
uv run jarvis mcp --vault /path/to/vault  # Start MCP server
uv run pytest resources/tests/            # Run tests
```

### 2. Project Structure (3 minutes)
```
src/jarvis/
├── mcp/           # MCP tools - user-facing functionality
├── services/      # Core business logic (vector, graph, vault)
├── models/        # Data structures and schemas
└── utils/         # Helpers and utilities

resources/tests/   # All tests in 4 files max
```

### 3. Adding a New MCP Tool (5 minutes)
1. **Create tool in `src/jarvis/mcp/plugins/`**
2. **Add to MCP server registration**
3. **Write one test in `resources/tests/test_mcp_tools.py`**
4. **Update tool documentation**

### 4. Common Development Patterns (3 minutes)

#### Service Access
```python
# Always use dependency injection
from jarvis.core.container import get_service
search_service = get_service("vector_service")
```

#### Error Handling
```python
# Return structured errors for MCP tools
return {
    "success": false,
    "error": {
        "code": "VAULT_NOT_FOUND",
        "message": "Vault not found at specified path",
        "suggestions": ["Check vault path", "Verify permissions"]
    }
}
```

#### Response Format
```python
# Always include metadata
return {
    "success": true,
    "data": results,
    "metadata": {
        "processing_time_ms": elapsed_ms,
        "confidence_score": 0.85
    }
}
```

### 5. Testing Strategy (2 minutes)
- **Test MCP tools**: Success + one error case
- **Test core services**: Happy path + edge cases
- **Skip**: Internal implementation details, getters/setters
- **Focus**: User-facing functionality that could break

### 6. When You're Stuck
- **Performance issues**: Check `performance.md`
- **Data modeling**: Check `data-modeling.md`
- **Common patterns**: Check `common-patterns.md`
- **Debugging**: Check `debugging.md`

## Ready to Code!
You now know enough to be productive. Dive into the codebase and refer to other steering docs when needed.