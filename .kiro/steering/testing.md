# Testing Standards & Quality Assurance

## Testing Philosophy
- **Strategic Testing**: Focus on critical paths and high-risk areas, not exhaustive coverage
- **Minimal Effective Coverage**: Test what matters - core functionality, edge cases, and integration points
- **Fast Feedback**: All tests should run in <60 seconds total
- **Pragmatic Approach**: Write tests that prevent real bugs, not just increase coverage numbers

## What to Test (Priority Order)

### 1. Critical User Paths (Must Test)
- MCP tool endpoints that users directly interact with
- Core search functionality (semantic, keyword, combined)
- Vault indexing and file parsing
- Error handling for common failure scenarios

### 2. High-Risk Areas (Should Test)
- Database operations and connection handling
- File system operations and edge cases
- Performance bottlenecks and memory usage
- Configuration and service initialization

### 3. Skip Testing (Don't Waste Time)
- Simple getters/setters and property access
- Trivial utility functions with no business logic
- Third-party library integrations (trust their tests)
- Internal implementation details that don't affect behavior

### Test Structure (Simplified)
```
resources/tests/
├── test_mcp_tools.py       # All MCP endpoints in one file
├── test_search_core.py     # Core search functionality
├── test_vault_ops.py       # File operations and indexing
└── test_integration.py     # End-to-end critical paths
```

## Minimal Testing Patterns

### Focus on Behavior, Not Implementation
```python
# Good: Test the behavior users care about
def test_search_returns_relevant_results():
    results = search_service.search("python programming")
    assert len(results) > 0
    assert any("python" in r.content.lower() for r in results)

# Skip: Testing internal implementation details
# def test_search_calls_embedding_service_with_correct_params()  # Don't do this
```

### Test Happy Path + One Error Case
```python
# Test the main success scenario
def test_mcp_search_tool_success():
    response = mcp_server.search_semantic("machine learning", limit=5)
    assert response["success"] is True
    assert len(response["data"]["results"]) <= 5

# Test one critical error scenario
def test_mcp_search_tool_handles_invalid_vault():
    response = mcp_server.search_semantic("query", vault="nonexistent")
    assert response["success"] is False
    assert "not found" in response["error"]["message"]
```

### Simple Test Data
```python
# Keep test data minimal and inline
def test_vault_indexing():
    test_notes = [
        ("note1.md", "# Python\nPython is great"),
        ("note2.md", "# JavaScript\nJS is also good")
    ]
    
    vault = create_temp_vault(test_notes)
    index_service.index_vault(vault)
    
    results = search_service.search("python")
    assert len(results) == 1
    assert "note1.md" in results[0].path
```

## Integration Testing (Minimal Approach)

### One End-to-End Test Per Critical Path
```python
def test_complete_search_workflow():
    """Single test covering: vault setup -> indexing -> search -> results"""
    # Setup minimal test vault
    vault_path = create_test_vault([
        ("ai.md", "# AI\nArtificial intelligence is fascinating"),
        ("python.md", "# Python\nPython programming language")
    ])
    
    # Index the vault
    indexer.index_vault(vault_path)
    
    # Test search works end-to-end
    results = mcp_server.search_semantic("artificial intelligence")
    
    assert results["success"] is True
    assert len(results["data"]["results"]) > 0
    assert "ai.md" in results["data"]["results"][0]["path"]
```

### Skip Detailed Integration Testing
- Don't test every combination of services
- Don't test database schema details
- Don't test file encoding edge cases unless they've caused real bugs
- Trust that individual components work if unit tests pass

## Performance Testing (Only When Needed)

### Skip Performance Tests Unless There's a Problem
- Don't write performance tests preemptively
- Add them only when users report slow performance
- Focus on real bottlenecks, not theoretical ones

### Simple Performance Check (If Needed)
```python
def test_search_not_extremely_slow():
    """Only test if search is catastrophically slow (>30 seconds)"""
    start_time = time.time()
    results = search_service.search("test query")
    elapsed = time.time() - start_time
    
    assert elapsed < 30.0  # Catch only catastrophic slowness
    assert len(results) >= 0  # Just ensure it doesn't crash
```

## Error Testing (Only Critical Errors)

### Test User-Facing Errors Only
```python
# Good: Test errors users will encounter
def test_mcp_tool_handles_missing_vault():
    response = mcp_server.search_semantic("query", vault="missing")
    assert response["success"] is False
    assert "vault" in response["error"]["message"].lower()

# Skip: Internal error handling details
# def test_database_connection_retry_logic()  # Don't test this
```

### One Error Test Per Tool
- Test one common error scenario per MCP tool
- Focus on errors that affect user experience
- Skip testing internal error recovery mechanisms

## Test Data (Keep It Simple)

### Minimal Test Data
```python
# Good: Simple, inline test data
def test_search():
    notes = [("test.md", "# Test\nSome content")]
    vault = create_temp_vault(notes)
    # ... rest of test

# Skip: Complex test data generation
# Don't create elaborate test data factories unless absolutely necessary
```

### Use Real Examples When Needed
- Copy actual problematic files when debugging specific issues
- Keep a small set of real vault examples for integration tests
- Don't generate synthetic data unless testing scale

## Continuous Integration (Streamlined)

### Pre-commit Hooks (Essential Only)
- Run linting (ruff) and formatting
- Type checking with mypy
- Run critical tests only (< 30 seconds)

### CI Pipeline (Fast and Focused)
```bash
# Essential quality checks
uv run ruff check src/ --no-fix
uv run ruff format src/ --check
uv run mypy src/
uv run pytest resources/tests/ --maxfail=3 -x  # Stop on first 3 failures
```

### Skip Heavy CI Setup
- Don't require isolated test environments for every PR
- Don't run full test suites on every commit
- Focus on code quality and critical functionality only

## Coverage Requirements (Realistic Targets)

### Coverage Philosophy
- **Ignore Coverage Metrics**: Don't chase coverage percentages
- **Focus on Critical Code**: Ensure MCP tools and core search work
- **Quality Over Quantity**: Better to have 10 good tests than 100 meaningless ones

### What Coverage Actually Means
- 60% coverage of critical paths > 95% coverage of everything
- Test the code that would break the user experience
- Skip coverage reporting unless debugging test gaps

### Coverage Reporting (Optional)
```bash
# Only run coverage when investigating test gaps
uv run pytest --cov=src/jarvis/mcp --cov-report=term-missing
# Focus on specific modules, not the entire codebase
```

## Test Maintenance

### Regular Test Review
- Remove obsolete tests when refactoring
- Update tests when requirements change
- Optimize slow tests
- Fix flaky tests immediately

### Test Documentation
- Document complex test scenarios
- Explain test data setup and teardown
- Provide examples of good test patterns
- Maintain test troubleshooting guides