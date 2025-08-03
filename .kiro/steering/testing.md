# Testing Standards & Quality Assurance

## Testing Philosophy
- **Test-Driven Development**: Write tests before implementation when possible
- **Comprehensive Coverage**: Aim for >90% code coverage with meaningful tests
- **Fast Feedback**: Unit tests should run in <30 seconds, integration tests in <5 minutes
- **Realistic Testing**: Use real data patterns and edge cases from actual vaults

## Test Structure & Organization

### Test Directory Layout
```
resources/tests/
├── unit/                   # Fast, isolated unit tests
│   ├── test_vector_service.py
│   ├── test_graph_indexer.py
│   └── test_mcp_tools.py
├── integration/            # End-to-end integration tests
│   ├── test_search_workflow.py
│   └── test_vault_indexing.py
└── mcp/                   # MCP protocol-specific tests
    ├── test_mcp_server.py
    └── test_tool_responses.py
```

### Test Naming Conventions
- Test files: `test_{module_name}.py`
- Test classes: `Test{ClassName}`
- Test methods: `test_{what_is_being_tested}_{expected_outcome}`
- Example: `test_semantic_search_returns_relevant_results()`

## Unit Testing Standards

### Test Isolation
- Mock all external dependencies (databases, file system, network)
- Use dependency injection for testable code
- Reset state between tests
- No shared test data between test methods

### Test Data Management
```python
# Use fixtures for reusable test data
@pytest.fixture
def sample_vault_notes():
    return [
        {"path": "note1.md", "content": "Sample content", "tags": ["test"]},
        {"path": "note2.md", "content": "Another note", "tags": ["example"]}
    ]

# Use factories for dynamic test data
def create_test_document(title="Test", content="Content"):
    return Document(title=title, content=content, created_at=datetime.now())
```

### Assertion Patterns
```python
# Prefer specific assertions over generic ones
assert result.status == "success"  # Good
assert result  # Too generic

# Test both positive and negative cases
def test_search_finds_relevant_results():
    results = search_service.search("python")
    assert len(results) > 0
    assert all("python" in r.content.lower() for r in results)

def test_search_returns_empty_for_nonexistent_term():
    results = search_service.search("xyznonexistent")
    assert len(results) == 0
```

## Integration Testing Standards

### Database Testing
- Use test databases (separate DuckDB files)
- Clean up test data after each test
- Test with realistic data volumes
- Verify database schema migrations

### File System Testing
- Use temporary directories for test vaults
- Create realistic vault structures
- Test with various file encodings and formats
- Clean up test files after execution

### MCP Protocol Testing
```python
# Test MCP tool responses
async def test_semantic_search_tool():
    request = {
        "method": "tools/call",
        "params": {
            "name": "search_semantic",
            "arguments": {"query": "machine learning", "limit": 5}
        }
    }
    
    response = await mcp_server.handle_request(request)
    
    assert response["success"] is True
    assert "results" in response["data"]
    assert len(response["data"]["results"]) <= 5
    assert "metadata" in response
```

## Performance Testing

### Response Time Testing
```python
import time

def test_search_performance():
    start_time = time.time()
    results = search_service.search("test query")
    elapsed = time.time() - start_time
    
    assert elapsed < 5.0  # Should complete within 5 seconds
    assert len(results) > 0
```

### Memory Usage Testing
```python
import psutil
import os

def test_memory_usage_within_limits():
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    large_search_results = search_service.search_large_vault()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Should not increase memory by more than 100MB
    assert memory_increase < 100 * 1024 * 1024
```

## Error Testing Standards

### Exception Testing
```python
def test_search_handles_invalid_vault_path():
    with pytest.raises(VaultNotFoundError) as exc_info:
        search_service.search_vault("/nonexistent/path", "query")
    
    assert "not found" in str(exc_info.value)
    assert exc_info.value.error_code == "VAULT_NOT_FOUND"
```

### Error Recovery Testing
```python
def test_service_recovers_from_database_failure():
    # Simulate database failure
    with mock.patch('jarvis.database.connection_pool.get_connection') as mock_conn:
        mock_conn.side_effect = DatabaseConnectionError()
        
        # Service should handle gracefully
        result = search_service.search("query")
        assert result.status == "error"
        assert "database unavailable" in result.message
```

## Test Data Management

### Realistic Test Data
- Use actual Obsidian vault structures
- Include various note formats (markdown, frontmatter, links)
- Test with different vault sizes (small, medium, large)
- Include edge cases (empty notes, very long notes, special characters)

### Test Data Generation
```python
def generate_test_vault(num_notes=100, avg_size=1000):
    """Generate realistic test vault with specified characteristics"""
    vault_path = tempfile.mkdtemp()
    
    for i in range(num_notes):
        note_path = os.path.join(vault_path, f"note_{i:03d}.md")
        content = generate_realistic_note_content(avg_size)
        
        with open(note_path, 'w') as f:
            f.write(content)
    
    return vault_path
```

## Continuous Integration Standards

### Pre-commit Hooks
- Run linting (ruff) and formatting
- Execute fast unit tests
- Type checking with mypy
- Security scanning

### CI Pipeline Requirements
```bash
# Quality checks that must pass
uv run ruff check src/ --no-fix
uv run ruff format src/ --check
uv run mypy src/
uv run pytest resources/tests/unit/ --cov=src/jarvis --cov-fail-under=90
uv run pytest resources/tests/integration/ --maxfail=5
```

### Test Environment Setup
- Isolated test environments for each PR
- Consistent Python version (3.11+)
- Clean database state for each test run
- Proper cleanup of test artifacts

## Coverage Requirements

### Coverage Targets
- **Unit Tests**: >90% line coverage
- **Integration Tests**: >80% feature coverage
- **Critical Paths**: 100% coverage (search, indexing, MCP tools)

### Coverage Exclusions
- Third-party code
- Configuration files
- Development utilities
- Deprecated code marked for removal

### Coverage Reporting
```bash
# Generate coverage reports
uv run pytest --cov=src/jarvis --cov-report=html --cov-report=term
uv run coverage xml  # For CI integration
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