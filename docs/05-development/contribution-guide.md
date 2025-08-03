# Contribution Guide

Welcome to the Jarvis Assistant project! This guide will help you understand how to contribute effectively to the project, from reporting issues to submitting code changes.

## Quick Navigation

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Review Process](#code-review-process)
- [Issue Guidelines](#issue-guidelines)
- [Documentation Contributions](#documentation-contributions)
- [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.11+** installed
- **Git** for version control
- **UV** package manager (`pip install uv`)
- **Neo4j** database (for graph functionality testing)
- Basic familiarity with MCP (Model Context Protocol)

### First-Time Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/jarvis-assistant.git
   cd jarvis-assistant
   ```

2. **Set Up Development Environment**
   ```bash
   # Install dependencies
   uv sync
   
   # Set up pre-commit hooks
   uv run pre-commit install
   
   # Verify installation
   uv run jarvis --help
   ```

3. **Configure Remote**
   ```bash
   # Add upstream remote
   git remote add upstream https://github.com/original-owner/jarvis-assistant.git
   
   # Verify remotes
   git remote -v
   ```

4. **Create Environment File**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit with your settings
   nano .env
   ```

---

## Development Setup

### Environment Configuration

Create a `.env` file in the project root:

```env
# Development Configuration
JARVIS_LOG_LEVEL=DEBUG
JARVIS_DB_PATH=./data/jarvis_dev.duckdb
JARVIS_VAULT_PATH=./resources/test_vault

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Testing Configuration
PYTEST_TIMEOUT=30
COVERAGE_THRESHOLD=80
```

### Local Testing Setup

```bash
# Create test vault
mkdir -p ./resources/test_vault
echo "# Test Note" > ./resources/test_vault/test.md
echo "This is a test note for development." >> ./resources/test_vault/test.md

# Index test data
uv run jarvis index --vault ./resources/test_vault

# Test MCP server
uv run jarvis mcp --vault ./resources/test_vault --watch
```

### Quality Assurance Tools

```bash
# Run all quality checks
./scripts/quality-check.sh

# Individual tools
uv run ruff check src/                    # Linting
uv run ruff format src/                   # Formatting
uv run mypy src/                         # Type checking
uv run pytest resources/tests/           # Testing
uv run pytest --cov=src/jarvis          # Coverage
```

---

## Contribution Workflow

### 1. Choose an Issue or Feature

- **For Beginners**: Look for issues labeled `good first issue` or `help wanted`
- **For Experienced**: Check `enhancement`, `bug`, or `feature` labels
- **For Documentation**: Look for `documentation` labeled issues

### 2. Create Feature Branch

```bash
# Sync with upstream
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/semantic-search-enhancement

# For bug fixes
git checkout -b bugfix/fix-graph-connection-error

# For documentation
git checkout -b docs/improve-api-examples
```

### 3. Development Process

#### Making Changes

```bash
# Make your changes
# ... edit files ...

# Test changes locally
uv run pytest resources/tests/
uv run jarvis index --vault ./resources/test_vault
uv run jarvis mcp --vault ./resources/test_vault

# Format and lint
uv run ruff format src/
uv run ruff check src/ --fix
```

#### Commit Guidelines

Follow conventional commit format:

```bash
# Feature commits
git commit -m "feat: add similarity threshold to semantic search"

# Bug fix commits
git commit -m "fix: resolve Neo4j connection timeout issue"

# Documentation commits
git commit -m "docs: improve MCP tool examples in API guide"

# Test commits
git commit -m "test: add integration tests for graph search"

# Other types: chore, style, refactor, perf
```

#### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Examples:**

```
feat(search): add similarity threshold filtering

Add optional similarity_threshold parameter to semantic search that
allows users to filter results below a specified similarity score.
This improves search precision for specific use cases.

- Add similarity_threshold parameter to VectorSearcher.search()
- Update MCP tool schema to include new parameter
- Add tests for threshold filtering
- Update documentation with examples

Closes #123
```

### 4. Testing Requirements

All contributions must include appropriate tests:

```bash
# Unit tests for new functions
# resources/tests/unit/services/test_new_feature.py

# Integration tests for new workflows
# resources/tests/integration/test_new_integration.py

# MCP tests for new tools
# resources/tests/mcp/test_new_mcp_tool.py
```

#### Test Examples

```python
# Example unit test
def test_similarity_threshold_filtering():
    """Test that similarity threshold properly filters results."""
    # Arrange
    searcher = VectorSearcher(mock_db, mock_encoder, vaults)
    
    # Act
    results = searcher.search("test query", similarity_threshold=0.8)
    
    # Assert
    assert all(r.similarity_score >= 0.8 for r in results)

# Example integration test
@pytest.mark.asyncio
async def test_mcp_tool_with_threshold():
    """Test MCP tool with similarity threshold parameter."""
    # Arrange
    server = create_mcp_server(test_vaults, test_db)
    
    # Act
    result = await server.call_tool("search-semantic", {
        "query": "test",
        "similarity_threshold": 0.7
    })
    
    # Assert
    assert result[0].type == "text"
    assert "results" in result[0].text
```

### 5. Documentation Updates

Update documentation for any user-facing changes:

```bash
# Update API documentation
# docs/06-reference/api-reference.md

# Update usage examples
# docs/04-usage/api-examples.md

# Update developer guide if needed
# docs/05-development/developer-guide.md
```

### 6. Submit Pull Request

```bash
# Push feature branch
git push origin feature/semantic-search-enhancement

# Create pull request on GitHub
# - Use descriptive title
# - Include detailed description
# - Reference related issues
# - Add screenshots if relevant
```

---

## Code Review Process

### Pull Request Template

When creating a pull request, include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests pass locally

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented appropriately
- [ ] Documentation updated
- [ ] No new warnings introduced

## Related Issues
Fixes #123
Related to #456
```

### Review Criteria

Reviewers will check for:

1. **Code Quality**
   - Follows [code standards](code-standards.md)
   - Proper error handling
   - Appropriate logging
   - Performance considerations

2. **Testing**
   - Adequate test coverage
   - Tests cover edge cases
   - Integration tests for new features
   - No failing tests

3. **Documentation**
   - Public APIs documented
   - Usage examples provided
   - Breaking changes documented
   - README updated if needed

4. **Functionality**
   - Feature works as described
   - No regression in existing features
   - Proper input validation
   - Graceful error handling

### Addressing Review Comments

```bash
# Make requested changes
# ... edit files ...

# Commit changes
git add .
git commit -m "address review comments: improve error handling"

# Push updates
git push origin feature/semantic-search-enhancement
```

---

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
## Bug Description
Clear description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Run command '...'
2. Use parameters '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g. macOS 14.0]
- Python: [e.g. 3.11.5]
- Jarvis Assistant: [e.g. 0.2.0]
- Neo4j: [e.g. 5.0.0]

## Additional Context
Any other context about the problem.

## Logs
```
Paste relevant log output here
```

### Feature Requests

Use the feature request template:

```markdown
## Feature Description
Clear description of the feature you'd like to see.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
Detailed description of how you envision the feature working.

## Alternative Solutions
Other approaches you've considered.

## Additional Context
Any other context or screenshots about the feature request.
```

### Issue Labels

Common labels and their meanings:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on
- `duplicate`: This issue already exists

---

## Documentation Contributions

### Documentation Structure

Follow the established documentation structure:

```
docs/
├── 01-overview/        # Project context
├── 02-system-design/   # Technical architecture
├── 03-getting-started/ # User onboarding
├── 04-usage/          # Practical examples
├── 05-development/    # Developer guides
├── 06-reference/      # API reference
└── 07-maintenance/    # Operational guides
```

### Documentation Standards

1. **Writing Style**
   - Use clear, concise language
   - Include code examples
   - Provide practical examples
   - Use consistent terminology

2. **Structure**
   - Start with overview/summary
   - Use logical section ordering
   - Include navigation links
   - Add cross-references

3. **Examples**
   - Include working code examples
   - Show expected outputs
   - Cover common use cases
   - Include error scenarios

### Documentation Workflow

```bash
# Create documentation branch
git checkout -b docs/improve-search-examples

# Edit documentation files
# docs/04-usage/api-examples.md

# Test documentation locally
# (if using documentation server)

# Commit changes
git commit -m "docs: improve search API examples with more use cases"

# Submit pull request
git push origin docs/improve-search-examples
```

---

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

1. **Be Respectful**: Treat all community members with respect
2. **Be Collaborative**: Work together constructively
3. **Be Inclusive**: Welcome people of all backgrounds
4. **Be Patient**: Help others learn and grow
5. **Be Professional**: Maintain professional communication

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests, questions
- **Pull Requests**: Code contributions, documentation improvements
- **Discussions**: General questions, ideas, showcase

### Getting Help

1. **Check Documentation**: Start with the [docs](../README.md)
2. **Search Issues**: Look for existing solutions
3. **Ask Questions**: Create an issue with the `question` label
4. **Join Discussions**: Participate in community discussions

### Recognition

Contributors are recognized through:

- **Contributor List**: Added to project contributors
- **Release Notes**: Significant contributions mentioned
- **Community Highlights**: Featured in project updates

---

## Development Tips

### Performance Considerations

```python
# Profile code performance
import cProfile
import pstats

def profile_search_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    results = searcher.search("test query")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### Debugging

```python
# Use structured logging for debugging
import logging

logger = logging.getLogger(__name__)

def debug_search_process(query: str):
    logger.debug(f"Starting search for: {query}")
    
    # Add debug points
    logger.debug(f"Encoded query length: {len(encoded_query)}")
    logger.debug(f"Database search returned: {len(results)} results")
    
    return results
```

### Testing Tips

```bash
# Run specific test categories
uv run pytest resources/tests/unit/           # Unit tests only
uv run pytest resources/tests/integration/   # Integration tests only
uv run pytest -k "search"                    # Tests with "search" in name
uv run pytest --tb=short                     # Shorter traceback format
uv run pytest -v                            # Verbose output
```

---

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

For maintainers creating releases:

1. **Version Bump**: Update version in `pyproject.toml`
2. **Changelog**: Update `CHANGELOG.md` with changes
3. **Documentation**: Ensure docs are up to date
4. **Testing**: Run full test suite
5. **Tag Release**: Create git tag and GitHub release

---

## Frequently Asked Questions

### Q: How do I run tests for a specific component?

```bash
# Vector search tests
uv run pytest resources/tests/unit/services/vector/

# MCP server tests
uv run pytest resources/tests/mcp/

# Graph search tests
uv run pytest resources/tests/unit/services/graph/
```

### Q: How do I add a new MCP tool?

1. Define the tool in `src/jarvis/mcp/server.py`
2. Add tool handler function
3. Update tool schema
4. Add comprehensive tests
5. Update documentation

### Q: How do I test with different Python versions?

```bash
# Use UV with specific Python version
uv python install 3.11
uv python install 3.12

# Test with specific version
uv run --python 3.11 pytest resources/tests/
uv run --python 3.12 pytest resources/tests/
```

### Q: How do I contribute to documentation?

1. Edit relevant files in `docs/`
2. Follow documentation standards
3. Include practical examples
4. Test documentation locally
5. Submit pull request

---

## Next Steps

- [Developer Guide](developer-guide.md) - Development setup and workflow
- [Testing Strategy](testing-strategy.md) - Comprehensive testing approach
- [Code Standards](code-standards.md) - Detailed coding guidelines
- [API Reference](../06-reference/api-reference.md) - Complete API documentation

Thank you for contributing to Jarvis Assistant! Your contributions help make this project better for everyone.