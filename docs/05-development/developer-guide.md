# Developer Guide

This guide provides comprehensive information for developers working on Jarvis Assistant. It covers the development workflow, architecture understanding, and contribution processes.

## Quick Navigation

- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Development Workflow](#development-workflow)
- [Testing Strategy](#testing-strategy)
- [Code Standards](#code-standards)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)

---

## Development Setup

### Prerequisites

- **Python 3.11+** with UV package manager
- **Git** for version control
- **IDE** with Python support (VS Code, PyCharm, etc.)
- **Neo4j Database** (optional, for graph functionality)
- **Obsidian vault** for testing

### Local Development Environment

```bash
# Clone the repository
git clone https://github.com/your-username/jarvis-assistant.git
cd jarvis-assistant

# Install dependencies using UV
uv sync

# Verify installation
uv run jarvis --help
```

### Environment Configuration

Jarvis Assistant uses Pydantic Settings with optional .env file support. Configuration can be set via:
1. **Environment variables** (with `JARVIS_` prefix)
2. **Optional config/.env file** in project root
3. **Command-line arguments** for vault path

Copy the example configuration:
```bash
cp config/.env.example config/.env
```

Edit `config/.env` for your development setup:

```env
# Development Settings
JARVIS_LOG_LEVEL=DEBUG
JARVIS_VAULT_PATH=./resources/test_vault
JARVIS_VECTOR_DB_PATH=./data/jarvis_dev.duckdb

# Optional: Neo4j Configuration (for graph features)
JARVIS_NEO4J_URI=bolt://localhost:7687
JARVIS_NEO4J_USER=neo4j
JARVIS_NEO4J_PASSWORD=your_password

# Optional: Enable advanced features
JARVIS_USE_DEPENDENCY_INJECTION=true
JARVIS_ANALYTICS_ENABLED=true
```

**Note**: Neo4j is optional - the system works without graph features.

### Development Dependencies

All dependencies are declared in `pyproject.toml` and installed with:

```bash
# Install all dependencies (including dev tools)
uv sync

# Verify development tools are available
uv run ruff --version
uv run pytest --version
uv run mypy --version

# Verify pytest-cov is installed (required for coverage reporting)
# Unix/macOS/Linux:
uv pip list | grep pytest-cov
# Windows PowerShell:
uv pip list | findstr pytest
# Should show: pytest-cov  6.2.1 (or similar version)
```

**Core dependencies are automatically installed:**
- MCP server framework (`mcp`)
- Vector search (`duckdb`, `sentence-transformers`, `torch`)
- Optional graph database (`neo4j-driver`)
- Development tools (`pytest`, `ruff`, `mypy`)

---

## Project Architecture

### Core Components

```
src/jarvis/
├── mcp/                   # MCP server implementation
│   ├── server.py         # Main MCP server
│   ├── tools/            # MCP tool implementations
│   └── resources/        # MCP resource handlers
├── services/             # Business logic services
│   ├── vector/          # Semantic search (DuckDB)
│   ├── graph/           # Graph operations (Neo4j)
│   └── vault/           # Obsidian file management
├── database/            # Database adapters
├── models/              # Data models and schemas
└── utils/               # Utilities and configuration
```

### Key Service Interactions

```python
# Example: How components interact
vault_reader = VaultReader(vault_path)
indexer = VectorIndexer(database, encoder)
searcher = VectorSearcher(database, encoder, vaults)

# MCP server coordinates all services
context = MCPServerContext(vaults, database_path)
server = create_mcp_server(vaults, database_path)
```

### Data Flow

1. **Indexing**: `VaultReader` → `VectorIndexer` → `DuckDB`
2. **Graph Building**: `VaultReader` → `GraphIndexer` → `Neo4j`
3. **Search**: `MCP Client` → `MCP Server` → `VectorSearcher` → `DuckDB`
4. **Graph Query**: `MCP Client` → `MCP Server` → `GraphDatabase` → `Neo4j`

---

## Development Workflow

### Feature Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/semantic-search-enhancement
   ```

2. **Implement Changes**
   - Write code following [code standards](#code-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Locally**
   ```bash
   # Run tests
   uv run pytest resources/tests/

   # Run linting
   uv run ruff check src/

   # Test MCP server
   uv run jarvis mcp --vault ./resources/test_vault
   ```

4. **Submit Pull Request**
   - Follow [contribution guidelines](contribution-guide.md)
   - Include test coverage
   - Update relevant documentation

### Local Testing Setup

```bash
# Set up test environment (via .env or environment variables)
export JARVIS_VAULT_PATH="./resources/test_vault"
export JARVIS_VECTOR_DB_PATH="./data/test.duckdb"
export JARVIS_LOG_LEVEL="DEBUG"

# Create test vault if needed
mkdir -p ./resources/test_vault
echo "# Test Note" > ./resources/test_vault/test.md

# Index test data
uv run jarvis index --vault ./resources/test_vault

# Start MCP server for testing
uv run jarvis mcp --vault ./resources/test_vault --watch
```

### Development Commands

```bash
# Core development commands
uv run jarvis --help                 # Show all commands
uv run jarvis index --vault /path    # Index vault for development
uv run jarvis mcp --vault /path      # Start MCP server
uv run jarvis graph-index --vault /path  # Index graph data

# Quality assurance
uv run ruff check src/               # Lint code
uv run ruff format src/              # Format code
uv run mypy src/                     # Type checking
uv run pytest resources/tests/       # Run test suite
```

---

## Testing Strategy

### Test Structure

```
resources/tests/
├── unit/                # Unit tests for individual components
├── integration/         # Integration tests for service interactions
├── mcp/                # MCP server and tool tests
└── fixtures/           # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
uv run pytest resources/tests/

# Run specific test categories
uv run pytest resources/tests/unit/
uv run pytest resources/tests/integration/
uv run pytest resources/tests/mcp/

# Run with coverage
uv run pytest --cov=src/jarvis resources/tests/

# Run specific test file
uv run pytest resources/tests/unit/test_vector_search.py
```

### Writing Tests

```python
# Example unit test
import pytest
from jarvis.services.vector.searcher import VectorSearcher

class TestVectorSearcher:
    def test_search_basic_functionality(self):
        # Arrange
        searcher = VectorSearcher(mock_database, mock_encoder, mock_vaults)
        
        # Act
        results = searcher.search("test query", top_k=5)
        
        # Assert
        assert len(results) <= 5
        assert all(result.similarity_score > 0 for result in results)

# Example integration test
import pytest
from jarvis.mcp.server import create_mcp_server

@pytest.mark.asyncio
async def test_mcp_semantic_search():
    # Arrange
    server = create_mcp_server(test_vaults, test_db_path)
    
    # Act
    results = await server.call_tool("search-semantic", {"query": "test"})
    
    # Assert
    assert len(results) > 0
    assert results[0].type == "text"
```

### Test Data Management

```python
# Use fixtures for consistent test data
@pytest.fixture
def test_vault_path():
    return Path("./resources/test_vault")

@pytest.fixture
def mock_database():
    return MockVectorDatabase()

@pytest.fixture
def sample_search_results():
    return [
        SearchResult(path="test.md", similarity_score=0.85, vault_name="test"),
        SearchResult(path="example.md", similarity_score=0.72, vault_name="test")
    ]
```

---

## Code Standards

### Python Code Style

```python
# Follow PEP 8 with these specific guidelines:

# 1. Use type hints
def search_documents(query: str, limit: int = 10) -> List[SearchResult]:
    """Search documents with semantic similarity."""
    pass

# 2. Use Pydantic models for data validation
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    vault_name: Optional[str] = None

# 3. Comprehensive error handling
try:
    results = searcher.search(query, limit)
except VectorSearchError as e:
    logger.error(f"Search failed: {e}")
    raise MCPToolError(f"Search operation failed: {str(e)}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise MCPToolError("An unexpected error occurred")

# 4. Logging for debugging
import logging
logger = logging.getLogger(__name__)

def process_search_request(request: SearchRequest) -> SearchResponse:
    logger.info(f"Processing search request: {request.query}")
    try:
        results = searcher.search(request.query, request.limit)
        logger.debug(f"Found {len(results)} results")
        return SearchResponse(results=results)
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise
```

### Documentation Standards

```python
def search_semantic(
    query: str, 
    limit: int = 10, 
    similarity_threshold: Optional[float] = None
) -> List[SearchResult]:
    """Perform semantic search across indexed documents.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return (1-50)
        similarity_threshold: Minimum similarity score (0.0-1.0)
        
    Returns:
        List of SearchResult objects sorted by similarity score
        
    Raises:
        VectorSearchError: If search operation fails
        ValueError: If parameters are invalid
        
    Example:
        >>> searcher = VectorSearcher(db, encoder, vaults)
        >>> results = searcher.search("machine learning", limit=5)
        >>> print(f"Found {len(results)} results")
    """
```

### Error Handling Patterns

```python
# Custom exception hierarchy
class JarvisError(Exception):
    """Base exception for Jarvis Assistant."""
    pass

class VectorSearchError(JarvisError):
    """Raised when vector search operations fail."""
    pass

class GraphSearchError(JarvisError):
    """Raised when graph search operations fail."""
    pass

class MCPToolError(JarvisError):
    """Raised when MCP tool execution fails."""
    pass

# Graceful degradation
def get_search_results(query: str) -> List[SearchResult]:
    try:
        return semantic_searcher.search(query)
    except VectorSearchError as e:
        logger.warning(f"Semantic search failed: {e}")
        logger.info("Falling back to traditional search")
        return traditional_searcher.search(query)
    except Exception as e:
        logger.error(f"All search methods failed: {e}")
        return []  # Return empty results rather than crashing
```

---

## Debugging and Troubleshooting

### Logging Configuration

```python
# Enable debug logging during development
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component-specific logging
logger = logging.getLogger('jarvis.services.vector.searcher')
logger.setLevel(logging.DEBUG)
```

### Common Development Issues

#### 1. Pytest Coverage Errors

If you see `pytest: error: unrecognized arguments: --cov=jarvis`, pytest-cov is missing:

**Unix/macOS/Linux:**
```bash
# Fix: Install dev dependencies
uv sync

# Or install pytest-cov specifically  
uv add pytest-cov

# Verify installation
uv pip list | grep pytest-cov

# Alternative: Run tests without coverage
uv run pytest resources/tests/ --no-cov
```

**Windows (PowerShell):**
```powershell
# Fix: Install dev dependencies
uv sync

# Or force reinstall if needed
uv sync --reinstall

# Verify installation
uv pip list | findstr pytest

# Alternative: Run tests without coverage
uv run pytest resources/tests/ --override-ini="addopts="
```

#### 2. Database Connection Problems

```python
# Debug database connections
from jarvis.services.vector.database import VectorDatabase

try:
    db = VectorDatabase("./data/jarvis.duckdb")
    logger.info("Database connection successful")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    # Check file permissions, path existence, etc.
```

#### 2. MCP Server Issues

```python
# Debug MCP server startup
import asyncio
from jarvis.mcp.server import run_mcp_server

async def debug_mcp_server():
    try:
        await run_mcp_server(vaults, db_path)
    except Exception as e:
        logger.error(f"MCP server failed: {e}", exc_info=True)

# Run with debug logging
asyncio.run(debug_mcp_server())
```

#### 3. Graph Search Problems

```python
# Debug Neo4j connections
from jarvis.services.graph.database import GraphDatabase

try:
    graph_db = GraphDatabase(uri, user, password)
    graph_db.check_connection()
    logger.info("Neo4j connection successful")
except Exception as e:
    logger.error(f"Neo4j connection failed: {e}")
    # Check Neo4j server status, credentials, etc.
```

### Performance Debugging

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@timing_decorator
def search_documents(query: str) -> List[SearchResult]:
    # Your search implementation
    pass
```

### Memory Usage Monitoring

```python
import psutil
import os

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.debug(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Call periodically during development
log_memory_usage()
```

---

## Development Tools and IDE Setup

### VS Code Configuration

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["resources/tests/"]
}
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

### Useful Development Scripts

```bash
# scripts/dev-setup.sh
#!/bin/bash
set -e

echo "Setting up development environment..."

# Install dependencies
uv sync

# Set up test data
mkdir -p ./resources/test_vault
echo "# Test Note" > ./resources/test_vault/test.md

# Create database directory
mkdir -p ./data

# Run initial indexing
uv run jarvis index --vault ./resources/test_vault

echo "Development setup complete!"
```

---

## Next Steps

- [Testing Strategy](testing-strategy.md) - Comprehensive testing approach
- [Code Standards](code-standards.md) - Detailed coding guidelines
- [Contribution Guide](contribution-guide.md) - How to contribute to the project
- [Architecture Reference](../02-architecture/component-interaction.md) - System design details
