# Code Standards

This document defines the coding standards, conventions, and best practices for Jarvis Assistant development. Following these standards ensures consistency, maintainability, and quality across the codebase.

## Quick Navigation

- [Python Style Guide](#python-style-guide)
- [Code Organization](#code-organization)
- [Documentation Standards](#documentation-standards)
- [Error Handling](#error-handling)
- [Testing Standards](#testing-standards)
- [Security Guidelines](#security-guidelines)
- [Performance Considerations](#performance-considerations)

---

## Python Style Guide

### Code Formatting

We use **Ruff** for code formatting and linting. Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
extend-select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
```

### Type Hints

**Required**: All functions and methods must have type hints.

```python
# Good
def search_documents(
    query: str, 
    limit: int = 10, 
    vault_name: Optional[str] = None
) -> List[SearchResult]:
    """Search documents with semantic similarity."""
    pass

# Bad
def search_documents(query, limit=10, vault_name=None):
    """Search documents with semantic similarity."""
    pass
```

### Import Organization

Use absolute imports and organize them in this order:

```python
# Standard library imports
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party imports
import mcp.types as types
from mcp.server import Server
from pydantic import BaseModel, Field

# Local imports
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.models.search import SearchResult
import logging
```

### Variable Naming

Follow PEP 8 naming conventions:

```python
# Variables and functions: snake_case
user_query = "machine learning"
search_results = []

def process_search_request(query: str) -> List[SearchResult]:
    pass

# Classes: PascalCase
class VectorSearcher:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_SEARCH_RESULTS = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# Private attributes: leading underscore
class DatabaseManager:
    def __init__(self):
        self._connection = None
        self._is_connected = False
```

---

## Code Organization

### Module Structure

Organize code into logical modules with clear responsibilities:

```python
# src/jarvis/services/vector/searcher.py
"""
Vector search service for semantic document retrieval.

This module provides the VectorSearcher class that implements semantic search
functionality using vector embeddings and similarity matching.
"""

import logging
from typing import List, Optional, Dict
from pathlib import Path

from jarvis.models.search import SearchResult, SearchRequest
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.exceptions import VectorSearchError

logger = logging.getLogger(__name__)


class VectorSearcher:
    """Semantic search service using vector embeddings."""
    
    def __init__(
        self, 
        database: VectorDatabase, 
        encoder: VectorEncoder, 
        vaults: Dict[str, Path]
    ):
        """Initialize the vector searcher.
        
        Args:
            database: Vector database instance
            encoder: Vector encoder for query embedding
            vaults: Dictionary mapping vault names to paths
        """
        self.database = database
        self.encoder = encoder
        self.vaults = vaults
        logger.info(f"Initialized VectorSearcher with {len(vaults)} vaults")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        vault_name: Optional[str] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Perform semantic search across indexed documents."""
        # Implementation here
        pass
```

### Class Design Principles

1. **Single Responsibility**: Each class should have one reason to change
2. **Dependency Injection**: Pass dependencies through constructors
3. **Composition over Inheritance**: Prefer composition for flexibility
4. **Immutable Data**: Use immutable data structures where possible

```python
# Good: Single responsibility and dependency injection
class VectorSearcher:
    def __init__(self, database: VectorDatabase, encoder: VectorEncoder):
        self.database = database
        self.encoder = encoder
    
    def search(self, query: str) -> List[SearchResult]:
        # Only handles search logic
        pass

class SearchResultFormatter:
    def format_for_mcp(self, results: List[SearchResult]) -> List[str]:
        # Only handles formatting logic
        pass

# Bad: Multiple responsibilities
class SearchService:
    def __init__(self):
        self.database = VectorDatabase()  # Creates dependency
        self.encoder = VectorEncoder()
    
    def search(self, query: str) -> List[SearchResult]:
        # Handles search logic
        pass
    
    def format_results(self, results: List[SearchResult]) -> List[str]:
        # Also handles formatting - violates SRP
        pass
```

---

## Documentation Standards

### Docstring Format

Use Google-style docstrings for all public functions and classes:

```python
def search_semantic(
    query: str, 
    limit: int = 10, 
    similarity_threshold: Optional[float] = None
) -> List[SearchResult]:
    """Perform semantic search across indexed documents.
    
    This function uses vector embeddings to find documents that are
    semantically similar to the input query, even if they don't contain
    the exact keywords.
    
    Args:
        query: Natural language search query
        limit: Maximum number of results to return (1-50)
        similarity_threshold: Minimum similarity score (0.0-1.0).
            If None, no threshold filtering is applied.
    
    Returns:
        List of SearchResult objects sorted by similarity score in
        descending order. Each result contains the document path,
        similarity score, and metadata.
    
    Raises:
        VectorSearchError: If the search operation fails due to
            database or encoding issues.
        ValueError: If query is empty or parameters are invalid.
    
    Example:
        >>> searcher = VectorSearcher(db, encoder, vaults)
        >>> results = searcher.search("machine learning", limit=5)
        >>> print(f"Found {len(results)} results")
        Found 5 results
        
        >>> # With similarity threshold
        >>> results = searcher.search(
        ...     "deep learning", 
        ...     limit=10, 
        ...     similarity_threshold=0.8
        ... )
        >>> all(r.similarity_score >= 0.8 for r in results)
        True
    """
```

### Module Documentation

Include comprehensive module-level documentation:

```python
"""
Vector search service for semantic document retrieval.

This module provides the core semantic search functionality for Jarvis Assistant.
It uses sentence transformers to generate embeddings and DuckDB for efficient
vector similarity search.

Key Components:
    VectorSearcher: Main search interface
    SearchResult: Result data model
    VectorSearchError: Search-specific exceptions

Usage:
    from jarvis.services.vector.searcher import VectorSearcher
    from jarvis.services.vector.database import VectorDatabase
    from jarvis.services.vector.encoder import VectorEncoder
    
    database = VectorDatabase("./jarvis.duckdb")
    encoder = VectorEncoder()
    searcher = VectorSearcher(database, encoder, vaults)
    
    results = searcher.search("machine learning", limit=10)

Dependencies:
    - DuckDB for vector storage and similarity search
    - sentence-transformers for text encoding
    - Pydantic for data validation

Performance Notes:
    - First search may be slower due to model loading
    - Subsequent searches are cached and faster
    - Similarity threshold filtering improves performance
"""
```

### Code Comments

Use comments sparingly and effectively:

```python
def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
    """Perform semantic search across indexed documents."""
    
    # Validate input parameters
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    
    # Generate query embedding
    # Note: This may take ~100ms for first call due to model loading
    query_embedding = self.encoder.encode(query)
    
    # Search for similar documents
    try:
        results = self.database.search_similar(
            query_embedding, 
            top_k=top_k
        )
    except DatabaseError as e:
        # Convert database-specific errors to search errors
        raise VectorSearchError(f"Search failed: {str(e)}") from e
    
    # Convert database results to SearchResult objects
    return [
        SearchResult(
            path=result.path,
            similarity_score=result.score,
            vault_name=result.vault_name
        )
        for result in results
    ]
```

---

## Error Handling

### Exception Hierarchy

Define a clear exception hierarchy:

```python
# src/jarvis/utils/exceptions.py
"""Custom exceptions for Jarvis Assistant."""

class JarvisError(Exception):
    """Base exception for all Jarvis Assistant errors."""
    pass

class ConfigurationError(JarvisError):
    """Raised when configuration is invalid."""
    pass

class DatabaseError(JarvisError):
    """Base class for database-related errors."""
    pass

class VectorDatabaseError(DatabaseError):
    """Raised when vector database operations fail."""
    pass

class GraphDatabaseError(DatabaseError):
    """Raised when graph database operations fail."""
    pass

class SearchError(JarvisError):
    """Base class for search-related errors."""
    pass

class VectorSearchError(SearchError):
    """Raised when vector search operations fail."""
    pass

class VaultError(JarvisError):
    """Raised when vault operations fail."""
    pass

class MCPError(JarvisError):
    """Raised when MCP operations fail."""
    pass
```

### Error Handling Patterns

Use consistent error handling patterns:

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def search_with_fallback(
    query: str, 
    primary_searcher: VectorSearcher,
    fallback_searcher: Optional[VaultReader] = None
) -> List[SearchResult]:
    """Search with fallback to traditional search if semantic search fails."""
    
    try:
        # Primary search attempt
        results = primary_searcher.search(query)
        logger.info(f"Semantic search returned {len(results)} results")
        return results
        
    except VectorSearchError as e:
        # Log the specific error for debugging
        logger.warning(f"Semantic search failed: {e}")
        
        # Try fallback if available
        if fallback_searcher:
            try:
                logger.info("Attempting fallback to traditional search")
                fallback_results = fallback_searcher.search_vault(query, search_content=True)
                
                # Convert to SearchResult format
                return [
                    SearchResult(
                        path=result["path"],
                        similarity_score=0.5,  # Default score for traditional search
                        vault_name="default"
                    )
                    for result in fallback_results
                ]
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                # Chain the original exception
                raise VectorSearchError(
                    f"Both semantic and fallback search failed: {str(e)}"
                ) from e
        else:
            # No fallback available, re-raise original error
            raise
    
    except Exception as e:
        # Catch unexpected errors and convert to known exception type
        logger.error(f"Unexpected error in search: {e}", exc_info=True)
        raise VectorSearchError(f"Search failed due to unexpected error: {str(e)}") from e
```

### Logging Standards

Use structured logging consistently:

```python
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class VectorSearcher:
    def __init__(self, database: VectorDatabase, encoder: VectorEncoder):
        self.database = database
        self.encoder = encoder
        logger.info("VectorSearcher initialized", extra={
            "component": "vector_searcher",
            "database_type": type(database).__name__,
            "encoder_model": encoder.model_name
        })
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform semantic search with comprehensive logging."""
        
        # Log search request
        logger.info("Starting semantic search", extra={
            "query_length": len(query),
            "top_k": top_k,
            "operation": "semantic_search"
        })
        
        try:
            # Generate embedding
            start_time = time.time()
            query_embedding = self.encoder.encode(query)
            encoding_time = time.time() - start_time
            
            logger.debug("Query encoded", extra={
                "encoding_time_ms": round(encoding_time * 1000, 2),
                "embedding_dimension": len(query_embedding)
            })
            
            # Perform search
            start_time = time.time()
            results = self.database.search_similar(query_embedding, top_k=top_k)
            search_time = time.time() - start_time
            
            logger.info("Search completed", extra={
                "results_count": len(results),
                "search_time_ms": round(search_time * 1000, 2),
                "min_score": min(r.similarity_score for r in results) if results else 0,
                "max_score": max(r.similarity_score for r in results) if results else 0
            })
            
            return results
            
        except Exception as e:
            logger.error("Search failed", extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query_length": len(query),
                "top_k": top_k
            }, exc_info=True)
            raise
```

---

## Testing Standards

### Test Structure

Organize tests to mirror the source code structure:

```python
# resources/tests/unit/services/vector/test_searcher.py
import pytest
from unittest.mock import Mock, patch
from jarvis.services.vector.searcher import VectorSearcher
from jarvis.models.search import SearchResult
from jarvis.utils.exceptions import VectorSearchError

class TestVectorSearcher:
    """Test suite for VectorSearcher class."""
    
    def test_search_basic_functionality(self):
        """Test basic search functionality with valid inputs."""
        # Arrange
        mock_database = Mock()
        mock_encoder = Mock()
        mock_database.search_similar.return_value = [
            Mock(path="test.md", similarity_score=0.95, vault_name="test")
        ]
        mock_encoder.encode.return_value = [0.1, 0.2, 0.3]
        
        searcher = VectorSearcher(mock_database, mock_encoder, {})
        
        # Act
        results = searcher.search("test query", top_k=5)
        
        # Assert
        assert len(results) == 1
        assert results[0].path == "test.md"
        assert results[0].similarity_score == 0.95
        mock_encoder.encode.assert_called_once_with("test query")
        mock_database.search_similar.assert_called_once()
    
    def test_search_empty_query_raises_error(self):
        """Test that empty query raises appropriate error."""
        # Arrange
        searcher = VectorSearcher(Mock(), Mock(), {})
        
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            searcher.search("", top_k=5)
    
    def test_search_invalid_top_k_raises_error(self):
        """Test that invalid top_k values raise appropriate errors."""
        # Arrange
        searcher = VectorSearcher(Mock(), Mock(), {})
        
        # Act & Assert
        with pytest.raises(ValueError, match="top_k must be between 1 and 50"):
            searcher.search("test", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be between 1 and 50"):
            searcher.search("test", top_k=51)
    
    def test_search_database_error_handling(self):
        """Test proper handling of database errors."""
        # Arrange
        mock_database = Mock()
        mock_encoder = Mock()
        mock_database.search_similar.side_effect = Exception("Database error")
        mock_encoder.encode.return_value = [0.1, 0.2, 0.3]
        
        searcher = VectorSearcher(mock_database, mock_encoder, {})
        
        # Act & Assert
        with pytest.raises(VectorSearchError, match="Search failed"):
            searcher.search("test query", top_k=5)
    
    @patch('jarvis.services.vector.searcher.logger')
    def test_search_logging(self, mock_logger):
        """Test that search operations are properly logged."""
        # Arrange
        mock_database = Mock()
        mock_encoder = Mock()
        mock_database.search_similar.return_value = []
        mock_encoder.encode.return_value = [0.1, 0.2, 0.3]
        
        searcher = VectorSearcher(mock_database, mock_encoder, {})
        
        # Act
        searcher.search("test query", top_k=5)
        
        # Assert
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()
```

### Test Data Management

Create reusable test fixtures:

```python
# resources/tests/fixtures/search_fixtures.py
import pytest
from pathlib import Path
from jarvis.models.search import SearchResult

@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        SearchResult(
            path="ai_concepts.md",
            similarity_score=0.95,
            vault_name="test_vault",
            content_preview="Machine learning is a subset of artificial intelligence..."
        ),
        SearchResult(
            path="programming.md",
            similarity_score=0.87,
            vault_name="test_vault",
            content_preview="Python is a versatile programming language..."
        ),
        SearchResult(
            path="data_science.md",
            similarity_score=0.82,
            vault_name="test_vault",
            content_preview="Data science combines statistics and programming..."
        )
    ]

@pytest.fixture
def test_vault_with_content(tmp_path):
    """Create a temporary vault with test content."""
    vault_path = tmp_path / "test_vault"
    vault_path.mkdir()
    
    # Create test files
    (vault_path / "ai_concepts.md").write_text("""
# AI Concepts

Machine learning is a subset of artificial intelligence that enables
computers to learn and improve from experience without being explicitly
programmed.
""")
    
    (vault_path / "programming.md").write_text("""
# Programming

Python is a versatile programming language that is widely used in
data science, web development, and artificial intelligence.
""")
    
    return vault_path
```

---

## Security Guidelines

### Input Validation

Always validate and sanitize inputs:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class SearchRequest(BaseModel):
    """Validated search request model."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    vault_name: Optional[str] = Field(None, max_length=100, description="Vault name")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query string."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        
        # Remove potentially dangerous characters
        cleaned = v.strip()
        if len(cleaned) != len(v):
            raise ValueError("Query contains invalid whitespace")
        
        return cleaned
    
    @field_validator('vault_name')
    @classmethod
    def validate_vault_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate vault name format."""
        if v is None:
            return v
        
        # Only allow alphanumeric characters, hyphens, and underscores
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Vault name contains invalid characters")
        
        return v

# Usage in MCP tools
def handle_search_request(arguments: dict) -> List[SearchResult]:
    """Handle search request with validation."""
    try:
        # Validate input using Pydantic
        request = SearchRequest(**arguments)
        
        # Proceed with validated data
        return searcher.search(
            query=request.query,
            top_k=request.limit,
            vault_name=request.vault_name,
            similarity_threshold=request.similarity_threshold
        )
    except ValidationError as e:
        logger.warning(f"Invalid search request: {e}")
        raise ValueError(f"Invalid request: {e}")
```

### Path Security

Prevent path traversal attacks:

```python
import os
from pathlib import Path

def safe_path_join(base_path: Path, user_path: str) -> Path:
    """Safely join paths preventing directory traversal."""
    
    # Normalize the user path
    user_path = os.path.normpath(user_path)
    
    # Remove leading path separators
    user_path = user_path.lstrip(os.sep)
    
    # Join with base path
    full_path = base_path / user_path
    
    # Resolve to absolute path
    full_path = full_path.resolve()
    base_path = base_path.resolve()
    
    # Ensure the result is within the base path
    if not str(full_path).startswith(str(base_path)):
        raise ValueError(f"Path traversal attempted: {user_path}")
    
    return full_path

# Usage in vault operations
def read_vault_file(vault_path: Path, requested_path: str) -> str:
    """Read file from vault with path validation."""
    try:
        safe_path = safe_path_join(vault_path, requested_path)
        
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {requested_path}")
        
        if not safe_path.is_file():
            raise ValueError(f"Not a file: {requested_path}")
        
        return safe_path.read_text(encoding='utf-8')
        
    except Exception as e:
        logger.warning(f"Unsafe file access attempt: {requested_path}")
        raise ValueError(f"Cannot read file: {str(e)}")
```

### Secrets Management

Never hardcode secrets or expose them in logs:

```python
import os
from typing import Optional

class DatabaseConfig:
    """Database configuration with secure credential handling."""
    
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not self.neo4j_password:
            raise ValueError("NEO4J_PASSWORD environment variable is required")
    
    def get_connection_string(self) -> str:
        """Get connection string without exposing credentials."""
        return f"{self.neo4j_uri} (user: {self.neo4j_user})"
    
    def __repr__(self) -> str:
        """String representation without secrets."""
        return f"DatabaseConfig(uri={self.neo4j_uri}, user={self.neo4j_user})"

# Logging without secrets
logger.info(f"Connecting to database: {config.get_connection_string()}")
# NOT: logger.info(f"Connecting with password: {config.neo4j_password}")
```

---

## Performance Considerations

### Caching Strategies

Implement appropriate caching for performance:

```python
import functools
import time
from typing import Dict, Any

class VectorSearcher:
    def __init__(self, database: VectorDatabase, encoder: VectorEncoder):
        self.database = database
        self.encoder = encoder
        self._query_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
    
    @functools.lru_cache(maxsize=128)
    def _encode_query(self, query: str) -> list:
        """Cache query encodings to avoid re-encoding identical queries."""
        return self.encoder.encode(query)
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search with caching for performance."""
        
        # Create cache key
        cache_key = f"{query}:{top_k}"
        
        # Check cache
        if cache_key in self._query_cache:
            cached_result, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        # Perform search
        query_embedding = self._encode_query(query)
        results = self.database.search_similar(query_embedding, top_k=top_k)
        
        # Cache results
        self._query_cache[cache_key] = (results, time.time())
        
        # Clean old cache entries periodically
        if len(self._query_cache) > 1000:
            self._cleanup_cache()
        
        return results
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._query_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            del self._query_cache[key]
```

### Resource Management

Properly manage resources and connections:

```python
import contextlib
from typing import Generator

class DatabaseManager:
    """Database manager with proper resource management."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None
    
    @contextlib.contextmanager
    def get_connection(self) -> Generator[DatabaseConnection, None, None]:
        """Context manager for database connections."""
        connection = None
        try:
            connection = self._create_connection()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
    
    def _create_connection(self) -> DatabaseConnection:
        """Create database connection."""
        return DatabaseConnection(self.db_path)

# Usage
db_manager = DatabaseManager("./jarvis.duckdb")

with db_manager.get_connection() as conn:
    results = conn.execute("SELECT * FROM documents WHERE vault = ?", ("test",))
```

### Memory Usage Optimization

Monitor and optimize memory usage:

```python
import gc
import psutil
from typing import Iterator

def process_large_dataset(data_source: Iterator[dict]) -> Iterator[SearchResult]:
    """Process large datasets with memory optimization."""
    
    batch_size = 1000
    batch = []
    
    for item in data_source:
        batch.append(item)
        
        if len(batch) >= batch_size:
            # Process batch
            yield from process_batch(batch)
            
            # Clear batch and force garbage collection
            batch.clear()
            gc.collect()
            
            # Monitor memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent}%")
    
    # Process remaining items
    if batch:
        yield from process_batch(batch)
```

---

## Tools and Automation

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, types-requests]
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

### Development Scripts

```bash
#!/bin/bash
# scripts/quality-check.sh

set -e

echo "Running code quality checks..."

# Format code
echo "Formatting code with ruff..."
uv run ruff format src/

# Lint code
echo "Linting code with ruff..."
uv run ruff check src/ --fix

# Type checking
echo "Running type checks with mypy..."
uv run mypy src/

# Run tests
echo "Running test suite..."
uv run pytest resources/tests/ --cov=src/jarvis --cov-report=html

echo "All quality checks passed!"
```

---

## Next Steps

- [Testing Strategy](testing-strategy.md) - Comprehensive testing approach
- [Contribution Guide](contribution-guide.md) - How to contribute to the project
- [Developer Guide](developer-guide.md) - Development setup and workflow
- [API Reference](../06-reference/api-reference.md) - Complete API documentation
