# Testing Strategy

This document outlines the comprehensive testing approach for Jarvis Assistant, covering unit tests, integration tests, MCP server validation, and quality assurance processes.

## Quick Navigation

- [Testing Philosophy](#testing-philosophy)
- [Test Structure](#test-structure)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [MCP Server Testing](#mcp-server-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Performance Testing](#performance-testing)
- [Test Data Management](#test-data-management)

---

## Testing Philosophy

### Core Principles

1. **Reliability First**: MCP tools must work consistently with Claude Desktop
2. **Graceful Degradation**: Services should handle failures without crashing
3. **Performance Awareness**: Tests should catch performance regressions
4. **Real-World Scenarios**: Tests should reflect actual usage patterns

### Test Categories

```
Unit Tests (60%)          - Individual component testing
Integration Tests (25%)   - Service interaction testing
MCP Server Tests (10%)    - End-to-end MCP functionality
Performance Tests (5%)    - Load and performance validation
Contract Tests (*)        - Interface/implementation validation
```

**Contract Tests** validate that implementations correctly match their interface definitions, preventing parameter mismatches and signature inconsistencies that could cause runtime errors.

---

## Test Structure

### Directory Organization

```
resources/tests/
├── unit/                    # Unit tests
│   ├── services/
│   │   ├── test_vector_search.py
│   │   ├── test_graph_search.py
│   │   ├── test_vault_reader.py
│   │   ├── test_health_checker.py
│   │   ├── test_result_ranker.py
│   │   └── test_mcp_cache.py
│   ├── mcp/
│   │   ├── test_server.py
│   │   └── test_tools.py
│   ├── test_interface_contracts.py  # Interface validation tests
│   └── utils/
│       ├── test_config.py
│       └── test_logging.py
├── integration/             # Integration tests
│   ├── test_search_flow.py
│   └── test_database_integration.py
├── mcp/                     # MCP-specific tests
│   ├── test_mcp_integration.py
│   ├── test_claude_integration.py
│   └── test_tool_schemas.py
├── performance/             # Performance tests
│   ├── test_search_performance.py
│   └── test_concurrent_access.py
├── fixtures/                # Test data and fixtures
│   ├── test_vault/
│   ├── mock_data.py
│   └── test_databases.py
└── conftest.py             # Pytest configuration
```

### Test Configuration

```python
# resources/tests/conftest.py
import pytest
from pathlib import Path
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vault.reader import VaultReader

@pytest.fixture(scope="session")
def test_vault_path():
    """Path to test vault with sample data."""
    return Path(__file__).parent / "fixtures" / "test_vault"

@pytest.fixture(scope="session")
def test_database_path(tmp_path_factory):
    """Temporary database for testing."""
    db_path = tmp_path_factory.mktemp("test_db") / "test.duckdb"
    return db_path

@pytest.fixture
def vector_database(test_database_path):
    """Vector database instance for testing."""
    db = VectorDatabase(test_database_path)
    yield db
    db.close()

@pytest.fixture
def vector_encoder():
    """Vector encoder instance for testing."""
    return VectorEncoder()

@pytest.fixture
def vault_reader(test_vault_path):
    """Vault reader instance for testing."""
    return VaultReader(str(test_vault_path))
```

---

## Unit Testing

### Vector Search Service Tests

```python
# resources/tests/unit/services/test_vector_search.py
import pytest
from unittest.mock import Mock, patch
from jarvis.services.vector.searcher import VectorSearcher
from jarvis.models.search import SearchResult

class TestVectorSearcher:
    
    def test_search_basic_functionality(self, vector_database, vector_encoder):
        """Test basic search functionality."""
        # Arrange
        vaults = {"test": Path("./test_vault")}
        searcher = VectorSearcher(vector_database, vector_encoder, vaults)
        
        # Act
        results = searcher.search("test query", top_k=5)
        
        # Assert
        assert isinstance(results, list)
        assert len(results) <= 5
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_with_similarity_threshold(self, vector_database, vector_encoder):
        """Test search with similarity threshold filtering."""
        # Arrange
        vaults = {"test": Path("./test_vault")}
        searcher = VectorSearcher(vector_database, vector_encoder, vaults)
        
        # Act
        results = searcher.search("test query", top_k=10, similarity_threshold=0.8)
        
        # Assert
        assert all(r.similarity_score >= 0.8 for r in results)
    
    def test_search_vault_specific(self, vector_database, vector_encoder):
        """Test vault-specific search functionality."""
        # Arrange
        vaults = {"vault1": Path("./vault1"), "vault2": Path("./vault2")}
        searcher = VectorSearcher(vector_database, vector_encoder, vaults)
        
        # Act
        results = searcher.search("test query", vault_name="vault1")
        
        # Assert
        assert all(r.vault_name == "vault1" for r in results)
    
    def test_search_empty_query(self, vector_database, vector_encoder):
        """Test search with empty query."""
        # Arrange
        vaults = {"test": Path("./test_vault")}
        searcher = VectorSearcher(vector_database, vector_encoder, vaults)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Query cannot be empty"):
            searcher.search("", top_k=5)
    
    def test_search_invalid_vault(self, vector_database, vector_encoder):
        """Test search with invalid vault name."""
        # Arrange
        vaults = {"test": Path("./test_vault")}
        searcher = VectorSearcher(vector_database, vector_encoder, vaults)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unknown vault"):
            searcher.search("test query", vault_name="nonexistent")
    
    @patch('jarvis.services.vector.searcher.logger')
    def test_search_database_error(self, mock_logger, vector_encoder):
        """Test search handles database errors gracefully."""
        # Arrange
        mock_database = Mock()
        mock_database.search_similar.side_effect = Exception("Database error")
        vaults = {"test": Path("./test_vault")}
        searcher = VectorSearcher(mock_database, vector_encoder, vaults)
        
        # Act & Assert
        with pytest.raises(Exception):
            searcher.search("test query")
        mock_logger.error.assert_called()
```

### Graph Search Service Tests

```python
# resources/tests/unit/services/test_graph_search.py
import pytest
from unittest.mock import Mock, patch
from jarvis.services.graph.database import GraphDatabase
from jarvis.models.graph import GraphResult

class TestGraphDatabase:
    
    def test_get_note_graph_basic(self):
        """Test basic graph retrieval functionality."""
        # Arrange
        with patch('jarvis.services.graph.database.GraphDatabase._create_driver'):
            graph_db = GraphDatabase("bolt://localhost:7687", "neo4j", "password")
            
            # Mock session and results
            mock_session = Mock()
            mock_result = Mock()
            mock_result.data.return_value = [
                {"n": {"path": "test.md", "title": "Test"}},
                {"r": {"type": "LINKS_TO"}}
            ]
            mock_session.run.return_value = mock_result
            graph_db._driver.session.return_value.__enter__.return_value = mock_session
            
            # Act
            result = graph_db.get_note_graph("test.md", depth=1)
            
            # Assert
            assert isinstance(result, dict)
            assert "nodes" in result
            assert "relationships" in result
    
    def test_connection_check(self):
        """Test database connection verification."""
        # Arrange
        with patch('jarvis.services.graph.database.GraphDatabase._create_driver'):
            graph_db = GraphDatabase("bolt://localhost:7687", "neo4j", "password")
            mock_driver = Mock()
            graph_db._driver = mock_driver
            
            # Act
            graph_db.check_connection()
            
            # Assert
            mock_driver.verify_connectivity.assert_called_once()
    
    def test_connection_failure(self):
        """Test handling of connection failures."""
        # Arrange
        with patch('jarvis.services.graph.database.GraphDatabase._create_driver') as mock_create:
            mock_create.side_effect = Exception("Connection failed")
            
            # Act & Assert
            with pytest.raises(Exception, match="Connection failed"):
                GraphDatabase("bolt://localhost:7687", "neo4j", "password")
```

### Vault Reader Tests

```python
# resources/tests/unit/services/test_vault_reader.py
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from jarvis.services.vault.reader import VaultReader

class TestVaultReader:
    
    def test_read_file_success(self, test_vault_path):
        """Test successful file reading."""
        # Arrange
        reader = VaultReader(str(test_vault_path))
        
        # Create test file
        test_file = test_vault_path / "test.md"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# Test Content\n\nThis is a test file.")
        
        # Act
        content, metadata = reader.read_file("test.md")
        
        # Assert
        assert content == "# Test Content\n\nThis is a test file."
        assert metadata["path"] == "test.md"
        assert "size" in metadata
        assert "modified" in metadata
    
    def test_read_file_not_found(self, test_vault_path):
        """Test reading non-existent file."""
        # Arrange
        reader = VaultReader(str(test_vault_path))
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            reader.read_file("nonexistent.md")
    
    def test_search_vault_filename(self, test_vault_path):
        """Test vault search by filename."""
        # Arrange
        reader = VaultReader(str(test_vault_path))
        
        # Create test files
        (test_vault_path / "meeting_notes.md").write_text("Meeting content")
        (test_vault_path / "project_meeting.md").write_text("Project content")
        
        # Act
        results = reader.search_vault("meeting", search_content=False)
        
        # Assert
        assert len(results) == 2
        assert all("meeting" in r["path"].lower() for r in results)
    
    def test_search_vault_content(self, test_vault_path):
        """Test vault search by content."""
        # Arrange
        reader = VaultReader(str(test_vault_path))
        
        # Create test files
        (test_vault_path / "file1.md").write_text("This contains machine learning")
        (test_vault_path / "file2.md").write_text("This has neural networks")
        
        # Act
        results = reader.search_vault("machine learning", search_content=True)
        
        # Assert
        assert len(results) == 1
        assert results[0]["path"] == "file1.md"
        assert "content_preview" in results[0]
```

---

## Interface Contract Testing

### Purpose

Interface contract tests validate that implementations correctly match their interface definitions, preventing parameter mismatches and signature inconsistencies that could cause runtime errors.

### Key Benefits

- **Early Detection**: Catch interface/implementation mismatches before deployment
- **Parameter Validation**: Ensure method signatures match between interfaces and implementations
- **Type Safety**: Validate return types and parameter types are consistent
- **Regression Prevention**: Prevent interface drift during development

### Contract Test Structure

```python
# resources/tests/unit/test_interface_contracts.py
import pytest
import inspect
from jarvis.core.interfaces import IVaultReader, IVectorSearcher
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vector.searcher import VectorSearcher

class TestInterfaceContracts:
    """Test that implementations match their interface contracts."""
    
    @pytest.mark.parametrize("interface_cls,implementation_cls", [
        (IVaultReader, VaultReader),
        (IVectorSearcher, VectorSearcher),
    ])
    def test_method_signatures_match(self, interface_cls, implementation_cls):
        """Test that interface and implementation method signatures match."""
        interface_methods = self._get_abstract_methods(interface_cls)
        
        for method_name in interface_methods:
            # Get signatures
            interface_method = getattr(interface_cls, method_name)
            impl_method = getattr(implementation_cls, method_name)
            
            interface_sig = inspect.signature(interface_method)
            impl_sig = inspect.signature(impl_method)
            
            # Compare signatures
            self._compare_signatures(
                interface_sig, impl_sig, 
                interface_cls.__name__, implementation_cls.__name__, 
                method_name
            )
    
    def test_critical_search_vault_contract(self):
        """Specific validation for search_vault method parameters."""
        sig = inspect.signature(VaultReader.search_vault)
        params = list(sig.parameters.keys())
        
        # Validate required parameters
        assert 'query' in params
        assert 'search_content' in params  
        assert 'limit' in params
        
        # Ensure old parameter names are not present
        assert 'max_results' not in params
        assert 'file_path' not in params
```

### Running Contract Tests

```bash
# Run all contract validation tests
uv run pytest resources/tests/unit/test_interface_contracts.py -v

# Run specific interface validation
uv run pytest resources/tests/unit/test_interface_contracts.py -k "search_vault" -v

# Include in CI/CD pipeline
uv run pytest resources/tests/unit/test_interface_contracts.py --tb=short -q
```

### Integration with Development Workflow

**Pre-commit Validation**:
```bash
# Add to git hooks or CI pipeline
uv run pytest resources/tests/unit/test_interface_contracts.py -x
```

**Interface Change Protocol**:
1. Update interface definition in `src/jarvis/core/interfaces.py`
2. Update implementation to match
3. Run contract validation tests
4. Update documentation if parameters changed
5. Update any usage examples

---

## Integration Testing

### Search Flow Integration

```python
# resources/tests/integration/test_search_flow.py
import pytest
from pathlib import Path
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vector.indexer import VectorIndexer
from jarvis.services.vector.searcher import VectorSearcher
from jarvis.services.vault.reader import VaultReader

class TestSearchFlowIntegration:
    
    def test_index_and_search_flow(self, test_vault_path, test_database_path):
        """Test complete index and search workflow."""
        # Arrange
        database = VectorDatabase(test_database_path)
        encoder = VectorEncoder()
        indexer = VectorIndexer(database, encoder)
        
        # Create test content
        test_vault_path.mkdir(parents=True, exist_ok=True)
        (test_vault_path / "ai_concepts.md").write_text(
            "# AI Concepts\n\nMachine learning and neural networks are fundamental to AI."
        )
        (test_vault_path / "programming.md").write_text(
            "# Programming\n\nPython is great for data science and machine learning."
        )
        
        # Act - Index the vault
        indexer.index_vault(str(test_vault_path), "test_vault")
        
        # Act - Search the indexed content
        vaults = {"test_vault": test_vault_path}
        searcher = VectorSearcher(database, encoder, vaults)
        results = searcher.search("machine learning", top_k=5)
        
        # Assert
        assert len(results) == 2
        assert all(result.vault_name == "test_vault" for result in results)
        assert any("ai_concepts.md" in result.path for result in results)
        assert any("programming.md" in result.path for result in results)
        
        # Cleanup
        database.close()
    
    def test_multi_vault_search(self, tmp_path, test_database_path):
        """Test searching across multiple vaults."""
        # Arrange
        vault1 = tmp_path / "vault1"
        vault2 = tmp_path / "vault2"
        vault1.mkdir()
        vault2.mkdir()
        
        (vault1 / "tech.md").write_text("Technology and programming")
        (vault2 / "science.md").write_text("Scientific research methods")
        
        database = VectorDatabase(test_database_path)
        encoder = VectorEncoder()
        indexer = VectorIndexer(database, encoder)
        
        # Index both vaults
        indexer.index_vault(str(vault1), "vault1")
        indexer.index_vault(str(vault2), "vault2")
        
        # Act
        vaults = {"vault1": vault1, "vault2": vault2}
        searcher = VectorSearcher(database, encoder, vaults)
        results = searcher.search("research", top_k=5)
        
        # Assert
        assert len(results) > 0
        vault_names = {result.vault_name for result in results}
        assert "vault2" in vault_names  # Should find science.md
        
        # Cleanup
        database.close()
```

### MCP Integration Tests

This section details the integration tests for the MCP server and its tools, ensuring proper communication and functionality with Claude Desktop.

#### `test_mcp_integration.py`

This file contains comprehensive integration tests for the MCP tools, including:

-   **`test_search_graph_fallback`**: Verifies that the `search-graph` tool correctly falls back to semantic search when Neo4j is disabled or unavailable.
-   **`test_get_health_status`**: Tests the `get-health-status` tool, ensuring it returns accurate health information for all services (Neo4j, VectorDB, Vault).
-   **`test_search_combined`**: Validates the `search-combined` tool, checking its ability to merge, deduplicate, and rank results from both semantic and keyword searches.
-   **`test_mcp_cache_behavior`**: Assesses the MCP-level caching mechanism, verifying cache hits, misses, and TTL-based expiration.

```python
# resources/tests/mcp/test_mcp_integration.py
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from jarvis.mcp.server import create_mcp_server
from jarvis.utils.config import JarvisSettings
import mcp.types as types

@pytest.fixture
def mock_settings():
    settings = JarvisSettings()
    settings.graph_enabled = True
    settings.mcp_cache_size = 2
    settings.mcp_cache_ttl = 1
    return settings

@pytest.mark.anyio
async def test_search_graph_fallback(mock_settings):
    mock_settings.graph_enabled = False
    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        mock_context = mock_context_class.return_value
        mock_context.graph_database.is_healthy = False
        mock_context.settings = mock_settings
        with patch('jarvis.mcp.server._handle_semantic_search', new_callable=AsyncMock) as mock_semantic_search:
            mock_semantic_search.return_value = [types.TextContent(type="text", text="Semantic search results")]
            server = create_mcp_server({}, MagicMock(), mock_settings)
            result = await server.call_tool('search-graph', {'query_note_path': 'some/note.md'})
            assert len(result) == 2
            assert "Graph search is unavailable" in result[0].text
            assert "Semantic search results" in result[1].text
            mock_semantic_search.assert_called_once_with(mock_context, {'query': 'some/note.md', 'limit': 10})

@pytest.mark.anyio
async def test_get_health_status(mock_settings):
    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        mock_context = mock_context_class.return_value
        mock_context.settings = mock_settings
        mock_context.health_checker.get_overall_health.return_value = {
            "overall_status": "HEALTHY",
            "services": [
                {"service": "Neo4j", "status": "HEALTHY"},
                {"service": "VectorDB", "status": "HEALTHY"},
                {"service": "Vault", "status": "HEALTHY"}
            ]
        }
        server = create_mcp_server({}, MagicMock(), mock_settings)
        result = await server.call_tool('get-health-status', {})
        assert len(result) == 1
        assert result[0].type == "text"
        health_data = json.loads(result[0].text)
        assert health_data["overall_status"] == "HEALTHY"
        mock_context.health_checker.get_overall_health.assert_called_once()

@pytest.mark.anyio
async def test_search_combined(mock_settings):
    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        mock_context = mock_context_class.return_value
        mock_context.settings = mock_settings
        mock_semantic_search_results = [
            MagicMock(spec=types.TextContent, path=Path("sem1.md"), vault_name="vault1", similarity_score=0.9),
            MagicMock(spec=types.TextContent, path=Path("common.md"), vault_name="vault1", similarity_score=0.8),
        ]
        mock_context.searcher.search.return_value = mock_semantic_search_results
        mock_keyword_search_results = [
            {"path": "key1.md", "vault_name": "vault1", "match_type": "name", "size": 100},
            {"path": "common.md", "vault_name": "vault1", "match_type": "content", "size": 200, "content_preview": "preview"},
        ]
        mock_vault_reader = MagicMock()
        mock_vault_reader.search_vault.return_value = mock_keyword_search_results
        mock_context.vault_readers = {"default": mock_vault_reader}
        mock_context.vaults = {"default": Path("/tmp/vault")}
        mock_context.ranker.merge_and_rank.return_value = [
            mock_semantic_search_results[0],
            mock_semantic_search_results[1],
            mock_keyword_search_results[0]
        ]
        server = create_mcp_server({}, MagicMock(), mock_settings)
        result = await server.call_tool('search-combined', {'query': 'test query', 'limit': 5, 'vault': 'default', 'search_content': True})
        response_text = result[0].text
        assert "Found 3 combined results for 'test query':" in response_text
        assert "[SEMANTIC] sem1.md" in response_text
        assert "[SEMANTIC] common.md" in response_text
        assert "[KEYWORD] key1.md" in response_text
        mock_context.searcher.search.assert_called_once_with(query='test query', limit=5, vault_name='default')
        mock_vault_reader.search_vault.assert_called_once_with(query='test query', limit=5, vault='default', search_content=True)
        mock_context.ranker.merge_and_rank.assert_called_once()

@pytest.mark.anyio
async def test_mcp_cache_behavior(mock_settings):
    mock_settings.mcp_cache_size = 1
    mock_settings.mcp_cache_ttl = 1
    with patch('jarvis.mcp.server.MCPServerContext') as mock_context_class:
        mock_context = mock_context_class.return_value
        mock_context.settings = mock_settings
        mock_context.mcp_cache = MCPToolCache(mock_settings.mcp_cache_size, mock_settings.mcp_cache_ttl)
        mock_tool_handler = AsyncMock(return_value=[types.TextContent(type="text", text="Unique result 1")])
        with patch.dict('jarvis.mcp.server.Server.request_handlers', {
            types.CallToolRequest: MagicMock(side_effect=lambda req: mock_tool_handler(req.params.name, req.params.arguments))
        }):
            server = create_mcp_server({}, MagicMock(), mock_settings)
            result1 = await server.call_tool('test-cached-tool', {"param": "value"})
            assert "Unique result 1" in result1[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 1
            mock_tool_handler.assert_called_once()
            mock_tool_handler.reset_mock()
            result2 = await server.call_tool('test-cached-tool', {"param": "value"})
            assert "Unique result 1" in result2[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 1
            assert mock_context.mcp_cache.get_stats()["misses"] == 1
            mock_tool_handler.assert_not_called()
            mock_context.clear_cache()
            mock_tool_handler.reset_mock()
            result3 = await server.call_tool('test-cached-tool', {"param": "value"})
            assert "Unique result 1" in result3[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 1
            mock_tool_handler.assert_called_once()
            mock_tool_handler.reset_mock()
            result4 = await server.call_tool('test-ttl-tool', {"param": "value"})
            assert "Unique result 1" in result4[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 2
            mock_tool_handler.assert_called_once()
            time.sleep(1.1)
            mock_tool_handler.reset_mock()
            result5 = await server.call_tool('test-ttl-tool', {"param": "value"})
            assert "Unique result 1" in result5[0].text
            assert mock_context.mcp_cache.get_stats()["hits"] == 0
            assert mock_context.mcp_cache.get_stats()["misses"] == 3
            mock_tool_handler.assert_called_once()
```

---

## MCP Server Testing

### Tool Schema Validation

```python
# resources/tests/mcp/test_tool_schemas.py
import pytest
import jsonschema
from jarvis.mcp.server import create_mcp_server

class TestMCPToolSchemas:
    
    @pytest.mark.asyncio
    async def test_search_semantic_schema(self, test_vault_path, test_database_path):
        """Test semantic search tool schema validation."""
        # Arrange
        vaults = {"test": test_vault_path}
        server = create_mcp_server(vaults, test_database_path)
        
        # Act
        tools = await server.list_tools()
        search_tool = next(tool for tool in tools if tool.name == "search-semantic")
        
        # Assert
        schema = search_tool.inputSchema
        assert schema["type"] == "object"
        assert "query" in schema["required"]
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        
        # Validate sample inputs
        valid_input = {"query": "test search"}
        jsonschema.validate(valid_input, schema)
        
        with pytest.raises(jsonschema.ValidationError):
            invalid_input = {"limit": 10}  # Missing required 'query'
            jsonschema.validate(invalid_input, schema)
    
    @pytest.mark.asyncio
    async def test_all_tools_have_valid_schemas(self, test_vault_path, test_database_path):
        """Test all MCP tools have valid schemas."""
        # Arrange
        vaults = {"test": test_vault_path}
        server = create_mcp_server(vaults, test_database_path)
        
        # Act
        tools = await server.list_tools()
        
        # Assert
        assert len(tools) == 5  # Expected number of tools
        
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')
            assert isinstance(tool.inputSchema, dict)
            assert "type" in tool.inputSchema
            assert "properties" in tool.inputSchema
```

### Claude Desktop Integration

```python
# resources/tests/mcp/test_claude_integration.py
import pytest
import json
from unittest.mock import Mock, patch
from jarvis.mcp.server import run_mcp_server

class TestClaudeIntegration:
    
    @pytest.mark.asyncio
    async def test_mcp_server_stdio_communication(self, test_vault_path, test_database_path):
        """Test MCP server communication over stdio."""
        # This test would require more complex setup to mock stdio streams
        # and test the actual protocol communication
        pass
    
    def test_mcp_server_initialization_message(self, test_vault_path, test_database_path):
        """Test MCP server initialization response."""
        # Arrange
        vaults = {"test": test_vault_path}
        
        # This would test the actual initialization handshake
        # Implementation depends on MCP protocol specifics
        pass
```

---

## End-to-End Testing

### Complete Workflow Tests

```python
# resources/tests/e2e/test_complete_workflow.py
import pytest
import asyncio
from pathlib import Path
from jarvis.cli.main import main
from jarvis.mcp.server import run_mcp_server

class TestCompleteWorkflow:
    
    def test_cli_index_and_search_workflow(self, test_vault_path, test_database_path, monkeypatch):
        """Test complete CLI workflow from indexing to search."""
        # Arrange
        test_vault_path.mkdir(parents=True, exist_ok=True)
        (test_vault_path / "test.md").write_text("Machine learning concepts")
        
        # Mock sys.argv for CLI testing
        monkeypatch.setattr('sys.argv', [
            'jarvis', 'index', 
            '--vault', str(test_vault_path),
            '--database', str(test_database_path)
        ])
        
        # Act & Assert
        # This would test the actual CLI execution
        # Implementation depends on CLI framework
        pass
    
    @pytest.mark.asyncio
    async def test_mcp_server_full_lifecycle(self, test_vault_path, test_database_path):
        """Test MCP server complete lifecycle."""
        # Arrange
        vaults = {"test": test_vault_path}
        
        # Create test content
        test_vault_path.mkdir(parents=True, exist_ok=True)
        (test_vault_path / "ai.md").write_text("Artificial intelligence concepts")
        (test_vault_path / "ml.md").write_text("Machine learning algorithms")
        
        # This would test the complete MCP server lifecycle
        # Including initialization, tool execution, and cleanup
        pass
```

---

## Performance Testing

### Search Performance Tests

```python
# resources/tests/performance/test_search_performance.py
import pytest
import time
from jarvis.services.vector.searcher import VectorSearcher

class TestSearchPerformance:
    
    def test_search_response_time(self, vector_database, vector_encoder):
        """Test search response time under normal conditions."""
        # Arrange
        vaults = {"test": Path("./test_vault")}
        searcher = VectorSearcher(vector_database, vector_encoder, vaults)
        
        # Act
        start_time = time.time()
        results = searcher.search("test query", top_k=10)
        end_time = time.time()
        
        # Assert
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_concurrent_search_performance(self, vector_database, vector_encoder):
        """Test search performance under concurrent load."""
        import concurrent.futures
        
        # Arrange
        vaults = {"test": Path("./test_vault")}
        searcher = VectorSearcher(vector_database, vector_encoder, vaults)
        
        def search_task(query_id):
            return searcher.search(f"test query {query_id}", top_k=5)
        
        # Act
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_task, i) for i in range(20)]
            results = [future.result() for future in futures]
        end_time = time.time()
        
        # Assert
        total_time = end_time - start_time
        assert total_time < 5.0  # Should complete within 5 seconds
        assert len(results) == 20
        assert all(len(result) <= 5 for result in results)
```

---

## Test Data Management

### Test Fixtures

```python
# resources/tests/fixtures/mock_data.py
from pathlib import Path
from jarvis.models.search import SearchResult
from jarvis.models.graph import GraphNode, GraphRelationship

def create_test_vault(vault_path: Path) -> None:
    """Create a test vault with sample content."""
    vault_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    (vault_path / "ai.md").write_text("""
# Artificial Intelligence

AI encompasses machine learning, neural networks, and deep learning.
Applications include natural language processing and computer vision.
""")
    
    (vault_path / "ml.md").write_text("""
# Machine Learning

Machine learning is a subset of AI that uses statistical methods.
Common algorithms include linear regression, decision trees, and neural networks.
""")
    
    (vault_path / "programming.md").write_text("""
# Programming

Python is widely used for data science and machine learning.
Libraries like scikit-learn, TensorFlow, and PyTorch are popular.
""")

def create_mock_search_results() -> List[SearchResult]:
    """Create mock search results for testing."""
    return [
        SearchResult(
            path="ai.md",
            similarity_score=0.95,
            vault_name="test",
            content_preview="AI encompasses machine learning..."
        ),
        SearchResult(
            path="ml.md",
            similarity_score=0.87,
            vault_name="test",
            content_preview="Machine learning is a subset..."
        )
    ]

def create_mock_graph_data() -> dict:
    """Create mock graph data for testing."""
    return {
        "nodes": [
            {"id": "ai.md", "label": "Artificial Intelligence", "path": "ai.md"},
            {"id": "ml.md", "label": "Machine Learning", "path": "ml.md"}
        ],
        "relationships": [
            {"source": "ai.md", "target": "ml.md", "type": "INCLUDES"}
        ]
    }
```

### Test Database Management

```python
# resources/tests/fixtures/test_databases.py
import pytest
from pathlib import Path
from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.indexer import VectorIndexer
from jarvis.services.vector.encoder import VectorEncoder

@pytest.fixture
def populated_test_database(test_database_path, test_vault_path):
    """Create a populated test database."""
    # Create test content
    create_test_vault(test_vault_path)
    
    # Initialize and populate database
    database = VectorDatabase(test_database_path)
    encoder = VectorEncoder()
    indexer = VectorIndexer(database, encoder)
    
    indexer.index_vault(str(test_vault_path), "test_vault")
    
    yield database
    
    # Cleanup
    database.close()
```

---

## Running Tests

### Basic Test Commands

```bash
# Run all tests
uv run pytest resources/tests/

# Run specific test categories
uv run pytest resources/tests/unit/
uv run pytest resources/tests/integration/
uv run pytest resources/tests/mcp/

# Run with coverage
uv run pytest --cov=src/jarvis --cov-report=html resources/tests/

# Run performance tests
uv run pytest resources/tests/performance/ -v

# Run specific test file
uv run pytest resources/tests/unit/services/test_vector_search.py
uv run pytest resources/tests/unit/test_health_checker.py
uv run pytest resources/tests/unit/test_result_ranker.py
uv run pytest resources/tests/unit/test_mcp_cache.py
uv run pytest resources/tests/mcp/test_mcp_integration.py

# Run with specific markers
uv run pytest -m "not slow" resources/tests/
```

### Test Configuration

```python
# pytest.ini
[tool:pytest]
testpaths = resources/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    mcp: marks tests as MCP-specific tests
    performance: marks tests as performance tests
```

### Continuous Integration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install UV
      run: pip install uv
    
    - name: Install dependencies
      run: uv sync
    
    - name: Run tests
      run: uv run pytest resources/tests/ --cov=src/jarvis --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

Pre-commit hooks are used to enforce code quality standards before commits are made. This ensures that `ruff` (for linting and formatting) and `mypy` (for type checking) are run automatically.

To set up pre-commit hooks, install `pre-commit` (already in `dev` dependencies) and run:

```bash
uv run pre-commit install
```

Configuration is in `pyproject.toml` for `ruff` and `mypy`.

---

## Next Steps

- [Code Standards](code-standards.md) - Detailed coding guidelines
- [Contribution Guide](contribution-guide.md) - How to contribute to the project
- [Developer Guide](developer-guide.md) - Development setup and workflow
- [Troubleshooting](../07-maintenance/troubleshooting.md) - Common issues and solutions