# Technology Stack & Build System

## Core Technologies
- **Python**: 3.11+ (required for modern type hints and performance)
- **Package Manager**: UV (fast Python package manager, replaces pip/poetry)
- **Database**: DuckDB (embedded vector database) + optional Neo4j (graph database)
- **AI/ML**: sentence-transformers (local embeddings), PyTorch backend
- **Protocol**: MCP (Model Context Protocol) for AI tool integration

## Key Dependencies
- **mcp**: 1.6.0+ (Model Context Protocol implementation)
- **duckdb**: 1.1.3+ (vector database with similarity search)
- **sentence-transformers**: 4.0.0+ (local text embeddings)
- **pydantic**: 2.0+ (data validation and settings)
- **click**: 8.1.7+ (CLI interface)
- **watchdog**: 6.0.0+ (file system monitoring)

## Development Tools
- **Linting/Formatting**: Ruff (replaces black, flake8, isort)
- **Type Checking**: MyPy with strict mode
- **Testing**: pytest with asyncio support and coverage
- **Pre-commit**: Automated code quality checks

## Common Commands

### Setup & Installation
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev
```

### Development Workflow
```bash
# Format code
uv run ruff format src/

# Lint and fix issues
uv run ruff check src/ --fix

# Type checking
uv run mypy src/

# Run tests with coverage
uv run pytest resources/tests/ --cov=src/jarvis --cov-report=html

# Run all quality checks
./scripts/quality-check.sh
```

### Application Commands
```bash
# Index a vault for semantic search
uv run jarvis index --vault /path/to/vault

# Start MCP server for Claude Desktop
uv run jarvis mcp --vault /path/to/vault --watch

# Interactive semantic search testing
uv run jarvis search --vault /path/to/vault

# System statistics and health
uv run jarvis stats --vault /path/to/vault

# Graph database indexing (optional)
uv run jarvis graph-index --vault /path/to/vault
```

## Architecture Patterns
- **Service Registry + Dependency Injection**: Centralized service management
- **Event-Driven Architecture**: Loose coupling between services via event bus
- **MCP Protocol**: JSON-RPC over stdio for AI tool integration
- **Graceful Degradation**: Core features work without optional components
- **Local-First**: All processing happens on user's machine

## Configuration
- **Settings**: Pydantic-based configuration with environment variable support
- **Logging**: Structured logging with JSON output for production
- **Caching**: Multi-layer caching (LRU for queries, TTL for results)
- **Error Handling**: Custom exception hierarchy with proper error propagation