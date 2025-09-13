# Project Structure & Organization

## Root Directory Layout
```
jarvis-assistant/
├── src/jarvis/              # Main application code
├── docs/                    # Comprehensive documentation
├── resources/               # Scripts, debug tools, tests, datasets
├── config/                  # Environment and YAML configuration
├── data/                    # Runtime databases (DuckDB files)
├── .kiro/                   # Kiro steering rules
├── pyproject.toml           # Project configuration and dependencies
└── uv.lock                  # Dependency lock file
```

## Source Code Organization (`src/jarvis/`)
```
src/jarvis/
├── __init__.py              # Package initialization
├── main.py                  # CLI entry point
├── core/                    # Core system components
│   ├── service_registry.py  # Service management
│   ├── container.py         # Dependency injection
│   ├── events.py           # Event bus system
│   └── interfaces.py       # Service interfaces
├── services/               # Business logic services
│   ├── vector/             # Semantic search (DuckDB)
│   ├── graph/              # Graph search (Neo4j)
│   └── vault/              # File system operations
├── mcp/                    # MCP protocol implementation
│   ├── server.py           # Main MCP server
│   ├── plugins/            # MCP tool implementations
│   └── mcp_main.py         # MCP entry point
├── models/                 # Data models and schemas
├── database/               # Database adapters and migrations
├── utils/                  # Utilities and helpers
├── monitoring/             # Health checks and metrics
└── extensions/             # Optional extension system
```

## Documentation Structure (`docs/`)
- **01-overview/**: Project overview and key concepts
- **02-architecture/**: Architecture and technical design
- **03-getting-started/**: Installation and setup guides
- **04-usage/**: API examples and workflows
- **05-development/**: Developer guides and standards
- **06-reference/**: Complete API and configuration reference
- **07-maintenance/**: Troubleshooting and performance
- **08-progression/**: Version history and milestones

## Resources Directory (`resources/`)
```
resources/
├── debug/                 # Debugging tools and helpers
├── scripts/               # Utility scripts
├── data-generation/       # Test data generation
└── tests/                 # Test suite
    ├── unit/              # Unit tests (mirror src structure)
    ├── integration/       # Integration tests
    └── mcp/               # MCP-specific tests
```

## Key Architectural Patterns

### Service Layer Organization
- **Core Services**: Essential system functionality (vector, graph, vault)
- **MCP Layer**: Protocol-specific implementations
- **Utility Layer**: Cross-cutting concerns (logging, config, exceptions)

### Naming Conventions
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private Members**: `_leading_underscore`

### Import Organization
1. Standard library imports
2. Third-party imports
3. Local imports (absolute paths from `jarvis.`)

### Configuration Management
- **Environment Variables**: `JARVIS_*` prefix
- **Settings Classes**: Pydantic-based with validation
- **Config Files**: `config/.env` and YAML (`config/base.yaml`, `config/local.yaml`)

## Config Directory (`config/`)
```
config/
├── .env.example          # Canonical environment template
├── .env                  # Local overrides (gitignored)
├── base.yaml             # Base YAML configuration
├── local.yaml            # Local overrides
└── neo4j/                # Neo4j schema/configuration
```

### Error Handling
- **Custom Exceptions**: Hierarchical exception classes in `utils/exceptions.py`
- **Error Propagation**: Service-specific errors converted to appropriate MCP errors
- **Logging**: Structured logging with context information

### Testing Structure
- **Unit Tests**: Mirror source structure in `resources/tests/unit/`
- **Integration Tests**: End-to-end scenarios in `resources/tests/integration/`
- **MCP Tests**: Protocol-specific tests in `resources/tests/mcp/`
- **Fixtures**: Reusable test data and mocks

## File Naming Patterns
- **Services**: `{domain}_service.py` (e.g., `vector_service.py`)
- **Models**: `{entity}.py` (e.g., `search_result.py`)
- **Tests**: `test_{module}.py` (e.g., `test_vector_service.py`)
- **Utilities**: `{purpose}.py` (e.g., `logging.py`, `config.py`)

## Development Workflow
1. **Feature Development**: Create in appropriate service directory
2. **Testing**: Add unit tests mirroring source structure
3. **Documentation**: Update relevant docs sections
4. **Integration**: Add MCP tool if user-facing functionality
5. **Quality Checks**: Run ruff, mypy, and pytest before commit
