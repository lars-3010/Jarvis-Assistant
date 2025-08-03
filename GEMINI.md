# JARVIS ASSISTANT - MCP TOOLS FOR AI SYSTEMS

*project philosophy: "Excellent MCP tools for AI systems"*

## PROJECT MISSION

Production-ready MCP server providing semantic search and knowledge graph capabilities for AI systems. Built for Claude Desktop integration via Model Context Protocol (MCP) servers. Focus: Reliable, efficient tools for AI-powered knowledge discovery in Obsidian vaults.

## CURRENT STATUS

-   **Status**: Stable MCP Tools - Production Ready ✅
-   **Focus**: Reliable MCP server with 5 working tools for AI systems
-   **Architecture**: Production-ready MCP server with semantic search (DuckDB) and graph capabilities (Neo4j)
-   **Next Priority**: Architecture improvements and enhanced testing

### Current MCP Tools (Production Ready)

✅ **5 Working MCP Tools**:
-   **search-semantic**: Semantic vector search with DuckDB backend
-   **search-vault**: Traditional keyword search with content matching
-   **search-graph**: Graph-based relationship search via Neo4j
-   **read-note**: File reading with metadata and error handling
-   **list-vaults**: Vault management with statistics and validation

### Architecture Highlights

✅ **Production-Ready Features**:
-   **Robust Error Handling**: Graceful degradation when services unavailable, including optional Neo4j integration with semantic search fallback.
-   **Performance Optimization**: Caching, batch processing, optimized search
-   **Clean Service Architecture**: Separated vector, graph, and vault services
-   **Comprehensive CLI**: Full command suite for indexing and server management
-   **Local-First Approach**: No external dependencies, privacy-focused

## DEVELOPMENT APPROACH

### Documentation-First Development
**CRITICAL**: Always maintain and update project documentation when making changes:

1. **Update docs/ first**: Before coding, check if relevant documentation exists in `docs/`
2. **Keep CLAUDE.md current**: This file should reflect the actual project state
3. **Link to detailed docs**: CLAUDE.md provides overview, `docs/` provides depth
4. **Archive obsolete content**: Move outdated docs to `docs/archived/`

### Comprehensive Documentation System

The project uses a **7-section documentation structure** designed for multiple audiences:

- **📋 [docs/README.md](docs/README.md)** - Documentation navigation hub
- **🎯 [01-overview/](docs/01-overview/)** - Project context and architecture
- **🏗️ [02-system-design/](docs/02-system-design/)** - Technical deep dive and component interaction
- **🚀 [03-getting-started/](docs/03-getting-started/)** - Progressive user onboarding
- **💡 [04-usage/](docs/04-usage/)** - Practical examples and workflows
- **🔧 [05-development/](docs/05-development/)** - Developer guides and contribution process
- **📚 [06-reference/](docs/06-reference/)** - Complete API and configuration reference
- **🛠️ [07-maintenance/](docs/07-maintenance/)** - Operational guides and troubleshooting

**For AI Tools (Claude Code)**: Start with `docs/02-system-design/component-interaction.md` for quick context, then explore relevant sections based on your task.

### MCP Tool Development Focus

When developing MCP tools, prioritize:

#### 🔧 Reliability & Robustness
-   Implement comprehensive error handling and graceful degradation
-   Validate MCP tools with Claude Desktop integration
-   Test tool parameter handling and edge cases

#### ⚡ Performance & Efficiency
-   Optimize search performance with caching and batch processing
-   Design efficient data flows and minimal resource usage
-   Monitor and improve tool response times

#### 📊 Testing & Quality
-   Create comprehensive test suites for all MCP tools
-   Validate integration with Claude Desktop
-   Maintain high code quality standards

📚 **For advanced development ideas**: See `docs/future-expansion-ideas.md`  
🔧 **For detailed development setup**: See [docs/05-development/developer-guide.md](docs/05-development/developer-guide.md)  
📊 **For testing strategy**: See [docs/05-development/testing-strategy.md](docs/05-development/testing-strategy.md)

## DEVELOPMENT ENVIRONMENT

### Technology Stack

-   **Primary**: Python 3.11+ with UV package management
-   **Databases**: DuckDB (vector search) + Neo4j (graph relationships)
-   **AI/ML**: sentence-transformers, PyTorch, numpy
-   **Architecture**: MCP protocol-based dual database system
-   **Development**: Direct UV execution, local-first approach

### Essential Commands

**Development Setup:**
-   `uv sync` - Install dependencies
-   `uv run jarvis --help` - Show all available commands

**Core Operations:**
-   `uv run jarvis mcp --vault /path --watch` - Start MCP server for Claude Desktop
-   `uv run jarvis index --vault /path` - Index Obsidian vault for semantic search
-   `uv run jarvis graph-index --vault /path` - Index vault for graph search

**Quality Assurance:**
-   `uv run ruff check src/` - Lint and format code
-   `uv run pytest resources/tests/` - Run test suite

📚 **For detailed setup and usage**: See [docs/03-getting-started/quick-start.md](docs/03-getting-started/quick-start.md)  
🎯 **For practical examples**: See [docs/04-usage/api-examples.md](docs/04-usage/api-examples.md)  
⚙️ **For configuration**: See [docs/06-reference/configuration-reference.md](docs/06-reference/configuration-reference.md)

### Project Structure

```
src/jarvis/                 # Core package
├── mcp/                   # MCP server & tools (5 working tools)
├── services/              # Business logic
│   ├── vector/           # DuckDB semantic search (active)
│   ├── graph/            # Neo4j graph operations (active)
│   └── vault/            # Obsidian file management (active)
├── future/               # Future development staging area
├── database/             # Database adapters
├── models/               # Data models
└── utils/                # Configuration & utilities
resources/                 # Development resources
├── tests/                # Test suites (unit, integration, mcp)
├── scripts/              # Setup and utility scripts
├── config/               # Environment configuration examples
└── media/                # Architecture diagrams
```

### Code Standards

-   **Python**: Follow UV package management and Pydantic validation patterns
-   **Quality**: Use Ruff for formatting/linting, type hints with mypy
-   **Architecture**: Implement comprehensive error handling and logging
-   **Documentation**: Update relevant docs/ files when adding features

🛠️ **For complete code standards**: See [docs/05-development/code-standards.md](docs/05-development/code-standards.md)  
📖 **For API documentation**: See [docs/06-reference/api-reference.md](docs/06-reference/api-reference.md)

### Quick Reference

**Current Active Services:**
-   `src/jarvis/services/vector/` - DuckDB semantic search (production ready)
-   `src/jarvis/services/graph/` - Neo4j graph operations (production ready)
-   `src/jarvis/services/vault/` - Obsidian file management (production ready)
-   `src/jarvis/mcp/` - MCP server with 5 working tools (production ready)

**Development Resources:**
-   `resources/tests/` - Test suites (comprehensive coverage implemented)
-   `resources/config/` - Environment configuration examples

**Documentation Navigation:**
-   **Getting Started**: [docs/03-getting-started/](docs/03-getting-started/) - Setup and first steps
-   **Usage Examples**: [docs/04-usage/](docs/04-usage/) - API examples and workflows
-   **Development**: [docs/05-development/](docs/05-development/) - Developer guides and contribution
-   **Reference**: [docs/06-reference/](docs/06-reference/) - Complete API and config documentation
-   **Maintenance**: [docs/07-maintenance/](docs/07-maintenance/) - Troubleshooting and performance

**Future Development:**
-   `src/jarvis/future/` - Staging area for future enhancements
-   `docs/future-expansion-ideas.md` - Advanced features and learning system ideas

🎯 **For troubleshooting**: See [docs/07-maintenance/troubleshooting.md](docs/07-maintenance/troubleshooting.md)  
⚡ **For performance optimization**: See [docs/07-maintenance/performance-tuning.md](docs/07-maintenance/performance-tuning.md)