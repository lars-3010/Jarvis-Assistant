# Graph Database Setup for Future Development

## Overview

The Jarvis Assistant project includes graph database functionality in `src/jarvis/future/` for future phases. This guide outlines setup options when you're ready to implement graph-based knowledge relationships.

## Current Status

- **Phase 2**: Graph functionality is in `src/jarvis/future/` (not active)
- **Phase 3**: Focus on semantic search optimization (DuckDB)
- **Future Phases**: Graph RAG integration

## Setup Options

### Option 1: Neo4j Desktop (Recommended for Development)

**Best for**: Local development, GUI access, easy management

1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Create new project: "Jarvis Assistant"
3. Create database with APOC plugin
4. Default connection: `bolt://localhost:7687`

### Option 2: Local Neo4j Installation

**Best for**: Production-like setup, headless servers

```bash
# macOS with Homebrew
brew install neo4j

# Start service
neo4j start

# Access browser: http://localhost:7474
```

### Option 3: Embedded Alternatives

**Best for**: Simplified deployment, no server management

- **DuckDB Graph Extensions**: Use DuckDB's graph capabilities
- **NetworkX + SQLite**: Python-native graph operations
- **Apache AGE**: PostgreSQL graph extension

### Option 4: Cloud Options

**Best for**: Production deployment, managed services

- **Neo4j AuraDB**: Managed Neo4j cloud
- **Amazon Neptune**: AWS graph database
- **Azure Cosmos DB**: Graph API

## Configuration

When implementing graph functionality, update:

1. `src/jarvis/future/config/settings.py` - Database connection settings
2. `resources/config/.env.backend.example` - Environment variables
3. `src/jarvis/services/graph/` - Move implementation from future folder

## Integration with MCP

Graph search will be exposed as MCP tools:
- `graph-search`: Traverse knowledge relationships
- `graph-explore`: Discover connected concepts
- `graph-insert`: Add new relationships

## Development Philosophy

Maintain the local-first approach:
- Prefer embedded solutions when possible
- Keep setup simple for end users
- Provide multiple deployment options
- Document all dependencies clearly