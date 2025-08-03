# Project Overview

*What is Jarvis Assistant and why does it exist?*

## Mission

Jarvis Assistant provides production-ready MCP (Model Context Protocol) tools for AI systems to search and analyze Obsidian vaults through semantic search and graph relationships.

## Problem Statement

AI systems need efficient ways to discover knowledge within personal knowledge bases. Obsidian vaults contain rich information but lack intelligent search capabilities that can understand context and relationships between notes.

## Solution Approach

Jarvis Assistant bridges this gap by providing:
- **Semantic Search**: Vector-based search using sentence transformers
- **Graph Relationships**: Neo4j-powered relationship discovery
- **MCP Integration**: Direct integration with Claude Desktop and other AI tools
- **Local-First**: All processing happens locally for privacy and control

## Key Benefits

1. **Intelligent Discovery**: Find related information through semantic similarity
2. **Relationship Mapping**: Understand connections between ideas and concepts
3. **AI-Native**: Built specifically for AI system integration
4. **Privacy-Focused**: Local processing with no external dependencies
5. **Production-Ready**: Robust error handling and performance optimization

## Current Status

**10+ Production-Ready MCP Tools:**
- `search-semantic` - Vector-based semantic search
- `search-vault` - Traditional keyword search  
- `search-graph` - Graph relationship discovery
- `search-combined` - Hybrid search strategies
- `read-note` - File reading with metadata
- `list-vaults` - Vault management and statistics
- `health-status` - System health monitoring
- `performance-metrics` - Performance analytics
- `analyze-domains` - Domain-specific content analysis
- `assess-quality` - Content quality assessment
- `get-vault-context` - Contextual vault information

**Advanced Features:**
- Plugin-based architecture for extensible MCP tools
- Dependency injection system for modular services
- Multiple database adapters (DuckDB, Chroma, Pinecone support)
- Extensions system for AI workflows and integrations
- Analytics services for content assessment and domain analysis

## Technology Stack

- **Language**: Python 3.11+ with UV package management
- **Search**: DuckDB for vector storage, sentence-transformers for embeddings
- **Graph**: Neo4j for relationship mapping and traversal (optional)
- **Integration**: MCP protocol for AI tool connectivity
- **Architecture**: Plugin-based with dependency injection, local-first processing
- **Extensions**: AI workflow automation, LLM integration, analytics services
- **Database Support**: Multi-backend (DuckDB, Chroma, Pinecone) with migration tools

## Next Steps

For getting started, see: [Quick Start Guide](../03-getting-started/quick-start.md)

For technical details, see: [System Design](../02-system-design/data-flow.md)

For usage examples, see: [Common Workflows](../04-usage/common-workflows.md)