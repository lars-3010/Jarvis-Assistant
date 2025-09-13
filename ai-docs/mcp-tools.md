**MCP Tool Commands**  
Docs version: 0.2.0 • Snapshot: 2025-09-13
- Summary: Current built‑in MCP tools, their purpose, and input parameters to help AI integrations call them consistently.

**search-semantic**
- Description: Perform semantic search across vault content using natural language queries
- Inputs:
  - query: string, required — Natural language search query
  - limit: integer, default 10, min 1, max 50 — Maximum number of results
  - vault: string, optional — Vault name to search within
  - similarity_threshold: number, optional (0.0–1.0)
  - format: string, default markdown, enum [markdown, json]

**read-note**
- Description: Read the content of a specific note from a vault
- Inputs:
  - path: string, required — Path to the note relative to vault root
  - vault: string, optional — Vault name (uses first available if not specified)
  - format: string, default markdown, enum [markdown, json]

**list-vaults**
- Description: List all available vaults and their statistics
- Inputs:
  - none (empty object)

**search-vault**
- Description: Search for files in vault by filename or content (traditional search)
- Inputs:
  - query: string, required — Search term for filenames or content
  - limit: integer, default 20, min 1, max 100 — Maximum number of results
  - vault: string, optional — Vault name to search within
  - search_content: boolean, default true — Include content search
  - format: string, default json, enum [json]

**search-graph**
- Description: Search for notes and their relationships in the knowledge graph
- Inputs:
  - query_note_path: string, required — Note path to center search, or keywords
  - depth: integer, default 1, min 1, max 5 — Graph traversal depth
  - format: string, default json, enum [json]

**search-combined**
- Description: Perform a combined semantic and keyword search across vault content
- Inputs:
  - query: string, required — Natural language search query
  - limit: integer, default 10, min 1, max 50
  - search_content: boolean, default true — Include keyword content search
  - vault: string, optional — Vault name to search within

**get-health-status**
- Description: Get the health status of all Jarvis Assistant services
- Inputs:
  - include_details: boolean, default false
  - format: string, default json, enum [json]

**get-performance-metrics**
- Description: Get performance metrics and statistics for MCP tools and services
- Inputs:
  - reset_after_read: boolean, default false
  - filter_prefix: string, optional — Filter metrics by prefix
  - format: string, default json, enum [json]

**get-vault-context**
- Description: Generate comprehensive vault context with organization patterns, quality metrics, and insights
- Inputs:
  - vault: string, default default — Vault to analyze
  - include_recommendations: boolean, default true
  - include_quality_gaps: boolean, default true
  - format: string, default markdown, enum [markdown, json]

**assess-quality**
- Description: Assess content quality for notes or vault with improvement suggestions
- Inputs:
  - scope: string, default vault, enum [note, vault]
  - note_path: string, required when scope=note
  - vault: string, default default
  - include_suggestions: boolean, default true
  - show_detailed_metrics: boolean, default false
  - format: string, default markdown, enum [markdown, json]

**analyze-domains**
- Description: Analyze knowledge domains, clusters, and bridges between related domains
- Inputs:
  - vault: string, default default
  - include_bridges: boolean, default true
  - include_clusters: boolean, default true
  - show_connections: boolean, default true
  - min_domain_size: integer, default 3, min 2
  - format: string, default markdown, enum [markdown, json]

**analytics-cache-status**
- Description: Get current analytics cache status and performance metrics
- Note: Schema currently failing validation in code; interface may change.

**analytics-invalidate-cache**
- Description: Invalidate analytics cache entries to force fresh analysis on next request
- Note: Schema currently failing validation in code; interface may change.

**search-graphrag**
- Description: Comprehensive GraphRAG search combining semantic retrieval with graph traversal
- Inputs:
  - query: string, required
  - mode: string, default quick, enum [quick, focused, comprehensive]
  - max_sources: integer, default 5, min 1, max 20
  - depth: integer, default 1, min 1, max 3
  - vault: string, optional
  - include_content: boolean, default true
  - enable_clustering: boolean, default true

**Notes**
- Error handling: Most tools return a single text item containing structured JSON for easy parsing.
- Caching: Server-level MCP cache may transparently cache results by tool name + arguments.
- Containers: Tools expect services via a DI container implementing `get(interface)` for required interfaces.
