# MCP API Examples

This guide provides copy-paste examples for each MCP tool available in Jarvis Assistant. All examples assume you have a properly configured MCP server running with indexed vaults.

## Quick Reference

| Tool | Purpose | Required Parameters | Optional Parameters |
|------|---------|-------------------|-------------------|
| `search-semantic` | Semantic vector search | `query` | `limit`, `vault`, `similarity_threshold` |
| `read-note` | Read specific note | `path` | `vault` |
| `list-vaults` | List all vaults | None | None |
| `search-vault` | Traditional keyword search | `query` | `vault`, `search_content`, `limit` |
| `search-graph` | Graph relationship search | `query_note_path` | `depth` |
| `get-health-status` | Get health status of services | None | None |

---

## 1. search-semantic

**Purpose**: Perform semantic search using natural language queries to find conceptually related content.

### Basic Usage

```json
{
  "tool": "search-semantic",
  "arguments": {
    "query": "project management techniques"
  }
}
```

### Advanced Usage

```json
{
  "tool": "search-semantic",
  "arguments": {
    "query": "machine learning algorithms for text processing",
    "limit": 15,
    "vault": "research",
    "similarity_threshold": 0.7
  }
}
```

### Example Response

```
Found 3 results for 'project management techniques':

1. **projects/agile-methodology.md** (vault: notes, score: 0.892)
2. **workflows/task-management.md** (vault: notes, score: 0.834)
3. **tools/productivity-systems.md** (vault: notes, score: 0.767)
```

### Parameters

- **query** (required): Natural language search query
- **limit** (optional): Maximum results to return (1-50, default: 10)
- **vault** (optional): Specific vault to search within
- **similarity_threshold** (optional): Minimum similarity score (0.0-1.0)

---

## 2. read-note

**Purpose**: Read the complete content of a specific note file.

### Basic Usage

```json
{
  "tool": "read-note",
  "arguments": {
    "path": "projects/jarvis-assistant.md"
  }
}
```

### With Specific Vault

```json
{
  "tool": "read-note",
  "arguments": {
    "path": "research/ai-models.md",
    "vault": "research"
  }
}
```

### Example Response

```
# projects/jarvis-assistant.md

**Size:** 4,832 bytes  
**Modified:** 2024-01-15T14:30:22  

---

# Jarvis Assistant Project

This document outlines the development of the Jarvis Assistant MCP server...
[full note content follows]
```

### Parameters

- **path** (required): Path to the note relative to vault root
- **vault** (optional): Vault name (uses first available if not specified)

---

## 3. list-vaults

**Purpose**: List all configured vaults with their statistics and status.

### Usage

```json
{
  "tool": "list-vaults",
  "arguments": {}
}
```

### Example Response

```
# Available Vaults

## personal
- **Status:** ✅ Available
- **Path:** `/Users/username/Documents/personal-vault`
- **Notes:** 1,247
- **Last Modified:** 2024-01-15T09:15:33

## research
- **Status:** ✅ Available
- **Path:** `/Users/username/Documents/research-vault`
- **Notes:** 543
- **Last Modified:** 2024-01-14T16:22:11

## Search Configuration
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Device:** cpu
- **Total Notes:** 1,790
```

### Parameters

None required.

---

## 4. search-vault

**Purpose**: Traditional keyword search through filenames and optionally file content.

### Filename Search

```json
{
  "tool": "search-vault",
  "arguments": {
    "query": "meeting"
  }
}
```

### Content Search

```json
{
  "tool": "search-vault",
  "arguments": {
    "query": "database optimization",
    "search_content": true,
    "limit": 10,
    "vault": "technical"
  }
}
```

### Example Response

```
Found 5 results in vault 'personal' for 'meeting':

1. **meetings/weekly-standup.md** (name match) (2,341 bytes)
2. **projects/meeting-notes-app.md** (name match) (1,892 bytes)
3. **daily/2024-01-15.md** (content match) (743 bytes)
   > Today's meeting covered project status, upcoming deadlines, and resource allocation...
```

### Parameters

- **query** (required): Search term for filenames or content
- **vault** (optional): Vault name to search in
- **search_content** (optional): Whether to search within file content (default: false)
- **limit** (optional): Maximum number of results (1-100, default: 20)

---

## 5. search-graph

**Purpose**: Explore note relationships and connections through the knowledge graph.

### Basic Graph Search

```json
{
  "tool": "search-graph",
  "arguments": {
    "query_note_path": "projects/jarvis-assistant.md"
  }
}
```

### Deep Graph Traversal

```json
{
  "tool": "search-graph",
  "arguments": {
    "query_note_path": "concepts/machine-learning.md",
    "depth": 2
  }
}
```

### Example Response

```
# Knowledge Graph for 'projects/jarvis-assistant.md'
**Traversal Depth:** 1
**Nodes Found:** 4
**Relationships Found:** 6

## 🎯 Center Node
**Jarvis Assistant** (`projects/jarvis-assistant.md`)
  Tags: project, mcp, ai-tools

## 🔗 Connected Notes
1. **MCP Protocol** (`concepts/mcp-protocol.md`)
   Tags: protocol, ai, integration
2. **Claude Desktop** (`tools/claude-desktop.md`)
   Tags: ai, desktop, claude
3. **Vector Search** (`concepts/vector-search.md`)
   Tags: search, vectors, ai

## 🌐 Relationships
### IMPLEMENTS
- **Jarvis Assistant** → **MCP Protocol**

### INTEGRATES_WITH
- **Jarvis Assistant** → **Claude Desktop**

### USES
- **Jarvis Assistant** → **Vector Search**

## 📊 Graph Structure
- **Total Connections:** 6
- **Unique Relationship Types:** 3
- **Relationship Distribution:**
  - USES: 2
  - IMPLEMENTS: 2
  - INTEGRATES_WITH: 2
```

### Parameters

- **query_note_path** (required): Path to the note to use as center of search
- **depth** (optional): How many relationship levels to traverse (default: 1)

---

## 6. search-combined

**Purpose**: Perform a combined semantic and keyword search across vault content.

### Basic Usage

```json
{
  "tool": "search-combined",
  "arguments": {
    "query": "project management best practices"
  }
}
```

### Advanced Usage

```json
{
  "tool": "search-combined",
  "arguments": {
    "query": "agile methodologies",
    "limit": 20,
    "vault": "work",
    "search_content": true
  }
}
```

### Example Response

```
Found 5 combined results for 'project management best practices':

1. **[SEMANTIC] projects/agile-scrum-guide.md** (vault: work, score: 0.912)
2. **[KEYWORD] meetings/daily-standup-notes.md** (name match) (1,234 bytes)
   > ...daily standup meetings are a core agile practice...
3. **[SEMANTIC] concepts/kanban-principles.md** (vault: work, score: 0.876)
4. **[KEYWORD] documents/project-charter-template.md** (content match) (3,456 bytes)
   > ...this template outlines best practices for project initiation...
5. **[SEMANTIC] workflows/project-planning.md** (vault: work, score: 0.850)
```

### Parameters

- **query** (required): Natural language search query.
- **limit** (optional): Maximum number of results to return (1-50, default: 10).
- **vault** (optional): Specific vault to search within.
- **search_content** (optional): Whether to include keyword search within file content (default: `true`).

---

## 7. get-health-status

**Purpose**: Get the current health status of all Jarvis Assistant services.

### Usage

```json
{
  "tool": "get-health-status",
  "arguments": {}
}
```

### Example Response

```json
{
  "overall_status": "HEALTHY",
  "services": [
    {
      "service": "Neo4j",
      "status": "HEALTHY",
      "details": "Successfully connected to Neo4j."
    },
    {
      "service": "VectorDB",
      "status": "HEALTHY",
      "details": "Successfully connected to Vector Database."
    },
    {
      "service": "Vault",
      "status": "HEALTHY",
      "details": "Vault found at /Users/username/Documents/MyVault."
    }
  ]
}
```

### Parameters

None required.

---

## Error Handling

All tools include comprehensive error handling:

### Common Error Responses

```
Error: Query parameter is required
Error: Unknown vault 'invalid-vault'
Error: Path parameter is required
Error reading note: File not found
Graph search is currently unavailable: Neo4j connection failed
```

### Service Availability

- **Vector Search**: Always available when database is indexed
- **Vault Reading**: Available when vault paths are accessible
- **Graph Search**: Requires Neo4j connection and indexed graph data

---

## Integration Examples

### Claude Desktop Usage

When using with Claude Desktop, these tools are automatically available:

```
Tell me about semantic search techniques.
```

Claude will use `search-semantic` to find relevant content and can follow up with `read-note` to get full details.

### Chained Operations

Common patterns include:

1. **Search → Read**: Use semantic search to find relevant notes, then read specific ones
2. **Graph → Read**: Use graph search to explore connections, then read connected notes
3. **List → Search**: Check vault status, then search within specific vaults

---

## Next Steps

- [Common Workflows](common-workflows.md) - Real-world usage patterns
- [Advanced Queries](advanced-queries.md) - Complex search techniques
- [Configuration Reference](../06-reference/configuration-reference.md) - Setup details