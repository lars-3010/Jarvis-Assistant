# API Reference

Complete reference documentation for all Jarvis Assistant MCP tools, parameters, and responses. This guide provides detailed specifications for developers and advanced users.

## Quick Navigation

- [MCP Tools Overview](#mcp-tools-overview)
- [search-semantic](#search-semantic)
- [read-note](#read-note)
- [list-vaults](#list-vaults)
- [search-vault](#search-vault)
- [search-graph](#search-graph)
- [Error Responses](#error-responses)
- [Data Models](#data-models)

---

## MCP Tools Overview

Jarvis Assistant provides 5 MCP tools for Claude Desktop integration:

| Tool | Purpose | Input Requirements | Output Type |
|------|---------|-------------------|-------------|
| `search-semantic` | Semantic vector search | `query` (required) | Search results with similarity scores |
| `read-note` | Read specific file | `path` (required) | File content with metadata |
| `list-vaults` | List all vaults | None | Vault statistics and status |
| `search-vault` | Traditional keyword search | `query` (required) | Search results with match types |
| `search-graph` | Graph relationship search | `query_note_path` (required) | Graph nodes and relationships |

---

## search-semantic

Performs semantic search using natural language queries and vector embeddings.

### Schema

```json
{
  "name": "search-semantic",
  "description": "Perform semantic search across vault content using natural language queries",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query"
      },
      "limit": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 50,
        "description": "Maximum number of results to return"
      },
      "vault": {
        "type": "string",
        "description": "Optional vault name to search within"
      },
      "similarity_threshold": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Minimum similarity score (0.0-1.0)"
      }
    },
    "required": ["query"]
  }
}
```

### Parameters

#### query (required)
- **Type**: `string`
- **Description**: Natural language search query
- **Example**: `"machine learning algorithms"`
- **Constraints**: Must be non-empty string

#### limit (optional)
- **Type**: `integer`
- **Default**: `10`
- **Range**: `1` to `50`
- **Description**: Maximum number of results to return
- **Example**: `15`

#### vault (optional)
- **Type**: `string`
- **Description**: Specific vault name to search within
- **Example**: `"research"`
- **Behavior**: If not specified, searches all available vaults

#### similarity_threshold (optional)
- **Type**: `number`
- **Range**: `0.0` to `1.0`
- **Description**: Minimum similarity score for results
- **Example**: `0.75`
- **Behavior**: Filters out results below threshold

### Response Format

```json
{
  "type": "text",
  "text": "Found 3 results for 'machine learning':\n\n1. **ai/neural-networks.md** (vault: research, score: 0.892)\n2. **projects/ml-project.md** (vault: personal, score: 0.834)\n3. **notes/deep-learning.md** (vault: research, score: 0.767)"
}
```

### Response Structure

- **Header**: `Found {count} results for '{query}':`
- **Results**: Numbered list with format: `{index}. **{path}** (vault: {vault_name}, score: {score})`
- **Sorting**: Results sorted by similarity score (descending)
- **Empty Results**: `No results found for query: '{query}'`

### Example Usage

```json
{
  "tool": "search-semantic",
  "arguments": {
    "query": "project management techniques",
    "limit": 5,
    "vault": "business",
    "similarity_threshold": 0.8
  }
}
```

### Error Conditions

- **Empty Query**: `Error: Query parameter is required`
- **Invalid Vault**: `Error: Unknown vault '{vault_name}'`
- **Search Failure**: `Search error: {error_message}`

---

## read-note

Reads the complete content of a specific note file from a vault.

### Schema

```json
{
  "name": "read-note",
  "description": "Read the content of a specific note from a vault",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Path to the note relative to vault root"
      },
      "vault": {
        "type": "string",
        "description": "Vault name (uses first available if not specified)"
      }
    },
    "required": ["path"]
  }
}
```

### Parameters

#### path (required)
- **Type**: `string`
- **Description**: Path to the note relative to vault root
- **Example**: `"projects/jarvis-assistant.md"`
- **Format**: Unix-style path separator (`/`)
- **Constraints**: Must be valid file path within vault

#### vault (optional)
- **Type**: `string`
- **Description**: Vault name to read from
- **Example**: `"research"`
- **Behavior**: Uses first available vault if not specified

### Response Format

```json
{
  "type": "text",
  "text": "# projects/jarvis-assistant.md\n\n**Size:** 4,832 bytes  \n**Modified:** 2024-01-15T14:30:22  \n\n---\n\n# Jarvis Assistant Project\n\nThis document outlines the development of the Jarvis Assistant MCP server...\n[full file content follows]"
}
```

### Response Structure

- **Header**: `# {path}`
- **Metadata**: Size and modification date
- **Separator**: `---`
- **Content**: Full file content as-is

### Example Usage

```json
{
  "tool": "read-note",
  "arguments": {
    "path": "concepts/artificial-intelligence.md",
    "vault": "research"
  }
}
```

### Error Conditions

- **Missing Path**: `Error: Path parameter is required`
- **Invalid Vault**: `Error: Unknown vault '{vault_name}'`
- **File Not Found**: `Error reading note: File not found`
- **No Vaults**: `Error: No vaults available`

---

## list-vaults

Lists all configured vaults with their statistics and operational status.

### Schema

```json
{
  "name": "list-vaults",
  "description": "List all available vaults and their statistics",
  "inputSchema": {
    "type": "object",
    "properties": {}
  }
}
```

### Parameters

None required.

### Response Format

```json
{
  "type": "text",
  "text": "# Available Vaults\n\n## personal\n- **Status:** ✅ Available\n- **Path:** `/Users/username/Documents/personal-vault`\n- **Notes:** 1,247\n- **Last Modified:** 2024-01-15T09:15:33\n\n## research\n- **Status:** ✅ Available\n- **Path:** `/Users/username/Documents/research-vault`\n- **Notes:** 543\n- **Last Modified:** 2024-01-14T16:22:11\n\n## Search Configuration\n- **Model:** sentence-transformers/all-MiniLM-L6-v2\n- **Device:** cpu\n- **Total Notes:** 1,790"
}
```

### Response Structure

- **Header**: `# Available Vaults`
- **Vault Sections**: Each vault as H2 header with:
  - **Status**: ✅ Available / ❌ Unavailable
  - **Path**: Full filesystem path
  - **Notes**: Count of indexed notes
  - **Last Modified**: ISO timestamp of latest modification
- **Search Configuration**: Model info and total note count

### Example Usage

```json
{
  "tool": "list-vaults",
  "arguments": {}
}
```

### Error Conditions

- **Service Error**: `Error listing vaults: {error_message}`

---

## search-vault

Performs traditional keyword search through filenames and optionally file content.

### Schema

```json
{
  "name": "search-vault",
  "description": "Search for files in vault by filename or content (traditional search)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search term for filenames or content"
      },
      "vault": {
        "type": "string",
        "description": "Vault name to search in"
      },
      "search_content": {
        "type": "boolean",
        "default": false,
        "description": "Whether to search within file content"
      },
      "limit": {
        "type": "integer",
        "default": 20,
        "minimum": 1,
        "maximum": 100,
        "description": "Maximum number of results"
      }
    },
    "required": ["query"]
  }
}
```

### Parameters

#### query (required)
- **Type**: `string`
- **Description**: Search term for filenames or content
- **Example**: `"meeting notes"`
- **Behavior**: Case-insensitive partial matching

#### vault (optional)
- **Type**: `string`
- **Description**: Vault name to search in
- **Example**: `"personal"`
- **Behavior**: Uses first available vault if not specified

#### search_content (optional)
- **Type**: `boolean`
- **Default**: `false`
- **Description**: Whether to search within file content
- **Behavior**: 
  - `false`: Only searches filenames
  - `true`: Searches both filenames and content

#### limit (optional)
- **Type**: `integer`
- **Default**: `20`
- **Range**: `1` to `100`
- **Description**: Maximum number of results to return

### Response Format

```json
{
  "type": "text",
  "text": "Found 3 results in vault 'personal' for 'meeting':\n\n1. **meetings/weekly-standup.md** (name match) (2,341 bytes)\n2. **projects/meeting-notes-app.md** (name match) (1,892 bytes)\n3. **daily/2024-01-15.md** (content match) (743 bytes)\n   > Today's meeting covered project status, upcoming deadlines..."
}
```

### Response Structure

- **Header**: `Found {count} results in vault '{vault_name}' for '{query}':`
- **Results**: Numbered list with:
  - **Path**: `**{path}**`
  - **Match Type**: `(name match)` or `(content match)`
  - **Size**: `({size} bytes)`
  - **Preview**: Content preview for content matches (truncated)

### Example Usage

```json
{
  "tool": "search-vault",
  "arguments": {
    "query": "machine learning",
    "vault": "research",
    "search_content": true,
    "limit": 10
  }
}
```

### Error Conditions

- **Missing Query**: `Error: Query parameter is required`
- **Invalid Vault**: `Error: Unknown vault '{vault_name}'`
- **Search Failure**: `Error searching vault: {error_message}`
- **No Results**: `No results found in {search_type} for query: '{query}'`

---

## search-graph

Searches for notes and their relationships in the knowledge graph.

### Schema

```json
{
  "name": "search-graph",
  "description": "Search for notes and their relationships in the knowledge graph",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query_note_path": {
        "type": "string",
        "description": "The path to the note to use as the center of the search"
      },
      "depth": {
        "type": "integer",
        "default": 1,
        "description": "How many relationship levels to traverse"
      }
    },
    "required": ["query_note_path"]
  }
}
```

### Parameters

#### query_note_path (required)
- **Type**: `string`
- **Description**: Path to the note to use as center of search
- **Example**: `"concepts/artificial-intelligence.md"`
- **Format**: Unix-style path separator (`/`)

#### depth (optional)
- **Type**: `integer`
- **Default**: `1`
- **Description**: How many relationship levels to traverse
- **Range**: Typically `1` to `3`
- **Behavior**: Higher values find more distant connections

### Response Format

```json
{
  "type": "text",
  "text": "# Knowledge Graph for 'concepts/artificial-intelligence.md'\n**Traversal Depth:** 1\n**Nodes Found:** 4\n**Relationships Found:** 6\n\n## 🎯 Center Node\n**Artificial Intelligence** (`concepts/artificial-intelligence.md`)\n  Tags: ai, machine-learning, technology\n\n## 🔗 Connected Notes\n1. **Machine Learning** (`ai/machine-learning.md`)\n   Tags: ml, algorithms, data-science\n2. **Neural Networks** (`ai/neural-networks.md`)\n   Tags: deep-learning, ai, networks\n\n## 🌐 Relationships\n### INCLUDES\n- **Artificial Intelligence** → **Machine Learning**\n- **Machine Learning** → **Neural Networks**\n\n### USES\n- **Neural Networks** → **Deep Learning**\n\n## 📊 Graph Structure\n- **Total Connections:** 6\n- **Unique Relationship Types:** 2\n- **Relationship Distribution:**\n  - INCLUDES: 2\n  - USES: 4"
}
```

### Response Structure

- **Header**: `# Knowledge Graph for '{query_note_path}'`
- **Metadata**: Traversal depth, node count, relationship count
- **Center Node**: 🎯 The queried note with tags
- **Connected Notes**: 🔗 Related notes with tags
- **Relationships**: 🌐 Grouped by relationship type
- **Graph Structure**: 📊 Statistics and distribution

### Example Usage

```json
{
  "tool": "search-graph",
  "arguments": {
    "query_note_path": "projects/jarvis-assistant.md",
    "depth": 2
  }
}
```

### Error Conditions

- **Missing Path**: `Error: query_note_path parameter is required`
- **Graph Unavailable**: `Graph search is currently unavailable: {reason}`
- **No Results**: `No results found for query: '{query_note_path}'`
- **Connection Error**: `Graph search failed due to database connection issues`

---

## Error Responses

### Common Error Format

All errors return a single TextContent object with error information:

```json
{
  "type": "text",
  "text": "Error: {error_message}"
}
```

### Error Categories

#### Input Validation Errors
- **Missing Required Parameters**: `Error: {parameter} parameter is required`
- **Invalid Parameter Values**: `Error: {parameter} must be between {min} and {max}`
- **Invalid Parameter Types**: `Error: {parameter} must be a {type}`

#### Resource Errors
- **Unknown Vault**: `Error: Unknown vault '{vault_name}'`
- **File Not Found**: `Error reading note: File not found`
- **No Vaults Available**: `Error: No vaults available`

#### Service Errors
- **Database Connection**: `Search error: Database connection failed`
- **Graph Service**: `Graph search is currently unavailable: Neo4j connection failed`
- **Encoding Error**: `Search error: Failed to encode query`

#### System Errors
- **Unexpected Errors**: `Error executing {tool_name}: {error_message}`
- **Service Unavailable**: `{service_name} is currently unavailable`

---

## Data Models

### SearchResult

Represents a search result from semantic or traditional search.

```typescript
interface SearchResult {
  path: string;              // Relative path to file
  similarity_score: number;  // Similarity score (0.0-1.0)
  vault_name: string;        // Name of vault containing file
  content_preview?: string;  // Optional content preview
}
```

### GraphNode

Represents a node in the knowledge graph.

```typescript
interface GraphNode {
  id: string;          // Unique node identifier
  label: string;       // Display name
  path: string;        // File path
  tags?: string[];     // Associated tags
  center?: boolean;    // True if this is the center node
}
```

### GraphRelationship

Represents a relationship between nodes in the knowledge graph.

```typescript
interface GraphRelationship {
  source: string;        // Source node ID
  target: string;        // Target node ID
  type: string;          // Relationship type
  original_type?: string; // Original relationship type
}
```

### GraphResult

Complete graph search result structure.

```typescript
interface GraphResult {
  nodes: GraphNode[];
  relationships: GraphRelationship[];
}
```

### VaultInfo

Information about a configured vault.

```typescript
interface VaultInfo {
  name: string;           // Vault name
  path: string;           // Filesystem path
  status: 'available' | 'unavailable';
  note_count: number;     // Number of indexed notes
  latest_modified?: number; // Unix timestamp of latest modification
}
```

### ModelInfo

Information about the search model and configuration.

```typescript
interface ModelInfo {
  encoder_info: {
    model_name: string;   // Model identifier
    device: string;       // Execution device (cpu/gpu)
  };
  database_note_count: number; // Total notes in database
}
```

---

## Response Examples

### Successful Search Response

```
Found 3 results for 'machine learning':

1. **ai/neural-networks.md** (vault: research, score: 0.892)
2. **projects/ml-project.md** (vault: personal, score: 0.834)
3. **notes/deep-learning.md** (vault: research, score: 0.767)
```

### Empty Search Response

```
No results found for query: 'quantum computing'
```

### File Read Response

```
# projects/jarvis-assistant.md

**Size:** 4,832 bytes  
**Modified:** 2024-01-15T14:30:22  

---

# Jarvis Assistant Project

This document outlines the development of the Jarvis Assistant MCP server for Claude Desktop integration.

## Overview

Jarvis Assistant provides semantic search and graph exploration capabilities...
```

### Vault List Response

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

### Graph Search Response

```
# Knowledge Graph for 'concepts/artificial-intelligence.md'
**Traversal Depth:** 2
**Nodes Found:** 7
**Relationships Found:** 12

## 🎯 Center Node
**Artificial Intelligence** (`concepts/artificial-intelligence.md`)
  Tags: ai, technology, computer-science

## 🔗 Connected Notes
1. **Machine Learning** (`ai/machine-learning.md`)
   Tags: ml, algorithms, data-science
2. **Neural Networks** (`ai/neural-networks.md`)
   Tags: deep-learning, ai, networks
3. **Natural Language Processing** (`ai/nlp.md`)
   Tags: nlp, language, processing

## 🌐 Relationships
### INCLUDES
- **Artificial Intelligence** → **Machine Learning**
- **Artificial Intelligence** → **Neural Networks**
- **Machine Learning** → **Deep Learning**

### USES
- **Neural Networks** → **Backpropagation**
- **NLP** → **Transformers**

## 📊 Graph Structure
- **Total Connections:** 12
- **Unique Relationship Types:** 3
- **Relationship Distribution:**
  - INCLUDES: 5
  - USES: 4
  - RELATES_TO: 3
```

---

## Integration Notes

### Claude Desktop Integration

These tools are automatically available when the MCP server is properly configured in Claude Desktop. The tools appear in Claude's interface and can be called directly through natural language interaction.

### MCP Protocol Compliance

All tools follow MCP (Model Context Protocol) specifications:
- Proper tool registration with schemas
- Consistent parameter validation
- Standardized response formats
- Appropriate error handling

### Performance Considerations

- **Semantic Search**: First query may be slower due to model loading
- **Graph Search**: Performance depends on graph size and depth
- **File Reading**: Limited by filesystem performance
- **Vault Search**: Performance scales with vault size

---

## Next Steps

- [Configuration Reference](configuration-reference.md) - Setup and configuration options
- [Error Codes](error-codes.md) - Complete error reference
- [Usage Examples](../04-usage/api-examples.md) - Practical usage patterns
- [Advanced Queries](../04-usage/advanced-queries.md) - Complex search techniques