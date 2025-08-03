# Common Workflows

This guide covers typical user scenarios and how to accomplish them using Jarvis Assistant's MCP tools. Each workflow includes step-by-step instructions and expected results.

## Quick Navigation

- [Discovery Workflows](#discovery-workflows)
- [Research Workflows](#research-workflows)
- [Content Management](#content-management)
- [Knowledge Graph Exploration](#knowledge-graph-exploration)
- [Troubleshooting Workflows](#troubleshooting-workflows)

---

## Discovery Workflows

### I want to explore what's in my vault

**Scenario**: You want to understand what content you have available.

**Steps**:
1. **Check vault status**:
   ```json
   {"tool": "list-vaults", "arguments": {}}
   ```

2. **Browse by broad topics**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "project management", "limit": 20}}
   ```

3. **Check filename patterns**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "meeting", "limit": 15}}
   ```

**Expected Result**: Overview of vault contents, key topics, and organizational patterns.

---

### I want to find notes related to a specific concept

**Scenario**: You remember writing something about "machine learning" but don't remember the exact file.

**Steps**:
1. **Semantic search first** (finds conceptually related content):
   ```json
   {"tool": "search-semantic", "arguments": {"query": "machine learning algorithms", "limit": 10}}
   ```

2. **Traditional search as backup** (finds exact keyword matches):
   ```json
   {"tool": "search-vault", "arguments": {"query": "machine learning", "search_content": true, "limit": 10}}
   ```

3. **Read promising results**:
   ```json
   {"tool": "read-note", "arguments": {"path": "ai/neural-networks.md"}}
   ```

**Expected Result**: Comprehensive list of related notes, both conceptually similar and keyword-matched.

---

## Research Workflows

### I want to research a complex topic across multiple notes

**Scenario**: You're researching "productivity systems" and want to gather information from multiple sources.

**Steps**:
1. **Initial semantic search**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "productivity systems time management", "limit": 15}}
   ```

2. **Read key foundational notes**:
   ```json
   {"tool": "read-note", "arguments": {"path": "productivity/gtd-system.md"}}
   ```

3. **Explore related concepts**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "task prioritization methods", "limit": 10}}
   ```

4. **Find specific implementations**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "todo", "search_content": true, "limit": 10}}
   ```

**Expected Result**: Comprehensive research compilation from multiple perspectives and sources.

---

### I want to find all my notes about a specific project

**Scenario**: You need to gather all information related to a particular project.

**Steps**:
1. **Search by project name**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "jarvis-assistant", "search_content": true, "limit": 20}}
   ```

2. **Find conceptually related notes**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "MCP server development Claude integration", "limit": 15}}
   ```

3. **Explore project connections**:
   ```json
   {"tool": "search-graph", "arguments": {"query_note_path": "projects/jarvis-assistant.md", "depth": 2}}
   ```

4. **Read detailed project documentation**:
   ```json
   {"tool": "read-note", "arguments": {"path": "projects/jarvis-assistant.md"}}
   ```

**Expected Result**: Complete project overview with all related documentation and connections.

---

## Content Management

### I want to verify my vault setup is working correctly

**Scenario**: You want to ensure your Jarvis Assistant setup is functioning properly.

**Steps**:
1. **Check vault status**:
   ```json
   {"tool": "list-vaults", "arguments": {}}
   ```
   
   Look for:
   - ✅ Available status for all vaults
   - Reasonable note counts
   - Recent modification dates

2. **Test semantic search**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "test search", "limit": 5}}
   ```

3. **Test traditional search**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "test", "limit": 5}}
   ```

4. **Test note reading**:
   ```json
   {"tool": "read-note", "arguments": {"path": "[path from search results]"}}
   ```

**Expected Result**: Confirmation that all services are operational and returning expected results.

---

### I want to find duplicate or similar content

**Scenario**: You suspect you might have written about the same topic multiple times.

**Steps**:
1. **Search for specific topic**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "project planning methodologies", "limit": 20, "similarity_threshold": 0.8}}
   ```

2. **Review high-similarity results** (scores > 0.85 often indicate similar content)

3. **Read suspected duplicates**:
   ```json
   {"tool": "read-note", "arguments": {"path": "planning/agile-methods.md"}}
   ```

4. **Check for filename patterns**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "planning", "limit": 15}}
   ```

**Expected Result**: Identification of duplicate or highly similar content for consolidation.

---

## Knowledge Graph Exploration

### I want to understand how concepts connect in my knowledge base

**Scenario**: You want to explore the relationships between your notes and identify knowledge clusters.

**Steps**:
1. **Start with a central concept**:
   ```json
   {"tool": "search-graph", "arguments": {"query_note_path": "concepts/artificial-intelligence.md", "depth": 1}}
   ```

2. **Explore deeper connections**:
   ```json
   {"tool": "search-graph", "arguments": {"query_note_path": "concepts/artificial-intelligence.md", "depth": 2}}
   ```

3. **Investigate interesting connections**:
   ```json
   {"tool": "search-graph", "arguments": {"query_note_path": "tools/claude-desktop.md", "depth": 1}}
   ```

4. **Read connected notes for context**:
   ```json
   {"tool": "read-note", "arguments": {"path": "tools/claude-desktop.md"}}
   ```

**Expected Result**: Map of knowledge relationships and identification of key connecting concepts.

---

### I want to find orphaned notes (notes with few connections)

**Scenario**: You want to identify notes that might need better integration into your knowledge system.

**Steps**:
1. **Search for notes by broad topics**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "random thoughts ideas", "limit": 20}}
   ```

2. **Check graph connections for each result**:
   ```json
   {"tool": "search-graph", "arguments": {"query_note_path": "thoughts/random-idea-2024.md", "depth": 1}}
   ```

3. **Look for notes with 0-1 connections**

4. **Read orphaned notes to understand integration opportunities**:
   ```json
   {"tool": "read-note", "arguments": {"path": "thoughts/random-idea-2024.md"}}
   ```

**Expected Result**: List of potentially orphaned notes that could benefit from better linking.

---

## Troubleshooting Workflows

### I can't find a note I know exists

**Scenario**: You're certain a note exists but searches aren't finding it.

**Steps**:
1. **Try multiple search strategies**:
   
   **By filename**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "partial-filename", "limit": 20}}
   ```
   
   **By content**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "unique phrase from note", "search_content": true, "limit": 20}}
   ```
   
   **By concept**:
   ```json
   {"tool": "search-semantic", "arguments": {"query": "concept from the note", "limit": 20}}
   ```

2. **Check all vaults**:
   ```json
   {"tool": "list-vaults", "arguments": {}}
   ```

3. **Search in specific vault**:
   ```json
   {"tool": "search-vault", "arguments": {"query": "search term", "vault": "specific-vault", "search_content": true}}
   ```

**Expected Result**: Location of the missing note or confirmation it needs to be re-indexed.

---

### I want to verify my graph search is working

**Scenario**: Graph search isn't returning expected results.

**Steps**:
1. **Check a known note with connections**:
   ```json
   {"tool": "search-graph", "arguments": {"query_note_path": "projects/main-project.md", "depth": 1}}
   ```

2. **If no results, check vault status**:
   ```json
   {"tool": "list-vaults", "arguments": {}}
   ```

3. **Try reading the note directly**:
   ```json
   {"tool": "read-note", "arguments": {"path": "projects/main-project.md"}}
   ```

4. **If graph search fails, you'll see an error message like**:
   ```
   Graph search is currently unavailable: Neo4j connection failed
   ```

**Expected Result**: Either working graph results or clear error messages indicating what needs to be fixed.

---

## Advanced Workflow Patterns

### Research → Connect → Expand Pattern

1. **Research**: Use semantic search to find initial relevant notes
2. **Connect**: Use graph search to understand relationships
3. **Expand**: Follow connections to discover new relevant content
4. **Read**: Deep dive into the most promising discoveries

### Validate → Search → Verify Pattern

1. **Validate**: Check vault status and service availability
2. **Search**: Use multiple search strategies (semantic, traditional, graph)
3. **Verify**: Read results to confirm relevance and accuracy

### Breadth → Depth → Integration Pattern

1. **Breadth**: Start with broad semantic searches
2. **Depth**: Focus on specific areas using traditional search
3. **Integration**: Use graph search to understand connections

---

## Tips for Effective Workflows

### Search Strategy Tips

- **Start broad, then narrow**: Begin with semantic search for concepts, then use traditional search for specifics
- **Use multiple search types**: Each tool finds different types of relevant content
- **Adjust similarity thresholds**: Higher thresholds (0.7-0.9) for focused results, lower (0.5-0.7) for broader discovery
- **Check all vaults**: Don't forget to search across different vault contexts

### Graph Exploration Tips

- **Start with central concepts**: Begin graph exploration from well-connected notes
- **Use depth gradually**: Start with depth 1, then increase to 2-3 for deeper exploration
- **Follow interesting connections**: Graph search often reveals unexpected but valuable relationships

### Performance Tips

- **Limit results appropriately**: Use lower limits (5-10) for initial exploration, higher (20-50) for comprehensive searches
- **Check service availability**: Always verify vault and graph service status when troubleshooting
- **Use specific vault targeting**: Search within specific vaults for better performance and relevance

---

## Next Steps

- [API Examples](api-examples.md) - Detailed tool usage reference
- [Advanced Queries](advanced-queries.md) - Complex search techniques
- [Troubleshooting](../07-maintenance/troubleshooting.md) - Detailed problem resolution