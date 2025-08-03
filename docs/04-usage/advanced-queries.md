# Advanced Queries

This guide covers sophisticated search techniques and power user patterns for Jarvis Assistant. These examples demonstrate how to leverage the full capabilities of semantic search, graph exploration, and multi-tool workflows.

## Quick Navigation

- [Semantic Search Mastery](#semantic-search-mastery)
- [Graph Search Patterns](#graph-search-patterns)
- [Multi-Tool Workflows](#multi-tool-workflows)
- [Performance Optimization](#performance-optimization)
- [Edge Cases and Troubleshooting](#edge-cases-and-troubleshooting)

---

## Semantic Search Mastery

### Precision Search with Similarity Thresholds

**Use Case**: Finding only highly relevant content while filtering out tangential matches.

```json
{
  "tool": "search-semantic",
  "arguments": {
    "query": "neural network backpropagation algorithms",
    "limit": 10,
    "similarity_threshold": 0.85
  }
}
```

**Advanced Pattern**: Iterative threshold adjustment

```json
// Start with high precision
{"tool": "search-semantic", "arguments": {"query": "deep learning optimization", "similarity_threshold": 0.9, "limit": 5}}

// If too few results, gradually lower threshold
{"tool": "search-semantic", "arguments": {"query": "deep learning optimization", "similarity_threshold": 0.75, "limit": 15}}

// For broad discovery
{"tool": "search-semantic", "arguments": {"query": "deep learning optimization", "similarity_threshold": 0.6, "limit": 25}}
```

### Concept Bridging Searches

**Use Case**: Finding connections between seemingly unrelated concepts.

```json
{
  "tool": "search-semantic",
  "arguments": {
    "query": "productivity systems knowledge management cognitive load",
    "limit": 20,
    "similarity_threshold": 0.7
  }
}
```

**Multi-Concept Pattern**: Exploring concept intersections

```json
// Primary concept
{"tool": "search-semantic", "arguments": {"query": "project management methodologies agile scrum", "limit": 15}}

// Related concept
{"tool": "search-semantic", "arguments": {"query": "team collaboration communication tools", "limit": 15}}

// Intersection concept
{"tool": "search-semantic", "arguments": {"query": "agile team collaboration communication", "limit": 10, "similarity_threshold": 0.8}}
```

### Contextual Semantic Search

**Use Case**: Finding content within specific knowledge domains.

```json
{
  "tool": "search-semantic",
  "arguments": {
    "query": "optimization techniques machine learning hyperparameter tuning",
    "vault": "research",
    "limit": 15,
    "similarity_threshold": 0.75
  }
}
```

**Domain-Specific Pattern**: Comparing across knowledge domains

```json
// Technical domain
{"tool": "search-semantic", "arguments": {"query": "system architecture patterns", "vault": "technical", "limit": 10}}

// Business domain
{"tool": "search-semantic", "arguments": {"query": "system architecture patterns", "vault": "business", "limit": 10}}

// Personal domain
{"tool": "search-semantic", "arguments": {"query": "system architecture patterns", "vault": "personal", "limit": 10}}
```

---

## Graph Search Patterns

### Deep Relationship Exploration

**Use Case**: Understanding complex knowledge networks and discovering distant connections.

```json
{
  "tool": "search-graph",
  "arguments": {
    "query_note_path": "concepts/artificial-intelligence.md",
    "depth": 3
  }
}
```

**Progressive Depth Pattern**: Building understanding layer by layer

```json
// Layer 1: Direct connections
{"tool": "search-graph", "arguments": {"query_note_path": "projects/jarvis-assistant.md", "depth": 1}}

// Layer 2: Secondary connections
{"tool": "search-graph", "arguments": {"query_note_path": "projects/jarvis-assistant.md", "depth": 2}}

// Layer 3: Tertiary connections (use sparingly)
{"tool": "search-graph", "arguments": {"query_note_path": "projects/jarvis-assistant.md", "depth": 3}}
```

### Hub Discovery

**Use Case**: Finding central nodes in your knowledge graph that connect many concepts.

```json
// Test multiple potential hubs
{"tool": "search-graph", "arguments": {"query_note_path": "concepts/productivity.md", "depth": 1}}
{"tool": "search-graph", "arguments": {"query_note_path": "concepts/learning.md", "depth": 1}}
{"tool": "search-graph", "arguments": {"query_note_path": "concepts/technology.md", "depth": 1}}
```

**Hub Analysis Pattern**: Identifying knowledge centers

```json
// High-level concept
{"tool": "search-graph", "arguments": {"query_note_path": "concepts/artificial-intelligence.md", "depth": 1}}

// If it returns many connections, it's a hub
// Follow up with specific branches
{"tool": "search-graph", "arguments": {"query_note_path": "ai/machine-learning.md", "depth": 1}}
{"tool": "search-graph", "arguments": {"query_note_path": "ai/neural-networks.md", "depth": 1}}
```

### Relationship Type Analysis

**Use Case**: Understanding how different concepts relate to each other.

```json
{
  "tool": "search-graph",
  "arguments": {
    "query_note_path": "methodologies/agile-development.md",
    "depth": 2
  }
}
```

**Pattern**: Look for relationship distribution in results:
- **IMPLEMENTS**: Concrete implementations of concepts
- **USES**: Tool/technique usage patterns
- **RELATES_TO**: General conceptual connections
- **DEPENDS_ON**: Dependency relationships

---

## Multi-Tool Workflows

### Comprehensive Topic Research

**Use Case**: Exhaustive research on a complex topic using all available tools.

```json
// Step 1: Semantic discovery
{"tool": "search-semantic", "arguments": {"query": "knowledge management systems information architecture", "limit": 20}}

// Step 2: Traditional keyword search
{"tool": "search-vault", "arguments": {"query": "knowledge management", "search_content": true, "limit": 15}}

// Step 3: Graph exploration from key results
{"tool": "search-graph", "arguments": {"query_note_path": "systems/pkm-system.md", "depth": 2}}

// Step 4: Deep reading of central documents
{"tool": "read-note", "arguments": {"path": "systems/pkm-system.md"}}

// Step 5: Follow connections
{"tool": "read-note", "arguments": {"path": "tools/obsidian-setup.md"}}
```

### Validation and Cross-Reference

**Use Case**: Verifying information across multiple sources and perspectives.

```json
// Primary source
{"tool": "search-semantic", "arguments": {"query": "agile project management principles", "limit": 10, "similarity_threshold": 0.8}}

// Alternative perspectives
{"tool": "search-vault", "arguments": {"query": "agile criticism challenges", "search_content": true, "limit": 10}}

// Implementation examples
{"tool": "search-vault", "arguments": {"query": "agile implementation", "search_content": true, "limit": 10}}

// Relationship mapping
{"tool": "search-graph", "arguments": {"query_note_path": "methodologies/agile.md", "depth": 2}}
```

### Knowledge Gap Analysis

**Use Case**: Identifying areas where your knowledge base lacks depth or connections.

```json
// Broad topic search
{"tool": "search-semantic", "arguments": {"query": "data science machine learning", "limit": 25}}

// Check graph connections for each result
{"tool": "search-graph", "arguments": {"query_note_path": "data/preprocessing.md", "depth": 1}}
{"tool": "search-graph", "arguments": {"query_note_path": "ml/supervised-learning.md", "depth": 1}}
{"tool": "search-graph", "arguments": {"query_note_path": "statistics/probability.md", "depth": 1}}

// Notes with few connections indicate potential gaps
```

---

## Performance Optimization

### Efficient Search Strategies

**Use Case**: Getting optimal results with minimal queries.

```json
// Start with moderate precision
{"tool": "search-semantic", "arguments": {"query": "machine learning optimization", "limit": 15, "similarity_threshold": 0.75}}

// If too many results, increase precision
{"tool": "search-semantic", "arguments": {"query": "machine learning optimization", "limit": 10, "similarity_threshold": 0.85}}

// If too few results, decrease precision
{"tool": "search-semantic", "arguments": {"query": "machine learning optimization", "limit": 20, "similarity_threshold": 0.65}}
```

### Vault-Specific Optimization

**Use Case**: Leveraging vault organization for better search performance.

```json
// Technical queries in technical vault
{"tool": "search-semantic", "arguments": {"query": "API design patterns", "vault": "technical", "limit": 10}}

// Research queries in research vault
{"tool": "search-semantic", "arguments": {"query": "academic paper synthesis", "vault": "research", "limit": 10}}

// Cross-vault comparison
{"tool": "search-semantic", "arguments": {"query": "productivity systems", "vault": "personal", "limit": 10}}
{"tool": "search-semantic", "arguments": {"query": "productivity systems", "vault": "business", "limit": 10}}
```

### Smart Limit Management

**Use Case**: Optimizing result counts for different search purposes.

```json
// Quick exploration (5-10 results)
{"tool": "search-semantic", "arguments": {"query": "new topic exploration", "limit": 8}}

// Research phase (15-25 results)
{"tool": "search-semantic", "arguments": {"query": "comprehensive research topic", "limit": 20}}

// Comprehensive survey (30-50 results)
{"tool": "search-semantic", "arguments": {"query": "exhaustive topic survey", "limit": 40}}
```

---

## Edge Cases and Troubleshooting

### Handling Ambiguous Queries

**Use Case**: Dealing with terms that have multiple meanings.

```json
// Ambiguous term
{"tool": "search-semantic", "arguments": {"query": "python", "limit": 15}}

// Add context to disambiguate
{"tool": "search-semantic", "arguments": {"query": "python programming language", "limit": 15}}
{"tool": "search-semantic", "arguments": {"query": "python snake animal", "limit": 15}}

// Domain-specific search
{"tool": "search-semantic", "arguments": {"query": "python", "vault": "programming", "limit": 15}}
```

### Recovering from Poor Results

**Use Case**: When initial searches don't return expected results.

```json
// Initial search fails
{"tool": "search-semantic", "arguments": {"query": "complex technical term", "limit": 10}}

// Try synonyms and related terms
{"tool": "search-semantic", "arguments": {"query": "related concept alternative term", "limit": 10}}

// Fall back to traditional search
{"tool": "search-vault", "arguments": {"query": "exact phrase", "search_content": true, "limit": 10}}

// Try filename matching
{"tool": "search-vault", "arguments": {"query": "partial filename", "limit": 10}}
```

### Graph Search Troubleshooting

**Use Case**: When graph search isn't returning expected connections.

```json
// Verify note exists
{"tool": "read-note", "arguments": {"path": "concepts/target-concept.md"}}

// Check with different depth
{"tool": "search-graph", "arguments": {"query_note_path": "concepts/target-concept.md", "depth": 1}}
{"tool": "search-graph", "arguments": {"query_note_path": "concepts/target-concept.md", "depth": 2}}

// Try related notes
{"tool": "search-semantic", "arguments": {"query": "target concept", "limit": 10}}
// Then use results for graph search
{"tool": "search-graph", "arguments": {"query_note_path": "found/related-note.md", "depth": 1}}
```

---

## Advanced Pattern Combinations

### Semantic → Graph → Semantic Pattern

**Use Case**: Using semantic search to find entry points, then exploring connections, then finding similar concepts.

```json
// 1. Find entry points
{"tool": "search-semantic", "arguments": {"query": "distributed systems architecture", "limit": 10, "similarity_threshold": 0.8}}

// 2. Explore connections from best match
{"tool": "search-graph", "arguments": {"query_note_path": "systems/microservices.md", "depth": 2}}

// 3. Find similar concepts to discovered connections
{"tool": "search-semantic", "arguments": {"query": "service mesh kubernetes", "limit": 15}}
```

### Validation Triangle

**Use Case**: Confirming information through multiple search approaches.

```json
// Semantic confirmation
{"tool": "search-semantic", "arguments": {"query": "specific concept or claim", "limit": 10, "similarity_threshold": 0.8}}

// Traditional confirmation
{"tool": "search-vault", "arguments": {"query": "key terms exact phrase", "search_content": true, "limit": 10}}

// Graph confirmation
{"tool": "search-graph", "arguments": {"query_note_path": "authoritative/source-note.md", "depth": 1}}
```

### Iterative Refinement

**Use Case**: Progressively refining search results through multiple iterations.

```json
// Iteration 1: Broad discovery
{"tool": "search-semantic", "arguments": {"query": "machine learning", "limit": 25, "similarity_threshold": 0.6}}

// Iteration 2: Focus on interesting subcategory
{"tool": "search-semantic", "arguments": {"query": "deep learning neural networks", "limit": 15, "similarity_threshold": 0.75}}

// Iteration 3: Specific technique
{"tool": "search-semantic", "arguments": {"query": "convolutional neural networks image recognition", "limit": 10, "similarity_threshold": 0.85}}

// Iteration 4: Implementation details
{"tool": "search-vault", "arguments": {"query": "CNN implementation", "search_content": true, "limit": 10}}
```

---

## Expert Tips

### Search Query Optimization

- **Use specific terminology**: "gradient descent optimization" vs "learning algorithms"
- **Combine concepts**: "project management agile methodology" vs "project management"
- **Include context**: "python machine learning" vs "python"
- **Use domain language**: Technical terms in technical vaults, business terms in business vaults

### Similarity Threshold Guidelines

- **0.9-1.0**: Near-exact matches only
- **0.8-0.9**: Highly relevant, focused results
- **0.7-0.8**: Good balance of relevance and diversity
- **0.6-0.7**: Broader discovery, may include tangential results
- **0.5-0.6**: Exploratory searches, high recall

### Graph Search Depth Strategy

- **Depth 1**: Direct relationships, immediate connections
- **Depth 2**: Secondary relationships, recommended for most use cases
- **Depth 3**: Tertiary relationships, use sparingly due to complexity
- **Depth 4+**: Rarely useful, may return noise

### Multi-Vault Strategies

- **Vault-specific searches**: Better performance and relevance
- **Cross-vault comparison**: Understanding different perspectives
- **Unified searches**: Discovering unexpected connections across domains

---

## Next Steps

- [API Examples](api-examples.md) - Basic tool usage reference
- [Common Workflows](common-workflows.md) - Standard usage patterns
- [Configuration Reference](../06-reference/configuration-reference.md) - Advanced setup options
- [Performance Tuning](../07-maintenance/performance-tuning.md) - Optimization techniques