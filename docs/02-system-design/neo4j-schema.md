# Neo4j Schema Design

*Graph database structure and relationships for knowledge discovery*

## Overview

The Neo4j graph database implements a **knowledge graph architecture** that models the rich interconnections within Obsidian vaults. Unlike traditional relational databases that store data in isolated tables, this graph approach captures the **natural connectivity** of human knowledge.

### Architectural Philosophy

The graph schema design reflects several key architectural decisions:

- **Node-Relationship Modeling**: Every piece of content (notes, headings, blocks) becomes a node with explicit relationships
- **Hierarchical Structure**: Preserves Obsidian's document structure while enabling cross-document connections  
- **Temporal Tracking**: All relationships include timestamps for change detection and temporal analysis
- **Flexible Schema**: JSON properties allow storage of arbitrary metadata without schema changes

### Integration with System Architecture

| Component | Role | File Location | Graph Interaction |
|-----------|------|---------------|-------------------|
| **Graph Service** | Query orchestration | `/src/jarvis/services/graph/` | Cypher query execution |
| **Graph Indexer** | Content → Graph conversion | `/src/jarvis/services/graph/indexer.py` | Node/relationship creation |
| **Graph Parser** | Link extraction | `/src/jarvis/services/graph/parser.py` | Relationship discovery |
| **Neo4j Adapter** | Database abstraction | `/src/jarvis/database/adapters/` | Connection management |

## Schema Architecture Overview

The graph schema implements a **multi-layered node hierarchy** that preserves both document structure and cross-document relationships:

## Node Types

### Note Node
Primary node type representing individual markdown files in the vault.

```cypher
CREATE (n:Note {
  id: "unique-note-id",
  path: "/path/to/note.md",
  title: "Note Title",
  content: "Full markdown content",
  excerpt: "First 200 characters...",
  created_at: datetime(),
  modified_at: datetime(),
  file_size: 1024,
  checksum: "sha256-hash",
  tags: ["tag1", "tag2"],
  frontmatter: {
    author: "John Doe",
    status: "draft",
    category: "research"
  }
})
```

**Properties:**
- `id`: Unique identifier (UUID or path-based)
- `path`: Absolute file path within vault
- `title`: Note title (from frontmatter or first heading)
- `content`: Full markdown content
- `excerpt`: Brief content preview
- `created_at`: File creation timestamp
- `modified_at`: Last modification timestamp
- `file_size`: Size in bytes
- `checksum`: Content hash for change detection
- `tags`: Array of tags from frontmatter and inline tags
- `frontmatter`: YAML frontmatter as nested object

### Heading Node
Represents headings within notes for fine-grained navigation.

```cypher
CREATE (h:Heading {
  id: "note-id#heading-anchor",
  note_id: "parent-note-id",
  text: "Heading Text",
  level: 2,
  anchor: "heading-anchor",
  position: 150,
  parent_heading_id: "parent-heading-id"
})
```

**Properties:**
- `id`: Unique identifier combining note and heading
- `note_id`: Reference to parent note
- `text`: Heading text content
- `level`: Heading level (1-6)
- `anchor`: URL-safe anchor for linking
- `position`: Character position in file
- `parent_heading_id`: Parent heading for hierarchy

### Block Node
Represents blocks of content (paragraphs, lists, code blocks).

```cypher
CREATE (b:Block {
  id: "note-id#block-hash",
  note_id: "parent-note-id",
  type: "paragraph",
  content: "Block content",
  position: 300,
  heading_id: "parent-heading-id"
})
```

**Properties:**
- `id`: Unique identifier for block
- `note_id`: Reference to parent note
- `type`: Block type (paragraph, list, code, quote, etc.)
- `content`: Block text content
- `position`: Character position in file
- `heading_id`: Parent heading if applicable

### Tag Node
Represents tags for categorization and filtering.

```cypher
CREATE (t:Tag {
  name: "machine-learning",
  count: 15,
  created_at: datetime(),
  description: "Notes about ML algorithms and techniques"
})
```

**Properties:**
- `name`: Tag name (normalized, lowercase)
- `count`: Number of notes with this tag
- `created_at`: First occurrence timestamp
- `description`: Optional tag description

## Relationship Types

### LINKS_TO
Direct wiki-style links between notes.

```cypher
CREATE (source:Note)-[:LINKS_TO {
  created_at: datetime(),
  context: "surrounding text context",
  anchor: "specific-heading",
  link_type: "wiki"
}]->(target:Note)
```

**Properties:**
- `created_at`: When link was created
- `context`: Surrounding text for context
- `anchor`: Specific section if linked
- `link_type`: Type of link (wiki, markdown, etc.)

### REFERENCES
Mentions or citations without direct links.

```cypher
CREATE (source:Note)-[:REFERENCES {
  created_at: datetime(),
  mention_context: "context where mentioned",
  confidence: 0.85,
  extraction_method: "natural_language_processing"
}]->(target:Note)
```

**Properties:**
- `created_at`: When reference was detected
- `mention_context`: Text context of mention
- `confidence`: Confidence score (0-1)
- `extraction_method`: How reference was found

### CONTAINS
Hierarchical relationship for headings and blocks.

```cypher
CREATE (note:Note)-[:CONTAINS {
  position: 100,
  level: 1
}]->(heading:Heading)

CREATE (heading:Heading)-[:CONTAINS {
  position: 200,
  order: 1
}]->(block:Block)
```

**Properties:**
- `position`: Position in document
- `level`: Hierarchy level (for headings)
- `order`: Sequential order within parent

### TAGGED_WITH
Connection between notes and tags.

```cypher
CREATE (note:Note)-[:TAGGED_WITH {
  created_at: datetime(),
  source: "frontmatter",
  context: "tag context if inline"
}]->(tag:Tag)
```

**Properties:**
- `created_at`: When tag was applied
- `source`: Source of tag (frontmatter, inline, etc.)
- `context`: Context if tag appears inline

### SIMILAR_TO
Computed semantic similarity between notes.

```cypher
CREATE (note1:Note)-[:SIMILAR_TO {
  similarity_score: 0.75,
  computed_at: datetime(),
  method: "sentence_transformers",
  common_concepts: ["concept1", "concept2"]
}]->(note2:Note)
```

**Properties:**
- `similarity_score`: Similarity score (0-1)
- `computed_at`: When similarity was computed
- `method`: Algorithm used for similarity
- `common_concepts`: Shared concepts or keywords

### FOLLOWS
Temporal or logical sequence between notes.

```cypher
CREATE (note1:Note)-[:FOLLOWS {
  created_at: datetime(),
  sequence_type: "chronological",
  gap_days: 5
}]->(note2:Note)
```

**Properties:**
- `created_at`: When sequence was established
- `sequence_type`: Type of sequence (chronological, logical, etc.)
- `gap_days`: Time gap between notes

## Indexing Strategy

### Primary Indexes
```cypher
-- Note path index for fast lookup
CREATE INDEX note_path_index FOR (n:Note) ON (n.path)

-- Note title index for search
CREATE INDEX note_title_index FOR (n:Note) ON (n.title)

-- Tag name index for filtering
CREATE INDEX tag_name_index FOR (t:Tag) ON (t.name)

-- Heading anchor index for navigation
CREATE INDEX heading_anchor_index FOR (h:Heading) ON (h.anchor)
```

### Composite Indexes
```cypher
-- Note modified time for change detection
CREATE INDEX note_modified_index FOR (n:Note) ON (n.modified_at)

-- Similarity score for ranking
CREATE INDEX similarity_score_index FOR ()-[r:SIMILAR_TO]-() ON (r.similarity_score)

-- Link creation time for temporal queries
CREATE INDEX link_time_index FOR ()-[r:LINKS_TO]-() ON (r.created_at)
```

### Full-Text Search
```cypher
-- Full-text search on note content
CREATE FULLTEXT INDEX note_content_index FOR (n:Note) ON EACH [n.content, n.title]

-- Full-text search on headings
CREATE FULLTEXT INDEX heading_text_index FOR (h:Heading) ON EACH [h.text]
```

## Common Query Patterns

### Find Related Notes
```cypher
// Find notes directly linked from a starting note
MATCH (start:Note {path: $start_path})-[:LINKS_TO]->(related:Note)
RETURN related
ORDER BY related.modified_at DESC
LIMIT 10

// Find notes with similar content
MATCH (start:Note {path: $start_path})-[:SIMILAR_TO]->(similar:Note)
WHERE similar.similarity_score > 0.7
RETURN similar
ORDER BY similar.similarity_score DESC
LIMIT 10
```

### Graph Traversal
```cypher
// Find all notes reachable within 3 hops
MATCH path = (start:Note {path: $start_path})-[:LINKS_TO*1..3]->(reachable:Note)
RETURN path, reachable
ORDER BY length(path), reachable.title

// Find shortest path between two notes
MATCH path = shortestPath((start:Note {path: $start_path})-[:LINKS_TO*]-(end:Note {path: $end_path}))
RETURN path
```

### Content Discovery
```cypher
// Find notes by tag with relationship context
MATCH (note:Note)-[:TAGGED_WITH]->(tag:Tag {name: $tag_name})
OPTIONAL MATCH (note)-[:LINKS_TO]->(linked:Note)
RETURN note, collect(linked) as linked_notes
ORDER BY note.modified_at DESC

// Find hub notes (high connectivity)
MATCH (hub:Note)
WITH hub, 
     size((hub)-[:LINKS_TO]->()) as outgoing_links,
     size((hub)<-[:LINKS_TO]-()) as incoming_links
WHERE outgoing_links + incoming_links > 5
RETURN hub, outgoing_links, incoming_links
ORDER BY (outgoing_links + incoming_links) DESC
```

### Temporal Analysis
```cypher
// Find recently modified notes and their connections
MATCH (note:Note)
WHERE note.modified_at > datetime() - duration('P7D')
OPTIONAL MATCH (note)-[:LINKS_TO]->(linked:Note)
RETURN note, collect(linked) as connections
ORDER BY note.modified_at DESC

// Find notes created in sequence
MATCH (note1:Note)-[:FOLLOWS]->(note2:Note)
WHERE note1.created_at > datetime() - duration('P30D')
RETURN note1, note2
ORDER BY note1.created_at DESC
```

## Schema Evolution

### Version Management
```cypher
// Schema version node
CREATE (v:SchemaVersion {
  version: "1.0.0",
  created_at: datetime(),
  description: "Initial schema with basic note relationships"
})

// Migration tracking
CREATE (m:Migration {
  id: "001_add_similarity_relationships",
  applied_at: datetime(),
  description: "Added SIMILAR_TO relationships"
})
```

### Backward Compatibility
```cypher
// Check for old schema patterns
MATCH (n:Note) WHERE n.version IS NULL
SET n.version = "1.0.0"

// Migrate deprecated relationships
MATCH (n1:Note)-[r:OLD_RELATIONSHIP]->(n2:Note)
CREATE (n1)-[:NEW_RELATIONSHIP {
  created_at: r.created_at,
  migrated_from: "OLD_RELATIONSHIP"
}]->(n2)
DELETE r
```

## Performance Optimization

### Query Optimization
```cypher
// Use parameters for better query planning
MATCH (n:Note {path: $path})
WHERE n.modified_at > $since
RETURN n

// Limit result sets early
MATCH (n:Note)-[:LINKS_TO]->(linked:Note)
WITH n, linked
ORDER BY linked.modified_at DESC
LIMIT 100
RETURN n, linked
```

### Memory Management
```cypher
// Use PERIODIC COMMIT for large operations
USING PERIODIC COMMIT 1000
LOAD CSV FROM 'file:///notes.csv' AS row
CREATE (n:Note {
  path: row.path,
  title: row.title,
  content: row.content
})
```

### Monitoring Queries
```cypher
// Find slow queries
CALL db.stats.query.list() 
YIELD query, elapsedTimeMillis
WHERE elapsedTimeMillis > 1000
RETURN query, elapsedTimeMillis
ORDER BY elapsedTimeMillis DESC

// Monitor relationship counts
MATCH ()-[r]->()
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC
```

## Data Consistency

### Constraint Definitions
```cypher
// Unique constraints
CREATE CONSTRAINT note_path_unique FOR (n:Note) REQUIRE n.path IS UNIQUE
CREATE CONSTRAINT tag_name_unique FOR (t:Tag) REQUIRE t.name IS UNIQUE
CREATE CONSTRAINT heading_id_unique FOR (h:Heading) REQUIRE h.id IS UNIQUE

// Existence constraints
CREATE CONSTRAINT note_path_exists FOR (n:Note) REQUIRE n.path IS NOT NULL
CREATE CONSTRAINT note_title_exists FOR (n:Note) REQUIRE n.title IS NOT NULL
```

### Validation Rules
```cypher
// Check for orphaned headings
MATCH (h:Heading)
WHERE NOT EXISTS((h)<-[:CONTAINS]-(:Note))
RETURN h

// Check for broken links
MATCH (n:Note)-[:LINKS_TO]->(target:Note)
WHERE NOT EXISTS(target.path)
RETURN n.path as source, target.path as broken_target
```

## For More Detail

- **Component Interaction**: [Component Interaction](component-interaction.md)
- **Data Flow**: [Data Flow Architecture](data-flow.md)
- **Search Implementation**: [Semantic Search Design](semantic-search-design.md)
- **Database Setup**: [Neo4j Setup Guide](../05-development/neo4j-setup.md)