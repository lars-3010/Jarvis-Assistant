// Neo4j Schema for Obsidian Notes with Semantic Relationships and Vector Search

// Basic schema constraints and indices
CREATE CONSTRAINT note_path_unique IF NOT EXISTS FOR (n:Note) REQUIRE n.path IS UNIQUE;
CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR (n:Tag) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

// Create indices for frequently queried properties
CREATE INDEX note_title_index IF NOT EXISTS FOR (n:Note) ON (n.title);
CREATE INDEX note_updated_index IF NOT EXISTS FOR (n:Note) ON (n.updated);
CREATE INDEX tag_name_index IF NOT EXISTS FOR (t:Tag) ON (t.name);
CREATE INDEX note_tags_index IF NOT EXISTS FOR (n:Note) ON (n.tags);
CREATE INDEX chunk_note_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.note_id);

// Create vector index for note embeddings (Neo4j 5.11+)
// If using Neo4j < 5.11, this will be skipped during initialization
CREATE VECTOR INDEX note_embedding_index IF NOT EXISTS 
FOR (n:Note) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,  // For 'all-MiniLM-L6-v2' model
    `vector.similarity_function`: 'cosine'
  }
};

// Create vector index for chunk embeddings (Neo4j 5.11+)
CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS 
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,  // For 'all-MiniLM-L6-v2' model
    `vector.similarity_function`: 'cosine'
  }
};

// Sample schema for creating notes and relationships

// Create a note with embedding
MERGE (n:Note {path: "path/to/note.md"})
ON CREATE SET 
  n.title = "Note Title",
  n.content = "Full note content...",
  n.created = timestamp(),
  n.tags = ["tag1", "tag2"],
  n.embedding = [0.1, 0.2, ...], // Vector embedding (normally 384 dimensions)
  n.has_embedding = true
ON MATCH SET
  n.title = "Note Title",
  n.content = "Full note content...",
  n.updated = timestamp(),
  n.tags = ["tag1", "tag2"],
  n.embedding = [0.1, 0.2, ...],
  n.has_embedding = true;

// Create a chunk with embedding
MERGE (c:Chunk {id: "unique-chunk-id"})
ON CREATE SET
  c.text = "Chunk text content...",
  c.embedding = [0.1, 0.2, ...], // Vector embedding
  c.start_pos = 0,
  c.end_pos = 500,
  c.note_id = 123,  // ID of parent note
  c.created = timestamp()
ON MATCH SET
  c.text = "Chunk text content...",
  c.embedding = [0.1, 0.2, ...],
  c.updated = timestamp();

// Connect note to chunk
MATCH (n:Note), (c:Chunk {note_id: id(n)})
MERGE (n)-[:HAS_CHUNK]->(c);

// Create tags and connect them to the note
UNWIND ["tag1", "tag2"] AS tagName
MERGE (t:Tag {name: tagName})
WITH t, n
MERGE (n)-[:HAS_TAG]->(t);

// Create semantic relationships
// UP relationship (parent concept)
MERGE (parent:Note {path: "path/to/parent.md"})
ON CREATE SET parent.title = "Parent"
MERGE (n)-[:UP {type: "up::"}]->(parent);

// SIMILAR relationship
MERGE (similar:Note {path: "path/to/similar.md"})
ON CREATE SET similar.title = "Similar"
MERGE (n)-[:SIMILAR {type: "similar"}]->(similar);

// LEADS_TO relationship
MERGE (next:Note {path: "path/to/next.md"})
ON CREATE SET next.title = "Next"
MERGE (n)-[:LEADS_TO {type: "leads_to"}]->(next);

// CONTRADICTS relationship
MERGE (opposing:Note {path: "path/to/opposing.md"})
ON CREATE SET opposing.title = "Opposing"
MERGE (n)-[:CONTRADICTS {type: "contradicts"}]->(opposing);

// EXTENDS relationship
MERGE (foundation:Note {path: "path/to/foundation.md"})
ON CREATE SET foundation.title = "Foundation"
MERGE (n)-[:EXTENDS {type: "extends"}]->(foundation);

// IMPLEMENTS relationship
MERGE (application:Note {path: "path/to/application.md"})
ON CREATE SET application.title = "Application"
MERGE (n)-[:IMPLEMENTS {type: "implements"}]->(application);

// Plain links (non-semantic relationships from markdown links)
MERGE (linked:Note {path: "path/to/linked.md"})
ON CREATE SET linked.title = "Linked"
MERGE (n)-[:LINKS_TO]->(linked);

// --- Vector Search Queries ---

// Basic vector search for similar notes
MATCH (n:Note)
WHERE n.embedding IS NOT NULL
WITH n, vector.similarity(n.embedding, $query_embedding) AS similarity
WHERE similarity >= 0.7
RETURN n.path as path, n.title as title, similarity
ORDER BY similarity DESC
LIMIT 5;

// Vector search for similar chunks
MATCH (c:Chunk)
WITH c, vector.similarity(c.embedding, $query_embedding) AS similarity
WHERE similarity >= 0.7
MATCH (n:Note)-[:HAS_CHUNK]->(c)
RETURN c.id as id, c.text as text, similarity,
       n.path as note_path, n.title as note_title
ORDER BY similarity DESC
LIMIT 10;

// Hybrid search example (combining vector similarity with keyword filtering)
MATCH (n:Note)
WHERE n.embedding IS NOT NULL AND any(tag IN n.tags WHERE tag IN $search_tags)
WITH n, vector.similarity(n.embedding, $query_embedding) AS similarity
WHERE similarity >= 0.6
RETURN n.path as path, n.title as title, similarity, n.tags as tags
ORDER BY similarity DESC
LIMIT 10;

// Graph-augmented vector search (find similar notes and their connected notes)
MATCH (n:Note)
WHERE n.embedding IS NOT NULL
WITH n, vector.similarity(n.embedding, $query_embedding) AS similarity
WHERE similarity >= 0.7
WITH n, similarity
ORDER BY similarity DESC
LIMIT 3
MATCH (n)-[r]-(connected:Note)
WHERE type(r) <> 'HAS_TAG'
RETURN n.path as source_path, n.title as source_title, similarity as source_similarity,
       connected.path as connected_path, connected.title as connected_title,
       type(r) as relationship_type
LIMIT 15;