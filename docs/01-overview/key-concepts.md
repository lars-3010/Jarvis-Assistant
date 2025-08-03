# Key Concepts

*Core terminology and concepts for understanding Jarvis Assistant*

## Essential Terminology

### MCP (Model Context Protocol)
- **Definition**: Standard protocol for AI tools to communicate with external systems
- **Purpose**: Enables Claude Desktop and other AI systems to access external data and tools
- **In Jarvis**: Provides the interface layer between AI systems and our search capabilities

### Semantic Search
- **Definition**: Search based on meaning and context rather than exact keyword matching
- **How it Works**: Converts text to vector embeddings, finds similar vectors in high-dimensional space
- **Benefits**: Finds related content even with different wording, understands context and intent

### Graph Relationships
- **Definition**: Connections between notes based on links, references, and shared concepts
- **Types**: Direct links, backlinks, concept relationships, hierarchical connections
- **Use Cases**: Finding related notes, understanding note clusters, discovering knowledge gaps

### Vector Embeddings
- **Definition**: Numerical representations of text that capture semantic meaning
- **Generation**: Created using sentence-transformers neural networks
- **Storage**: Stored in DuckDB for efficient similarity search

### Obsidian Vault
- **Definition**: Collection of markdown files organized as a knowledge base
- **Structure**: Hierarchical folders with interconnected notes
- **Features**: Wiki-style links, tags, metadata, plugins ecosystem

## Core Concepts

### Local-First Architecture
- **Principle**: All processing happens on your local machine
- **Benefits**: Privacy, control, no internet dependency, fast access
- **Trade-offs**: Requires local resources, setup complexity

### Hybrid Search Strategy
- **Semantic Search**: Find conceptually related content
- **Keyword Search**: Find specific terms and phrases
- **Graph Search**: Discover relationships and connections
- **Combined**: Best results from multiple search approaches

### Caching Strategy
- **Embedding Cache**: Stores computed embeddings to avoid recomputation
- **Search Cache**: Caches recent search results for faster retrieval
- **Graph Cache**: Caches relationship queries for improved performance

### Indexing Process
- **Content Extraction**: Parse markdown files for text content
- **Embedding Generation**: Convert text to vector representations
- **Graph Construction**: Build relationship maps from links and references
- **Database Storage**: Store vectors and relationships in optimized formats

## Mental Models

### Think of Jarvis Assistant as...

#### A Librarian
- **Traditional Search**: Like asking for books with specific words in the title
- **Semantic Search**: Like asking for books about similar topics, even if they use different terms
- **Graph Search**: Like asking what other books reference this one, or what topics are related

#### A Knowledge Map
- **Nodes**: Individual notes in your vault
- **Edges**: Relationships between notes (links, references, concepts)
- **Paths**: Ways to navigate from one idea to another
- **Clusters**: Groups of related notes about similar topics

#### A Translation Layer
- **Input**: Natural language questions from AI systems
- **Processing**: Converts questions to database queries
- **Output**: Structured results that AI systems can understand and use

## Search Strategies

### When to Use Each Search Type

#### Semantic Search
- **Best For**: Exploring topics, finding related ideas, research discovery
- **Example**: "Find notes about productivity techniques" → Returns notes about GTD, time management, focus methods

#### Keyword Search
- **Best For**: Finding specific information, exact quotes, technical terms
- **Example**: "DuckDB configuration" → Returns notes containing that exact phrase

#### Graph Search
- **Best For**: Understanding relationships, finding note clusters, exploring connections
- **Example**: Starting from a note about "Machine Learning" → Find all connected notes about AI, algorithms, data science

### Combining Search Methods
- **Start Broad**: Use semantic search for topic exploration
- **Get Specific**: Use keyword search for precise information
- **Understand Context**: Use graph search to see how ideas connect

## Common Patterns

### Knowledge Discovery Workflow
1. **Explore**: Use semantic search to find topic-related notes
2. **Navigate**: Use graph search to find related concepts
3. **Locate**: Use keyword search to find specific details
4. **Synthesize**: Combine results to understand the full picture

### Content Organization
- **Hierarchical**: Organize notes in folder structures
- **Associative**: Connect notes through links and references
- **Thematic**: Group related notes with tags and categories
- **Temporal**: Track information evolution over time

## For More Detail

- **Technical Implementation**: [System Design](../02-system-design/data-flow.md)
- **Practical Usage**: [Common Workflows](../04-usage/common-workflows.md)
- **API Reference**: [MCP Tools Reference](../06-reference/api-reference.md)