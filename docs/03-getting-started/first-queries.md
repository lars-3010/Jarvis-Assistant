# First Queries

*Testing your setup and validating functionality*

## Overview

After installation and configuration, it's important to verify that all components are working correctly. This guide provides systematic tests for each part of the system.

## System Health Check

### Basic System Validation

#### Step 1: Check Installation
```bash
# Verify Jarvis is installed
uv run jarvis --version

# Check system information
uv run jarvis --system-info

# Expected output:
# Jarvis Assistant v0.3.0
# Python: 3.11.x
# Platform: darwin/linux/win32
# Dependencies: OK
```

#### Step 2: Validate Configuration
```bash
# Show current configuration
uv run jarvis --show-config

# Check vault accessibility
uv run jarvis --check-vault /path/to/your/vault

# Expected output:
# Vault path: /path/to/your/vault
# Vault accessible: ✓
# Markdown files found: 150
# Vault size: 2.3 MB
```

#### Step 3: Test Dependencies
```bash
# Test all dependencies
uv run jarvis --test-dependencies

# Expected output:
# torch: ✓
# sentence_transformers: ✓
# duckdb: ✓
# neo4j: ✓ (if configured)
# mcp: ✓
```

## Vector Search Testing

### Basic Semantic Search

#### Step 1: Simple Search Query
```bash
# Test basic semantic search
uv run jarvis search --vault /path/to/your/vault --query "productivity tips"

# Expected output:
# Semantic search results for: "productivity tips"
# 
# 1. /Daily Notes/2024-01-15.md (Score: 0.89)
#    Today I learned about the Pomodoro Technique for better productivity...
# 
# 2. /Projects/GTD System.md (Score: 0.82)
#    Getting Things Done methodology for organizing tasks...
# 
# Found 5 results in 0.3s
```

#### Step 2: Adjust Similarity Threshold
```bash
# Test with different similarity thresholds
uv run jarvis search \
  --vault /path/to/your/vault \
  --query "machine learning" \
  --similarity-threshold 0.8 \
  --limit 5

# Higher threshold = more precise results
# Lower threshold = more comprehensive results
```

#### Step 3: Test Different Query Types
```bash
# Conceptual query
uv run jarvis search --vault /path/to/your/vault --query "creative writing techniques"

# Technical query
uv run jarvis search --vault /path/to/your/vault --query "python programming best practices"

# Personal query
uv run jarvis search --vault /path/to/your/vault --query "morning routine habits"
```

### Advanced Search Testing

#### Step 1: Test Search Performance
```bash
# Benchmark search performance
uv run jarvis --benchmark-search --vault /path/to/your/vault

# Expected output:
# Search Performance Benchmark
# ===========================
# Vault: /path/to/your/vault
# Total files: 150
# Total embeddings: 342
# 
# Query: "test query"
# Search time: 0.045s
# Results: 10
# 
# Average search time: 0.052s
# Embeddings per second: 6,580
```

#### Step 2: Test Cache Performance
```bash
# Test same query multiple times (should be faster after first)
time uv run jarvis search --vault /path/to/your/vault --query "productivity"
time uv run jarvis search --vault /path/to/your/vault --query "productivity"
time uv run jarvis search --vault /path/to/your/vault --query "productivity"
```

## Vault Operations Testing

### File Reading Tests

#### Step 1: Test File Reading
```bash
# List files in vault
uv run jarvis list-files --vault /path/to/your/vault --limit 10

# Read specific file
uv run jarvis read-file --vault /path/to/your/vault --path "Daily Notes/2024-01-15.md"

# Expected output:
# File: Daily Notes/2024-01-15.md
# Size: 1,234 bytes
# Modified: 2024-01-15 10:30:45
# 
# Content:
# # Daily Note - January 15, 2024
# 
# Today I worked on...
```

#### Step 2: Test Vault Statistics
```bash
# Get vault statistics
uv run jarvis stats --vault /path/to/your/vault

# Expected output:
# Vault Statistics
# ================
# Path: /path/to/your/vault
# Total files: 150
# Total size: 2.3 MB
# Indexed files: 150
# Last indexed: 2024-01-15 14:30:00
# 
# Search Index:
# - Embeddings: 342
# - Vector dimension: 384
# - Database size: 15.2 MB
```

### Traditional Search Tests

#### Step 1: Keyword Search
```bash
# Test keyword search
uv run jarvis vault-search --vault /path/to/your/vault --query "docker"

# Expected output:
# Keyword search results for: "docker"
# 
# 1. /Development/Docker Guide.md
#    Docker containers provide isolated environments...
# 
# 2. /Notes/DevOps Setup.md
#    Setting up Docker for development workflow...
# 
# Found 3 results in 0.1s
```

#### Step 2: Content vs Filename Search
```bash
# Search in content only
uv run jarvis vault-search --vault /path/to/your/vault --query "python" --search-content

# Search in filenames only
uv run jarvis vault-search --vault /path/to/your/vault --query "python" --search-filenames
```

## Graph Search Testing (Optional)

### Basic Graph Operations

#### Step 1: Test Graph Connection
```bash
# Test Neo4j connection
uv run jarvis --test-neo4j

# Expected output:
# Neo4j Connection Test
# =====================
# URI: bolt://localhost:7687
# Username: neo4j
# Connection: ✓
# Database: ✓
# Version: 5.x.x
```

#### Step 2: Test Graph Indexing
```bash
# Index vault for graph search
uv run jarvis graph-index --vault /path/to/your/vault

# Expected output:
# Graph indexing started...
# Processing 150 files...
# Created 150 note nodes
# Created 324 link relationships
# Created 89 reference relationships
# Graph index complete in 12.3s
```

#### Step 3: Test Graph Search
```bash
# Search for related notes
uv run jarvis graph-search \
  --vault /path/to/your/vault \
  --query-note-path "Projects/Machine Learning.md" \
  --depth 2

# Expected output:
# Graph search from: Projects/Machine Learning.md (depth 2)
# 
# Connected Notes:
# • Neural Networks Basics.md (Projects/AI/)
# • Deep Learning Resources.md (Resources/)
# • Python ML Libraries.md (Development/)
# 
# Relationships:
# • Machine Learning.md → Neural Networks Basics.md (links_to)
# • Machine Learning.md → Deep Learning Resources.md (references)
# 
# Found 8 connected notes
```

## MCP Server Testing

### Server Startup Tests

#### Step 1: Start MCP Server
```bash
# Start MCP server with verbose output
uv run jarvis mcp --vault /path/to/your/vault --verbose

# Expected output:
# MCP Server starting...
# Vault path: /path/to/your/vault
# Vector index: ✓ (342 embeddings)
# Graph index: ✓ (150 nodes, 324 relationships)
# 
# Available tools:
# - search-semantic: Semantic search using vector similarity
# - search-vault: Traditional keyword search
# - search-graph: Graph relationship discovery
# - read-note: Read specific note content
# - list-vaults: Vault management and statistics
# 
# MCP server listening on stdio
# Press Ctrl+C to stop
```

#### Step 2: Test MCP Protocol
```bash
# Test MCP protocol compliance
uv run jarvis --test-mcp --vault /path/to/your/vault

# Expected output:
# MCP Protocol Test
# =================
# Server initialization: ✓
# Tool discovery: ✓
# Tool execution: ✓
# Error handling: ✓
# Resource management: ✓
# 
# Available tools: 5
# Protocol version: 1.0.0
```

### Claude Desktop Integration Tests

#### Step 1: Configuration Check
```bash
# Check Claude Desktop configuration
cat ~/.claude.json

# Verify paths are correct
ls -la /path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio
```

#### Step 2: Test MCP Tools in Claude Desktop

Start Claude Desktop and test each tool:

##### Semantic Search Tool
```
Query: "Search my vault for notes about productivity techniques"
Expected: Claude uses search-semantic tool and returns relevant notes
```

##### Vault Search Tool
```
Query: "Find all notes that mention 'docker' in my vault"
Expected: Claude uses search-vault tool for keyword search
```

##### Graph Search Tool
```
Query: "What notes are related to my machine learning project?"
Expected: Claude uses search-graph tool to find connected notes
```

##### Read Note Tool
```
Query: "Read my daily note from January 15th"
Expected: Claude uses read-note tool to access specific file
```

##### List Vaults Tool
```
Query: "Show me statistics about my vault"
Expected: Claude uses list-vaults tool to display vault information
```

## Performance Testing

### Embedding Performance

#### Step 1: Test Embedding Generation
```bash
# Test embedding generation performance
uv run jarvis --benchmark-embeddings

# Expected output:
# Embedding Performance Benchmark
# ===============================
# Model: paraphrase-MiniLM-L6-v2
# Device: cpu/cuda/mps
# 
# Single text embedding: 0.045s
# Batch embedding (32): 0.891s
# Embeddings per second: 35.9
# 
# Memory usage: 245 MB
# Cache hit rate: 85%
```

#### Step 2: Test Different Batch Sizes
```bash
# Test different batch sizes
uv run jarvis --benchmark-embeddings --batch-size 16
uv run jarvis --benchmark-embeddings --batch-size 32
uv run jarvis --benchmark-embeddings --batch-size 64
```

### Search Performance

#### Step 1: Search Latency Test
```bash
# Test search latency
uv run jarvis --benchmark-search --vault /path/to/your/vault --queries 100

# Expected output:
# Search Performance Benchmark
# ============================
# Vault: /path/to/your/vault
# Total queries: 100
# 
# Average search time: 0.052s
# Median search time: 0.045s
# 95th percentile: 0.089s
# 99th percentile: 0.142s
# 
# Throughput: 19.2 queries/second
```

#### Step 2: Concurrent Search Test
```bash
# Test concurrent search performance
uv run jarvis --benchmark-concurrent --vault /path/to/your/vault --workers 4

# Expected output:
# Concurrent Search Benchmark
# ===========================
# Workers: 4
# Queries per worker: 25
# Total queries: 100
# 
# Total time: 3.2s
# Average time per query: 0.032s
# Throughput: 31.2 queries/second
```

## Troubleshooting First Queries

### Common Issues

#### No Search Results
```bash
# Check if vault is indexed
uv run jarvis stats --vault /path/to/your/vault

# Re-index if needed
uv run jarvis index --vault /path/to/your/vault --force

# Check file permissions
ls -la /path/to/your/vault/
```

#### Slow Search Performance
```bash
# Check embedding model
uv run jarvis --show-config | grep EMBEDDING_MODEL_NAME

# Try smaller batch size
export EMBEDDING_BATCH_SIZE=16
uv run jarvis search --vault /path/to/your/vault --query "test"

# Check available memory
free -h  # Linux
vm_stat  # macOS
```

#### MCP Server Not Starting
```bash
# Check dependencies
uv run jarvis --test-dependencies

# Check vault path
uv run jarvis --check-vault /path/to/your/vault

# Check permissions
ls -la .venv/bin/jarvis-mcp-stdio
chmod +x .venv/bin/jarvis-mcp-stdio  # If needed
```

#### Claude Desktop Not Connecting
```bash
# Check configuration syntax
python -m json.tool ~/.claude.json

# Check binary path
ls -la /path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio

# Test MCP server manually
uv run jarvis mcp --vault /path/to/your/vault --verbose
```

### Diagnostic Commands

#### System Diagnostics
```bash
# Complete system check
uv run jarvis --diagnose

# Check specific component
uv run jarvis --diagnose --component search
uv run jarvis --diagnose --component mcp
uv run jarvis --diagnose --component graph
```

#### Performance Diagnostics
```bash
# Profile search performance
uv run jarvis --profile-search --vault /path/to/your/vault --query "test"

# Memory usage analysis
uv run jarvis --memory-profile --vault /path/to/your/vault

# Cache analysis
uv run jarvis --cache-stats --vault /path/to/your/vault
```

## Validation Checklist

### Basic Functionality
- [ ] System installation verified
- [ ] Configuration validated
- [ ] Vault accessibility confirmed
- [ ] Dependencies tested

### Search Functionality
- [ ] Semantic search working
- [ ] Keyword search working
- [ ] Search performance acceptable
- [ ] Cache functioning properly

### MCP Integration
- [ ] MCP server starts successfully
- [ ] All 5 tools available
- [ ] Claude Desktop configured
- [ ] Tools working in Claude Desktop

### Performance
- [ ] Search latency < 100ms
- [ ] Embedding generation working
- [ ] Memory usage reasonable
- [ ] No error messages

## Next Steps

### If Everything Works
1. **[Common Workflows](../04-usage/common-workflows.md)** - Learn typical usage patterns
2. **[API Examples](../04-usage/api-examples.md)** - Explore all available tools
3. **[Advanced Queries](../04-usage/advanced-queries.md)** - Master complex searches

### If Issues Persist
1. **[Troubleshooting Guide](../07-maintenance/troubleshooting.md)** - Detailed problem solving
2. **[Performance Tuning](../07-maintenance/performance-tuning.md)** - Optimization tips
3. **[GitHub Issues](https://github.com/your-username/jarvis-assistant/issues)** - Report problems

### For Customization
1. **[Configuration Reference](../06-reference/configuration-reference.md)** - All settings
2. **[Developer Guide](../05-development/developer-guide.md)** - Extending the system

---

**Congratulations!** If all tests pass, your Jarvis Assistant is ready to provide intelligent search capabilities to AI systems. The system is now configured and validated for production use.
