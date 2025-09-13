# Detailed Installation Guide

*Complete setup instructions with all options and configurations*

## Prerequisites

### System Requirements

- **Operating System**: macOS 10.15+, Linux (Ubuntu 20.04+), or Windows 10+
- **Python**: 3.11 or higher
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: 2GB+ free space for dependencies and indexes
- **Network**: Internet connection for initial setup

### Required Software

#### 1. Python 3.11+
```bash
# Check current Python version
python3 --version

# macOS with Homebrew
brew install python@3.11

# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-dev python3.11-venv

# Windows
# Download from python.org or use winget
winget install Python.Python.3.11
```

#### 2. UV Package Manager
```bash
# Install UV (cross-platform)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Verify installation
uv --version
```

#### 3. Git (for cloning repository)
```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt install git

# Windows
winget install Git.Git
```

#### 4. Claude Desktop (for MCP integration)
Download from: https://claude.ai/download

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository
```bash
# Clone the repository
git clone https://github.com/lars-3010/Jarvis-Assistant.git
cd jarvis-assistant

# Verify project structure
ls -la
```

#### Step 2: Install Dependencies
```bash
# Install all dependencies (creates virtual environment)
uv sync

# Verify installation
uv run jarvis --help
```

#### Step 3: Verify Installation
```bash
# Check all components
uv run jarvis --version
uv run python -c "import torch; print('PyTorch:', torch.__version__)"
uv run python -c "import sentence_transformers; print('SentenceTransformers: OK')"
uv run python -c "import duckdb; print('DuckDB: OK')"
```

### Method 2: Development Installation

#### Step 1: Development Setup
```bash
# Clone with development branches
git clone https://github.com/lars-3010/Jarvis-Assistant.git
cd jarvis-assistant

# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

#### Step 2: Verify Development Environment
```bash
# Run linting
uv run ruff check src/

# Run type checking
uv run mypy src/

# Run tests
uv run pytest resources/tests/
```

### Method 3: Docker Installation (Optional)

#### Step 1: Docker Setup
```bash
# Build Docker image
docker build -t jarvis-assistant .

# Run with vault mounted
docker run -it -v /path/to/vault:/vault jarvis-assistant
```

## Database Setup

### DuckDB (Automatic)
DuckDB is embedded and requires no separate installation. It will be automatically created on first run.

### Neo4j (Optional)

#### Option 1: Docker Neo4j
```bash
# Run Neo4j in Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v neo4j_data:/data \
  neo4j:latest

# Verify Neo4j is running
curl http://localhost:7474/
```

#### Option 2: Native Neo4j Installation
```bash
# macOS
brew install neo4j

# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j

# Start Neo4j
sudo systemctl start neo4j
```

#### Step 3: Configure Neo4j
```bash
# Set environment variables
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=password

# Or create config/.env file
cp config/.env.example config/.env
# Edit config/.env with your Neo4j credentials
```

## Vault Setup

### Obsidian Vault Preparation

#### Step 1: Locate Your Vault
```bash
# Common vault locations:

# macOS iCloud
/Users/username/Library/Mobile Documents/iCloud~md~obsidian/Documents/VaultName

# macOS Local
/Users/username/Documents/ObsidianVault

# Windows
C:\Users\username\Documents\ObsidianVault

# Linux
/home/username/Documents/ObsidianVault
```

#### Step 2: Verify Vault Structure
```bash
# Check vault contents
ls -la /path/to/your/vault/

# Should contain .md files
find /path/to/your/vault -name "*.md" | head -5
```

#### Step 3: Create Vault Backup (Recommended)
```bash
# Create backup before indexing
cp -r /path/to/your/vault /path/to/your/vault.backup

# Or use git if vault is version controlled
cd /path/to/your/vault
git add .
git commit -m "Pre-indexing backup"
```

## Initial Indexing

### Semantic Search Index

#### Step 1: Basic Indexing
```bash
# Index your vault
uv run jarvis index --vault /path/to/your/vault

# Example output:
# Indexing vault: /path/to/your/vault
# Found 150 markdown files
# Processing files... ████████████████████████████████ 100%
# Generated embeddings for 150 files
# Vector index created successfully
# Processed 150 files in 45.2s
```

#### Step 2: Verify Index
```bash
# Check index status
uv run jarvis stats --vault /path/to/your/vault

# Test search
uv run jarvis search --vault /path/to/your/vault --query "test query"
```

### Graph Search Index (Optional)

#### Step 1: Neo4j Index
```bash
# Index vault for graph search (requires Neo4j)
uv run jarvis graph-index --vault /path/to/your/vault

# Example output:
# Connecting to Neo4j at bolt://localhost:7687
# Processing 150 markdown files
# Created 150 note nodes
# Created 324 link relationships
# Graph index created successfully
```

#### Step 2: Verify Graph Index
```bash
# Test graph search
uv run jarvis graph-search --vault /path/to/your/vault --query-note-path "example.md"
```

## Claude Desktop Configuration

### Configuration File Setup

#### Step 1: Locate Configuration File
```bash
# Configuration file locations:

# macOS/Linux
~/.claude.json

# Windows
%APPDATA%\Claude\claude.json
```

#### Step 2: Create Configuration
```json
{
  "mcpServers": {
    "jarvis": {
      "command": "/path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio",
      "args": ["/path/to/your/obsidian/vault"],
      "type": "stdio",
      "cwd": "/path/to/jarvis-assistant",
      "env": {
        "PYTHONUNBUFFERED": "1",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

#### Step 3: Platform-Specific Paths
```bash
# macOS/Linux binary path
/path/to/jarvis-assistant/.venv/bin/jarvis-mcp-stdio

# Windows binary path
C:\path\to\jarvis-assistant\.venv\Scripts\jarvis-mcp-stdio.exe
```

### Verify MCP Integration

#### Step 1: Test MCP Server
```bash
# Start MCP server manually
uv run jarvis mcp --vault /path/to/your/vault --verbose

# Should show:
# MCP server started for vault: /path/to/your/vault
# Listening on stdio for Claude Desktop
# Available tools: search-semantic, search-vault, search-graph, read-note, list-vaults
```

#### Step 2: Test Claude Desktop Connection
1. **Restart Claude Desktop** after configuration
2. **Open new conversation**
3. **Test integration**:
   ```
   Test message: "Search my vault for notes about productivity"
   Expected: Claude should use the search-semantic tool
   ```

## Advanced Configuration

### Environment Variables

#### Step 1: Create Environment File
```bash
# Copy example configuration
cp config/.env.example config/.env

# Edit configuration
nano config/.env
```

#### Step 2: Configuration Options
```bash
# Core settings
JARVIS_VAULT_PATH=/path/to/default/vault
JARVIS_DB_PATH=/path/to/database/directory

# Embedding settings
EMBEDDING_MODEL_NAME=paraphrase-MiniLM-L6-v2
EMBEDDING_DEVICE=auto  # auto, cpu, cuda, mps
EMBEDDING_BATCH_SIZE=32

# Neo4j settings (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Performance settings
MAX_WORKERS=4
CACHE_SIZE=1000
INDEX_BATCH_SIZE=100

# Logging
LOG_LEVEL=INFO
LOG_FILE=/path/to/logs/jarvis.log
```

### Performance Optimization

#### Step 1: Memory Optimization
```bash
# For systems with limited memory
export EMBEDDING_BATCH_SIZE=16
export MAX_WORKERS=2

# For high-memory systems
export EMBEDDING_BATCH_SIZE=64
export MAX_WORKERS=8
```

#### Step 2: GPU Acceleration (Optional)
```bash
# Check for GPU support
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Enable GPU if available
export EMBEDDING_DEVICE=cuda  # or mps for Apple Silicon
```

## Troubleshooting Installation

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python3 --version

# Install specific version if needed
brew install python@3.11  # macOS
sudo apt install python3.11  # Ubuntu

# Use specific Python version
uv python install 3.11
uv sync
```

#### Permission Issues
```bash
# Fix binary permissions (Unix/macOS)
chmod +x .venv/bin/jarvis-mcp-stdio

# Fix directory permissions
chmod -R 755 .venv/
```

#### Missing Dependencies
```bash
# Reinstall dependencies
uv sync --reinstall

# Install system dependencies (Ubuntu)
sudo apt install build-essential python3-dev

# Install system dependencies (macOS)
xcode-select --install
```

#### Database Connection Issues
```bash
# Check Neo4j connection
uv run python -c "from neo4j import GraphDatabase; print('Neo4j connection OK')"

# Check DuckDB
uv run python -c "import duckdb; print('DuckDB OK')"
```

### Diagnostic Commands

#### System Information
```bash
# Check system info
uv run jarvis --system-info

# Check dependency versions
uv run jarvis --version --verbose

# Check vault accessibility
uv run jarvis --check-vault /path/to/vault
```

#### Performance Testing
```bash
# Test embedding generation
uv run jarvis --test-embeddings

# Test search performance
uv run jarvis --benchmark-search --vault /path/to/vault

# Test MCP protocol
uv run jarvis --test-mcp
```

## Post-Installation

### Verification Checklist

- [ ] UV package manager installed and working
- [ ] Jarvis Assistant dependencies installed
- [ ] Vault path accessible and contains .md files
- [ ] Vector index created successfully
- [ ] MCP server starts without errors
- [ ] Claude Desktop configuration updated
- [ ] Basic search functionality working
- [ ] Neo4j connected (if using graph search)

### Next Steps

1. **[Configuration Guide](configuration.md)** - Customize settings
2. **[First Queries](first-queries.md)** - Test your setup
3. **[Common Workflows](../04-usage/common-workflows.md)** - Learn usage patterns
4. **[API Examples](../04-usage/api-examples.md)** - Explore all tools

### Getting Help

- **[Troubleshooting Guide](../07-maintenance/troubleshooting.md)** - Common issues
- **[Performance Tuning](../07-maintenance/performance-tuning.md)** - Optimization
- **[GitHub Issues](https://github.com/your-username/jarvis-assistant/issues)** - Report problems

---

**Installation complete!** Your Jarvis Assistant is ready to provide intelligent search capabilities to AI systems through the MCP protocol.
