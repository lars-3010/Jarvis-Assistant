# Troubleshooting

Comprehensive troubleshooting guide for Jarvis Assistant. This guide covers common issues, diagnostic procedures, and step-by-step resolution strategies.

## Quick Navigation

- [Quick Diagnostics](#quick-diagnostics)
- [MCP Server Issues](#mcp-server-issues)
- [Search Problems](#search-problems)
- [Database Issues](#database-issues)
- [Performance Problems](#performance-problems)
- [Configuration Issues](#configuration-issues)
- [System Integration](#system-integration)

---

## Quick Diagnostics

### System Health Check

Run this comprehensive health check to identify common issues:

```bash
#!/bin/bash
# Quick diagnostic script

echo "=== Jarvis Assistant Health Check ==="

# Check Python and UV
echo "1. Checking Python and UV..."
python --version
uv --version

# Check Jarvis installation
echo "2. Checking Jarvis installation..."
uv run jarvis --help > /dev/null 2>&1 && echo "✅ Jarvis CLI working" || echo "❌ Jarvis CLI failed"

# Check environment variables
echo "3. Checking environment variables..."
[ -n "$JARVIS_VAULT_PATH" ] && echo "✅ JARVIS_VAULT_PATH set: $JARVIS_VAULT_PATH" || echo "❌ JARVIS_VAULT_PATH not set"
[ -n "$NEO4J_PASSWORD" ] && echo "✅ NEO4J_PASSWORD set" || echo "❌ NEO4J_PASSWORD not set"

# Check vault access
echo "4. Checking vault access..."
if [ -d "$JARVIS_VAULT_PATH" ]; then
    file_count=$(find "$JARVIS_VAULT_PATH" -name "*.md" | wc -l)
    echo "✅ Vault accessible: $file_count markdown files found"
else
    echo "❌ Vault not accessible or doesn't exist"
fi

# Check database
echo "5. Checking database..."
if [ -f ~/.jarvis/jarvis.duckdb ]; then
    echo "✅ Database file exists"
else
    echo "❌ Database file not found"
fi

# Check Neo4j
echo "6. Checking Neo4j connection..."
if command -v cypher-shell >/dev/null 2>&1; then
    if cypher-shell -a bolt://localhost:7687 -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1" >/dev/null 2>&1; then
        echo "✅ Neo4j connection working"
    else
        echo "❌ Neo4j connection failed"
    fi
else
    echo "⚠️ cypher-shell not available"
fi

# Check model availability
echo "7. Checking ML model..."
uv run python -c "
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('✅ ML model loaded successfully')
except Exception as e:
    print(f'❌ ML model error: {e}')
" 2>/dev/null

echo "=== Health Check Complete ==="
```

### Common Status Commands

```bash
# Check all services
uv run jarvis list-vaults

# Test MCP server (run in background)
timeout 10s uv run jarvis mcp --vault "$JARVIS_VAULT_PATH" --test

# Check database size
ls -lh ~/.jarvis/jarvis.duckdb

# Check Neo4j status
neo4j status

# Check system resources
df -h ~/.jarvis/
free -h
```

---

## MCP Server Issues

### MCP Server Won't Start

#### Symptom
```
Error: MCP server failed to start
```

#### Diagnostic Steps
```bash
# 1. Check environment variables
echo "JARVIS_VAULT_PATH: $JARVIS_VAULT_PATH"
echo "NEO4J_PASSWORD: $NEO4J_PASSWORD"

# 2. Test vault access
ls -la "$JARVIS_VAULT_PATH"

# 3. Check database file
ls -la ~/.jarvis/jarvis.duckdb

# 4. Test in debug mode
JARVIS_LOG_LEVEL=DEBUG uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"
```

#### Common Causes and Solutions

**1. Missing Environment Variables**
```bash
# Set required variables
export JARVIS_VAULT_PATH="/path/to/your/vault"
export NEO4J_PASSWORD="your_password"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
```

**2. Vault Path Issues**
```bash
# Check if vault exists
if [ ! -d "$JARVIS_VAULT_PATH" ]; then
    echo "Creating vault directory..."
    mkdir -p "$JARVIS_VAULT_PATH"
    echo "# Test Note" > "$JARVIS_VAULT_PATH/test.md"
fi
```

**3. Database Issues**
```bash
# Remove corrupted database
rm -f ~/.jarvis/jarvis.duckdb

# Re-index vault
uv run jarvis index --vault "$JARVIS_VAULT_PATH"
```

**4. Permission Problems**
```bash
# Fix permissions
chmod 755 ~/.jarvis/
chmod 644 ~/.jarvis/jarvis.duckdb
chmod -R 755 "$JARVIS_VAULT_PATH"
```

### Claude Desktop Integration Issues

#### Symptom
MCP server starts but Claude Desktop doesn't show tools

#### Diagnostic Steps
```bash
# 1. Check Claude Desktop configuration
cat ~/.config/claude-desktop/claude_desktop_config.json

# 2. Test MCP server manually
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"

# 3. Check logs
tail -f ~/.claude/logs/mcp.log
```

#### Common Solutions

**1. Configuration Format**
```json
{
  "mcpServers": {
    "jarvis-assistant": {
      "command": "uv",
      "args": ["run", "jarvis", "mcp", "--vault", "/absolute/path/to/vault"],
      "env": {
        "NEO4J_PASSWORD": "your_password"
      }
    }
  }
}
```

**2. Path Issues**
```bash
# Use absolute paths only
export JARVIS_VAULT_PATH="$(realpath ~/Documents/ObsidianVault)"
```

**3. Restart Claude Desktop**
- Completely quit Claude Desktop
- Wait 5 seconds
- Restart Claude Desktop

### MCP Server Crashes

#### Symptom
```
MCP server connection lost
Process exited with code 1
```

#### Diagnostic Steps
```bash
# 1. Run with debug logging
JARVIS_LOG_LEVEL=DEBUG uv run jarvis mcp --vault "$JARVIS_VAULT_PATH" 2>&1 | tee mcp-debug.log

# 2. Check for memory issues
while true; do
    ps aux | grep jarvis | grep -v grep
    sleep 5
done

# 3. Monitor file descriptors
lsof -p $(pgrep -f "jarvis mcp")
```

#### Common Causes and Solutions

**1. Memory Issues**
```bash
# Reduce memory usage
export JARVIS_BATCH_SIZE=8
export JARVIS_CACHE_SIZE=32
```

**2. Database Corruption**
```bash
# Test database integrity
uv run python -c "
import duckdb
try:
    conn = duckdb.connect('~/.jarvis/jarvis.duckdb')
    result = conn.execute('SELECT COUNT(*) FROM documents').fetchone()
    print(f'Database OK: {result[0]} documents')
except Exception as e:
    print(f'Database error: {e}')
"
```

**3. Neo4j Connection Issues**
```bash
# Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1"

# If fails, restart Neo4j
neo4j restart
```

---

## Search Problems

### Semantic Search Returns No Results

#### Symptom
```
No results found for query: 'machine learning'
```

#### Diagnostic Steps
```bash
# 1. Check if vault is indexed
uv run python -c "
import duckdb
conn = duckdb.connect('~/.jarvis/jarvis.duckdb')
try:
    result = conn.execute('SELECT COUNT(*) FROM documents').fetchone()
    print(f'Indexed documents: {result[0]}')
    if result[0] == 0:
        print('Database is empty - run indexing')
except:
    print('Database not initialized - run indexing')
"

# 2. Check vault contents
find "$JARVIS_VAULT_PATH" -name "*.md" | head -10

# 3. Test model loading
uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Model loaded successfully')
embedding = model.encode('test query')
print(f'Embedding dimension: {len(embedding)}')
"
```

#### Common Solutions

**1. Re-index Vault**
```bash
# Full re-indexing
rm -f ~/.jarvis/jarvis.duckdb
uv run jarvis index --vault "$JARVIS_VAULT_PATH"
```

**2. Check File Formats**
```bash
# Ensure files are readable
find "$JARVIS_VAULT_PATH" -name "*.md" -exec file {} \; | grep -v "UTF-8\|ASCII"
```

**3. Model Issues**
```bash
# Clear model cache
rm -rf ~/.cache/torch/sentence_transformers/

# Force model re-download
uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./models')
"
```

### Search Results Are Poor Quality

#### Symptom
Search returns irrelevant or unexpected results

#### Diagnostic Steps
```bash
# 1. Test with different queries
uv run jarvis mcp --vault "$JARVIS_VAULT_PATH" --test-query "artificial intelligence"
uv run jarvis mcp --vault "$JARVIS_VAULT_PATH" --test-query "project management"

# 2. Check similarity scores
# Look for scores below 0.5 which might indicate poor matches

# 3. Examine indexed content
uv run python -c "
import duckdb
conn = duckdb.connect('~/.jarvis/jarvis.duckdb')
samples = conn.execute('SELECT path, content FROM documents LIMIT 5').fetchall()
for path, content in samples:
    print(f'{path}: {content[:100]}...')
"
```

#### Common Solutions

**1. Adjust Similarity Threshold**
```bash
# Use higher threshold for more precise results
export JARVIS_SIMILARITY_THRESHOLD=0.75
```

**2. Better Query Phrasing**
```bash
# Instead of: "AI"
# Use: "artificial intelligence machine learning"

# Instead of: "notes"
# Use: "meeting notes project planning"
```

**3. Re-index with Better Content**
```bash
# Check for empty or corrupted files
find "$JARVIS_VAULT_PATH" -name "*.md" -size 0
find "$JARVIS_VAULT_PATH" -name "*.md" -exec grep -L "." {} \;

# Remove or fix problematic files
```

### Graph Search Fails

#### Symptom
```
Graph search is currently unavailable: Neo4j connection failed
```

#### Diagnostic Steps
```bash
# 1. Check Neo4j status
neo4j status

# 2. Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1"

# 3. Check graph data
cypher-shell -a bolt://localhost:7687 -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n)"

# 4. Check environment variables
echo "NEO4J_URI: $NEO4J_URI"
echo "NEO4J_USER: $NEO4J_USER"
echo "NEO4J_PASSWORD: [set: $([ -n "$NEO4J_PASSWORD" ] && echo "yes" || echo "no")]"
```

#### Common Solutions

**1. Start Neo4j**
```bash
# Start Neo4j service
neo4j start

# Or for system service
sudo systemctl start neo4j
```

**2. Reset Neo4j Password**
```bash
# Stop Neo4j
neo4j stop

# Reset password
neo4j-admin set-initial-password new_password

# Update environment
export NEO4J_PASSWORD=new_password

# Start Neo4j
neo4j start
```

**3. Re-index Graph Data**
```bash
# Clear graph database
cypher-shell -a bolt://localhost:7687 -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) DETACH DELETE n"

# Re-index
uv run jarvis graph-index --vault "$JARVIS_VAULT_PATH"
```

---

## Database Issues

### Database File Corruption

#### Symptom
```
VectorDatabase error: Unable to connect to database
Database file appears to be corrupted
```

#### Diagnostic Steps
```bash
# 1. Check file integrity
file ~/.jarvis/jarvis.duckdb
ls -la ~/.jarvis/jarvis.duckdb

# 2. Try to open database
uv run python -c "
import duckdb
try:
    conn = duckdb.connect('~/.jarvis/jarvis.duckdb')
    print('Database opened successfully')
    conn.execute('PRAGMA database_size')
    print('Database integrity check passed')
except Exception as e:
    print(f'Database error: {e}')
"

# 3. Check disk space
df -h ~/.jarvis/
```

#### Solutions

**1. Rebuild Database**
```bash
# Backup if possible
cp ~/.jarvis/jarvis.duckdb ~/.jarvis/jarvis.duckdb.backup

# Remove corrupted database
rm ~/.jarvis/jarvis.duckdb

# Re-index from vault
uv run jarvis index --vault "$JARVIS_VAULT_PATH"
```

**2. Recover from Backup**
```bash
# If you have a backup
cp ~/.jarvis/jarvis.duckdb.backup ~/.jarvis/jarvis.duckdb
```

**3. Change Database Location**
```bash
# Use different location
export JARVIS_DB_PATH="/tmp/jarvis.duckdb"
uv run jarvis index --vault "$JARVIS_VAULT_PATH"
```

### Database Performance Issues

#### Symptom
Searches are very slow or hang

#### Diagnostic Steps
```bash
# 1. Check database size
ls -lh ~/.jarvis/jarvis.duckdb

# 2. Check available space
df -h ~/.jarvis/

# 3. Monitor during search
iostat -x 1

# 4. Check for locks
lsof ~/.jarvis/jarvis.duckdb
```

#### Solutions

**1. Optimize Database**
```bash
# Run VACUUM to optimize
uv run python -c "
import duckdb
conn = duckdb.connect('~/.jarvis/jarvis.duckdb')
conn.execute('VACUUM')
conn.execute('ANALYZE')
print('Database optimized')
"
```

**2. Adjust Performance Settings**
```bash
# Reduce batch size
export JARVIS_BATCH_SIZE=8

# Increase similarity threshold to reduce results
export JARVIS_SIMILARITY_THRESHOLD=0.7
```

**3. Index Management**
```bash
# Check if re-indexing is needed
uv run python -c "
import duckdb
import os
from datetime import datetime

conn = duckdb.connect('~/.jarvis/jarvis.duckdb')
try:
    result = conn.execute('SELECT MAX(indexed_at) FROM documents').fetchone()
    if result[0]:
        print(f'Last indexed: {result[0]}')
    else:
        print('No indexing timestamp found')
except:
    print('Database needs re-indexing')
"
```

---

## Performance Problems

### Slow Startup

#### Symptom
MCP server or searches take a long time to start

#### Diagnostic Steps
```bash
# 1. Time the startup
time uv run jarvis list-vaults

# 2. Check model loading time
time uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Model loaded')
"

# 3. Check disk I/O
iostat -x 1

# 4. Check memory usage
free -h
```

#### Solutions

**1. Use Smaller Model**
```bash
# Switch to smaller, faster model
export JARVIS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

**2. Preload Model**
```bash
# Create model cache
uv run python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Model is now cached
"
```

**3. Optimize System**
```bash
# Increase available memory
sudo sysctl vm.swappiness=10

# For SSD, enable TRIM
sudo fstrim -av
```

### High Memory Usage

#### Symptom
System becomes slow, high memory usage reported

#### Diagnostic Steps
```bash
# 1. Monitor memory usage
while true; do
    ps aux | grep jarvis | grep -v grep
    free -h
    sleep 5
done

# 2. Check for memory leaks
valgrind --tool=memcheck --leak-check=full uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"

# 3. Monitor file descriptors
lsof -p $(pgrep -f jarvis)
```

#### Solutions

**1. Reduce Memory Usage**
```bash
# Smaller batch size
export JARVIS_BATCH_SIZE=4

# Smaller cache
export JARVIS_CACHE_SIZE=16

# Use CPU instead of GPU
export JARVIS_DEVICE=cpu
```

**2. Restart Service Periodically**
```bash
# Create restart script
cat > restart_jarvis.sh << 'EOF'
#!/bin/bash
pkill -f "jarvis mcp"
sleep 5
uv run jarvis mcp --vault "$JARVIS_VAULT_PATH" &
EOF

chmod +x restart_jarvis.sh
```

**3. Monitor and Alert**
```bash
# Memory monitoring script
cat > monitor_memory.sh << 'EOF'
#!/bin/bash
while true; do
    memory_percent=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
    if (( $(echo "$memory_percent > 85.0" | bc -l) )); then
        echo "High memory usage: ${memory_percent}%"
    fi
    sleep 60
done
EOF
```

---

## Configuration Issues

### Environment Variable Problems

#### Symptom
```
Configuration error: Required environment variable not set
```

#### Diagnostic Steps
```bash
# 1. Check all environment variables
env | grep JARVIS
env | grep NEO4J

# 2. Check shell configuration
echo $SHELL
grep -r JARVIS ~/.bashrc ~/.zshrc ~/.profile 2>/dev/null

# 3. Check if variables persist
echo $JARVIS_VAULT_PATH
exec bash -l
echo $JARVIS_VAULT_PATH
```

#### Solutions

**1. Set Environment Variables Permanently**
```bash
# Add to shell profile
echo 'export JARVIS_VAULT_PATH="/path/to/vault"' >> ~/.bashrc
echo 'export NEO4J_PASSWORD="your_password"' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

**2. Use Configuration File**
```bash
# Create config file
cat > ~/.jarvis/config.env << EOF
JARVIS_VAULT_PATH="/path/to/vault"
JARVIS_DB_PATH="$HOME/.jarvis/jarvis.duckdb"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your_password"
EOF

# Load before running
source ~/.jarvis/config.env
uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"
```

**3. Use Environment File**
```bash
# Create config/.env file in project directory
mkdir -p config
cat > config/.env << EOF
JARVIS_VAULT_PATH=/path/to/vault
NEO4J_PASSWORD=your_password
EOF

# Load with direnv or manually
source config/.env
```

### Path Configuration Issues

#### Symptom
```
Error: Vault path does not exist
File not found errors
```

#### Diagnostic Steps
```bash
# 1. Check path resolution
echo "JARVIS_VAULT_PATH: $JARVIS_VAULT_PATH"
realpath "$JARVIS_VAULT_PATH" 2>/dev/null || echo "Path does not exist"

# 2. Check permissions
ls -ld "$JARVIS_VAULT_PATH"
ls -la "$JARVIS_VAULT_PATH" | head -5

# 3. Check for spaces or special characters
echo "$JARVIS_VAULT_PATH" | od -c
```

#### Solutions

**1. Use Absolute Paths**
```bash
# Convert to absolute path
export JARVIS_VAULT_PATH="$(realpath ~/Documents/ObsidianVault)"
```

**2. Handle Spaces in Paths**
```bash
# Proper quoting
export JARVIS_VAULT_PATH="/Users/username/Documents/My Vault"

# Or use escape characters
export JARVIS_VAULT_PATH="/Users/username/Documents/My\ Vault"
```

**3. Create Missing Directories**
```bash
# Create vault directory
mkdir -p "$JARVIS_VAULT_PATH"

# Create sample content
echo "# Test Note" > "$JARVIS_VAULT_PATH/test.md"
```

---

## System Integration

### Claude Desktop Not Finding MCP Server

#### Symptom
Claude Desktop starts but doesn't show Jarvis Assistant tools

#### Diagnostic Steps
```bash
# 1. Check Claude Desktop config location
ls -la ~/.config/claude-desktop/claude_desktop_config.json

# Or on macOS:
ls -la ~/Library/Application\ Support/Claude/claude_desktop_config.json

# 2. Validate JSON format
python -m json.tool ~/.config/claude-desktop/claude_desktop_config.json

# 3. Test MCP server manually
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}' | uv run jarvis mcp --vault "$JARVIS_VAULT_PATH"
```

#### Solutions

**1. Correct Configuration Path**
```bash
# Linux/WSL
mkdir -p ~/.config/claude-desktop/

# macOS
mkdir -p ~/Library/Application\ Support/Claude/
```

**2. Validate Configuration**
```json
{
  "mcpServers": {
    "jarvis-assistant": {
      "command": "uv",
      "args": ["run", "jarvis", "mcp", "--vault", "/absolute/path/to/vault"],
      "env": {
        "NEO4J_PASSWORD": "your_password",
        "JARVIS_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**3. Restart and Test**
```bash
# Kill Claude Desktop completely
pkill -f Claude

# Wait and restart
sleep 5
open -a "Claude Desktop"  # macOS
# or
claude-desktop  # Linux
```

### Permission Denied Errors

#### Symptom
```
Permission denied accessing vault
Cannot create database file
```

#### Diagnostic Steps
```bash
# 1. Check file permissions
ls -la ~/.jarvis/
ls -la "$JARVIS_VAULT_PATH"

# 2. Check user ownership
whoami
ls -la ~/.jarvis/ | head -2

# 3. Check filesystem
df -h ~/.jarvis/
mount | grep $(df ~/.jarvis/ | tail -1 | awk '{print $1}')
```

#### Solutions

**1. Fix File Permissions**
```bash
# Fix directory permissions
chmod 755 ~/.jarvis/
chmod -R 755 "$JARVIS_VAULT_PATH"

# Fix file permissions
find ~/.jarvis/ -type f -exec chmod 644 {} \;
find "$JARVIS_VAULT_PATH" -name "*.md" -exec chmod 644 {} \;
```

**2. Fix Ownership**
```bash
# Take ownership
sudo chown -R $(whoami):$(id -gn) ~/.jarvis/
sudo chown -R $(whoami):$(id -gn) "$JARVIS_VAULT_PATH"
```

**3. SELinux Issues (Linux)**
```bash
# Check SELinux status
getenforce

# If enforcing, check context
ls -Z ~/.jarvis/

# Fix SELinux context if needed
restorecon -R ~/.jarvis/
```

### Network Connectivity Issues

#### Symptom
```
Neo4j connection failed: Connection refused
Network timeout errors
```

#### Diagnostic Steps
```bash
# 1. Check Neo4j is listening
netstat -tlnp | grep 7687
ss -tlnp | grep 7687

# 2. Test connectivity
telnet localhost 7687
nc -zv localhost 7687

# 3. Check firewall
sudo iptables -L | grep 7687
sudo ufw status | grep 7687

# 4. Check Neo4j logs
tail -f /var/log/neo4j/neo4j.log
```

#### Solutions

**1. Start Neo4j**
```bash
# Check status
neo4j status

# Start if stopped
neo4j start

# Enable auto-start
sudo systemctl enable neo4j
```

**2. Fix Network Configuration**
```bash
# Check Neo4j configuration
grep -r "bolt" /etc/neo4j/neo4j.conf

# Ensure Neo4j listens on all interfaces
echo "dbms.connector.bolt.listen_address=0.0.0.0:7687" | sudo tee -a /etc/neo4j/neo4j.conf
```

**3. Firewall Configuration**
```bash
# Allow Neo4j port
sudo ufw allow 7687
# or
sudo iptables -A INPUT -p tcp --dport 7687 -j ACCEPT
```

---

## Emergency Recovery

### Complete System Reset

If all else fails, perform a complete reset:

```bash
#!/bin/bash
echo "=== Emergency Recovery ==="

# 1. Stop all services
pkill -f jarvis
neo4j stop

# 2. Backup important data
mkdir -p ~/jarvis-backup
cp -r ~/.jarvis/ ~/jarvis-backup/ 2>/dev/null || echo "No existing data"

# 3. Clean installation
rm -rf ~/.jarvis/
rm -rf ~/.cache/torch/sentence_transformers/

# 4. Reinstall
cd /path/to/jarvis-assistant
uv sync

# 5. Reset Neo4j
neo4j-admin set-initial-password new_password
neo4j start

# 6. Set environment
export JARVIS_VAULT_PATH="/path/to/vault"
export NEO4J_PASSWORD="new_password"

# 7. Re-index everything
uv run jarvis index --vault "$JARVIS_VAULT_PATH"
uv run jarvis graph-index --vault "$JARVIS_VAULT_PATH"

# 8. Test
uv run jarvis list-vaults

echo "=== Recovery Complete ==="
```

### Data Recovery

If you need to recover from corrupted data:

```bash
# Check for backups
ls -la ~/.jarvis/*.backup
ls -la ~/jarvis-backup/

# Restore from backup
cp ~/.jarvis/jarvis.duckdb.backup ~/.jarvis/jarvis.duckdb

# Or rebuild from source
find "$JARVIS_VAULT_PATH" -name "*.md" -exec echo "Found: {}" \;
uv run jarvis index --vault "$JARVIS_VAULT_PATH" --force
```

---

## Getting Help

### Information to Collect

When seeking help, collect this information:

```bash
# System information
uname -a
python --version
uv --version

# Environment
env | grep JARVIS
env | grep NEO4J

# Status
uv run jarvis --version
neo4j status
ls -la ~/.jarvis/

# Logs (sanitize sensitive information)
tail -100 ~/.jarvis/logs/jarvis.log 2>/dev/null || echo "No log file"
```

### Log Collection Script

```bash
#!/bin/bash
# Collect diagnostic information

echo "=== Jarvis Assistant Diagnostics ===" > jarvis-diag.txt
echo "Date: $(date)" >> jarvis-diag.txt
echo "System: $(uname -a)" >> jarvis-diag.txt
echo "Python: $(python --version)" >> jarvis-diag.txt
echo "UV: $(uv --version)" >> jarvis-diag.txt
echo "" >> jarvis-diag.txt

echo "=== Environment ===" >> jarvis-diag.txt
env | grep JARVIS | sed 's/PASSWORD=.*/PASSWORD=***/' >> jarvis-diag.txt
env | grep NEO4J | sed 's/PASSWORD=.*/PASSWORD=***/' >> jarvis-diag.txt
echo "" >> jarvis-diag.txt

echo "=== File Status ===" >> jarvis-diag.txt
ls -la ~/.jarvis/ >> jarvis-diag.txt 2>&1
echo "" >> jarvis-diag.txt

echo "=== Recent Errors ===" >> jarvis-diag.txt
tail -50 ~/.jarvis/logs/jarvis.log >> jarvis-diag.txt 2>/dev/null || echo "No log file" >> jarvis-diag.txt

echo "Diagnostics saved to jarvis-diag.txt"
echo "Please sanitize any sensitive information before sharing"
```

---

## Next Steps

- [Performance Tuning](performance-tuning.md) - Optimization techniques
- [Backup Recovery](backup-recovery.md) - Data protection strategies
- [Error Codes](../06-reference/error-codes.md) - Complete error reference
- [Configuration Reference](../06-reference/configuration-reference.md) - Setup options
