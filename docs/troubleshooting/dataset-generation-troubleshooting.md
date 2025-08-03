# Dataset Generation Troubleshooting Guide

This guide provides solutions to common issues encountered when using the dataset generation tool in Jarvis Assistant.

## Quick Diagnostics

### Health Check Commands

Run these commands to quickly diagnose issues:

```bash
# Check tool installation
python resources/scripts/dataset_tool_deployment.py --status

# Verify tool functionality
python resources/scripts/dataset_tool_deployment.py --verify-only

# Test CLI command
jarvis generate-dataset --help

# Check Jarvis configuration
jarvis stats --vault /path/to/vault
```

### Log Analysis

Enable verbose logging to get detailed information:

```bash
# Enable verbose logging
jarvis -v generate-dataset --vault /path/to/vault

# Check log files
tail -f ~/.jarvis/mcp_server.log
```

## Installation Issues

### Issue: Command Not Found

**Error**: `jarvis: command not found` or `generate-dataset: command not found`

**Diagnosis**:
```bash
# Check if you're in the right directory
pwd
ls -la src/jarvis/main.py

# Check Python path
python -c "import sys; print(sys.path)"
```

**Solutions**:
1. **Use full Python module path**:
   ```bash
   python -m jarvis.main generate-dataset --help
   ```

2. **Verify installation**:
   ```bash
   python resources/scripts/dataset_tool_deployment.py --verify-only
   ```

3. **Reinstall the tool**:
   ```bash
   python resources/scripts/dataset_tool_deployment.py --force
   ```

### Issue: Import Errors

**Error**: `ImportError: No module named 'jarvis.tools.dataset_generation'`

**Diagnosis**:
```bash
# Check if tool directory exists
ls -la src/jarvis/tools/dataset_generation/

# Check Python path
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from jarvis.tools.dataset_generation import DatasetGenerator
    print('Import successful')
except ImportError as e:
    print(f'Import failed: {e}')
"
```

**Solutions**:
1. **Verify tool installation**:
   ```bash
   python resources/scripts/dataset_tool_deployment.py --status
   ```

2. **Reinstall dependencies**:
   ```bash
   uv sync
   ```

3. **Check file permissions**:
   ```bash
   find src/jarvis/tools/dataset_generation -name "*.py" -exec ls -la {} \;
   ```

### Issue: Missing Dependencies

**Error**: `ImportError: No module named 'pandas'` or similar

**Diagnosis**:
```bash
# Check installed packages
uv pip list | grep -E "(pandas|networkx|numpy|scipy|sentence-transformers)"

# Test imports
python -c "
import pandas
import networkx
import numpy
import scipy
from sentence_transformers import SentenceTransformer
print('All dependencies available')
"
```

**Solutions**:
1. **Install missing dependencies**:
   ```bash
   uv sync
   ```

2. **Install specific packages**:
   ```bash
   uv add pandas networkx numpy scipy sentence-transformers
   ```

## Configuration Issues

### Issue: Vault Path Not Found

**Error**: `Error: Vault path does not exist: /path/to/vault`

**Diagnosis**:
```bash
# Check if path exists
ls -la /path/to/vault

# Check for .md files
find /path/to/vault -name "*.md" | head -5

# Check permissions
ls -ld /path/to/vault
```

**Solutions**:
1. **Use absolute path**:
   ```bash
   jarvis generate-dataset --vault "$(pwd)/path/to/vault"
   ```

2. **Check path expansion**:
   ```bash
   # If using ~, make sure it expands correctly
   echo ~/path/to/vault
   ```

3. **Set environment variable**:
   ```bash
   export JARVIS_VAULT_PATH="/absolute/path/to/vault"
   jarvis generate-dataset
   ```

### Issue: Output Directory Permissions

**Error**: `PermissionError: [Errno 13] Permission denied: '/path/to/output'`

**Diagnosis**:
```bash
# Check directory permissions
ls -ld /path/to/output
ls -ld /path/to/output/..

# Check if directory exists
mkdir -p /path/to/output
```

**Solutions**:
1. **Use writable directory**:
   ```bash
   jarvis generate-dataset --vault /path/to/vault --output ./datasets
   ```

2. **Fix permissions**:
   ```bash
   chmod 755 /path/to/output
   ```

3. **Use default output**:
   ```bash
   # Let the tool create the default directory
   jarvis generate-dataset --vault /path/to/vault
   ```

## Processing Issues

### Issue: No Notes Found

**Error**: `No notes found to process` or `Total Notes: 0`

**Diagnosis**:
```bash
# Check for markdown files
find /path/to/vault -name "*.md" -type f | wc -l

# Check file extensions
find /path/to/vault -type f | head -10

# Check for hidden files
ls -la /path/to/vault
```

**Solutions**:
1. **Verify file extensions**:
   ```bash
   # Look for different extensions
   find /path/to/vault -name "*.markdown" -o -name "*.txt"
   ```

2. **Check excluded folders**:
   ```bash
   # Check if notes are in excluded folders
   echo $JARVIS_EXCLUDED_FOLDERS
   ```

3. **Test with specific folder**:
   ```bash
   # Create a test vault with known files
   mkdir test_vault
   echo "# Test Note\nThis is a test." > test_vault/test.md
   jarvis generate-dataset --vault test_vault
   ```

### Issue: Link Extraction Returns Zero Links

**Error**: `Total Links: 0` or `Positive Pairs: 0`

**Diagnosis**:
```bash
# Check for different link formats
grep -r "\[\[" /path/to/vault | head -5
grep -r "\](" /path/to/vault | head -5
grep -r "![[" /path/to/vault | head -5

# Check specific files
head -20 /path/to/vault/some_note.md
```

**Solutions**:
1. **Verify link formats**: Ensure your notes use supported link formats:
   - Wikilinks: `[[Note Name]]`
   - Markdown links: `[Display Text](note.md)`
   - Embedded links: `![[Image.png]]`

2. **Test link extraction manually**:
   ```python
   from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
   from jarvis.services.vault.reader import VaultReader
   
   reader = VaultReader("/path/to/vault")
   extractor = LinkExtractor(reader)
   
   # Test with a specific file
   content, _ = reader.read_file("some_note.md")
   links = extractor._extract_links_from_content(content, "some_note.md")
   print(f"Found {len(links)} links")
   ```

3. **Check file encoding**:
   ```bash
   file /path/to/vault/*.md | head -5
   ```

### Issue: Memory Errors

**Error**: `MemoryError` or `Out of memory`

**Diagnosis**:
```bash
# Check available memory
free -h

# Check vault size
du -sh /path/to/vault
find /path/to/vault -name "*.md" | wc -l

# Monitor memory usage during processing
top -p $(pgrep -f "jarvis.*generate-dataset")
```

**Solutions**:
1. **Reduce batch size**:
   ```bash
   jarvis generate-dataset \
       --vault /path/to/vault \
       --batch-size 8 \
       --max-pairs 100
   ```

2. **Use sampling**:
   ```bash
   jarvis generate-dataset \
       --vault /path/to/vault \
       --negative-ratio 2.0 \
       --sampling random
   ```

3. **Process in chunks**: For very large vaults, process subsets:
   ```bash
   # Process specific folders
   jarvis generate-dataset --vault /path/to/vault/folder1
   jarvis generate-dataset --vault /path/to/vault/folder2
   ```

### Issue: Slow Processing

**Problem**: Dataset generation takes too long

**Diagnosis**:
```bash
# Check vault size
find /path/to/vault -name "*.md" | wc -l
du -sh /path/to/vault

# Monitor CPU usage
top -p $(pgrep -f "jarvis.*generate-dataset")

# Check if GPU is being used for embeddings
nvidia-smi  # If you have NVIDIA GPU
```

**Solutions**:
1. **Optimize batch size**:
   ```bash
   # Increase batch size if you have memory
   jarvis generate-dataset \
       --vault /path/to/vault \
       --batch-size 64
   ```

2. **Reduce dataset size**:
   ```bash
   jarvis generate-dataset \
       --vault /path/to/vault \
       --negative-ratio 3.0 \
       --max-pairs 500
   ```

3. **Use faster sampling**:
   ```bash
   jarvis generate-dataset \
       --vault /path/to/vault \
       --sampling random
   ```

## Data Quality Issues

### Issue: Poor Link Prediction Accuracy

**Problem**: Generated dataset doesn't train good models

**Diagnosis**:
```python
import pandas as pd

# Load and analyze the dataset
pairs_df = pd.read_csv("pairs_dataset.csv")

print("Dataset shape:", pairs_df.shape)
print("Link ratio:", pairs_df['link_exists'].mean())
print("Feature statistics:")
print(pairs_df.describe())

# Check for missing values
print("Missing values:")
print(pairs_df.isnull().sum())
```

**Solutions**:
1. **Adjust negative sampling ratio**:
   ```bash
   # Try different ratios
   jarvis generate-dataset --negative-ratio 5.0  # More balanced
   jarvis generate-dataset --negative-ratio 10.0  # More negatives
   ```

2. **Use stratified sampling**:
   ```bash
   jarvis generate-dataset --sampling stratified
   ```

3. **Check vault quality**:
   - Ensure notes are properly linked
   - Remove or fix broken links
   - Use consistent linking patterns

### Issue: Missing or Invalid Features

**Problem**: Features contain NaN or unexpected values

**Diagnosis**:
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("pairs_dataset.csv")

# Check for problematic values
print("NaN values per column:")
print(df.isnull().sum())

print("Infinite values:")
print(np.isinf(df.select_dtypes(include=[np.number])).sum())

print("Feature ranges:")
print(df.describe())
```

**Solutions**:
1. **Handle missing embeddings**:
   ```bash
   # Regenerate with smaller batch size
   jarvis generate-dataset --batch-size 16
   ```

2. **Check graph connectivity**:
   ```python
   # Verify graph structure
   pairs_df = pd.read_csv("pairs_dataset.csv")
   connected_pairs = pairs_df[pairs_df['shortest_path_length'] != -1]
   print(f"Connected pairs: {len(connected_pairs)}/{len(pairs_df)}")
   ```

3. **Validate note content**:
   ```bash
   # Check for empty or corrupted files
   find /path/to/vault -name "*.md" -size 0
   ```

## Service Integration Issues

### Issue: VaultReader Errors

**Error**: `VaultReader initialization failed` or file reading errors

**Diagnosis**:
```python
# Test VaultReader directly
from jarvis.services.vault.reader import VaultReader

try:
    reader = VaultReader("/path/to/vault")
    files = reader.get_markdown_files()
    print(f"Found {len(files)} files")
    
    # Test reading a file
    if files:
        content, metadata = reader.read_file(files[0])
        print(f"Successfully read {files[0]}")
except Exception as e:
    print(f"VaultReader error: {e}")
```

**Solutions**:
1. **Check file permissions**:
   ```bash
   find /path/to/vault -name "*.md" ! -readable
   ```

2. **Test with simple vault**:
   ```bash
   mkdir simple_vault
   echo "# Test\n[[Other Note]]" > simple_vault/test.md
   echo "# Other Note\nContent here" > simple_vault/other.md
   jarvis generate-dataset --vault simple_vault
   ```

### Issue: VectorEncoder Errors

**Error**: `Embedding generation failed` or CUDA errors

**Diagnosis**:
```python
# Test VectorEncoder directly
from jarvis.services.vector.encoder import VectorEncoder

try:
    encoder = VectorEncoder()
    test_docs = ["This is a test document", "Another test document"]
    embeddings = encoder.encode_documents(test_docs)
    print(f"Generated embeddings shape: {embeddings.shape}")
except Exception as e:
    print(f"VectorEncoder error: {e}")
```

**Solutions**:
1. **Check device configuration**:
   ```bash
   # Set CPU-only mode if GPU issues
   export JARVIS_EMBEDDING_DEVICE="cpu"
   jarvis generate-dataset --vault /path/to/vault
   ```

2. **Try different model**:
   ```bash
   export JARVIS_EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
   jarvis generate-dataset --vault /path/to/vault
   ```

3. **Reduce batch size**:
   ```bash
   jarvis generate-dataset --batch-size 8
   ```

### Issue: GraphDatabase Errors

**Error**: Graph database connection or query errors

**Diagnosis**:
```python
# Test GraphDatabase directly
from jarvis.services.graph.database import GraphDatabase
from jarvis.utils.config import get_settings

try:
    settings = get_settings()
    if settings.graph_enabled:
        db = GraphDatabase(settings)
        # Test connection
        print("Graph database connection successful")
        db.close()
    else:
        print("Graph database disabled in settings")
except Exception as e:
    print(f"GraphDatabase error: {e}")
```

**Solutions**:
1. **Disable graph features temporarily**:
   ```bash
   export JARVIS_GRAPH_ENABLED=false
   jarvis generate-dataset --vault /path/to/vault
   ```

2. **Check Neo4j connection**:
   ```bash
   # Test Neo4j connectivity
   curl -u neo4j:password http://localhost:7474/db/data/
   ```

## Performance Optimization

### Memory Optimization

For large vaults or limited memory:

```bash
# Minimal memory usage
jarvis generate-dataset \
    --vault /path/to/vault \
    --batch-size 8 \
    --negative-ratio 2.0 \
    --max-pairs 200 \
    --sampling random
```

### Speed Optimization

For faster processing:

```bash
# Maximum speed (requires more memory)
jarvis generate-dataset \
    --vault /path/to/vault \
    --batch-size 128 \
    --negative-ratio 3.0 \
    --sampling random
```

### Quality vs Speed Trade-offs

| Priority | Batch Size | Negative Ratio | Sampling | Max Pairs |
|----------|------------|----------------|----------|-----------|
| Quality | 32 | 10.0 | stratified | 2000 |
| Balanced | 32 | 5.0 | stratified | 1000 |
| Speed | 64 | 3.0 | random | 500 |
| Memory | 8 | 2.0 | random | 200 |

## Recovery and Cleanup

### Partial Generation Recovery

If generation fails partway through:

```bash
# Check for partial output files
ls -la ./datasets/

# Clean up and restart
rm -f ./datasets/*.csv
jarvis generate-dataset --vault /path/to/vault
```

### Complete Tool Removal

If you need to remove the tool completely:

```bash
# Preview what will be removed
python resources/scripts/dataset_tool_cleanup.py --dry-run

# Remove the tool
python resources/scripts/dataset_tool_cleanup.py

# Verify core functionality still works
jarvis --help
```

### Backup and Restore

Before making changes:

```bash
# Create backup
python resources/scripts/dataset_tool_deployment.py --status
cp -r src/jarvis/tools/dataset_generation ~/backup_dataset_tool

# Restore if needed
rm -rf src/jarvis/tools/dataset_generation
cp -r ~/backup_dataset_tool src/jarvis/tools/dataset_generation
```

## Getting Additional Help

### Debug Information Collection

When reporting issues, collect this information:

```bash
# System information
python --version
uv --version
echo $JARVIS_VAULT_PATH

# Tool status
python resources/scripts/dataset_tool_deployment.py --status

# Test with minimal example
mkdir debug_vault
echo "# Note 1\n[[Note 2]]" > debug_vault/note1.md
echo "# Note 2\nContent" > debug_vault/note2.md
jarvis -v generate-dataset --vault debug_vault --output debug_output
```

### Log Analysis

Enable detailed logging:

```bash
export JARVIS_LOG_LEVEL=DEBUG
jarvis -v generate-dataset --vault /path/to/vault 2>&1 | tee generation.log
```

### Common Log Patterns

Look for these patterns in logs:

- `LinkExtractionError`: Link parsing issues
- `FeatureEngineeringError`: Feature computation problems
- `DataQualityError`: Data validation failures
- `MemoryError`: Memory exhaustion
- `TimeoutError`: Processing timeouts

This troubleshooting guide covers the most common issues. For complex problems, the debug information and logs will help identify the specific cause.