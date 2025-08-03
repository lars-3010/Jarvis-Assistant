# Dataset Generation Tool Migration Guide

This guide helps you migrate from the existing dataset generation script (`resources/data-generation/generate_dataset.py`) to the new integrated dataset generation tool in Jarvis Assistant.

## Overview

The dataset generation functionality has been moved from a standalone script to a fully integrated CLI command within Jarvis Assistant. This provides better integration with existing services, improved error handling, and enhanced features.

## Key Changes

### Before (Old Script)
- **Location**: `resources/data-generation/generate_dataset.py`
- **Usage**: `python resources/data-generation/generate_dataset.py --vault /path/to/vault --output dataset.csv`
- **Scope**: Limited to `/Areas` folder only
- **Features**: Basic link prediction dataset generation
- **Integration**: Standalone script with minimal Jarvis integration

### After (New Tool)
- **Location**: Integrated into `jarvis` CLI as `generate-dataset` command
- **Usage**: `jarvis generate-dataset --vault /path/to/vault --output /path/to/output`
- **Scope**: Full vault processing with configurable filtering
- **Features**: Comprehensive dataset generation with multiple output formats
- **Integration**: Full integration with Jarvis services and configuration

## Migration Steps

### Step 1: Verify New Tool Installation

First, verify that the new dataset generation tool is properly installed:

```bash
# Check if the command is available
jarvis generate-dataset --help

# Verify installation status
python resources/scripts/dataset_tool_deployment.py --status
```

### Step 2: Update Your Workflow Scripts

If you have scripts that call the old dataset generation script, update them to use the new CLI command.

#### Old Script Call
```bash
python resources/data-generation/generate_dataset.py \
    --vault /path/to/vault \
    --output areas_dataset.csv
```

#### New CLI Command
```bash
jarvis generate-dataset \
    --vault /path/to/vault \
    --output /path/to/output \
    --notes-filename areas_notes_dataset.csv \
    --pairs-filename areas_pairs_dataset.csv
```

### Step 3: Update Configuration

The new tool uses Jarvis configuration settings. Update your environment or configuration:

#### Environment Variables
```bash
# Set default vault path
export JARVIS_VAULT_PATH="/path/to/your/vault"

# Set default dataset output directory
export JARVIS_DATASET_OUTPUT_DIR="/path/to/datasets"

# Configure embedding settings
export JARVIS_EMBEDDING_MODEL_NAME="paraphrase-MiniLM-L6-v2"
export JARVIS_EMBEDDING_BATCH_SIZE=32
```

#### Configuration File (.env)
```env
JARVIS_VAULT_PATH=/path/to/your/vault
JARVIS_DATASET_OUTPUT_DIR=./datasets
JARVIS_EMBEDDING_MODEL_NAME=paraphrase-MiniLM-L6-v2
JARVIS_EMBEDDING_BATCH_SIZE=32
```

### Step 4: Migrate Custom Processing Logic

If you've customized the old script, here's how to adapt your changes:

#### Custom Note Filtering
The old script was hardcoded to process only the `/Areas` folder. The new tool processes the entire vault by default, but you can filter using:

```python
# In your custom script, you can still use the DatasetGenerator directly
from jarvis.tools.dataset_generation import DatasetGenerator
from pathlib import Path

# Create a custom generator with filtering
vault_path = Path("/path/to/vault")
output_dir = Path("./datasets")

with DatasetGenerator(vault_path, output_dir) as generator:
    # You can implement custom filtering in the generator
    # or pre-filter the notes before processing
    result = generator.generate_datasets(
        notes_filename="custom_notes.csv",
        pairs_filename="custom_pairs.csv"
    )
```

#### Custom Feature Engineering
The new tool provides more comprehensive features. If you had custom features in the old script:

```python
# Old script custom features
def custom_features(note_a, note_b):
    return {
        "custom_metric": calculate_custom_metric(note_a, note_b)
    }

# New tool: extend the PairFeatures model or use post-processing
import pandas as pd

# Generate standard dataset
result = generator.generate_datasets()

# Load and extend with custom features
pairs_df = pd.read_csv(result.pairs_dataset_path)
pairs_df['custom_metric'] = pairs_df.apply(
    lambda row: calculate_custom_metric_from_row(row), axis=1
)
pairs_df.to_csv("extended_pairs_dataset.csv", index=False)
```

### Step 5: Update Data Processing Pipelines

If you have downstream processing that depends on the old dataset format:

#### Schema Changes

**Old Schema** (single CSV file):
```csv
note_a,note_b,cosine_similarity,shortest_path,adamic_adar,jaccard_coefficient,common_tags,tag_jaccard,length_difference,length_ratio,link_exists
```

**New Schema** (two CSV files):

**Notes Dataset** (`notes_dataset.csv`):
```csv
note_path,note_title,word_count,tag_count,quality_stage,creation_date,last_modified,outgoing_links_count,incoming_links_count,betweenness_centrality,closeness_centrality,pagerank_score,semantic_cluster_id,semantic_summary,...
```

**Pairs Dataset** (`pairs_dataset.csv`):
```csv
note_a_path,note_b_path,cosine_similarity,semantic_cluster_match,tag_overlap_count,tag_jaccard_similarity,vault_path_distance,shortest_path_length,common_neighbors_count,adamic_adar_score,word_count_ratio,creation_time_diff_days,link_exists,...
```

#### Adaptation Script
If you need to convert the new format to match your existing pipeline:

```python
import pandas as pd

def convert_to_old_format(notes_csv, pairs_csv, output_csv):
    """Convert new format back to old format for compatibility."""
    notes_df = pd.read_csv(notes_csv)
    pairs_df = pd.read_csv(pairs_csv)
    
    # Create mapping from note path to features
    note_features = notes_df.set_index('note_path').to_dict('index')
    
    # Convert pairs dataset to old format
    old_format_data = []
    for _, row in pairs_df.iterrows():
        note_a = row['note_a_path']
        note_b = row['note_b_path']
        
        # Map new columns to old column names
        old_row = {
            'note_a': note_a,
            'note_b': note_b,
            'cosine_similarity': row['cosine_similarity'],
            'shortest_path': row['shortest_path_length'],
            'adamic_adar': row['adamic_adar_score'],
            'jaccard_coefficient': row.get('jaccard_coefficient', 0),  # May need calculation
            'common_tags': row['tag_overlap_count'],
            'tag_jaccard': row['tag_jaccard_similarity'],
            'length_difference': abs(note_features[note_a]['word_count'] - note_features[note_b]['word_count']),
            'length_ratio': row['word_count_ratio'],
            'link_exists': row['link_exists']
        }
        old_format_data.append(old_row)
    
    # Save in old format
    old_df = pd.DataFrame(old_format_data)
    old_df.to_csv(output_csv, index=False)
    print(f"Converted dataset saved to {output_csv}")

# Usage
convert_to_old_format("notes_dataset.csv", "pairs_dataset.csv", "legacy_format.csv")
```

### Step 6: Performance Considerations

The new tool includes performance optimizations not available in the old script:

#### Batch Processing
```bash
# Configure batch size for your system
jarvis generate-dataset \
    --vault /path/to/vault \
    --batch-size 64  # Increase for more memory, decrease for less
```

#### Sampling for Large Vaults
```bash
# Use intelligent sampling for large vaults
jarvis generate-dataset \
    --vault /path/to/vault \
    --negative-ratio 5.0 \
    --sampling stratified \
    --max-pairs 1000
```

### Step 7: Testing and Validation

After migration, validate that the new tool produces equivalent results:

```bash
# Generate datasets with new tool
jarvis generate-dataset --vault /path/to/vault --output ./new_datasets

# Compare with old script output (if you have it)
python -c "
import pandas as pd
old_df = pd.read_csv('old_dataset.csv')
new_pairs_df = pd.read_csv('./new_datasets/pairs_dataset.csv')

# Compare key metrics
print('Old dataset shape:', old_df.shape)
print('New pairs dataset shape:', new_pairs_df.shape)
print('Link ratio old:', old_df['link_exists'].mean())
print('Link ratio new:', new_pairs_df['link_exists'].mean())
"
```

### Step 8: Clean Up Old Script (Optional)

Once you've verified the migration works correctly, you can remove the old script:

```bash
# Backup the old script first
cp resources/data-generation/generate_dataset.py ~/backup_generate_dataset.py

# Remove the old script
rm resources/data-generation/generate_dataset.py

# Or keep it for reference but rename it
mv resources/data-generation/generate_dataset.py resources/data-generation/generate_dataset_legacy.py
```

## Troubleshooting

### Common Migration Issues

#### Issue: "Command not found: jarvis"
**Solution**: Make sure you're in the project directory and using the correct Python environment:
```bash
cd /path/to/jarvis-assistant
python -m jarvis.main generate-dataset --help
```

#### Issue: "ImportError: No module named 'jarvis.tools.dataset_generation'"
**Solution**: Verify the tool is properly installed:
```bash
python resources/scripts/dataset_tool_deployment.py --verify-only
```

#### Issue: Different results compared to old script
**Solution**: The new tool processes the entire vault, not just `/Areas`. To match old behavior:
1. Use a vault that contains only the Areas folder, or
2. Implement custom filtering in your processing pipeline

#### Issue: Performance is slower than old script
**Solution**: Adjust batch size and sampling parameters:
```bash
jarvis generate-dataset \
    --batch-size 64 \
    --negative-ratio 3.0 \
    --max-pairs 500
```

### Getting Help

If you encounter issues during migration:

1. **Check the logs**: Look for detailed error messages in the Jarvis logs
2. **Verify installation**: Run the deployment verification script
3. **Test with small dataset**: Try with a small vault first
4. **Check configuration**: Ensure environment variables are set correctly

### Rollback Procedure

If you need to rollback to the old script:

1. **Restore the old script** from backup
2. **Remove the new tool** using the cleanup script:
   ```bash
   python resources/scripts/dataset_tool_cleanup.py --dry-run  # Preview changes
   python resources/scripts/dataset_tool_cleanup.py           # Perform cleanup
   ```
3. **Verify core functionality** still works after cleanup

## Advanced Migration Scenarios

### Automated Migration Script

For organizations with multiple users, create an automated migration script:

```bash
#!/bin/bash
# migrate_dataset_tool.sh

echo "Starting dataset generation tool migration..."

# Backup old script
if [ -f "resources/data-generation/generate_dataset.py" ]; then
    cp resources/data-generation/generate_dataset.py resources/data-generation/generate_dataset_backup.py
    echo "✓ Backed up old script"
fi

# Verify new tool installation
python resources/scripts/dataset_tool_deployment.py --verify-only
if [ $? -ne 0 ]; then
    echo "✗ New tool verification failed"
    exit 1
fi
echo "✓ New tool verified"

# Test with sample vault
if [ -d "test_vault" ]; then
    jarvis generate-dataset --vault test_vault --output test_migration
    if [ $? -eq 0 ]; then
        echo "✓ Migration test successful"
        rm -rf test_migration
    else
        echo "✗ Migration test failed"
        exit 1
    fi
fi

echo "✓ Migration completed successfully"
```

### Batch Migration for Multiple Vaults

If you process multiple vaults, create a batch migration script:

```python
#!/usr/bin/env python3
"""Batch migrate multiple vaults to new dataset generation tool."""

import subprocess
import sys
from pathlib import Path

vaults = [
    "/path/to/vault1",
    "/path/to/vault2", 
    "/path/to/vault3"
]

output_base = Path("./migrated_datasets")
output_base.mkdir(exist_ok=True)

for i, vault_path in enumerate(vaults):
    vault_name = Path(vault_path).name
    output_dir = output_base / vault_name
    
    print(f"Processing vault {i+1}/{len(vaults)}: {vault_name}")
    
    try:
        result = subprocess.run([
            "jarvis", "generate-dataset",
            "--vault", vault_path,
            "--output", str(output_dir),
            "--notes-filename", f"{vault_name}_notes.csv",
            "--pairs-filename", f"{vault_name}_pairs.csv"
        ], check=True, capture_output=True, text=True)
        
        print(f"✓ Successfully processed {vault_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to process {vault_name}: {e}")
        print(f"Error output: {e.stderr}")

print("Batch migration completed")
```

This migration guide provides comprehensive instructions for moving from the old dataset generation script to the new integrated tool while maintaining compatibility and providing troubleshooting support.