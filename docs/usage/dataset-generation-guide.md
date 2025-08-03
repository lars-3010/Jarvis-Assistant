# Dataset Generation Tool User Guide

This guide provides comprehensive instructions for using the dataset generation tool in Jarvis Assistant to create machine learning datasets from Obsidian vaults.

## Overview

The dataset generation tool creates two complementary datasets from your Obsidian vault:

1. **Notes Dataset**: Individual note characteristics and properties
2. **Pairs Dataset**: Comparative analysis between note pairs for link prediction

These datasets are designed for machine learning applications, particularly link prediction and knowledge graph analysis.

## Installation and Setup

### Prerequisites

- Python 3.11 or higher
- Jarvis Assistant properly installed
- Required Python packages (automatically installed with Jarvis)
- Access to an Obsidian vault

### Verification

Verify the tool is properly installed:

```bash
# Check if the command is available
jarvis generate-dataset --help

# Verify installation status
python resources/scripts/dataset_tool_deployment.py --status
```

If the tool is not installed, run the deployment script:

```bash
python resources/scripts/dataset_tool_deployment.py
```

## Basic Usage

### Simple Dataset Generation

Generate datasets from your vault with default settings:

```bash
jarvis generate-dataset --vault /path/to/your/vault
```

This creates two files in the default output directory (`./datasets`):
- `notes_dataset.csv`: Individual note properties
- `pairs_dataset.csv`: Note pair comparisons

### Specifying Output Location

```bash
jarvis generate-dataset \
    --vault /path/to/your/vault \
    --output /path/to/output/directory \
    --notes-filename my_notes.csv \
    --pairs-filename my_pairs.csv
```

## Configuration Options

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--vault` | Path to Obsidian vault | From config |
| `--output` | Output directory | `./datasets` |
| `--notes-filename` | Notes dataset filename | `notes_dataset.csv` |
| `--pairs-filename` | Pairs dataset filename | `pairs_dataset.csv` |
| `--negative-ratio` | Negative to positive pairs ratio | `5.0` |
| `--sampling` | Sampling strategy (`random`, `stratified`) | `stratified` |
| `--batch-size` | Processing batch size | `32` |
| `--max-pairs` | Maximum pairs per note | `1000` |

### Environment Configuration

Set default values using environment variables:

```bash
export JARVIS_VAULT_PATH="/path/to/your/vault"
export JARVIS_DATASET_OUTPUT_DIR="/path/to/datasets"
export JARVIS_EMBEDDING_MODEL_NAME="paraphrase-MiniLM-L6-v2"
export JARVIS_EMBEDDING_BATCH_SIZE=32
```

Or create a `.env` file:

```env
JARVIS_VAULT_PATH=/path/to/your/vault
JARVIS_DATASET_OUTPUT_DIR=./datasets
JARVIS_EMBEDDING_MODEL_NAME=paraphrase-MiniLM-L6-v2
JARVIS_EMBEDDING_BATCH_SIZE=32
```

## Dataset Schemas

### Notes Dataset Schema

The notes dataset contains individual note characteristics:

| Column | Type | Description |
|--------|------|-------------|
| `note_path` | string | Relative path to note in vault |
| `note_title` | string | Note title (filename without extension) |
| `word_count` | integer | Number of words in note content |
| `tag_count` | integer | Number of tags in note |
| `quality_stage` | string | Quality indicator (🌱🌿🌲⚛️🗺️⧉🎓) |
| `creation_date` | datetime | File creation timestamp |
| `last_modified` | datetime | Last modification timestamp |
| `outgoing_links_count` | integer | Number of outbound links |
| `incoming_links_count` | integer | Number of inbound links |
| `betweenness_centrality` | float | Graph centrality measure |
| `closeness_centrality` | float | Graph centrality measure |
| `pagerank_score` | float | PageRank score in link graph |
| `semantic_cluster_id` | integer | Semantic cluster assignment |
| `semantic_summary` | string | AI-generated or extracted summary |
| `aliases_count` | integer | Number of note aliases |
| `domains_count` | integer | Number of knowledge domains |
| `concepts_count` | integer | Number of key concepts |
| `has_summary_field` | boolean | Whether note has frontmatter summary |
| `progress_state` | string | Progress indicator from tags/frontmatter |
| `heading_count` | integer | Number of headings in note |
| `technical_term_density` | float | Density of technical terminology |

### Pairs Dataset Schema

The pairs dataset contains comparative features between note pairs:

| Column | Type | Description |
|--------|------|-------------|
| `note_a_path` | string | Path to first note |
| `note_b_path` | string | Path to second note |
| `cosine_similarity` | float | Semantic similarity (0-1) |
| `semantic_cluster_match` | boolean | Whether notes are in same cluster |
| `tag_overlap_count` | integer | Number of shared tags |
| `tag_jaccard_similarity` | float | Jaccard similarity of tag sets |
| `vault_path_distance` | integer | Folder hierarchy distance |
| `shortest_path_length` | integer | Graph distance (-1 if no path) |
| `common_neighbors_count` | integer | Number of shared linked notes |
| `adamic_adar_score` | float | Adamic-Adar link prediction score |
| `word_count_ratio` | float | Ratio of word counts |
| `creation_time_diff_days` | float | Difference in creation time (days) |
| `quality_stage_compatibility` | integer | Quality stage compatibility score |
| `source_centrality` | float | Centrality of source note |
| `target_centrality` | float | Centrality of target note |
| `clustering_coefficient` | float | Local clustering coefficient |
| `link_exists` | boolean | **Target variable**: Whether notes are linked |

## Advanced Usage

### Performance Optimization

For large vaults, optimize performance with these settings:

```bash
# Increase batch size for more memory/faster processing
jarvis generate-dataset \
    --vault /path/to/vault \
    --batch-size 64 \
    --negative-ratio 3.0 \
    --max-pairs 500
```

### Sampling Strategies

Control dataset size and quality with sampling options:

```bash
# Stratified sampling for balanced datasets
jarvis generate-dataset \
    --vault /path/to/vault \
    --sampling stratified \
    --negative-ratio 10.0

# Random sampling for speed
jarvis generate-dataset \
    --vault /path/to/vault \
    --sampling random \
    --negative-ratio 5.0
```

### Custom Processing

For advanced use cases, use the tool programmatically:

```python
from jarvis.tools.dataset_generation import DatasetGenerator
from pathlib import Path

vault_path = Path("/path/to/vault")
output_dir = Path("./custom_datasets")

# Custom progress callback
def progress_callback(message: str, step: int, total: int):
    print(f"[{step}/{total}] {message}")

# Generate with custom settings
with DatasetGenerator(vault_path, output_dir) as generator:
    result = generator.generate_datasets(
        notes_filename="custom_notes.csv",
        pairs_filename="custom_pairs.csv",
        negative_sampling_ratio=8.0,
        sampling_strategy="stratified",
        batch_size=64,
        max_pairs_per_note=2000,
        progress_callback=progress_callback
    )
    
    if result.success:
        print(f"Generated {result.summary.total_notes} notes")
        print(f"Generated {result.summary.pairs_generated} pairs")
    else:
        print(f"Generation failed: {result.error_message}")
```

## Use Cases and Examples

### Link Prediction Model Training

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the pairs dataset
pairs_df = pd.read_csv("pairs_dataset.csv")

# Prepare features and target
feature_columns = [
    'cosine_similarity', 'tag_overlap_count', 'tag_jaccard_similarity',
    'shortest_path_length', 'common_neighbors_count', 'adamic_adar_score',
    'word_count_ratio', 'creation_time_diff_days'
]

X = pairs_df[feature_columns].fillna(0)
y = pairs_df['link_exists']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Knowledge Graph Analysis

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load notes dataset
notes_df = pd.read_csv("notes_dataset.csv")

# Analyze note quality distribution
quality_dist = notes_df['quality_stage'].value_counts()
print("Quality Distribution:")
print(quality_dist)

# Find high-centrality notes
top_central = notes_df.nlargest(10, 'betweenness_centrality')
print("\nMost Central Notes:")
print(top_central[['note_title', 'betweenness_centrality', 'outgoing_links_count']])

# Analyze semantic clusters
cluster_sizes = notes_df['semantic_cluster_id'].value_counts()
print(f"\nFound {len(cluster_sizes)} semantic clusters")
print(f"Largest cluster: {cluster_sizes.iloc[0]} notes")
```

### Content Quality Assessment

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load notes dataset
notes_df = pd.read_csv("notes_dataset.csv")

# Analyze quality factors
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Word count distribution by quality
sns.boxplot(data=notes_df, x='quality_stage', y='word_count', ax=axes[0,0])
axes[0,0].set_title('Word Count by Quality Stage')

# Link count distribution by quality
sns.boxplot(data=notes_df, x='quality_stage', y='outgoing_links_count', ax=axes[0,1])
axes[0,1].set_title('Outgoing Links by Quality Stage')

# Centrality by quality
sns.boxplot(data=notes_df, x='quality_stage', y='betweenness_centrality', ax=axes[1,0])
axes[1,0].set_title('Centrality by Quality Stage')

# Technical term density by quality
sns.boxplot(data=notes_df, x='quality_stage', y='technical_term_density', ax=axes[1,1])
axes[1,1].set_title('Technical Term Density by Quality Stage')

plt.tight_layout()
plt.show()
```

## Troubleshooting

### Common Issues

#### Issue: "No notes found to process"
**Cause**: Vault path is incorrect or vault is empty
**Solution**: 
```bash
# Verify vault path
ls -la /path/to/your/vault

# Check for .md files
find /path/to/your/vault -name "*.md" | head -5
```

#### Issue: "Memory error during processing"
**Cause**: Vault is too large for available memory
**Solution**: Reduce batch size and limit pairs
```bash
jarvis generate-dataset \
    --vault /path/to/vault \
    --batch-size 16 \
    --max-pairs 500 \
    --negative-ratio 3.0
```

#### Issue: "Link extraction returns 0 positive links"
**Cause**: Notes don't contain recognizable link formats
**Solution**: Check your note linking format
```bash
# Look for different link patterns in your notes
grep -r "\[\[" /path/to/vault | head -5
grep -r "\](" /path/to/vault | head -5
```

#### Issue: "Embedding generation fails"
**Cause**: Model download or GPU/CPU issues
**Solution**: Check embedding configuration
```bash
# Test embedding generation
python -c "
from jarvis.services.vector.encoder import VectorEncoder
encoder = VectorEncoder()
result = encoder.encode_documents(['test document'])
print('Embedding shape:', result.shape)
"
```

### Performance Issues

#### Slow Processing
1. **Increase batch size** (if you have enough memory):
   ```bash
   --batch-size 64
   ```

2. **Reduce negative sampling ratio**:
   ```bash
   --negative-ratio 3.0
   ```

3. **Limit pairs per note**:
   ```bash
   --max-pairs 500
   ```

#### High Memory Usage
1. **Decrease batch size**:
   ```bash
   --batch-size 16
   ```

2. **Use random sampling** (faster than stratified):
   ```bash
   --sampling random
   ```

### Data Quality Issues

#### Low Link Prediction Accuracy
1. **Check link extraction quality**: Look at the link statistics in the output
2. **Adjust negative sampling ratio**: Try different ratios (3.0, 5.0, 10.0)
3. **Verify vault structure**: Ensure notes are properly linked

#### Missing Features
1. **Check frontmatter parsing**: Ensure YAML frontmatter is properly formatted
2. **Verify embedding generation**: Test with a small subset first
3. **Check graph connectivity**: Ensure notes are actually linked

## Best Practices

### Vault Preparation
1. **Clean up broken links** before generation
2. **Ensure consistent frontmatter** format
3. **Use meaningful note titles** and file names
4. **Organize notes** in a logical folder structure

### Dataset Generation
1. **Start with small vaults** to test settings
2. **Use stratified sampling** for balanced datasets
3. **Monitor memory usage** during generation
4. **Validate results** with known link patterns

### Model Training
1. **Handle missing values** appropriately
2. **Scale numerical features** if needed
3. **Use cross-validation** for robust evaluation
4. **Consider class imbalance** in link prediction

## API Reference

For programmatic usage, see the full API documentation:

```python
from jarvis.tools.dataset_generation import DatasetGenerator
from jarvis.tools.dataset_generation.models import DatasetGenerationResult

# See docstrings for detailed parameter descriptions
help(DatasetGenerator.generate_datasets)
```

## Support and Feedback

If you encounter issues or have suggestions:

1. **Check the logs** for detailed error messages
2. **Run verification scripts** to diagnose problems
3. **Test with smaller datasets** to isolate issues
4. **Review the migration guide** if upgrading from old scripts

The dataset generation tool is designed to be robust and handle various vault structures, but optimal results depend on well-structured vaults with consistent linking patterns.