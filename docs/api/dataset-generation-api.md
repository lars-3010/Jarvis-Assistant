# Dataset Generation API Reference

This document provides comprehensive API reference for the dataset generation tool components.

## Core Classes

### DatasetGenerator

Main orchestrator class for dataset generation.

```python
from jarvis.tools.dataset_generation import DatasetGenerator
```

#### Constructor

```python
DatasetGenerator(vault_path: Path, output_dir: Path)
```

**Parameters:**
- `vault_path` (Path): Path to the Obsidian vault
- `output_dir` (Path): Directory for output files

**Example:**
```python
from pathlib import Path
generator = DatasetGenerator(
    vault_path=Path("/path/to/vault"),
    output_dir=Path("./datasets")
)
```

#### Methods

##### generate_datasets()

```python
def generate_datasets(
    self,
    notes_filename: str = "notes_dataset.csv",
    pairs_filename: str = "pairs_dataset.csv",
    negative_sampling_ratio: float = 5.0,
    sampling_strategy: str = "stratified",
    batch_size: int = 32,
    max_pairs_per_note: int = 1000,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> DatasetGenerationResult
```

**Parameters:**
- `notes_filename` (str): Output filename for notes dataset
- `pairs_filename` (str): Output filename for pairs dataset
- `negative_sampling_ratio` (float): Ratio of negative to positive examples
- `sampling_strategy` (str): "random" or "stratified"
- `batch_size` (int): Processing batch size
- `max_pairs_per_note` (int): Maximum pairs to generate per note
- `progress_callback` (callable): Optional progress callback function

**Returns:**
- `DatasetGenerationResult`: Result object with success status and metadata

**Example:**
```python
def progress_callback(message: str, step: int, total: int):
    print(f"[{step}/{total}] {message}")

with DatasetGenerator(vault_path, output_dir) as generator:
    result = generator.generate_datasets(
        notes_filename="my_notes.csv",
        pairs_filename="my_pairs.csv",
        negative_sampling_ratio=8.0,
        sampling_strategy="stratified",
        batch_size=64,
        progress_callback=progress_callback
    )
```

##### validate_vault()

```python
def validate_vault(self) -> ValidationResult
```

**Returns:**
- `ValidationResult`: Validation status and any issues found

**Example:**
```python
validation = generator.validate_vault()
if not validation.valid:
    for error in validation.errors:
        print(f"Error: {error}")
```

### LinkExtractor

Extracts links from vault notes and builds knowledge graph.

```python
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
```

#### Constructor

```python
LinkExtractor(vault_reader: VaultReader)
```

**Parameters:**
- `vault_reader` (VaultReader): Initialized vault reader service

#### Methods

##### extract_all_links()

```python
def extract_all_links(self) -> Tuple[nx.DiGraph, LinkStatistics]
```

**Returns:**
- `Tuple[nx.DiGraph, LinkStatistics]`: Knowledge graph and link statistics

**Example:**
```python
from jarvis.services.vault.reader import VaultReader

reader = VaultReader("/path/to/vault")
extractor = LinkExtractor(reader)
graph, stats = extractor.extract_all_links()

print(f"Found {stats.total_links} total links")
print(f"Graph has {graph.number_of_nodes()} nodes")
```

### NotesDatasetGenerator

Generates individual note features dataset.

```python
from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator
```

#### Constructor

```python
NotesDatasetGenerator(
    vault_reader: VaultReader,
    vector_encoder: VectorEncoder,
    graph_database: GraphDatabase,
    markdown_parser: MarkdownParser
)
```

#### Methods

##### generate_dataset()

```python
def generate_dataset(
    self,
    notes: List[str],
    link_graph: nx.DiGraph,
    batch_size: int = 32,
    progress_callback: Optional[Callable] = None
) -> pd.DataFrame
```

**Parameters:**
- `notes` (List[str]): List of note paths to process
- `link_graph` (nx.DiGraph): Knowledge graph from LinkExtractor
- `batch_size` (int): Processing batch size
- `progress_callback` (callable): Optional progress callback

**Returns:**
- `pd.DataFrame`: Notes dataset with features

### PairsDatasetGenerator

Generates note pairs comparison dataset.

```python
from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import PairsDatasetGenerator
```

#### Constructor

```python
PairsDatasetGenerator(
    vector_encoder: VectorEncoder,
    graph_database: GraphDatabase
)
```

#### Methods

##### generate_dataset()

```python
def generate_dataset(
    self,
    notes_data: Dict[str, NoteData],
    link_graph: nx.DiGraph,
    negative_sampling_ratio: float = 5.0,
    sampling_strategy: str = "stratified",
    max_pairs_per_note: int = 1000,
    progress_callback: Optional[Callable] = None
) -> pd.DataFrame
```

**Parameters:**
- `notes_data` (Dict[str, NoteData]): Note data dictionary
- `link_graph` (nx.DiGraph): Knowledge graph
- `negative_sampling_ratio` (float): Negative to positive ratio
- `sampling_strategy` (str): Sampling strategy
- `max_pairs_per_note` (int): Maximum pairs per note
- `progress_callback` (callable): Optional progress callback

**Returns:**
- `pd.DataFrame`: Pairs dataset with features

## Data Models

### DatasetGenerationResult

Result object returned by dataset generation.

```python
@dataclass
class DatasetGenerationResult:
    success: bool
    notes_dataset_path: Optional[Path]
    pairs_dataset_path: Optional[Path]
    summary: GenerationSummary
    error_message: Optional[str] = None
```

**Attributes:**
- `success` (bool): Whether generation succeeded
- `notes_dataset_path` (Path): Path to generated notes dataset
- `pairs_dataset_path` (Path): Path to generated pairs dataset
- `summary` (GenerationSummary): Detailed generation summary
- `error_message` (str): Error message if generation failed

### GenerationSummary

Detailed summary of generation process.

```python
@dataclass
class GenerationSummary:
    total_notes: int
    notes_processed: int
    notes_failed: int
    pairs_generated: int
    positive_pairs: int
    negative_pairs: int
    total_time_seconds: float
    link_statistics: Optional[LinkStatistics]
    validation_result: Optional[ValidationResult]
    performance_metrics: Optional[Dict[str, float]]
```

### NoteData

Complete note information for dataset generation.

```python
@dataclass
class NoteData:
    path: str
    title: str
    content: str
    metadata: Dict[str, Any]
    tags: List[str]
    outgoing_links: List[str]
    embedding: np.ndarray
    quality_stage: Optional[str] = None
    all_frontmatter_properties: Dict[str, Any] = field(default_factory=dict)
    semantic_relationships: Dict[str, List[str]] = field(default_factory=dict)
    progress_indicators: List[str] = field(default_factory=list)
```

### NoteFeatures

Individual note features for dataset.

```python
@dataclass
class NoteFeatures:
    note_path: str
    note_title: str
    word_count: int
    tag_count: int
    quality_stage: str
    creation_date: datetime
    last_modified: datetime
    outgoing_links_count: int
    incoming_links_count: int
    betweenness_centrality: float
    closeness_centrality: float
    pagerank_score: float
    semantic_cluster_id: int
    semantic_summary: str
    
    # Enhanced frontmatter-derived features
    aliases_count: int = 0
    domains_count: int = 0
    concepts_count: int = 0
    sources_count: int = 0
    has_summary_field: bool = False
    progress_state: str = "unknown"
    semantic_up_links: int = 0
    semantic_similar_links: int = 0
    semantic_leads_to_links: int = 0
    semantic_extends_links: int = 0
    semantic_implements_links: int = 0
    
    # Content analysis features
    heading_count: int = 0
    max_heading_depth: int = 0
    technical_term_density: float = 0.0
    concept_density_score: float = 0.0
```

### PairFeatures

Note pair features for dataset.

```python
@dataclass
class PairFeatures:
    note_a_path: str
    note_b_path: str
    cosine_similarity: float
    semantic_cluster_match: bool
    tag_overlap_count: int
    tag_jaccard_similarity: float
    vault_path_distance: int
    shortest_path_length: int
    common_neighbors_count: int
    adamic_adar_score: float
    word_count_ratio: float
    creation_time_diff_days: float
    quality_stage_compatibility: int
    source_centrality: float
    target_centrality: float
    clustering_coefficient: float
    link_exists: bool
```

### LinkStatistics

Statistics about extracted links.

```python
@dataclass
class LinkStatistics:
    total_links: int
    unique_links: int
    broken_links: int
    self_links: int
    bidirectional_links: int
    link_types: Dict[str, int]
```

### ValidationResult

Validation result with errors and warnings.

```python
@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

## Utility Classes

### MemoryMonitor

Monitors memory usage during processing.

```python
from jarvis.tools.dataset_generation.utils.memory_monitor import MemoryMonitor
```

#### Constructor

```python
MemoryMonitor(threshold_mb: int = 1000)
```

#### Methods

```python
def current_usage(self) -> int
def should_pause(self) -> bool
def force_cleanup(self) -> None
```

### ProgressTracker

Tracks progress of long-running operations.

```python
from jarvis.tools.dataset_generation.utils.progress_tracker import ProgressTracker
```

#### Constructor

```python
ProgressTracker(total_steps: int, callback: Optional[Callable] = None)
```

#### Methods

```python
def update(self, step: int, message: str) -> None
def complete(self) -> None
```

### ParallelProcessor

Handles parallel processing of tasks.

```python
from jarvis.tools.dataset_generation.utils.parallel_processor import ParallelProcessor
```

#### Constructor

```python
ParallelProcessor(max_workers: int = 4)
```

#### Methods

```python
def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]
```

## Exception Classes

### DatasetGenerationError

Base exception for dataset generation errors.

```python
class DatasetGenerationError(JarvisError):
    """Base exception for dataset generation errors."""
    pass
```

### LinkExtractionError

Errors during link extraction.

```python
class LinkExtractionError(DatasetGenerationError):
    """Errors during link extraction process."""
    pass
```

### FeatureEngineeringError

Errors during feature computation.

```python
class FeatureEngineeringError(DatasetGenerationError):
    """Errors during feature computation."""
    pass
```

### DataQualityError

Data quality validation failures.

```python
class DataQualityError(DatasetGenerationError):
    """Data quality validation failures."""
    pass
```

## Configuration

### Environment Variables

The tool respects these environment variables:

- `JARVIS_VAULT_PATH`: Default vault path
- `JARVIS_DATASET_OUTPUT_DIR`: Default output directory
- `JARVIS_EMBEDDING_MODEL_NAME`: Embedding model name
- `JARVIS_EMBEDDING_BATCH_SIZE`: Embedding batch size
- `JARVIS_EMBEDDING_DEVICE`: Device for embeddings (cpu/cuda/mps)

### Settings Integration

The tool integrates with Jarvis settings:

```python
from jarvis.utils.config import get_settings

settings = get_settings()
dataset_output_path = settings.get_dataset_output_path()
```

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from jarvis.tools.dataset_generation import DatasetGenerator

vault_path = Path("/path/to/vault")
output_dir = Path("./datasets")

with DatasetGenerator(vault_path, output_dir) as generator:
    result = generator.generate_datasets()
    
    if result.success:
        print(f"Generated datasets:")
        print(f"  Notes: {result.notes_dataset_path}")
        print(f"  Pairs: {result.pairs_dataset_path}")
        print(f"  Total notes: {result.summary.total_notes}")
        print(f"  Total pairs: {result.summary.pairs_generated}")
    else:
        print(f"Generation failed: {result.error_message}")
```

### Advanced Usage with Custom Settings

```python
from pathlib import Path
from jarvis.tools.dataset_generation import DatasetGenerator

def custom_progress_callback(message: str, step: int, total: int):
    percentage = (step / total) * 100
    print(f"[{percentage:.1f}%] {message}")

vault_path = Path("/path/to/vault")
output_dir = Path("./custom_datasets")

with DatasetGenerator(vault_path, output_dir) as generator:
    # Validate vault first
    validation = generator.validate_vault()
    if not validation.valid:
        print("Vault validation failed:")
        for error in validation.errors:
            print(f"  - {error}")
        return
    
    # Generate with custom settings
    result = generator.generate_datasets(
        notes_filename="research_notes.csv",
        pairs_filename="research_pairs.csv",
        negative_sampling_ratio=10.0,
        sampling_strategy="stratified",
        batch_size=64,
        max_pairs_per_note=2000,
        progress_callback=custom_progress_callback
    )
    
    if result.success:
        # Access detailed statistics
        summary = result.summary
        print(f"Processing time: {summary.total_time_seconds:.2f}s")
        print(f"Notes processed: {summary.notes_processed}/{summary.total_notes}")
        print(f"Positive pairs: {summary.positive_pairs}")
        print(f"Negative pairs: {summary.negative_pairs}")
        
        if summary.link_statistics:
            link_stats = summary.link_statistics
            print(f"Total links found: {link_stats.total_links}")
            print(f"Broken links: {link_stats.broken_links}")
            print(f"Link types: {dict(link_stats.link_types)}")
        
        if summary.performance_metrics:
            perf = summary.performance_metrics
            print(f"Notes/second: {perf.get('notes_per_second', 0):.2f}")
            print(f"Pairs/second: {perf.get('pairs_per_second', 0):.2f}")
```

### Direct Component Usage

```python
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.graph.database import GraphDatabase
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
from jarvis.tools.dataset_generation.generators.notes_dataset_generator import NotesDatasetGenerator

# Initialize services
vault_reader = VaultReader("/path/to/vault")
vector_encoder = VectorEncoder()
graph_database = GraphDatabase(settings)

# Extract links
link_extractor = LinkExtractor(vault_reader)
link_graph, link_stats = link_extractor.extract_all_links()

# Generate notes dataset
notes_generator = NotesDatasetGenerator(
    vault_reader, vector_encoder, graph_database, markdown_parser
)

notes = vault_reader.get_markdown_files()
notes_df = notes_generator.generate_dataset(notes, link_graph)

print(f"Generated notes dataset with {len(notes_df)} rows")
print(f"Columns: {list(notes_df.columns)}")
```

This API reference provides comprehensive documentation for all public classes and methods in the dataset generation tool.