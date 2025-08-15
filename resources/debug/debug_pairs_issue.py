#!/usr/bin/env python3
"""
Debug script to understand the pairs dataset filtering issue.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

# Add the src directory to the path
import sys
sys.path.insert(0, '../../src')

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator

def create_test_vault():
    """Create a test vault with Areas/ content."""
    temp_dir = Path(tempfile.mkdtemp())
    vault_path = temp_dir / "test_vault"
    vault_path.mkdir()
    
    # Create .obsidian directory
    obsidian_dir = vault_path / ".obsidian"
    obsidian_dir.mkdir()
    
    # Create Areas/ folder structure
    areas_dir = vault_path / "Areas"
    areas_dir.mkdir()
    
    areas_cs_dir = areas_dir / "Computer Science"
    areas_cs_dir.mkdir()
    
    # Create Areas/ content (need at least 5 notes for validation)
    areas_notes = [
        (areas_cs_dir / "machine_learning.md", """---
title: Machine Learning
tags: [ml, ai]
---

# Machine Learning

Core concepts in machine learning.

## Connection
Uses [[deep_learning.md]] techniques.
"""),
        (areas_cs_dir / "deep_learning.md", """---
title: Deep Learning
tags: [dl, neural-networks]
---

# Deep Learning

Advanced machine learning using neural networks.

## Connection
Built on [[machine_learning.md]] principles.
"""),
        (areas_cs_dir / "algorithms.md", """---
title: Algorithms
tags: [algorithms, cs]
---

# Algorithms

Fundamental algorithms and data structures.
"""),
        (areas_cs_dir / "data_structures.md", """---
title: Data Structures
tags: [ds, cs]
---

# Data Structures

Core data structures.
"""),
        (areas_cs_dir / "programming.md", """---
title: Programming
tags: [programming, cs]
---

# Programming

Programming concepts.
"""),
    ]
    
    # Create non-Areas/ content (should be excluded)
    journal_dir = vault_path / "Journal"
    journal_dir.mkdir()
    
    non_areas_notes = [
        (journal_dir / "2024-01-01.md", """---
title: Daily Journal
tags: [journal, personal]
---

# Daily Journal Entry

Personal thoughts and reflections.
"""),
    ]
    
    # Write all notes
    for note_path, content in areas_notes + non_areas_notes:
        note_path.write_text(content, encoding='utf-8')
    
    return vault_path

def debug_pairs_generation():
    """Debug the pairs generation issue."""
    print("Creating test vault...")
    vault_path = create_test_vault()
    
    print(f"Test vault created at: {vault_path}")
    
    # Create output directory
    output_dir = Path(tempfile.mkdtemp())
    print(f"Output directory: {output_dir}")
    
    # Create generator with Areas filtering enabled
    print("Creating DatasetGenerator with areas_only=True...")
    generator = DatasetGenerator(
        vault_path=vault_path,
        output_dir=output_dir,
        areas_only=True,
        skip_validation=True
    )
    
    print(f"Generator areas_only: {generator.areas_only}")
    
    # Mock vector encoder to avoid actual embedding generation
    mock_encoder = Mock()
    def encode_documents(documents):
        import numpy as np
        embeddings = []
        for i, doc in enumerate(documents):
            np.random.seed(i)
            embedding = np.random.rand(384)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    mock_encoder.encode_documents.side_effect = encode_documents
    
    # Replace the vector encoders
    generator.notes_generator.vector_encoder = mock_encoder
    generator.pairs_generator.vector_encoder = mock_encoder
    
    print("Starting dataset generation...")
    
    try:
        # Step 1: Get filtered notes
        print("\n=== Step 1: Getting filtered notes ===")
        all_notes = [str(path) for path in generator.vault_reader.get_markdown_files()]
        print(f"Filtered notes count: {len(all_notes)}")
        print(f"Filtered notes: {all_notes}")
        
        # Step 2: Extract links
        print("\n=== Step 2: Extracting links ===")
        link_graph, link_statistics = generator.link_extractor.extract_all_links()
        print(f"Link graph nodes: {list(link_graph.nodes())}")
        print(f"Link graph edges: {list(link_graph.edges())}")
        
        # Step 3: Filter link graph
        print("\n=== Step 3: Filtering link graph ===")
        if generator.areas_only:
            original_nodes = list(link_graph.nodes())
            link_graph = link_graph.subgraph([node for node in link_graph.nodes() if node in all_notes]).copy()
            filtered_nodes = list(link_graph.nodes())
            print(f"Original nodes: {original_nodes}")
            print(f"Filtered nodes: {filtered_nodes}")
            print(f"Nodes removed: {set(original_nodes) - set(filtered_nodes)}")
        
        # Step 4: Load notes data
        print("\n=== Step 4: Loading notes data ===")
        notes_data = generator._load_notes_data_for_pairs(all_notes)
        print(f"Notes data keys: {list(notes_data.keys())}")
        print(f"Notes data count: {len(notes_data)}")
        
        # Step 5: Test pairs generation
        print("\n=== Step 5: Testing pairs generation ===")
        
        # Extract positive pairs
        positive_pairs = generator.pairs_generator._extract_positive_pairs(link_graph)
        print(f"Positive pairs: {positive_pairs}")
        print(f"Positive pairs count: {len(positive_pairs)}")
        
        # Test negative sampling
        print("\n=== Step 6: Testing negative sampling ===")
        all_notes_for_sampling = list(notes_data.keys())
        print(f"All notes for sampling: {all_notes_for_sampling}")
        print(f"All notes for sampling count: {len(all_notes_for_sampling)}")
        
        if len(all_notes_for_sampling) == 0:
            print("ERROR: all_notes_for_sampling is empty!")
            return
        
        # Try to create sampling strategy - test both random and stratified
        from jarvis.tools.dataset_generation.generators.pairs_dataset_generator import RandomSamplingStrategy, StratifiedSamplingStrategy
        
        target_count = 5
        
        # Test Random sampling
        print(f"=== Testing Random Sampling ===")
        random_strategy = RandomSamplingStrategy()
        try:
            negative_pairs = random_strategy.sample_negative_pairs(
                positive_pairs, all_notes_for_sampling, target_count
            )
            print(f"Random: Successfully generated {len(negative_pairs)} negative pairs")
        except Exception as e:
            print(f"Random: ERROR in negative sampling: {e}")
        
        # Test Stratified sampling (this is what the test uses by default)
        print(f"\n=== Testing Stratified Sampling ===")
        try:
            stratified_strategy = StratifiedSamplingStrategy(notes_data)
            print(f"Stratified strategy created successfully")
            print(f"Folder groups: {stratified_strategy._folder_groups}")
            print(f"Tag groups: {stratified_strategy._tag_groups}")
            
            negative_pairs = stratified_strategy.sample_negative_pairs(
                positive_pairs, all_notes_for_sampling, target_count
            )
            print(f"Stratified: Successfully generated {len(negative_pairs)} negative pairs")
        except Exception as e:
            print(f"Stratified: ERROR in negative sampling: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pairs_generation()