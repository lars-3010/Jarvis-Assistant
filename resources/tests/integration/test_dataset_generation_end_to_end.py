"""
End-to-end testing with real data for dataset generation.

This test suite performs comprehensive end-to-end testing with actual Obsidian vaults,
validates link extraction accuracy against manual verification, and tests performance
and memory usage with realistic data volumes.

Requirements tested:
- 7.5: Validate extracted links against actual vault files
- 6.6: Test performance and memory usage with realistic data volumes
"""

import gc
import os
import tempfile
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple
from unittest.mock import Mock, patch
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from jarvis.tools.dataset_generation.dataset_generator import DatasetGenerator
from jarvis.tools.dataset_generation.extractors.link_extractor import LinkExtractor
from jarvis.tools.dataset_generation.models.data_models import LinkStatistics
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vault.parser import MarkdownParser


class RealVaultTestHelper:
    """Helper class for creating realistic test vaults and validating results."""
    
    @staticmethod
    def create_realistic_obsidian_vault(vault_path: Path, complexity: str = "medium") -> Dict[str, any]:
        """Create a realistic Obsidian vault with proper structure and content.
        
        Args:
            vault_path: Path to create vault
            complexity: "small", "medium", or "large" vault complexity
            
        Returns:
            Dictionary with vault metadata for validation
        """
        vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create .obsidian directory structure
        obsidian_dir = vault_path / ".obsidian"
        obsidian_dir.mkdir(exist_ok=True)
        
        # Create realistic folder structure
        folders = [
            "00-Inbox",
            "01-Projects", 
            "02-Areas",
            "03-Resources",
            "04-Archive",
            "Daily Notes",
            "Templates",
            "Attachments"
        ]
        
        for folder in folders:
            (vault_path / folder).mkdir(exist_ok=True)
        
        # Define complexity parameters
        complexity_params = {
            "small": {"notes": 25, "max_links_per_note": 3, "avg_content_size": 300},
            "medium": {"notes": 100, "max_links_per_note": 8, "avg_content_size": 800},
            "large": {"notes": 500, "max_links_per_note": 15, "avg_content_size": 1200}
        }
        
        params = complexity_params.get(complexity, complexity_params["medium"])
        
        # Create realistic note content with proper frontmatter and links
        notes_metadata = {}
        note_titles = []
        
        # Generate note titles first for realistic linking
        domains = [
            "Machine Learning", "Data Science", "Software Engineering", "Product Management",
            "Research Methods", "Statistics", "Python Programming", "Web Development",
            "System Design", "Database Design", "User Experience", "Project Planning"
        ]
        
        for i in range(params["notes"]):
            domain = np.random.choice(domains)
            note_num = i + 1
            title = f"{domain} - Topic {note_num:03d}"
            note_titles.append(title)
        
        # Create notes with realistic content and cross-references
        for i, title in enumerate(note_titles):
            note_filename = f"{title.replace(' ', '_').replace('-', '').lower()}.md"
            note_path = vault_path / RealVaultTestHelper._get_folder_for_note(i, folders) / note_filename
            note_path.parent.mkdir(exist_ok=True)
            
            # Generate realistic frontmatter
            creation_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            modified_date = creation_date + timedelta(days=np.random.randint(0, 30))
            
            tags = RealVaultTestHelper._generate_realistic_tags(title, i)
            aliases = RealVaultTestHelper._generate_aliases(title)
            
            frontmatter = {
                "title": title,
                "created": creation_date.isoformat(),
                "modified": modified_date.isoformat(),
                "tags": tags,
                "aliases": aliases,
                "status": np.random.choice(["🌱", "🌿", "🌲", "🗺️"], p=[0.3, 0.4, 0.25, 0.05])
            }
            
            # Add domain-specific frontmatter
            if "Machine Learning" in title:
                frontmatter["domains"] = ["artificial-intelligence", "data-science"]
                frontmatter["concepts"] = ["supervised-learning", "neural-networks", "algorithms"]
            elif "Software Engineering" in title:
                frontmatter["domains"] = ["software-development", "engineering"]
                frontmatter["concepts"] = ["design-patterns", "architecture", "testing"]
            
            # Generate realistic content with proper links
            content_parts = []
            content_parts.append(RealVaultTestHelper._generate_yaml_frontmatter(frontmatter))
            content_parts.append(f"# {title}\n")
            
            # Add introduction
            content_parts.append(RealVaultTestHelper._generate_introduction(title))
            
            # Add main content sections
            content_parts.extend(RealVaultTestHelper._generate_content_sections(title, params["avg_content_size"]))
            
            # Add realistic links to other notes
            links = RealVaultTestHelper._generate_realistic_links(i, note_titles, params["max_links_per_note"])
            if links:
                content_parts.append("\n## Related Notes\n")
                for link in links:
                    content_parts.append(f"- {link}\n")
            
            # Add references section
            content_parts.append(RealVaultTestHelper._generate_references_section())
            
            full_content = "\n".join(content_parts)
            note_path.write_text(full_content, encoding='utf-8')
            
            # Store metadata for validation
            notes_metadata[str(note_path)] = {
                "title": title,
                "filename": note_filename,
                "folder": str(note_path.parent.relative_to(vault_path)),
                "frontmatter": frontmatter,
                "expected_links": links,
                "content_size": len(full_content),
                "word_count": len(full_content.split())
            }
        
        # Create some template files
        templates_dir = vault_path / "Templates"
        (templates_dir / "Daily Note Template.md").write_text("""# {{date}}

## Tasks
- [ ] 

## Notes


## References
""")
        
        # Create a few daily notes
        daily_notes_dir = vault_path / "Daily Notes"
        for i in range(5):
            date = datetime.now() - timedelta(days=i)
            daily_note = daily_notes_dir / f"{date.strftime('%Y-%m-%d')}.md"
            daily_note.write_text(f"""# {date.strftime('%Y-%m-%d')}

## Daily Reflection
Today I worked on various projects and made progress on understanding key concepts.

## Key Insights
- Important insight about [[{note_titles[i % len(note_titles)]}]]
- Connection between different concepts

## Tomorrow's Focus
- Continue research
- Review notes
""")
        
        return {
            "vault_path": str(vault_path),
            "complexity": complexity,
            "total_notes": len(notes_metadata) + 5,  # +5 for daily notes
            "notes_metadata": notes_metadata,
            "expected_folders": folders,
            "creation_date": datetime.now().isoformat()
        }
    
    @staticmethod
    def _get_folder_for_note(index: int, folders: List[str]) -> str:
        """Distribute notes across folders realistically."""
        if index < 5:
            return folders[0]  # Inbox
        elif index < 20:
            return folders[1]  # Projects
        elif index < 40:
            return folders[2]  # Areas
        else:
            return folders[3]  # Resources
    
    @staticmethod
    def _generate_realistic_tags(title: str, index: int) -> List[str]:
        """Generate realistic tags based on content."""
        base_tags = []
        
        if "Machine Learning" in title:
            base_tags.extend(["ml", "ai", "data-science"])
        elif "Software Engineering" in title:
            base_tags.extend(["software", "engineering", "development"])
        elif "Data Science" in title:
            base_tags.extend(["data", "analytics", "statistics"])
        elif "Product Management" in title:
            base_tags.extend(["product", "management", "strategy"])
        
        # Add status tags
        if index % 10 == 0:
            base_tags.append("important")
        if index % 15 == 0:
            base_tags.append("review")
        
        return base_tags[:5]  # Limit to 5 tags
    
    @staticmethod
    def _generate_aliases(title: str) -> List[str]:
        """Generate realistic aliases for notes."""
        aliases = []
        
        # Create abbreviated version
        words = title.split()
        if len(words) > 2:
            aliases.append(" ".join(words[:2]))
        
        # Create acronym for longer titles
        if len(words) >= 3:
            acronym = "".join(word[0].upper() for word in words if word[0].isupper())
            if len(acronym) >= 2:
                aliases.append(acronym)
        
        return aliases[:3]  # Limit to 3 aliases
    
    @staticmethod
    def _generate_yaml_frontmatter(frontmatter: Dict) -> str:
        """Generate YAML frontmatter string."""
        lines = ["---"]
        
        for key, value in frontmatter.items():
            if isinstance(value, list):
                if value:  # Only add non-empty lists
                    lines.append(f"{key}:")
                    for item in value:
                        lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        
        lines.append("---")
        return "\n".join(lines)
    
    @staticmethod
    def _generate_introduction(title: str) -> str:
        """Generate realistic introduction based on title."""
        domain = title.split(" - ")[0] if " - " in title else title
        
        intros = {
            "Machine Learning": f"This note explores key concepts in {domain.lower()}, focusing on practical applications and theoretical foundations.",
            "Data Science": f"An overview of {domain.lower()} methodologies and their implementation in real-world scenarios.",
            "Software Engineering": f"This document covers {domain.lower()} principles and best practices for building robust systems.",
            "Product Management": f"Key insights into {domain.lower()} strategies and frameworks for successful product development."
        }
        
        default_intro = f"This note provides an in-depth analysis of {title.lower()} and its applications."
        
        for key, intro in intros.items():
            if key in title:
                return intro + "\n"
        
        return default_intro + "\n"
    
    @staticmethod
    def _generate_content_sections(title: str, target_size: int) -> List[str]:
        """Generate realistic content sections."""
        sections = []
        current_size = 0
        
        # Common section templates
        section_templates = [
            ("## Overview", "This section provides a comprehensive overview of the key concepts and principles."),
            ("## Key Concepts", "The following concepts are fundamental to understanding this topic:"),
            ("## Implementation", "Practical implementation details and considerations:"),
            ("## Best Practices", "Industry best practices and recommendations:"),
            ("## Common Pitfalls", "Common mistakes to avoid and how to prevent them:"),
            ("## Examples", "Real-world examples and case studies:"),
            ("## Tools and Resources", "Useful tools and additional resources:")
        ]
        
        # Add sections until we reach target size
        for section_title, section_intro in section_templates:
            if current_size >= target_size:
                break
            
            sections.append(f"\n{section_title}\n")
            sections.append(f"{section_intro}\n")
            
            # Add bullet points or numbered lists
            for i in range(np.random.randint(2, 6)):
                point = f"- Important point {i+1} about {title.lower().split()[-1]}"
                sections.append(f"{point}\n")
                current_size += len(point)
            
            sections.append("")  # Empty line
            current_size += len(section_title) + len(section_intro) + 20
        
        return sections
    
    @staticmethod
    def _generate_realistic_links(current_index: int, all_titles: List[str], max_links: int) -> List[str]:
        """Generate realistic links to other notes."""
        links = []
        num_links = np.random.randint(1, min(max_links + 1, len(all_titles)))
        
        # Create links to related notes (prefer nearby indices and similar domains)
        potential_targets = []
        
        for i, title in enumerate(all_titles):
            if i == current_index:
                continue
            
            # Higher probability for notes in similar domain
            current_domain = all_titles[current_index].split(" - ")[0]
            target_domain = title.split(" - ")[0]
            
            if current_domain == target_domain:
                potential_targets.extend([title] * 3)  # Higher weight
            else:
                potential_targets.append(title)
        
        # Select random links
        if potential_targets:
            selected_titles = np.random.choice(
                potential_targets, 
                size=min(num_links, len(set(potential_targets))), 
                replace=False
            )
            
            for title in selected_titles:
                # Create different link formats
                link_format = np.random.choice([
                    f"[[{title}]]",
                    f"[[{title}|{title.split(' - ')[-1]}]]",
                    f"See also: [[{title}]]"
                ])
                links.append(link_format)
        
        return links
    
    @staticmethod
    def _generate_references_section() -> str:
        """Generate a references section."""
        return """
## References

1. Academic paper or book reference
2. Online resource or documentation
3. Related research or case study
"""
    
    @staticmethod
    def validate_link_extraction_accuracy(vault_path: Path, extracted_graph: nx.DiGraph, 
                                        notes_metadata: Dict) -> Dict[str, any]:
        """Validate link extraction accuracy against expected links.
        
        Args:
            vault_path: Path to the test vault
            extracted_graph: Graph extracted by LinkExtractor
            notes_metadata: Expected metadata from vault creation
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_expected_links": 0,
            "total_extracted_links": extracted_graph.number_of_edges(),
            "correctly_extracted": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "detailed_errors": []
        }
        
        # Build expected links set
        expected_links = set()
        for note_path, metadata in notes_metadata.items():
            source_path = Path(note_path)
            
            for link_text in metadata.get("expected_links", []):
                # Parse link to extract target
                target_title = RealVaultTestHelper._extract_link_target(link_text)
                if target_title:
                    # Find corresponding file path
                    target_path = RealVaultTestHelper._find_note_by_title(
                        target_title, notes_metadata, vault_path
                    )
                    if target_path:
                        expected_links.add((str(source_path), str(target_path)))
        
        validation_results["total_expected_links"] = len(expected_links)
        
        # Build extracted links set
        extracted_links = set()
        for source, target in extracted_graph.edges():
            extracted_links.add((str(source), str(target)))
        
        # Calculate accuracy metrics
        correctly_extracted = expected_links.intersection(extracted_links)
        false_positives = extracted_links - expected_links
        false_negatives = expected_links - extracted_links
        
        validation_results["correctly_extracted"] = len(correctly_extracted)
        validation_results["false_positives"] = len(false_positives)
        validation_results["false_negatives"] = len(false_negatives)
        
        # Calculate metrics
        if len(expected_links) > 0:
            validation_results["recall"] = len(correctly_extracted) / len(expected_links)
        
        if len(extracted_links) > 0:
            validation_results["precision"] = len(correctly_extracted) / len(extracted_links)
        
        if len(expected_links) + len(false_positives) > 0:
            validation_results["accuracy"] = len(correctly_extracted) / (len(expected_links) + len(false_positives))
        
        # Record detailed errors for analysis
        for source, target in false_negatives:
            validation_results["detailed_errors"].append({
                "type": "false_negative",
                "source": source,
                "target": target,
                "description": f"Expected link from {Path(source).name} to {Path(target).name} not found"
            })
        
        for source, target in false_positives:
            validation_results["detailed_errors"].append({
                "type": "false_positive", 
                "source": source,
                "target": target,
                "description": f"Unexpected link from {Path(source).name} to {Path(target).name}"
            })
        
        return validation_results
    
    @staticmethod
    def _extract_link_target(link_text: str) -> str:
        """Extract target from link text."""
        # Handle different link formats
        if link_text.startswith("[[") and link_text.endswith("]]"):
            inner = link_text[2:-2]
            if "|" in inner:
                return inner.split("|")[0].strip()
            return inner.strip()
        elif "[[" in link_text and "]]" in link_text:
            start = link_text.find("[[") + 2
            end = link_text.find("]]", start)
            inner = link_text[start:end]
            if "|" in inner:
                return inner.split("|")[0].strip()
            return inner.strip()
        return None
    
    @staticmethod
    def _find_note_by_title(title: str, notes_metadata: Dict, vault_path: Path) -> str:
        """Find note path by title."""
        for note_path, metadata in notes_metadata.items():
            if metadata["title"] == title:
                return note_path
            if title in metadata.get("aliases", []):
                return note_path
        return None


class TestDatasetGenerationEndToEnd:
    """Comprehensive end-to-end tests with real data."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_vector_encoder(self):
        """Create mock vector encoder for consistent testing."""
        mock_encoder = Mock()
        
        def encode_documents(documents):
            # Create deterministic embeddings based on content
            embeddings = []
            for doc in documents:
                # Use hash for deterministic but varied embeddings
                seed = hash(doc[:100]) % 10000  # Use first 100 chars
                np.random.seed(seed)
                embedding = np.random.rand(384)
                embeddings.append(embedding)
            return np.array(embeddings)
        
        mock_encoder.encode_documents.side_effect = encode_documents
        mock_encoder.encode_batch.side_effect = encode_documents
        mock_encoder.get_embedding_dimension.return_value = 384
        
        return mock_encoder
    
    @pytest.mark.parametrize("vault_complexity", ["small", "medium"])
    def test_end_to_end_with_realistic_vault(self, vault_complexity, temp_output_dir, mock_vector_encoder):
        """Test complete end-to-end dataset generation with realistic Obsidian vault."""
        # Create realistic test vault
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / f"realistic_vault_{vault_complexity}"
        
        try:
            # Create realistic vault
            vault_metadata = RealVaultTestHelper.create_realistic_obsidian_vault(
                vault_path, complexity=vault_complexity
            )
            
            print(f"\nTesting {vault_complexity} vault with {vault_metadata['total_notes']} notes")
            
            # Create dataset generator with Areas/ filtering disabled for full vault testing
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir,
                areas_only=False  # Disable Areas/ filtering for comprehensive testing
            )
            
            # Mock vector encoder for consistent results
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Measure performance
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                # Generate datasets
                result = generator.generate_datasets(
                    notes_filename=f"{vault_complexity}_notes.csv",
                    pairs_filename=f"{vault_complexity}_pairs.csv",
                    batch_size=15,
                    negative_sampling_ratio=5.0
                )
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                # Verify success
                assert result.success is True, f"Dataset generation failed: {result.error_message}"
                
                # Performance validation
                total_time = end_time - start_time
                memory_increase = end_memory - start_memory
                
                print(f"Performance metrics:")
                print(f"  Total time: {total_time:.2f} seconds")
                print(f"  Memory increase: {memory_increase:.2f} MB")
                print(f"  Processing rate: {vault_metadata['total_notes'] / total_time:.2f} notes/second")
                
                # Performance requirements based on complexity
                max_time = {"small": 60, "medium": 180, "large": 300}[vault_complexity]
                max_memory = {"small": 200, "medium": 500, "large": 1000}[vault_complexity]
                
                assert total_time <= max_time, f"Processing too slow: {total_time:.2f}s > {max_time}s"
                assert memory_increase <= max_memory, f"Memory usage too high: {memory_increase:.2f}MB > {max_memory}MB"
                
                # Validate output files
                notes_path = temp_output_dir / f"{vault_complexity}_notes.csv"
                pairs_path = temp_output_dir / f"{vault_complexity}_pairs.csv"
                
                assert notes_path.exists(), "Notes dataset file not created"
                assert pairs_path.exists(), "Pairs dataset file not created"
                
                # Load and validate datasets
                notes_df = pd.read_csv(notes_path)
                pairs_df = pd.read_csv(pairs_path)
                
                # Basic structure validation
                assert len(notes_df) > 0, "Notes dataset is empty"
                assert len(pairs_df) > 0, "Pairs dataset is empty"
                
                # Validate notes dataset structure
                required_note_columns = [
                    'note_path', 'note_title', 'word_count', 'tag_count',
                    'outgoing_links_count', 'semantic_summary'
                ]
                for col in required_note_columns:
                    assert col in notes_df.columns, f"Missing column in notes dataset: {col}"
                
                # Validate pairs dataset structure
                required_pair_columns = [
                    'note_a_path', 'note_b_path', 'cosine_similarity', 'link_exists'
                ]
                for col in required_pair_columns:
                    assert col in pairs_df.columns, f"Missing column in pairs dataset: {col}"
                
                # Data quality validation
                assert notes_df['word_count'].min() >= 0, "Invalid word counts"
                assert notes_df['tag_count'].min() >= 0, "Invalid tag counts"
                assert pairs_df['cosine_similarity'].between(0, 1).all(), "Invalid similarity scores"
                
                # Validate data consistency
                note_paths_in_notes = set(notes_df['note_path'])
                note_paths_in_pairs = set(pairs_df['note_a_path']) | set(pairs_df['note_b_path'])
                
                missing_refs = note_paths_in_pairs - note_paths_in_notes
                assert len(missing_refs) == 0, f"Pairs reference missing notes: {missing_refs}"
                
                # Validate summary statistics
                summary = result.summary
                assert summary.total_notes > 0, "Summary should report total notes"
                assert summary.notes_processed > 0, "Summary should report processed notes"
                assert summary.pairs_generated > 0, "Summary should report generated pairs"
                
                print(f"Dataset validation successful:")
                print(f"  Notes processed: {summary.notes_processed}")
                print(f"  Pairs generated: {summary.pairs_generated}")
                print(f"  Processing rate: {summary.processing_rate:.2f} notes/second")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_link_extraction_accuracy_validation(self, temp_output_dir):
        """Test link extraction accuracy against manual verification."""
        # Create test vault with known link structure
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "link_accuracy_test"
        
        try:
            # Create vault with precisely known links for validation
            vault_metadata = RealVaultTestHelper.create_realistic_obsidian_vault(
                vault_path, complexity="small"
            )
            
            # Extract links using our LinkExtractor
            vault_reader = VaultReader(str(vault_path))
            link_extractor = LinkExtractor(vault_reader)
            
            extracted_graph, link_statistics = link_extractor.extract_all_links()
            
            # Validate link extraction accuracy
            validation_results = RealVaultTestHelper.validate_link_extraction_accuracy(
                vault_path, extracted_graph, vault_metadata["notes_metadata"]
            )
            
            print(f"\nLink extraction validation results:")
            print(f"  Expected links: {validation_results['total_expected_links']}")
            print(f"  Extracted links: {validation_results['total_extracted_links']}")
            print(f"  Correctly extracted: {validation_results['correctly_extracted']}")
            print(f"  False positives: {validation_results['false_positives']}")
            print(f"  False negatives: {validation_results['false_negatives']}")
            print(f"  Accuracy: {validation_results['accuracy']:.3f}")
            print(f"  Precision: {validation_results['precision']:.3f}")
            print(f"  Recall: {validation_results['recall']:.3f}")
            
            # Accuracy requirements (allowing for some variance in realistic scenarios)
            assert validation_results['accuracy'] >= 0.7, f"Link extraction accuracy too low: {validation_results['accuracy']:.3f}"
            assert validation_results['precision'] >= 0.6, f"Link extraction precision too low: {validation_results['precision']:.3f}"
            assert validation_results['recall'] >= 0.6, f"Link extraction recall too low: {validation_results['recall']:.3f}"
            
            # Validate link statistics
            assert isinstance(link_statistics, LinkStatistics)
            assert link_statistics.total_links >= 0
            assert link_statistics.unique_links >= 0
            assert link_statistics.broken_links >= 0
            
            # Print detailed errors for analysis
            if validation_results['detailed_errors']:
                print(f"\nDetailed errors (first 5):")
                for error in validation_results['detailed_errors'][:5]:
                    print(f"  {error['type']}: {error['description']}")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_performance_with_realistic_data_volumes(self, temp_output_dir, mock_vector_encoder):
        """Test performance and memory usage with realistic data volumes."""
        # Test with progressively larger vaults
        vault_sizes = [
            ("small", 50),
            ("medium", 150), 
            ("large", 300)
        ]
        
        performance_results = []
        
        for complexity, expected_notes in vault_sizes:
            temp_dir = tempfile.mkdtemp()
            vault_path = Path(temp_dir) / f"performance_test_{complexity}"
            
            try:
                # Create realistic vault
                vault_metadata = RealVaultTestHelper.create_realistic_obsidian_vault(
                    vault_path, complexity=complexity
                )
                
                # Create generator
                generator = DatasetGenerator(
                    vault_path=vault_path,
                    output_dir=temp_output_dir
                )
                
                with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                     patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                    
                    # Monitor performance
                    gc.collect()  # Clean up before measurement
                    
                    start_time = time.time()
                    start_memory = self._get_memory_usage()
                    
                    # Generate datasets with appropriate batch size
                    batch_size = {"small": 10, "medium": 20, "large": 30}[complexity]
                    result = generator.generate_datasets(batch_size=batch_size)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    # Calculate metrics
                    total_time = end_time - start_time
                    memory_increase = end_memory - start_memory
                    
                    performance_data = {
                        "complexity": complexity,
                        "expected_notes": expected_notes,
                        "actual_notes": result.summary.notes_processed if result.success else 0,
                        "total_time": total_time,
                        "memory_increase": memory_increase,
                        "processing_rate": result.summary.processing_rate if result.success else 0,
                        "success": result.success
                    }
                    
                    performance_results.append(performance_data)
                    
                    print(f"\n{complexity.title()} vault performance:")
                    print(f"  Notes processed: {performance_data['actual_notes']}")
                    print(f"  Total time: {total_time:.2f} seconds")
                    print(f"  Memory increase: {memory_increase:.2f} MB")
                    print(f"  Processing rate: {performance_data['processing_rate']:.2f} notes/second")
                    
                    # Performance requirements
                    max_times = {"small": 90, "medium": 240, "large": 480}
                    max_memory = {"small": 300, "medium": 600, "large": 1200}
                    
                    assert result.success is True, f"Dataset generation failed for {complexity} vault"
                    assert total_time <= max_times[complexity], f"Processing too slow for {complexity}: {total_time:.2f}s"
                    assert memory_increase <= max_memory[complexity], f"Memory usage too high for {complexity}: {memory_increase:.2f}MB"
                    
                    # Validate output quality
                    notes_df = pd.read_csv(result.notes_dataset_path)
                    pairs_df = pd.read_csv(result.pairs_dataset_path)
                    
                    assert len(notes_df) > 0, f"Empty notes dataset for {complexity}"
                    assert len(pairs_df) > 0, f"Empty pairs dataset for {complexity}"
                    
                    # Data quality checks
                    assert notes_df['word_count'].min() >= 0, f"Invalid word counts in {complexity}"
                    assert pairs_df['cosine_similarity'].between(0, 1).all(), f"Invalid similarities in {complexity}"
                    
            finally:
                import shutil
                shutil.rmtree(temp_dir)
                gc.collect()  # Clean up after each test
        
        # Analyze scaling behavior
        print(f"\nPerformance scaling analysis:")
        for i, result in enumerate(performance_results):
            if i > 0:
                prev_result = performance_results[i-1]
                time_ratio = result['total_time'] / prev_result['total_time']
                memory_ratio = result['memory_increase'] / max(prev_result['memory_increase'], 1)
                notes_ratio = result['actual_notes'] / max(prev_result['actual_notes'], 1)
                
                print(f"  {prev_result['complexity']} -> {result['complexity']}:")
                print(f"    Notes ratio: {notes_ratio:.2f}x")
                print(f"    Time ratio: {time_ratio:.2f}x")
                print(f"    Memory ratio: {memory_ratio:.2f}x")
                
                # Scaling should be reasonable (not exponential)
                assert time_ratio <= notes_ratio * 2, f"Time scaling too poor: {time_ratio:.2f}x vs {notes_ratio:.2f}x notes"
                assert memory_ratio <= notes_ratio * 1.5, f"Memory scaling too poor: {memory_ratio:.2f}x vs {notes_ratio:.2f}x notes"
    
    def test_memory_efficiency_with_large_vault(self, temp_output_dir, mock_vector_encoder):
        """Test memory efficiency with large vault to ensure no memory leaks."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "memory_efficiency_test"
        
        try:
            # Create large vault for memory testing
            vault_metadata = RealVaultTestHelper.create_realistic_obsidian_vault(
                vault_path, complexity="large"
            )
            
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                # Monitor memory usage throughout processing
                memory_samples = []
                
                def memory_monitor():
                    for _ in range(60):  # Monitor for 60 seconds
                        memory_samples.append(self._get_memory_usage())
                        time.sleep(1)
                
                import threading
                monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
                monitor_thread.start()
                
                # Generate datasets
                start_memory = self._get_memory_usage()
                result = generator.generate_datasets(batch_size=25)
                end_memory = self._get_memory_usage()
                
                # Analyze memory usage pattern
                if memory_samples:
                    max_memory = max(memory_samples)
                    min_memory = min(memory_samples)
                    memory_variance = max_memory - min_memory
                    
                    print(f"\nMemory efficiency analysis:")
                    print(f"  Start memory: {start_memory:.2f} MB")
                    print(f"  End memory: {end_memory:.2f} MB")
                    print(f"  Peak memory: {max_memory:.2f} MB")
                    print(f"  Memory variance: {memory_variance:.2f} MB")
                    print(f"  Final increase: {end_memory - start_memory:.2f} MB")
                    
                    # Memory efficiency requirements
                    assert result.success is True, "Large vault processing should succeed"
                    assert memory_variance <= 800, f"Memory variance too high: {memory_variance:.2f}MB"
                    assert end_memory - start_memory <= 600, f"Final memory increase too high: {end_memory - start_memory:.2f}MB"
                    
                    # Check for memory leaks (end memory should be close to start)
                    memory_leak = end_memory - start_memory
                    assert memory_leak <= 200, f"Possible memory leak detected: {memory_leak:.2f}MB increase"
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
            gc.collect()
    
    def test_concurrent_processing_with_real_data(self, temp_output_dir, mock_vector_encoder):
        """Test concurrent processing capabilities with realistic data."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "concurrent_test"
        
        try:
            # Create medium-sized vault for concurrent testing
            vault_metadata = RealVaultTestHelper.create_realistic_obsidian_vault(
                vault_path, complexity="medium"
            )
            
            # Test concurrent dataset generation
            import threading
            results = []
            errors = []
            
            def generate_concurrent_dataset(thread_id):
                try:
                    thread_output_dir = temp_output_dir / f"concurrent_{thread_id}"
                    thread_output_dir.mkdir(exist_ok=True)
                    
                    generator = DatasetGenerator(
                        vault_path=vault_path,
                        output_dir=thread_output_dir
                    )
                    
                    with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                         patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                        
                        start_time = time.time()
                        result = generator.generate_datasets(batch_size=15)
                        end_time = time.time()
                        
                        results.append({
                            "thread_id": thread_id,
                            "success": result.success,
                            "time": end_time - start_time,
                            "notes_processed": result.summary.notes_processed if result.success else 0,
                            "error": result.error_message if not result.success else None
                        })
                        
                except Exception as e:
                    errors.append({"thread_id": thread_id, "error": str(e)})
            
            # Start multiple concurrent threads
            threads = []
            num_threads = 3
            
            for i in range(num_threads):
                thread = threading.Thread(target=generate_concurrent_dataset, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Analyze concurrent processing results
            print(f"\nConcurrent processing results:")
            print(f"  Threads: {num_threads}")
            print(f"  Errors: {len(errors)}")
            print(f"  Successful results: {len([r for r in results if r['success']])}")
            
            # Validate concurrent processing
            assert len(errors) == 0, f"Concurrent processing errors: {errors}"
            assert len(results) == num_threads, "Not all threads completed"
            
            successful_results = [r for r in results if r['success']]
            assert len(successful_results) == num_threads, "Not all concurrent processes succeeded"
            
            # Check consistency across concurrent runs
            processing_times = [r['time'] for r in successful_results]
            notes_processed = [r['notes_processed'] for r in successful_results]
            
            # All should process same number of notes
            assert len(set(notes_processed)) == 1, f"Inconsistent notes processed: {notes_processed}"
            
            # Processing times should be reasonably consistent
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Max time: {max_time:.2f}s")
            print(f"  Time variance: {max_time - avg_time:.2f}s")
            
            assert max_time <= avg_time * 1.5, "Concurrent processing times too inconsistent"
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_error_recovery_with_realistic_corrupted_data(self, temp_output_dir, mock_vector_encoder):
        """Test error recovery with realistic corrupted data scenarios."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "error_recovery_test"
        
        try:
            # Create base vault
            vault_metadata = RealVaultTestHelper.create_realistic_obsidian_vault(
                vault_path, complexity="small"
            )
            
            # Add realistic corrupted files
            corrupted_scenarios = [
                ("empty_file.md", ""),  # Empty file
                ("binary_file.md", b'\xff\xfe\x00\x00\x01\x02'),  # Binary content
                ("malformed_frontmatter.md", """---
title: Malformed
tags: [unclosed
---
# Content"""),  # Malformed YAML
                ("broken_links.md", """# Broken Links
[[Unclosed link
[Malformed](
[[]]
[[   ]]
"""),  # Malformed links
                ("very_large_file.md", "# Large File\n" + "Content line.\n" * 10000),  # Very large file
            ]
            
            for filename, content in corrupted_scenarios:
                corrupted_file = vault_path / filename
                if isinstance(content, bytes):
                    corrupted_file.write_bytes(content)
                else:
                    corrupted_file.write_text(content, encoding='utf-8')
            
            # Test dataset generation with corrupted files
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_vector_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_vector_encoder):
                
                result = generator.generate_datasets(batch_size=10)
                
                print(f"\nError recovery test results:")
                print(f"  Success: {result.success}")
                print(f"  Notes processed: {result.summary.notes_processed if result.success else 0}")
                print(f"  Error message: {result.error_message if not result.success else 'None'}")
                
                # Should either succeed with partial data or fail gracefully
                if result.success:
                    # If successful, should have processed some valid files
                    assert result.summary.notes_processed > 0, "Should process some valid files"
                    
                    # Validate output quality
                    notes_df = pd.read_csv(result.notes_dataset_path)
                    pairs_df = pd.read_csv(result.pairs_dataset_path)
                    
                    assert len(notes_df) > 0, "Should generate some notes data"
                    assert len(pairs_df) > 0, "Should generate some pairs data"
                    
                    print(f"  Successfully processed {len(notes_df)} notes despite corrupted files")
                    
                else:
                    # If failed, should provide meaningful error message
                    assert result.error_message is not None, "Should provide error message"
                    assert len(result.error_message) > 0, "Error message should not be empty"
                    
                    print(f"  Graceful failure with error: {result.error_message}")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB (simplified approach without psutil)."""
        try:
            # Try to get memory usage from /proc/self/status on Linux/macOS
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        # Extract memory in KB and convert to MB
                        memory_kb = int(line.split()[1])
                        return memory_kb / 1024
        except (FileNotFoundError, OSError, IndexError, ValueError):
            pass
        
        # Fallback: use resource module if available
        try:
            import resource
            # Get memory usage in KB and convert to MB
            memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On macOS, ru_maxrss is in bytes, on Linux it's in KB
            import platform
            if platform.system() == 'Darwin':  # macOS
                return memory_kb / (1024 * 1024)  # bytes to MB
            else:  # Linux
                return memory_kb / 1024  # KB to MB
        except (ImportError, AttributeError):
            pass
        
        # Final fallback: return 0 (memory monitoring disabled)
        return 0.0


class TestDatasetGenerationRealWorldScenarios:
    """Test real-world scenarios and edge cases."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_academic_research_vault_scenario(self, temp_output_dir):
        """Test with academic research vault structure."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "academic_vault"
        
        try:
            # Create academic-style vault structure
            vault_path.mkdir(parents=True)
            
            # Create academic folders
            folders = [
                "Literature Review",
                "Research Papers", 
                "Methodology",
                "Data Analysis",
                "Thesis Chapters",
                "Meeting Notes",
                "References"
            ]
            
            for folder in folders:
                (vault_path / folder).mkdir()
            
            # Create academic-style notes with citations and complex links
            academic_notes = [
                ("Literature Review/machine_learning_survey.md", """---
title: Machine Learning Survey
authors: [Smith et al., 2023]
tags: [literature-review, ml, survey]
status: 🌲
---

# Machine Learning Survey

## Overview
Comprehensive survey of machine learning techniques [[Methodology/research_methods.md|research methods]].

## Key Papers
- [[Research Papers/neural_networks_2023.md]]
- [[Research Papers/deep_learning_advances.md]]

## Connections
This survey informs our [[Thesis Chapters/chapter_2_background.md|background chapter]].
"""),
                ("Research Papers/neural_networks_2023.md", """---
title: Neural Networks 2023
authors: [Johnson, A.]
venue: ICML 2023
tags: [paper, neural-networks, 2023]
status: 🌿
---

# Neural Networks 2023

## Abstract
Recent advances in neural network architectures.

## Methodology
See [[Methodology/experimental_design.md]] for similar approaches.

## Related Work
Builds on [[Research Papers/deep_learning_advances.md]].
"""),
                ("Thesis Chapters/chapter_2_background.md", """---
title: Chapter 2 - Background
chapter: 2
tags: [thesis, background]
status: 🌱
---

# Chapter 2: Background

## Literature Review
Based on [[Literature Review/machine_learning_survey.md]].

## Theoretical Foundation
Key concepts from [[Research Papers/neural_networks_2023.md]].
""")
            ]
            
            for file_path, content in academic_notes:
                full_path = vault_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
            
            # Test dataset generation
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            # Mock vector encoder
            mock_encoder = Mock()
            mock_encoder.encode_documents.return_value = np.random.rand(10, 384)
            mock_encoder.encode_batch.return_value = np.random.rand(10, 384)
            mock_encoder.get_embedding_dimension.return_value = 384
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_encoder):
                
                result = generator.generate_datasets()
                
                assert result.success is True, "Academic vault processing should succeed"
                
                # Validate academic-specific features
                notes_df = pd.read_csv(result.notes_dataset_path)
                
                # Should capture academic metadata
                assert 'tag_count' in notes_df.columns
                assert notes_df['tag_count'].max() > 0, "Should capture academic tags"
                
                print(f"Academic vault test successful: {len(notes_df)} notes processed")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_personal_knowledge_management_scenario(self, temp_output_dir):
        """Test with personal knowledge management vault structure."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "pkm_vault"
        
        try:
            # Create PKM-style structure (Zettelkasten-inspired)
            vault_path.mkdir(parents=True)
            
            # Create PKM folders
            folders = ["Permanent Notes", "Literature Notes", "Daily Notes", "Projects", "MOCs"]
            for folder in folders:
                (vault_path / folder).mkdir()
            
            # Create interconnected notes with timestamps and IDs
            pkm_notes = [
                ("Permanent Notes/202401011200_learning_systems.md", """---
title: Learning Systems
id: 202401011200
tags: [permanent, learning, systems]
status: 🌲
---

# Learning Systems

Core principles of learning systems design.

## Connections
- [[202401011300_feedback_loops.md|Feedback Loops]]
- [[Literature Notes/202401010900_educational_theory.md]]

## Development
Started from [[Daily Notes/2024-01-01.md]] insights.
"""),
                ("Permanent Notes/202401011300_feedback_loops.md", """---
title: Feedback Loops
id: 202401011300
tags: [permanent, feedback, systems]
status: 🌿
---

# Feedback Loops

Essential component of [[202401011200_learning_systems.md|learning systems]].

## Types
1. Positive feedback
2. Negative feedback

## Applications
Used in [[Projects/learning_app.md]].
"""),
                ("Daily Notes/2024-01-01.md", """---
title: 2024-01-01
date: 2024-01-01
tags: [daily]
---

# 2024-01-01

## Insights
Key insight about learning systems led to [[Permanent Notes/202401011200_learning_systems.md]].

## Reading
- [[Literature Notes/202401010900_educational_theory.md]]
""")
            ]
            
            for file_path, content in pkm_notes:
                full_path = vault_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
            
            # Test dataset generation
            generator = DatasetGenerator(
                vault_path=vault_path,
                output_dir=temp_output_dir
            )
            
            # Mock vector encoder
            mock_encoder = Mock()
            mock_encoder.encode_documents.return_value = np.random.rand(5, 384)
            mock_encoder.encode_batch.return_value = np.random.rand(5, 384)
            mock_encoder.get_embedding_dimension.return_value = 384
            
            with patch.object(generator.notes_generator, 'vector_encoder', mock_encoder), \
                 patch.object(generator.pairs_generator, 'vector_encoder', mock_encoder):
                
                result = generator.generate_datasets()
                
                assert result.success is True, "PKM vault processing should succeed"
                
                # Validate PKM-specific patterns
                notes_df = pd.read_csv(result.notes_dataset_path)
                pairs_df = pd.read_csv(result.pairs_dataset_path)
                
                # Should capture timestamp-based IDs and connections
                assert len(notes_df) > 0
                assert len(pairs_df) > 0
                
                # Should have some linked pairs (PKM vaults are highly connected)
                linked_pairs = pairs_df[pairs_df['link_exists'] == True]
                assert len(linked_pairs) > 0, "PKM vault should have linked notes"
                
                print(f"PKM vault test successful: {len(notes_df)} notes, {len(linked_pairs)} linked pairs")
                
        finally:
            import shutil
            shutil.rmtree(temp_dir)