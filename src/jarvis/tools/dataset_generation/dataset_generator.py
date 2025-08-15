"""
Main orchestrator for dataset generation process.

This module coordinates the entire dataset generation workflow, including
link extraction, feature engineering, and dataset creation for both
individual notes and note pairs.
"""

import re
import time
from datetime import datetime
from pathlib import Path

from jarvis.services.graph.database import GraphDatabase
from jarvis.services.vault.reader import VaultReader
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.utils.config import get_settings
from jarvis.utils.logging import setup_logging

from .extractors.link_extractor import LinkExtractor
from .generators.notes_dataset_generator import NotesDatasetGenerator
from .generators.pairs_dataset_generator import (
    PairsDatasetGenerator,
    RandomSamplingStrategy,
    StratifiedSamplingStrategy,
)
from .models.data_models import (
    DatasetGenerationResult,
    GenerationSummary,
    NoteData,
    ValidationResult,
)
from .models.exceptions import (
    ConfigurationError,
    InsufficientDataError,
    VaultValidationError,
)
from .utils.progress_tracker import ProgressTracker, BatchProgressTracker

logger = setup_logging(__name__)


class DatasetGenerator:
    """Main orchestrator for dataset generation process."""

    def __init__(self, vault_path: Path, output_dir: Path = None, areas_only: bool = None, 
                 skip_validation: bool = False):
        """Initialize the dataset generator.
        
        Args:
            vault_path: Path to the Obsidian vault
            output_dir: Directory for output files (uses default from settings if None)
            areas_only: Whether to filter content to Areas/ folder only (uses settings if None)
            skip_validation: Skip validation during initialization (for testing)
            
        Raises:
            VaultValidationError: If vault is invalid
            ConfigurationError: If configuration is invalid
        """
        self.vault_path = vault_path.resolve()
        
        # Resolve configuration settings
        settings = get_settings()
        
        # Set Areas filtering mode
        if areas_only is None:
            areas_only = settings.dataset_areas_only
        self.areas_only = areas_only
        
        # Resolve output directory with tilde expansion
        if output_dir is None:
            output_dir = settings.get_dataset_output_path()
        else:
            output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir = output_dir

        # Initialize services first (needed for validation)
        try:
            self.vault_reader = VaultReader(str(vault_path), areas_only=self.areas_only)
            self.vector_encoder = VectorEncoder()

            # GraphDatabase is optional
            settings = get_settings()
            self.graph_database = None
            if settings.graph_enabled:
                try:
                    self.graph_database = GraphDatabase(settings)
                    logger.info("GraphDatabase service initialized")
                except Exception as e:
                    logger.warning(f"GraphDatabase initialization failed: {e}, continuing without graph features")

            # Initialize components
            self.link_extractor = LinkExtractor(self.vault_reader)
            self.notes_generator = NotesDatasetGenerator(
                self.vault_reader, self.vector_encoder, self.graph_database
            )
            self.pairs_generator = PairsDatasetGenerator(
                self.vector_encoder, self.graph_database
            )

            logger.info(f"DatasetGenerator initialized with vault: {vault_path}")

        except Exception as e:
            logger.error(f"Failed to initialize DatasetGenerator: {e}")
            raise ConfigurationError(f"Service initialization failed: {e}") from e

        # Validate inputs (unless skipped for testing) - after services are initialized
        if not skip_validation:
            validation_result = self.validate_vault()
            if not validation_result.valid:
                raise VaultValidationError(
                    f"Vault validation failed: {', '.join(validation_result.errors)}",
                    vault_path=str(vault_path)
                )

    def generate_datasets(self,
                         notes_filename: str = "notes_dataset.csv",
                         pairs_filename: str = "pairs_dataset.csv",
                         negative_sampling_ratio: float = 5.0,
                         sampling_strategy: str = "stratified",
                         batch_size: int = 32,
                         max_pairs_per_note: int = 1000,
                         progress_callback=None) -> DatasetGenerationResult:
        """Generate both notes and pairs datasets.
        
        Args:
            notes_filename: Output filename for notes dataset
            pairs_filename: Output filename for pairs dataset
            negative_sampling_ratio: Ratio of negative to positive examples
            sampling_strategy: Sampling strategy ('random' or 'stratified')
            batch_size: Batch size for processing
            max_pairs_per_note: Maximum pairs per note
            progress_callback: Optional callback for progress updates
            
        Returns:
            DatasetGenerationResult with generation summary and file paths
        """
        start_time = time.time()
        logger.info("Starting comprehensive dataset generation")
        
        # Enhanced logging for filtering status with privacy messaging
        filtering_metadata = self._get_filtering_metadata()
        if self.areas_only:
            logger.info("🔒 PRIVACY MODE: Processing Areas/ folder only - personal content excluded")
            logger.info(f"📁 Areas folder: {filtering_metadata.get('areas_folder_path', 'Areas/')}")
            if filtering_metadata.get('excluded_folders'):
                excluded_count = len(filtering_metadata['excluded_folders'])
                logger.info(f"🚫 Excluding {excluded_count} folders for privacy: {', '.join(filtering_metadata['excluded_folders'][:5])}")
                if excluded_count > 5:
                    logger.info(f"    ... and {excluded_count - 5} more folders")
            logger.info("ℹ️  This ensures personal journals, people notes, and private content remain private")
        else:
            logger.info("📂 Processing entire vault - all content included")
            logger.warning("⚠️  Personal content will be included in dataset generation")

        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: Extract links and build graph
            logger.info("Step 1: Extracting links and building graph")
            if progress_callback:
                progress_callback("Extracting links...", 0, 5)

            link_graph, link_statistics = self.link_extractor.extract_all_links()

            # Step 2: Get all notes with filtering-aware discovery
            logger.info("Step 2: Discovering notes")
            if progress_callback:
                progress_callback("Discovering notes...", 1, 5)

            all_notes = [str(path) for path in self.vault_reader.get_markdown_files()]
            
            # Filter the link_graph to only include nodes from all_notes
            if self.areas_only:
                link_graph = link_graph.subgraph([node for node in link_graph.nodes() if node in all_notes]).copy()
            
            # Enhanced logging for filtered content discovery
            if self.areas_only:
                # Get total vault count for comparison
                try:
                    from jarvis.services.vault.reader import VaultReader
                    full_vault_reader = VaultReader(str(self.vault_path), areas_only=False)
                    total_vault_notes = len(list(full_vault_reader.get_markdown_files()))
                    excluded_count = total_vault_notes - len(all_notes)
                    
                    logger.info(f"📊 Content filtering results:")
                    logger.info(f"   • Areas/ notes found: {len(all_notes)}")
                    logger.info(f"   • Total vault notes: {total_vault_notes}")
                    logger.info(f"   • Notes excluded for privacy: {excluded_count}")
                    logger.info(f"   • Privacy protection: {excluded_count/total_vault_notes*100:.1f}% of content excluded")
                except Exception as e:
                    logger.info(f"📊 Found {len(all_notes)} notes in Areas/ folder (privacy filtering active)")
                    logger.debug(f"Could not calculate exclusion stats: {e}")
            else:
                logger.info(f"📊 Found {len(all_notes)} notes in vault (full vault processing)")

            if len(all_notes) < 5:
                raise InsufficientDataError(
                    f"Insufficient notes for dataset generation: {len(all_notes)} < 5",
                    required_minimum=5,
                    actual_count=len(all_notes)
                )

            # Step 3: Generate notes dataset
            logger.info("Step 3: Generating notes dataset")
            if progress_callback:
                progress_callback("Generating notes dataset...", 2, 5)

            def notes_progress(processed, total):
                if progress_callback:
                    progress_callback(f"Processing notes: {processed}/{total}", 2, 5)

            notes_dataset = self.notes_generator.generate_dataset(
                all_notes, link_graph, batch_size=batch_size, progress_callback=notes_progress
            )

            # Save notes dataset
            notes_output_path = self.output_dir / notes_filename
            notes_dataset.to_csv(notes_output_path, index=False)
            logger.info(f"Notes dataset saved to: {notes_output_path}")

            # Step 4: Load note data for pairs generation
            logger.info("Step 4: Preparing note data for pairs generation")
            if progress_callback:
                progress_callback("Preparing note data...", 3, 5)

            notes_data = self._load_notes_data_for_pairs(all_notes)

            # Step 5: Generate pairs dataset
            logger.info("Step 5: Generating pairs dataset")
            if progress_callback:
                progress_callback("Generating pairs dataset...", 4, 5)

            # Set up sampling strategy
            if sampling_strategy == "stratified":
                sampling_strategy_obj = StratifiedSamplingStrategy(notes_data)
            else:
                sampling_strategy_obj = RandomSamplingStrategy()

            self.pairs_generator.sampling_strategy = sampling_strategy_obj

            def pairs_progress(processed, total):
                if progress_callback:
                    progress_callback(f"Processing pairs: {processed}/{total}", 4, 5)

            pairs_dataset = self.pairs_generator.generate_dataset(
                notes_data, link_graph,
                negative_sampling_ratio=negative_sampling_ratio,
                max_pairs_per_note=max_pairs_per_note,
                batch_size=batch_size,
                progress_callback=pairs_progress
            )

            # Save pairs dataset
            pairs_output_path = self.output_dir / pairs_filename
            pairs_dataset.to_csv(pairs_output_path, index=False)
            logger.info(f"Pairs dataset saved to: {pairs_output_path}")

            # Step 6: Generate summary
            if progress_callback:
                progress_callback("Finalizing...", 5, 5)

            total_time = time.time() - start_time

            # Create generation summary
            positive_pairs = pairs_dataset['link_exists'].sum() if 'link_exists' in pairs_dataset.columns else 0
            negative_pairs = len(pairs_dataset) - positive_pairs

            # Get filtering metadata if Areas filtering is enabled
            filtering_metadata = self._get_filtering_metadata()

            # Enhanced validation result with filtering metadata
            validation_result = ValidationResult(
                valid=True,
                notes_processed=len(notes_dataset),
                links_extracted=link_statistics.total_links,
                links_broken=link_statistics.broken_links,
                # Add filtering metadata to validation result
                areas_folder_exists=filtering_metadata.get("areas_folder_path") is not None,
                areas_notes_count=filtering_metadata.get("areas_notes_count", len(all_notes)),
                filtering_mode="areas_only" if filtering_metadata["filtering_enabled"] else "full_vault",
                excluded_notes_count=filtering_metadata.get("excluded_folder_count", 0),
                areas_folder_path=filtering_metadata.get("areas_folder_path"),
                areas_validation_passed=filtering_metadata.get("content_protection_level") != "unknown"
            )

            # Calculate privacy protection percentage if filtering is enabled
            privacy_protection_percentage = None
            total_vault_notes = None
            if filtering_metadata["filtering_enabled"]:
                try:
                    from jarvis.services.vault.reader import VaultReader
                    full_vault_reader = VaultReader(str(self.vault_path), areas_only=False)
                    total_vault_notes = len(list(full_vault_reader.get_markdown_files()))
                    if total_vault_notes > 0:
                        excluded_count = total_vault_notes - len(all_notes)
                        privacy_protection_percentage = (excluded_count / total_vault_notes) * 100
                except Exception as e:
                    logger.debug(f"Could not calculate privacy protection percentage: {e}")

            summary = GenerationSummary(
                total_notes=len(all_notes),
                notes_processed=len(notes_dataset),
                notes_failed=len(all_notes) - len(notes_dataset),
                pairs_generated=len(pairs_dataset),
                positive_pairs=positive_pairs,
                negative_pairs=negative_pairs,
                total_time_seconds=total_time,
                link_statistics=link_statistics,
                validation_result=validation_result,
                output_files={
                    "notes_dataset": str(notes_output_path),
                    "pairs_dataset": str(pairs_output_path)
                },
                performance_metrics={
                    "notes_per_second": len(notes_dataset) / total_time if total_time > 0 else 0,
                    "pairs_per_second": len(pairs_dataset) / total_time if total_time > 0 else 0
                },
                # Enhanced filtering metadata
                filtering_enabled=filtering_metadata["filtering_enabled"],
                areas_folder_path=filtering_metadata["areas_folder_path"],
                excluded_folders=filtering_metadata["excluded_folders"],
                privacy_mode=filtering_metadata["privacy_mode"],
                filtering_summary=filtering_metadata.get("filtering_summary"),
                content_protection_level=filtering_metadata.get("content_protection_level", "none"),
                privacy_message=filtering_metadata.get("privacy_message"),
                excluded_folder_count=filtering_metadata.get("excluded_folder_count", 0),
                areas_notes_count=filtering_metadata.get("areas_notes_count", len(all_notes)),
                total_vault_notes=total_vault_notes,
                privacy_protection_percentage=privacy_protection_percentage
            )

            result = DatasetGenerationResult(
                success=True,
                summary=summary,
                notes_dataset_path=str(notes_output_path),
                pairs_dataset_path=str(pairs_output_path)
            )

            # Enhanced completion logging with filtering information
            logger.info(f"✅ Dataset generation completed successfully in {total_time:.2f}s")
            logger.info(f"📊 Generation Results:")
            logger.info(f"   • Notes dataset: {len(notes_dataset)} rows")
            logger.info(f"   • Pairs dataset: {len(pairs_dataset)} rows ({positive_pairs} positive, {negative_pairs} negative)")
            
            # Log filtering summary
            if filtering_metadata["filtering_enabled"]:
                logger.info(f"🔒 Privacy Protection Summary:")
                logger.info(f"   • Content source: Areas/ folder only")
                logger.info(f"   • Excluded folders: {len(filtering_metadata.get('excluded_folders', []))}")
                logger.info(f"   • Privacy mode: Active - personal content protected")
            else:
                logger.info(f"📂 Full vault processing - all content included")

            return result

        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")

            total_time = time.time() - start_time

            # Create failed result
            failed_summary = GenerationSummary(
                total_notes=len(all_notes) if 'all_notes' in locals() else 0,
                notes_processed=0,
                notes_failed=0,
                pairs_generated=0,
                positive_pairs=0,
                negative_pairs=0,
                total_time_seconds=total_time,
                link_statistics=link_statistics if 'link_statistics' in locals() else None,
                validation_result=ValidationResult(valid=False, errors=[str(e)])
            )

            result = DatasetGenerationResult(
                success=False,
                summary=failed_summary,
                error_message=str(e)
            )

            return result

    def generate_datasets_with_progress_tracking(self,
                                               notes_filename: str = "notes_dataset.csv",
                                               pairs_filename: str = "pairs_dataset.csv",
                                               negative_sampling_ratio: float = 5.0,
                                               sampling_strategy: str = "stratified",
                                               batch_size: int = 32,
                                               max_pairs_per_note: int = 1000,
                                               use_parallel: bool = True,
                                               progress_callback=None) -> DatasetGenerationResult:
        """Generate datasets with comprehensive progress tracking and performance monitoring.
        
        Args:
            notes_filename: Output filename for notes dataset
            pairs_filename: Output filename for pairs dataset
            negative_sampling_ratio: Ratio of negative to positive examples
            sampling_strategy: Sampling strategy ('random' or 'stratified')
            batch_size: Batch size for processing
            max_pairs_per_note: Maximum pairs per note
            use_parallel: Whether to use parallel processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            DatasetGenerationResult with generation summary and file paths
        """
        start_time = time.time()
        logger.info("Starting dataset generation with comprehensive progress tracking")
        
        # Enhanced logging for filtering status with privacy messaging
        filtering_metadata = self._get_filtering_metadata()
        if self.areas_only:
            logger.info("🔒 PRIVACY MODE: Processing Areas/ folder only - personal content excluded")
            logger.info(f"📁 Areas folder: {filtering_metadata.get('areas_folder_path', 'Areas/')}")
            if filtering_metadata.get('excluded_folders'):
                excluded_count = len(filtering_metadata['excluded_folders'])
                logger.info(f"🚫 Excluding {excluded_count} folders for privacy: {', '.join(filtering_metadata['excluded_folders'][:5])}")
                if excluded_count > 5:
                    logger.info(f"    ... and {excluded_count - 5} more folders")
            logger.info("ℹ️  This ensures personal journals, people notes, and private content remain private")
        else:
            logger.info("📂 Processing entire vault - all content included")
            logger.warning("⚠️  Personal content will be included in dataset generation")

        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get all notes first to estimate total work with filtering-aware discovery
            all_notes = [str(path) for path in self.vault_reader.get_markdown_files()]
            
            # Enhanced logging for filtered content discovery
            if self.areas_only:
                # Get total vault count for comparison
                try:
                    from jarvis.services.vault.reader import VaultReader
                    full_vault_reader = VaultReader(str(self.vault_path), areas_only=False)
                    total_vault_notes = len(list(full_vault_reader.get_markdown_files()))
                    excluded_count = total_vault_notes - len(all_notes)
                    
                    logger.info(f"📊 Content filtering results:")
                    logger.info(f"   • Areas/ notes found: {len(all_notes)}")
                    logger.info(f"   • Total vault notes: {total_vault_notes}")
                    logger.info(f"   • Notes excluded for privacy: {excluded_count}")
                    logger.info(f"   • Privacy protection: {excluded_count/total_vault_notes*100:.1f}% of content excluded")
                except Exception as e:
                    logger.info(f"📊 Found {len(all_notes)} notes in Areas/ folder (privacy filtering active)")
                    logger.debug(f"Could not calculate exclusion stats: {e}")
            else:
                logger.info(f"📊 Found {len(all_notes)} notes in vault (full vault processing)")

            if len(all_notes) < 5:
                raise InsufficientDataError(
                    f"Insufficient notes for dataset generation: {len(all_notes)} < 5",
                    required_minimum=5,
                    actual_count=len(all_notes)
                )

            # Initialize comprehensive progress tracking
            # Estimate total work: link extraction + notes processing + pairs processing
            estimated_pairs = min(len(all_notes) * 50, 10000)  # Rough estimate
            total_work_items = len(all_notes) + estimated_pairs
            
            main_progress = ProgressTracker(
                total_items=total_work_items,
                description="Dataset Generation",
                update_interval=2.0
            )

            # Add progress callback if provided
            if progress_callback:
                def progress_wrapper(info):
                    progress_callback(
                        f"{info['description']}: {info['processed_items']}/{info['total_items']} "
                        f"({info['progress_percent']:.1f}%)",
                        info['processed_items'],
                        info['total_items']
                    )
                main_progress.add_progress_callback(progress_wrapper)

            # Step 1: Extract links and build graph
            logger.info("Step 1: Extracting links and building graph")
            main_progress.update(0, 0, force_update=True, context="Extracting links")
            
            link_graph, link_statistics = self.link_extractor.extract_all_links()
            
            # Step 2: Generate notes dataset with progress tracking
            logger.info("Step 2: Generating notes dataset")
            notes_progress = ProgressTracker(
                total_items=len(all_notes),
                description="Notes Processing",
                update_interval=1.0
            )

            def notes_progress_callback(processed, total):
                notes_progress.update(processed, 0, context="processing notes")
                # Update main progress (notes are first half of work)
                main_progress.update(processed, 0, context="notes dataset")

            # Choose processing method based on use_parallel flag
            if use_parallel and hasattr(self.notes_generator, 'generate_dataset_parallel'):
                notes_dataset = self.notes_generator.generate_dataset_parallel(
                    all_notes, link_graph, batch_size=batch_size, 
                    progress_callback=notes_progress_callback
                )
            else:
                notes_dataset = self.notes_generator.generate_dataset(
                    all_notes, link_graph, batch_size=batch_size, 
                    progress_callback=notes_progress_callback
                )

            notes_progress.finish("notes dataset generation")

            # Save notes dataset
            notes_output_path = self.output_dir / notes_filename
            notes_dataset.to_csv(notes_output_path, index=False)
            logger.info(f"Notes dataset saved to: {notes_output_path}")

            # Step 3: Load note data for pairs generation
            logger.info("Step 3: Preparing note data for pairs generation")
            main_progress.update(len(all_notes), 0, force_update=True, context="preparing pairs data")
            
            notes_data = self._load_notes_data_for_pairs(all_notes)

            # Step 4: Generate pairs dataset with progress tracking
            logger.info("Step 4: Generating pairs dataset")
            
            # Set up sampling strategy
            if sampling_strategy == "stratified":
                sampling_strategy_obj = StratifiedSamplingStrategy(notes_data)
            else:
                sampling_strategy_obj = RandomSamplingStrategy()

            self.pairs_generator.sampling_strategy = sampling_strategy_obj

            # Estimate actual number of pairs
            positive_pairs = self.pairs_generator._extract_positive_pairs(link_graph)
            target_negative_count = min(
                int(len(positive_pairs) * negative_sampling_ratio),
                (len(all_notes) * (len(all_notes) - 1)) // 2 - len(positive_pairs)
            )
            actual_pairs_count = len(positive_pairs) + target_negative_count

            pairs_progress = ProgressTracker(
                total_items=actual_pairs_count,
                description="Pairs Processing",
                update_interval=1.0
            )

            def pairs_progress_callback(processed, total):
                pairs_progress.update(processed, 0, context="processing pairs")
                # Update main progress (pairs are second half of work)
                main_progress.update(len(all_notes) + processed, 0, context="pairs dataset")

            # Choose processing method based on use_parallel flag
            if use_parallel and hasattr(self.pairs_generator, 'generate_dataset_parallel'):
                pairs_dataset = self.pairs_generator.generate_dataset_parallel(
                    notes_data, link_graph,
                    negative_sampling_ratio=negative_sampling_ratio,
                    max_pairs_per_note=max_pairs_per_note,
                    batch_size=batch_size,
                    progress_callback=pairs_progress_callback
                )
            else:
                pairs_dataset = self.pairs_generator.generate_dataset(
                    notes_data, link_graph,
                    negative_sampling_ratio=negative_sampling_ratio,
                    max_pairs_per_note=max_pairs_per_note,
                    batch_size=batch_size,
                    progress_callback=pairs_progress_callback
                )

            pairs_progress.finish("pairs dataset generation")

            # Save pairs dataset
            pairs_output_path = self.output_dir / pairs_filename
            pairs_dataset.to_csv(pairs_output_path, index=False)
            logger.info(f"Pairs dataset saved to: {pairs_output_path}")

            # Step 5: Finalize and generate summary
            main_progress.update(total_work_items, 0, force_update=True, context="finalizing")
            main_progress.finish("dataset generation")

            total_time = time.time() - start_time

            # Get comprehensive performance metrics
            main_performance = main_progress.get_performance_summary()
            notes_performance = notes_progress.get_performance_summary()
            pairs_performance = pairs_progress.get_performance_summary()

            # Create generation summary with enhanced metrics
            positive_pairs_count = pairs_dataset['link_exists'].sum() if 'link_exists' in pairs_dataset.columns else 0
            negative_pairs_count = len(pairs_dataset) - positive_pairs_count

            # Get filtering metadata if Areas filtering is enabled
            filtering_metadata = self._get_filtering_metadata()

            # Enhanced validation result with filtering metadata
            validation_result = ValidationResult(
                valid=True,
                notes_processed=len(notes_dataset),
                links_extracted=link_statistics.total_links,
                links_broken=link_statistics.broken_links,
                # Add filtering metadata to validation result
                areas_folder_exists=filtering_metadata.get("areas_folder_path") is not None,
                areas_notes_count=filtering_metadata.get("areas_notes_count", len(all_notes)),
                filtering_mode="areas_only" if filtering_metadata["filtering_enabled"] else "full_vault",
                excluded_notes_count=filtering_metadata.get("excluded_folder_count", 0),
                areas_folder_path=filtering_metadata.get("areas_folder_path"),
                areas_validation_passed=filtering_metadata.get("content_protection_level") != "unknown"
            )

            # Calculate privacy protection percentage if filtering is enabled
            privacy_protection_percentage = None
            total_vault_notes = None
            if filtering_metadata["filtering_enabled"]:
                try:
                    from jarvis.services.vault.reader import VaultReader
                    full_vault_reader = VaultReader(str(self.vault_path), areas_only=False)
                    total_vault_notes = len(list(full_vault_reader.get_markdown_files()))
                    if total_vault_notes > 0:
                        excluded_count = total_vault_notes - len(all_notes)
                        privacy_protection_percentage = (excluded_count / total_vault_notes) * 100
                except Exception as e:
                    logger.debug(f"Could not calculate privacy protection percentage: {e}")

            summary = GenerationSummary(
                total_notes=len(all_notes),
                notes_processed=len(notes_dataset),
                notes_failed=len(all_notes) - len(notes_dataset),
                pairs_generated=len(pairs_dataset),
                positive_pairs=positive_pairs_count,
                negative_pairs=negative_pairs_count,
                total_time_seconds=total_time,
                link_statistics=link_statistics,
                validation_result=validation_result,
                output_files={
                    "notes_dataset": str(notes_output_path),
                    "pairs_dataset": str(pairs_output_path)
                },
                performance_metrics={
                    "notes_per_second": notes_performance.average_rate,
                    "pairs_per_second": pairs_performance.average_rate,
                    "peak_memory_mb": main_performance.peak_memory_mb,
                    "memory_increase_mb": main_performance.current_memory_mb - main_performance.initial_memory_mb,
                    "notes_success_rate": notes_performance.success_rate,
                    "pairs_success_rate": pairs_performance.success_rate,
                    "parallel_processing": use_parallel,
                    "batch_size": batch_size
                },
                # Enhanced filtering metadata
                filtering_enabled=filtering_metadata["filtering_enabled"],
                areas_folder_path=filtering_metadata["areas_folder_path"],
                excluded_folders=filtering_metadata["excluded_folders"],
                privacy_mode=filtering_metadata["privacy_mode"],
                filtering_summary=filtering_metadata.get("filtering_summary"),
                content_protection_level=filtering_metadata.get("content_protection_level", "none"),
                privacy_message=filtering_metadata.get("privacy_message"),
                excluded_folder_count=filtering_metadata.get("excluded_folder_count", 0),
                areas_notes_count=filtering_metadata.get("areas_notes_count", len(all_notes)),
                total_vault_notes=total_vault_notes,
                privacy_protection_percentage=privacy_protection_percentage
            )

            result = DatasetGenerationResult(
                success=True,
                summary=summary,
                notes_dataset_path=str(notes_output_path),
                pairs_dataset_path=str(pairs_output_path)
            )

            # Enhanced completion logging with filtering information
            logger.info(f"✅ Dataset generation with progress tracking completed successfully in {total_time:.2f}s")
            logger.info(f"📊 Generation Results:")
            logger.info(f"   • Notes dataset: {len(notes_dataset)} rows (rate: {notes_performance.average_rate:.2f}/sec)")
            logger.info(f"   • Pairs dataset: {len(pairs_dataset)} rows (rate: {pairs_performance.average_rate:.2f}/sec)")
            logger.info(f"   • Peak memory usage: {main_performance.peak_memory_mb:.1f}MB")
            
            # Log filtering summary
            if filtering_metadata["filtering_enabled"]:
                logger.info(f"🔒 Privacy Protection Summary:")
                logger.info(f"   • Content source: Areas/ folder only")
                logger.info(f"   • Excluded folders: {len(filtering_metadata.get('excluded_folders', []))}")
                logger.info(f"   • Privacy mode: Active - personal content protected")
            else:
                logger.info(f"📂 Full vault processing - all content included")

            return result

        except Exception as e:
            logger.error(f"Dataset generation with progress tracking failed: {e}")

            total_time = time.time() - start_time

            # Create failed result
            failed_summary = GenerationSummary(
                total_notes=len(all_notes) if 'all_notes' in locals() else 0,
                notes_processed=0,
                notes_failed=0,
                pairs_generated=0,
                positive_pairs=0,
                negative_pairs=0,
                total_time_seconds=total_time,
                link_statistics=link_statistics if 'link_statistics' in locals() else None,
                validation_result=ValidationResult(valid=False, errors=[str(e)])
            )

            result = DatasetGenerationResult(
                success=False,
                summary=failed_summary,
                error_message=str(e)
            )

            return result

    def _load_notes_data_for_pairs(self, note_paths: list[str]) -> dict[str, NoteData]:
        """Load note data needed for pairs generation.
        
        Args:
            note_paths: List of note paths
            
        Returns:
            Dictionary mapping paths to NoteData objects
        """
        logger.info(f"Loading note data for {len(note_paths)} notes")
        notes_data = {}
        valid_notes = []
        note_contents = []

        # First pass: load all note data without embeddings
        for note_path in note_paths:
            try:
                # Read file content and metadata
                content, metadata = self.vault_reader.read_file(note_path)

                # Extract basic information
                path_obj = Path(note_path)
                title = path_obj.stem

                # Extract tags
                tags = self._extract_tags(content, metadata)

                # Extract outgoing links (simplified)
                outgoing_links = self._extract_outgoing_links(content)

                # Get file statistics
                full_path = self.vault_reader.get_absolute_path(note_path)
                file_stat = full_path.stat()

                # Create NoteData object (without embedding initially)
                note_data = NoteData(
                    path=note_path,
                    title=title,
                    content=content,
                    metadata=metadata,
                    tags=tags,
                    outgoing_links=outgoing_links,
                    word_count=len(content.split()) if content else 0,
                    creation_date=datetime.fromtimestamp(file_stat.st_ctime),
                    last_modified=datetime.fromtimestamp(file_stat.st_mtime),
                    quality_stage=metadata.get('quality_stage', 'unknown')
                )

                notes_data[note_path] = note_data
                
                # Collect content for batch embedding generation
                if content and content.strip():
                    valid_notes.append(note_path)
                    note_contents.append(content)

            except Exception as e:
                logger.warning(f"Failed to load note data for {note_path}: {e}")

        # Second pass: generate embeddings in batch for efficiency
        logger.info(f"Generating embeddings for {len(valid_notes)} notes with content")
        try:
            if note_contents:
                embeddings = self.vector_encoder.encode_documents(note_contents)
                
                # Assign embeddings back to note data
                for i, note_path in enumerate(valid_notes):
                    if i < len(embeddings):
                        notes_data[note_path].embedding = embeddings[i]
                    else:
                        notes_data[note_path].embedding = None
                        
                logger.info(f"✅ Successfully generated {len(embeddings)} embeddings")
            else:
                logger.warning("No valid content found for embedding generation")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings in batch: {e}")
            # Set all embeddings to None if batch generation fails
            for note_path in valid_notes:
                notes_data[note_path].embedding = None

        logger.info(f"Successfully loaded {len(notes_data)} note data objects")
        return notes_data

    def _extract_tags(self, content: str, metadata: dict) -> list[str]:
        """Extract tags from content and metadata."""
        tags = set()

        # Extract from metadata
        if 'tags' in metadata:
            meta_tags = metadata['tags']
            if isinstance(meta_tags, list):
                tags.update(meta_tags)
            elif isinstance(meta_tags, str):
                tags.add(meta_tags)

        # Extract hashtags from content
        hashtag_pattern = re.compile(r'#([a-zA-Z0-9_/-]+)')
        content_tags = hashtag_pattern.findall(content or '')
        tags.update(content_tags)

        return list(tags)

    def _extract_outgoing_links(self, content: str) -> list[str]:
        """Extract outgoing links from content (simplified version)."""
        if not content:
            return []

        links = []

        # Simple regex for wikilinks [[link]]
        wikilink_pattern = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')
        wikilinks = wikilink_pattern.findall(content)
        links.extend(wikilinks)

        return links

    def _get_filtering_metadata(self) -> dict[str, any]:
        """Get comprehensive metadata about filtering configuration and status.
        
        Returns:
            Dictionary with detailed filtering metadata
        """
        metadata = {
            "filtering_enabled": self.areas_only,
            "areas_folder_path": None,
            "excluded_folders": [],
            "privacy_mode": self.areas_only,
            "filtering_summary": None,
            "content_protection_level": "none"
        }
        
        if self.areas_only:
            try:
                from jarvis.tools.dataset_generation.filters.areas_filter import AreasContentFilter
                areas_filter = AreasContentFilter(str(self.vault_path))
                
                # Get Areas folder path
                metadata["areas_folder_path"] = str(areas_filter.areas_folder_path)
                
                # Get comprehensive exclusion summary
                exclusion_summary = areas_filter.get_exclusion_summary()
                metadata["excluded_folders"] = exclusion_summary.get("excluded_folders", [])
                metadata["excluded_folder_count"] = exclusion_summary.get("excluded_folder_count", 0)
                
                # Get Areas structure information
                areas_structure = areas_filter.get_areas_structure()
                metadata["areas_notes_count"] = areas_structure.get("total_files", 0)
                metadata["areas_subdirectories"] = areas_structure.get("subdirectories", [])
                metadata["areas_total_size_bytes"] = areas_structure.get("total_size_bytes", 0)
                
                # Create filtering summary message
                excluded_count = len(metadata["excluded_folders"])
                if excluded_count > 0:
                    metadata["filtering_summary"] = (
                        f"Privacy filtering active: {excluded_count} folders excluded "
                        f"({', '.join(metadata['excluded_folders'][:3])}"
                        f"{', ...' if excluded_count > 3 else ''})"
                    )
                    metadata["content_protection_level"] = "high"
                else:
                    metadata["filtering_summary"] = "Areas/ filtering active (no other folders found)"
                    metadata["content_protection_level"] = "medium"
                
                # Add privacy messaging
                metadata["privacy_message"] = (
                    "Only content from the Areas/ folder is included in dataset generation. "
                    "Personal journals, people notes, and private content are excluded."
                )
                
            except Exception as e:
                logger.warning(f"Could not get filtering metadata: {e}")
                metadata["filtering_summary"] = "Areas/ filtering enabled (metadata unavailable)"
                metadata["content_protection_level"] = "unknown"
        else:
            metadata["filtering_summary"] = "Full vault processing - all content included"
            metadata["content_protection_level"] = "none"
            metadata["privacy_message"] = (
                "All vault content is included in dataset generation. "
                "Consider enabling Areas/ filtering for privacy protection."
            )
        
        return metadata

    def validate_vault(self) -> ValidationResult:
        """Validate vault structure and accessibility with comprehensive checks.
        
        Returns:
            ValidationResult with detailed validation status
        """
        result = ValidationResult(valid=True)
        logger.info(f"Starting comprehensive vault validation for: {self.vault_path}")

        # 1. Basic path validation
        if not self._validate_vault_path(result):
            return result

        # 2. Vault structure and content validation
        if not self._validate_vault_structure(result):
            return result

        # 3. Output directory validation
        if not self._validate_output_directory(result):
            return result

        # 4. Service dependencies validation
        self._validate_service_dependencies(result)

        # 5. Data quality pre-checks
        self._validate_data_quality_requirements(result)

        # 6. Memory and performance pre-checks
        self._validate_system_requirements(result)

        logger.info(f"Vault validation completed. Valid: {result.valid}, "
                   f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
        
        return result

    def _validate_vault_path(self, result: ValidationResult) -> bool:
        """Validate basic vault path requirements."""
        # Check if vault path exists
        if not self.vault_path.exists():
            result.valid = False
            result.errors.append(f"Vault path does not exist: {self.vault_path}")
            return False

        # Check if vault path is a directory
        if not self.vault_path.is_dir():
            result.valid = False
            result.errors.append(f"Vault path is not a directory: {self.vault_path}")
            return False

        # Check if vault path is readable
        try:
            # Test readability by trying to list directory contents
            list(self.vault_path.iterdir())
        except PermissionError:
            result.valid = False
            result.errors.append(f"Vault path is not readable: {self.vault_path}")
            return False

        # Check for common Obsidian vault indicators
        obsidian_config = self.vault_path / ".obsidian"
        if not obsidian_config.exists():
            result.warnings.append("No .obsidian directory found - this may not be an Obsidian vault")

        return True

    def _validate_vault_structure(self, result: ValidationResult) -> bool:
        """Validate vault structure and content requirements."""
        try:
            # Initialize vault reader with same filtering settings as the main instance
            vault_reader = VaultReader(str(self.vault_path), areas_only=self.areas_only)
            
            # If Areas filtering is enabled, validate Areas folder first
            if self.areas_only:
                try:
                    from jarvis.tools.dataset_generation.filters.areas_filter import AreasContentFilter
                    
                    # Get minimum content threshold from settings
                    settings = get_settings()
                    min_threshold = getattr(settings, 'dataset_min_areas_content', 5)
                    
                    areas_filter = AreasContentFilter(
                        str(self.vault_path), 
                        min_content_threshold=min_threshold
                    )
                    areas_validation = areas_filter.validate_areas_folder()
                    
                    logger.info(f"Areas folder validation: {areas_validation['markdown_file_count']} files found")
                    
                    # Populate comprehensive Areas-specific information in result
                    result.areas_folder_exists = areas_validation.get('areas_folder_exists', False)
                    result.areas_notes_count = areas_validation.get('markdown_file_count', 0)
                    result.areas_folder_path = areas_validation.get('areas_folder_path')
                    result.areas_folder_name = areas_validation.get('areas_folder_name')
                    result.areas_subdirectory_count = areas_validation.get('subdirectory_count', 0)
                    result.areas_total_size_bytes = areas_validation.get('total_size_bytes', 0)
                    result.min_content_threshold = min_threshold
                    result.areas_validation_passed = areas_validation.get('validation_passed', False)
                    result.filtering_mode = "areas_only"
                    
                    # Add informative messages about Areas filtering
                    if result.areas_validation_passed:
                        logger.info(f"Areas/ folder validation passed: {result.areas_notes_count} files, "
                                  f"{result.areas_subdirectory_count} subdirectories, "
                                  f"{result.areas_total_size_bytes / 1024:.1f}KB total")
                        
                        # Add success message to warnings for user visibility
                        result.warnings.append(
                            f"Areas/ filtering active: Processing {result.areas_notes_count} files "
                            f"from {result.areas_subdirectory_count} subdirectories - personal content excluded"
                        )
                    
                except Exception as e:
                    # Handle Areas-specific validation errors with detailed messages
                    result.valid = False
                    result.filtering_mode = "areas_only"
                    result.areas_validation_passed = False
                    
                    # Provide specific error messages based on exception type
                    from jarvis.tools.dataset_generation.models.exceptions import (
                        AreasNotFoundError, InsufficientAreasContentError
                    )
                    
                    if isinstance(e, AreasNotFoundError):
                        result.errors.append(
                            f"Areas/ folder not found: {e}. "
                            f"Please create an Areas/ folder in your vault and organize your "
                            f"knowledge content there for dataset generation."
                        )
                    elif isinstance(e, InsufficientAreasContentError):
                        result.errors.append(
                            f"Insufficient content in Areas/ folder: {e}. "
                            f"Please add more markdown files to Areas/ subdirectories "
                            f"(minimum {result.min_content_threshold} files required)."
                        )
                    else:
                        result.errors.append(f"Areas folder validation failed: {e}")
                    
                    return False
            else:
                result.filtering_mode = "full_vault"
                result.areas_validation_passed = True  # Not applicable for full vault mode
            
            markdown_files = list(vault_reader.get_markdown_files())

            # Check minimum file count with Areas-specific messaging
            if len(markdown_files) == 0:
                result.valid = False
                if self.areas_only:
                    result.errors.append(
                        "No markdown files found in Areas/ folder. "
                        "Please add knowledge content to Areas/ subdirectories for dataset generation."
                    )
                else:
                    result.errors.append("No markdown files found in vault")
                return False
            elif len(markdown_files) < result.min_content_threshold:
                if self.areas_only:
                    result.valid = False
                    result.errors.append(
                        f"Insufficient content in Areas/ folder: {len(markdown_files)} files found, "
                        f"but minimum {result.min_content_threshold} required for meaningful dataset generation. "
                        f"Please add more markdown files to the '{result.areas_folder_name}' folder or its subdirectories."
                    )
                    return False
                else:
                    result.warnings.append(
                        f"Very few markdown files found: {len(markdown_files)} "
                        f"(minimum {result.min_content_threshold} recommended for meaningful datasets)"
                    )

            result.notes_processed = len(markdown_files)
            
            # If Areas filtering is enabled, also count excluded files for reporting
            if self.areas_only:
                try:
                    # Count total files in vault for comparison
                    full_vault_reader = VaultReader(str(self.vault_path), areas_only=False)
                    all_files = list(full_vault_reader.get_markdown_files())
                    result.excluded_notes_count = len(all_files) - len(markdown_files)
                    
                    if result.excluded_notes_count > 0:
                        logger.info(f"Areas filtering: {len(markdown_files)} files included, {result.excluded_notes_count} files excluded")
                        result.warnings.append(
                            f"Privacy mode active: {result.excluded_notes_count} files excluded from processing "
                            f"(only Areas/ content included)"
                        )
                    else:
                        logger.info("Areas filtering: All vault content is within Areas/ folder")
                        
                except Exception as e:
                    logger.warning(f"Could not count excluded files: {e}")
                    result.excluded_notes_count = 0

            # Check for reasonable file sizes
            total_size = 0
            empty_files = 0
            large_files = 0
            
            for file_path in markdown_files[:min(100, len(markdown_files))]:  # Sample first 100 files
                try:
                    full_path = vault_reader.get_absolute_path(str(file_path))
                    file_size = full_path.stat().st_size
                    total_size += file_size
                    
                    if file_size == 0:
                        empty_files += 1
                    elif file_size > 1024 * 1024:  # 1MB
                        large_files += 1
                        
                except Exception as e:
                    logger.warning(f"Could not check file size for {file_path}: {e}")

            # Report file size statistics
            if empty_files > len(markdown_files) * 0.5:
                result.warnings.append(f"Many empty files detected: {empty_files}/{len(markdown_files)}")
            
            if large_files > 0:
                result.warnings.append(f"Large files detected: {large_files} files > 1MB (may impact performance)")

            avg_size = total_size / min(100, len(markdown_files)) if markdown_files else 0
            logger.info(f"Vault statistics: {len(markdown_files)} files, average size: {avg_size:.0f} bytes")

            # Test reading a sample of files
            sample_size = min(10, len(markdown_files))
            readable_files = 0
            
            for file_path in markdown_files[:sample_size]:
                try:
                    content, metadata = vault_reader.read_file(str(file_path))
                    if content is not None:
                        readable_files += 1
                except Exception as e:
                    logger.warning(f"Could not read sample file {file_path}: {e}")

            if readable_files < sample_size * 0.8:
                result.valid = False
                result.errors.append(f"Many files are unreadable: {sample_size - readable_files}/{sample_size} failed")
                return False

        except Exception as e:
            result.valid = False
            result.errors.append(f"Failed to access vault structure: {e}")
            return False

        return True

    def _validate_output_directory(self, result: ValidationResult) -> bool:
        """Validate output directory requirements."""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = self.output_dir / "test_write.tmp"
            test_content = "test dataset generation write permissions"
            test_file.write_text(test_content)
            
            # Verify we can read it back
            read_content = test_file.read_text()
            if read_content != test_content:
                result.valid = False
                result.errors.append("Output directory write/read verification failed")
                return False
                
            test_file.unlink()

            # Check available disk space
            import shutil
            free_space = shutil.disk_usage(self.output_dir).free
            if free_space < 100 * 1024 * 1024:  # 100MB minimum
                result.warnings.append(f"Low disk space in output directory: {free_space / (1024*1024):.1f}MB available")

        except Exception as e:
            result.valid = False
            result.errors.append(f"Cannot write to output directory: {e}")
            return False

        return True

    def _validate_service_dependencies(self, result: ValidationResult):
        """Validate service dependencies and configurations."""
        # Check VaultReader
        try:
            if not hasattr(self, 'vault_reader') or self.vault_reader is None:
                result.errors.append("VaultReader service not initialized")
                result.valid = False
        except Exception as e:
            result.errors.append(f"VaultReader validation failed: {e}")
            result.valid = False

        # Check VectorEncoder
        try:
            if not hasattr(self, 'vector_encoder') or self.vector_encoder is None:
                result.errors.append("VectorEncoder service not initialized")
                result.valid = False
            else:
                # Test encoding capability
                test_text = "test encoding capability"
                try:
                    embedding = self.vector_encoder.encode([test_text])
                    if embedding is None or len(embedding) == 0:
                        result.warnings.append("VectorEncoder test encoding returned empty result")
                except Exception as e:
                    result.warnings.append(f"VectorEncoder test failed: {e}")
        except Exception as e:
            result.errors.append(f"VectorEncoder validation failed: {e}")
            result.valid = False

        # Check GraphDatabase (optional)
        if hasattr(self, 'graph_database') and self.graph_database is not None:
            try:
                # Test basic graph database connectivity
                logger.info("GraphDatabase service available - enhanced features enabled")
            except Exception as e:
                result.warnings.append(f"GraphDatabase available but test failed: {e}")
        else:
            result.warnings.append("GraphDatabase not available - some features will be limited")

    def _validate_data_quality_requirements(self, result: ValidationResult):
        """Validate data quality requirements for meaningful dataset generation."""
        try:
            # Sample a few files to check content quality
            markdown_files = list(self.vault_reader.get_markdown_files())
            sample_size = min(20, len(markdown_files))
            
            files_with_content = 0
            files_with_links = 0
            files_with_metadata = 0
            total_word_count = 0
            
            for file_path in markdown_files[:sample_size]:
                try:
                    content, metadata = self.vault_reader.read_file(str(file_path))
                    
                    if content and len(content.strip()) > 50:  # Minimum meaningful content
                        files_with_content += 1
                        total_word_count += len(content.split())
                    
                    if content and ('[[' in content or '](' in content):
                        files_with_links += 1
                    
                    if metadata and len(metadata) > 0:
                        files_with_metadata += 1
                        
                except Exception as e:
                    logger.warning(f"Could not analyze sample file {file_path}: {e}")

            # Quality checks
            content_ratio = files_with_content / sample_size if sample_size > 0 else 0
            if content_ratio < 0.5:
                result.warnings.append(f"Many files have minimal content: {files_with_content}/{sample_size} have substantial content")

            link_ratio = files_with_links / sample_size if sample_size > 0 else 0
            if link_ratio < 0.2:
                result.warnings.append(f"Few files contain links: {files_with_links}/{sample_size} have links (may limit pair generation)")

            metadata_ratio = files_with_metadata / sample_size if sample_size > 0 else 0
            if metadata_ratio < 0.3:
                result.warnings.append(f"Few files have metadata: {files_with_metadata}/{sample_size} have frontmatter")

            avg_words = total_word_count / files_with_content if files_with_content > 0 else 0
            if avg_words < 50:
                result.warnings.append(f"Average content length is low: {avg_words:.0f} words per file")

            logger.info(f"Data quality sample: {content_ratio:.1%} substantial content, "
                       f"{link_ratio:.1%} with links, {metadata_ratio:.1%} with metadata")

        except Exception as e:
            result.warnings.append(f"Could not perform data quality validation: {e}")

    def _validate_system_requirements(self, result: ValidationResult):
        """Validate system requirements for dataset generation."""
        try:
            import psutil
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 1.0:
                result.warnings.append(f"Low available memory: {available_gb:.1f}GB (may impact performance)")
            elif available_gb < 0.5:
                result.errors.append(f"Insufficient memory: {available_gb:.1f}GB (minimum 0.5GB required)")
                result.valid = False

            # Check CPU count for parallel processing
            cpu_count = psutil.cpu_count()
            if cpu_count and cpu_count < 2:
                result.warnings.append("Single CPU core detected - parallel processing will be limited")

            # Estimate processing requirements
            estimated_memory_mb = result.notes_processed * 0.1  # Rough estimate: 0.1MB per note
            if estimated_memory_mb > available_gb * 1024 * 0.8:  # Use max 80% of available memory
                result.warnings.append(f"Large vault may require {estimated_memory_mb:.0f}MB memory "
                                     f"(consider using smaller batch sizes)")

        except ImportError:
            result.warnings.append("psutil not available - cannot check system requirements")
        except Exception as e:
            result.warnings.append(f"System requirements check failed: {e}")

    def validate_dataset_quality(self, notes_dataset_path: str | None = None, 
                               pairs_dataset_path: str | None = None) -> ValidationResult:
        """Validate the quality of generated datasets.
        
        Args:
            notes_dataset_path: Path to notes dataset CSV file
            pairs_dataset_path: Path to pairs dataset CSV file
            
        Returns:
            ValidationResult with comprehensive quality assessment
        """
        logger.info("Starting comprehensive dataset quality validation")
        result = ValidationResult(valid=True)
        
        try:
            # Validate notes dataset if provided
            if notes_dataset_path:
                notes_validation = self._validate_notes_dataset_quality(notes_dataset_path)
                result.errors.extend(notes_validation.errors)
                result.warnings.extend(notes_validation.warnings)
                if not notes_validation.valid:
                    result.valid = False
                    
            # Validate pairs dataset if provided
            if pairs_dataset_path:
                pairs_validation = self._validate_pairs_dataset_quality(pairs_dataset_path)
                result.errors.extend(pairs_validation.errors)
                result.warnings.extend(pairs_validation.warnings)
                if not pairs_validation.valid:
                    result.valid = False
                    
            # Cross-dataset validation if both are provided
            if notes_dataset_path and pairs_dataset_path:
                cross_validation = self._validate_cross_dataset_consistency(
                    notes_dataset_path, pairs_dataset_path
                )
                result.errors.extend(cross_validation.errors)
                result.warnings.extend(cross_validation.warnings)
                if not cross_validation.valid:
                    result.valid = False
                    
            logger.info(f"Dataset quality validation completed. Valid: {result.valid}, "
                       f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
                       
        except Exception as e:
            logger.error(f"Dataset quality validation failed: {e}")
            result.valid = False
            result.errors.append(f"Quality validation failed: {e}")
            
        return result

    def _validate_notes_dataset_quality(self, dataset_path: str) -> ValidationResult:
        """Validate notes dataset quality and completeness.
        
        Args:
            dataset_path: Path to notes dataset CSV file
            
        Returns:
            ValidationResult for notes dataset
        """
        result = ValidationResult(valid=True)
        
        try:
            import pandas as pd
            
            # Load dataset
            try:
                df = pd.read_csv(dataset_path)
                logger.info(f"Loaded notes dataset: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                result.valid = False
                result.errors.append(f"Failed to load notes dataset: {e}")
                return result
                
            # Check basic structure
            if len(df) == 0:
                result.valid = False
                result.errors.append("Notes dataset is empty")
                return result
                
            # Check required columns
            required_columns = [
                'note_path', 'note_title', 'word_count', 'tag_count',
                'quality_stage', 'outgoing_links_count', 'incoming_links_count'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                result.valid = False
                result.errors.append(f"Missing required columns: {missing_columns}")
                
            # Check data types and ranges
            if 'word_count' in df.columns:
                if df['word_count'].dtype not in ['int64', 'float64']:
                    result.warnings.append("word_count should be numeric")
                if (df['word_count'] < 0).any():
                    result.errors.append("word_count contains negative values")
                    result.valid = False
                    
            if 'tag_count' in df.columns:
                if df['tag_count'].dtype not in ['int64', 'float64']:
                    result.warnings.append("tag_count should be numeric")
                if (df['tag_count'] < 0).any():
                    result.errors.append("tag_count contains negative values")
                    result.valid = False
                    
            # Check for missing values in critical columns
            critical_columns = ['note_path', 'note_title']
            for col in critical_columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        result.warnings.append(f"{col} has {null_count} null values")
                        if null_count > len(df) * 0.1:  # More than 10% null
                            result.valid = False
                            result.errors.append(f"Too many null values in {col}: {null_count}/{len(df)}")
                            
            # Check for duplicates
            if 'note_path' in df.columns:
                duplicate_count = df['note_path'].duplicated().sum()
                if duplicate_count > 0:
                    result.warnings.append(f"Found {duplicate_count} duplicate note paths")
                    
            # Check data distribution
            if 'word_count' in df.columns:
                word_count_stats = df['word_count'].describe()
                if word_count_stats['mean'] < 10:
                    result.warnings.append(f"Very low average word count: {word_count_stats['mean']:.1f}")
                if word_count_stats['std'] == 0:
                    result.warnings.append("No variation in word count (all notes same length)")
                    
            # Check quality stage distribution
            if 'quality_stage' in df.columns:
                quality_dist = df['quality_stage'].value_counts()
                if len(quality_dist) == 1:
                    result.warnings.append("All notes have the same quality stage")
                unknown_ratio = quality_dist.get('unknown', 0) / len(df)
                if unknown_ratio > 0.8:
                    result.warnings.append(f"High ratio of unknown quality stages: {unknown_ratio:.1%}")
                    
            # Check centrality metrics if present
            centrality_columns = ['betweenness_centrality', 'closeness_centrality', 'pagerank_score']
            for col in centrality_columns:
                if col in df.columns:
                    if (df[col] < 0).any() or (df[col] > 1).any():
                        result.warnings.append(f"{col} values outside expected range [0,1]")
                        
            result.notes_processed = len(df)
            
        except Exception as e:
            logger.error(f"Notes dataset validation failed: {e}")
            result.valid = False
            result.errors.append(f"Notes dataset validation error: {e}")
            
        return result

    def _validate_pairs_dataset_quality(self, dataset_path: str) -> ValidationResult:
        """Validate pairs dataset quality and completeness.
        
        Args:
            dataset_path: Path to pairs dataset CSV file
            
        Returns:
            ValidationResult for pairs dataset
        """
        result = ValidationResult(valid=True)
        
        try:
            import pandas as pd
            
            # Load dataset
            try:
                df = pd.read_csv(dataset_path)
                logger.info(f"Loaded pairs dataset: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                result.valid = False
                result.errors.append(f"Failed to load pairs dataset: {e}")
                return result
                
            # Check basic structure
            if len(df) == 0:
                result.valid = False
                result.errors.append("Pairs dataset is empty")
                return result
                
            # Check required columns
            required_columns = [
                'note_a_path', 'note_b_path', 'link_exists', 'cosine_similarity',
                'tfidf_similarity', 'combined_similarity',
                'tag_overlap_count', 'tag_jaccard_similarity'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                result.valid = False
                result.errors.append(f"Missing required columns: {missing_columns}")
                
            # Check link_exists distribution
            if 'link_exists' in df.columns:
                link_dist = df['link_exists'].value_counts()
                positive_ratio = link_dist.get(True, 0) / len(df)
                negative_ratio = link_dist.get(False, 0) / len(df)
                
                if positive_ratio == 0:
                    result.valid = False
                    result.errors.append("No positive examples (links) in pairs dataset")
                elif positive_ratio > 0.8:
                    result.warnings.append(f"Very high positive ratio: {positive_ratio:.1%}")
                elif positive_ratio < 0.05:
                    result.warnings.append(f"Very low positive ratio: {positive_ratio:.1%}")
                    
                logger.info(f"Pairs dataset distribution: {positive_ratio:.1%} positive, {negative_ratio:.1%} negative")
                
            # Check similarity scores
            if 'cosine_similarity' in df.columns:
                sim_stats = df['cosine_similarity'].describe()
                if sim_stats['min'] < -1 or sim_stats['max'] > 1:
                    result.warnings.append("Cosine similarity values outside expected range [-1,1]")
                if sim_stats['std'] < 0.01:
                    result.warnings.append("Very low variation in cosine similarity scores")
                    
            # Check for self-pairs
            if 'note_a_path' in df.columns and 'note_b_path' in df.columns:
                self_pairs = (df['note_a_path'] == df['note_b_path']).sum()
                if self_pairs > 0:
                    result.warnings.append(f"Found {self_pairs} self-pairs (note paired with itself)")
                    
            # Check for duplicate pairs
            if 'note_a_path' in df.columns and 'note_b_path' in df.columns:
                # Create normalized pair representation for duplicate detection
                df_normalized = df.copy()
                df_normalized['pair_key'] = df_normalized.apply(
                    lambda row: tuple(sorted([row['note_a_path'], row['note_b_path']])), axis=1
                )
                duplicate_count = df_normalized['pair_key'].duplicated().sum()
                if duplicate_count > 0:
                    result.warnings.append(f"Found {duplicate_count} duplicate pairs")
                    
            # Check tag overlap consistency
            if 'tag_overlap_count' in df.columns and 'tag_jaccard_similarity' in df.columns:
                # Jaccard should be 0 when overlap is 0
                inconsistent = ((df['tag_overlap_count'] == 0) & (df['tag_jaccard_similarity'] > 0)).sum()
                if inconsistent > 0:
                    result.warnings.append(f"Found {inconsistent} inconsistent tag overlap/Jaccard pairs")
                    
            result.notes_processed = len(df)
            
        except Exception as e:
            logger.error(f"Pairs dataset validation failed: {e}")
            result.valid = False
            result.errors.append(f"Pairs dataset validation error: {e}")
            
        return result

    def _validate_cross_dataset_consistency(self, notes_path: str, pairs_path: str) -> ValidationResult:
        """Validate consistency between notes and pairs datasets.
        
        Args:
            notes_path: Path to notes dataset CSV file
            pairs_path: Path to pairs dataset CSV file
            
        Returns:
            ValidationResult for cross-dataset consistency
        """
        result = ValidationResult(valid=True)
        
        try:
            import pandas as pd
            
            # Load both datasets
            try:
                notes_df = pd.read_csv(notes_path)
                pairs_df = pd.read_csv(pairs_path)
            except Exception as e:
                result.valid = False
                result.errors.append(f"Failed to load datasets for cross-validation: {e}")
                return result
                
            # Check that all notes in pairs exist in notes dataset
            if 'note_path' in notes_df.columns and 'note_a_path' in pairs_df.columns and 'note_b_path' in pairs_df.columns:
                notes_set = set(notes_df['note_path'])
                pairs_notes_a = set(pairs_df['note_a_path'])
                pairs_notes_b = set(pairs_df['note_b_path'])
                all_pairs_notes = pairs_notes_a | pairs_notes_b
                
                missing_notes = all_pairs_notes - notes_set
                if missing_notes:
                    result.warnings.append(f"Found {len(missing_notes)} notes in pairs dataset not present in notes dataset")
                    if len(missing_notes) > len(all_pairs_notes) * 0.1:  # More than 10% missing
                        result.valid = False
                        result.errors.append(f"Too many missing notes: {len(missing_notes)}/{len(all_pairs_notes)}")
                        
                # Check coverage - what percentage of notes appear in pairs
                coverage = len(all_pairs_notes & notes_set) / len(notes_set) if notes_set else 0
                if coverage < 0.5:
                    result.warnings.append(f"Low coverage: only {coverage:.1%} of notes appear in pairs dataset")
                    
                logger.info(f"Cross-dataset coverage: {coverage:.1%} of notes appear in pairs")
                
            # Check feature consistency (e.g., word counts should be consistent)
            # This would require more complex validation based on specific features
            
        except Exception as e:
            logger.error(f"Cross-dataset validation failed: {e}")
            result.valid = False
            result.errors.append(f"Cross-dataset validation error: {e}")
            
        return result

    def generate_quality_report(self, notes_dataset_path: str | None = None,
                              pairs_dataset_path: str | None = None) -> dict[str, any]:
        """Generate comprehensive quality report for datasets.
        
        Args:
            notes_dataset_path: Path to notes dataset CSV file
            pairs_dataset_path: Path to pairs dataset CSV file
            
        Returns:
            Dictionary containing detailed quality metrics and statistics
        """
        logger.info("Generating comprehensive dataset quality report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': {},
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Validate datasets
            validation_result = self.validate_dataset_quality(notes_dataset_path, pairs_dataset_path)
            report['validation_results'] = {
                'overall_valid': validation_result.valid,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'notes_processed': validation_result.notes_processed
            }
            
            # Generate detailed statistics
            if notes_dataset_path:
                notes_stats = self._generate_notes_statistics(notes_dataset_path)
                report['statistics']['notes'] = notes_stats
                
            if pairs_dataset_path:
                pairs_stats = self._generate_pairs_statistics(pairs_dataset_path)
                report['statistics']['pairs'] = pairs_stats
                
            # Generate recommendations based on findings
            recommendations = self._generate_quality_recommendations(
                validation_result, report['statistics']
            )
            report['recommendations'] = recommendations
            
            logger.info("Quality report generation completed")
            
        except Exception as e:
            logger.error(f"Quality report generation failed: {e}")
            report['error'] = str(e)
            
        return report

    def _generate_notes_statistics(self, dataset_path: str) -> dict[str, any]:
        """Generate detailed statistics for notes dataset."""
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            
            stats = {
                'total_notes': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            # Numeric column statistics
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                stats[f'{col}_stats'] = df[col].describe().to_dict()
                
            # Categorical column statistics
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in ['note_path', 'note_title']:
                    continue  # Skip path/title columns
                value_counts = df[col].value_counts().head(10).to_dict()
                stats[f'{col}_distribution'] = value_counts
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate notes statistics: {e}")
            return {'error': str(e)}

    def _generate_pairs_statistics(self, dataset_path: str) -> dict[str, any]:
        """Generate detailed statistics for pairs dataset."""
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            
            stats = {
                'total_pairs': len(df),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            # Link distribution
            if 'link_exists' in df.columns:
                link_dist = df['link_exists'].value_counts().to_dict()
                stats['link_distribution'] = link_dist
                stats['positive_ratio'] = link_dist.get(True, 0) / len(df)
                
            # Similarity score distribution
            if 'cosine_similarity' in df.columns:
                stats['cosine_similarity_stats'] = df['cosine_similarity'].describe().to_dict()
                
            # Tag overlap statistics
            if 'tag_overlap_count' in df.columns:
                stats['tag_overlap_stats'] = df['tag_overlap_count'].describe().to_dict()
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate pairs statistics: {e}")
            return {'error': str(e)}

    def _generate_quality_recommendations(self, validation_result: ValidationResult,
                                        statistics: dict[str, any]) -> list[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Recommendations based on validation errors
        if not validation_result.valid:
            recommendations.append("Address validation errors before using datasets for training")
            
        # Recommendations based on statistics
        if 'notes' in statistics:
            notes_stats = statistics['notes']
            if 'word_count_stats' in notes_stats:
                avg_words = notes_stats['word_count_stats'].get('mean', 0)
                if avg_words < 50:
                    recommendations.append("Consider filtering out very short notes (< 50 words) for better quality")
                    
        if 'pairs' in statistics:
            pairs_stats = statistics['pairs']
            positive_ratio = pairs_stats.get('positive_ratio', 0)
            if positive_ratio < 0.1:
                recommendations.append("Consider increasing positive examples ratio for better model training")
            elif positive_ratio > 0.5:
                recommendations.append("Consider balancing dataset with more negative examples")
                
        # General recommendations
        if validation_result.warnings:
            recommendations.append("Review and address validation warnings for optimal dataset quality")
            
        return recommendations

    def get_generation_summary(self) -> GenerationSummary | None:
        """Provide comprehensive summary of generation process.
        
        Returns:
            GenerationSummary from last generation run or None
        """
        # This would typically be stored from the last run
        # For now, return None - could be enhanced to store state
        return None

    def cleanup(self):
        """Clean up resources and close connections."""
        try:
            if hasattr(self, 'graph_database') and self.graph_database:
                self.graph_database.close()
                logger.info("GraphDatabase connection closed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
