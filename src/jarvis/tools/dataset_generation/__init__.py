"""
Dataset Generation Tool for Jarvis Assistant.

This module provides comprehensive dataset generation capabilities for machine learning
analysis of Obsidian vaults. It creates two complementary datasets:
1. Notes dataset: Individual note characteristics and properties
2. Pairs dataset: Comparative analysis and relationship modeling

The tool integrates with existing Jarvis services (VaultReader, VectorEncoder, GraphDatabase)
and follows the established architecture patterns for maintainability.
"""

from .dataset_generator import DatasetGenerator
from .extractors.link_extractor import LinkExtractor
from .generators.notes_dataset_generator import NotesDatasetGenerator
from .generators.pairs_dataset_generator import (
    PairsDatasetGenerator,
    RandomSamplingStrategy,
    StratifiedSamplingStrategy,
)
from .models.data_models import (
    CentralityMetrics,
    DatasetGenerationResult,
    GenerationSummary,
    Link,
    LinkStatistics,
    NoteData,
    NoteFeatures,
    PairFeatures,
    ProcessingStats,
    ValidationResult,
)
from .models.exceptions import (
    DataQualityError,
    DatasetGenerationError,
    FeatureEngineeringError,
    LinkExtractionError,
    SamplingError,
    VaultValidationError,
)

__all__ = [
    # Components
    'DatasetGenerator',
    'LinkExtractor',
    'NotesDatasetGenerator',
    'PairsDatasetGenerator',
    'RandomSamplingStrategy',
    'StratifiedSamplingStrategy',
    
    # Data models
    'CentralityMetrics',
    'DatasetGenerationResult',
    'GenerationSummary',
    'Link',
    'LinkStatistics',
    'NoteData',
    'NoteFeatures',
    'PairFeatures',
    'ProcessingStats',
    'ValidationResult',
    
    # Exceptions
    'DataQualityError',
    'DatasetGenerationError',
    'FeatureEngineeringError',
    'LinkExtractionError',
    'SamplingError',
    'VaultValidationError',
]