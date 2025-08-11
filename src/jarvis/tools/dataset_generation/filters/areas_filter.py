"""
Areas content filter for dataset generation.

This module provides filtering functionality to process only content from
the Areas/ folder in an Obsidian vault, supporting privacy-focused dataset
generation by excluding personal content from other folders.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from jarvis.utils.config import get_settings
from jarvis.utils.errors import ValidationError
from jarvis.tools.dataset_generation.models.exceptions import (
    VaultValidationError,
    InsufficientDataError,
    ConfigurationError,
    AreasNotFoundError,
    InsufficientAreasContentError
)

logger = logging.getLogger(__name__)


class AreasContentFilter:
    """Filter for processing only Areas/ folder content in dataset generation.
    
    This filter implements privacy-focused content filtering by restricting
    dataset generation to only the Areas/ folder, which typically contains
    work-related or public content, excluding personal journals, people notes,
    and other private content.
    """
    
    def __init__(self, vault_path: str, areas_folder_name: Optional[str] = None,
                 min_content_threshold: int = 5):
        """Initialize the Areas content filter.
        
        Args:
            vault_path: Path to the Obsidian vault
            areas_folder_name: Name of the Areas folder (from config if None)
            min_content_threshold: Minimum number of files required in Areas/
        """
        self.vault_path = Path(vault_path).resolve()
        self.min_content_threshold = min_content_threshold
        
        # Get areas folder name from config if not provided
        if areas_folder_name is None:
            settings = get_settings()
            areas_folder_name = settings.dataset_areas_folder_name
        
        self.areas_folder_name = areas_folder_name
        self.areas_folder_path = self.vault_path / self.areas_folder_name
        
        logger.info(f"AreasContentFilter initialized for vault: {self.vault_path}")
        logger.info(f"Areas folder: {self.areas_folder_name} -> {self.areas_folder_path}")
    
    def validate_areas_folder(self) -> Dict[str, Any]:
        """Validate that the Areas folder exists and has sufficient content.
        
        Returns:
            Dictionary with validation results and metadata
            
        Raises:
            AreasNotFoundError: If Areas folder doesn't exist
            InsufficientAreasContentError: If Areas folder has insufficient content
        """
        validation_result = {
            "areas_folder_exists": False,
            "areas_folder_path": str(self.areas_folder_path),
            "areas_folder_name": self.areas_folder_name,
            "markdown_file_count": 0,
            "subdirectory_count": 0,
            "total_size_bytes": 0,
            "validation_passed": False,
            "error_message": None
        }
        
        # Check if vault path exists
        if not self.vault_path.exists():
            error_msg = f"Vault path does not exist: {self.vault_path}"
            validation_result["error_message"] = error_msg
            raise VaultValidationError(error_msg, str(self.vault_path), "vault_not_found")
        
        # Check if Areas folder exists
        if not self.areas_folder_path.exists():
            error_msg = (f"Areas folder '{self.areas_folder_name}' not found in vault. "
                        f"Expected path: {self.areas_folder_path}")
            validation_result["error_message"] = error_msg
            raise AreasNotFoundError(
                str(self.vault_path), 
                self.areas_folder_name
            )
        
        if not self.areas_folder_path.is_dir():
            error_msg = f"Areas path exists but is not a directory: {self.areas_folder_path}"
            validation_result["error_message"] = error_msg
            raise AreasNotFoundError(
                str(self.vault_path),
                self.areas_folder_name
            )
        
        validation_result["areas_folder_exists"] = True
        
        # Count markdown files in Areas folder
        markdown_files = list(self.areas_folder_path.rglob("*.md"))
        # Filter out hidden files and system files
        markdown_files = [f for f in markdown_files if not f.name.startswith('.')]
        
        validation_result["markdown_file_count"] = len(markdown_files)
        
        # Count subdirectories
        subdirs = [d for d in self.areas_folder_path.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
        validation_result["subdirectory_count"] = len(subdirs)
        
        # Calculate total size
        total_size = 0
        for md_file in markdown_files:
            try:
                total_size += md_file.stat().st_size
            except (OSError, IOError):
                # Skip files that can't be accessed
                pass
        validation_result["total_size_bytes"] = total_size
        
        # Check minimum content threshold
        if len(markdown_files) < self.min_content_threshold:
            error_msg = (f"Insufficient content in Areas folder. "
                        f"Found {len(markdown_files)} markdown files, "
                        f"but need at least {self.min_content_threshold} for dataset generation.")
            validation_result["error_message"] = error_msg
            raise InsufficientAreasContentError(
                str(self.areas_folder_path),
                len(markdown_files),
                self.min_content_threshold
            )
        
        validation_result["validation_passed"] = True
        logger.info(f"Areas folder validation passed: {len(markdown_files)} files found")
        
        return validation_result
    
    def is_areas_content(self, file_path: str) -> bool:
        """Check if a file path is within the Areas folder.
        
        Args:
            file_path: File path to check (relative to vault root)
            
        Returns:
            True if file is within Areas folder, False otherwise
        """
        # Normalize path for comparison
        normalized_path = str(file_path).replace("\\", "/").strip("/")
        areas_prefix = self.areas_folder_name.replace("\\", "/").strip("/")
        
        # Check if path is in Areas folder or its subfolders
        return (normalized_path.startswith(f"{areas_prefix}/") or 
                normalized_path == areas_prefix)
    
    def filter_file_paths(self, file_paths: List[Path]) -> List[Path]:
        """Filter a list of file paths to include only Areas content.
        
        Args:
            file_paths: List of file paths (relative to vault root)
            
        Returns:
            Filtered list containing only Areas content
        """
        filtered_paths = []
        
        for file_path in file_paths:
            # Convert to string for path checking
            path_str = str(file_path)
            
            if self.is_areas_content(path_str):
                filtered_paths.append(file_path)
        
        logger.debug(f"Filtered {len(file_paths)} paths to {len(filtered_paths)} Areas paths")
        return filtered_paths
    
    def get_areas_structure(self) -> Dict[str, Any]:
        """Get detailed structure information about the Areas folder.
        
        Returns:
            Dictionary with Areas folder structure information
        """
        if not self.areas_folder_path.exists():
            return {
                "exists": False,
                "path": str(self.areas_folder_path),
                "error": "Areas folder not found"
            }
        
        structure = {
            "exists": True,
            "path": str(self.areas_folder_path),
            "name": self.areas_folder_name,
            "subdirectories": [],
            "markdown_files": [],
            "total_files": 0,
            "total_size_bytes": 0
        }
        
        try:
            # Get all markdown files
            markdown_files = list(self.areas_folder_path.rglob("*.md"))
            markdown_files = [f for f in markdown_files if not f.name.startswith('.')]
            
            # Convert to relative paths from vault root
            relative_files = []
            total_size = 0
            
            for md_file in markdown_files:
                try:
                    rel_path = md_file.relative_to(self.vault_path)
                    relative_files.append(str(rel_path))
                    total_size += md_file.stat().st_size
                except (ValueError, OSError):
                    # Skip files that can't be processed
                    continue
            
            structure["markdown_files"] = relative_files
            structure["total_files"] = len(relative_files)
            structure["total_size_bytes"] = total_size
            
            # Get subdirectories
            subdirs = []
            for item in self.areas_folder_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    try:
                        rel_path = item.relative_to(self.vault_path)
                        subdirs.append(str(rel_path))
                    except ValueError:
                        continue
            
            structure["subdirectories"] = subdirs
            
        except Exception as e:
            structure["error"] = f"Error analyzing Areas structure: {e}"
            logger.error(f"Error analyzing Areas structure: {e}")
        
        return structure
    
    def get_exclusion_summary(self) -> Dict[str, Any]:
        """Get summary of what content will be excluded by Areas filtering.
        
        Returns:
            Dictionary with exclusion summary information
        """
        summary = {
            "filtering_enabled": True,
            "areas_folder_name": self.areas_folder_name,
            "areas_folder_path": str(self.areas_folder_path),
            "excluded_folders": [],
            "privacy_note": "Only content from the Areas folder will be included in dataset generation"
        }
        
        try:
            # Find top-level folders that will be excluded
            excluded_folders = []
            
            for item in self.vault_path.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and 
                    item.name != self.areas_folder_name):
                    excluded_folders.append(item.name)
            
            summary["excluded_folders"] = excluded_folders
            summary["excluded_folder_count"] = len(excluded_folders)
            
        except Exception as e:
            summary["error"] = f"Error analyzing exclusions: {e}"
            logger.error(f"Error analyzing exclusions: {e}")
        
        return summary
    
    def __str__(self) -> str:
        """String representation of the filter."""
        return f"AreasContentFilter(vault={self.vault_path}, areas={self.areas_folder_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the filter."""
        return (f"AreasContentFilter(vault_path='{self.vault_path}', "
                f"areas_folder_name='{self.areas_folder_name}', "
                f"min_content_threshold={self.min_content_threshold})")