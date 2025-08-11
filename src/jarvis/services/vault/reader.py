"""
Vault Reader service for Obsidian filesystem operations.

This module handles all interactions with the Obsidian vault filesystem,
including secure file access, directory traversal, and content search.
"""

import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from jarvis.utils.logging import setup_logging
from jarvis.utils.config import get_settings
from jarvis.utils.errors import JarvisError, ServiceError, ConfigurationError, ValidationError
from jarvis.core.interfaces import IVaultReader

logger = setup_logging(__name__)


class VaultReader(IVaultReader):
    """Service for accessing and managing Obsidian vault files."""
    
    def __init__(self, vault_path: Optional[str] = None, areas_only: Optional[bool] = None):
        """Initialize the vault reader.
        
        Args:
            vault_path: Path to the Obsidian vault (from settings if None)
            areas_only: Whether to filter content to Areas/ folder only (from settings if None)
        """
        if vault_path is None:
            settings = get_settings()
            vault_path = settings.vault_path
        
        if not vault_path:
            raise ConfigurationError("No vault path provided and none configured in settings")
        
        self.vault_path = Path(vault_path).resolve()
        
        if not self.vault_path.exists():
            raise ConfigurationError(f"Vault path not found: {self.vault_path}")
        
        if not self.vault_path.is_dir():
            raise ConfigurationError(f"Vault path is not a directory: {self.vault_path}")
        
        # Initialize Areas filtering
        settings = get_settings()
        if areas_only is None:
            # Get default from settings, defaulting to False for backward compatibility
            areas_only = getattr(settings, 'dataset_areas_only', False)
        
        self.areas_only = areas_only
        self.areas_filter = None
        
        if self.areas_only:
            try:
                # Lazy import to avoid circular imports
                from jarvis.tools.dataset_generation.filters.areas_filter import AreasContentFilter
                self.areas_filter = AreasContentFilter(str(self.vault_path))
                logger.info(f"VaultReader initialized with Areas/ filtering enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Areas filter: {e}")
                # Fall back to no filtering to maintain functionality
                self.areas_only = False
                self.areas_filter = None
        
        logger.info(f"VaultReader initialized with path: {self.vault_path}, areas_only: {self.areas_only}")
    
    def is_within_vault(self, path: Path) -> bool:
        """Check if a path is within the vault.
        
        Args:
            path: Path to check
        
        Returns:
            True if path is within vault, False otherwise
        """
        try:
            resolved_path = path.resolve()
            return self.vault_path in resolved_path.parents or resolved_path == self.vault_path
        except (OSError, ValueError) as e:
            raise ServiceError(f"Error resolving path {path}: {e}") from e
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert a vault-relative path to an absolute path.
        
        Args:
            relative_path: Path relative to vault root
        
        Returns:
            Absolute path
        """
        # Clean the relative path
        clean_rel_path = str(relative_path).lstrip("/")
        absolute_path = self.vault_path / clean_rel_path
        
        # Security check
        if not self.is_within_vault(absolute_path):
            raise ValidationError(f"Path outside vault: {relative_path}")
        
        return absolute_path
    
    def get_relative_path(self, absolute_path: Path) -> Path:
        """Convert an absolute path to a vault-relative path.
        
        Args:
            absolute_path: Absolute path
        
        Returns:
            Path relative to vault root
        """
        resolved_path = absolute_path.resolve()
        
        # Security check
        if not self.is_within_vault(resolved_path):
            raise ValidationError(f"Path outside vault: {absolute_path}")
        
        return resolved_path.relative_to(self.vault_path)
    
    def read_file(self, relative_path: str) -> Tuple[str, Dict[str, Any]]:
        """Read a file from the vault.
        
        Args:
            relative_path: Path relative to vault root
        
        Returns:
            Tuple of (content, metadata)
        """
        if self.is_excluded_path(relative_path):
            raise PermissionError(f"Access to this path is restricted: {relative_path}")
        
        absolute_path = self.get_absolute_path(relative_path)
        
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        
        if not absolute_path.is_file():
            raise ValidationError(f"Not a file: {relative_path}")
        
        # Read file with fallback encoding
        try:
            content = absolute_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = absolute_path.read_text(encoding='latin-1')
                logger.warning(f"File {relative_path} read with latin-1 encoding")
            except Exception as e:
                logger.error(f"Failed to read file {relative_path}: {e}")
                raise ServiceError(f"Cannot read file with supported encodings: {relative_path}") from e
        
        # Get file metadata
        stats = absolute_path.stat()
        metadata = {
            "path": relative_path,
            "size": stats.st_size,
            "created": stats.st_ctime,
            "modified": stats.st_mtime,
            "created_formatted": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_formatted": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "checksum": self._calculate_checksum(absolute_path)
        }
        
        return content, metadata
    
    def write_file(self, relative_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file in the vault.
        
        Args:
            relative_path: Path relative to vault root
            content: Content to write
        
        Returns:
            File metadata
        """
        if self.is_excluded_path(relative_path):
            raise PermissionError(f"Access to this path is restricted: {relative_path}")
        
        absolute_path = self.get_absolute_path(relative_path)
        
        # Create directory if it doesn't exist
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            absolute_path.write_text(content, encoding='utf-8')
            
            # Get updated metadata
            stats = absolute_path.stat()
            metadata = {
                "path": relative_path,
                "size": stats.st_size,
                "created": stats.st_ctime,
                "modified": stats.st_mtime,
                "created_formatted": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_formatted": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "checksum": self._calculate_checksum(absolute_path)
            }
            
            logger.debug(f"File written: {relative_path}")
            return metadata
        except Exception as e:
            logger.error(f"Error writing file {relative_path}: {e}")
            raise ServiceError(f"Error writing file {relative_path}: {e}") from e
    
    def get_vault_structure(self, relative_path: str = "", max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get the structure of the vault or a subdirectory.
        
        Args:
            relative_path: Path relative to vault root
            max_depth: Maximum depth to traverse
        
        Returns:
            List of items in the directory
        """
        absolute_path = self.get_absolute_path(relative_path)
        
        if not absolute_path.exists():
            raise FileNotFoundError(f"Path not found: {relative_path}")
        
        if not absolute_path.is_dir():
            raise ValidationError(f"Not a directory: {relative_path}")
        
        def build_tree(current_path: Path, depth: int = 0) -> List[Dict[str, Any]]:
            """Recursively build directory structure."""
            if depth > max_depth:
                return []
            
            items = []
            
            try:
                for item_path in current_path.iterdir():
                    # Skip .obsidian and hidden files
                    if item_path.name.startswith('.'):
                        continue
                    
                    # Check if excluded
                    rel_item_path = self.get_relative_path(item_path)
                    if self.is_excluded_path(str(rel_item_path)):
                        continue
                    
                    stats = item_path.stat()
                    is_dir = item_path.is_dir()
                    is_md = item_path.suffix.lower() == '.md'
                    
                    vault_item = {
                        "path": str(rel_item_path),
                        "name": item_path.name,
                        "type": 'directory' if is_dir else 'file',
                        "is_markdown": is_md if not is_dir else None,
                        "size": stats.st_size if not is_dir else None,
                        "modified": stats.st_mtime,
                        "modified_formatted": datetime.fromtimestamp(stats.st_mtime).isoformat()
                    }
                    
                    # Recursively process subdirectories
                    if is_dir and depth < max_depth:
                        try:
                            children = build_tree(item_path, depth + 1)
                            vault_item["children"] = children
                        except Exception as e:
                            logger.warning(f"Error accessing directory {item_path}: {e}")
                            vault_item["children"] = []
                    
                    items.append(vault_item)
                    
            except Exception as e:
                logger.error(f"Error listing directory {current_path}: {e}")
            
            return sorted(items, key=lambda x: (x["type"] == "file", x["name"].lower()))
        
        return build_tree(absolute_path)
    
    def search_vault(self, query: str, search_content: bool = False, 
                    limit: int = 20) -> List[Dict[str, Any]]:
        """Search the vault for files matching the query.
        
        Args:
            query: Search term
            search_content: Whether to search in file content
            limit: Maximum number of results to return
        
        Returns:
            List of matching files
        """
        results = []
        query_lower = query.lower()
        
        def search_recursively(directory: Path) -> List[Dict[str, Any]]:
            """Recursively search directory."""
            items = []
            
            if len(results) >= limit:
                return items
            
            try:
                for item_path in directory.iterdir():
                    # Skip .obsidian and hidden files
                    if item_path.name.startswith('.'):
                        continue
                    
                    # Check if excluded
                    try:
                        rel_path = self.get_relative_path(item_path)
                        if self.is_excluded_path(str(rel_path)):
                            continue
                        
                        # Apply Areas/ filtering if enabled
                        if self.areas_only and self.areas_filter:
                            if not self.areas_filter.is_areas_content(str(rel_path)):
                                continue
                                
                    except ValueError as e:
                        raise ServiceError(f"Error getting relative path for {item_path}: {e}") from e
                    
                    # Check filename match
                    name_match = query_lower in item_path.name.lower()
                    
                    if item_path.is_dir():
                        # Recursively search subdirectories
                        subitems = search_recursively(item_path)
                        if name_match or subitems:
                            items.append({
                                "path": str(rel_path),
                                "name": item_path.name,
                                "type": 'directory',
                                "match_type": "name" if name_match else "children",
                                "children": subitems if subitems else None
                            })
                    elif item_path.is_file():
                        is_md = item_path.suffix.lower() == '.md'
                        content_match = False
                        content_preview = None
                        
                        # Search in content if requested and is markdown
                        if search_content and is_md:
                            try:
                                content = item_path.read_text(encoding='utf-8')
                                content_lower = content.lower()
                                content_match = query_lower in content_lower
                                
                                # Extract a snippet with the match
                                if content_match:
                                    pos = content_lower.find(query_lower)
                                    start = max(0, pos - 40)
                                    end = min(len(content), pos + len(query) + 40)
                                    content_preview = content[start:end].strip()
                            except (FileNotFoundError, PermissionError, OSError) as e:
                                logger.warning(f"Could not read file {item_path}: {e}")
                            except UnicodeDecodeError as e:
                                logger.warning(f"Could not decode file {item_path}: {e}")
                            except Exception as e:
                                logger.error(f"Unexpected error reading file {item_path}: {e}")
                                raise ServiceError(f"File read failed: {e}") from e
                        
                        if name_match or content_match:
                            items.append({
                                "path": str(rel_path),
                                "name": item_path.name,
                                "type": 'file',
                                "is_markdown": is_md,
                                "match_type": "name" if name_match else "content",
                                "content_preview": content_preview,
                                "size": item_path.stat().st_size
                            })
                            
                            if len(items) >= limit:
                                break
                                
            except Exception as e:
                logger.error(f"Error searching directory {directory}: {e}")
            
            return items
        
        # Start search from vault root
        results = search_recursively(self.vault_path)
        return results[:limit]
    
    def get_markdown_files(self, relative_path: str = "", recursive: bool = True,
                          pattern: str = "*.md") -> List[Path]:
        """Get all markdown files in a directory.
        
        Args:
            relative_path: Path relative to vault root
            recursive: Whether to search recursively
            pattern: File pattern to match
        
        Returns:
            List of relative paths to markdown files
        """
        if self.is_excluded_path(relative_path):
            return []
        
        absolute_path = self.get_absolute_path(relative_path)
        
        if not absolute_path.exists():
            raise FileNotFoundError(f"Path not found: {relative_path}")
        
        if not absolute_path.is_dir():
            raise ValidationError(f"Not a directory: {relative_path}")
        
        markdown_files = []
        
        if recursive:
            # Use rglob for recursive search
            for file_path in absolute_path.rglob(pattern):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        rel_path = self.get_relative_path(file_path)
                        if not self.is_excluded_path(str(rel_path)):
                            markdown_files.append(rel_path)
                    except ValueError as e:
                        raise ServiceError(f"Error getting relative path for {file_path}: {e}") from e
        else:
            # Use glob for non-recursive search
            for file_path in absolute_path.glob(pattern):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        rel_path = self.get_relative_path(file_path)
                        if not self.is_excluded_path(str(rel_path)):
                            markdown_files.append(rel_path)
                    except ValueError as e:
                        raise ServiceError(f"Error getting relative path for {file_path}: {e}") from e
        
        # Apply Areas/ filtering if enabled
        if self.areas_only and self.areas_filter:
            try:
                markdown_files = self.areas_filter.filter_file_paths(markdown_files)
                logger.debug(f"Areas filtering applied: {len(markdown_files)} files after filtering")
            except Exception as e:
                logger.warning(f"Areas filtering failed, returning unfiltered results: {e}")
        
        return markdown_files
    
    def is_excluded_path(self, path: str) -> bool:
        """Check if a path is in an excluded folder.
        
        Args:
            path: Path to check (relative to vault root)
            
        Returns:
            True if path is excluded, False otherwise
        """
        # Normalize path for comparison
        normalized_path = str(path).replace("\\", "/").strip("/")
        
        # Get excluded folders from settings
        settings = get_settings()
        excluded_folders = getattr(settings, 'excluded_folders', [
            '.obsidian', '.git', '.trash', 'node_modules'
        ])
        
        # Check if path is in any excluded folder
        for excluded in excluded_folders:
            excluded_norm = excluded.replace("\\", "/").strip("/")
            
            # Check if path is the excluded folder or in a subfolder
            if (normalized_path == excluded_norm or 
                normalized_path.startswith(f"{excluded_norm}/") or
                # Handle special case for root-level exclusions
                (excluded_norm.find("/") == -1 and normalized_path.startswith(f"{excluded_norm}/"))):
                return True
        
        return False
    
    def get_file_info(self, relative_path: str) -> Dict[str, Any]:
        """Get detailed information about a file.
        
        Args:
            relative_path: Path relative to vault root
            
        Returns:
            File information dictionary
        """
        if self.is_excluded_path(relative_path):
            raise PermissionError(f"Access to this path is restricted: {relative_path}")
        
        absolute_path = self.get_absolute_path(relative_path)
        
        if not absolute_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        
        stats = absolute_path.stat()
        
        info = {
            "path": relative_path,
            "name": absolute_path.name,
            "type": "directory" if absolute_path.is_dir() else "file",
            "size": stats.st_size,
            "created": stats.st_ctime,
            "modified": stats.st_mtime,
            "created_formatted": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_formatted": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "is_markdown": absolute_path.suffix.lower() == '.md' if absolute_path.is_file() else False,
            "extension": absolute_path.suffix if absolute_path.is_file() else None
        }
        
        if absolute_path.is_file():
            info["checksum"] = self._calculate_checksum(absolute_path)
        
        return info
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate MD5 checksum of file content.
        
        Args:
            path: Path to the file
            
        Returns:
            MD5 checksum as hex string
        """
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum for {path}: {e}")
            return ""
    
    def get_recent_files(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently modified files from the vault.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of files to return
            
        Returns:
            List of recently modified files
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_files = []
        
        for file_path in self.vault_path.rglob("*.md"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    rel_path = self.get_relative_path(file_path)
                    if not self.is_excluded_path(str(rel_path)):
                        # Apply Areas/ filtering if enabled
                        if self.areas_only and self.areas_filter:
                            if not self.areas_filter.is_areas_content(str(rel_path)):
                                continue
                        
                        stats = file_path.stat()
                        if stats.st_mtime > cutoff_time:
                            recent_files.append({
                                "path": str(rel_path),
                                "name": file_path.name,
                                "modified": stats.st_mtime,
                                "modified_formatted": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                                "size": stats.st_size
                            })
                except ValueError as e:
                    raise ServiceError(f"Error getting relative path for {file_path}: {e}") from e
        
        recent_files.sort(key=lambda x: x["modified"], reverse=True)
        return recent_files[:limit]
    
    def get_areas_filter_status(self) -> Dict[str, Any]:
        """Get the current status of Areas/ filtering.
        
        Returns:
            Dictionary with Areas filtering status and information
        """
        status = {
            "areas_filtering_enabled": self.areas_only,
            "areas_filter_initialized": self.areas_filter is not None,
            "vault_path": str(self.vault_path)
        }
        
        if self.areas_filter:
            try:
                # Get Areas folder information
                areas_structure = self.areas_filter.get_areas_structure()
                exclusion_summary = self.areas_filter.get_exclusion_summary()
                
                status.update({
                    "areas_folder_name": self.areas_filter.areas_folder_name,
                    "areas_folder_path": str(self.areas_filter.areas_folder_path),
                    "areas_folder_exists": areas_structure.get("exists", False),
                    "areas_file_count": areas_structure.get("total_files", 0),
                    "excluded_folders": exclusion_summary.get("excluded_folders", [])
                })
            except Exception as e:
                status["error"] = f"Error getting Areas filter information: {e}"
                logger.warning(f"Error getting Areas filter status: {e}")
        
        return status
    
    def validate_areas_filtering(self) -> Dict[str, Any]:
        """Validate Areas/ filtering configuration and folder structure.
        
        Returns:
            Dictionary with validation results
        """
        if not self.areas_only:
            return {
                "areas_filtering_enabled": False,
                "validation_passed": True,
                "message": "Areas filtering is disabled"
            }
        
        if not self.areas_filter:
            return {
                "areas_filtering_enabled": True,
                "validation_passed": False,
                "error": "Areas filter not initialized"
            }
        
        try:
            validation_result = self.areas_filter.validate_areas_folder()
            return {
                "areas_filtering_enabled": True,
                "validation_passed": validation_result.get("validation_passed", False),
                **validation_result
            }
        except Exception as e:
            return {
                "areas_filtering_enabled": True,
                "validation_passed": False,
                "error": str(e)
            }
    
    # Interface method required by IVaultReader
    def list_files(self, extension: str = ".md") -> List[Path]:
        """List all files in the vault with specified extension.
        
        Args:
            extension: File extension to filter by
            
        Returns:
            List of relative paths to files
        """
        return self.get_markdown_files(pattern=f"*{extension}")