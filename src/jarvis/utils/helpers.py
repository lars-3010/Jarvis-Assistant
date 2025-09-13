"""
Common utility functions for Jarvis Assistant.

This module provides reusable functions for common patterns found throughout the codebase.
"""

import sys
from pathlib import Path

import click

from jarvis.utils.config import get_settings
import logging

logger = logging.getLogger(__name__)


def validate_settings_or_exit() -> None:
    """
    Validate settings and exit with error message if invalid.
    
    This function consolidates the common pattern of validating settings
    and exiting with appropriate error messages.
    """
    settings = get_settings()
    validation_result = settings.validate_settings()

    if not validation_result.valid:
        for error in validation_result.errors:
            click.echo(f"Error: {error}")
        sys.exit(1)


def resolve_vault_path_or_exit(vault: Path | None = None) -> Path:
    """
    Resolve vault path from parameter or settings, exit if none found.
    
    Args:
        vault: Optional vault path from command line
        
    Returns:
        Resolved vault path
        
    Raises:
        SystemExit: If no vault path can be resolved
    """
    settings = get_settings()
    vault_path = vault or settings.get_vault_path()

    if not vault_path:
        click.echo("Error: No vault path specified. Use --vault parameter or configure vault_path in settings.")
        sys.exit(1)

    vault_path = Path(vault_path).resolve()

    if not vault_path.exists():
        click.echo(f"Error: Vault path not found: {vault_path}")
        sys.exit(1)

    if not vault_path.is_dir():
        click.echo(f"Error: Vault path is not a directory: {vault_path}")
        sys.exit(1)

    return vault_path


def resolve_database_path_or_exit(database: Path | None = None, vault_path: Path | None = None) -> Path:
    """
    Resolve database path from parameter or settings, exit if invalid.
    
    Args:
        database: Optional database path from command line
        vault_path: Vault path for default database location
        
    Returns:
        Resolved database path
        
    Raises:
        SystemExit: If database path cannot be resolved
    """
    settings = get_settings()

    if database:
        db_path = Path(database).resolve()
    else:
        db_path = Path(settings.vector_db_path).resolve()

    # Create parent directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return db_path


def setup_logging_with_level(log_level: str = "INFO") -> None:
    """
    Set up logging with specified level.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging

    # Set the root logger level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    # Also set for jarvis logger
    logging.getLogger("jarvis").setLevel(getattr(logging, log_level.upper()))

    logger.info(f"Logging level set to {log_level}")


def validate_directory_exists(path: Path, description: str) -> None:
    """
    Validate that a directory exists and is actually a directory.
    
    Args:
        path: Path to validate
        description: Description of the path for error messages
        
    Raises:
        SystemExit: If path is invalid
    """
    if not path.exists():
        click.echo(f"Error: {description} not found: {path}")
        sys.exit(1)

    if not path.is_dir():
        click.echo(f"Error: {description} is not a directory: {path}")
        sys.exit(1)


def create_directory_if_not_exists(path: Path, description: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Path to create
        description: Description of the path for logging
    """
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {description}: {path}")
        except Exception as e:
            click.echo(f"Error: Could not create {description} {path}: {e}")
            sys.exit(1)
