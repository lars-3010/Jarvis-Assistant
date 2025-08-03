"""
Main entry point for Jarvis Assistant CLI.

This module provides the command-line interface for Jarvis Assistant,
including MCP server startup and vault indexing operations.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Dict

import click

from jarvis.services.vector.database import VectorDatabase
from jarvis.services.vector.encoder import VectorEncoder
from jarvis.services.vector.indexer import VectorIndexer
from jarvis.services.graph.database import GraphDatabase
from jarvis.services.graph.indexer import GraphIndexer
from jarvis.mcp.server import run_mcp_server
from jarvis.mcp.container_context import ContainerAwareMCPServerContext as MCPServerContext
from jarvis.utils.logging import setup_logging
from jarvis.utils.config import JarvisSettings, get_settings, ValidationResult
from jarvis.utils.helpers import validate_settings_or_exit, resolve_vault_path_or_exit, resolve_database_path_or_exit

# Import dataset generation components
from jarvis.tools.dataset_generation import DatasetGenerator

logger = setup_logging(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Jarvis Assistant - AI-augmented learning system."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logger.info("Verbose logging enabled")


@cli.command()
@click.option('--vault', type=click.Path(path_type=Path),
              help='Path to Obsidian vault')
@click.option('--database', type=click.Path(path_type=Path),
              help='Path to DuckDB database file')
@click.option('--reindex', is_flag=True, help='Force complete reindexing')
@click.option('--watch', is_flag=True, help='Watch for file changes')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def mcp(ctx: click.Context, vault: Optional[Path], database: Optional[Path], 
        reindex: bool, watch: bool, debug: bool) -> None:
    """Start the MCP server for Claude Desktop integration."""
    
    # Enable debug logging if requested
    if debug or ctx.obj.get('verbose', False):
        import logging
        logging.getLogger('jarvis').setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    logger.info("🚀 Starting Jarvis MCP server...")
    
    try:
        # Validate settings
        logger.debug("🔧 Validating settings")
        validate_settings_or_exit()
        settings = get_settings()
        logger.debug(f"🔧 Settings loaded successfully")
        
        # Show warnings if any
        validation_result = settings.validate_settings()
        for warning in validation_result.warnings:
            click.echo(f"Warning: {warning}")
            logger.warning(f"Settings warning: {warning}")

        # Determine vault path
        logger.debug("📁 Resolving vault path")
        vault_path = resolve_vault_path_or_exit(vault)
        logger.debug(f"📁 Vault path resolved: {vault_path}")
        
        # Special handling for iCloud paths
        if "iCloud" in str(vault_path):
            logger.info(f"📱 iCloud vault detected: {vault_path}")
            # Check if path exists and is accessible
            if not vault_path.exists():
                logger.error(f"❌ iCloud vault path does not exist: {vault_path}")
                click.echo(f"Error: iCloud vault path does not exist: {vault_path}")
                click.echo("Make sure iCloud is synced and the vault folder is downloaded locally.")
                sys.exit(1)
            elif not vault_path.is_dir():
                logger.error(f"❌ iCloud vault path is not a directory: {vault_path}")
                click.echo(f"Error: iCloud vault path is not a directory: {vault_path}")
                sys.exit(1)
            else:
                logger.info(f"✅ iCloud vault path validated: {vault_path}")
        
        # Determine database path
        logger.debug("💾 Resolving database path")
        db_path = resolve_database_path_or_exit(database, vault_path)
        logger.debug(f"💾 Database path resolved: {db_path}")
        
        # Setup vault configuration
        vaults = {"default": vault_path}
        
        click.echo(f"Vault: {vault_path}")
        click.echo(f"Database: {db_path}")
        
        # Check if indexing is needed
        if reindex or not db_path.exists():
            click.echo("Indexing vault before starting MCP server...")
            logger.info(f"📚 Starting indexing: reindex={reindex}, db_exists={db_path.exists()}")
            _index_vault(vault_path, db_path, force_reindex=reindex)
            logger.info("✅ Indexing completed")
        
        if watch:
            click.echo("File watching enabled - vault changes will trigger automatic reindexing")
            logger.info("🔍 File watching mode enabled")
        
        click.echo("Starting MCP server...")
        logger.info(f"🚀 MCP server starting with vault: {vault_path}, database: {db_path}, watch: {watch}")
        
        # Run the MCP server
        logger.info("🌊 Running MCP server")
        asyncio.run(run_mcp_server(vaults, db_path, settings, watch))
        
    except KeyboardInterrupt:
        click.echo("\nMCP server stopped")
        logger.info("⌨️ MCP server stopped by user")
    except Exception as e:
        logger.error(f"💥 MCP server error: {e}")
        logger.error(f"🔍 Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"🔍 Full traceback:\n{traceback.format_exc()}")
        click.echo(f"Error: {str(e)}")
        click.echo("Check logs for detailed error information.")
        sys.exit(1)


@cli.command()
@click.option('--vault', type=click.Path(path_type=Path),
              help='Path to Obsidian vault')
@click.option('--database', type=click.Path(path_type=Path),
              help='Path to DuckDB database file')
@click.option('--force', is_flag=True, help='Force complete reindexing')
@click.option('--batch-size', type=int, default=32, help='Batch size for processing')
def index(vault: Optional[Path], database: Optional[Path], force: bool, batch_size: int) -> None:
    """Index a vault for semantic search."""
    try:
        # Validate settings
        validate_settings_or_exit()
        settings = get_settings()
        
        # Show warnings if any
        validation_result = settings.validate_settings()
        for warning in validation_result.warnings:
            click.echo(f"Warning: {warning}")

        # Determine vault path
        vault_path = resolve_vault_path_or_exit(vault)
        
        # Determine database path
        db_path = resolve_database_path_or_exit(database, vault_path)
        
        click.echo(f"Indexing vault: {vault_path}")
        click.echo(f"Database: {db_path}")
        click.echo(f"Batch size: {batch_size}")
        click.echo(f"Force reindex: {force}")
        
        # Perform indexing
        stats = _index_vault(vault_path, db_path, force_reindex=force, batch_size=batch_size)
        
        # Clear MCP cache after indexing
        with MCPServerContext({"default": vault_path}, db_path, settings) as mcp_context:
            mcp_context.clear_cache()

        # Display results
        click.echo("\n" + "="*50)
        click.echo("INDEXING COMPLETE")
        click.echo("="*50)
        click.echo(f"Total files: {stats.total_files}")
        click.echo(f"Processed: {stats.processed_files}")
        click.echo(f"Skipped: {stats.skipped_files}")
        click.echo(f"Failed: {stats.failed_files}")
        click.echo(f"Total time: {stats.total_time:.2f}s")
        if stats.processed_files > 0:
            click.echo(f"Speed: {stats.files_per_second:.2f} files/sec")
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


def _index_vault(
    vault_path: Path, 
    database_path: Path, 
    force_reindex: bool = False,
    batch_size: int = 32
):
    """Helper function to index a vault."""
    vaults = {"default": vault_path}
    
    # Initialize services
    with VectorDatabase(database_path) as database:
        encoder = VectorEncoder()
        indexer = VectorIndexer(database, encoder, vaults, batch_size=batch_size)
        
        # Progress callback
        def progress_callback(processed: int, total: int):
            if total > 0:
                percentage = (processed / total) * 100
                click.echo(f"Progress: {processed}/{total} ({percentage:.1f}%)")
        
        # Index the vault
        click.echo("Starting indexing...")
        stats = indexer.index_vault(
            vault_name="default", 
            force_reindex=force_reindex,
            progress_callback=progress_callback
        )
        
        return stats


@cli.command()
@click.option('--vault', type=click.Path(path_type=Path),
              help='Path to Obsidian vault')
@click.option('--database', type=click.Path(path_type=Path),
              help='Path to DuckDB database file')
@click.option('--limit', type=int, default=10, help='Maximum results to show')
def search(vault: Optional[Path], database: Optional[Path], limit: int) -> None:
    """Interactive semantic search for testing."""
    try:
        # Validate settings
        validate_settings_or_exit()
        settings = get_settings()
        
        # Show warnings if any
        validation_result = settings.validate_settings()
        for warning in validation_result.warnings:
            click.echo(f"Warning: {warning}")

        # Determine vault path
        vault_path = resolve_vault_path_or_exit(vault)
        
        # Determine database path
        db_path = resolve_database_path_or_exit(database, vault_path)
        
        if not db_path.exists():
            click.echo(f"Error: Database does not exist: {db_path}")
            click.echo("Run 'jarvis index' first to create the database.")
            sys.exit(1)
        
        # Initialize search system
        vaults = {"default": vault_path}
        
        with VectorDatabase(db_path, read_only=True) as database:
            encoder = VectorEncoder()
            from jarvis.services.vector.searcher import VectorSearcher
            searcher = VectorSearcher(database, encoder, vaults)
            
            click.echo(f"Vault: {vault_path}")
            click.echo(f"Database: {db_path}")
            click.echo(f"Notes in database: {database.num_notes()}")
            click.echo("\nEnter search queries (empty line to quit):")
            
            while True:
                try:
                    query = input("\n> ").strip()
                    if not query:
                        break
                    
                    results = searcher.search(query, top_k=limit)
                    
                    if not results:
                        click.echo("No results found.")
                        continue
                    
                    click.echo(f"\nFound {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        click.echo(f"{i}. {result.path} (score: {result.similarity_score:.3f})")
                        
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
        
        click.echo("\nSearch session ended.")
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option('--vault', type=click.Path(path_type=Path),
              help='Path to Obsidian vault')
@click.option('--database', type=click.Path(path_type=Path),
              help='Path to DuckDB database file')
def stats(vault: Optional[Path], database: Optional[Path]) -> None:
    """Show system statistics and performance metrics."""
    try:
        # Validate settings
        validate_settings_or_exit()
        settings = get_settings()
        
        # Show warnings if any
        validation_result = settings.validate_settings()
        for warning in validation_result.warnings:
            click.echo(f"Warning: {warning}")

        # Determine vault path
        vault_path = resolve_vault_path_or_exit(vault)
        
        # Determine database path
        db_path = resolve_database_path_or_exit(database, vault_path)
        
        if not db_path.exists():
            click.echo(f"Error: Database does not exist: {db_path}")
            click.echo("Run 'jarvis index' first to create the database.")
            sys.exit(1)
        
        # Initialize system components
        vaults = {"default": vault_path}
        
        with VectorDatabase(db_path, read_only=True) as database:
            encoder = VectorEncoder()
            from jarvis.services.vector.searcher import VectorSearcher
            searcher = VectorSearcher(database, encoder, vaults)
            
            # Get system info
            model_info = searcher.get_model_info()
            search_stats = searcher.get_search_stats()
            vault_stats = searcher.get_vault_stats()
            
            # Display statistics
            click.echo("="*60)
            click.echo("JARVIS ASSISTANT SYSTEM STATISTICS")
            click.echo("="*60)
            
            # System Configuration
            click.echo("\n📊 SYSTEM CONFIGURATION")
            click.echo(f"Vault Path: {vault_path}")
            click.echo(f"Database: {db_path}")
            click.echo(f"Model: {model_info['encoder_info']['model_name']}")
            click.echo(f"Device: {model_info['encoder_info']['device']}")
            click.echo(f"Vector Dimension: {model_info['encoder_info']['vector_dimension']}")
            
            # Database Statistics
            click.echo("\n📚 DATABASE STATISTICS")
            click.echo(f"Total Notes: {model_info['database_note_count']}")
            click.echo(f"Configured Vaults: {model_info['vault_count']}")
            
            for vault_name, stats in vault_stats.items():
                click.echo(f"\nVault '{vault_name}':")
                click.echo(f"  Notes: {stats['note_count']}")
                if stats.get('latest_modified'):
                    from datetime import datetime
                    latest = datetime.fromtimestamp(stats['latest_modified']).strftime("%Y-%m-%d %H:%M:%S")
                    click.echo(f"  Last Modified: {latest}")
            
            # Performance Statistics
            click.echo("\n⚡ PERFORMANCE STATISTICS")
            click.echo(f"Cache Enabled: {'Yes' if model_info['cache_enabled'] else 'No'}")
            click.echo(f"Total Searches: {search_stats['total_searches']}")
            
            if search_stats['total_searches'] > 0:
                click.echo(f"Cache Hit Rate: {search_stats.get('cache_hit_rate', 0):.2%}")
                avg_time = search_stats.get('avg_search_time', 0)
                click.echo(f"Avg Search Time: {avg_time*1000:.1f}ms")
            
            # Cache Statistics (if enabled)
            if model_info['cache_enabled'] and 'cache_stats' in model_info:
                cache_stats = model_info['cache_stats']
                click.echo("\n🗄️  CACHE STATISTICS")
                click.echo(f"Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
                click.echo(f"Cache Hits: {cache_stats['hits']}")
                click.echo(f"Cache Misses: {cache_stats['misses']}")
                click.echo(f"Cache Hit Rate: {cache_stats['hit_rate']:.2%}")
                click.echo(f"Cache Evictions: {cache_stats['evictions']}")
                click.echo(f"TTL: {cache_stats['ttl_seconds']}s")
            
            click.echo("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


@cli.command('graph-index')
@click.option('--vault', type=click.Path(path_type=Path),
              help='Path to Obsidian vault')
@click.option('--uri', type=str, help='Neo4j connection URI')
@click.option('--user', type=str, help='Neo4j username')
@click.option('--password', type=str, help='Neo4j password')
def graph_index(vault: Optional[Path], uri: Optional[str], user: Optional[str], password: Optional[str]) -> None:
    """Index a vault for graph-based search."""
    try:
        settings = get_settings()
        validation_result = settings.validate_settings()

        if not validation_result.valid:
            for error in validation_result.errors:
                click.echo(f"Error: {error}")
            sys.exit(1)
        
        for warning in validation_result.warnings:
            click.echo(f"Warning: {warning}")

        if not settings.graph_enabled:
            click.echo("Graph database integration is disabled in the settings.")
            sys.exit(1)

        vault_path = vault or settings.get_vault_path()
        if not vault_path:
            click.echo("Error: No vault path specified. Use --vault option or configure JARVIS_VAULT_PATH")
            sys.exit(1)
        
        if not vault_path.exists():
            click.echo(f"Error: Vault path does not exist: {vault_path}")
            sys.exit(1)

        db_uri = uri or settings.neo4j_uri
        db_user = user or settings.neo4j_user
        db_password = password or settings.neo4j_password

        click.echo(f"Indexing vault: {vault_path}")
        click.echo(f"Graph Database: {db_uri}")

        vaults = {"default": vault_path}
        database = GraphDatabase(settings) # Pass settings object
        try:
            indexer = GraphIndexer(database, vaults)
            indexer.index_vault("default")
        finally:
            database.close()

        click.echo("\nGraph indexing complete.")

    except Exception as e:
        logger.error(f"Graph indexing error: {e}")
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


@cli.command('graph-search')
@click.option('--uri', type=str, help='Neo4j connection URI')
@click.option('--user', type=str, help='Neo4j username')
@click.option('--password', type=str, help='Neo4j password')
@click.option('--query', type=str, help='Note path to search for')
@click.option('--depth', type=int, default=1, help='Search depth')
def graph_search(uri: Optional[str], user: Optional[str], password: Optional[str], query: str, depth: int) -> None:
    """Interactive graph search for testing."""
    try:
        settings = get_settings()
        validation_result = settings.validate_settings()

        if not validation_result.valid:
            for error in validation_result.errors:
                click.echo(f"Error: {error}")
            sys.exit(1)
        
        for warning in validation_result.warnings:
            click.echo(f"Warning: {warning}")

        if not settings.graph_enabled:
            click.echo("Graph database integration is disabled in the settings.")
            sys.exit(1)

        db_uri = uri or settings.neo4j_uri
        db_user = user or settings.neo4j_user
        db_password = password or settings.neo4j_password

        database = GraphDatabase(settings) # Pass settings object
        try:
            click.echo(f"Graph Database: {db_uri}")
            click.echo(f"Searching for note: {query}")

            graph = database.get_note_graph(query, depth)
            
            if not graph or not graph.get('nodes'):
                click.echo("No results found.")
                return

            click.echo(f"\nFound {len(graph['nodes'])} nodes and {len(graph['relationships'])} relationships:")
            for node in graph['nodes']:
                click.echo(f"  - Node: {node['label']} ({node['path']})")
            for rel in graph['relationships']:
                click.echo(f"  - Relationship: {rel['type']}")
        finally:
            database.close()

    except Exception as e:
        logger.error(f"Graph search error: {e}")
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


@cli.command('generate-dataset')
@click.option('--vault', type=click.Path(path_type=Path),
              help='Path to Obsidian vault')
@click.option('--output', type=click.Path(path_type=Path),
              help='Output directory for datasets (default: ./datasets)')
@click.option('--notes-filename', type=str, default='notes_dataset.csv',
              help='Filename for notes dataset')
@click.option('--pairs-filename', type=str, default='pairs_dataset.csv',
              help='Filename for pairs dataset')
@click.option('--negative-ratio', type=float, default=5.0,
              help='Negative to positive pairs ratio')
@click.option('--sampling', type=click.Choice(['random', 'stratified']),
              default='stratified', help='Negative sampling strategy')
@click.option('--batch-size', type=int, default=32,
              help='Batch size for processing')
@click.option('--max-pairs', type=int, default=1000,
              help='Maximum pairs per note')
@click.pass_context
def generate_dataset(ctx: click.Context, vault: Optional[Path], output: Optional[Path],
                    notes_filename: str, pairs_filename: str, negative_ratio: float,
                    sampling: str, batch_size: int, max_pairs: int) -> None:
    """Generate machine learning datasets from Obsidian vault."""
    
    # Enable debug logging if verbose
    if ctx.obj.get('verbose', False):
        import logging
        logging.getLogger('jarvis').setLevel(logging.DEBUG)
        logger.info("Debug logging enabled for dataset generation")
    
    logger.info("🚀 Starting dataset generation...")
    
    try:
        # Validate settings
        validate_settings_or_exit()
        settings = get_settings()
        
        # Show warnings if any
        validation_result = settings.validate_settings()
        for warning in validation_result.warnings:
            click.echo(f"Warning: {warning}")

        # Determine vault path
        vault_path = resolve_vault_path_or_exit(vault)
        
        # Determine output directory
        if output is None:
            # Use settings to get dataset output directory
            output_dir = settings.get_dataset_output_path()
        else:
            output_dir = output.resolve()
        
        click.echo("="*60)
        click.echo("JARVIS DATASET GENERATION")
        click.echo("="*60)
        click.echo(f"Vault: {vault_path}")
        click.echo(f"Output Directory: {output_dir}")
        click.echo(f"Notes Dataset: {notes_filename}")
        click.echo(f"Pairs Dataset: {pairs_filename}")
        click.echo(f"Negative Sampling Ratio: {negative_ratio}")
        click.echo(f"Sampling Strategy: {sampling}")
        click.echo(f"Batch Size: {batch_size}")
        click.echo(f"Max Pairs per Note: {max_pairs}")
        click.echo("="*60)
        
        # Progress tracking
        current_step = 0
        total_steps = 5
        
        def progress_callback(message: str, step: int, total: int):
            nonlocal current_step
            if step != current_step:
                current_step = step
                percentage = (step / total) * 100
                click.echo(f"[{step}/{total}] ({percentage:.1f}%) {message}")
        
        # Initialize and run dataset generator
        with DatasetGenerator(vault_path, output_dir) as generator:
            result = generator.generate_datasets(
                notes_filename=notes_filename,
                pairs_filename=pairs_filename,
                negative_sampling_ratio=negative_ratio,
                sampling_strategy=sampling,
                batch_size=batch_size,
                max_pairs_per_note=max_pairs,
                progress_callback=progress_callback
            )
        
        # Display results
        if result.success:
            click.echo("\n" + "="*60)
            click.echo("DATASET GENERATION COMPLETE")
            click.echo("="*60)
            
            summary = result.summary
            click.echo(f"Total Notes: {summary.total_notes}")
            click.echo(f"Notes Processed: {summary.notes_processed}")
            if summary.notes_failed > 0:
                click.echo(f"Notes Failed: {summary.notes_failed}")
            
            click.echo(f"Total Pairs: {summary.pairs_generated}")
            click.echo(f"Positive Pairs: {summary.positive_pairs}")
            click.echo(f"Negative Pairs: {summary.negative_pairs}")
            
            click.echo(f"Generation Time: {summary.total_time_seconds:.2f}s")
            if summary.performance_metrics:
                click.echo(f"Notes/Second: {summary.performance_metrics.get('notes_per_second', 0):.2f}")
                click.echo(f"Pairs/Second: {summary.performance_metrics.get('pairs_per_second', 0):.2f}")
            
            click.echo("\nOutput Files:")
            if result.notes_dataset_path:
                click.echo(f"  Notes Dataset: {result.notes_dataset_path}")
            if result.pairs_dataset_path:
                click.echo(f"  Pairs Dataset: {result.pairs_dataset_path}")
            
            # Link extraction statistics
            if summary.link_statistics:
                link_stats = summary.link_statistics
                click.echo(f"\nLink Statistics:")
                click.echo(f"  Total Links: {link_stats.total_links}")
                click.echo(f"  Unique Links: {link_stats.unique_links}")
                if link_stats.broken_links > 0:
                    click.echo(f"  Broken Links: {link_stats.broken_links}")
                if link_stats.link_types:
                    click.echo(f"  Link Types: {dict(link_stats.link_types)}")
            
            click.echo("\n✅ Dataset generation completed successfully!")
            
        else:
            click.echo("\n" + "="*60)
            click.echo("DATASET GENERATION FAILED")
            click.echo("="*60)
            click.echo(f"Error: {result.error_message}")
            
            if result.summary and result.summary.validation_result:
                validation = result.summary.validation_result
                if validation.errors:
                    click.echo("\nValidation Errors:")
                    for error in validation.errors:
                        click.echo(f"  - {error}")
                if validation.warnings:
                    click.echo("\nValidation Warnings:")
                    for warning in validation.warnings:
                        click.echo(f"  - {warning}")
            
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Dataset generation error: {e}")
        click.echo(f"Error: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()