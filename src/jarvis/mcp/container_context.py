"""
Container-aware MCP server context using dependency injection.

This module provides a new implementation of MCPServerContext that uses
the dependency injection container for better modularity and testability.
"""

from pathlib import Path
from typing import Dict, Optional

from jarvis.core.container import ServiceContainer
from jarvis.core.interfaces import (
    IVectorDatabase, IVectorEncoder, IVectorSearcher,
    IGraphDatabase, IVaultReader, IHealthChecker, IMetrics
)
from jarvis.mcp.cache import MCPToolCache
from jarvis.services.ranking import ResultRanker
from jarvis.utils.logging import setup_logging
from jarvis.utils.config import JarvisSettings

logger = setup_logging(__name__)


class ContainerAwareMCPServerContext:
    """MCP server context that uses dependency injection container."""
    
    def __init__(
        self,
        vaults: Dict[str, Path],
        database_path: Path,
        settings: Optional[JarvisSettings] = None
    ):
        """Initialize container-aware MCP server context.
        
        Args:
            vaults: Dictionary mapping vault names to paths
            database_path: Path to DuckDB database file
            settings: Optional settings override
        """
        logger.info("🚀 Initializing container-aware MCP server context")
        logger.debug(f"📂 Vaults: {list(vaults.keys())}")
        logger.debug(f"💾 Database path: {database_path}")
        
        self.vaults = vaults
        self.database_path = database_path
        self.settings = settings or JarvisSettings()
        
        logger.debug(f"⚙️ Settings loaded: use_dependency_injection={self.settings.use_dependency_injection}, metrics_enabled={self.settings.metrics_enabled}, graph_enabled={self.settings.graph_enabled}")
        
        # Initialize service container
        logger.info("🔧 Initializing service container")
        try:
            self.container = ServiceContainer(self.settings)
            logger.debug("✅ ServiceContainer created successfully")
            
            logger.info("🔧 Configuring default services")
            self.container.configure_default_services()
            logger.debug("✅ Default services configured successfully")
            
            # Log registered services
            container_info = self.container.get_service_info()
            registered_services = container_info.get('registered_services', [])
            logger.info(f"📋 Registered services: {registered_services}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize service container: {e}")
            raise
        
        # Initialize additional services not managed by container
        logger.debug("🔧 Initializing additional services")
        try:
            self.ranker = ResultRanker()
            logger.debug("✅ ResultRanker initialized")
            
            self.mcp_cache = MCPToolCache(
                self.settings.mcp_cache_size, 
                self.settings.mcp_cache_ttl
            )
            logger.debug(f"✅ MCPToolCache initialized (size: {self.settings.mcp_cache_size}, ttl: {self.settings.mcp_cache_ttl})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize additional services: {e}")
            raise
        
        logger.info(f"🎉 Container-aware MCP server context initialized successfully with {len(vaults)} vaults")
    
    @property
    def database(self) -> IVectorDatabase:
        """Get vector database service."""
        logger.debug("🔍 Retrieving vector database service")
        try:
            service = self.container.get(IVectorDatabase)
            logger.debug(f"✅ Vector database service retrieved: {service is not None}")
            return service
        except Exception as e:
            logger.error(f"❌ Failed to retrieve vector database service: {e}")
            raise
    
    @property
    def encoder(self) -> IVectorEncoder:
        """Get vector encoder service."""
        logger.debug("🔍 Retrieving vector encoder service")
        try:
            service = self.container.get(IVectorEncoder)
            logger.debug(f"✅ Vector encoder service retrieved: {service is not None}")
            return service
        except Exception as e:
            logger.error(f"❌ Failed to retrieve vector encoder service: {e}")
            raise
    
    @property
    def searcher(self) -> IVectorSearcher:
        """Get vector searcher service."""
        logger.debug("🔍 Retrieving vector searcher service")
        try:
            service = self.container.get(IVectorSearcher)
            logger.debug(f"✅ Vector searcher service retrieved: {service is not None}")
            return service
        except Exception as e:
            logger.error(f"❌ Failed to retrieve vector searcher service: {e}")
            raise
    
    @property
    def graph_database(self) -> IGraphDatabase:
        """Get graph database service."""
        logger.debug("🔍 Retrieving graph database service")
        try:
            service = self.container.get(IGraphDatabase)
            logger.debug(f"✅ Graph database service retrieved: {service is not None}")
            return service
        except Exception as e:
            logger.warning(f"⚠️ Graph database not available: {e}")
            return None
    
    @property
    def vault_readers(self) -> Dict[str, IVaultReader]:
        """Get vault readers."""
        logger.debug("🔍 Retrieving vault reader service")
        # For now, return a single reader for all vaults
        # In the future, this could be expanded to support multiple readers
        try:
            reader = self.container.get(IVaultReader)
            logger.debug(f"✅ Vault reader service retrieved: {reader is not None}")
            readers_dict = {vault_name: reader for vault_name in self.vaults.keys()}
            logger.debug(f"📂 Created vault readers for {len(readers_dict)} vaults")
            return readers_dict
        except Exception as e:
            logger.warning(f"⚠️ Vault reader not available: {e}")
            return {}
    
    @property
    def health_checker(self) -> IHealthChecker:
        """Get health checker service."""
        return self.container.get(IHealthChecker)
    
    @property
    def metrics(self) -> Optional[IMetrics]:
        """Get metrics service if enabled."""
        if not self.settings.metrics_enabled:
            return None
        try:
            return self.container.get(IMetrics)
        except Exception as e:
            logger.warning(f"Metrics service not available: {e}")
            return None
    
    def get_service_info(self) -> Dict[str, any]:
        """Get information about all services."""
        info = {
            "container_info": self.container.get_service_info(),
            "vault_count": len(self.vaults),
            "database_path": str(self.database_path),
            "settings": {
                "metrics_enabled": self.settings.metrics_enabled,
                "graph_enabled": self.settings.graph_enabled,
                "use_dependency_injection": self.settings.use_dependency_injection
            }
        }
        
        # Add service health status
        try:
            health_checker = self.health_checker
            info["health_status"] = health_checker.get_overall_health()
        except Exception as e:
            info["health_status"] = {"error": str(e)}
        
        return info
    
    def close(self):
        """Clean up resources."""
        logger.info("Closing container-aware MCP server context")
        
        # Close MCP cache
        if hasattr(self, 'mcp_cache') and self.mcp_cache:
            self.mcp_cache.clear()
        
        # Dispose of service container (this will close all managed services)
        if hasattr(self, 'container'):
            self.container.dispose()
        
        logger.info("Container-aware MCP server context closed")
    
    def clear_cache(self):
        """Clear the MCP tool cache."""
        if self.mcp_cache:
            self.mcp_cache.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()