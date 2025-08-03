
import logging
from typing import Dict, Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from jarvis.utils.config import JarvisSettings
from jarvis.utils.errors import ServiceError
from jarvis.services.vector.database import VectorDatabase
from jarvis.core.interfaces import IHealthChecker

logger = logging.getLogger(__name__)


class HealthChecker(IHealthChecker):
    """Provides health checks for various Jarvis Assistant services."""

    def __init__(self, settings: JarvisSettings):
        self.settings = settings

    def check_neo4j_health(self) -> Dict[str, Any]:
        """
        Checks the health of the Neo4j database.
        """
        status = {"service": "Neo4j", "status": "UNAVAILABLE", "details": ""}
        if not self.settings.graph_enabled:
            status["status"] = "DISABLED"
            status["details"] = "Neo4j integration is disabled in configuration."
            return status

        try:
            with GraphDatabase.driver(self.settings.neo4j_uri, auth=(self.settings.neo4j_user, self.settings.neo4j_password)) as driver:
                driver.verify_connectivity()
            status["status"] = "HEALTHY"
            status["details"] = "Successfully connected to Neo4j."
        except ServiceUnavailable as e:
            status["details"] = f"Connection failed: {e}"
        except Exception as e:
            status["details"] = f"An unexpected error occurred: {e}"

        return status

    def check_vector_db_health(self) -> Dict[str, Any]:
        """
        Checks the health of the DuckDB vector database.
        """
        status = {"service": "VectorDB", "status": "UNAVAILABLE", "details": ""}
        try:
            # Attempt to connect and run a simple query
            from jarvis.services.vector.database import VectorDatabase
            db = VectorDatabase(self.settings.get_vector_db_path(), read_only=True)
            db.num_notes() # Try a simple operation
            db.close()
            status["status"] = "HEALTHY"
            status["details"] = "Successfully connected to Vector Database."
        except ServiceError as e:
            status["details"] = f"Connection failed: {e}"
        except Exception as e:
            status["details"] = f"An unexpected error occurred: {e}"
        return status

    def check_vault_health(self) -> Dict[str, Any]:
        """
        Checks the health of the Obsidian vault.
        """
        status = {"service": "Vault", "status": "UNAVAILABLE", "details": ""}
        vault_path = self.settings.get_vault_path()
        if not vault_path:
            status["status"] = "UNCONFIGURED"
            status["details"] = "Vault path is not configured."
            return status

        if vault_path.exists() and vault_path.is_dir():
            status["status"] = "HEALTHY"
            status["details"] = f"Vault found at {vault_path}."
        else:
            status["details"] = f"Vault path does not exist or is not a directory: {vault_path}"
        return status

    def get_overall_health(self) -> Dict[str, Any]:
        """
        Gets the health status of all key services.
        """
        neo4j_health = self.check_neo4j_health()
        vector_db_health = self.check_vector_db_health()
        vault_health = self.check_vault_health()

        overall_status = "HEALTHY"
        if neo4j_health["status"] == "UNAVAILABLE" or \
           vector_db_health["status"] == "UNAVAILABLE" or \
           vault_health["status"] == "UNAVAILABLE" or \
           vault_health["status"] == "UNCONFIGURED":
            overall_status = "DEGRADED"

        return {
            "overall_status": overall_status,
            "services": [
                neo4j_health,
                vector_db_health,
                vault_health
            ]
        }
    
    # Interface methods required by IHealthChecker
    def check_vector_database(self) -> bool:
        """Check vector database health."""
        health = self.check_vector_db_health()
        return health["status"] == "HEALTHY"
    
    def check_graph_database(self) -> bool:
        """Check graph database health."""
        health = self.check_neo4j_health()
        return health["status"] == "HEALTHY"
    
    def check_vault_access(self) -> Dict[str, bool]:
        """Check vault accessibility."""
        health = self.check_vault_health()
        return {"default": health["status"] == "HEALTHY"}


def check_neo4j_health(uri: str, auth: tuple) -> bool:
    """
    Standalone function to check Neo4j health with explicit parameters.
    Used by GraphDatabase for backward compatibility.
    
    Args:
        uri: Neo4j connection URI
        auth: Tuple of (username, password)
        
    Returns:
        True if Neo4j is healthy, False otherwise
    """
    username, password = auth
    logger.info(f"Checking Neo4j health at URI: {uri}")
    logger.info(f"Neo4j credentials - User: '{username}', Password: {'*' * len(password)} (length: {len(password)})")
    
    try:
        logger.debug(f"Creating Neo4j driver with URI: {uri}")
        with GraphDatabase.driver(uri, auth=auth) as driver:
            logger.debug("Verifying Neo4j connectivity...")
            driver.verify_connectivity()
            logger.info("Neo4j health check PASSED - connection successful")
        return True
    except ServiceUnavailable as e:
        logger.error(f"Neo4j health check FAILED - ServiceUnavailable: {e}")
        return False
    except Exception as e:
        logger.error(f"Neo4j health check FAILED - Exception: {e}")
        return False
