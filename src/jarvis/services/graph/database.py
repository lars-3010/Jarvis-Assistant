"""
Graph Database Service for Neo4j Operations
Handles all interactions with the Neo4j graph database.
"""
import logging
import os
from typing import Dict, Any, List, Optional

from neo4j import GraphDatabase as Neo4jGraphDatabase, Driver, Transaction
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from jarvis.services.health import check_neo4j_health
from jarvis.utils.config import JarvisSettings
from jarvis.utils.errors import ServiceUnavailableError, JarvisError
from jarvis.core.interfaces import IGraphDatabase

logger = logging.getLogger(__name__)


class GraphDatabase(IGraphDatabase):
    """Service for all Neo4j graph database operations"""

    def __init__(self, settings: JarvisSettings):
        """
        Initialize Neo4j connection and schema based on settings.

        Args:
            settings: The application settings object.
        """
        self.driver: Optional[Driver] = None
        self.enabled = settings.graph_enabled

        if not self.enabled:
            logger.info("Neo4j integration is disabled by configuration.")
            return

        try:
            logger.info(f"Attempting Neo4j connection with URI: {settings.neo4j_uri}")
            logger.info(f"Neo4j user: {settings.neo4j_user}")
            logger.info(f"Neo4j password: {'*' * len(settings.neo4j_password)} (length: {len(settings.neo4j_password)})")
            
            # Check environment variables for debugging
            env_password = os.environ.get("JARVIS_NEO4J_PASSWORD")
            env_user = os.environ.get("JARVIS_NEO4J_USER")
            env_uri = os.environ.get("JARVIS_NEO4J_URI")
            
            logger.debug(f"Environment variables - Password: {env_password}, User: {env_user}, URI: {env_uri}")
            
            is_healthy = check_neo4j_health(
                uri=settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )

            if not is_healthy:
                logger.error(f"Neo4j health check failed for {settings.neo4j_uri} with user '{settings.neo4j_user}'")
                raise ServiceUnavailable(f"Neo4j connection to {settings.neo4j_uri} failed.")

            logger.info(f"Neo4j health check passed, creating driver connection...")
            self.driver = Neo4jGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            
            logger.info("Testing driver connectivity...")
            self.driver.verify_connectivity()
            logger.info("Neo4j driver connectivity verified successfully!")
            
            self._initialize_schema()
            logger.info(f"Connected to Neo4j at {settings.neo4j_uri}")

        except (ServiceUnavailable, Neo4jError) as e:
            logger.warning(f"Neo4j connection to {settings.neo4j_uri} failed. Graph features will be unavailable. Reason: {e}")
            self.driver = None

    @classmethod
    def from_config(cls, config) -> "GraphDatabase":
        """Create GraphDatabase instance from DatabaseConfig.
        
        Args:
            config: DatabaseConfig instance with Neo4j configuration
            
        Returns:
            GraphDatabase instance
        """
        from jarvis.utils.config import JarvisSettings
        
        # Create a temporary settings object with the config values
        settings = JarvisSettings(
            graph_enabled=config.get('enabled', True),
            neo4j_uri=config.get('uri', 'bolt://localhost:7687'),
            neo4j_user=config.get('username', 'neo4j'),
            neo4j_password=config.get('password', 'password')
        )
        
        return cls(settings)

    @property
    def is_healthy(self) -> bool:
        """Check if the Neo4j connection is alive and configured."""
        return self.driver is not None

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self.close()
        return False  # Don't suppress exceptions

    def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indices"""
        if not self.is_healthy:
            return

        with self.driver.session() as session:
            try:
                session.run("""
                    CREATE CONSTRAINT note_path_unique IF NOT EXISTS
                    FOR (n:Note) REQUIRE n.path IS UNIQUE
                """)

                session.run("""
                    CREATE CONSTRAINT tag_name_unique IF NOT EXISTS
                    FOR (t:Tag) REQUIRE t.name IS UNIQUE
                """)

                session.run("""
                    CREATE INDEX note_title_index IF NOT EXISTS
                    FOR (n:Note) ON (n.title)
                """)

                logger.info("Neo4j schema initialized successfully")
            except Neo4jError as e:
                logger.error(f"Error initializing Neo4j schema: {e}")

    def create_or_update_note(self, note_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a note with all its relationships
        """
        if not self.is_healthy:
            raise ServiceUnavailableError("Graph database is not available.")

        if "path" not in note_data or "title" not in note_data:
            raise ValueError("Note must have path and title")

        try:
            with self.driver.session() as session:
                result = session.execute_write(self._create_note_tx, note_data)

                if "relationships" in note_data and note_data["relationships"]:
                    rel_result = session.execute_write(
                        self._create_relationships_tx,
                        note_data["path"],
                        note_data["relationships"]
                    )
                    result["relationships_created"] = rel_result

                if "links" in note_data and note_data["links"]:
                    link_result = session.execute_write(
                        self._create_links_tx,
                        note_data["path"],
                        note_data["links"]
                    )
                    result["links_created"] = link_result

                return result
        except Neo4jError as e:
            logger.error(f"Error creating/updating note: {e}")
            return {"error": str(e)}

    def _create_note_tx(self, tx: Transaction, note_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transaction function to create or update a note"""
        logger.debug(f"Creating/updating note: {note_data['path']}")
        
        query = """
        MERGE (n:Note {path: $path})
        ON CREATE SET
            n.title = $title,
            n.name = $name,
            n.content = $content,
            n.created = timestamp()
        ON MATCH SET
            n.title = $title,
            n.name = $name,
            n.content = $content,
            n.updated = timestamp()
        """

        params = {
            "path": note_data["path"],
            "title": note_data["title"],
            "name": note_data.get("name", note_data["title"]),  # Fallback to title if name not provided
            "content": note_data.get("content", ""),  # Fallback to empty string if content not provided
        }
        
        logger.debug(f"Database params for {note_data['path']}: title='{params['title']}', name='{params['name']}', content_length={len(params['content'])}")

        if "tags" in note_data and note_data["tags"]:
            query += ", n.tags = $tags"
            params["tags"] = note_data["tags"]
            logger.debug(f"Adding tags to {note_data['path']}: {note_data['tags']}")

            tag_query = """
            MATCH (n:Note {path: $path})
            UNWIND $tags AS tagName
            MERGE (t:Tag {name: tagName})
            MERGE (n)-[:HAS_TAG]->(t)
            """
            tx.run(tag_query, {"tags": note_data["tags"], "path": note_data["path"]})

        result = tx.run(query, params)
        summary = result.consume()
        
        operation = "created" if summary.counters.nodes_created > 0 else "updated"
        logger.debug(f"Note {operation}: {note_data['path']}")

        return {
            "operation": operation,
            "path": note_data["path"],
            "title": note_data["title"]
        }

    def _create_relationships_tx(
        self, tx: Transaction,
        source_path: str,
        relationships: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, int]:
        """Transaction function to create semantic relationships"""
        relationship_counts = {}

        for rel_type, targets in relationships.items():
            neo4j_rel_type = rel_type.replace("::", "").upper()

            for target_info in targets:
                target_path = target_info["target"]

                query = f"""
                MATCH (source:Note {{path: $source_path}})
                MERGE (target:Note {{path: $target_path}})
                ON CREATE SET target.title = $target_title
                MERGE (source)-[r:{neo4j_rel_type} {{type: $rel_type}}]->(target)
                RETURN count(r) as rel_count
                """

                target_title = target_info.get("alias")
                if not target_title:
                    filename = os.path.basename(target_path)
                    target_title = os.path.splitext(filename)[0]

                result = tx.run(
                    query,
                    {
                        "source_path": source_path,
                        "target_path": target_path,
                        "target_title": target_title,
                        "rel_type": rel_type
                    }
                )

                record = result.single()
                if record:
                    relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + record["rel_count"]

        return relationship_counts

    def _create_links_tx(
        self, tx: Transaction,
        source_path: str,
        links: List[Dict[str, str]]
    ) -> int:
        """Transaction function to create regular markdown link relationships"""
        query = """
        MATCH (source:Note {path: $source_path})
        MERGE (target:Note {path: $target_path})
        ON CREATE SET target.title = $target_title
        MERGE (source)-[r:LINKS_TO]->(target)
        RETURN count(r) as link_count
        """

        link_count = 0
        for link in links:
            target_path = link["target"]

            target_title = link.get("alias")
            if not target_title:
                filename = os.path.basename(target_path)
                target_title = os.path.splitext(filename)[0]

            result = tx.run(
                query,
                {
                    "source_path": source_path,
                    "target_path": target_path,
                    "target_title": target_title
                }
            )

            record = result.single()
            if record:
                link_count += record["link_count"]

        return link_count

    def get_note_graph(self, path: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get a knowledge graph centered on a specific note
        """
        if not self.is_healthy:
            raise ServiceUnavailableError("Graph database is not available.")

        logger.info(f"Getting knowledge graph for path: '{path}' with depth: {depth}")
        
        with self.driver.session() as session:
            try:
                # First, let's see what paths are available in the database
                available_paths = session.execute_read(self._get_available_paths_tx, path)
                logger.info(f"Available paths matching '{path}': {available_paths}")
                
                result = session.execute_read(self._get_knowledge_graph_tx, path, depth)
                logger.debug(f"Knowledge graph result: {result}")
                return result
            except Neo4jError as e:
                logger.error(f"Error getting knowledge graph: {e}")
                return {"nodes": [], "relationships": []}

    def _get_available_paths_tx(self, tx: Transaction, search_term: str) -> List[str]:
        """Transaction function to get available paths that match a search term"""
        # First try exact match
        exact_query = "MATCH (n:Note {path: $path}) RETURN n.path as path"
        exact_result = tx.run(exact_query, path=search_term).data()
        
        if exact_result:
            return [record["path"] for record in exact_result]
        
        # Then try partial matches (case-insensitive)
        partial_query = """
        MATCH (n:Note) 
        WHERE toLower(n.path) CONTAINS toLower($search_term) 
           OR toLower(n.title) CONTAINS toLower($search_term)
           OR toLower(n.content) CONTAINS toLower($search_term)
        RETURN n.path as path, n.title as title
        LIMIT 10
        """
        partial_result = tx.run(partial_query, search_term=search_term).data()
        
        return [f"{record['path']} (title: {record['title']})" for record in partial_result]

    def _get_knowledge_graph_tx(self, tx: Transaction, path: str, depth: int) -> Dict[str, Any]:
        """Transaction function to get knowledge graph"""
        query = f"""
        MATCH path = (center:Note {{path: $path}})-[*0..{depth}]-(related:Note)
        WITH nodes(path) as nodes, relationships(path) as rels
        UNWIND nodes as node
        WITH collect(distinct node) as allNodes, rels
        UNWIND rels as rel
        WITH allNodes, collect(distinct rel) as allRels

        RETURN
            [node IN allNodes | {{
                id: toString(id(node)),
                label: node.title,
                path: node.path,
                tags: node.tags,
                center: node.path = $path
            }}] as nodes,
            [rel IN allRels | {{
                id: toString(id(rel)),
                source: toString(id(startNode(rel))),
                target: toString(id(endNode(rel))),
                type: type(rel),
                original_type: rel.type
            }}] as relationships
        """

        result = tx.run(query, {"path": path})
        record = result.single()

        if not record:
            return {"nodes": [], "relationships": []}

        return {
            "nodes": record["nodes"],
            "relationships": record["relationships"]
        }