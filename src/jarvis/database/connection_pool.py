"""
Database connection pooling implementation.

This module provides connection pooling for database connections to improve
performance and resource utilization under heavy load.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Generic, TypeVar

import duckdb
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from jarvis.utils.errors import ServiceUnavailableError
from jarvis.utils.logging import setup_logging

logger = setup_logging(__name__)

T = TypeVar('T')  # Connection type

class ConnectionPool(Generic[T], ABC):
    """Abstract base class for database connection pools."""

    def __init__(
        self,
        max_size: int = 10,
        min_size: int = 1,
        max_idle_time: float = 60.0,
        connection_timeout: float = 5.0
    ):
        """Initialize the connection pool.
        
        Args:
            max_size: Maximum number of connections in the pool
            min_size: Minimum number of connections to maintain
            max_idle_time: Maximum time in seconds a connection can be idle
            connection_timeout: Timeout in seconds for acquiring a connection
        """
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout

        self._pool: list[dict[str, Any]] = []  # List of {connection, last_used} dicts
        self._in_use: int = 0
        self._lock = asyncio.Lock()
        self._initialized = False

        logger.info(f"Connection pool created: max_size={max_size}, min_size={min_size}")

    async def initialize(self) -> None:
        """Initialize the connection pool with minimum connections."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:  # Double-check under lock
                return

            logger.info(f"Initializing connection pool with {self.min_size} connections")
            for _ in range(self.min_size):
                connection = await self._create_connection()
                self._pool.append({
                    'connection': connection,
                    'last_used': time.time()
                })

            self._initialized = True
            logger.info(f"Connection pool initialized with {len(self._pool)} connections")

    @abstractmethod
    async def _create_connection(self) -> T:
        """Create a new database connection.
        
        Returns:
            A new database connection
        """
        pass

    @abstractmethod
    async def _close_connection(self, connection: T) -> None:
        """Close a database connection.
        
        Args:
            connection: Connection to close
        """
        pass

    @abstractmethod
    async def _test_connection(self, connection: T) -> bool:
        """Test if a connection is still valid.
        
        Args:
            connection: Connection to test
            
        Returns:
            True if connection is valid, False otherwise
        """
        pass

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[T]:
        """Acquire a connection from the pool.
        
        Returns:
            Database connection
            
        Raises:
            ServiceUnavailableError: If no connection could be acquired
        """
        if not self._initialized:
            await self.initialize()

        connection = await self._acquire_connection()
        try:
            yield connection
        finally:
            await self._release_connection(connection)

    async def _acquire_connection(self) -> T:
        """Get a connection from the pool or create a new one.
        
        Returns:
            Database connection
            
        Raises:
            ServiceUnavailableError: If no connection could be acquired
        """
        # Try to get a connection with timeout
        try:
            return await asyncio.wait_for(
                self._get_connection(),
                timeout=self.connection_timeout
            )
        except TimeoutError:
            logger.error("Timed out waiting for database connection")
            raise ServiceUnavailableError("Database connection pool exhausted")

    async def _get_connection(self) -> T:
        """Internal method to get or create a connection.
        
        Returns:
            Database connection
        """
        while True:
            async with self._lock:
                # Check for available connection in pool
                if self._pool:
                    conn_data = self._pool.pop(0)
                    connection = conn_data['connection']

                    # Check if connection is still valid
                    try:
                        is_valid = await self._test_connection(connection)
                        if not is_valid:
                            logger.debug("Discarding invalid connection from pool")
                            await self._close_connection(connection)
                            continue
                    except Exception as e:
                        logger.debug(f"Error testing connection: {e}")
                        try:
                            await self._close_connection(connection)
                        except Exception:
                            pass
                        continue

                    # Valid connection found
                    self._in_use += 1
                    return connection

                # No connection available, create new if under max_size
                if self._in_use < self.max_size:
                    try:
                        connection = await self._create_connection()
                        self._in_use += 1
                        return connection
                    except Exception as e:
                        logger.error(f"Failed to create new connection: {e}")
                        raise ServiceUnavailableError(f"Failed to create database connection: {e}")

            # Pool is at max capacity, wait and retry
            await asyncio.sleep(0.1)

    async def _release_connection(self, connection: T) -> None:
        """Return a connection to the pool.
        
        Args:
            connection: Connection to return to the pool
        """
        async with self._lock:
            try:
                # Check if connection is still valid
                is_valid = await self._test_connection(connection)

                if is_valid:
                    # Return to pool
                    self._pool.append({
                        'connection': connection,
                        'last_used': time.time()
                    })
                else:
                    # Close invalid connection
                    logger.debug("Closing invalid connection on release")
                    try:
                        await self._close_connection(connection)
                    except Exception as e:
                        logger.warning(f"Error closing invalid connection: {e}")
            except Exception as e:
                logger.warning(f"Error testing connection on release: {e}")
                try:
                    await self._close_connection(connection)
                except Exception:
                    pass
            finally:
                self._in_use -= 1

    async def maintenance(self) -> None:
        """Perform pool maintenance tasks, like closing idle connections."""
        if not self._initialized:
            return

        async with self._lock:
            now = time.time()
            idle_connections = []

            # Identify idle connections
            for i, conn_data in enumerate(self._pool):
                if now - conn_data['last_used'] > self.max_idle_time:
                    # Don't remove below min_size
                    if len(self._pool) - len(idle_connections) > self.min_size:
                        idle_connections.append((i, conn_data['connection']))

            # Remove and close idle connections (in reverse order to maintain indices)
            for i, connection in sorted(idle_connections, reverse=True):
                try:
                    logger.debug("Closing idle connection")
                    await self._close_connection(connection)
                    self._pool.pop(i)
                except Exception as e:
                    logger.warning(f"Error closing idle connection: {e}")

    async def close(self) -> None:
        """Close all connections in the pool."""
        if not self._initialized:
            return

        async with self._lock:
            logger.info(f"Closing connection pool ({len(self._pool)} connections)")

            # Close all pooled connections
            for conn_data in self._pool:
                try:
                    await self._close_connection(conn_data['connection'])
                except Exception as e:
                    logger.warning(f"Error closing connection during pool shutdown: {e}")

            self._pool.clear()
            self._initialized = False

            logger.info("Connection pool closed")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the connection pool.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            'max_size': self.max_size,
            'min_size': self.min_size,
            'available': len(self._pool),
            'in_use': self._in_use,
            'total': len(self._pool) + self._in_use
        }


class DuckDBPool(ConnectionPool[duckdb.DuckDBPyConnection]):
    """Connection pool for DuckDB connections."""

    def __init__(
        self,
        database_path: str,
        read_only: bool = False,
        max_size: int = 10,
        min_size: int = 1,
        max_idle_time: float = 60.0
    ):
        """Initialize the DuckDB connection pool.
        
        Args:
            database_path: Path to the DuckDB database file
            read_only: Whether to open connections in read-only mode
            max_size: Maximum number of connections
            min_size: Minimum number of connections
            max_idle_time: Maximum idle time in seconds
        """
        super().__init__(
            max_size=max_size,
            min_size=min_size,
            max_idle_time=max_idle_time
        )
        self.database_path = database_path
        self.read_only = read_only

        logger.info(f"DuckDB pool created: path={database_path}, read_only={read_only}")

    async def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection.
        
        Returns:
            DuckDB connection
            
        Raises:
            Exception: If connection creation fails
        """
        try:
            # DuckDB doesn't have async API, use loop.run_in_executor
            conn = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: duckdb.connect(self.database_path, read_only=self.read_only)
            )
            logger.debug("Created new DuckDB connection")
            return conn
        except Exception as e:
            logger.error(f"Failed to create DuckDB connection: {e}")
            raise

    async def _close_connection(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Close a DuckDB connection.
        
        Args:
            connection: DuckDB connection to close
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                connection.close
            )
        except Exception as e:
            logger.warning(f"Error closing DuckDB connection: {e}")

    async def _test_connection(self, connection: duckdb.DuckDBPyConnection) -> bool:
        """Test if a DuckDB connection is still valid.
        
        Args:
            connection: Connection to test
            
        Returns:
            True if connection is valid
        """
        try:
            # Test with a simple query
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: connection.execute("SELECT 1").fetchall()
            )
            return True
        except Exception:
            return False


class Neo4jPool(ConnectionPool[AsyncSession]):
    """Connection pool for Neo4j sessions."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        max_size: int = 5,
        min_size: int = 1,
        max_idle_time: float = 60.0
    ):
        """Initialize the Neo4j connection pool.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            max_size: Maximum number of connections
            min_size: Minimum number of connections
            max_idle_time: Maximum idle time in seconds
        """
        super().__init__(
            max_size=max_size,
            min_size=min_size,
            max_idle_time=max_idle_time
        )
        self.uri = uri
        self.username = username
        self.password = password
        self._driver: AsyncDriver | None = None

        logger.info(f"Neo4j pool created: uri={uri}")

    async def initialize(self) -> None:
        """Initialize the Neo4j connection pool."""
        if self._initialized:
            return

        # Create the driver first if needed
        if not self._driver:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                # Test driver connection
                await self._driver.verify_connectivity()
                logger.info("Neo4j driver connection verified")
            except Exception as e:
                logger.error(f"Failed to create Neo4j driver: {e}")
                raise ServiceUnavailableError(f"Failed to connect to Neo4j: {e}")

        # Initialize pool connections
        await super().initialize()

    async def _create_connection(self) -> AsyncSession:
        """Create a new Neo4j session.
        
        Returns:
            Neo4j session
        """
        if not self._driver:
            await self.initialize()

        try:
            session = self._driver.session()
            logger.debug("Created new Neo4j session")
            return session
        except Exception as e:
            logger.error(f"Failed to create Neo4j session: {e}")
            raise

    async def _close_connection(self, connection: AsyncSession) -> None:
        """Close a Neo4j session.
        
        Args:
            connection: Neo4j session to close
        """
        try:
            await connection.close()
        except Exception as e:
            logger.warning(f"Error closing Neo4j session: {e}")

    async def _test_connection(self, connection: AsyncSession) -> bool:
        """Test if a Neo4j session is still valid.
        
        Args:
            connection: Session to test
            
        Returns:
            True if session is valid
        """
        try:
            # Simple test query
            await connection.run("RETURN 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the connection pool and the Neo4j driver."""
        await super().close()

        if self._driver:
            try:
                await self._driver.close()
                self._driver = None
            except Exception as e:
                logger.warning(f"Error closing Neo4j driver: {e}")


class DatabaseConnectionManager:
    """Manager for database connection pools."""

    def __init__(
        self,
        duckdb_path: str,
        duckdb_read_only: bool = False,
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        duckdb_pool_size: int = 10,
        neo4j_pool_size: int = 5
    ):
        """Initialize the connection pool manager.
        
        Args:
            duckdb_path: Path to the DuckDB database file
            duckdb_read_only: Whether to open DuckDB in read-only mode
            neo4j_uri: Optional Neo4j URI
            neo4j_username: Optional Neo4j username
            neo4j_password: Optional Neo4j password
            duckdb_pool_size: Maximum DuckDB connections
            neo4j_pool_size: Maximum Neo4j connections
        """
        self.duckdb_pool = DuckDBPool(
            database_path=duckdb_path,
            read_only=duckdb_read_only,
            max_size=duckdb_pool_size,
            min_size=1
        )

        # Neo4j pool is optional
        self.neo4j_pool = None
        if neo4j_uri and neo4j_username and neo4j_password:
            self.neo4j_pool = Neo4jPool(
                uri=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
                max_size=neo4j_pool_size,
                min_size=1
            )

        self._maintenance_task = None

        logger.info("DatabaseConnectionManager initialized")

    async def initialize(self) -> None:
        """Initialize all connection pools."""
        await self.duckdb_pool.initialize()

        if self.neo4j_pool:
            try:
                await self.neo4j_pool.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j pool: {e}")

    def start_maintenance(self, interval: float = 60.0) -> None:
        """Start periodic maintenance task.
        
        Args:
            interval: Maintenance interval in seconds
        """
        if self._maintenance_task is not None:
            return

        async def maintenance_loop():
            while True:
                try:
                    await self.duckdb_pool.maintenance()
                    if self.neo4j_pool:
                        await self.neo4j_pool.maintenance()
                except Exception as e:
                    logger.error(f"Error in pool maintenance: {e}")
                await asyncio.sleep(interval)

        self._maintenance_task = asyncio.create_task(maintenance_loop())
        logger.info(f"Started connection pool maintenance task (interval={interval}s)")

    def stop_maintenance(self) -> None:
        """Stop the maintenance task."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            self._maintenance_task = None
            logger.info("Stopped connection pool maintenance task")

    @asynccontextmanager
    async def get_duckdb_connection(self) -> AsyncIterator[duckdb.DuckDBPyConnection]:
        """Get a DuckDB connection from the pool.
        
        Returns:
            DuckDB connection
        """
        async with self.duckdb_pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def get_neo4j_session(self) -> AsyncIterator[AsyncSession]:
        """Get a Neo4j session from the pool.
        
        Returns:
            Neo4j session
            
        Raises:
            ServiceUnavailableError: If Neo4j is not configured
        """
        if not self.neo4j_pool:
            raise ServiceUnavailableError("Neo4j is not configured")

        async with self.neo4j_pool.acquire() as session:
            yield session

    async def close(self) -> None:
        """Close all connection pools."""
        self.stop_maintenance()

        await self.duckdb_pool.close()

        if self.neo4j_pool:
            await self.neo4j_pool.close()

        logger.info("DatabaseConnectionManager closed")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for all connection pools.
        
        Returns:
            Dictionary with pool statistics
        """
        stats = {
            'duckdb_pool': self.duckdb_pool.get_stats()
        }

        if self.neo4j_pool:
            stats['neo4j_pool'] = self.neo4j_pool.get_stats()

        return stats
