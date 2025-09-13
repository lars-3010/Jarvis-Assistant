"""
Database adapters for alternative backends.

This module provides adapter implementations for various database backends,
allowing the system to work with different vector and graph databases.
"""

# Import adapters to register them with the factory
try:
    from .chroma_adapter import ChromaVectorDatabase
    __all__ = ["ChromaVectorDatabase"]
except ImportError:
    __all__ = []

try:
    from .pinecone_adapter import PineconeVectorDatabase
    __all__.append("PineconeVectorDatabase")
except ImportError:
    pass

# Note: ArangoGraphDatabase not yet implemented
# try:
#     from .arangodb_adapter import ArangoGraphDatabase
#     __all__.append("ArangoGraphDatabase")
# except ImportError:
#     pass
