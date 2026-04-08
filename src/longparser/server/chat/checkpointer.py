"""LangGraph MongoDB Checkpointer singleton.

Holds the global per-worker instance of the MongoDBSaver.
"""
import logging
from typing import Optional
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

logger = logging.getLogger(__name__)

_mongo_client: Optional[MongoClient] = None
_checkpointer: Optional[MongoDBSaver] = None


async def init_checkpointer(mongo_uri: str, db_name: str) -> None:
    """Initialize the MongoDB checkpointer on app startup."""
    global _mongo_client, _checkpointer
    if _checkpointer is not None:
        return

    logger.info("Initializing LangGraph MongoDB checkpointer...")
    # Initialize the sync MongoClient
    _mongo_client = MongoClient(mongo_uri)
    
    # Initialize the saver
    _checkpointer = MongoDBSaver(_mongo_client, db_name=db_name)


def get_checkpointer() -> MongoDBSaver:
    """Get the active checkpointer instance."""
    global _checkpointer
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Call init_checkpointer first.")
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the database checkpointer on app shutdown."""
    global _mongo_client, _checkpointer
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
    _checkpointer = None
    logger.info("LangGraph MongoDB checkpointer closed.")
