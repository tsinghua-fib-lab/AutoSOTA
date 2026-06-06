import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from litellm import ModelResponse
from loguru import logger
from pydantic import BaseModel

from ..core.globals import global_vars
from ..core.types.caching import DBCacheBase
from .utils import dict_to_pydantic, encode_to_uuid_v5, pydantic_to_dict


class DBCache(DBCacheBase):
    """SQLite-based caching implementation."""

    def __init__(
        self,
        db_path: str = "cache.db",
        max_size: Optional[int] = None,
        time_to_live: Optional[int] = None,
        cleanup_interval: Optional[int] = None,
    ):
        """Initialize the database cache.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.max_size = max_size or global_vars.configs.settings.caching.max_size
        self.time_to_live = (
            time_to_live or global_vars.configs.settings.caching.time_to_live
        )
        self.cleanup_interval = (
            cleanup_interval or global_vars.configs.settings.caching.cleanup_interval
        )
        self._init_db()
        self._init_cleanup_tracker()
        self._maybe_cleanup()

    def _init_db(self) -> None:
        """Initialize the database and create the cache table if it doesn't exist."""
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    uuid TEXT PRIMARY KEY,
                    key TEXT,
                    value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _init_cleanup_tracker(self) -> None:
        """Initialize the cleanup tracker table."""
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cleanup_tracker (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_cleanup DATETIME
                )
            """)
            # Insert initial cleanup time if not exists
            conn.execute("""
                INSERT OR IGNORE INTO cleanup_tracker (id, last_cleanup)
                VALUES (1, CURRENT_TIMESTAMP)
            """)

    def _should_cleanup(self, now: Optional[datetime] = None) -> bool:
        """Check if cleanup should be performed based on the last cleanup time."""
        with self.connect() as conn:
            now = now or datetime.now()
            cursor = conn.execute(
                "SELECT last_cleanup FROM cleanup_tracker WHERE id = 1"
            )
            last_cleanup = datetime.fromisoformat(cursor.fetchone()[0])
            return now - last_cleanup > timedelta(minutes=self.cleanup_interval)

    def _update_cleanup_time(self, now: Optional[datetime] = None) -> None:
        """Update the last cleanup timestamp."""
        with self.connect() as conn:
            now = now or datetime.now()
            conn.execute(
                """
                UPDATE cleanup_tracker 
                SET last_cleanup = ? 
                WHERE id = 1
                """,
                (now.isoformat(),),
            )

    def _maybe_cleanup(self, now: Optional[datetime] = None) -> None:
        """Perform cleanup if enough time has passed since last cleanup."""
        if self._should_cleanup(now=now):
            self._cleanup(now=now)
            self._update_cleanup_time(now=now)

    def _cleanup(self, now: Optional[datetime] = None) -> None:
        """Remove expired entries and enforce size limit."""
        with self.connect() as conn:
            # Remove expired entries
            now = now or datetime.now()
            expiry_time = now - timedelta(minutes=self.time_to_live)
            logger.info(f"Cleaning up expired caching entries before {expiry_time}")
            conn.execute(
                "DELETE FROM cache WHERE timestamp < ?", (expiry_time.isoformat(),)
            )

            # Enforce size limit by removing oldest entries
            conn.execute(
                """
                DELETE FROM cache 
                WHERE uuid IN (
                    SELECT uuid FROM cache 
                    ORDER BY timestamp ASC 
                    LIMIT max(0, (SELECT COUNT(*) FROM cache) - ?)
                )
            """,
                (self.max_size,),
            )

    def connect(self) -> sqlite3.Connection:
        """Create and return a database connection.

        Returns:
            sqlite3.Connection: Database connection object
        """
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def find(self, key: str) -> Optional[Any]:
        """Find a value in the cache by key.

        Args:
            key: Cache key to look up

        Returns:
            The cached value if found, None otherwise
        """
        self._maybe_cleanup()  # Only clean if needed

        uuid = encode_to_uuid_v5(key)

        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT key, value FROM cache WHERE uuid = ?", (uuid,)
            )
            result = cursor.fetchone()

            if result is None:
                return None

            stored_key, value = result
            assert stored_key == key
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value

    def insert(
        self, key: str, value: Any, timestamp: Optional[datetime] = None
    ) -> None:
        """Insert a value to the cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
        """
        self._maybe_cleanup()  # Only clean if needed

        uuid = encode_to_uuid_v5(key)

        serialized_value = json.dumps(value)
        if timestamp is None:
            timestamp = datetime.now()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (uuid, key, value, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (uuid, key, serialized_value, timestamp.isoformat()),
            )


def _serialize_args(args: Dict[str, Any]) -> str:
    args = args.copy()
    for k, v in args.items():
        # dump as schema if it is a pydantic model
        if isinstance(v, type):
            if issubclass(v, BaseModel):
                args[k] = v.model_json_schema()
            else:
                # TODO: convert to a schema
                logger.warning(f"Unknown type during serialization: {type(v)}")
                args[k] = str(v)
    return json.dumps(args)


def find_in_cache(
    args: Dict[str, Any], cache: Optional[DBCacheBase] = None
) -> Optional[ModelResponse]:
    """Find a value in the LLM cache by key.

    Args:
        args: The arguments of the completion.
        cache: The cache to search in. Defaults to the global LLM cache.

    Returns:
        The completion result if found, otherwise None.
    """
    cache = cache or global_vars.llm_cache
    if cache is None:
        return None
    if (
        args.get("temperature", 1.0) > 0.0
        and not global_vars.configs.settings.caching.allow_temp_greater_than_0
    ):
        return None
    # only cache the completions with temperature == 0
    value = cache.find(_serialize_args(args))
    if value is None:
        return None
    return dict_to_pydantic(value, ModelResponse)


def add_to_cache(
    args: Dict[str, Any], value: ModelResponse, cache: Optional[DBCacheBase] = None
) -> None:
    """Add a value to the LLM cache."""
    cache = cache or global_vars.llm_cache
    if cache is None:
        logger.warning("No cache to add to")
        return
    if (
        args.get("temperature", 1.0) > 0.0
        and not global_vars.configs.settings.caching.allow_temp_greater_than_0
    ):
        return
    args_str = _serialize_args(args)
    value_dict = pydantic_to_dict(value)
    logger.info(f"Adding to cache, args: {args_str}, value: {value_dict}")
    cache.insert(args_str, value_dict)
