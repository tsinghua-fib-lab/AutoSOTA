import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from appl.caching import DBCache, encode_to_uuid_v5


@pytest.fixture
def cache():
    """Fixture to create a temporary database for each test."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_cache.db"
    cache = DBCache(str(db_path), max_size=10, time_to_live=10, cleanup_interval=10)
    yield cache
    if db_path.exists():
        db_path.unlink()


def test_init_creates_tables(cache: DBCache):
    """Test that initialization creates the required tables."""
    with sqlite3.connect(cache.db_path) as conn:
        # Check cache table exists
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='cache'
            """
        )
        assert cursor.fetchone() is not None

        # Check cleanup_tracker table exists
        cursor = conn.execute(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='cleanup_tracker'
            """
        )
        assert cursor.fetchone() is not None


def test_write_and_find(cache: DBCache):
    """Test basic write and find operations."""
    test_data = {"key1": "value1", "key2": [1, 2, 3], "key3": {"a": 1}}

    # Write test data
    for key, value in test_data.items():
        cache.insert(key, value)

    # Read and verify test data
    for key, value in test_data.items():
        assert cache.find(key) == value


def test_find_nonexistent(cache: DBCache):
    """Test finding a non-existent key returns None."""
    assert cache.insert("key1", "value1") is None
    assert cache.find("nonexistent_key") is None


def test_overwrite(cache: DBCache):
    """Test overwriting an existing key."""
    cache.insert("key1", "value1")
    cache.insert("key1", "value2")
    assert cache.find("key1") == "value2"


def test_cleanup_expired_entries(cache: DBCache):
    """Test that expired entries are cleaned up."""
    # Set current time
    current_time = datetime(2024, 1, 1, 12, 0)

    # Write entries with different timestamps
    cache.insert(
        "key1", "value1", current_time - timedelta(minutes=cache.time_to_live + 10)
    )  # Expired
    cache.insert(
        "key2", "value2", current_time - timedelta(minutes=cache.time_to_live - 10)
    )  # Not expired

    # Set last cleanup time to force cleanup
    cache._update_cleanup_time(
        now=current_time - timedelta(minutes=cache.cleanup_interval + 10)
    )

    # Force cleanup
    cache._cleanup(now=current_time)
    cache._update_cleanup_time(now=datetime.now())  # disable cleanup

    # Verify expired entry is gone but non-expired remains
    assert cache.find("key1") is None  # Should be cleaned up
    assert cache.find("key2") == "value2"  # Should still exist


def test_cleanup_size_limit(cache: DBCache):
    """Test that size limit is enforced during cleanup."""

    # Write more entries than the limit
    for i in range(11):
        cache.insert(f"key{i}", f"value{i}")

    # Force cleanup
    cache._cleanup()

    # Verify only the newest entries remain
    for i in range(11):
        if i < 1:
            assert cache.find(f"key{i}") is None
        else:
            assert cache.find(f"key{i}") == f"value{i}"


def test_cleanup_interval(cache: DBCache):
    """Test that cleanup only occurs after the cleanup interval."""
    # Set initial time
    current_time = datetime(2024, 1, 1, 12, 0)

    # Set initial cleanup time
    cache._update_cleanup_time(
        now=current_time - timedelta(minutes=cache.cleanup_interval - 10)
    )
    assert not cache._should_cleanup(now=current_time)

    # Check cleanup needed when time > cleanup_interval
    cache._update_cleanup_time(
        now=current_time - timedelta(minutes=cache.cleanup_interval + 10)
    )
    assert cache._should_cleanup(now=current_time)


def test_json_serialization(cache: DBCache):
    """Test that complex objects are properly JSON serialized."""
    test_obj = {
        "string": "test",
        "number": 123,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "null": None,
        "bool": True,
    }

    cache.insert("complex", test_obj)
    retrieved = cache.find("complex")

    assert retrieved == test_obj
    assert isinstance(retrieved, dict)


def test_invalid_json(cache: DBCache):
    """Test handling of invalid JSON data."""
    # Directly insert invalid JSON into database
    with cache.connect() as conn:
        conn.execute(
            """
            INSERT INTO cache (uuid, key, value) 
            VALUES (?, ?, ?)
            """,
            (encode_to_uuid_v5("invalid_json"), "invalid_json", "{invalid json}"),
        )

    # Should return raw string when JSON parsing fails
    assert cache.find("invalid_json") == "{invalid json}"
