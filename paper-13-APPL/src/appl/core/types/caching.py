from abc import ABC, abstractmethod
from typing import Any, Optional


class DBCacheBase(ABC):
    """Base class for DBCache."""

    @abstractmethod
    def find(self, key: str) -> Optional[Any]:
        """Find a value in the cache by key.

        Args:
            key: Cache key to look up

        Returns:
            The cached value if found, None otherwise
        """
        pass

    @abstractmethod
    def insert(self, key: str, value: Any) -> None:
        """Insert a value to the cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
        """
        pass
