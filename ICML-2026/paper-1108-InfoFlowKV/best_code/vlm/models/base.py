"""Base classes for model patches."""

from abc import ABC, abstractmethod


class BasePatch(ABC):
    """Base class for all model patches."""

    @abstractmethod
    def apply(self):
        """Apply the patch to the model."""
        pass

    @abstractmethod
    def remove(self):
        """Remove the patch and restore original behavior."""
        pass

    def clear(self):
        """Clear any captured data."""
        pass

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False
