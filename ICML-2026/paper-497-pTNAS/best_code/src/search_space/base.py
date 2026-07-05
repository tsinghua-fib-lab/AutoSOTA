"""
Base Search Space Class

This module contains the base class for search spaces with genetic operations.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseSearchSpace(ABC):
    """Base class for search spaces with genetic operations"""

    # These should be defined in subclasses
    blocks_choices: List[int] = []
    channel_choices: List[int] = []
    blocks_choices_large: List[int] = []
    channel_choices_large: List[int] = []

    @staticmethod
    @abstractmethod
    def mutate_architecture(architecture: List[int], mutation_rate: float = 0.3) -> List[int]:
        """
        Mutate an architecture by randomly changing some channels

        Args:
            architecture: Original architecture (list of channel sizes)
            mutation_rate: Probability of mutating each channel

        Returns:
            Mutated architecture
        """
        pass

    @staticmethod
    @abstractmethod
    def crossover_architectures(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Crossover two architectures to create two children

        Args:
            parent1: First parent architecture
            parent2: Second parent architecture

        Returns:
            Two child architectures
        """
        pass

