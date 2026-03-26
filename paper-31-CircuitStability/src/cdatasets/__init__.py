"""Dataset management module for the circuit stability framework.

This module provides a factory pattern for creating datasets with configurable
parameters and a unified interface for different prompting strategies.

Available datasets:
- arith: Arithmetic computation tasks
- bool: Boolean logic reasoning
- csense: Common sense question answering
- dyck: Dyck language recognition
- date: Date understanding tasks
- sports: Sports understanding tasks
- movie: Movie recommendation tasks

Available prompting strategies:
- zero-shot: Direct question answering
- few-shot: Learning from examples
- chain-of-thought: Step-by-step reasoning
"""

from .arith_dataset import ArithDataset
from .bool_dataset import BooleanDataset
from .csense_dataset import CommonSenseDataset
from .dyck_dataset import DyckDataset
from .date_dataset import DateDataset
from .sports_dataset import SportsDataset
from .movie_dataset import MovieDataset
from .prompt_dataaset import PromptDataset
from .prompts import PromptFormatter, ZeroShot, FewShot, ChainOfThought


class DatasetBuilder:
    """Factory class for creating datasets with configurable parameters.
    
    This class provides a fluent interface for creating datasets with specific
    parameters. It supports all built-in dataset types and allows for easy
    parameter configuration through method chaining.
    
    The builder pattern allows for clean, readable code when creating datasets
    with multiple parameters.
    
    Attributes:
        ids: Dictionary mapping dataset names to their corresponding classes
        cls: The dataset class to instantiate
        params: Dictionary of parameters to pass to the dataset constructor
    
    Example:
        >>> dataset = DatasetBuilder("bool")\
        ...     .set_param("length", 5)\
        ...     .set_param("complexity", "medium")\
        ...     .build()
        >>> print(f"Created dataset with {len(dataset)} examples")
    """
    
    # Registry of available dataset classes
    ids = {
        "arith": ArithDataset,
        "bool": BooleanDataset,
        "csense": CommonSenseDataset,
        "dyck": DyckDataset,
        "date": DateDataset,
        "sports": SportsDataset,
        "movie": MovieDataset,
    }

    def __init__(self, name: str):
        """Initialize the dataset builder with a dataset name.
        
        Args:
            name: Name of the dataset to create. Must be one of the keys
                 in the ids dictionary.
                 
        Raises:
            ValueError: If the dataset name is not recognized.
        """
        if name not in self.ids:
            available = ", ".join(self.ids.keys())
            raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
        
        self.cls = self.ids[name]
        self.params = {}

    def set_param(self, name: str, val):
        """Set a parameter for the dataset.
        
        This method allows for method chaining to set multiple parameters
        in a readable way.
        
        Args:
            name: Parameter name to set
            val: Parameter value
            
        Returns:
            Self for method chaining
            
        Example:
            >>> builder = DatasetBuilder("bool")\
            ...     .set_param("length", 5)\
            ...     .set_param("complexity", "medium")
        """
        self.params[name] = val
        return self

    def build(self):
        """Build and return the configured dataset.
        
        Creates an instance of the dataset class with all the parameters
        that have been set using set_param().
        
        Returns:
            Configured dataset instance
            
        Example:
            >>> dataset = DatasetBuilder("arith")\
            ...     .set_param("max_digits", 3)\
            ...     .set_param("operations", ["+", "-"])\
            ...     .build()
        """
        return self.cls(**self.params)


def get_dataset_strategy(name: str, **kwargs):
    """Get a dataset builder for the specified dataset.
    
    This is a convenience function that creates a DatasetBuilder instance
    for the given dataset name. It's equivalent to DatasetBuilder(name).
    
    Args:
        name: Name of the dataset to create
        **kwargs: Additional keyword arguments (currently unused)
        
    Returns:
        DatasetBuilder instance for the specified dataset
        
    Example:
        >>> builder = get_dataset_strategy("bool")
        >>> dataset = builder.set_param("length", 5).build()
    """
    return DatasetBuilder(name)


def get_prompt_strategy(name: str, **kwargs):
    """Get a prompting strategy by name.
    
    Creates and returns an instance of the specified prompting strategy
    with the given parameters.
    
    Args:
        name: Name of the prompting strategy. Must be one of:
              'zero-shot', 'few-shot', 'chain-of-thought'
        **kwargs: Parameters for the prompting strategy (e.g., 'shots' for few-shot)
        
    Returns:
        PromptFormatter instance for the specified strategy
        
    Raises:
        ValueError: If the strategy name is not recognized
        
    Example:
        >>> formatter = get_prompt_strategy("few-shot", shots=3)
        >>> zero_shot = get_prompt_strategy("zero-shot")
    """
    if name == "zero-shot":
        return ZeroShot()
    elif name == "few-shot":
        return FewShot(kwargs["shots"])
    elif name == "chain-of-thought":
        return ChainOfThought()
    else:
        available = "zero-shot, few-shot, chain-of-thought"
        raise ValueError(
            f"Unknown prompt strategy: {name}; must be one of {available}."
        )


# Add convenience methods to classes for easier access
DatasetBuilder.get_strategy = get_dataset_strategy
PromptFormatter.get_strategy = get_prompt_strategy
PromptFormatter.ids = {
    "zero-shot": ZeroShot,
    "few-shot": FewShot,
    "chain-of-thought": ChainOfThought,
}
