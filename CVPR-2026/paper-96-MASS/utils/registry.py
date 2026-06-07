"""Central registry used to construct MASS components by config name.

Models, datasets, trainers, losses, optimizers, and schedulers register
themselves here so YAML configs can refer to them by a stable string key.
"""

from typing import Dict, Callable, Any, Optional, Type, Union, List
import inspect


class Registry:
    """
    A registry to store objects by name, supporting multiple categories.
    
    Examples:
    ```
    # Register a model
    @Registry.register("model")
    class MyModel(nn.Module):
        pass
    
    # Register with a custom name
    @Registry.register("model", name="fancy_model")
    class MyComplexModel(nn.Module):
        pass
    
    model_cls = Registry.get("model", "MyModel")
    fancy_model_cls = Registry.get("model", "fancy_model")
    ```
    """
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, category: str, name: Optional[str] = None) -> Callable:
        """
        Register an object in the registry under the specified category.
        
        Args:
            category: Category name (e.g., "model", "dataset", "optimizer")
            name: Optional custom name, defaults to the object's __name__
            
        Returns:
            Decorator function that registers the object
        """
        def _register(obj: Any) -> Any:
            nonlocal name
            key = name or obj.__name__
            
            if category not in cls._registry:
                cls._registry[category] = {}
                
            if key in cls._registry[category]:
                raise ValueError(f"Object with name '{key}' already registered in '{category}'")
                
            cls._registry[category][key] = obj
            
            return obj
        
        def decorator(obj_or_name: Any = None) -> Any:
            # Called as @register
            if obj_or_name is not None and not isinstance(obj_or_name, str):
                return _register(obj_or_name)
                
            # Called as @register("name")
            nonlocal name
            if isinstance(obj_or_name, str):
                name = obj_or_name
                
            return _register
            
        return decorator
    
    @classmethod
    def get(cls, category: str, name: str) -> Any:
        """
        Retrieve an object from the registry.
        
        Args:
            category: Category name (e.g., "model", "dataset")
            name: Object name
            
        Returns:
            The registered object
            
        Raises:
            KeyError: If the category or name doesn't exist
        """
        if category not in cls._registry:
            raise KeyError(f"Category '{category}' not found in registry")
            
        if name not in cls._registry[category]:
            raise KeyError(f"'{name}' not found in '{category}' registry")
            
        return cls._registry[category][name]
    
    @classmethod
    def list(cls, category: str) -> List[str]:
        """
        List all registered names in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of registered object names in the category
        """
        if category not in cls._registry:
            return []
            
        return list(cls._registry[category].keys())
    
    @classmethod
    def contains(cls, category: str, name: str) -> bool:
        """
        Check if an object is registered.
        
        Args:
            category: Category name
            name: Object name
            
        Returns:
            True if the object is registered, False otherwise
        """
        return category in cls._registry and name in cls._registry[category]


# Convenience functions for common registry operations

def register_model(name: Optional[str] = None) -> Callable:
    """Register a model in the registry."""
    return Registry.register("model", name)


def register_dataset(name: Optional[str] = None) -> Callable:
    """Register a dataset in the registry."""
    return Registry.register("dataset", name)


def register_optimizer(name: Optional[str] = None) -> Callable:
    """Register an optimizer in the registry."""
    return Registry.register("optimizer", name)


def register_scheduler(name: Optional[str] = None) -> Callable:
    """Register a learning rate scheduler in the registry."""
    return Registry.register("scheduler", name)


def register_criterion(name: Optional[str] = None) -> Callable:
    """Register a loss function in the registry."""
    return Registry.register("criterion", name)

def register_trainer(name: Optional[str] = None) -> Callable:
    """Register a trainer in the registry."""
    return Registry.register("trainer", name)


def get_model(name: str) -> Any:
    """Get a model from the registry."""
    return Registry.get("model", name)


def get_dataset(name: str) -> Any:
    """Get a dataset from the registry."""
    return Registry.get("dataset", name)


def get_optimizer(name: str) -> Any:
    """Get an optimizer from the registry."""
    return Registry.get("optimizer", name)


def get_scheduler(name: str) -> Any:
    """Get a learning rate scheduler from the registry."""
    return Registry.get("scheduler", name)


def get_criterion(name: str) -> Any:
    """Get a loss function from the registry."""
    return Registry.get("criterion", name)


def get_trainer(name: str) -> Any:
    """Get a trainer from the registry."""
    return Registry.get("trainer", name)


def list_models() -> List[str]:
    """List all registered model names."""
    return Registry.list("model")


def list_datasets() -> List[str]:
    """List all registered dataset names."""
    return Registry.list("dataset")


def list_optimizers() -> List[str]:
    """List all registered optimizer names."""
    return Registry.list("optimizer")


def list_schedulers() -> List[str]:
    """List all registered scheduler names."""
    return Registry.list("scheduler")


def list_criteria() -> List[str]:
    """List all registered loss function names."""
    return Registry.list("criterion")


def list_trainers() -> List[str]:
    """List all registered trainer names."""
    return Registry.list("trainer")


def build_model_from_config(config: Dict[str, Any]) -> Any:
    """
    Build a model instance from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'type' and optional kwargs
        
    Returns:
        Instantiated model
    """
    config = config.copy()
    model_type = config.pop("type")
    model_cls = get_model(model_type)
    model = model_cls(**config)
    return model


def build_from_config(category: str, config: Dict[str, Any]) -> Any:
    """
    Build an object from a configuration dictionary.
    
    Args:
        category: Registry category (e.g., "model", "dataset")
        config: Configuration dictionary with 'type' and optional kwargs
        
    Returns:
        Instantiated object
    """
    config = config.copy()  # Avoid modifying the original
    obj_type = config.pop("type")
    obj_cls = Registry.get(category, obj_type)
    
    signature = inspect.signature(obj_cls.__init__)
    valid_params = signature.parameters.keys()
    
    if "self" in valid_params:
        valid_params = [p for p in valid_params if p != "self"]
    
    # Keep only valid kwargs
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    
    # Instantiate the object
    obj = obj_cls(**filtered_config)
    return obj
