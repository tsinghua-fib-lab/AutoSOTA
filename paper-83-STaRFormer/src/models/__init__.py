from .fcn import *
from .global_modules import *
from .gru import *
from .lstm import *
from .rnn import *
from .transformer import *

import importlib
import pkgutil

from typing import Dict

def get_all_models() -> Dict[str, object]:
    """
    Dynamically discover and retrieve all model classes in the current package.

    This function iterates through all modules in the current package (as determined by __name__),
    imports them, and collects all classes defined within these modules. The resulting dictionary
    maps class names to class objects.

    Returns:
        dict: A dictionary where keys are class names (str) and values are the corresponding class objects.

    Notes:
        - This function assumes that models are defined as classes inside the modules of the current package.
        - The function uses pkgutil and importlib to scan and import modules dynamically.

    Example:
        >>> models = get_all_models()
        >>> 'LSTMNet' in models
        True
    """
    models = {}
    package_name = __name__  # This will be 'src.models'

    # Iterate through all modules in the package
    for _, module_name, _ in pkgutil.iter_modules(importlib.import_module(package_name).__path__):
        module = importlib.import_module(f"{package_name}.{module_name}")
        
        # Assuming models are defined as classes in the module
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type):  # Check if it's a class
                models[name] = obj

    return models


def get_rnn_models() -> Dict[str, object]:
    """Retrieve all RNN-related model classes from the current package.

    This function calls `get_all_models()` and filters the result to include only those class names
    that start with 'LSTM', 'RNN', or 'GRU'.

    Returns:
        dict: A dictionary where keys are class names (str) related to RNNs and values are the class objects.

    Example:
        >>> rnn_models = get_rnn_models()
        >>> 'LSTMNet' in rnn_models
        True
    """
    all_models = get_all_models()
    return {k: v  for k, v in all_models.items() \
            if k.startswith('LSTM') or k.startswith('RNN') or k.startswith('GRU')}
