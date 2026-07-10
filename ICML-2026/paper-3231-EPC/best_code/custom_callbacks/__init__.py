# This file automatically imports all of the Callbacks that are found in files in the same directory as this file.

import importlib
import inspect
import os

from lightning.pytorch.callbacks import Callback

# Get the directory path where the __init__.py file resides
current_dir = os.path.dirname(__file__)

# Get a list of all Python files in the directory
files = [f for f in os.listdir(current_dir) if f.endswith(".py") and f != "__init__.py"]

# Import modules and dynamically add Callback classes to __all__
__all__ = []
for file in files:
    module_name = file[:-3]  # Remove the '.py' extension
    module = importlib.import_module(f".{module_name}", package=__name__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Callback) and obj.__module__ == module.__name__:
            __all__.append(name)
            globals()[name] = obj  # add custom callback to global workspace

# Write the imported class names to __all__ in __init__.py
__all__ = sorted(__all__)  # Sort the list alphabetically
