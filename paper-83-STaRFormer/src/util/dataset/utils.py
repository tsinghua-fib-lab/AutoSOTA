import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Union


__all__ = [
    'BaseData',
]


class BaseData(object):
    def __init__(self, **kwargs) -> None:
        # Add additional features from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    
    def __repr__(self) -> str:
        """Returns a string representation of the Data object."""
        attr_str = []
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                attr_str.append(f"{k}={v.size()}")
            elif isinstance(v, list):
                attr_str.append(f'{k}=list([{len(v)}])')
            elif isinstance(v, pd.Series):
                attr_str.append(f'{k}=pd.Series([{len(v)}])')
            else:
                attr_str.append(f"{k}='{v}'")
        return f'{self.__class__.__name__}({", ".join(attr_str)})'


    def __getitem__(self, key):
        """Allows slicing the Data object as a dictionary."""
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise KeyError(f"Key '{key}' not found in Data object.")
    

    def __setitem__(self, key, value):
        """Allows adding attributes to the Data object."""
        setattr(self, key, value)
