import pickle 
import ujson
import torch


__all__ = [
    "store_asset",
    "load_asset",
]

def store_asset(format: str='pickle', file_path: str=None, obj: object=None, write: str='w') -> None:
    """
    Stores an object to a file in the specified format.

    This method saves the given object to a file using the specified format. 
    Supported formats are 'pickle', 'pt' (PyTorch), and 'json'.

    Args:
        format (str): The format in which to store the object. Default is 'pickle'.
        file_path (str): The path to the file where the object will be stored. 
                        This argument cannot be None.
        obj (object): The object to be stored.
        write (str): The mode in which to open the file for writing. Default is 'w'.

    Raises:
        AssertionError: If `file_path` is None.
        RuntimeError: If the specified format is not implemented.

    Returns:
        None: This method performs the storage operation and does not return any value.
    """
    assert file_path is not None, f'file_path cannot be None'
    if format == 'pickle':
        with open(file_path, write) as file:
            pickle.dump(obj, file)
    elif format == 'pt':
        torch.save(obj, file_path)

    elif format == 'json':
        with open(file_path, write) as f:
            ujson.dump(obj, f)
    else:
        raise RuntimeError(f'{format} not implemented!')

def load_asset(format: str='pickle', file_path: str=None, buffer: bool=False, read: str='r') -> object:
    """
    Loads an object from a file in the specified format.

    This method loads an object from a file using the specified format. 
    Supported formats are 'pickle', 'pt' (PyTorch), and 'json'.

    Args:
        format (str): The format in which the object is stored. Default is 'pickle'.
        file_path (str): The path to the file from which the object will be loaded. 
                        This argument cannot be None.
        buffer (bool): If True, load the object from a buffer. Default is False.
        read (str): The mode in which to open the file for reading. Default is 'r'.

    Raises:
        AssertionError: If `file_path` is None.
        RuntimeError: If the specified format is not implemented.

    Returns:
        object: The object loaded from the file.
    """
    assert file_path is not None, f'file_path cannot be None'
    if format == 'pickle':
        with open(file_path, read) as file:
            if buffer:
                asset = pickle.loads(file)
            else:
                asset = pickle.load(file)

    elif format == 'pt':
        asset = torch.load(file_path)

    elif format == 'json':
        with open(file_path, read) as file:
            if buffer:
                asset = ujson.loads(file)
            else:
                asset = ujson.load(file)
    else:
        raise RuntimeError(f'{format} not implemented!')
    
    return asset