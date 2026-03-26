import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from matplotlib import colormaps
from typing import Dict, List, Union


__all__ = [
    'BaseData',
    'plot_client_data_distribution',
    'check_distribution_lengths',
    'compute_dirichlet_distribution',
    'get_dirichlet_distribution'
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


    #def pin_memory(self):
    #    raise NotImplementedError


    #def to(self, device):
    #    for k, v in self.__dict__.items():
    #        if isinstance(v, torch.Tensor):
    #            self.__dict__[k] = v.to(device)
            
            

def compute_dirichlet_distribution(
        dir_alpha: float, 
        num_classes: int, 
        num_clients: int, 
        amount: Union[List, int]
    ) -> np.ndarray:
    """ Return dirichlet distirubtion for partition of iid data to create client data. """
    print(amount)
    if isinstance(amount, list):
        amount = np.array([amount])[None, :].reshape(-1, 1) # (num_clients, 1)
    elif isinstance(amount, int):
        amount = np.array([amount]*num_clients).reshape(-1, 1) # (num_clients, 1)
    try: 
        assert len(amount.shape) == 2 and amount.shape == num_clients
    except Exception as e:
        print(f'{e}')
        amount = amount.reshape(-1,1)
        
    dirichlet_class_priors = np.random.dirichlet(
        alpha=[dir_alpha]*num_classes, 
        size=num_clients
    )*amount # (num_clients, num_classes)
    return dirichlet_class_priors.round().astype(np.int64)


def get_dirichlet_distribution(
        dir_alpha: float,
        num_classes: int,
        num_clients: int,
        amount_samples_per_client: Union[List , int ]
    ) -> np.ndarray:
    while True:
        with np.errstate(invalid='raise'): # checkt for floating point errors
            try:
                dirichlet_dist = compute_dirichlet_distribution(
                    dir_alpha, num_classes, num_clients, amount_samples_per_client
                )
                break
            except Exception as error:
                print(f'FloatingPointError: {error}')
                continue
    return dirichlet_dist


def plot_client_data_distribution(
    client_labels: Dict, 
    num_clients: int, 
    num_classes: int, 
    title: str=None, 
    xlabel: str='Num. of Samples',
    ylabel: str='Client',
    xticks: list=None,
    yticks: list=None,
    cmap_name: str='viridis',
    colors: list=None,
    bar_height: float=1.,
    legend_kwargs: Dict=None,
    store_path: str=None,
    dpi: int=300,
    ) -> np.ndarray:
    if cmap_name is not None:
        cmap = colormaps[cmap_name]
        colors = [cmap(i) for i in np.arange(0, 1, 1/(num_classes))]
    else:
        assert colors is not None
    assert len(colors) == num_classes
    res_array = np.zeros((num_clients, num_classes))
    #print(client_labels)
    for client, data in client_labels.items():
        unique, counts = np.unique(data, return_counts=True)
        #print(data, unique, counts)
        for u, c in zip(unique, counts):
            #print(u, c)
            res_array[client, u] = c

    res_array = res_array.transpose()

    for i in range(num_classes):
        plt.barh(range(num_clients), res_array[i], left=res_array[:i].sum(axis=0), color=colors[i], height=bar_height)
    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if xticks is not None: plt.xticks(xticks)
    if yticks is not None: plt.yticks(yticks)

    assert len(list(range(num_classes))) == len(colors)

    #plt.legend(list(range(num_classes)), colors)
    if legend_kwargs is None: 
        legend_kwargs = {'labels': list(range(num_classes))}
    plt.legend(**legend_kwargs)
    if store_path is not None: 
        dpi = 300 if dpi is None else dpi
        plt.savefig(store_path, dpi=dpi, bbox_inches='tight')
    plt.show()

    

    return res_array.transpose().astype(np.int64)


def check_distribution_lengths(distribution: np.ndarray, amount_samples_per_client):
    """ 
    if distribution lenghts differ from inputs values, adjust them but 
    adding or subtracting the difference to the argmin or argmax respectively. 

    e.g.
    input: (dirichlet distribution (num_clients x num_classes))
    [[ 12  47  92   9]
    [ 50  33   2  75]
    [ 24  85   5  45]
    [ 68  45  10  37]
    [ 17  23 113   8]]
    sum(axis=1): [160 160 159 160 161]

    output: 
    [[[ 12  47  92   9]
    [ 50  33   2  75]
    [ 24  85   6  45]
    [ 68  45  10  37]
    [ 17  23 112   8]]
    sum(axis=1): [160 160 160 160 160]

    """
    check  = distribution.sum(axis=1) == amount_samples_per_client
    for i, elem in enumerate(check):
        if elem == False:
            diff = distribution.sum(axis=1)[i] - amount_samples_per_client[i]
            #print(i, diff)
            if diff > 0:
                j = np.argmax(distribution[i]) # select max value and subtract diff
            if diff < 0:
                j = np.argmin(distribution[i]) # select max value and subtract diff
            distribution[i, j] = distribution[i, j] - diff
    
    return distribution
