import torch
from torch.utils.data import Dataset
import sys
from templates import flooding
import numpy as np
import json

class FloodingDataset(Dataset):
    def __init__(self, l, D, mode='train', json_path=None):
        """
        Initialize the training dataset for a graph flooding algorithm 
        with a given message length l in bits
        also the max number used in a square-graph coloring, d 
        (square graph coloring determines local slot ids for message passing )
        the training dataset created, works for any test time graph, whose max
        number of colores used in its square graph coloring is d
        """
        super().__init__()
        self.l = l
        self.D = D
        self.d = D**2+1

        self.mode = mode
        
        if mode == 'train':
            self.inputs, self.targets = generate_template_vectors(self.l,  self.d)
        elif mode == 'test':
            with open(json_path) as f:
                data = json.load(f)
            
            # Extract test samples
            test_samples = data['test_samples']
            
            # Convert to numpy arrays
            self.inputs = np.array([sample['encoded_features'] for sample in test_samples])
            self.targets = np.array([sample['target'] for sample in test_samples])
        else:
            raise ValueError("Mode must be either 'train' or 'test'")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.inputs)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            sample: Dictionary containing input and target, or tuple (input, target)
        """
        # Get the data
        input_sample = self.inputs[idx]
        target_sample = self.targets[idx]
        
        # Convert to PyTorch tensors
        input_tensor = torch.FloatTensor(input_sample)
        target_tensor = torch.FloatTensor(target_sample)
        
        return input_tensor, target_tensor

def generate_template_vectors(l, d):

    """
    create the x and y templates based on the instructions of the flooding algorithm
    gets l number of bits, d the max color used in square graph coloring

    """

    in_templ, out_templ = flooding.get_dataset(l,d)
    n0 = len(in_templ)
    X_train = np.eye(n0, dtype=np.float64)
    Y_train = np.array(out_templ, dtype=np.float64)

    return X_train, Y_train

def load_test_samples (json_path):

    with open(json_path) as f:
        data = json.load(f)
    
    # Extract test samples
    test_samples = data['test_samples']
    
    # Convert to numpy arrays
    inputs = np.array([sample['encoded_features'] for sample in test_samples], dtype =np.float64)
    targets = np.array([sample['target'] for sample in test_samples], dtype = np.float64)

    # Convert to PyTorch tensors
    inputs = torch.FloatTensor(inputs)
    targets = torch.FloatTensor(targets)

    return inputs, targets

def load_test_cases(json_path):

    with open(json_path) as f:
        data = json.load(f)
    
    # Extract test samples
    test_cases = data['test_cases']
    
    # Convert to numpy arrays
    cases = [case['encoded_inputs'] for case in test_cases]
    return cases