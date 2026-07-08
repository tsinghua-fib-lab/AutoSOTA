
import numpy as np
import torch
import time
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset, Subset
from models_sudoku import BLOSudokuLearnA, OptNetSudokuLearnA, SingleOptLayerSudoku

from utils_sudoku import decode_onehot, computeErr

if __name__=="__main__":
    n = 2
    board_side_len = n**2
    Qpenalty = 0.1
    batch_size = 1
    method = "lpgd"
    weight_path = f"../sudoku_results_32/{method}_per_5_model/model_epoch0.pt"
    
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #################### PREPARE DATA ##################
    
    train_data_dir_path = f"sudoku/data/{n}"
    features = torch.load(os.path.join(train_data_dir_path, "features.pt"))
    labels = torch.load(os.path.join(train_data_dir_path, "labels.pt"))
    features = torch.tensor(features, dtype=torch.float32).to(device)[:50000]
    labels   = torch.tensor(labels, dtype=torch.float32).to(device)[:50000]
    print(features.shape)
    print(labels.shape)
    #assert(1==0)
    
    # Create TensorDataset
    dataset = TensorDataset(features, labels)   
    
   # Fixed split indices
    num_samples = len(dataset)
    train_split = 0.9
    train_size = int(num_samples * train_split)
    test_size = num_samples - train_size

    # Use first 80% for training, last 20% for testing
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, num_samples))

    train_dataset = Subset(dataset, train_indices)
    test_dataset  = Subset(dataset, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    ###############################################
    model = SingleOptLayerSudoku(n, learnable_parts=["eq"], layer_type=method, Qpenalty=0.1, alpha=1000)
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        print(x.shape)
        
        pred = model(x)
        print("pred logit: ", pred)
        
        print(f"pred error: {computeErr(pred)}")
        print(f"GT error: {computeErr(y)}")
        
        
        print(f"test example: {i}")
        print(f"input puzzle: \n{decode_onehot(x[0])}")
        print(f"gt sol:\n{decode_onehot(y[0])}")
        print(f"pred sol: \n{decode_onehot(pred[0])}")
        
        input("Press Enter to continue...")
    
    