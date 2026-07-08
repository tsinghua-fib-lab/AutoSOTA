import numpy as np
import torch


import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset, Subset

from models_sudoku import OptNetSudokuLearnA, CvxpyLayerSudokuLearnA, SingleOptLayerSudoku



    
if __name__ == '__main__':
   
    seed = 1
    batch_size = 1 #32
    
    n = 2
    board_side_len = n**2
    Qpenalty = 0.1
    
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #################### PREPARE DATA ##################
    
    train_data_dir_path = f"sudoku/data/{n}"
    features = torch.load(os.path.join(train_data_dir_path, "features.pt"))
    labels = torch.load(os.path.join(train_data_dir_path, "labels.pt"))
    features = torch.tensor(features, dtype=torch.float32).to(device)[:1000]
    labels   = torch.tensor(labels, dtype=torch.float32).to(device)[:1000]
    print(features.shape)
    print(labels.shape)
    
    # Create TensorDataset
    dataset = TensorDataset(features, labels)   
    
   # Fixed split indices
    num_samples = len(dataset)
    train_split = 0.8
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
    alpha = 100
    init_A = torch.rand((40, 64)).double()
    model_cvxpylayer_A = CvxpyLayerSudokuLearnA(n, Qpenalty)
    model_qpth_A = OptNetSudokuLearnA(n, Qpenalty)
    model_cvxpylayer = SingleOptLayerSudoku(n, learnable_parts=["eq"], layer_type="cvxpylayer", Qpenalty=0.1, alpha=alpha)
    model_qpth = SingleOptLayerSudoku(n, learnable_parts=["eq"], layer_type="qpth", Qpenalty=0.1, alpha=alpha)
    print("finish setup models")
    
    model_cvxpylayer_A.train()
    model_qpth_A.train()
    model_cvxpylayer.train()
    model_qpth.train()
    
    print("finish setup model train")
    
    true_idx = 2
    models_list = [model_qpth_A, model_cvxpylayer_A, model_cvxpylayer, model_qpth]
    A_grad_list = [None, None, None, None, None]
    # z0_grad_list = [None, None, None, None, None]

    loss_fn = torch.nn.MSELoss()
    
    forward_time = [0 for _ in models_list]
    backward_time = [0 for _ in models_list]
    print("start training")
    for i, (x, y) in enumerate(train_loader):
        if i%1==0:
            print(f"\t\t train example: {i}/{len(train_loader)}")
            
        x = x.to(device)
        y = y.to(device)
        
        for model_idx, model in enumerate(models_list):
            print(f"\t\t --- model: {model_idx}")
            
            start_time = time.time()
            pred = model(x)
            loss = loss_fn(pred, y)
            
            forward_time[model_idx] += time.time() - start_time

            start_time = time.time()
            loss.backward()
            backward_time[model_idx] += time.time() - start_time
            
            # for name, param in model.named_parameters():
            #     print(name, param.grad.shape)
            A_grad_list[model_idx] = model.A.grad.detach().cpu()#.numpy()
            # z0_grad_list[model_idx] = model.z0_a.grad.detach().cpu()#.numpy()
            

        break

    print('Forward time {}, \nbackward time {}'.format(forward_time, backward_time))
    # print(f"A_grad_list: {A_grad_list}")
    # print(f"z0 grad list: {z0_grad_list}")
    
    A_grad_list = [x.reshape(-1) for x in A_grad_list]
    #z0_grad_list = [x.reshape(-1) for x in z0_grad_list]
    
    A_grad_diffs = []
    #z0_grad_diffs = []
    A_grad_cos_sims = []
    #z0_grad_cos_sims = []
    for model_idx, model in enumerate(models_list):
        A_true_grad = A_grad_list[true_idx]
        #z0_true_grad = z0_grad_list[true_idx]
        
        grad_diff = torch.norm(A_grad_list[model_idx] - A_true_grad, p=1)
        cos_sim = torch.nn.functional.cosine_similarity(A_grad_list[model_idx], A_true_grad, dim=0)
        A_grad_diffs.append(grad_diff.item())
        A_grad_cos_sims.append(cos_sim.item())

        #z0_grad_diff = torch.norm(z0_grad_list[model_idx] - z0_true_grad, p=1)
        #z0_cos_sim = torch.nn.functional.cosine_similarity(z0_grad_list[model_idx], z0_true_grad, dim=0)
        #z0_grad_diffs.append(z0_grad_diff.item())
        #0_grad_cos_sims.append(z0_cos_sim.item())

    models_names = ["qpth_A", "cvxpylayer_A", "cvxpylayer", "qpth"]
    rows = ["A grad diffs", "A grad cos sims"]
    data = [A_grad_diffs, A_grad_cos_sims]
    table = np.array(data)
    import pandas as pd
    df = pd.DataFrame(table, columns=models_names, index=rows)
    print(df)
    
    