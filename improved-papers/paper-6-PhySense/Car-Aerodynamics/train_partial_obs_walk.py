import numpy as np
import time, json, os
import torch
import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from typing import Optional
from sklearn.neighbors import KDTree


@torch.no_grad()
def project_to_surface(xyz: torch.Tensor,
                       kdt: KDTree,
                       surf_pts_np) -> torch.Tensor:
    dev   = xyz.device
    dtype = xyz.dtype

    xyz_np = xyz.detach().cpu().numpy()
    dist, idx = kdt.query(xyz_np, k=1)          # (N,1)

    nearest_np = surf_pts_np[idx[:, 0]]         # (N,3)
    nearest = torch.as_tensor(nearest_np, device=dev, dtype=dtype)
    return nearest




def knn_idw_interp(
    data: torch.Tensor,         # (B, N, C)  features of known points
    pts: torch.Tensor,          # (N, 3)     positions of known points
    queries: torch.Tensor,      # (M, 3)     positions of quert points
    k: int = 8,                 # the nearest k
    p: float = 2.0,             # power
    mask: Optional[torch.Tensor] = None  # (N,) 1=valid
) -> torch.Tensor:              # (B, M, C)

    device  = data.device
    pts     = pts.to(device)
    queries = queries.to(device)
    if mask is not None:
        mask = mask.to(device, data.dtype)              # (N,)

    d2 = torch.cdist(queries, pts, p=2) + 1e-12         

    d2, idx = torch.topk(d2, k, dim=1, largest=False)   # (M, k)
    w = 1.0 / (d2 ** (p / 2))                           # (M, k)

    if mask is not None:
        valid = mask[idx]                               # (M, k)
        w = w * valid
        w_sum = w.sum(dim=1, keepdim=True) + 1e-12
        w = w / w_sum
    else:
        w = w / w.sum(dim=1, keepdim=True)


    feat_k = data[:, idx, :]                            # (B, M, k, C)
    w = w.unsqueeze(0).unsqueeze(-1)                    # (1, M, k, 1)
    out = (feat_k * w).sum(dim=2)                      # (B, M, C)
    return out




def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, model_name, train_loader, optimizer, scheduler, kdt, pos, reg=1, lr=1):
    model.train()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    total_loss = []
    
    # with torch.no_grad():
    for cfd_data in tqdm(train_loader):
        
        cfd_data = cfd_data[0]

        
        cfd_data = cfd_data.to(device)
        optimizer.zero_grad()
        
        loss = model(cfd_data)

        
        loss.backward()
        optimizer.step()
                
        
        total_loss.append(loss.item())

    scheduler.step()

    return np.mean(total_loss)


@torch.no_grad()
def test(device, model, test_loader):
    model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses_press = []
    losses_velo = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = criterion_func(out[cfd_data.surf, -1], targets[cfd_data.surf, -1]).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())

    return np.mean(losses_press), np.mean(losses_velo)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(local_rank, device, train_dataset, val_dataset, Net, model_name, hparams, path, reg=1, val_iter=1, coef_norm=[], ):
    
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False, drop_last=True, sampler=sampler)
        
    model = Net.to(device)
    
    
    # sensor number and seed
    
    m = hparams['sensor_num']
    seed = hparams['seed']
    
    print(m, seed)
    torch.manual_seed(seed)            
    
    pos = train_dataset.get(0).pos
    
     
    random_indices = torch.randperm(pos.shape[0])[:m]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    xyz = pos[random_indices]
    
    # torch.save(xyz, f'./origin_sensorpos_15_seed{seed}.pt')
    
    model.module.xyz_sens = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device), requires_grad=True)
    

    car_pts_np = pos.cpu().numpy()           
    kdt = KDTree(car_pts_np, leaf_size=512) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    
    lr = hparams['lr']
    lr_main = 0
    print(lr, lr_main)
    
    sensor_pos_group = {
        'params'        : [model.module.xyz_sens],  # ★
        'lr'            : lr,
        'weight_decay'  : 0.0                                          
    }
    other_parameters = [p for p in model.module.parameters()
                        if p is not model.module.xyz_sens]

    main_group = {
        'params'        : other_parameters,
        'lr'            : lr_main,
        'weight_decay'  : 1e-4
    }

    # -------------------------------------------------------------
    optimizer = torch.optim.Adam([sensor_pos_group, main_group])
    
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=5)
    
    
    start = time.time()

    train_loss, val_loss = 1e5, 1e5
    # pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in range(hparams['nb_epochs']):

        train_loader.sampler.set_epoch(epoch)
        train_loss = train(device, model, model_name, train_loader, optimizer, lr_scheduler, kdt, pos, reg=reg, lr=lr)
    
        print(f"epoch:{epoch}, loss:{train_loss}")
            
        if epoch % hparams['save_freq'] == 0:
            if local_rank == 0:
                torch.save(model.module.state_dict(), path + os.sep + f'model_{hparams["nb_epochs"]}_{epoch}_walk_{lr}_{m}_this_seed{seed}.pth')

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))


    return model
