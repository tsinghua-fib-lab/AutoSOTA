import numpy as np
import time, json, os
import torch
import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler

from torch_geometric.loader import DataLoader
from tqdm import tqdm


def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(device, model, model_name, train_loader, optimizer, scheduler, epoch, reg=1):
    

    
    model.train()
    total_loss = []
    
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

    print(hparams['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=hparams['nb_epochs'], eta_min=0)
    
    start = time.time()

    train_loss, val_loss = 1e5, 1e5

    for epoch in range(hparams['nb_epochs']):

        train_loader.sampler.set_epoch(epoch)
        train_loss = train(device, model, model_name, train_loader, optimizer, lr_scheduler, epoch, reg=reg)
    
        print(f"epoch:{epoch}, loss:{train_loss}")

        if epoch % hparams['save_freq'] == 0:
            if local_rank == 0:
                torch.save(model.module.state_dict(), path + os.sep + f'model_{hparams["nb_epochs"]}_{epoch}_cosine.pth')

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))

    return model
