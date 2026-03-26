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

@torch.no_grad()
def test(device, model, model_name, train_loader, sensor_number):

    total_loss = []
    
    loss_sum = 0
    index = 0

    for p in model.parameters():
        p.requires_grad_(False) 
    
    for cfd_data in tqdm(train_loader):
        
        cfd_data = cfd_data[0]
        cfd_data = cfd_data.to(device)
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            
        loss = model.sample(cfd_data, sensor_number=sensor_number)
        
        loss_sum += loss
        index += 1
        
        print(index, loss_sum / index, loss)
        
        total_loss.append(loss.item())

    print(total_loss)
    return np.mean(total_loss)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(local_rank, device, train_dataset, val_dataset, Net, model_name, hparams, path, reg=1, val_iter=1, coef_norm=[], ):
    
    sampler = DistributedSampler(train_dataset)
    test_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=False, drop_last=True, sampler=sampler)
        
    model = Net.to(device)

    start = time.time()

    test_loss = 1e5
    
    sensor_number = hparams['sensor_num']
    
    test_loader.sampler.set_epoch(0)
    test_loss = test(device, model, model_name, test_loader, sensor_number)

    print(f"loss:{test_loss}")

    return model
