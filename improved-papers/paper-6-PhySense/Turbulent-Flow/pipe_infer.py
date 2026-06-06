import numpy as np
import matplotlib.pyplot as plt
import torch
from dotwiz import DotWiz
import yaml
from datasets.senseiver_dataloader import load_data

from main import parse_args_and_config

torch.cuda.set_device(5)
test_data, x_sens, y_sens = load_data('pipe', 1, 123)

print('load data successfully')

test_data = test_data[:]
test_dataset = torch.utils.data.TensorDataset(test_data)


from model_dict import count_params
from models import physense_for_pipe_crossattn


with open('./configs/sensor_rect_pipe.yml', "r") as f:
    config = yaml.safe_load(f)

config = DotWiz(config)
print(config)
model = physense_for_pipe_crossattn.Model(args=config)
print(count_params(model))


ckpt_path = './checkpoints/pipe_best_base_model.pth'


print("Loading checkpoint {}".format(ckpt_path))
model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
model.cuda()

def sample_image(noise, model, row_indices, col_indices, sensor_value):
    with torch.no_grad():
        N = 10
        
        dt = (1./N)
        z = noise
        batchsize = z.shape[0]

        for i in range(N):
            t = (torch.ones((batchsize)) * i / N).to('cuda')
            x = z
            pred = model(x, t, row_indices, col_indices, sensor_value)
            z = z + pred * dt

    return z  


import torch.utils.data as data

import tqdm
import random
test_loader = data.DataLoader(test_dataset, batch_size=100)

with torch.no_grad():
    device = 'cuda'
    model.eval()
    
    index = 0
    known_field_loss = 0
    mse_loss = 0
    relative_loss = 0
    for field in (test_loader):
        
        with torch.cuda.amp.autocast():
        
            field = field[0]
            field = field.to(device)
            m = 100

            
            H, W = field.shape[1], field.shape[2]
            indices = torch.randperm(H * W)[:m].to(device)
            

            row_indices = indices // W
            col_indices = indices % W
            
            sensor_value = field[:,row_indices, col_indices,:].to(device)
            
            batch_size = field.shape[0]
            
            
            noise = torch.randn_like(field).to(device)
            
            
            im = sample_image(noise, model, row_indices, col_indices, sensor_value)
        
            mse_loss += (im - field).square().sum(dim=(1, 2, 3)).sum(dim=0)
            relative_loss += (torch.norm(field.reshape(batch_size, -1) - im.reshape(batch_size, -1), 2, 1) / torch.norm(field.reshape(batch_size, -1), 2, 1)).sum(dim=0)
            
            print(f'{((index+1)*batch_size)}: mse_loss: {mse_loss/((index+1)*batch_size)}, relative_loss:{relative_loss/((index+1)*batch_size)}')
           
            index += 1