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
from models import physense_for_pipe_crossattn_walk


with open('./configs/sensor_rect_pipe_walk.yml', "r") as f:
    config = yaml.safe_load(f)

config = DotWiz(config)
print(config)
model = physense_for_pipe_crossattn_walk.Model(args=config)
print(count_params(model))


sensor_num = 50

model.x_sens = torch.nn.Parameter(torch.Tensor((sensor_num)), requires_grad=True)
model.y_sens = torch.nn.Parameter(torch.Tensor((sensor_num)), requires_grad=True)


# optimized model path
ckpt_path = f'./checkpoints/pipe_{sensor_num}sensor_opt.pth'

print("Loading checkpoint {}".format(ckpt_path))
model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
model.cuda()


def sample_image(noise, model, row_indices, col_indices, sensor_value, sensor_pos):
    with torch.no_grad():
        N = 10
        
        dt = (1./N)
        z = noise
        batchsize = z.shape[0]

        for i in range(N):
            t = (torch.ones((batchsize)) * i / N).to('cuda')
            x = z
            pred = model(x, t, row_indices, col_indices, sensor_value, sensor_pos)
            z = z + pred * dt

    return z  




def interp2d_single(data2: torch.Tensor, x_sens: torch.Tensor, y_sens: torch.Tensor) -> torch.Tensor:
    
    xx = torch.arange(0, data2.shape[1], 1)
    xy = torch.arange(0, data2.shape[2], 1)
    itp = torch.empty(data2.shape[0],len(x_sens), data2.shape[-1])

    for i, (x_sens, y_sens) in enumerate(zip(x_sens,y_sens)):
        idx_xx = torch.sum(torch.ge(x_sens[None], xx[None,:]), 1) -1
        idx_xy = torch.sum(torch.ge(y_sens[None], xy[None,:]), 1) -1
        idx_xx = torch.clamp(idx_xx, 0, data2.shape[1] - 2) 
        idx_xy = torch.clamp(idx_xy, 0, data2.shape[2] - 2) 
        x0 = xx[idx_xx]
        x1 = xx[idx_xx+1]
        y0 = xy[idx_xy]
        y1 = xy[idx_xy+1]
        wa = (x1-x_sens) * (y1-y_sens)
        wb = (x1-x_sens) * (y_sens-y0)
        wc = (x_sens-x0) * (y1-y_sens)
        wd = (x_sens-x0) * (y_sens-y0)
        # print(data2.shape)
        Ia = data2[ :, idx_xx, idx_xy ,:]
        Ib = data2[ :, idx_xx, idx_xy+1,: ]
        Ic = data2[ :, idx_xx+1, idx_xy,: ]
        Id = data2[ :, idx_xx+1, idx_xy+1 ,:]
    
        itp[:,i,:] = ((Ia*wa + Ib*wb+ Ic*wc + Id*wd)/((x1-x0)*(y1-y0)))[:,0,:]
    return itp


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
            
            batch_size = field.shape[0]
            
            current_sensor_num = 100
            
            sparse_sensor_field = interp2d_single(
                    field.cpu(), model.x_sens[:current_sensor_num].cpu(), model.y_sens[:current_sensor_num].cpu()
                )
            
            sparse_pos = interp2d_single(
                    model.pos.cpu(), model.x_sens[:current_sensor_num].cpu(), model.y_sens[:current_sensor_num].cpu()
                ).repeat(batch_size, 1, 1)
            
            
            noise = torch.randn_like(field).to(device)
            
            
            im = sample_image(noise, model, model.x_sens[:current_sensor_num].to(device), model.y_sens[:current_sensor_num].to(device), sparse_sensor_field.to(device), sparse_pos.to(device))


            mse_loss += (im - field).square().sum(dim=(1, 2, 3)).sum(dim=0)
            relative_loss += (torch.norm(field.reshape(batch_size, -1) - im.reshape(batch_size, -1), 2, 1) / torch.norm(field.reshape(batch_size, -1), 2, 1)).sum(dim=0)
            
            print(f'{((index+1)*batch_size)}: mse_loss: {mse_loss/((index+1)*batch_size)}, relative_loss:{relative_loss/((index+1)*batch_size)}')
            
            index += 1