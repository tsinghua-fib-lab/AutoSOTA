import numpy as np
import matplotlib.pyplot as plt
import torch
from dotwiz import DotWiz
import yaml

torch.cuda.set_device(0)


data_path = '/workspace/mayuezhou/ddim/sea_data_large/'

sea_test_data = torch.from_numpy(np.load(data_path + 'sea_test_2020to2021_correct.npy').astype(np.float32)).unsqueeze(-1)

# normalize 
# divide the maximum in train data
sea_test_data /= 37.584736


sea_test_time = torch.from_numpy(np.load(data_path + 'sea_test_2020to2021_time_processed.npy'))


test_dataset = torch.utils.data.TensorDataset(sea_test_data, sea_test_time)

print('load data successfully')


from model_dict import count_params
from models import physense_for_sea_crossattn_walk


with open('./configs/sensor_rect_sea.yml', "r") as f:
    config = yaml.safe_load(f)

config = DotWiz(config)
print(config)
model = physense_for_sea_crossattn_walk.Model(args=config)
print(count_params(model))


sensor_num = 50

model.x_sens = torch.nn.Parameter(torch.Tensor((sensor_num)), requires_grad=True)
model.y_sens = torch.nn.Parameter(torch.Tensor((sensor_num)), requires_grad=True)


# optimized model path
ckpt_path = f'./checkpoints/sea_{sensor_num}sensor_opt.pth'

print("Loading checkpoint {}".format(ckpt_path))
model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
model.cuda()



def sample_image(noise, model, land_mask, sparse_sensor_field, sparse_pos, week):
    with torch.no_grad():
        N = 2
        dt = (1./N)
        # traj = [] # to store the trajectory
        z = noise
        batchsize = z.shape[0]

        # traj.append(z.detach().clone())
        for i in range(N):
            t = (torch.ones((batchsize)) * i / N).to('cuda')
            x = torch.concat([z, land_mask.unsqueeze(0).repeat(batchsize,1,1,1)],dim=-1)
            pred = model(x, t , week, model.x_sens.to(device), model.y_sens.to(device), sparse_sensor_field.to(device), sparse_pos.to(device))

        
            z = z + pred * dt
            
        return z
    

def sea_n_sensors(data, n_sensors, seed=-1):
    
    if seed != -1:
        np.random.seed(seed)
        torch.manual_seed(seed)
    im = torch.clone(data).squeeze()
    
    # print('Picking up sensor locations \n')
    coords = []
    
    for n in range(n_sensors):
        while True:
            new_x = np.random.randint(0,data.shape[0],1)[0]
            new_y = np.random.randint(0,data.shape[1],1)[0]
            if im[new_x,new_y] != 0:
                coords.append([new_x,new_y])
                im[new_x,new_y] = 0
                break
    coords = np.array(coords)  
    return coords[:,0], coords[:,1]




import torch
from typing import Optional

def bilinear_interp_with_mask(
        data: torch.Tensor,      # (B,H,W,C)
        xs: torch.Tensor,        # (N,)
        ys: torch.Tensor,        # (N,)
        mask: Optional[torch.Tensor] = None  # (H,W) 1=valid,0=invalid
    ) -> torch.Tensor:

    B, H, W, C = data.shape
    device = data.device
    xs = xs.to(device)
    ys = ys.to(device)

    grid_x = torch.arange(H, device=device)
    grid_y = torch.arange(W, device=device)

    N = xs.numel()
    out = torch.empty(B, N, C, device=device)

    for i, (x, y) in enumerate(zip(xs, ys)):

        ix = torch.clamp(torch.searchsorted(grid_x, x, right=True) - 1, 0, H-2)
        iy = torch.clamp(torch.searchsorted(grid_y, y, right=True) - 1, 0, W-2)

        x0, x1 = ix, ix + 1
        y0, y1 = iy, iy + 1

        Ia = data[:, x0, y0, :]   
        Ib = data[:, x0, y1, :]   
        Ic = data[:, x1, y0, :]   
        Id = data[:, x1, y1, :]   

        if mask is not None:
            ma = mask[x0, y0]
            mb = mask[x0, y1]
            mc = mask[x1, y0]
            md = mask[x1, y1]
        else:
            ma = mb = mc = md = 1.0
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        weights = torch.tensor([wa, wb, wc, wd], device=device)
        masks   = torch.tensor([ma, mb, mc, md], device=device, dtype=weights.dtype)

        if masks.sum() == 0:
            out[:, i, :] = torch.nan
            continue

        weights = weights * masks
        weights = weights / weights.sum()

        out[:, i, :] = (
            Ia * weights[0] +
            Ib * weights[1] +
            Ic * weights[2] +
            Id * weights[3]
        )

    return out

import torch.utils.data as data


device = 'cuda'


land_mask = torch.load('./land_mask_sealarge.pt').to(device).unsqueeze(-1)
test_loader = data.DataLoader(
            test_dataset,
            batch_size=40,
            shuffle=False,
        )

with torch.no_grad():
    device = 'cuda'
    model.eval()
    
    index = 0
    known_field_loss = 0
    mse_loss = 0
    relative_loss = 0
    for field, time in (test_loader):
        
        with torch.cuda.amp.autocast():
            
            field = field.to(device)
            time = time.to(device)
            
            batch_size = field.shape[0]
            
            current_sensor_num = sensor_num
            
            sparse_sensor_field = bilinear_interp_with_mask(
                    field.cpu(), model.x_sens[:current_sensor_num].cpu(), model.y_sens[:current_sensor_num].cpu(), land_mask.squeeze().cpu()
                )
            
            sparse_pos = bilinear_interp_with_mask(
                    model.pos.cpu(), model.x_sens[:current_sensor_num].cpu(), model.y_sens[:current_sensor_num].cpu(), land_mask.squeeze().cpu()
                ).repeat(batch_size, 1, 1)
            
            
            K = 25  # Antithetic pairs: K positive + K negative noise = 2K samples
            im = torch.zeros_like(field).to(device)
            for _k in range(K):
                noise = torch.randn_like(field).to(device)
                # Antithetic: average +noise and -noise for variance reduction
                im = im + sample_image(noise, model, land_mask, sparse_sensor_field.to(device), sparse_pos.to(device), time)
                im = im + sample_image(-noise, model, land_mask, sparse_sensor_field.to(device), sparse_pos.to(device), time)
            im = im / (2 * K)
            
            im = im * land_mask
            
            # Sensor enforcement: overwrite sensor locations with exact readings
            # x_sens, y_sens are float sensor positions; round to nearest grid
            xs_int = model.x_sens[:current_sensor_num].detach().clamp(0, im.shape[1]-1).round().long()
            ys_int = model.y_sens[:current_sensor_num].detach().clamp(0, im.shape[2]-1).round().long()
            # Get exact sensor values from ground truth field (before masking)
            sensor_readings = bilinear_interp_with_mask(
                field.cpu(), model.x_sens[:current_sensor_num].cpu(), model.y_sens[:current_sensor_num].cpu(), land_mask.squeeze().cpu()
            ).to(device)  # (B, N, C)
            # Overwrite at integer sensor positions with exact readings
            for si in range(current_sensor_num):
                xi = xs_int[si].item()
                yi = ys_int[si].item()
                if land_mask[xi, yi, 0] > 0:  # only if on ocean (not land)
                    im[:, xi, yi, :] = sensor_readings[:, si, :]
            
            field = field * land_mask
            
            mse_loss += (im - field).square().sum(dim=(1, 2, 3)).sum(dim=0)
            relative_loss += (torch.norm(field.reshape(batch_size, -1) - im.reshape(batch_size, -1), 2, 1) / torch.norm(field.reshape(batch_size, -1), 2, 1)).sum(dim=0)
            
            print(f'{((index+1)*batch_size)}: mse_loss: {mse_loss/((index+1)*batch_size)}, relative_loss:{relative_loss/((index+1)*batch_size)}')
            
            index += 1
            
    print(f'{(sea_test_data.shape[0])}: mse_loss: {mse_loss/sea_test_data.shape[0]}, relative_loss:{relative_loss/sea_test_data.shape[0]}')