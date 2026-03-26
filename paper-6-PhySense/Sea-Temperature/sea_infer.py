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
from models import physense_for_sea_crossattn


with open('./configs/sensor_rect_sea.yml', "r") as f:
    config = yaml.safe_load(f)

config = DotWiz(config)
print(config)
model = physense_for_sea_crossattn.Model(args=config)
print(count_params(model))


ckpt_path = './checkpoints/sea_best_base_model.pth'


print("Loading checkpoint {}".format(ckpt_path))

model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
model.cuda()

# print(model.t_embedder.mlp[0].weight)


def sample_image(noise, model, land_mask, row_indices, col_indices, sensor_value, week):
    with torch.no_grad():
        N = 2
        dt = (1./N)

        z = noise
        batchsize = z.shape[0]

        # traj.append(z.detach().clone())
        for i in range(N):
            t = (torch.ones((batchsize)) * i / N).to('cuda')
            x = torch.concat([z, land_mask.unsqueeze(0).repeat(batchsize,1,1,1)],dim=-1)
            pred = model(x, t , week, row_indices, col_indices, sensor_value)
        
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


import torch.utils.data as data


device = 'cuda'


land_mask = torch.load('./land_mask_sealarge.pt').to(device).unsqueeze(-1)
test_loader = data.DataLoader(
            test_dataset,
            batch_size=40,
            shuffle=False,
        )

with torch.no_grad():
    
    model.eval()
    
    index = 0
    known_field_loss = 0
    mse_loss = 0
    relative_loss = 0
    for field, week in (test_loader):

        with torch.cuda.amp.autocast():
            
            field = field.to(device)
            week = week.to(device)
            
            sensor_num = 100

            
            row_indices, col_indices = sea_n_sensors(land_mask.squeeze(), sensor_num, seed=-1)

            sensor_value = field[:,row_indices, col_indices,:].to(device)
            
            batch_size = field.shape[0]
            
            
            noise = torch.randn_like(field).to(device)
            
            im = sample_image(noise, model, land_mask, row_indices, col_indices, sensor_value, week)
            
            im = im * land_mask
            
            field = field * land_mask
    
            mse_loss += (im - field).square().sum(dim=(1, 2, 3)).sum(dim=0)
            relative_loss += (torch.norm(field.reshape(batch_size, -1) - im.reshape(batch_size, -1), 2, 1) / torch.norm(field.reshape(batch_size, -1), 2, 1)).sum(dim=0)
            
            print(f'{((index+1)*batch_size)}: mse_loss: {mse_loss/((index+1)*batch_size)}, relative_loss:{relative_loss/((index+1)*batch_size)}')
           
            index += 1
            
    print(f'{(sea_test_data.shape[0])}: mse_loss: {mse_loss/sea_test_data.shape[0]}, relative_loss:{relative_loss/sea_test_data.shape[0]}')