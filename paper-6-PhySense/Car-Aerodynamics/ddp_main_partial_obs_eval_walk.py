import train_partial_obs_eval
import os
import torch
import argparse

from pathlib import Path
import numpy as np

from torch_geometric.data import Data, Dataset

import torch.distributed as dist

from dataset.load_dataset import load_train_val_fold
from dataset.dataset import GraphDataset

from model_dict import get_model, count_params

import re, math


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir')
parser.add_argument('--train_data_num', default=75, type=int)
parser.add_argument('--cfd_model')
parser.add_argument('--base_model_path')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=1, type=float)
parser.add_argument('--nb_epochs', default=201, type=int)
parser.add_argument('--save_freq', default=1, type=int)


parser.add_argument('--sensor_num', default=30, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--model_path')

args = parser.parse_args()
print(args)


ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
port = os.environ.get("MASTER_PORT", "64209")
hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
rank = int(os.environ.get("RANK", "0"))  # node id
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
gpus = torch.cuda.device_count()  # gpus per node
args.local_rank = local_rank
print(ip, port, hosts, rank, local_rank, gpus)

dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                        rank=rank)
torch.cuda.set_device(local_rank)

hparams = {'lr': args.lr, 'batch_size': args.batch_size, 'nb_epochs': args.nb_epochs, 'sensor_num': args.sensor_num}
print(hparams)

device = torch.device("cuda", local_rank)

# print(device)

print(args.cfd_model)
model = get_model(args.cfd_model)
print(count_params(model))


# load train data

dataset = []



p_min = -844.3360
p_max = 602.6890

case_num = 25

base_path = args.data_dir

for index in range(76, case_num + 76):
    
    press_path = base_path + 'pressure_files/' + f'case_{index}_p_car_patch.raw'
    velocity_path = base_path + 'velocity_files/' + f'case_{index}_initialConditions'
    
    arr = np.loadtxt(press_path,         
                    comments='#',     
                    dtype=np.float32) 

    sigma_clip = 3
    coords = arr[:, :3]
    pressures = arr[:, 3]

    p_mean = pressures.mean()
    p_std = pressures.std()

    lower_bound = p_mean - sigma_clip * p_std
    upper_bound = p_mean + sigma_clip * p_std

    mask = (pressures >= lower_bound) & (pressures <= upper_bound)

    coords_filtered = torch.from_numpy(coords[mask])
    pressures_filtered = torch.from_numpy(pressures[mask])


    pressures_filtered = (pressures_filtered - p_min) / (p_max - p_min)


    pattern = re.compile(
        r"flowVelocity\s*\(\s*([-\d\.eE+]+)\s+([-\d\.eE+]+)\s+([-\d\.eE+]+)\s*\)"
    )

    file_path = Path(velocity_path)

    text = file_path.read_text()
    m = pattern.search(text)


    vx, vy, vz = map(float, m.groups())

    v = math.sqrt(vx**2 + vy**2 + vz**2)
    angle = math.degrees(math.acos(vx / v))        

    data = Data(pos=coords_filtered, y=pressures_filtered.unsqueeze(-1), v=v, angle=angle)
    dataset.append(data)



train_ds = GraphDataset(dataset)

val_ds = None
coef_norm = None



m = args.sensor_num

print(m)


pos = train_ds.get(0).pos

    
random_indices = torch.randperm(pos.shape[0])[:m]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
xyz = pos[random_indices]

model.xyz_sens = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device), requires_grad=True)

ckpt_path = args.model_path
print(ckpt_path)

model.load_state_dict(torch.load(ckpt_path, map_location=device))


model=torch.nn.parallel.DistributedDataParallel(model)

model = train_partial_obs_eval.main(local_rank, device, train_ds, val_ds, model, args.cfd_model, hparams, path=None, val_iter=1, reg=1,
                   coef_norm=coef_norm)
