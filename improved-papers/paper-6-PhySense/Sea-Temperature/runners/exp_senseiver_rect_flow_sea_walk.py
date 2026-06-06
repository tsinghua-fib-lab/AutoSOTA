import os
import logging

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from model_dict import get_model, count_params
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


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device


    def train(self):

        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        # print(os.environ)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
        
        gpu_num = self.config.training.gpu_num


        # DDP init
        device = torch.device("cuda", local_rank)
        
        data_path = '/workspace/mayuezhou/ddim/sea_data_large/'
        
        sea_train_data = torch.from_numpy(np.load(data_path + 'sea_train_1993to2019.npy')).unsqueeze(-1)
        sea_train_data /= sea_train_data.max()

        sea_train_time = torch.from_numpy(np.load(data_path + 'sea_train_1993to2019_time_processed.npy'))
        
        train_dataset = torch.utils.data.TensorDataset(sea_train_data, sea_train_time)
        
        print('load data successfully')
        
        # optimized sensor number
        m = self.config.training.sensor_number
        print(m)
                    
        seed = self.config.training.seed
        print(seed)
        torch.manual_seed(seed)
        
        
        def sea_n_sensors(data, n_sensors, seed=-1):
    
            if seed != -1:
                np.random.seed(seed)
                torch.manual_seed(seed)
            im = torch.clone(data).squeeze()
                        
            coords = []
            
            for n in range(n_sensors):
                while True:
                    new_x = np.random.randint(0,data.shape[0],1)[0]
                    new_y = np.random.randint(0,data.shape[1],1)[0]
                    if im[new_x,new_y] != 0:
                        coords.append([new_x,new_y])
                        im[new_x,new_y] = 0
                        break
            coords = torch.from_numpy(np.array(coords)  )
            return coords[:,0], coords[:,1]
        
        
        land_mask = torch.load('./land_mask_sealarge.pt').to(device).unsqueeze(-1)
        row_indices, col_indices = sea_n_sensors(land_mask, m, seed)


        
        

        batch_size = config.training.batch_size
        
        sampler = DistributedSampler(train_dataset)
        
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler = sampler,
            num_workers=config.data.num_workers,
        )


        model = get_model(config)
        
        print(row_indices.shape)
        
        ckpt_path = self.config.training.base_model_path

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        model.x_sens = torch.nn.Parameter(torch.tensor(row_indices, dtype=torch.float32).to(device), requires_grad=True)
        model.y_sens = torch.nn.Parameter(torch.tensor(col_indices, dtype=torch.float32).to(device), requires_grad=True)
        
        print(model.x_sens.is_leaf, model.x_sens.grad_fn)
        
        model = model.to(device)


        lr = config.optim.lr
        lr_main = 0 # freeze the model parameter
        print(lr, lr_main)

        sensor_pos_parameter = [
            {'params': model.x_sens, 'lr': lr},  
            {'params': model.y_sens, 'lr': lr}
        ]
        other_parameters = [
            param for param in model.parameters() 
            if param.size() != model.x_sens.size() and param.size() != model.y_sens.size()
        ]

        params = sensor_pos_parameter + [{'params': other_parameters, 'lr': lr_main}]

        optimizer = torch.optim.Adam(params, weight_decay=0)
        
        save_freq = self.config.training.save_freq
        total_epoch = self.config.training.n_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=0)
        
        model=torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        print(count_params(model))
        
        torch.autograd.set_detect_anomaly(False)
        
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(0, total_epoch):
            train_loader.sampler.set_epoch(epoch)
            train_loss_step = 0

            for field, time in tqdm.tqdm(train_loader):
                with torch.cuda.amp.autocast():
                    
                    field = field.to(device).float()
                    time = time.to(device).float()
                    optimizer.zero_grad()
                    
                    field = field * land_mask
                    
                    noise = torch.randn_like(field)
                    
                    # logit normal sampling
                    batch_size = field.shape[0]
                    u = torch.normal(mean=0.0, std=1.0, size=(batch_size,)).to(device)
                    t = torch.sigmoid(u)


                    t_tmp = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, field.shape[1], field.shape[2], 1)
                    target = (field - noise) * land_mask
                    
                    noised_field = t_tmp * field + (1 - t_tmp) * noise
                    
                    model_input = torch.concat((noised_field, land_mask.unsqueeze(0).repeat(batch_size,1,1,1)), dim=-1)
                    
                    current_sensor_num = m

                    sparse_sensor_field = bilinear_interp_with_mask(
                            field.cpu(), model.module.x_sens[:current_sensor_num].cpu(), model.module.y_sens[:current_sensor_num].cpu(), land_mask.squeeze().cpu()
                        )
                    
                    sparse_pos = bilinear_interp_with_mask(
                            model.module.pos.cpu(), model.module.x_sens[:current_sensor_num].cpu(), model.module.y_sens[:current_sensor_num].cpu(), land_mask.squeeze().cpu()
                        ).repeat(batch_size, 1, 1)

                    predict_v = model(model_input.to(device), t.to(device), time, model.module.x_sens.to(device), model.module.y_sens.to(device), sparse_sensor_field.to(device), sparse_pos.to(device))
                    
                    predict_v = predict_v * land_mask
                    
                    loss = (predict_v - target).square().sum(dim=(1, 2, 3)).sum(dim=0)
                    
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.module.x_sens.data.clamp_(0, field.shape[1])
                model.module.y_sens.data.clamp_(0, field.shape[2])
                    
                train_loss_step += loss.item()
                
                
            scheduler.step()
            

            ntrain = sea_train_data.shape[0] / gpu_num
            logging.info( f"epoch:{epoch}/{self.config.training.n_epochs}, loss:{train_loss_step / (ntrain)}, x_sens_6:{model.module.x_sens[6]}" )
            tb_logger.add_scalar('train_l2_full', train_loss_step / (ntrain), epoch)
            
            
            
            your_local_dir = self.config.training.ckpt_path
            
            if epoch % save_freq == 0:
                os.makedirs(os.path.join(your_local_dir, args.doc), exist_ok=True)
                print('save model')
                if local_rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(your_local_dir,args.doc, "ckpt_{}".format(epoch)))