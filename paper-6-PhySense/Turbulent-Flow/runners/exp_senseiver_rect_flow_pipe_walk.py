import os
import logging

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from model_dict import get_model, count_params

from datasets.senseiver_dataloader import load_data


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

        Ia = data2[ :, idx_xx, idx_xy ,:]
        Ib = data2[ :, idx_xx, idx_xy+1,: ]
        Ic = data2[ :, idx_xx+1, idx_xy,: ]
        Id = data2[ :, idx_xx+1, idx_xy+1 ,:]
        itp[:,i,:] = ((Ia*wa + Ib*wb+ Ic*wc + Id*wd)/((x1-x0)*(y1-y0)))[:,0,:]
        
    return itp


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
        
        data_config = {
            'data_name': 'pipe',
            'num_sensors': 10,
            'seed': 123,
            'training_frames': 10000,
            'space_bands': 32,
            'sample_train_dataset': False,
        }
        
        train_data, x_sens, y_sens = load_data('pipe', 1, 123)
        
        
        # optimized sensor number
        m = self.config.training.sensor_number
        print(m)
                    
        seed = self.config.training.seed
        print(seed)
        torch.manual_seed(seed)
        
        H, W = train_data.shape[1], train_data.shape[2]
        indices = torch.randperm(H * W)[:m].to(device)
        

        row_indices = indices // W
        col_indices = indices % W
        

        train_dataset = torch.utils.data.TensorDataset(train_data)
        
        print('load data successfully')

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

        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
        
        save_freq = self.config.training.save_freq
        total_epoch = self.config.training.n_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=0)
        
        model=torch.nn.parallel.DistributedDataParallel(model)
        print(count_params(model))
        
        torch.autograd.set_detect_anomaly(False)
        
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(0, total_epoch):
            train_loader.sampler.set_epoch(epoch)
            train_loss_step = 0

            for field in tqdm.tqdm(train_loader):
                with torch.cuda.amp.autocast():
                    
                    field = field[0]
                    field = field.to(device)
                    optimizer.zero_grad()
                    
                    noise = torch.randn_like(field)
                    
                    # logit normal sampling
                    batch_size = field.shape[0]
                    u = torch.normal(mean=0.0, std=1.0, size=(batch_size,)).to(device)
                    t = torch.sigmoid(u)
                    
                    # uniform sampling (not used)
                    # t = torch.rand(batch_size).to(device)

                    t_tmp = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, field.shape[1], field.shape[2], 1)
                    target = field - noise
                    
                    noised_field = t_tmp * field + (1 - t_tmp) * noise
                    model_input = noised_field

                    sparse_sensor_field = interp2d_single(
                            field.cpu(), model.module.x_sens.cpu(), model.module.y_sens.cpu()
                        )

                    sparse_pos = interp2d_single(
                            model.module.pos.cpu(), model.module.x_sens.cpu(), model.module.y_sens.cpu()
                        ).repeat(batch_size, 1, 1)


                    predict_v = model(model_input.to(device), t.to(device), model.module.x_sens.to(device), model.module.y_sens.to(device), sparse_sensor_field.to(device), sparse_pos.to(device))
                    loss = (predict_v - target).square().sum(dim=(1, 2, 3)).sum(dim=0)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.module.x_sens.data.clamp_(0, field.shape[1])
                model.module.y_sens.data.clamp_(0, field.shape[2])
                    
                train_loss_step += loss.item()
                
                
            scheduler.step()
            

            ntrain = data_config['training_frames'] / gpu_num
            logging.info( f"epoch:{epoch}/{self.config.training.n_epochs}, loss:{train_loss_step / (ntrain)}, x_sens_6:{model.module.x_sens[6]}" )
            tb_logger.add_scalar('train_l2_full', train_loss_step / (ntrain), epoch)
            
            
            
            your_local_dir = self.config.training.ckpt_path
            
            if epoch % save_freq == 0:
                os.makedirs(os.path.join(your_local_dir, args.doc), exist_ok=True)
                print('save model')
                if local_rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(your_local_dir,args.doc, "ckpt_{}".format(epoch)))