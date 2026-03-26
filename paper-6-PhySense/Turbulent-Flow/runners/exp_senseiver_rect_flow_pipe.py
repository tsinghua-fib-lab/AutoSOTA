import os
import logging
import random

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from model_dict import get_model, count_params

from datasets.senseiver_dataloader import load_data



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

        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
        
        gpu_num = self.config.training.gpu_num


        # DDP init
        device = torch.device("cuda", local_rank)
        
        data_config = {
            'data_name': 'pipe',
            'num_sensors': 8,
            'seed': 123,
            'training_frames': 10000,
            'space_bands': 32,
            'sample_train_dataset': False,
        }
        
        train_data, x_sens, y_sens = load_data('pipe', 1, 123)
        

        train_dataset = torch.utils.data.TensorDataset(train_data[:10000])
        
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
        
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            # find_unused_parameters=True
        )
        print(count_params(model))
        

        model = model.to(device)
        
        save_freq = self.config.training.save_freq
        total_epoch = self.config.training.n_epochs

        optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr, weight_decay=1e-4)
    
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        
        torch.autograd.set_detect_anomaly(False)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(0, total_epoch + 1):

            
            train_loader.sampler.set_epoch(epoch)
            train_loss_step = 0

    
            for field in tqdm.tqdm(train_loader):
                
                
                with torch.cuda.amp.autocast():
                    
                    field = field[0]
                    field = field.to(device) 
                    
                    # get sensors
                    m = random.randint(25,300)
                    H, W = field.shape[1], field.shape[2]
                    indices = torch.randperm(H * W)[:m].to(device)

                    row_indices = indices // W
                    col_indices = indices % W
                    sensor_value = field[:,row_indices, col_indices,:].to(device)
        
                    optimizer.zero_grad()

                    noise = torch.randn_like(field)
                    
                    # logit normal sampling
                    batch_size = field.shape[0]
                    u = torch.normal(mean=0.0, std=1.0, size=(batch_size,)).to(device)
                    t = torch.sigmoid(u)
                    
                
                    t_tmp = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, field.shape[1], field.shape[2], 1)
                    target = field - noise
                    noised_field = t_tmp * field + (1 - t_tmp) * noise
                    model_input = noised_field
                    predict_v = model(model_input, t, row_indices, col_indices, sensor_value)
                    loss = (predict_v - target).square().sum(dim=(1, 2, 3)).sum(dim=0)
                
                scaler.scale(loss).backward()
                
                # for name, p in model.named_parameters():
                #     if torch.isnan(p).any():
                #         print(name,'NAN')
                scaler.step(optimizer)
                scaler.update()
                    
                train_loss_step += loss.item()
                
                
            scheduler.step()
            

            ntrain = data_config['training_frames'] / gpu_num

            logging.info( f"epoch:{epoch}/{self.config.training.n_epochs}, loss:{train_loss_step / (ntrain)}" )

            tb_logger.add_scalar('train_l2_full', train_loss_step / (ntrain), epoch)
            
            your_local_dir = self.config.training.ckpt_path
            
            if epoch % save_freq == 0:
                os.makedirs(os.path.join(your_local_dir, args.doc), exist_ok=True)
                print('save model')
                if local_rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(your_local_dir,args.doc, "ckpt_{}".format(epoch)))
                