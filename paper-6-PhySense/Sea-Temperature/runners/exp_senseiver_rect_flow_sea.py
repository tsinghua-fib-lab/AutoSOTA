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
        
        data_path = '/workspace/mayuezhou/ddim/sea_data_large/'
        
        sea_train_data = torch.from_numpy(np.load(data_path + 'sea_train_1993to2019.npy')).unsqueeze(-1)
        sea_train_data /= sea_train_data.max()

        sea_train_time = torch.from_numpy(np.load(data_path + 'sea_train_1993to2019_time_processed.npy'))
        
        train_dataset = torch.utils.data.TensorDataset(sea_train_data, sea_train_time)
        
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
    
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        
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

        
        torch.autograd.set_detect_anomaly(False)
        scaler = torch.cuda.amp.GradScaler()
        
        
        land_mask = torch.load('./land_mask_sealarge.pt').to(device).unsqueeze(-1)
        

        for epoch in range(0, total_epoch + 1):

            
            train_loader.sampler.set_epoch(epoch)
            train_loss_step = 0

    
            for field, time in tqdm.tqdm(train_loader):
                
                
                with torch.cuda.amp.autocast():
                    
                    field = field.to(device).float()
                    time = time.to(device).float()
                    
                    # get sensors
                    field = field * land_mask
                    
                    m = random.randint(10, 100)    
                    row_indices, col_indices = sea_n_sensors(land_mask, m, seed=-1)
                    
                    
                    sensor_value = field[:,row_indices, col_indices,:].to(device)
        
                    optimizer.zero_grad()

                    noise = torch.randn_like(field)
                    
                    # logit normal sampling
                    batch_size = field.shape[0]
                    u = torch.normal(mean=0.0, std=1.0, size=(batch_size,)).to(device)
                    t = torch.sigmoid(u)
                    
                
                    t_tmp = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, field.shape[1], field.shape[2], 1)
                    target = (field - noise) * land_mask
                    noised_field = t_tmp * field + (1 - t_tmp) * noise
                    model_input = torch.concat((noised_field, land_mask.unsqueeze(0).repeat(batch_size,1,1,1)), dim=-1)
                    
                    predict_v = model(model_input, t, time, row_indices, col_indices, sensor_value)
                    predict_v = predict_v * land_mask
                    
                    loss = (predict_v - target).square().sum(dim=(1, 2, 3)).sum(dim=0)
                
                scaler.scale(loss).backward()
                
                # for name, p in model.named_parameters():
                #     if torch.isnan(p).any():
                #         print(name,'NAN')
                scaler.step(optimizer)
                scaler.update()
                    
                train_loss_step += loss.item()
                
                
            scheduler.step()
            

            ntrain = sea_train_data.shape[0] / gpu_num

            logging.info( f"epoch:{epoch}/{self.config.training.n_epochs}, loss:{train_loss_step / (ntrain)}" )

            tb_logger.add_scalar('train_l2_full', train_loss_step / (ntrain), epoch)
            
            your_local_dir = self.config.training.ckpt_path
            
            if epoch % save_freq == 0:
                os.makedirs(os.path.join(your_local_dir, args.doc), exist_ok=True)
                print('save model')
                if local_rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(your_local_dir,args.doc, "ckpt_{}".format(epoch)))
                