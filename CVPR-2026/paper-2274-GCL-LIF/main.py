import argparse
import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging
from resnet import resnet18, resnet19, resnet34
from vgg import vgg11, vgg13
from dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_TinyImageNet, load_ImageNet_dataset, build_dvscifar
from utils import *
from torch.cuda import amp
from timm.data import Mixup
import time


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def init_distributed(distributed_init_mode):
    if not all(k in os.environ for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')):
        print('Not using distributed mode')
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    backend = 'nccl'
    print('Distributed init rank {}'.format(rank))
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_mode, world_size=world_size, rank=rank)
    return True, rank, world_size, local_rank
    
    
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt
    

def get_gpu_memory_usage(device_id):
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
    return allocated + reserved


def train_one_epoch_STBP(model, loss_fn, optimizer, train_dataloader, sim_len, local_rank, scaler=None, mixup=None, distributed=False, use_dvs=False):
    epoch_loss, gpu_mem, lenth = 0, 0, 0
    model.train()
    for img, label in train_dataloader:
        img = img.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)
        lenth += len(img)
        if mixup:
            img, label = mixup(img, label)
        if use_dvs is True:
            img = img.transpose(0, 1).contiguous()
        else:
            img = img.unsqueeze(0).repeat(sim_len, 1, 1, 1, 1)
        
        optimizer.zero_grad()
        spikes = 0
        for t in range(sim_len):
            spikes += model(img[t])
        reset_model(model)

        if scaler is not None:
            with amp.autocast():
                loss = loss_fn(spikes, label)
                scaler.scale(loss).backward()
                gpu_mem = max(gpu_mem, get_gpu_memory_usage(local_rank))
                scaler.step(optimizer)
                scaler.update()
        else:
            loss = loss_fn(spikes, label)             
            loss.backward()
            gpu_mem = max(gpu_mem, get_gpu_memory_usage(local_rank))
            optimizer.step()
            
        if distributed:
            vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
            epoch_loss += vis_loss.item()
        else:
            epoch_loss += loss.item()     

    return epoch_loss/lenth, gpu_mem


def train_one_epoch(model, loss_fn, optimizer, train_dataloader, sim_len, local_rank, scaler=None, mixup=None, distributed=False, use_dvs=False, opt_backprop=False):
    epoch_loss, gpu_mem, lenth, index = 0, 0, 0, 0
    if opt_backprop:
        if local_rank == 0:
            backprop_list = np.random.choice(range(sim_len), len(train_dataloader))
        else:
            backprop_list = np.empty(len(train_dataloader), dtype=int)
        if distributed:
            torch.distributed.broadcast(
                torch.from_numpy(backprop_list), src=0
            )
    model.train()
    for img, label in train_dataloader:
        img = img.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)
        lenth += len(img)
        index += 1
        if mixup:
            img, label = mixup(img, label)
        if use_dvs is True:
            img = img.transpose(0, 1).contiguous()
        else:
            img = img.unsqueeze(0).repeat(sim_len, 1, 1, 1, 1)
        
        for t in range(sim_len):
            optimizer.zero_grad()
            if scaler is not None:
                with amp.autocast():
                    spikes = model(img[t])
                    loss = loss_fn(spikes, label)
                    if (opt_backprop is False) or ((opt_backprop is True) and (t == backprop_list[index-1])):
                        scaler.scale(loss).backward()
                        gpu_mem = max(gpu_mem, get_gpu_memory_usage(local_rank))
                        scaler.step(optimizer)
                        scaler.update()
            else:
                spikes = model(img[t])
                loss = loss_fn(spikes, label)     
                if (opt_backprop is False) or ((opt_backprop is True) and (t == backprop_list[index-1])):
                    loss.backward()
                    gpu_mem = max(gpu_mem, get_gpu_memory_usage(local_rank))
                    optimizer.step()
            
            if distributed:
                vis_loss = reduce_mean(loss, torch.distributed.get_world_size())
                epoch_loss += vis_loss.item()
            else:
                epoch_loss += loss.item()

            if (opt_backprop is True) and (t == backprop_list[index-1]):
                break

        reset_model(model)

    return epoch_loss/lenth, gpu_mem


def eval_one_epoch(model, test_dataloader, sim_len, local_rank, distributed=False, use_dvs=False):
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    tot = torch.zeros(sim_len).to(device)
    model.eval()
    lenth = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            spikes = 0
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            lenth += len(img)
            if use_dvs is True:
                img = img.transpose(0, 1).contiguous()
            else:
                img = img.unsqueeze(0).repeat(sim_len, 1, 1, 1, 1)

            for t in range(sim_len):
                out = model(img[t])
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum().item()
            reset_model(model)

    if distributed:
        tot = reduce_mean(tot, torch.distributed.get_world_size())
    
    return tot/lenth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='/home/cifar100/', help='Directory where the dataset is saved')
    parser.add_argument('--trainsnn_epochs', type=int, default=300, help='Training Epochs of SNNs')
    parser.add_argument('--net_arch', type=str, default='resnet18', help='Network Architecture')
    parser.add_argument('--batchsize', type=int, default=64, help='Batchsize')
    parser.add_argument('--time_step', type=int, default=4, help='Training Time-steps for SNNs')
    parser.add_argument('--snn_lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--snn_wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_inference', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dev', type=str, default='0')
    parser.add_argument('--snn_resume', action='store_true', default=False)
    parser.add_argument('--distributed_init_mode', type=str, default='env://')
    parser.add_argument("--sync_bn", action="store_true", help="Use sync batch norm")
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--use_eca', type=int, default=0, help='Use ECA Attention')
    parser.add_argument('--opt_backprop', action='store_true', help='Use Backpropagation Optimization')
    parser.add_argument('--use_mem_bn', action='store_true', help='Use Membrane BatchNorm')
    parser.add_argument('--use_ter', action='store_true', help='Use Ternary Weight')
    parser.add_argument('--use_parallel', action='store_true', help='Use Parallel Block')
    parser.add_argument('--mode', type=str, default='vanilla', choices=['vanilla', 'compression'], help='Firing mode')
    parser.add_argument('--mixup', action='store_true', help='Mixup')
    parser.add_argument('--amp', action='store_true', help='Use AMP training')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    
    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)
    

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{args.dataset}-{args.net_arch}-eca_{args.use_eca}_mem-bn_{args.use_mem_bn}-parallel_{args.use_parallel}-T{args.time_step}-B{args.batchsize}-use_ter_{args.use_ter}-{args.mode}')
    identifier = f'snn_epoch_{args.trainsnn_epochs}_lr_{args.snn_lr}_wd_{args.snn_wd}-opt_backprop_{args.opt_backprop}_mixup_{args.mixup}_seed_{args.seed}'
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    
    distributed, rank, world_size, local_rank = init_distributed(args.distributed_init_mode)

    use_ter, dvs_data = args.use_ter, False
    if args.dataset == 'CIFAR10':
        train_dataloader, test_dataloader, train_sampler, test_sampler = PreProcess_Cifar10(args.datadir, args.batchsize, distributed)
        cls_num = 10
        input_size = (3, 32, 32)
    elif args.dataset == 'CIFAR100':
        train_dataloader, test_dataloader, train_sampler, test_sampler = PreProcess_Cifar100(args.datadir, args.batchsize, distributed)
        cls_num = 100
        input_size = (3, 32, 32)
    elif args.dataset == 'ImageNet200':
        train_dataloader, test_dataloader, train_sampler, test_sampler = PreProcess_TinyImageNet(args.datadir, args.batchsize, distributed)
        cls_num = 200
        input_size = (3, 64, 64)
    elif args.dataset == 'ImageNet-100':
        train_dataloader, test_dataloader, train_sampler, test_sampler = load_ImageNet_dataset(args.batchsize, os.path.join(args.datadir, 'train'), os.path.join(args.datadir, 'val'), distributed)
        cls_num = 100
        input_size = (3, 224, 224)
    elif args.dataset == 'ImageNet-1k':
        train_dataloader, test_dataloader, train_sampler, test_sampler = load_ImageNet_dataset(args.batchsize, os.path.join(args.datadir, 'train'), os.path.join(args.datadir, 'val'), distributed)
        use_ter = True
        cls_num = 1000
        input_size = (3, 224, 224)
    elif args.dataset == 'DVSCIFAR':
        train_dataloader, test_dataloader, train_sampler, test_sampler = build_dvscifar(args.datadir, args.batchsize, distributed)        
        dvs_data = True
        cls_num = 10
        input_size = (2, 48, 48)
    elif local_rank == 0:
        print('unable to find dataset ' + args.dataset)

    if args.net_arch == 'resnet18':
        model = resnet18(args.time_step, use_ternary=use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode=args.mode)
    if args.net_arch == 'resnet18_fp':
        model = resnet18(args.time_step, use_ternary=use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode=args.mode, weight_mode='vanilla')
    elif args.net_arch == 'resnet18_lif':
        model = resnet18(args.time_step, use_ternary=use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode='vanilla', weight_mode='vanilla', use_lif='vanilla')
    elif args.net_arch == 'resnet18_lif_online':
        model = resnet18(args.time_step, use_ternary=use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode='vanilla', weight_mode='vanilla', use_lif='online')
    elif args.net_arch == 'resnet18_lif_online_ter':
        model = resnet18(args.time_step, use_ternary=True, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode='vanilla', use_lif='online')
    elif args.net_arch == 'resnet18_lif_online_learnable':
        model = resnet18(args.time_step, use_ternary=use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode='vanilla', weight_mode='vanilla', use_lif='online-learnable')
    elif args.net_arch == 'resnet19':
        model = resnet19(args.time_step, use_ternary=use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=True, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode=args.mode)
    elif args.net_arch == 'resnet34':
        model = resnet34(args.time_step, use_ternary=use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode=args.mode)
    elif args.net_arch == 'vgg11':
        model = vgg11(args.time_step, cls_num, use_ter, dvs_data, args.use_eca, args.use_mem_bn, args.mode)
    elif args.net_arch == 'vgg13':
        model = vgg13(args.time_step, cls_num, use_ter, dvs_data, args.use_eca, args.use_mem_bn, args.mode)
    elif local_rank == 0:
        print('unable to find model ' + args.net_arch)
    
    if local_rank == 0:
        logger.info(print_model_param_info(model))
    
    model.cuda()
    if distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    mixup = None
    if args.mixup:
        mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0,
                      switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=cls_num)

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
        
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    snn_optimizer = torch.optim.SGD(model.parameters(), lr=args.snn_lr, momentum=0.9, weight_decay=args.snn_wd, nesterov=True)
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(snn_optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(snn_optimizer, T_max=args.trainsnn_epochs - warmup_epochs)
    snn_scheduler = torch.optim.lr_scheduler.SequentialLR(snn_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])       
    
    model_without_ddp = model
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, broadcast_buffers=False)
        model_without_ddp = model.module

    if args.snn_resume:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        snn_optimizer.load_state_dict(checkpoint['snn_optimizer'])
        start_epoch = checkpoint['snn_epoch'] + 1
        best_acc = checkpoint['snn_max_acc1']
        snn_scheduler.load_state_dict(checkpoint['snn_scheduler'])
        print(best_acc, start_epoch)
    else:
        start_epoch = 0
        best_acc = 0


    if args.direct_inference is not True:   
        for epoch in range(start_epoch, args.trainsnn_epochs):
            if distributed:
                train_sampler.set_epoch(epoch)
            if args.net_arch == 'resnet18_lif':
                epoch_loss, gpu_mem = train_one_epoch_STBP(model, loss_fn, snn_optimizer, train_dataloader, args.time_step, local_rank, scaler, mixup, distributed, dvs_data)
            else:
                epoch_loss, gpu_mem = train_one_epoch(model, loss_fn, snn_optimizer, train_dataloader, args.time_step, local_rank, scaler, mixup, distributed, dvs_data, args.opt_backprop)
            snn_scheduler.step()

            if distributed:
                torch.distributed.barrier()

            acc = eval_one_epoch(model, test_dataloader, args.time_step, local_rank, distributed, dvs_data)

            if local_rank == 0:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'snn_optimizer': snn_optimizer.state_dict(),
                    'snn_scheduler': snn_scheduler.state_dict(),
                    'snn_epoch': epoch,
                    'snn_max_acc1': acc[-1].item()
                    }
                if best_acc < acc[-1].item():
                    best_acc = acc[-1].item()
                    torch.save(checkpoint, os.path.join(log_dir, f'{identifier}_snn_best_checkpoint.pth'))
                torch.save(checkpoint, os.path.join(log_dir, f'{identifier}_snn_current_checkpoint.pth'))

                logger.info(f"SNNs training Epoch {epoch}: Val_loss: {epoch_loss} GPU_Mem: {gpu_mem} Acc: {acc} Best Acc: {best_acc}")
    else:
        if local_rank == 0:
            print(f'=== Load Pretrained SNNs ===')
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            model_without_ddp.cuda()
            new_acc = eval_one_epoch(model, test_dataloader, args.time_step, local_rank, distributed, dvs_data)
            print(new_acc)
