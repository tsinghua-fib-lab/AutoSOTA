import argparse
import torch
import torch.nn as nn
import random
import os
import numpy as np
from resnet import resnet18
from dataprocess import PreProcess_Cifar100, build_dvscifar
from utils import *
from monitor import SOPMonitor


def eval_one_epoch(model, test_dataloader, sim_len, input_size, args, use_dvs=False):
    mon = SOPMonitor(model)
    tot, sops, tot_sops, nops, fire_spikes, tot_spikes = torch.zeros(sim_len).cuda(), 0, 0, 0, 0, 0
    model.eval()
    mon.enable()
    lenth = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            spikes = 0
            img = img.to(torch.device('cuda'), non_blocking=True)
            label = label.to(torch.device('cuda'), non_blocking=True)
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
            
    for name in mon.monitored_layers:
        sop_sublist, nop_sublist, tot_sublist = mon[name]
        if len(sop_sublist) > 0:
            sops += torch.tensor([f_s for (f_s, _) in sop_sublist]).mean(0).item() * sim_len
            tot_sops += torch.tensor([t_s for (_, t_s) in sop_sublist]).mean(0).item() * sim_len
        if len(tot_sublist) > 0:
            fire_spikes += torch.tensor([f_s for (f_s, _) in tot_sublist]).mean(0).item() * sim_len
            tot_spikes += torch.tensor([t_s for (_, t_s) in tot_sublist]).mean(0).item() * sim_len
        if len(nop_sublist) > 0:
            nops += torch.tensor(nop_sublist).mean(0).item() * sim_len

    sops = sops / args.batchsize
    tot_sops = tot_sops / args.batchsize
    nops = nops / args.batchsize

    for t in range(sim_len):
        print(f"T={t}, Acc={tot[t]/lenth}")
    print('Avg SOPs: {:.2f} M, Avg NOPs: {:.2f} M'.format(sops, nops))
    print('Avg Firing Rate: {:.2f}, Avg Event Rate: {:.2f}'.format(fire_spikes/tot_spikes, sops/tot_sops))

    if args.mode == 'vanilla':
        print('Avg Power: {:.2f} mJ'.format(sops*0.9/1000))
    else:
        print('Avg Power: {:.2f} mJ'.format(sops*0.9/8000))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='/home/cifar100/', help='Directory where the dataset is saved')
    parser.add_argument('--net_arch', type=str, default='resnet18', help='Network Architecture')
    parser.add_argument('--batchsize', type=int, default=64, help='Batchsize')
    parser.add_argument('--time_step', type=int, default=4, help='Training Time-steps for SNNs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dev', type=str, default='0')
    parser.add_argument('--use_eca', type=int, default=0, help='Use ECA Attention')
    parser.add_argument('--use_mem_bn', action='store_true', help='Use Membrane BatchNorm')
    parser.add_argument('--use_parallel', action='store_true', help='Use Parallel Block')
    parser.add_argument('--mode', type=str, default='vanilla', choices=['vanilla', 'compression'], help='Firing mode')
    parser.add_argument('--use_ter', action='store_true', help='Use Ternary Weight')
    parser.add_argument('--checkpoint_path', type=str, default='')

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

    dvs_data = False
    if args.dataset == 'CIFAR100':
        train_dataloader, test_dataloader, train_sampler, test_sampler = PreProcess_Cifar100(args.datadir, args.batchsize, False)
        cls_num = 100
        input_size = (3, 32, 32)
    elif args.dataset == 'DVSCIFAR':
        train_dataloader, test_dataloader, train_sampler, test_sampler = build_dvscifar(args.datadir, args.batchsize, False)        
        dvs_data = True
        cls_num = 10
        input_size = (2, 48, 48)
    else:
        print('unable to find dataset ' + args.dataset)
        
    if args.net_arch == 'resnet18':
        model = resnet18(args.time_step, use_ternary=args.use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode=args.mode)
    elif args.net_arch == 'resnet18_lif':
        model = resnet18(args.time_step, use_ternary=args.use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode='vanilla', weight_mode='vanilla', use_lif='vanilla')
    elif args.net_arch == 'resnet18_lif_online':
        model = resnet18(args.time_step, use_ternary=args.use_ter, num_classes=cls_num, use_dvs=dvs_data, use_resnet19=False, use_eca=args.use_eca, mem_bn=args.use_mem_bn, parallel_mode=args.use_parallel, mode='vanilla', weight_mode='vanilla', use_lif='online')
    else:
        print('unable to find model ' + args.net_arch)
    
    if len(args.checkpoint_path) > 0:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
    print(print_model_param_info(model))
    reset_BConv(model)
    model.cuda()
    eval_one_epoch(model, test_dataloader, args.time_step, input_size, args, dvs_data)
