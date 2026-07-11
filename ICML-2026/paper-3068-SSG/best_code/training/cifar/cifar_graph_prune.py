"""
CIFAR100 graph-pruning training script.
Based on the InfoBatch and graph-pruning strategy.

Usage examples:
    # Single GPU
    python cifar_graph_prune.py --graph-dir /path/to/train_graph \
        --dataloader-ratio 0.625 --score-weight 1.0 --model r18 --num_epoch 200

    # Multi-GPU DDP
    torchrun --nproc_per_node=4 cifar_graph_prune.py --graph-dir /path/to/train_graph \
        --dataloader-ratio 0.625 --score-weight 1.0 --model r18 --num_epoch 200 --use_ddp
"""

import argparse
import datetime
import json
import os
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from model import *
import torch.distributed as dist

# Import the CIFAR100-specific GraphProbPrune implementation.
from cifar_graph_dataloader import (
    CIFARGraphProbPrune,
    DistributedSamplerWrapper,
    concat_all_gather,
    split_index,
    recombine_index,
    is_master
)

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))


def compute_uncertainty_metrics(logits, top_k=2, reduction='mean'):
    """
    Compute entropy-based uncertainty metrics from logits.

    Args:
        logits: model outputs [B, C]
        top_k: unused; kept for compatibility
        reduction: 'mean', 'max', 'none'

    Returns:
        entropies: entropy value for each sample
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropies = -(probs * log_probs).sum(dim=-1)  # [B]

    def reduce(x):
        if reduction == 'mean':
            return x.mean()
        elif reduction == 'max':
            return x.max()
        elif reduction == 'none':
            return x
        else:
            raise ValueError("reduction must be 'mean', 'max', or 'none'")

    return entropies if reduction == 'none' else reduce(entropies)


def safe_print(*args, **kwargs):
    if RANK in (-1, 0):
        print(*args, **kwargs)


def setup_ddp():
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group('nccl', rank=RANK, world_size=world_size)


def destroy_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def format_duration(seconds: float) -> str:
    """Format a duration."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == '__main__':
    # Start total timing.
    SCRIPT_START_TIME = time.time()

    parser = argparse.ArgumentParser(description='CIFAR100 Graph Prune Training')
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_ddp', action='store_true', help='use distributed data parallel')
    parser.add_argument('--fp16', action='store_true', help='use mixed precision training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N', help='test batch size')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W', help='weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    # OneCycle scheduling arguments
    parser.add_argument('--max-lr', default=0.05, type=float)
    parser.add_argument('--div-factor', default=25, type=float)
    parser.add_argument('--final-div', default=10000, type=float)
    parser.add_argument('--num_epoch', default=200, type=int, help='training epochs')
    parser.add_argument('--pct-start', default=0.3, type=float)
    parser.add_argument('--ratio', default=0.5, type=float, help='prune ratio (for GraphProbPrune)')
    parser.add_argument('--delta', default=0.875, type=float, help='annealing delta')
    parser.add_argument('--model', default='r18', type=str, help='model architecture')
    # Graph prune specific arguments
    parser.add_argument('--graph-dir', type=str, required=True, help='graph file root directory')
    parser.add_argument('--dataloader-ratio', type=float, default=0.625, help='GraphProbPrune ratio')
    parser.add_argument('--score-weight', type=float, default=1.0, help='uncertainty score weight')
    parser.add_argument('--prune-mode', type=str, default='prob', choices=['prob', 'topk'], help='prune selection mode')

    args = parser.parse_args()

    # Device setup.
    if not torch.cuda.is_available():
        device = 'cpu'
    elif args.use_ddp:
        device = f'cuda:{LOCAL_RANK}'
        setup_ddp()
    else:
        device = 'cuda:0'

    use_amp = args.fp16 and torch.cuda.is_available()

    safe_print('==> Building model..')

    # Model construction.
    if args.model.lower() == 'r18':
        net = ResNet18(num_classes=100)
    elif args.model.lower() == 'r50':
        net = ResNet50(num_classes=100)
    elif args.model.lower() == 'r101':
        net = ResNet101(num_classes=100)
    else:
        net = ResNet50(num_classes=100)
    net = net.to(device)

    if args.use_ddp:
        safe_print('use ddp')
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    else:
        if torch.cuda.is_available():
            safe_print('use normal data parallel')
            net = torch.nn.DataParallel(net)
        else:
            safe_print('use single device (cpu)')

    # Loss function.
    try:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing, reduction='none').to(device)
    except Exception:
        safe_print('warning! This version has no label smooth.')
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    test_criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0
    best_loss = 1e3
    best_epoch = 0
    start_epoch = 0

    # Data preprocessing.
    stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # Load the CIFAR100 dataset.
    base_trainset = torchvision.datasets.CIFAR100(
        root='./cifar100', train=True, transform=train_transform, download=True
    )

    # Wrap the dataset with GraphProbPrune.
    safe_print(f'Using CIFARGraphProbPrune with graph_dir: {args.graph_dir}')
    safe_print(f'dataloader_ratio: {args.dataloader_ratio}, prune_mode: {args.prune_mode}')

    t_load_start = time.time()
    trainset = CIFARGraphProbPrune(
        base_trainset,
        input_graph_dir=args.graph_dir,
        ratio=args.dataloader_ratio,
        num_epoch=args.num_epoch,
        delta=args.delta,
        mode=args.prune_mode
    )
    t_load_end = time.time()
    graph_load_time = t_load_end - t_load_start
    safe_print(f'[Time] graph loading time: {graph_load_time:.2f} s')

    # Sampler
    if args.use_ddp:
        train_sampler = DistributedSamplerWrapper(trainset.pruning_sampler())
    else:
        train_sampler = trainset.pruning_sampler()

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        sampler=train_sampler
    )

    testset = torchvision.datasets.CIFAR100(
        root='./cifar100', train=False, download=True, transform=test_transform
    )
    testloader = DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            net.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'lars':
        from lars import Lars
        optimizer = Lars(
            net.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'lamb':
        from lamb import Lamb
        optimizer = Lamb(
            net.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.max_lr, steps_per_epoch=len(trainloader),
        epochs=args.num_epoch, div_factor=args.div_factor,
        final_div_factor=args.final_div, pct_start=args.pct_start
    )

    train_acc = []
    valid_acc = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Pruning-time statistics.
    total_prune_time = 0.0

    def train_epoch(epoch):
        global total_prune_time
        safe_print('\nEpoch: %d, iterations %d' % (epoch, len(trainloader)))
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        epoch_prune_time = 0.0

        for batch_idx, blobs in enumerate(trainloader):
            inputs, targets, indices, weights = blobs
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                # Compute uncertainty as the score.
                scores = compute_uncertainty_metrics(outputs, reduction='none')
                scores = scores * args.score_weight

                # Update dataset scores.
                t_prune_start = time.time()
                if args.use_ddp:
                    low, high = split_index(indices)
                    low = low.to(device, non_blocking=True).float()
                    high = high.to(device, non_blocking=True).float()
                    scores_f = scores.float()
                    tuple_tensor = torch.stack([low, high, scores_f], dim=0)
                    tuple_all = concat_all_gather(tuple_tensor, dim=1)
                    low_all = tuple_all[0].long()
                    high_all = tuple_all[1].long()
                    scores_all = tuple_all[2]
                    indices_all = recombine_index(low_all, high_all)
                    trainset.__setscore__(
                        indices_all.detach().cpu().numpy(),
                        scores_all.detach().cpu().numpy()
                    )
                else:
                    trainset.__setscore__(
                        indices.detach().cpu().numpy(),
                        scores.detach().cpu().numpy()
                    )
                epoch_prune_time += time.time() - t_prune_start

                # Weighted loss.
                loss = loss * weights
                loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        total_prune_time += epoch_prune_time
        safe_print('epoch: %d, Training Accuracy: %.3f, Train loss: %.4f, Prune time: %.3f s' % (
            epoch, 100. * correct / total, train_loss / len(trainloader), epoch_prune_time))
        train_acc.append(correct / total)

    def test(epoch):
        global best_acc, best_loss, best_epoch
        net.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = net(inputs)
                loss = test_criterion(outputs, targets)

                bs = targets.size(0)
                test_loss += loss.item() * bs
                _, predicted = outputs.max(1)
                total += bs
                correct += predicted.eq(targets).sum().item()

        if args.use_ddp:
            stat_tensor = torch.tensor(
                [test_loss, correct, total],
                dtype=torch.float64,
                device=device
            )
            dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
            test_loss = stat_tensor[0].item()
            correct = stat_tensor[1].item()
            total = stat_tensor[2].item()

        cur_acc = round(100. * correct / total, 3)
        cur_loss = round(test_loss / total, 4)
        safe_print('epoch: %d, Test Acc: %.3f, Test loss: %.4f, Best: epoch %d, acc %.3f, loss %.4f' % (
            epoch, cur_acc, cur_loss, best_epoch, best_acc, best_loss))

        if cur_acc > best_acc:
            best_acc = cur_acc
            best_epoch = epoch
        if cur_loss < best_loss:
            best_loss = cur_loss
        valid_acc.append(cur_acc)

    # Training loop.
    total_time = 0
    for epoch in range(args.num_epoch):
        if args.use_ddp:
            trainloader.sampler.set_epoch(epoch)

        # Update the learning-rate scheduler.
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.max_lr,
            steps_per_epoch=len(trainloader),
            epochs=args.num_epoch, div_factor=args.div_factor,
            final_div_factor=args.final_div, pct_start=args.pct_start,
            last_epoch=epoch * len(trainloader) - 1
        )

        end = time.time()
        train_epoch(epoch)
        total_time += time.time() - end
        test(epoch)

    # Training summary.
    SCRIPT_END_TIME = time.time()
    total_script_time = SCRIPT_END_TIME - SCRIPT_START_TIME

    safe_print('=' * 60)
    safe_print('[Time] graph loading time: %.2f s' % graph_load_time)
    safe_print('[Time] cumulative pruning time: %.2f s' % total_prune_time)
    safe_print('[Time] total training time: %.2f s' % total_time)
    safe_print('[Time] total script time: %.2f s (%s)' % (total_script_time, format_duration(total_script_time)))
    safe_print('=' * 60)
    safe_print('Total saved sample iteration: %d' % trainset.total_save())

    # Save logs.
    pref = 'graph_prune'
    fn = '{}-{}-ratio{}-score{}-epoch{}-bs{}-{}_cifar100_{}.json'.format(
        args.model, args.max_lr / args.div_factor, args.dataloader_ratio, args.score_weight,
        args.num_epoch, args.batch_size, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), pref
    )

    if RANK in (-1, 0):
        with open(fn, 'w+') as f:
            json.dump({
                'total_time': total_time,
                'total_saved': trainset.total_save(),
                'ratio': args.ratio,
                'dataloader_ratio': args.dataloader_ratio,
                'score_weight': args.score_weight,
                'train_acc': train_acc,
                'valid_acc': valid_acc,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
            }, f, indent=2)

        # Save model.
        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        model_name = f"{args.model}_cifar100_graph_prune_ratio{args.dataloader_ratio}_epoch{args.num_epoch}"
        model_name += f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        save_path = os.path.join(save_dir, model_name)

        state_dict = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        torch.save({
            'epoch': args.num_epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'best_loss': best_loss,
            'args': vars(args),
        }, save_path)
        safe_print(f"Model saved to: {save_path}")

    if args.use_ddp:
        destroy_ddp()




