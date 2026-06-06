"""
run_cifar10.py - Evaluation script for Stochastic Forward-Forward CIFAR10
Matches the eval command: python run_cifar10.py --config=./config.yaml --data_dir=...
With TTA (test-time augmentation): horizontal flip average at inference.
"""
import sys
import os
import copy
import torch
import numpy as np
import time

# Allow argparse with --config and --data_dir
import argparse
import utils
import prepared_configs
from pipeline import GreedyTrainPipeline, greedy_training_pipeline, AverageMeter, ProgressMeter


def run_cifar10_proposed(args):
    """Run the proposed method on CIFAR10 with the paper's default settings."""
    args.epochs = 3
    args.lr = 0.001
    args.bs = 128
    args.phase2_epoch = 60
    args.LR_SCHEDULER = {'name': "CosineAnnealingLR", 'args': {'T_max': args.epochs, 'eta_min': 0.0001}}

    args.dataset = 'CIFAR10'
    net_config = prepared_configs.default_network_config(
        consistency_mode='feature',
        diversity_factor=0.5,
        consistency_factor=0.5,
        sampling_len=20,
        inference_mode='sampling',
        projecting=True,
        projecting_dim=30,
        local_grad=True
    )
    for i in range(1, 3):
        net_config['conv_config'][i]['conv_type'] = 'depthwise'

    for j in range(3):
        net_config['conv_config'][j]['projecting_dim'] = 30 - j * 10

    args.MODEL = net_config
    args.MODEL['conv_config'][-1]['is_last'] = True
    args.AUXILIARY_CONFIG = copy.deepcopy(net_config)
    args.save_name = 'unsup_cifar10_0'

    # Monkey-patch the validate method to add TTA (horizontal flip) and print format
    original_validate = GreedyTrainPipeline.validate

    def patched_validate_with_tta(self, val_loader, model, criterion, args, epoch=0):
        """Validate with TTA: average logits from original and horizontally flipped images."""
        phase2_epoch = getattr(args, 'phase2_epoch', 60)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        model = self.model_processing_before_eval(model, args)

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images, target = self.data2device(images, target, args)

                # Original forward pass
                output_orig = self.compute_model_output(model, images, args)

                # Flip horizontally (dim=3 is width)
                images_flipped = torch.flip(images, dims=[3])
                output_flipped = self.compute_model_output(model, images_flipped, args)

                # Average logits
                output = (output_orig + output_flipped) / 2.0

                loss = self.compute_loss(output=output_orig, target=target, criterion=criterion, model=model, args=args, inputs=images)
                acc1, = self.accuracy(output, target, topk=(1, ))
                top1.update(acc1.item(), images.size(0))
                losses.update(loss.item(), images.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

        print(f"Test [Epoch {epoch}/{phase2_epoch}]: Loss={top1.avg:.4f} acc1={top1.avg:.2f}")
        sys.stdout.flush()
        return top1.avg

    GreedyTrainPipeline.validate = patched_validate_with_tta

    greedy_training_pipeline(args, train_func=GreedyTrainPipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 evaluation for Stochastic Forward-Forward')
    parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', help='YAML config file')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('-p', '--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--dir', default='results', type=str)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('--save_name', default='unsup_cifar10_0', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=False)
    parser.add_argument('--world_size', default=-1, type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--start-epoch', default=0, type=int)

    args = parser.parse_args()
    args.start_epoch = 0

    if args.config is not None:
        args = utils.set_config2args(args)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.world_size = torch.cuda.device_count()
    args.local_rank = 0
    if torch.cuda.is_available() and not args.cpu:
        args.use_cuda = True
    else:
        args.use_cuda = False

    # Override data_dir from command line if specified
    if '--data_dir' in sys.argv:
        idx = sys.argv.index('--data_dir')
        args.data_dir = sys.argv[idx + 1]

    run_cifar10_proposed(args)
