import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.pretraining_dataset import PretrainingDataset
from models.CSBrain import *
from pretrain_trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='lr_scheduler')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')
    parser.add_argument('--dataset_dir', type=str, default='path/to/dataset', help='dataset_dir')
    parser.add_argument('--model_dir', type=str, default='path/to/models', help='model_dir') # eg. 'CSBrain/pth'
    parser.add_argument('--TemEmbed_kernel_sizes', type=str, default="[(1,), (3,), (5,)]")
    parser.add_argument('--use_SmallerToken', type=bool, default=False, help='SmallerToken->dataset.py')
    parser.add_argument('--model', type=str, default='CSBrain', help='CSBrain')

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)

    pretrained_dataset = PretrainingDataset(
        dataset_dir=params.dataset_dir,
        SmallerToken=params.use_SmallerToken,
    )
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=8,
        shuffle=True,
    )

    brain_regions = [
        0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1
    ]
    electrode_labels = [
        "FP1-REF", "FP2-REF", "F3-REF", "F4-REF",
        "C3-REF", "C4-REF", "P3-REF", "P4-REF",
        "O1-REF", "O2-REF", "F7-REF", "F8-REF",
        "T3-REF", "T4-REF", "T5-REF", "T6-REF",
        "FZ-REF", "CZ-REF", "PZ-REF"
    ]
    topology = {
        0: ["FP1-REF", "F7-REF", "F3-REF", "FZ-REF", "F4-REF", "F8-REF", "FP2-REF"], 
        4: ["C3-REF", "CZ-REF", "C4-REF"],
        1: ["P3-REF", "PZ-REF", "P4-REF"],
        3: ["O1-REF", "O2-REF"],
        2: ["T3-REF", "T5-REF", "T6-REF", "T4-REF"],
        -1: ["A1-REF"]
    }
    region_groups = {}
    for i, region in enumerate(brain_regions):
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append((i, electrode_labels[i]))
    sorted_indices = []
    for region in sorted(region_groups.keys()):
        region_electrodes = region_groups[region]
        sorted_electrodes = sorted(region_electrodes, key=lambda x: topology[region].index(x[1]))
        sorted_indices.extend([e[0] for e in sorted_electrodes])

    print("Sorted Indices:", sorted_indices)

    if params.model == 'CSBrain':
        model = CSBrain(
            params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len, params.n_layer,
            params.nhead,
            eval(params.TemEmbed_kernel_sizes),
            brain_regions,
            sorted_indices
        )
  
    trainer = Trainer(params, data_loader, model)
    trainer.train()
    pretrained_dataset.db.close()


if __name__ == '__main__':
    main()
