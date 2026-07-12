"""Evaluation script for ConOrd on BID dataset.

Evaluates a trained checkpoint using SRCC and PCC metrics with
three-crops averaged features (as described in Appendix B.2).

Usage:
    python eval_bid.py --checkpoint /path/to/checkpoint.pth --data_root /datasets/BID/BID
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torchvision import transforms

from config.basic import ConfigBasic
from utils.util import write_log, to_np, make_dir, log_configs, extract_embs
from utils.util import cal_srocc_plcc
from utils.comparison_utils import find_kNN
from networks.util import prepare_model
from data.get_datasets_BIQA import BIDDataset, get_datasets_BIQA


def parse_args():
    parser = argparse.ArgumentParser(description='ConOrd BID Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='/datasets/BID/BID')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--k_nn', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def set_config(args):
    cfg = ConfigBasic()
    cfg.dataset = 'bid'
    cfg.data_root = args.data_root
    cfg.set_biqa_dataset()

    cfg.model = 'ConOrd'
    cfg.backbone = 'vitB16'
    cfg.ref_mode = 'flex'
    cfg.ref_point_num = 60
    cfg.start_norm = True
    cfg.metric = 'L2'
    cfg.k = args.k_nn
    cfg.test_batch_size = 1000
    cfg.batch_size = args.batch_size
    cfg.num_workers = 0
    cfg.device = torch.device(f'cuda:{args.gpu}')
    cfg.gpu_ids = [args.gpu]
    cfg.n_gpu = 1
    cfg.n_ranks = 100  # Will be overridden
    cfg.fiducial_point_num = 100
    cfg.rank_min = 0.0
    cfg.rank_max = 5.0
    cfg.margin = 0.05
    cfg.tau = 2.0
    cfg.temp = 0.07
    cfg.epsilon = 1e-7

    return cfg


def extract_embs_three_crop(encoder, data_loader, cfg):
    """Extract features using three-crops averaging."""
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    def get_three_crops(img_tensor):
        _, h, w = img_tensor.shape
        if h != 256 or w != 256:
            img_pil = transforms.ToPILImage()(img_tensor)
            img_pil = transforms.Resize(256)(img_pil)
            img_tensor = transforms.ToTensor()(img_pil)
        tl = normalize(img_tensor[:, :224, :224])
        br = normalize(img_tensor[:, -224:, -224:])
        c = normalize(transforms.CenterCrop(224)(img_tensor))
        return torch.stack([tl, br, c])

    encoder.eval()
    embs_list = []
    inds_list = []

    with torch.no_grad():
        for batch in data_loader:
            x_base = batch[0]
            item = batch[-1]
            bsz = x_base.size(0)
            all_crops = []
            for i in range(bsz):
                crops = get_three_crops(x_base[i])
                all_crops.append(crops)
            all_crops = torch.cat(all_crops, dim=0).to(cfg.device)
            feats = encoder(all_crops)
            feats = feats.view(bsz, 3, -1).mean(dim=1)
            embs_list.append(feats.cpu())
            inds_list.append(item)

    embs = torch.cat(embs_list)
    inds = torch.cat(inds_list)
    embs_temp = deepcopy(embs)
    embs[inds] = embs_temp
    return embs


def main():
    args = parse_args()
    cfg = set_config(args)

    # Data
    loader_dict = get_datasets_BIQA(cfg)

    # Model - detect ref_point_num from checkpoint if available
    state = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in state:
        ref_shape = state['model']['ref_points'].shape
        cfg.ref_point_num = ref_shape[0]
        cfg.fiducial_point_num = ref_shape[0]
        cfg.n_ranks = ref_shape[0]
        print(f'Detected ref_point_num={ref_shape[0]} from checkpoint')

    model = prepare_model(cfg)
    model = model.to(cfg.device)
    model.load_state_dict(state['model'])
    model.eval()

    # Extract features
    print('Extracting train features with 3-crops...')
    embs_train = extract_embs_three_crop(model.encoder, loader_dict['train_for_val'], cfg)
    embs_train = embs_train.to(cfg.device)

    print('Extracting test features with 3-crops...')
    embs_test = extract_embs_three_crop(model.encoder, loader_dict['val'], cfg)
    embs_test = embs_test.to(cfg.device)

    test_labels = loader_dict['val'].dataset.mos
    train_labels = loader_dict['train_for_val'].dataset.mos

    if isinstance(test_labels, torch.Tensor):
        test_labels = to_np(test_labels)
    if isinstance(train_labels, torch.Tensor):
        train_labels = to_np(train_labels)

    # k-NN prediction
    k = cfg.k
    with torch.no_grad():
        _, inds = find_kNN(embs_test, embs_train, k=k, metric=cfg.metric)
        inds = np.squeeze(to_np(inds), 0)
        if inds.ndim == 1:
            inds = inds[np.newaxis, :]
        nn_labels = train_labels[inds[:, :k]]
        pred_scores = np.mean(nn_labels, axis=-1)

    srcc, plcc = cal_srocc_plcc(pred_scores, test_labels)

    print(f'\n=== BID Evaluation Results ===')
    print(f'SRCC: {srcc:.4f}')
    print(f'PLCC: {plcc:.4f}')
    print(f'Checkpoint: {args.checkpoint}')

    return srcc, plcc


if __name__ == '__main__':
    main()
