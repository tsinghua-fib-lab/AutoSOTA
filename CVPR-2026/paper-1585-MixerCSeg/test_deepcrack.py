import time
import numpy as np
import torch
import argparse
import os
import cv2
from datasets import create_dataset
from models import build_MixerCSeg as build_model
from main import get_args_parser

parser = argparse.ArgumentParser('newcseg', parents=[get_args_parser()])
args = parser.parse_args([])
args.phase = 'test'
args.batch_size_train = 1
args.batch_size = 1
args.dataset_path = '/repo/data/DeepCrack'
args.dataset_mode = 'crack'
args.serial_batches = True
args.num_threads = 1
args.load_width = 512
args.load_height = 512
args.device = 'cuda'

device = torch.device(args.device)
test_dl = create_dataset(args)
load_model_file = '/repo/checkpoints/weights/checkpoint_DeepCrack/checkpoint_DeepCrack.pth'
data_size = len(test_dl)
model, criterion = build_model(args)
state_dict = torch.load(load_model_file, map_location='cpu')
model.load_state_dict(state_dict['model'])
model.to(device)
print('Load Model Successful!')
suffix = 'DeepCrack_eval'
save_root = '/repo/results/results_test/' + suffix
if not os.path.isdir(save_root):
    os.makedirs(save_root)
with torch.no_grad():
    model.eval()
    for batch_idx, data in enumerate(test_dl):
        x = data['image']
        target = data['label']
        if device != 'cpu':
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
        out = model(x)
        out_sigmoid = torch.sigmoid(out)
        target = target[0, 0, ...].cpu().numpy()
        pred = out_sigmoid[0, 0, ...].cpu().numpy()
        root_name = data['A_paths'][0].split('/')[-1][0:-4]
        target_save = 255 * (target / np.max(target) if np.max(target) > 0 else target)
        pred_save = (pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_root, '{}_lab.png'.format(root_name)), target_save)
        cv2.imwrite(os.path.join(save_root, '{}_pre.png'.format(root_name)), pred_save)
        if batch_idx % 20 == 0:
            print(f'Processed {batch_idx+1}/{data_size}')
print('Finished inference!')
