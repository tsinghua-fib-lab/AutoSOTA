import os

import numpy as np
import torch
from PIL import Image
import pyiqa

import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters for LLFormer')
parser.add_argument("--dataset_dir", default=r'./output/HM-TIR', type=str, help="test_dataset_dir")
parser.add_argument('--label_dir', default=r'./datasets/HM-TIR/test', type=str, help='Directory for GT')
parser.add_argument('--metrics', default=["maniqa"], nargs='+', help='Metric name(s)')
parser.add_argument('--subfolders', default=["BH", "NH"], nargs='+', help='Subfolder name(s)')

args = parser.parse_args()
img_list = sorted(os.listdir(args.label_dir))

metric_dict = {}
metric_str = ""
for metric in args.metrics:
    metric_dict[metric] = pyiqa.create_metric(metric, device="cuda")
    metric_str += f", {metric.upper()}"

for deg in args.subfolders:
    print(deg + metric_str)

    method_list = sorted(os.listdir(os.path.join(args.dataset_dir, deg)))
    for method in method_list:
        metric_val = {k: [] for k in args.metrics}

        for img in img_list:
            inp = Image.open(os.path.join(args.dataset_dir, deg, method, img)).convert('L').convert("RGB")
            inp = np.array(inp, dtype=np.float32) / 255.0
            inp = inp[np.newaxis, :]
            inp = torch.from_numpy(np.ascontiguousarray(inp)).cuda().permute(0, 3, 1, 2)
            gt = Image.open(os.path.join(args.label_dir, img)).convert('L').convert("RGB")
            gt = np.array(gt, dtype=np.float32) / 255.0
            gt = gt[np.newaxis, :]
            gt = torch.from_numpy(np.ascontiguousarray(gt)).cuda().permute(0, 3, 1, 2)

            for metric in args.metrics:
                if metric != "fid":
                    metric_val[metric].append(metric_dict[metric](inp, gt))

        metrics_val_mean = ""
        for metric in args.metrics:
            if metric != "fid":
                val_mean = torch.stack(metric_val[metric]).mean().detach().cpu().numpy()
                metrics_val_mean += f", {val_mean}"
            else:
                fid_val = metric_dict["fid"](os.path.join(args.dataset_dir, deg, method), os.path.join(args.label_dir), num_workers=0, verbose=False)
                metrics_val_mean += f", {fid_val}"
        print(method + metrics_val_mean)
    print()

print('finish !')
