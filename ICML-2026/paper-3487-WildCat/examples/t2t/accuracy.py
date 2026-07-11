"""Accuracy script for T2T-ViT on the ILSVRC2012 validation set.

This script reproduces the experiment from Sec. 4.2 of the following paper:
    Annabelle Michael Carrell, Albert Gong, Abhishek Shetty, Raaz Dwivedi, Lester Mackey
    Low-Rank Thinning
    https://arxiv.org/pdf/2502.12063

We adapted the following script:
- https://github.com/majid-daliri/kdeformer/blob/main/demo_imagenet.py
- License: MIT
- Copyright (c) 2025 KDEFormer
- Paper reference:
    Amir Zandieh, Insu Han, Majid Daliri, Amin Karbasi
    KDEformer: Accelerating transformers via kernel density estimation
    https://arxiv.org/pdf/2302.02451

NOTE: We made the following minor modifications to the original script:
1. We use a helper function to load the model (see `util_experiments.py:get_model()`)
2. We use a helper function to load the datamodule (see `imagenet.py:get_imagenet_datamodule()`)
Importantly, these modifications have no effect on the accuracy of the model.

Example usage:
To calculate accuracy of METHOD1 and METHOD2, run:
```bash
python accuracy.py -m1 METHOD1 -m2 METHOD2 -op OUTPUT_PATH
```
For example, to calculate full attention accuracy, set METHOD1=full and METHOD2=full.
This script saves the accuracy results to OUTPUT_PATH/acc/acc-METHOD1-METHOD2-DEVICE-sSEED.csv

To compute accuracy with fp16, add the --fp16 flag:
```bash
python accuracy.py -m1 METHOD1 -m2 METHOD2 -op OUTPUT_PATH --fp16
```

Example usage: display script arguments
```bash
python accuracy.py --help
```
"""

import os
import time
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict

from imagenet import get_imagenet_datamodule
from util_experiments import get_model, get_base_parser

parser = get_base_parser()
parser.add_argument(
    "--method1",
    "-m1",
    default="wildcat",
    type=str,
    help="attention method for first attention layer"
    "e.g., full, performer, reformer, scatterbrain, kdeformer, thinformer",
)
parser.add_argument(
    "--method2",
    "-m2",
    default="wildcat",
    type=str,
    help="attention method for second attention layer"
    "e.g., full, performer, reformer, scatterbrain, kdeformer, thinformer",
)
args = parser.parse_args()
method1 = args.method1
method2 = args.method2
device = args.device
ckpt_path = args.ckpt_path
dataset_path = args.dataset_path
output_path = args.output_path
seed = args.seed
force = args.force

acc_dir = os.path.join(output_path, "acc")
os.makedirs(acc_dir, exist_ok=True)
method1str = method1
method2str = method2
if method1 == "wildcat":
    # if args.q_kernel:
    #     method1str += "_q"
    if args.rank1 is not None:
        method1str += f"_r{args.rank1}"
    if args.bins1 is not None:
        method1str += f"_b{args.bins1}"
if method2 == "wildcat":
    # if args.q_kernel:
    #     method2str += "_q"
    if args.rank2 is not None:
        method2str += f"_r{args.rank2}"
    if args.bins2 is not None:
        method2str += f"_b{args.bins2}"
acc_file = os.path.join(acc_dir, f"acc-{method1str}-{method2str}-{device}-s{seed}.csv")
if os.path.exists(acc_file) and not force:
    print(f"Skipping {acc_file} as it already exists")
    exit()


print(f"Setting torch random seed to {seed}")
torch.manual_seed(seed)

print("Loading model...")
dtype = torch.float16 if args.fp16 else torch.float32
model = get_model(method1, method2, args, ckpt_path, device, dtype)

print("Loading data...")
datamodule = get_imagenet_datamodule(dataset_path, batch_size=64)

cnt = 0
corrects = 0
pbar = tqdm(datamodule.val_dataloader())
output = defaultdict(list)
tic = time.time()
print("Running model with torch.no_grad()...")
for i, (images, labels) in enumerate(pbar):
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        out = model(images)
    batch_corrects = (out.detach().cpu().argmax(-1) == labels).sum().item()
    batch_cnt = len(labels)
    output["corrects"].append(batch_corrects)
    output["cnt"].append(batch_cnt)
    # Compute cumulative corrects
    corrects += batch_corrects
    cnt += batch_cnt
    pbar.set_description(
        f"batch #{i}: {100 * corrects / cnt:.2f}% ({corrects} / {cnt})"
    )

toc = time.time() - tic
accuracy = float(corrects) / cnt
print(f"corrects: {corrects}, cnt: {cnt}")
name = f"{method1} {method2}"
res_str = f"[{name:<10}] dtype: {dtype}, time: {toc:.4f}, accuracy: {accuracy:.8f}, corrects: {corrects}, seed: {seed}\n"

print(res_str)

acc_df = pd.DataFrame(output)

print(f"Saving acc to {acc_file}:")
acc_df.to_csv(acc_file)

summ_file = os.path.join(acc_dir, f"accuracies-seed{seed}.txt")
print(f"Saving acc summary to {summ_file}")
with open(summ_file, 'a') as f:
    method = f"""Attention Method:
    layer1: {method1str}
    layer2: {method2str}"""
    f.write(method + '\n')
    f.write(f'Seed: {seed}\n')
    f.write(res_str)
    f.write('\n')