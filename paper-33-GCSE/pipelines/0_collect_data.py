# %%
import os
from argparse import ArgumentParser

import sys
sys.path.append("../")
cmd_args = True
#
parser = ArgumentParser()
parser.add_argument('--file_dir', default='../dataset/mix', help='file name')
parser.add_argument('--output_name', default='combine', help='file name of the dataset')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

files = os.listdir(args.file_dir)

SAVE_DIR = args.file_dir

result = []
for file_name in files:
    if file_name.find('__') >= 0:
        continue
    path = os.path.join(args.file_dir, file_name)
    with open(path) as f:
        ori_data = f.readlines()
    result += ori_data

with open(os.path.join(SAVE_DIR, f'__{args.output_name}.txt'), mode='w+') as f:
    for item in result:
        f.write(item.strip() + '\n')

# %%
