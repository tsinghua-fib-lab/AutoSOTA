# %%
import os
import re
import json
import json_repair
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append("../")
cmd_args = True
#
parser = ArgumentParser()
parser.add_argument('--file_dir', default='../dataset/mix', help='file name')
parser.add_argument('--file_name', default='__combine.txt', help='file name of the dataset')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()


# %%
SOURCE_FILE = os.path.join(args.file_dir, args.file_name)
basename = os.path.basename(SOURCE_FILE)
SAVE_DIR = os.path.join(os.path.dirname(SOURCE_FILE), basename.split('.')[0] + f'_{args.save_type_name}_DA')
SAVE_FILE = os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_ext_ori.tsv')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 
with open(SAVE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()

results = []
for item in tqdm(ori_data):
    item = json.loads(item)
    item = json_repair.loads(item)
    if type(item) != dict:
        continue
    if 'polish' not in item:
        continue
    polish = item['polish']
    polish = polish.replace('[', '')
    polish = polish.replace(']', '')
    polish = polish.replace('\n', '')
    polish = re.sub(r' +', ' ', polish)
    results.append(polish)
    

with open(os.path.join(args.file_dir, basename.split('.')[0] + f'_{args.save_type_name}_ext.txt'), mode='w+') as f:
    for item in results:
        f.write(item + '\n')

# %%
