# %%
import os
import json
import json_repair
from tqdm import tqdm
from argparse import ArgumentParser

import sys
sys.path.append("../")
cmd_args = False
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
                         basename.split('.')[0]+'_syn_samples.jsonl')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 
with open(SOURCE_FILE, encoding='utf-8') as f:
    ori_data = f.readlines()

with open(os.path.join(SAVE_DIR, basename.split('.')[0] + '_syn_samples.tsv')) as f:
    generation_data = f.readlines()

results = []
for item, generation_item in tqdm(zip(ori_data, generation_data), total=len(ori_data)):
    item = item.strip()
    _, pos, neg = generation_item.strip().split('\t')
    pos = json.loads(pos)
    neg = json.loads(neg)
    pos = json_repair.loads(pos)
    neg = json_repair.loads(neg)
    if type(pos) == dict and 'text' in pos:
        pos = pos['text']
    else:
        pos = str(pos)
    if type(neg) == dict and 'text' in neg:
        neg = neg['text']
    else:
        neg = str(neg)
    results.append({
        'ori': item,
        'pos': pos,
        'neg': neg
    })

with open(SAVE_FILE, mode='w+') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# %%
