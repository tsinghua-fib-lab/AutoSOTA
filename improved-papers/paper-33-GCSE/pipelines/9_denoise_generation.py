# %%
import os
import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoTokenizer

import sys
sys.path.append("../")

from main.predictors.cse_predictor import Predictor

cmd_args = False
#
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=1, help='n_gpu')
parser.add_argument('--file_dir', default='../dataset/mix', help='file name')
parser.add_argument('--file_name', default='__combine.txt', help='file name of the dataset')
parser.add_argument('--save_type_name', default='GLM4', help='the prefix name of save dir (usually is the LLM name)')
parser.add_argument('--model_from_pretrained', default='/home/lpc/models/bert-base-uncased/', help='model from pretrained')
parser.add_argument('--alpha', default=16, help='for filtering false positives')
parser.add_argument('--beta', default=15, help='for filtering false negatives')
parser.add_argument('--max_seq_len', default=32, help='model from pretrained')
parser.add_argument('--batch_size', default=128, help='batch size')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

tokenizer = AutoTokenizer.from_pretrained(args.model_from_pretrained)
pred = Predictor(tokenizer=tokenizer,
                  from_pretrained=args.model_from_pretrained,
                  max_seq_len=int(args.max_seq_len),
                  hard_negative_weight=0,
                  batch_size=int(args.batch_size),
                  temp=0.05)

# %%
SOURCE_FILE = os.path.join(args.file_dir, args.file_name)
basename = os.path.basename(SOURCE_FILE)
SAVE_DIR = os.path.join(os.path.dirname(SOURCE_FILE))
DA_DIR = os.path.join(os.path.dirname(SOURCE_FILE), basename.split('.')[0] + f'_{args.save_type_name}_DA')
SAVE_FILE = os.path.join(SAVE_DIR,
                         basename.split('.')[0]+'_syn_train.jsonl')
BATCH_SIZE = int(args.batch_size)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

with open(os.path.join(DA_DIR, basename.split('.')[0]+'_syn_samples.jsonl'), encoding='utf-8') as f:
    lines = f.readlines()

scores = []
inputs = []

for idx, line in tqdm(enumerate(lines), total=len(lines)):
    line = json.loads(line)
    ori, pos, neg = line['ori'], line['pos'], line['neg']
    inputs.append([ori, pos])
    inputs.append([ori, neg])

for output in pred.pred(inputs):
    scores += torch.diag(output.logits).tolist()

# %%
result = []
groups = {}
for idx, line in tqdm(enumerate(lines), total=len(lines)):
    line = json.loads(line)
    ori, pos, neg = line['ori'], line['pos'], line['neg']
    pos_score = scores[2 * idx]
    neg_score = scores[2 * idx + 1]
    
    groups[str(idx)] = {
        'ori': ori,
        'samples': [{
            'score': pos_score,
            'text': pos
        },{
            'score': neg_score,
            'text': neg
        }]
    }

neg_from_other_batch = 0
iter = tqdm(groups.values())
for idx, group in enumerate(iter):
    text1 = group['ori']
    text2 = text1
    neg = groups[str(idx + 6)]['ori'] if idx + \
            6 < len(groups.values()) else groups[str(idx - 6)]['ori']
    
    exists_pos = False
    for sample in group['samples']:
        if sample['score'] >= int(args.alpha):
            sample['used'] = True
            text2 = sample['text']
            exists_pos = True
            break
    
    exists_neg = False

    for sample in group['samples']:
        if sample['score'] < int(args.beta):
            sample['used'] = True
            neg = sample['text']
            exists_neg = True
            break

    result.append({
        'text1': text1,
        'text2': text2,
        'neg': neg
    })

    for sample in group['samples']:
        if 'used' not in sample or not sample['used']:
            result.append({
                'text1': sample['text'],
                'text2': sample['text'],
                'neg': neg
            })
    
    iter.set_description(f'neg_from_other_batch: {neg_from_other_batch}')
    if not exists_neg:
        neg_from_other_batch += 1
        continue
    

with open(SAVE_FILE, 'w') as f:
    for line in result:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')

# %%
