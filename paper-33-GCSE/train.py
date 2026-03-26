# %%
from argparse import ArgumentParser

use_cmd = False
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=1)
parser.add_argument('--model', default='/home/lpc/models/bert-base-uncased/')
parser.add_argument('--data_name', default='DomainFullUn')
if use_cmd:
    args = parser.parse_args()
else:
    args = parser.parse_args([])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.n_gpu}"
from main.trainers.gcse_trainer import Trainer
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained(args.model)
trainer = Trainer(tokenizer=tokenizer,
                  from_pretrained=args.model,
                  base_from_pretrained=args.model,
                  data_present_path='./dataset/present.json',
                  max_seq_len=64,
                  hard_negative_weight=0,
                  batch_size=128,
                  temp=0.05,
                  data_name=f'{args.data_name}',
                  task_name=f'GCSE_{args.data_name}_unsup')

for i in trainer(num_epochs=2, lr=1e-5, gpu=[0], eval_call_step=lambda x: x % 125 == 0, save_per_call=True):
    a = i

# %%
