# %%
import os
from main.evaluation import *
from main.models.gcse import GCSE
from argparse import ArgumentParser

cmd_args = False
# 添加 参数 n_gpu
parser = ArgumentParser()
parser.add_argument('--n_gpu', default=1, help='n_gpu')
parser.add_argument('--task_set', default='sts', help='batch size')
parser.add_argument('--model_path', default='/home/lpc/models/bert-base-uncased/', help='model name or path')
parser.add_argument('--tokenizer_path', default='/home/lpc/models/bert-base-uncased/', help='tokenizer path')

if not cmd_args:
    args = parser.parse_args([]) # You can directly set above parameters in the default.
else:
    args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

model_path = args.model_path
tokenizer_path = args.tokenizer_path
model = GCSE(from_pretrained=model_path,
                                pooler_type='cls')

main([
    '--model_name_or_path', model_path,
    '--tokenizer_path', tokenizer_path,
    '--task_set', args.task_set
], model)

