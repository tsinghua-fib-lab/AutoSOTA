import argparse
import os
import sys
import numpy as np
import time
from evaluation.metrics import get_metrics
from model.impact import IMPACT
from utils import get_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=f'./datasets/')
parser.add_argument('--data', type=str,
                    default='SAD',
                    help='dataset name',
                    choices=['PSM', 'ASD', 'SMD', 'UCR', 'CT', 'SAD', 'PTBXL', 'TUSZ'])
parser.add_argument('--model', type=str, default='IMPACT')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--epoch_steps', type=int, default=40)
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument('--rep_dim', help='', type=int, default=64)
parser.add_argument('--hidden_dims', help='', type=str, default='64')
parser.add_argument('--act', help='', type=str, default='ReLU')
parser.add_argument('--lr', help='', type=float, default=0.0003)
parser.add_argument('--batch_size', help='', type=int, default=64)
parser.add_argument('--setting', help='', type=str, default='general')
parser.add_argument('--anomaly_class_idx', help='', type=int, default=0)
parser.add_argument('--lambd', help='', type=float, default=1.0)
parser.add_argument('--k', help='', type=int, default=5)

args = parser.parse_args()
model_configs = {
    'epochs': args.num_epochs,
    'epoch_steps': args.epoch_steps,
    'batch_size': args.batch_size,
    'lr': args.lr,
    'rep_dim': args.rep_dim,
    'hidden_dims': args.hidden_dims,
    'act': args.act,
    'lambd': args.lambd,
    'k': args.k,
}

datasets = args.data.split(',')
for dataset in datasets:
    print(dataset)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    eval_metrics_lst = []
    t_lst = []
    train_data, train_label, test_data, test_label = get_data(dataset, args.data_root, setting=args.setting, anomaly_class_idx=args.anomaly_class_idx)
    for i in range(args.runs):
        print(f'\n\nRunning [{args.model}] on [{dataset}]  [{i+1}/{args.runs}], '
              f'cur_time: {time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())}')
        start_time = time.time()

        model = IMPACT(**model_configs, random_state=42+i)

        model.fit(train_data, train_label)
        scores = model.decision_function(test_data, test_label)
        end_time = time.time()
        run_time = end_time - start_time
        evaluation_result = get_metrics(scores, test_label)
        txt = f'{args.model}-{dataset}, '
        txt += f'AUC: {evaluation_result[0]:.4f}'
        txt += f', {run_time:.2f}s'
        txt += f', runs {i + 1}/{args.runs}'
        print(txt)
        eval_metrics_lst.append(evaluation_result)
        t_lst.append(run_time)

    avg, std = np.average(np.array(eval_metrics_lst), axis=0), np.std(np.array(eval_metrics_lst), axis=0)
    avg_t, std_t = np.average(np.array(t_lst)), np.std(np.array(t_lst))
    txt = f'{args.model}-{dataset}, '
    txt += f'AUC: {avg[0]:.4f}'
    txt += f', Time: {avg_t:.2f}s'
    txt += f', avg'
    print(txt)
    txt = f'{args.model}-{dataset}, '
    txt += f'std: {std[0]:.4f}'
    txt += f', T_std: {std_t:.2f}s'
    txt += f', std'
    print(txt)