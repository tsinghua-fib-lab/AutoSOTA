import sys
from collections import defaultdict

import numpy as np
import random
import torch

from config.basic import ConfigBasic
from utils.util import write_log, get_current_time, to_np, make_dir, log_configs
from utils.util import extract_embs, print_eval_result_by_groups_and_k, evaluate_metric
from utils.comparison_utils import find_kNN
from networks.util import prepare_model
from data.get_datasets_AGE import get_datasets_AGE

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ConOrd Test')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint file')
    parser.add_argument('--dataset', type=str, default='clap',
                        help='dataset')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which cuda device to use')
    return parser.parse_args()


def set_local_config(cfg, args):
    cfg.dataset = args.dataset
    cfg.logscale = False
    cfg.set_dataset()
    cfg.tau = 0

    # model
    cfg.model = 'ConOrd'
    cfg.backbone = 'vitB16'
    cfg.ref_mode = 'flex'
    cfg.ref_point_num = 60
    cfg.start_norm = True

    # eval
    cfg.k = np.arange(2, 60, 2)
    cfg.metric = 'L2'
    cfg.batch_size = 128
    cfg.test_batch_size = 1000

    cfg.wandb = False
    cfg.experiment_name = f'test_{cfg.dataset}'
    cfg.save_folder = f'../results_test/{cfg.dataset}/{cfg.experiment_name}_{get_current_time()}'
    make_dir(cfg.save_folder)

    cfg.n_gpu = torch.cuda.device_count()
    cfg.num_workers = 1
    cfg.gpu_ids = [args.gpu]
    cfg.device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
    return cfg


def main():
    random_seed = 999
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

    args = parse_args()
    cfg = ConfigBasic()
    cfg = set_local_config(cfg, args)
    cfg.logfile = log_configs(cfg, log_file='test_log.txt')

    # dataloader
    loader_dict = get_datasets_AGE(cfg)
    cfg.n_ranks = loader_dict['train'].dataset.ranks.max() + 1
    cfg.ref_point_num = cfg.n_ranks
    cfg.fiducial_point_num = cfg.n_ranks
    print(f'[*] {cfg.n_ranks} ranks exist.')

    # model
    model = prepare_model(cfg)
    model = model.to(cfg.device)

    state = torch.load(args.checkpoint, map_location=cfg.device)
    model.load_state_dict(state['model'])
    write_log(cfg.logfile, f'[*] Loaded checkpoint: {args.checkpoint}')

    mae, cs, best_k = test_AGE(loader_dict, model, cfg)
    write_log(cfg.logfile, f'[Result] MAE: {mae:.3f}, CS: {cs:.4f}, best_k: {best_k}')


def test_AGE(loader_dict, model, cfg):
    model.eval()

    embs_train = extract_embs(model.encoder, loader_dict['train_for_val'], cfg)
    embs_train = embs_train.to(cfg.device)

    embs_test = extract_embs(model.encoder, loader_dict['val'], cfg)
    embs_test = embs_test.to(cfg.device)

    n_test = len(embs_test)
    n_batch = int(np.ceil(n_test / cfg.test_batch_size))

    test_labels = loader_dict['val'].dataset.labels
    train_labels = loader_dict['train_for_val'].dataset.labels

    preds_all = defaultdict(list)
    with torch.no_grad():
        for idx in range(n_batch):
            i_st = idx * cfg.test_batch_size
            i_end = min(i_st + cfg.test_batch_size, n_test)

            _, inds = find_kNN(
                embs_test[i_st:i_end].view(i_end - i_st, -1),
                embs_train,
                k=max(cfg.k),
                metric=cfg.metric
            )
            inds = np.squeeze(to_np(inds), 0)
            if inds.ndim == 1:
                inds = inds[np.newaxis, :]

            for k in cfg.k:
                nn_labels = train_labels[inds[:, :k]]
                pred_mean = np.round(np.mean(nn_labels, axis=-1, dtype=np.float32))
                preds_all[k].append(pred_mean)

    for key in preds_all.keys():
        preds_all[key] = np.concatenate(preds_all[key])

    best_mae, best_k = print_eval_result_by_groups_and_k(
        test_labels, train_labels, preds_all, cfg.logfile, interval=3
    )
    mae, cs, _ = evaluate_metric(preds_all[best_k], test_labels)

    write_log(cfg.logfile, f'MAE: {mae:.3f}, CS: {cs:.4f}')
    sys.stdout.flush()

    return mae, cs, best_k


if __name__ == '__main__':
    main()
