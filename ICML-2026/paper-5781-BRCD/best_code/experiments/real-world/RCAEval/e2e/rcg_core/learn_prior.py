import argparse
from multiprocessing import Pool

import learn_kess_g
from utils import base_utils as bu
from config import LearnPriorConf, load_config

DEFAULT_CONFIG = 'learn_prior.yaml'


def learn_prior(src_dir, cfg: LearnPriorConf):
    if cfg.threading:
        t_pool = Pool(cfg.workers)

    for _k in cfg.k:
        if cfg.verbose:
            print(f'Learning prior graph for k={_k}')
        for node, n_path in bu.dir_iterator(src_dir):
            if cfg.verbose:
                print(f'Working on {node} nodes')
            if cfg.threading:
                future = list()
            for _, i_path in bu.dir_iterator(n_path):
                if cfg.threading:
                    future.append(t_pool.starmap_async(learn_kess_g.learn,
                                                       [(i_path, None, _k, cfg.oracle)]))
                else:
                    learn_kess_g.learn(i_path, k=_k, oracle=cfg.oracle)
            if cfg.threading:
                for f in future: f.get()
    if cfg.threading:
        t_pool.close()
        t_pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates prior graphs from a dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the data')
    args = parser.parse_args()
    path = args.path
    cfg: LearnPriorConf = load_config(DEFAULT_CONFIG, LearnPriorConf)

    s_time = bu.current_time()
    learn_prior(path, cfg)
    print(f'Learning prior took {round(bu.current_time() - s_time, 3)} sec')
