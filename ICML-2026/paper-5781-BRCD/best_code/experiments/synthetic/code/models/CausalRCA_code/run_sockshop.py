#!/usr/bin/env python3

import os
import datetime
import argparse
from multiprocessing import Pool
from collections import defaultdict

import numpy as np

import causalRCA

SEED = 42
ITERATIONS = 50
RESULT_DIR = 'exp_results/sockshop'

K = [1, 3, 5]
THREADING = True
WORKERS = 3


def _run(path, true_service, logs_file):
    # if THREADING:
    #     t_pool = Pool(WORKERS)
    #     futures = []

    # k_counter = {k: 0 for k in K}
    # k_time = {k: 0 for k in K}
    # def _get_result(result):
    #     nonlocal k_counter, k_time
    #     for k in K:
    #         services = [x.split('_')[0] for x in result['root_cause'][:k]]
    #         if true_service in services:
    #             k_counter[k] += 1
    #         k_time[k] += result['time']

    # for _ in range(ITERATIONS):
    #     seed = np.random.randint(1, 2**16 - 1)
    # if THREADING:
    #     futures.append(t_pool.starmap_async(causalRCA.multi_exp, [(path, true_service, ITERATIONS, K, )]))
    # else:
    counts, times = causalRCA.multi_exp(path, true_service, ITERATIONS, K)
    logs_file.write(f'{path} | counts={counts} | times={times}\n')

    # if THREADING:
    #     for f in futures:
    #         result = f.get()[0]
    #         logs_file.write(f'{path} | seed={seed} | result={result}\n')
    #         _get_result(result)
    #     t_pool.close()
    #     t_pool.join()

    # counts = {k: count / ITERATIONS for k, count in k_counter.items()}
    # times = {k: t / ITERATIONS for k, t in k_time.items()}
    return counts, times

def store_result(result, store_path):
    file = open(f'{store_path}/result.txt', 'w')
    services = ['carts', 'catalogue', 'orders', 'payment', 'user']
    metrics = ['cpu', 'mem']
    for k in K:
        file.write(f'k={k}\n')
        for m in metrics:
            file.write(f'{m}\n')
            for s in services:
                file.write(f"{s} => recall={round(result[k][s][m]['recall'], 2)}, time={round(result[k][s][m]['time'], 2)}\n")
    file.close()
    print(f'The result is stored at {store_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the algorithm on sockshop data')
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    path = args.path

    store_path = f'{RESULT_DIR}/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    os.makedirs(store_path)

    config = f"""
        seed = {SEED}
        ITERATIONS = {ITERATIONS}
        K = {K}
        path = {path}
    """
    with open(f'{store_path}/config.txt', 'w') as f:
        f.write(config)

    np.random.seed(SEED)
    logs_file = open(f'{store_path}/logs.txt', 'w')
    result = defaultdict(lambda: defaultdict(dict))
    for service in os.listdir(path):
        _serv, _met = service.split('-')
        p_path = os.path.join(path, service)
        if not os.path.isdir(p_path): continue
        print(f'Working on {service}...')
        logs_file.write(f'_serv={_serv} | _met={_met}\n')

        counter = 0
        k_sum = {k: 0 for k in K}
        k_time = {k: 0 for k in K}

        for itr in os.listdir(p_path):
            c_path = os.path.join(p_path, itr)
            if not os.path.isdir(c_path): continue
            _k_recall, _k_time = _run(c_path, _serv, logs_file)
            for k in K:
                k_sum[k] += _k_recall[k]
                k_time[k] += _k_time[k]
            counter += 1
        for k in K:
            result[k][_serv][_met] = {'recall': k_sum[k] / counter,
                                      'time': k_time[k] / counter}
        print(result)
    logs_file.close()
    store_result(result, store_path)
