import os
import time
import argparse
from multiprocessing import Pool

import pandas as pd

import rcd
# import find_root_cause as ft
import mutual_info as mt
# import marginal_ci as mci
# import ikpc
# import cmi
# import mi_cmi
# import mi_graph
import m_igs
import page_rank
# import toca
# import random_selection
# import boss
import rcg
import baro
import smooth

from utils import base_utils as bu
from config import ExperimentConf, load_config, dump_config

PAGERANK = 'page_rank'
BARO = 'baro'
MUTUAL_INFO = 'mutual_info'
RCD = 'rcd'
M_IGS = 'm_igs'
SMOOTH_CH = 'smooth'
RCG_0 = 'rcg_0'
RCG_1 = 'rcg_1'
RCG_CPDAG = 'rcg_cpdag'
RCG_DAG = 'rcg_dag'

BASELINES = [
    PAGERANK,
    BARO,
    MUTUAL_INFO,
    SMOOTH_CH,
    RCD,
    M_IGS,
    RCG_CPDAG,
    RCG_DAG,
    
    # For the sampled version
    # RCG_0,
    # RCG_1,
]

RESULT_DIR = 'exp_results'
DEFAULT_CONFIG = 'experiments.yaml'


def run_baselines(src_dir, seed, cfg: ExperimentConf):
    a_node = bu.load_data(f'{src_dir}/{bu.GRAPH_GEN_INFO}')[bu.ANOMALOUS_NODE]
    result = {'l': cfg.l_value, 'seed': seed,
              'a_node': a_node, 'int_samples': cfg.interventional_samples}
    def _extract_result(result, prefix):
        l_rc = result['root_cause'][:cfg.l_value]
        accuracy = 1 if a_node in l_rc else 0
        return {
            f"{prefix}_tests": result['tests'],
            f"{prefix}_time": result['time'],
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_top_l_targets": l_rc,
        }
    (normal_df, anomalous_df) = bu.load_datasets(src_dir)

    # We use 10,000 samples for normal period and 1,000 samples for anomalous
    # By default, data_generator generates 10,000 samples for both normal
    # and anomalous dataset.
    anomalous_df = anomalous_df.sample(n=cfg.interventional_samples, random_state=seed, replace=False)
    anomalous_df.reset_index(drop=True, inplace=True)

    if PAGERANK in BASELINES:
        page_rank_r = page_rank.rank_variables(src_dir)
        result = {**result, **_extract_result(page_rank_r, PAGERANK)}

    if BARO in BASELINES:
        baro_r = baro.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True))
        result = {**result, **_extract_result(baro_r, BARO)}

    if SMOOTH_CH in BASELINES:
        smooth_new_r = smooth.rank_variables(normal_df.copy(deep=True),
                                                anomalous_df.copy(deep=True),
                                                src_dir)
        result = {**result, **_extract_result(smooth_new_r, SMOOTH_CH)}

    if MUTUAL_INFO in BASELINES:
        mutual_info_r = mt.rank_variables(normal_df.copy(deep=True), anomalous_df.copy(deep=True))
        result = {**result, **_extract_result(mutual_info_r, MUTUAL_INFO)}

    if RCD in BASELINES:
        rcd_r = rcd.top_k_rc(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir,
                            cfg.l_value, None, seed=seed, oracle=False, localized=True, verbose=cfg.verbose)
        result = {**result, **_extract_result(rcd_r, RCD)}

    if M_IGS in BASELINES:
        igs_r = m_igs.run_algo(normal_df.copy(deep=True), anomalous_df.copy(deep=True), src_dir,
                            perfect_ci=False, max_l=cfg.l_value)
        result = {**result, **_extract_result(igs_r, M_IGS)}

    if RCG_0 in BASELINES:
        alpha_r = rcg.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
                                src_dir, cfg.l_value, k=0, oracle=cfg.oracle)
        result = {**result, **_extract_result(alpha_r, RCG_0)}

    if RCG_1 in BASELINES:
        alpha_r = rcg.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
                                src_dir, cfg.l_value, k=1, oracle=cfg.oracle)
        result = {**result, **_extract_result(alpha_r, RCG_1)}

    if RCG_CPDAG in BASELINES:
        alpha_r = rcg.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
                                src_dir, cfg.l_value, k=-1, oracle=cfg.oracle)
        result = {**result, **_extract_result(alpha_r, RCG_CPDAG)}

    if RCG_DAG in BASELINES:
        rcg_dag_r = rcg.run(normal_df.copy(deep=True), anomalous_df.copy(deep=True),
                            src_dir, cfg.l_value, dag=True)
        result = {**result, **_extract_result(rcg_dag_r, RCG_DAG)}

    if cfg.verbose:
        print(f"Output: {result}")
    return result

# Change the total number of nodes
def different_nodes(src_dir, cfg: ExperimentConf):
    out_dir = f"{src_dir}/{RESULT_DIR}/{bu.readable_time()}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dump_config(cfg, out_dir)
    if cfg.threading:
        t_pool = Pool(cfg.workers)

    df_counter = 0
    df = pd.DataFrame()
    def _store(row):
        nonlocal df
        nonlocal df_counter
        if row is not None:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        if df_counter % 10 == 0: # Write in batch of 10
            df.to_csv(f'{out_dir}/data.csv', mode='a', header=(df_counter == 0), index=False)
            df = pd.DataFrame()
        df_counter += 1

    # Number of interventional samples to use
    cfg.interventional_samples = cfg.anomalous_samples
    for node, n_path in bu.dir_iterator(src_dir):
        print(f"Running the experiment with {node} nodes")
        for l in cfg.L:
            cfg.l_value = l
            if cfg.threading:
                future = list()
            for i, i_path in bu.dir_iterator(n_path):
                if cfg.threading:
                    future.append(t_pool.starmap_async(run_baselines, [(i_path, int(i), cfg)]))
                else:
                    _store({'nodes': node, **run_baselines(i_path, int(i), cfg)})

            if cfg.threading:
                for f in future:
                    _store({'nodes': node, **(f.get()[0])})
        
    _store(None)

    if cfg.threading:
        t_pool.close()
        t_pool.join()
    return out_dir

# Change the number of interventional samples
def different_int_samples(src_dir, cfg: ExperimentConf):
    out_dir = f"{src_dir}/{RESULT_DIR}/{bu.readable_time()}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dump_config(cfg, out_dir)
    if cfg.threading:
        t_pool = Pool(cfg.workers)

    df_counter = 0
    df = pd.DataFrame()
    def _store(row):
        nonlocal df
        nonlocal df_counter
        if row is not None:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        if df_counter % 10 == 0: # Write in batch of 10
            df.to_csv(f'{out_dir}/data.csv', mode='a', header=(df_counter == 0), index=False)
            df = pd.DataFrame()
        df_counter += 1

    node = '25'
    n_path = os.path.join(src_dir, bu.get_nodes_dir_name(node))
    for _int_sample in cfg.int_samples:
        print(f"Running the experiment with {_int_sample} interventional samples")
        cfg.l_value = 1 # top-1
        cfg.interventional_samples = _int_sample
        if cfg.threading:
            future = list()
        for i, i_path in bu.dir_iterator(n_path):
            if cfg.threading:
                future.append(t_pool.starmap_async(run_baselines, [(i_path, int(i), cfg)]))
            else:
                _store({'nodes': node, **run_baselines(i_path, int(i), cfg)})

        if cfg.threading:
            for f in future:
                _store({'nodes': node, **(f.get()[0])})
    _store(None)

    if cfg.threading:
        t_pool.close()
        t_pool.join()
    return out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all the baselines on the given dataset')
    parser.add_argument('--path', type=str, required=True, help='Path to the data')
    parser.add_argument('--exp', type=int, required=True, help='The type of experiment')
    args = parser.parse_args()
    path = args.path
    exp = args.exp
    cfg: ExperimentConf = load_config(DEFAULT_CONFIG, ExperimentConf)

    fn = {
            1: different_nodes, # Figure 3(a)
            2: different_int_samples # Figure3 (b)
        }
    start = time.perf_counter()
    src_dir = fn[exp](path, cfg)
    end = time.perf_counter()
    print(f"The experiment took {round(end - start, 3)} seconds")
    print(f"The result of the experiment is stored at {src_dir}")
