import os
import copy
from multiprocessing import Pool

import graph_gen
from utils import base_utils as bu
from config import DataGenConf, DataGenParams, GraphGenConf, load_config, dump_config

DATA_DIR = 'datasets'
DEFAULT_CONFIG = 'data_gen.yaml'


def generate_data(cfg: DataGenConf):
    src_dir = os.path.join(os.path.dirname(__file__), DATA_DIR, bu.readable_time())
    os.makedirs(src_dir)
    dump_config(cfg, src_dir)

    graph_cfg: GraphGenConf = cfg.graph_gen_conf
    params: DataGenParams =  cfg.params
    graph_cfg.verbose = False
    if params.threading:
        t_pool = Pool(params.workers)

    for node in params.nodes:
        if params.verbose:
            print(f'Generating data for {node} nodes')
        p_path = f'{src_dir}/{bu.get_nodes_dir_name(node)}'
        os.mkdir(p_path)
        if params.threading:
            future = list()
        graph_cfg.nodes = node
        for i in range(params.n):
            if params.threading:
                _g_cfg = copy.deepcopy(graph_cfg)
                _g_cfg.seed = i
                future.append(t_pool.starmap_async(graph_gen.generate_graph,
                                                   [(_g_cfg, f'{p_path}/{i}-sample')]))
            else:
                graph_cfg.seed = i
                graph_gen.generate_graph(graph_cfg, f'{p_path}/{i}')
        if params.threading:
            for f in future: f.get()
    if params.threading:
        t_pool.close()
        t_pool.join()
    if params.verbose:
        print(f'Generated data is stored at {src_dir}')


if __name__ == '__main__':
    cfg: DataGenConf = load_config(DEFAULT_CONFIG, DataGenConf)
    s_time = bu.current_time()
    generate_data(cfg)
    print(f'Generating dataset took {round(bu.current_time() - s_time, 3)} sec')
