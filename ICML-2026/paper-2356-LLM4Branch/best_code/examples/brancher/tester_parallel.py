import time, csv
import numpy as np
import os, psutil, multiprocessing
import ecole, pyscipopt
from multiprocessing import Pool, current_process
import logging
from pathlib import Path
import traceback 
import argparse
from functools import partial
from utils.utils import numerical_mean_stable, geometric_mean_stable, get_scip_params, load_program, normalize, format_list, create_instance

from global_core_manager import GlobalCoreManager

logger = logging.getLogger(__name__)
logger.propagate = False

def configure_logger_for_process(log_level, log_file=None):
    logger = logging.getLogger() 
    logger.handlers = [] 
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def solver(dataset, level, seed = 999999999, branch_method = 'replscost', scip_params = None, score_function = None, used_features = None, device = 'cpu'):
    
    solver_result = {
        "validity": True,
        'seed': seed, 
        'wall_time': 0.0,
        "proc_time": 0.0,
        "nodes": 0.0,
        "gap": 0.0,
        "error_info": None
    }

    try:
        instance = create_instance(dataset, seed, level)
        if branch_method == 'evolve':
            flash_khalil_obs = ecole.observation.Khalil2016Flash()
            flash_node_obs = ecole.observation.NodeBipartiteFlash()
            extract_features_khalil = [fea - 19 for fea in used_features if fea >= 19]
            extract_features_node = [fea for fea in used_features if fea < 19]
            flash_khalil_obs.set_active_features(extract_features_khalil)
            flash_node_obs.set_active_features(extract_features_node)
            env = ecole.environment.Branching(observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs}, scip_params=scip_params)
            env.seed(0)
            proc_time = time.process_time()
            wall_time =  time.perf_counter()
            observation, action_set, _, done, info = env.reset(instance)
            while not done:
                variable_features = np.concatenate((observation['Node'].variable_features[action_set, :], observation['Khalil'].features[action_set, :]), axis = 1)
                variable_features[:, used_features] = np.nan_to_num(variable_features[:, used_features], nan=0.0, posinf=0.0, neginf=0.0) 
                variable_features = normalize(variable_features, used_features)
                sb_scores = score_function(variable_features)
                action = action_set[sb_scores.argmax()]
                observation, action_set, _, done, info = env.step(action)
            solver_result["proc_time"] = time.process_time() - proc_time
            solver_result["wall_time"] = time.perf_counter() - wall_time
            
        elif branch_method in ['relpscost', 'fullstrong', 'vanillafullstrong', 'pscost']:
            env = ecole.environment.Configuring(scip_params=scip_params)
            env.seed(0)
            proc_time = time.process_time()
            wall_time =  time.perf_counter()
            observation, action_set, _, done, info = env.reset(instance)
            _, _, _, _, info = env.step({})
            solver_result["proc_time"] = time.process_time() - proc_time
            solver_result["wall_time"] = time.perf_counter() - wall_time
       
        scip_model = env.model.as_pyscipopt()
        solver_result["validity"] = True
        solver_result['nodes'] = scip_model.getNNodes()
        solver_result['nlps'] = scip_model.getNLPs()
        solver_result['gap'] = abs(env.model.primal_bound - env.model.dual_bound) / min([env.model.primal_bound, env.model.dual_bound])
    except Exception as exc:
        solver_result["error_info"] = f"Exception occurred: {type(exc).__name__}: {exc}"
        print(solver_result["error_info"])
    finally:
        return solver_result


def evaluate_one(args):
    program_path, seed, cores_list, lock_dir, methods, dataset, level, log_level, log_file = args
    p = psutil.Process(current_process().pid)
    try:
        configure_logger_for_process(log_level, log_file)
        core_manager = GlobalCoreManager(cores_list, lock_dir)
        core_id = core_manager.get_core()
        logging.debug(f"Process {p.pid} got assigned to CPU core {core_id}.")
        p.cpu_affinity([core_id])
        logging.debug(f"Worker process {p.pid} successfully pinned to CPU core {core_id}.")
        scip_params_plain = get_scip_params()
        result_list = []
        for method in methods:
            if method == 'evolve':
                score_function, used_features, function_params, _ = load_program(program_path)
                score_function = partial(score_function, params=function_params)
                solver_result = solver(dataset, level, seed, 'evolve', scip_params_plain, score_function, used_features)
            elif method in ['relpscost', 'fullstrong', 'vanillafullstrong', 'pscost']:
                scip_params = {**scip_params_plain, f"branching/{method}/priority": 9999999}
                solver_result = solver(dataset, level, seed, method, scip_params)
            else:
                raise ValueError(f"Unknown reference method: {method}")
            result_list.append(solver_result)
        logging.info(f"Core: {core_id}, Seed: {seed}, Method: {methods}, Proc Time: {format_list([result['proc_time'] for result in result_list])}, Node: {format_list([result['nodes'] for result in result_list])}")
        return {
            "validity": True,
            "seed": seed,
            "results": result_list,
            "error_info": None
        }
    except Exception as exc:
        error_details = traceback.format_exc()
        print("An exception occurred:\n", error_details)
        return {
            "validity": False,
            "nodes": 0.0, 
            "gap": 0.0,
            "error_info": f"Exception occurred: {type(exc).__name__}: {exc}"
        }
    finally:
        if core_id != -1:
            core_manager.release_core(core_id)

def evaluate(program_path, dataset, level, cores_list = list(range(1,32)), lock_dir = './tmp/cpu_locks/eval', methods = [], max_workers = 8, instance_num = 80, seed = 999999999, csv = False, log_level = "INFO", log_file = None):
    pool = None 
    final_result = {
        "validity": True,
        "methods": methods,
        "avg_time": [],
        "max_time": [],
        "mean_time": [],
        "avg_nodes": [],
        "mean_nodes": [],
        "combined_score": 0.0, 
        "error_info": None,
    }
    try:
        configure_logger_for_process(log_level, log_file)
        logger.debug(f"Start evaluation")
        seeds = list(range(seed, seed + instance_num))
        tasks = [(program_path, seed, cores_list, lock_dir, methods, dataset, level, log_level, log_file) for seed in seeds]
        multiprocessing.set_start_method('spawn', force=True)
        pool = Pool(processes=max_workers)
        async_result = pool.map_async(evaluate_one, tasks)
        results = async_result.get()
        result_list = []
        logger.debug(f"Concatenating results from {len(results)} instances.")
        for i, result in enumerate(results):
            final_result["validity"] = final_result["validity"] and result["validity"]
            result_list.append(result["results"])
        if csv:
            fieldnames = [key for key in result_list[0][0].keys() if (key != 'error_info' and key != 'validity' and key != 'done')]
            for i, method in enumerate(methods):
                each_result = [result[i] for result in result_list]
                csv_dir = f"./examples/brancher/result/{dataset}/{level}"
                os.makedirs(csv_dir, exist_ok=True)
                method_result = sorted(each_result, key=lambda result: result['seed'])
                with open(F"{csv_dir}/{method}.csv", 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in method_result:
                        writer.writerow({
                            **{key: result[key] for key in fieldnames}
                        })
                        csvfile.flush()

        time_list = [[result['proc_time'] for result in solver_result] for solver_result in result_list]
        nodes_list = [[result['nodes'] for result in solver_result] for solver_result in result_list]
        gap_list = [[result['gap'] for result in solver_result] for solver_result in result_list]
        final_result["avg_time"] = np.apply_along_axis(numerical_mean_stable, axis = 0, arr = np.array(time_list)).tolist()
        final_result["mean_time"] = np.apply_along_axis(geometric_mean_stable, axis = 0, arr = time_list).tolist()
        final_result["max_time"] =  np.apply_along_axis(max, axis = 0, arr = time_list).tolist()
        final_result["avg_nodes"] = np.apply_along_axis(numerical_mean_stable, axis = 0, arr = nodes_list).tolist()
        final_result["mean_nodes"] = np.apply_along_axis(geometric_mean_stable, axis = 0, arr = nodes_list).tolist()
        final_result["max_nodes"] =  np.apply_along_axis(max, axis = 0, arr = nodes_list).tolist()
        final_result["avg_gap"] = np.apply_along_axis(numerical_mean_stable, axis = 0, arr = gap_list).tolist()
        final_result["mean_gap"] = np.apply_along_axis(geometric_mean_stable, axis = 0, arr = gap_list).tolist()
        final_result["max_gap"] =  np.apply_along_axis(max, axis = 0, arr = gap_list).tolist()

    except Exception as exc:
        logger.debug(f"Generate an exception {exc}")
        final_result =  {
            "validity": False,
            "error_info": f"Exception occurred: {type(exc).__name__}: {exc}"
        }
    finally:
        if pool is not None:
            pool.terminate() 
            pool.join()
        return final_result

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluation for Program")
    parser.add_argument("--program_path", help="Path to the program file", default=None)
    parser.add_argument("--dataset", "-d", help="dataset name", default='setcover', choices=['setcover', 'cauctions', 'facilities', 'indset'])
    parser.add_argument("--easy", help="dataset level", action="store_true")
    parser.add_argument("--medium", help="dataset level", action="store_true")
    parser.add_argument("--hard", help="dataset level", action="store_true")
    parser.add_argument("--csv", help="save to csv file", action="store_true")
    parser.add_argument("--cores_list", help="cores_list", default="1-31")
    parser.add_argument("--lock_dir", help="lock dir", default=None)
    parser.add_argument("--output", "-o", help="output_dir", default=None)
    parser.add_argument("--log_level", "-l", help="logger file", default="INFO")
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    METHODS = ['relpscost', 'evolve'] # METHODS = ['fullstrong', 'relpscost', 'evolve']  

    Path(args.output).mkdir(exist_ok=True, parents=True)
    log_file = os.path.join(args.output, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.log")
    configure_logger_for_process(args.log_level, log_file)
    start, end = args.cores_list.split('-')
    cores_list = list(range(int(start), int(end)+1))

    # Test Easy
    if args.easy:
        logging.info(f"Testing program: {args.program_path} in Easy setting")
        logging.info(f"Easy Result: {evaluate(args.program_path, dataset = args.dataset, level = 'easy', cores_list = cores_list, lock_dir = args.lock_dir, methods = METHODS, max_workers = 20, instance_num = 80, csv = args.csv, log_level = args.log_level, log_file = log_file)}")

    # Test Medium
    if args.medium:
        logging.info(f"Testing program: {args.program_path} in Medium setting")
        logging.info(f"Medium Result: {evaluate(args.program_path, dataset = args.dataset, level = 'medium', cores_list = cores_list, lock_dir = args.lock_dir, methods = METHODS, max_workers = 20, instance_num = 80, csv = args.csv, log_level = args.log_level, log_file = log_file)}")
    
    # Test Hard
    if args.hard:
        logging.info(f"Testing program: {args.program_path} in Hard setting")
        logging.info(f"Hard Result: {evaluate(args.program_path, dataset = args.dataset, level = 'hard', cores_list = cores_list, lock_dir = args.lock_dir, methods = METHODS, max_workers = 20, instance_num = 80, csv = args.csv, log_level = args.log_level, log_file = log_file)}")