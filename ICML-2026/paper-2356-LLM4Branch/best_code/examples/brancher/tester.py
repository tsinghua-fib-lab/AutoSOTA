import time
import numpy as np
import os, csv
import ecole # , pyscipopt
import logging
import argparse
from functools import partial
from utils.utils import geometric_mean_stable, get_scip_params, load_program, normalize, create_instance

logger = logging.getLogger(__name__)
logger.propagate = False

def solver(dataset, level, seed = 999999999, branch_method = 'replscost', scip_params = None, score_function = None, used_features = None):
    solver_result = {
        "validity": True,
        'seed': seed, 
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

        elif branch_method in ['relpscost', 'fullstrong', 'vanillafullstrong', 'pseudo']:
            env = ecole.environment.Configuring(scip_params=scip_params)
            env.seed(0)
            proc_time = time.process_time()
            wall_time =  time.perf_counter()
            observation, action_set, _, done, info = env.reset(instance)
            _, _, _, _, info = env.step({})
            scip_model = env.model.as_pyscipopt()
            print((env.model.primal_bound - env.model.dual_bound) / min([env.model.primal_bound, env.model.dual_bound]))
            print(env.model.primal_bound, env.model.dual_bound)
            solver_result["proc_time"] = time.process_time() - proc_time
            solver_result["wall_time"] = time.perf_counter() - wall_time

        scip_model = env.model.as_pyscipopt()
        solver_result["validity"] = True
        solver_result['nodes'] = scip_model.getNNodes()
        solver_result['nlps'] = scip_model.getNLPs()
        solver_result['gap'] = abs(env.model.primal_bound - env.model.dual_bound) / min([env.model.primal_bound, env.model.dual_bound])

    except Exception as exc:
        solver_result["error_info"] = False
        solver_result["error_info"] = f"Exception occurred: {type(exc).__name__}: {exc}"
        print(solver_result["error_info"])
    finally:
        return solver_result

def evaluate(program_path, dataset, level, method = None, instance_num = 80, seed = 999999999):
    pool = None 
    final_result = {
        "validity": True,
        "error_info": None
    }
    try:
        print(f"Start evaluation {dataset} in {level} by {method}")
        seeds = list(range(seed, seed + instance_num))
        result_list = []

        scip_params_plain = get_scip_params(time_limit=3600)
        for seed in seeds:
            if method == 'evolve':
                score_function, used_features, function_params, _ = load_program(program_path)
                score_function = partial(score_function, params=function_params)
                instance_result = solver(dataset, level, seed = seed, branch_method = "evolve", scip_params = scip_params_plain, score_function = score_function, used_features = used_features)
            elif method in ['relpscost', 'fullstrong', 'vanillafullstrong']:
                scip_params = {**scip_params_plain, f"branching/{method}/priority": 9999999}
                instance_result = solver(dataset, level, seed = seed, branch_method = method, scip_params = scip_params, score_function = None, used_features = None)
            print(f"seed: {seed}, walltime: {instance_result['wall_time']}, proctime: {instance_result['proc_time']}, nodes: {instance_result['nodes']}, gap: {instance_result['gap']}")
            result_list.append(instance_result)

        print(f"finish evaluation {dataset} in {level} by {method}")
        fieldnames = [key for key in result_list[0].keys() if (key != 'error_info' and key != 'validity' and key != 'done')]
        csv_dir = f"./examples/brancher/result/{dataset}/{level}"
        os.makedirs(csv_dir, exist_ok=True)
        method_result = sorted(result_list, key=lambda result: result['seed'])
        with open(F"{csv_dir}/{method}.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in method_result:
                writer.writerow({
                    **{key: result[key] for key in fieldnames}
                })
                csvfile.flush()
        print(f'branching policy result: {geometric_mean_stable([result["proc_time"] for result in result_list])} (time), {geometric_mean_stable([result["nodes"] for result in result_list])} (nodes), {geometric_mean_stable([result["gap"] for result in result_list])} (gap)')
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
    parser.add_argument("--program_path", help="Path to the program file, needed if method is evolve or symb4co", default=None)
    parser.add_argument("--dataset", "-d", help="dataset name", default='setcover', choices=['setcover', 'cauctions', 'facilities', 'indset'])
    parser.add_argument("--method", "-m", help="method name", default='evolve', choices=['evolve', 'relpscost', 'fullstrong'])
    parser.add_argument("--num_instances", "-n", help="number of instances to evaluate", default=80, type=int)
    parser.add_argument("--easy", help="dataset level", action="store_true")
    parser.add_argument("--medium", help="dataset level", action="store_true")
    parser.add_argument("--hard", help="dataset level", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.easy:
        print(evaluate(args.program_path, dataset = args.dataset, level = 'easy', method = args.method, instance_num = args.num_instances))
    if args.medium:
        print(evaluate(args.program_path, dataset = args.dataset, level = 'medium', method = args.method, instance_num = args.num_instances))
    if args.hard:
        print(evaluate(args.program_path, dataset = args.dataset, level = 'hard', method = args.method, instance_num = args.num_instances))