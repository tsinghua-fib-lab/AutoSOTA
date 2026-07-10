import time
import numpy as np
import ecole
import psutil
from multiprocessing import Pool, current_process, TimeoutError
import logging, traceback
from functools import partial
from skopt import Optimizer
from skopt.space import Real

# 从您的项目中导入必要的辅助函数
from utils.utils import numerical_mean_stable, geometric_mean_stable, get_scip_params, load_program, normalize, create_instance
from global_core_manager import GlobalCoreManager

# --- 配置部分 ---
logger = logging.getLogger(__name__)

def execute_ecole(args):
    task, dataset, seed, cores_list, lock_dir, score_function, used_features, function_params, scip_params, explore_index, explore_prob, dataset_dir = args
    core_manager = GlobalCoreManager(cores_list, lock_dir)
    p = psutil.Process(current_process().pid)
    result = {
        "task": task,
        "seed": seed,
        "validity": False
    }
    try:
        core_id = core_manager.get_core()
        if score_function and function_params:
            score_function = partial(score_function, params=function_params)
        instance = create_instance(dataset, seed, 'easy', dataset_dir)
        logger.debug(f"Process {p.pid} got assigned to CPU core {core_id}.")
        p.cpu_affinity([core_id])
        if task == "evolve":
            flash_khalil_obs = ecole.observation.Khalil2016Flash()
            flash_node_obs = ecole.observation.NodeBipartiteFlash()
            extract_features_khalil = [fea - 19 for fea in used_features if fea >= 19]
            extract_features_node = [fea for fea in used_features if fea < 19]
            flash_khalil_obs.set_active_features(extract_features_khalil)
            flash_node_obs.set_active_features(extract_features_node)
            env = ecole.environment.Branching(observation_function={"Node": flash_node_obs, "Khalil": flash_khalil_obs}, scip_params=scip_params)
            env.seed(0)
            proc_time = time.process_time()
            observation, action_set, _, done, info = env.reset(instance)
            while not done:
                variable_features = np.concatenate((observation['Node'].variable_features[action_set, :], observation['Khalil'].features[action_set, :]), axis = 1)
                variable_features = normalize(variable_features, used_features)
                sb_scores = score_function(variable_features)
                action = action_set[sb_scores.argmax()]
                observation, action_set, _, done, info = env.step(action)
            result["time"] = time.process_time() - proc_time
        elif task == "relpscost":
            env = ecole.environment.Configuring(scip_params=scip_params)
            env.seed(0)
            proc_time = time.process_time()
            observation, action_set, _, done, info = env.reset(instance)
            _, _, _, _, info = env.step({})
            result["time"] = time.process_time() - proc_time

        scip_model = env.model.as_pyscipopt()
        result['validity'] = True
        result['nodes'] = scip_model.getNNodes()
        result['gap'] = abs(env.model.primal_bound - env.model.dual_bound) / (abs(min([env.model.primal_bound, env.model.dual_bound])) + 1e-6)
        logger.debug(f"seed {seed}: {result}")
        return result
    except Exception as exc:
        error_details = traceback.format_exc()
        logger.debug(f"Generate an exception {error_details}")
        return {
            "task": task,
            "validity": False,
            "time": 0.0,
            "nodes": 0.0,
            "seed": seed,
            "error_info": f"Generate an exception {error_details}"
        }
    finally:
        if core_id != -1:
            core_manager.release_core(core_id)
        logger.debug(f"{task} in seed {seed} is finished!")

# Bayesian Optimization for program parameters
def params_optimizer(dataset, seeds, cores_list, lock_dir, score_function, used_features, function_params, bound_params, scip_params, config):
    logger.debug("--- Starting Parallel Bayesian Optimization ---")
    param_space = [Real(min_val, max_val) for min_val, max_val in bound_params]
    opt = Optimizer(dimensions=param_space, acq_func="gp_hedge", random_state=123)
    dataset_dir = None if config.dataset in ['setcover', 'cauctions', 'facilities', 'indset'] else config.dataset_dir
    def objective_function_parallel(function_params, seeds, score_function, used_features, scip_params, pool):
        sub_tasks = []
        for seed in seeds:
            sub_tasks.append(('evolve', dataset, seed, cores_list, lock_dir, score_function, used_features, function_params, scip_params, None, None, dataset_dir))
        results = pool.map(execute_ecole, sub_tasks)
        all_result = {seed: {} for seed in seeds}
        for result in results:
            if not result["validity"]:
                logger.warning(f"A sub-task failed: {result.get('error_info')}")
                return 0.0
            seed_index = result["seed"]
            all_result[seed_index]["evolve_time"] = result["time"]
            all_result[seed_index]["evolve_nodes"] = result["nodes"]
            all_result[seed_index]["evolve_gap"] = result["gap"]
        mean_nodes = numerical_mean_stable([all_result[seed]["evolve_nodes"] for seed in seeds])
        mean_time = numerical_mean_stable([all_result[seed]["evolve_time"] for seed in seeds])
        mean_gap = numerical_mean_stable([all_result[seed]["evolve_gap"] for seed in seeds])
        vanilla_score = 0.0
        if config.target == 'time':
            vanilla_score = mean_time
        elif config.target == 'nodes':
            vanilla_score = mean_nodes
        elif config.target == 'gap':
            vanilla_score = mean_gap
        logger.debug(f"Params: {[f'{p:.3f}' for p in function_params]}, Time: {mean_time:.5f}, Node: {mean_nodes:.5f}")
        return vanilla_score

    with Pool(processes=config.max_workers) as pool:
        f_val_initial = objective_function_parallel(function_params, seeds, score_function, used_features, scip_params, pool)
        logger.debug(f"Initial Score: {f_val_initial}")
        opt.tell([function_params], [f_val_initial])
        x0 = opt.ask(config.n_initial - 1)
        y0 = [objective_function_parallel(x, seeds, score_function, used_features, scip_params, pool) for x in x0]
        opt.tell(x0, y0)
        for i in range(config.n_initial, config.n_calls):
            logger.debug(f"\n--- Optimization Step {i+1}/{config.n_calls} ---")
            next_x = opt.ask()
            f_val = objective_function_parallel(next_x, seeds, score_function, used_features, scip_params, pool)
            opt.tell(next_x, f_val)

    result = opt.get_result()
    logger.debug("\n--- Bayesian Optimization Finished ---")
    logger.info(f"New Score Found by Parameter Opt: {f_val_initial:.6f} -> {result.fun:.6f}")
    logger.info(f"Optimal parameters found: {result.x}")
    return result.x

# end to end evaluate
def evaluate(program_path, config):
    pool = None 
    final_result = {
        "validity": True,
        "combined_score": 0.0,
        "error_info": None,
        "avg_time": 0.0,
        "max_time": 0.0,
        "mean_time": 0.0,
        "avg_nodes": 0.0,
        "max_nodes": 0.0,
        "mean_nodes": 0.0,
        "avg_gap": 0.0,
        "max_gap": 0.0,
        "mean_gap": 0.0
    }
    try:
        logger.debug(f"Start main evaluation")
        train_seeds = list(range(config.train_seed, config.train_seed + config.train_num))
        seeds = list(range(config.valid_seed, config.valid_seed + config.valid_num))
        score_function, used_features, function_params, bound_params = load_program(program_path)
        # scip setting
        if config.target in ['time', 'nodes']:
            scip_params = get_scip_params(time_limit = 180)
        elif config.target == 'gap':
            scip_params = get_scip_params(time_limit = 60)
        if config.parameter_opt:
            function_params = params_optimizer(config.dataset, train_seeds, config.cores_list, config.lock_dir, score_function, used_features, function_params, bound_params, scip_params, config)
            final_result["params"] = function_params
        tasks = []
        dataset_dir = None if config.dataset in ['setcover', 'cauctions', 'facilities', 'indset'] else config.dataset_dir
        for seed in seeds:
            tasks.append(('evolve', config.dataset, seed, config.cores_list, config.lock_dir, score_function, used_features, function_params, scip_params, None, None, dataset_dir))
            if config.rpb:
                tasks.append(('relpscost', config.dataset, seed, config.cores_list, config.lock_dir, None, None, None, {**scip_params, f"branching/relpscost/priority": 9999999}, None, None, dataset_dir))
        pool = Pool(processes=config.max_workers)
        async_result = pool.map_async(execute_ecole, tasks)
        results = async_result.get(timeout=config.stage2_timeout)
        # Collect all data
        all_result = {}
        for seed in seeds:
            all_result[seed] = {}

        for result in results:
            seed_index = result["seed"]
            if not result["validity"]:
                return {
                    "validity": False,
                    "combined_score": 0.0,
                    "error_info": result["error_info"]
                } 
            # evolve
            if result["task"] == 'evolve':
                all_result[seed_index]["evolve_time"] = result["time"]
                all_result[seed_index]["evolve_nodes"] = result["nodes"]
                all_result[seed_index]["evolve_gap"] = result["gap"]
            # rpb
            if result["task"] == 'relpscost':
                all_result[seed_index]["rpb_time"] = result["time"]
                all_result[seed_index]["rpb_nodes"] = result["nodes"]
                all_result[seed_index]["rpb_gap"] = result["gap"]

        # Statistic 
        evolve_time_list = [all_result[seed]["evolve_time"] for seed in seeds]
        evolve_nodes_list = [all_result[seed]["evolve_nodes"] for seed in seeds]
        evolve_gap_list = [all_result[seed]["evolve_gap"] for seed in seeds]
        final_result["avg_time"] = numerical_mean_stable(evolve_time_list)
        final_result["max_time"] = max(evolve_time_list)
        final_result["mean_time"] = geometric_mean_stable(evolve_time_list)
        final_result["avg_nodes"] = numerical_mean_stable(evolve_nodes_list)
        final_result["max_nodes"] = max(evolve_nodes_list)
        final_result["mean_nodes"] = geometric_mean_stable(evolve_nodes_list)
        final_result["avg_gap"] = numerical_mean_stable(evolve_gap_list)
        final_result["max_gap"] = max(evolve_gap_list)
        final_result["mean_gap"] = geometric_mean_stable(evolve_gap_list)

        if config.rpb:
            rpb_time_list = [all_result[seed]["rpb_time"] for seed in seeds]
            rpb_nodes_list = [all_result[seed]["rpb_nodes"] for seed in seeds]
            rpb_gap_list = [all_result[seed]["rpb_gap"] for seed in seeds]
            final_result["avg_rpb_time"] = numerical_mean_stable(rpb_time_list)
            final_result["max_rpb_time"] = max(rpb_time_list)
            final_result["mean_rpb_time"] = geometric_mean_stable(rpb_time_list)
            final_result["avg_rpb_nodes"] = numerical_mean_stable(rpb_nodes_list)
            final_result["max_rpb_nodes"] = max(rpb_nodes_list)
            final_result["mean_rpb_nodes"] = geometric_mean_stable(rpb_nodes_list)
            final_result["avg_rpb_gap"] = numerical_mean_stable(rpb_gap_list)
            final_result["max_rpb_gap"] = max(rpb_gap_list)
            final_result["mean_rpb_gap"] = geometric_mean_stable(rpb_gap_list)
        
        # Calculate Combined Scores
        scaling_param = 100 if config.target == 'time' else 10000
        if config.rpb:
            scaling_param = final_result["max_rpb_time"] if config.target  == 'time' else final_result["max_rpb_nodes"]

        if config.target  == 'time':
            final_result["combined_score"] = float(np.log(scaling_param / final_result["avg_time"] + 1))
        elif config.target  == 'nodes':
            final_result["combined_score"] = float(np.log(scaling_param / final_result["avg_nodes"] + 1)) 
        elif config.target == 'gap':
            final_result["combined_score"] = 100 / (1e-6 + float(final_result["avg_gap"]))

        return final_result

    except TimeoutError: 
        logger.debug(f"Evaluation timed out after {config.stage2_timeout} seconds.")
        final_result = {
            "validity": False,
            "combined_score": 0.0,
            "error_info": f"Evaluation timed out after {config.stage2_timeout} seconds."
        }

    except Exception as exc:
        error_details = traceback.format_exc()
        logger.debug(f"Generate an exception {error_details}")

        final_result =  {
            "validity": False,
            "combined_score": 0.0,
            "error_info": f"Exception occurred: {type(exc).__name__}: {exc}"
        }
    finally:
        if pool is not None:
            pool.terminate() 
            pool.join()
        return final_result

def evaluate_stage1(program_path, config):
    pool = None 
    final_result = {
        "stage1_score": 0.0,
        "error_info_stage1": None
    }
    try:
        logger.debug(f"Start fast filter")
        seeds = list(range(config.train_seed, config.train_seed + config.train_num))
        score_function, used_features, function_params, bound_params = load_program(program_path)
        # scip heuristic setting
        if config.target in ['time', 'nodes']:
            scip_params = get_scip_params()
        elif config.target == 'gap':
            scip_params = get_scip_params(time_limit = 180)
        dataset_dir = None if config.dataset in ['setcover', 'cauctions', 'facilities', 'indset'] else config.dataset_dir
        tasks = []
        for seed in seeds:
            tasks.append(('evolve', config.dataset, seed, config.cores_list, config.lock_dir, score_function, used_features, function_params, scip_params, None, None, dataset_dir))
            tasks.append(('relpscost', config.dataset, seed, config.cores_list, config.lock_dir, None, None, None, {**scip_params, f"branching/relpscost/priority": 9999999}, None, None, dataset_dir))

        pool = Pool(processes=config.max_workers)
        async_result = pool.map_async(execute_ecole, tasks)
        results = async_result.get(timeout=config.stage1_timeout)

        # Collect all data
        all_result = {}
        for seed in seeds:
            all_result[seed] = {}

        for result in results:
            seed_index = result["seed"]
            if not result["validity"]:
                return {
                    "validity": False,
                    "staga1_score": 0.0,
                    "error_info_stage1": result["error_info"]
                } 
            if result["task"] == 'evolve':
                all_result[seed_index]["evolve_time"] = result["time"]
                all_result[seed_index]["evolve_gap"] = result["gap"] + 1e-6
            if result["task"] == 'relpscost':
                all_result[seed_index]["rpb_time"] = result["time"]
                all_result[seed_index]["rpb_gap"] = result["gap"] + 1e-6

        # Statistic 
        logger.debug([all_result[seed]["rpb_time"] for seed in seeds])
        logger.debug([all_result[seed]["evolve_time"] for seed in seeds])
        logger.debug([all_result[seed]["rpb_gap"] for seed in seeds])
        logger.debug([all_result[seed]["evolve_gap"] for seed in seeds])
        speedup_list = [all_result[seed]["rpb_time"] / all_result[seed]["evolve_time"] for seed in seeds] if config.target in ['time', 'nodes'] else \
            [all_result[seed]["rpb_gap"] / all_result[seed]["evolve_gap"] for seed in seeds]
        logger.debug(f"Speed up: {speedup_list}")
        # Calculate Stage1 Scores
        final_result["stage1_score"] = min(speedup_list)

        return final_result

    except TimeoutError: 
        logger.debug(f"Evaluation timed out after {config.stage1_timeout} seconds.")
        final_result = {
            "speedup_score": 0.0,
            "error_info": f"Evaluation timed out after {config.stage1_timeout} seconds."
        }

    except Exception as exc:
        error_details = traceback.format_exc()
        logger.debug(f"Generate an exception {error_details}")

        final_result =  {
            "speedup_score": 0.0,
            "error_info": f"Exception occurred: {type(exc).__name__}: {exc}"
        }
    finally:
        if pool is not None:
            pool.terminate() 
            pool.join()
        return final_result

def evaluate_stage2(program_path, config):
    return evaluate(program_path, config)


if __name__ == "__main__":
    # DEBUG Setting
    from openevolve.config import EvaConfig
    logger.handlers = [] 
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    config = EvaConfig.from_yaml('./examples/brancher/evaluator_config.yaml', 'cauctions', '43-64', './tmp/cpu_locks/debug')
    config._config["parameter_opt"] = False
    # config._config["n_calls"] = 5
    # config._config["n_initial"] = 2
    config._config["rpb"] = True
    config._config["valid_num"] = 8
    config._config["stage2_timeout"] = 3000
    program_path = f"examples/brancher/initial_program.py"
    print(program_path)
    # print(evaluate_stage1(program_path, config))
    print(evaluate_stage2(program_path, config))
