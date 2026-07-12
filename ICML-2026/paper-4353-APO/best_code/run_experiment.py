import argparse
import pandas as pd
import os
import torch
import concurrent.futures
from tqdm import tqdm
import itertools
import time
import sys 
import traceback 
import yaml
import signal
import random
import numpy as np

def set_global_seed(base_seed: int):
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("="*60)
    print("Warning: `psutil` library not installed.")
    print("Child processes may not be fully cleaned up after Ctrl+C.")
    print("Please run: pip install psutil")
    print("="*60)

from src.experiment import run_single_seed
from src.samplers import SAMPLER_REGISTRY


def _terminate_child_processes():
    """
    Called in the main process to gracefully terminate all child processes.
    """
    if not PSUTIL_AVAILABLE:
        return
    
    parent = psutil.Process()
    children = parent.children(recursive=True)
    if not children:
        return
    
    print("\nTerminating child processes...")
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue
    
    gone, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        print(f"Child process PID {child.pid} did not respond, forcing kill.")
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def run_worker(job_args):
    """
    Independent worker function called by ProcessPoolExecutor.
    Sets up the GPU environment and runs a single experiment.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    seed, sampler_name, current_cli_args, device_id, num_threads, group_name = job_args
    
    if device_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    worker_seed = job_args[0]  # seed
    set_global_seed(worker_seed)

    try:
        result_list = run_single_seed(
            dataset_name=current_cli_args['dataset'],
            sampler_name=sampler_name,
            seed=seed,
            N_total=current_cli_args['n_total'],
            B=current_cli_args['batch_size'],
            N_init=current_cli_args['n_init'],
            beta=current_cli_args['beta'],
            num_threads=num_threads,
            n_pool=current_cli_args['n_pool'],
            test_ratio=current_cli_args['test_ratio'],
            n_candidates=current_cli_args['n_candidates'],
            gpu_batch_size=current_cli_args['gpu_batch_size'],
            t_grid_size=current_cli_args['t_grid_size'],
            init_strategy=current_cli_args['init_strategy'],
            validate_theory=current_cli_args['validate_theory'],
            lr_x = current_cli_args['sampler_lr_x'],
            lr_t= current_cli_args['sampler_lr_t']
        )
        
        for res in result_list:
            res['seed'] = seed
            res['sampler'] = sampler_name
            res['dataset'] = current_cli_args['dataset']
            res['setting'] = current_cli_args['setting_name']
            res['group'] = group_name
        
        return result_list
    except Exception as e:
        print(f"\n!!! WORKER FAILED (PID {os.getpid()}) on job ({group_name}/{sampler_name}, seed {seed}) !!!")
        print(f"Error: {e}")
        traceback.print_exc() 
        return [] 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SemiSynthNews')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--beta', type=float, default=1.96)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_threads', type=int, default=8)
    parser.add_argument('--n_pool', type=int, default=2000)
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--n_candidates', type=int, default=500)
    parser.add_argument('--gpu_batch_size', type=int, default=8)
    parser.add_argument('--t_grid_size', type=int, default=101)
    parser.add_argument('--config_file', type=str, default='experiments.yml')
    parser.add_argument('--group', type=str, default='Group_Init_Ablation',
                        help="Optional: Run only a specific group from the config file (e.g., Group_BatchSize_Ablation)")
    parser.add_argument('--validate_theory', action='store_true',
                        help="Enable theory validation logging (slows down execution).")
    
    parser.add_argument('--sampler_lr_x', type=float, default=0.1)
    parser.add_argument('--sampler_lr_t', type=float, default=0.1)
    
    args = parser.parse_args()
    cli_args_dict = vars(args) 

    samplers = list(SAMPLER_REGISTRY.keys())
    samplers += ['PG']

    print(f"Running with samplers: {samplers}")

    if torch.cuda.is_available():
        # Manually specify the GPU IDs to use
        device_ids = [0, 1, 2, 3] 
        num_gpus = len(device_ids)
        print(f"Using specified GPUs: {device_ids}")
    else:
        num_gpus = 0
        device_ids = ["cpu"]
        print("No GPUs found. Running on CPU.")
        
    if args.num_workers is None:
        max_workers = num_gpus if num_gpus > 0 else 1
    else:
        max_workers = args.num_workers
    print(f"Starting ProcessPoolExecutor with {max_workers} worker(s).")
    
    base_n_pool = args.n_pool
    print(f"Base N_Pool for ratio calculation: {base_n_pool}")
    print(f"Loading experiment settings from: {args.config_file}")

    try:
        with open(args.config_file, 'r') as f:
            ALL_GROUPS = yaml.safe_load(f)
        if not ALL_GROUPS or not isinstance(ALL_GROUPS, dict):
            raise ValueError("Config file is empty or not a dictionary (key: group_name, value: list_of_settings).")
    except Exception as e:
        print(f"Error parsing YAML file {args.config_file}: {e}")
        sys.exit(1)

    if args.group:
        if args.group not in ALL_GROUPS:
            print(f"Error: Group '{args.group}' not found in {args.config_file}.")
            print(f"Available groups: {list(ALL_GROUPS.keys())}")
            sys.exit(1)
        GROUPS_TO_RUN = {args.group: ALL_GROUPS[args.group]}
        print(f"--- Running ONLY selected group: {args.group} ---")
    else:
        GROUPS_TO_RUN = ALL_GROUPS
        print(f"--- Running all {len(GROUPS_TO_RUN)} groups from config file ---")
        
    all_results = []
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    start_time = time.time()
    interrupted = False
    
    def build_job_list(setting_ratios, group_name):
        current_cli_args = cli_args_dict.copy()
        current_cli_args.update(setting_ratios)
        current_cli_args['n_total'] = int(setting_ratios['total_budget_ratio'] * base_n_pool)
        current_cli_args['n_init'] = int(setting_ratios['init_ratio'] * base_n_pool)
        current_cli_args['n_init'] = max(current_cli_args.get('batch_size', 1), current_cli_args['n_init'])
        if current_cli_args['n_total'] <= current_cli_args['n_init']:
            current_cli_args['n_total'] = current_cli_args['n_init'] + current_cli_args.get('batch_size', 1)
        return current_cli_args

    torch.multiprocessing.set_start_method('spawn', force=True)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for group_name, settings_list in GROUPS_TO_RUN.items():
                print(f"\n{'#'*80}\n### Starting Experiment Group: {group_name} ###\n{'#'*80}\n")
                
                for setting_ratios in settings_list:
                    current_cli_args = build_job_list(setting_ratios, group_name)
                    setting_name = setting_ratios["setting_name"]

                    print(f"\n{'='*80}")
                    print(f"--- Starting experiment setting: {setting_name} (Group: {group_name}) ---")
                    print(f"--- Parameters: N_total={current_cli_args['n_total']}, "
                          f"N_init={current_cli_args['n_init']}, B={current_cli_args.get('batch_size', 'N/A')}")
                    
                    if current_cli_args['validate_theory']:
                        print(f"--- !!! Theory validation enabled (execution will slow down) !!! ---")
                    print(f"{'='*80}\n")

                    jobs = []
                    job_device_cycler = itertools.cycle(device_ids)
                    for seed in range(args.n_seeds):
                        for sampler_name in samplers:
                            device_id = next(job_device_cycler)
                            job_args = (seed, sampler_name, current_cli_args, device_id, args.num_threads, group_name)
                            jobs.append(job_args)
                    
                    print(f"Setting '{setting_name}' has total {len(jobs)} jobs to run.")
                    
                    output_dir_group_setting_dataset = os.path.join("results", group_name, setting_name, args.dataset)
                    os.makedirs(output_dir_group_setting_dataset, exist_ok=True)
                    print(f"Live summary CSVs will be saved to: {output_dir_group_setting_dataset}")

                    futures = {executor.submit(run_worker, job): job for job in jobs}
                    
                    pbar_desc = f"Group: {group_name} | Setting: {setting_name}"
                    pbar = tqdm(concurrent.futures.as_completed(futures), total=len(jobs), desc=pbar_desc)
                    for future in pbar:
                        job_args = futures[future]
                        sampler_name, seed = job_args[1], job_args[0]
                        pbar.set_description(f"{pbar_desc} | {sampler_name} (Seed {seed})")
                        
                        try:
                            result_list = future.result()
                        except Exception as e:
                            if not interrupted:
                                print(f"\n!!! JOB FAILED (future level): ({group_name}/{setting_name}/{sampler_name}, Seed {seed}) error: {e} !!!")
                            continue
                        if not result_list:
                            print(f"\n--- Job ({group_name}/{setting_name}/{sampler_name}, Seed {seed}) returned no results. ---")
                            continue

                        all_results.extend(result_list)
                        
                        try:
                            sampler_setting_results = [
                                r for r in all_results
                                if r['sampler'] == sampler_name and
                                   r['setting'] == setting_name and
                                   r['group'] == group_name
                            ]
                            if not sampler_setting_results:
                                continue
                            sampler_df = pd.DataFrame(sampler_setting_results)
                            grouped = sampler_df.groupby('N')
                            mean_df = grouped.mean(numeric_only=True)
                            std_df = grouped.std(numeric_only=True)
                            summary_df = pd.concat([mean_df, std_df], keys=['mean', 'std'], axis=1)
                            csv_path = os.path.join(output_dir_group_setting_dataset, f"{sampler_name}_{run_timestamp}.csv")
                            summary_df.to_csv(csv_path)
                        except Exception as e:
                            if not interrupted:
                                print(f"!!! Error updating summary CSV for {sampler_name} in {setting_name}: {e} !!!")

                    print(f"\n--- Setting {setting_name} (Group {group_name}) Finished ---")
                    setting_df = pd.DataFrame([
                        r for r in all_results
                        if r['setting'] == setting_name and r['group'] == group_name
                    ])
                    if not setting_df.empty:
                        output_path_pkl = os.path.join(
                            output_dir_group_setting_dataset,
                            f"results_{args.dataset}_beta{args.beta}.pkl"
                        )
                        setting_df.to_pickle(output_path_pkl)
                        print(f"Setting PKL results saved to {output_path_pkl}")
                
                if interrupted:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n!!! KeyboardInterrupt (Ctrl+C) received !!!")
        _terminate_child_processes()
        print("Main process exiting.")
        sys.exit(1)

    if interrupted:
        sys.exit(1)

    end_time = time.time()
    print(f"\n--- All experiments finished in {end_time - start_time:.2f} seconds ---")
    
    print("\n--- Final Aggregated Means (from all settings) ---")
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        print(results_df.groupby(['group', 'setting', 'sampler', 'N']).mean(numeric_only=True))
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()