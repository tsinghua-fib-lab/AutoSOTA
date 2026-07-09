from collections import defaultdict
from saa import SAA
import time 
import numpy as np
import gurobipy as gp
from utils import load_config
import pickle,json 
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_M(args):
    """Worker function to process a single M value"""
    np.random.seed(1234)
    c, n_items, n_machines, xi_samples_list = args
    tt1 = []
    cuts = []
    random_numbers = np.random.randint(0, 200, size=30)
    
    for i in range(30):
        saa = SAA(model=gp.Model(), c=c, n_items=n_items, n_machines=n_machines)
        start1 = time.perf_counter()
        _,_,it=saa.solve_nf(xi_samples_list[random_numbers[i]], max_iter=30, tol=1e-4)
        end1 = time.perf_counter()

        tt1.append(end1 - start1)
        cuts.append(it)
    
    return {
        'N': len(xi_samples_list[0]),
        'time': tt1,
        'cuts': cuts
    }


def evaluate_M_T_performance(c, n_items, n_machines, save_path='./results/saa_time2.json'):
    result = defaultdict(lambda: defaultdict(dict))

    n_cores = max(1, mp.cpu_count() - 1)
    xi_samples_listl = []
    for data_size in [120, 240, 480, 960]:
        # Load data for this size
        with open(f"./samples/{data_size}.pkl", "rb") as f:
            xi_samples_list = pickle.load(f)['train_samples']
            xi_samples_listl.append(xi_samples_list)

    # Prepare arguments for parallel processing
    process_args = [
        (c, n_items, n_machines, xi_samples_list)
        for xi_samples_list in xi_samples_listl
    ]
    
    # Process M values in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_M, args) for args in process_args]
        
        # Collect results as they complete
        for future in as_completed(futures):
            res = future.result()
            result['extensive form'][res['N']] = res['time']
            result['cuts'][res['N']] = res['cuts']

            print(f"[Data={res['N']}] EF mean: {np.mean(res['time']):.2f}s, EF dev: {np.std(res['time']):.2f}s | "
                    f"Iterations mean: {np.mean(res['cuts']):.2f}, Iterations dev: {np.std(res['cuts']):.2f}")
    
    # Convert defaultdict to regular dict for JSON serialization
    result_dict = {k: dict(v) for k, v in result.items()}

    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    cfg_path = "config.yaml"
    np.random.seed(42) 
    full_cfg = load_config(cfg_path)
    rg_cfg = full_cfg.get("random_generator", full_cfg)
    I = int(rg_cfg["I"]) 
    J = int(rg_cfg["J"])
    c = list(rg_cfg["c"])
    evaluate_M_T_performance(c=c, n_items=I, n_machines=J, save_path='./results/saa_time.json')
