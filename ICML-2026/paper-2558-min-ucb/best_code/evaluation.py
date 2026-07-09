import pickle
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from apub import APUB
import time
import multiprocessing as mp
from params_generator import ParametersGenerator
import json
from collections import defaultdict
from utils import sample_from_config
from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_oos(certificate, x_optimal, test_samples, c, n_items, n_machines):
    costs = []
    N = len(test_samples['h'])

    for m in range(N):
        W_test = test_samples['W'][m]
        h_test = test_samples['h'][m]
        T_test = test_samples['T']
        q_test = test_samples['q'][m]

        # second stage 
        sub_model = gp.Model("OOS_Evaluation")
        y = sub_model.addVars(3*n_machines, lb=0)
        sub_model.setObjective(gp.quicksum(q_test[j] * y[j] for j in range(n_machines)), GRB.MINIMIZE)
        # constraint：Wy = h - Tx
        for i in range(n_machines):
            sub_model.addConstr(
                gp.quicksum(W_test[i, j] * y[j] for j in range(3*n_machines)) == -gp.quicksum(
                    T_test[i, j] * x_optimal[j] for j in range(n_items)),name=f"Sub_Constr_{i}")
        sub_model.addConstr(gp.quicksum(y[j] for j in range(n_machines, 2*n_machines)) == h_test[-1], name="Cap_Constr")

        sub_model.setParam('OutputFlag', 0)
        sub_model.optimize()

        temp = 0
        if sub_model.status == GRB.OPTIMAL:
            for i in range(len(x_optimal)):
                temp += c[i] * x_optimal[i]
            total_cost = temp + sub_model.ObjVal
            costs.append(total_cost)
        else:
            costs.append(np.inf) 

    mean_cost = np.mean(costs)
    return {
        'mean_cost': mean_cost,
        'reliability': int(certificate >= mean_cost)
    }


def process_M(args):
    """Worker function to process a single M value"""
    np.random.seed(1234)
    A, b, c, n_items, n_machines, M, xi_samples_list = args
    tt1 = []
    tt2 = []
    cuts = []
    random_numbers = np.random.randint(0, 200, size=30)
    
    for i in range(30):
        apub = APUB(A, b, c=c, n_items=n_items, n_machines=n_machines, model=gp.Model())
        start1 = time.perf_counter()
        apub.extensive_form(xi_samples_list[random_numbers[i]], alpha=0.1, M_bootstrap=M)
        end1 = time.perf_counter()
        
        # L-shape method timing
        start2 = time.perf_counter()
        _, _, _, num_optimal_cuts = apub.solve_two_stage_apub(
            xi_samples_list[random_numbers[i]],
            alpha=0.1,
            M_bootstrap=M
        )
        end2 = time.perf_counter()
        
        tt1.append(end1 - start1)
        tt2.append(end2 - start2)
        cuts.append(num_optimal_cuts)
    
    return {
        'M': M,
        'extensive_form_time': tt1,
        'lshape_time': tt2,
        'cuts': cuts
    }


def evaluate_M_T_performance(A, b, c, M_list, n_items, n_machines, save_path='./results/time.json'):
    result = defaultdict(lambda: defaultdict(dict))

    n_cores = max(1, mp.cpu_count() - 1)

    for data_size in [120,240,480,960]:
        # Load data for this size
        with open(f"./samples/{data_size}.pkl", "rb") as f:
            xi_samples_list = pickle.load(f)['train_samples']
        
        # Prepare arguments for parallel processing
        process_args = [
            (A, b, c, n_items, n_machines, M, xi_samples_list)
            for M in M_list
        ]
        
        # Process M values in parallel
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(process_M, args) for args in process_args]
            
            # Collect results as they complete
            for future in as_completed(futures):
                res = future.result()
                M = res['M']
                result['extensive form'][data_size][M] = res['extensive_form_time']
                result['ours'][data_size][M] = res['lshape_time']
                result['cuts'][data_size][M] = res['cuts']

                print(f"[Data={data_size}, M={M}] EF mean: {np.mean(res['extensive_form_time']):.2f}s, EF dev: {np.std(res['extensive_form_time']):.2f}s | "
                      f"L-shape mean: {np.mean(res['lshape_time']):.2f}s, L-shape dev: {np.std(res['lshape_time']):.2f}s | Iterations mean: {np.mean(res['cuts']):.2f}, Iterations dev: {np.std(res['cuts']):.2f}")
    
    # Convert defaultdict to regular dict for JSON serialization
    result_dict = {k: dict(v) for k, v in result.items()}

    with open(save_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to {save_path}")

   
def worker(alpha, train_samples, test_samples, A, b, c, M, n_items, n_machines):
    apub = APUB(A, b, c=c, n_items=n_items, n_machines=n_machines, model=gp.Model())
    x_optimal, _, certificate,num_optimal_cut = apub.solve_two_stage_apub(train_samples, alpha=alpha, M_bootstrap=M)
    eval_result = evaluate_oos(certificate, x_optimal, test_samples, c=c, n_items=n_items, n_machines=n_machines)
    return alpha, eval_result['mean_cost'], eval_result['reliability'], certificate, num_optimal_cut


def run_experiment(A, b, c, M, n_items, n_machines, data_size, K=30, alpha_list=None, max_workers=None, data_path=None):
    if alpha_list is None:
        alpha_list = [0.05 * i for i in range(1, 21)]
    alpha_list = np.array(alpha_list)

    if data_path is not None:
        with open(f"{data_path}", "rb") as f:
            samples = pickle.load(f)
            train_samples_list = samples['train_samples']
            test_samples = samples['test_samples']
    else:
        pg = ParametersGenerator()
        test_samples = pg.generate_parameters(sample_from_config(cfg_or_path="config.yaml", train=False))

    results = {alpha: {'costs': [], 'reliabilities': []} for alpha in alpha_list}
    
    for trial in range(K):
        if data_path is not None:
            train_samples = train_samples_list[trial]
        else:
            pg = ParametersGenerator()
            train_samples = pg.generate_parameters(sample_from_config(cfg_or_path="config.yaml", train=True))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(worker, alpha, train_samples, test_samples, A, b, c, M, n_items, n_machines): alpha
                for alpha in alpha_list
            }
            for future in as_completed(futures):
                alpha, cost, reliability, certificate, num_optimal_cut = future.result()
                results[alpha]['costs'].append(cost)
                results[alpha]['reliabilities'].append(reliability)
                print(f'epoch {trial+1} of {K}, alpha={alpha:.2f}, '
                      f'cost: {np.mean(results[alpha]["costs"]):.2f}, '
                      f'reliability: {np.mean(results[alpha]["reliabilities"]):.2f}, '
                      f'certificate: {certificate:.2f}, '
                      f'num_optimal_cut: {num_optimal_cut:.2f}')

    serializable_results = {
        str(alpha): {
            'costs': [float(c) for c in vals['costs']],
            'reliabilities': [float(r) for r in vals['reliabilities']]
            #'num_optimal_cuts': [float(n) for n in vals['num_optimal_cuts']]
        }
        for alpha, vals in results.items()
    }

    save_path = f"apub_results_ee{data_size}.json"
    with open(save_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    print(f"\n Results saved to {save_path}")
    return results


def process_optimal_M(args):
    """Worker function to process a single random sample for all M values"""
    A, b, c, n_items, n_machines, M_list, xi_samples, sample_idx = args
    results = []
    
    for M in M_list:
        apub = APUB(A, b, c=c, n_items=n_items, n_machines=n_machines, model=gp.Model())
        _, _, optimal_value, _ = apub.solve_two_stage_apub(
            xi_samples,
            alpha=0.2,
            M_bootstrap=M,
        )
        results.append(optimal_value)
        print(f"Sample {sample_idx}, M={M}: optimal_value={optimal_value:.2f}")
    
    return results


def find_optimal_M(A, b, c, M_list, n_items, n_machines, save_path='./results/m.json'):
    """Find optimal M value using parallel processing for multiple random samples"""
    np.random.seed(999)
    # Generate random sample indices
    random_numbers = np.random.randint(0, 200, size=10)
    print(f"Processing samples: {random_numbers}")
    
    # Load samples
    with open(f"./samples/120_4000.pkl", "rb") as f:
        xi_samples_list = pickle.load(f)['train_samples']
    
    # Prepare arguments for parallel processing
    process_args = [
        (A, b, c, n_items, n_machines, M_list, xi_samples_list[idx], idx)
        for idx in random_numbers
    ]
    
    # Use multiprocessing to process samples in parallel
    n_cores = max(1, mp.cpu_count() - 1)
    results = []
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_optimal_M, args) for args in process_args]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Error processing sample: {str(e)}")
   
    # Save results to JSON
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {save_path}")
    