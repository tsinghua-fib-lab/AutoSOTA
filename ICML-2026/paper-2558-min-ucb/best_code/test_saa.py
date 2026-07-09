import gurobipy as gp
from apub import APUB
from saa import SAA
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from evaluation import evaluate_oos
from utils import load_config, sample_from_config
from params_generator import ParametersGenerator


def run_trial_with_data(args):
    r, cfg_path, J, I, c, M, A, b, train_samples, test_samples = args
    # Initialize solvers with optimized settings
    saa_model = initialize_solver(c, I, J)
    apub_model = initialize_solver(c, I, J)
    saa = SAA(c=c, n_items=I, n_machines=J, model=saa_model)
    apub_solver = APUB(A, b, c=c, n_items=I, n_machines=J, model=apub_model)
    x_opt, _, _ = saa.solve_nf(train_samples)
    val = saa.saa_oos(x_opt, test_samples)
    x_optimal1, _, certificate1,_ = apub_solver.solve_two_stage_apub(train_samples, alpha=1, M_bootstrap=M)
    apub_solver = APUB(A, b, c=c, n_items=I, n_machines=J, model=initialize_solver(c, I, J))
    x_optimal9, _, certificate9,_ = apub_solver.solve_two_stage_apub(train_samples, alpha=0.05, M_bootstrap=M)
    eval_result1 = evaluate_oos(certificate1, x_optimal1, test_samples, c=c, n_items=I, n_machines=J)
    eval_result9 = evaluate_oos(certificate9, x_optimal9, test_samples, c=c, n_items=I, n_machines=J)
    return {
        'train_samples': train_samples,
        'test_samples': test_samples,
        'saa_cost': val,
        'apub_cost': eval_result1['mean_cost'],
        'apub_cost_0.05': eval_result9['mean_cost'],
        'apub_reliability1': eval_result1['reliability'],
        'apub_reliability9': eval_result9['reliability'],
        'trial': r
    }

def initialize_solver(c, I, J):
    """Initialize solvers with optimized settings"""
    model = gp.Model()
    # Optimize solver settings for faster solving
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('Threads', 1)  # Single thread per process for better parallel performance
    model.setParam('Method', 1)  # Dual simplex - usually faster for this type of problem
    return model


if __name__ == "__main__":
    np.random.seed(42)
    use_saved_data = False
    cfg_path = "./config/config1.yaml"
    full_cfg = load_config(cfg_path)
    rg_cfg = full_cfg.get("random_generator", full_cfg)
    J = int(rg_cfg["J"]) 
    I = int(rg_cfg["I"]) 
    c = list(rg_cfg["c"]) 
    M = int(rg_cfg["M"]) 
    n = int(rg_cfg["train_n"])
    epochs = int(rg_cfg["epochs"])
    
    b = np.zeros(J)
    A = np.zeros((J,I))
    obj_values = []
    apub_costs = []
    apub_costs_005 = []
    apub_reliabilities = []
    if use_saved_data:
        with open("samples/120.pkl", "rb") as f:
            sample_data = pickle.load(f)
        train_samples_list = sample_data['train_samples']
        test_samples = sample_data['test_samples']
    else:
        train_samples_list = []
        pg = ParametersGenerator()
        test_samples = pg.generate_parameters(samples=sample_from_config(cfg_path, train=False))
        for _ in range(epochs):
            train_samples = pg.generate_parameters(samples=sample_from_config(cfg_path, train=True))
            train_samples_list.append(train_samples)
        sample_data = {
            'train_samples': train_samples_list,
            'test_samples': test_samples
        }
        os.makedirs("samples", exist_ok=True)
        with open("samples/samples2.pkl", "wb") as f:
            pickle.dump(sample_data, f)
            print("Samples saved to samples/samples2.pkl")

    # Use process pool for parallel execution
    n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    apub_reliabilities1 = []
    apub_reliabilities9 = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        args_list = [
            (r, cfg_path, J, I, c, M, A, b, train_samples_list[r], test_samples)
            for r in range(epochs)
        ]
        futures = [executor.submit(run_trial_with_data, args) for args in args_list]
        for future in as_completed(futures):
            result = future.result()
            obj_values.append(result['saa_cost'])
            apub_costs.append(result['apub_cost'])
            apub_costs_005.append(result['apub_cost_0.05'])
            apub_reliabilities1.append(result['apub_reliability1'])
            apub_reliabilities9.append(result['apub_reliability9'])
            r = result['trial']
            print(f'SAA trial {r+1}/{epochs}: cost={result["saa_cost"]:.2f}')
            print(f'APUB trial {r+1}/{epochs}: cost={result["apub_cost"]:.2f}, reliability1={result["apub_reliability1"]:.3f}, reliability01={result["apub_reliability9"]:.3f}')

    # Create side-by-side boxplot with reliability information
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate APUB costs for alpha=0.05
    apub_costs_005 = [eval_result9['mean_cost'] for eval_result9 in [evaluate_oos(certificate9, x_optimal9, test_samples, c=c, n_items=I, n_machines=J) for _, _, certificate9, x_optimal9 in [APUB(A, b, c=c, n_items=I, n_machines=J, model=initialize_solver(c, I, J)).solve_two_stage_apub(train_samples, alpha=0.05, M_bootstrap=M) for train_samples in train_samples_list]]]
    
    #box_data = [obj_values, apub_costs, apub_costs_005]
    box_data = [obj_values]
    box_labels = ['SAA', 'APUB (α=1)', 'APUB (α=0.05)']
    box_colors = ['lightblue', 'lightgreen', 'lightpink']
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    ax1.set_ylabel("Cost Value")
    ax1.set_title("SAA vs APUB Performance Comparison")
    ax1.grid(True, linestyle='--', alpha=0.6)

    saa_reliability = "N/A"  # SAA doesn't have reliability concept
    apub_reliability1 = f"{np.mean(apub_reliabilities1):.3f}"
    apub_reliability9 = f"{np.mean(apub_reliabilities9):.3f}"
    ax1.text(0.02, 0.98, f'SAA Reliability: {saa_reliability}', 
              transform=ax1.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.text(0.02, 0.92, f'APUB Reliability (α=1): {apub_reliability1}', 
              transform=ax1.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax1.text(0.02, 0.86, f'APUB Reliability (α=0.05): {apub_reliability9}', 
              transform=ax1.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))

    #Reliability distribution plots for alpha=1, 0.9
    ax2.hist(apub_reliabilities1, bins=15, alpha=0.7, color='lightgreen', edgecolor='green')
    ax2.axvline(np.mean(apub_reliabilities1), color='red', linestyle='--', linewidth=2, 
                 label=f'Mean: {np.mean(apub_reliabilities1):.3f}')
    ax2.set_xlabel("Reliability (α=1)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("APUB Reliability Distribution (α=1)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.hist(apub_reliabilities9, bins=15, alpha=0.7, color='lightblue', edgecolor='navy')
    ax3.axvline(np.mean(apub_reliabilities9), color='red', linestyle='--', linewidth=2, 
                 label=f'Mean: {np.mean(apub_reliabilities9):.3f}')
    ax3.set_xlabel("Reliability (α=0.05)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("APUB Reliability Distribution (α=0.05)")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\nSAA Summary:")
    print(f"Mean cost: {np.mean(obj_values):.2f}")
    print(f"Std cost: {np.std(obj_values):.2f}")

    print(f"\nAPUB Summary (α=1):")
    print(f"Mean out-of-sample cost: {np.mean(apub_costs):.2f}")
    print(f"Std out-of-sample cost: {np.std(apub_costs):.2f}")
    print(f"Mean reliability: {np.mean(apub_reliabilities1):.3f}")
    print(f"Std reliability: {np.std(apub_reliabilities1):.3f}")

    print(f"\nAPUB Summary (α=0.05):")
    print(f"Mean out-of-sample cost: {np.mean(apub_costs_005):.2f}")
    print(f"Std out-of-sample cost: {np.std(apub_costs_005):.2f}")
    print(f"Mean reliability: {np.mean(apub_reliabilities9):.3f}")
    print(f"Std reliability: {np.std(apub_reliabilities9):.3f}")

