import numpy as np
import torch
from src.datasets import DATASET_REGISTRY
from src.samplers import SAMPLER_REGISTRY
from src.gp_model import GPModel
from src.evaluator import evaluate_metrics
from tqdm import tqdm
import os
import gpytorch
from src.validation_utils import ValidationLogger 

SAMPLERS = SAMPLER_REGISTRY
DATASETS = DATASET_REGISTRY

try:
    from scipy.stats import qmc
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: Scipy not installed. LHS init_strategy will not be available.")


def run_single_seed(dataset_name, sampler_name, seed, N_total, B, N_init, beta, num_threads, 
                    n_pool, test_ratio, init_strategy="random", 
                    n_candidates=2000, gpu_batch_size=128, t_grid_size=101,
                    validate_theory=False,lr_x=0.1,lr_t=0.1,): 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)            # In case dataset.observe uses torch internally
    torch.cuda.manual_seed_all(seed)
    
    # 1. Load Dataset
    dataset = DATASETS[dataset_name](
        seed=seed, 
        n_pool=n_pool, 
        test_ratio=test_ratio
    )
    dataset._plot_samples(5)
    X_pool = dataset.X_pool
    X_test = dataset.X_test
    n_pool = len(X_pool)
    available_indices = np.arange(n_pool)
    
    # 2. Initialize Model and Sampler
    model = GPModel(dim_x=dataset.dim_x, device=device, normalize_x=True, normalize_y=False)
    sampler_args = {
        'rng': rng, 'beta': beta, 'num_threads': num_threads,
        'n_candidates': n_candidates, 'gpu_batch_size': gpu_batch_size,
        't_grid_size': t_grid_size,
    }
    if sampler_name=='PG':
        sampler_args['lr_x'] = lr_x
        sampler_args['lr_t'] = lr_t
    
    sampler = SAMPLERS[sampler_name](**sampler_args)
    
    # Initialize validation logger
    validation_logger = None
    if validate_theory and sampler_name.startswith("Theoretical"):
        # Ensure log directory exists
        log_dir = f"results/validation_logs/{dataset_name}/{sampler_name}/"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"seed_{seed}.jsonl")
        
        validation_logger = ValidationLogger(log_path)
    elif validate_theory:
        print(f"Note: --validate_theory is set, but sampler {sampler_name} is not a 'Theoretical' sampler. Skipping validation.")
    
    results = [] 
    
    # 3. Initial Sampling (LHS or Random)
    N_init = min(N_init, len(available_indices))
    
    if init_strategy.lower() == 'lhs':
        if not SCIPY_AVAILABLE:
            raise ImportError("LHS initialization requires 'scipy' to be installed.")
        
        pbar_desc = f"Seed {seed} | LHS Init"
        print(f"{pbar_desc} (N_init={N_init})")
        
        dim_x = dataset.dim_x
        x_min = X_pool[available_indices].min(axis=0)
        x_max = X_pool[available_indices].max(axis=0)
        
        dims = dim_x + 1
        lower_bounds = np.append(x_min, 0.0)
        upper_bounds = np.append(x_max, 1.0)
        upper_bounds[lower_bounds == upper_bounds] += 1e-6 

        lhs_sampler = qmc.LatinHypercube(d=dims, seed=rng)
        samples_unit_cube = lhs_sampler.random(n=N_init)
        samples_scaled = qmc.scale(samples_unit_cube, lower_bounds, upper_bounds)
        
        lhs_x = samples_scaled[:, :dim_x]
        init_t_list = list(samples_scaled[:, dim_x])

        distance_matrix = cdist(lhs_x, X_pool[available_indices])
        sorted_pool_indices_local = np.argsort(distance_matrix, axis=1)
        
        init_indices_local = []
        init_t_final = []
        used_local_indices = set()
        
        for i in range(N_init):
            assigned = False
            for local_idx in sorted_pool_indices_local[i]:
                if local_idx not in used_local_indices:
                    init_indices_local.append(local_idx)
                    init_t_final.append(init_t_list[i])
                    used_local_indices.add(local_idx)
                    assigned = True
                    break
            if not assigned:
                print(f"Warning: Could not find unique index for LHS point {i}")
                pass 
        
        init_indices = available_indices[init_indices_local]
        init_t = np.array(init_t_final)
        
        if len(init_indices) < N_init:
             print(f"Warning: LHS could only assign {len(init_indices)} unique points. Requested {N_init}.")
             N_init = len(init_indices) 

    elif init_strategy.lower() == 'random':
        init_sampler_args = sampler_args.copy()
        init_sampler_args['rng'] = rng 
        init_sampler = SAMPLERS["RAND"](**init_sampler_args)
        init_indices, init_t = init_sampler.select_batch(
            None, X_pool, available_indices, N_init, train_y=None
        )
    
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")

    init_x = X_pool[init_indices]
    init_y = dataset.observe(init_x, init_t)
    
    available_indices = np.setdiff1d(available_indices, init_indices)
    
    # --- Optimization ---
    # Maintain a NumPy copy of train_y for the EI Sampler
    current_train_y_np = init_y 
    
    # Combine (x,t) for the first update
    init_train_x_t = np.hstack([init_x, init_t.reshape(-1, 1)])
    
    # Use model.update() instead of model.fit()
    with gpytorch.settings.cholesky_jitter(1e-1): 
        model.update(init_train_x_t, init_y)

        # ================= [Debug/Validation START] =================
        
        # 1. Check training data scale (Y Scale)
        train_y = model.train_y
        y_min, y_max, y_mean = train_y.min(), train_y.max(), train_y.mean()
        if hasattr(train_y, 'item'): # Handle tensor/numpy difference
            print(f"TRAIN Y Stats: Min={y_min.item():.4f}, Max={y_max.item():.4f}, Mean={y_mean.item():.4f}")
        else:
            print(f"TRAIN Y Stats: Min={y_min:.4f}, Max={y_max:.4f}, Mean={y_mean:.4f}")

        # 2. Check ground truth scale (Test Truth Scale)
        sample_test_x = X_test[:5]
        t_star_true = dataset.get_t_star(sample_test_x)
        f_star_true = dataset.get_f(sample_test_x, t_star_true)
        print(f"TRUE F (First 5): {f_star_true.flatten()}")
        print(f"TRUE T* (First 5): {t_star_true.flatten()}")

        # 3. Check model raw prediction values
        if isinstance(model.train_x, torch.Tensor): # GPU/Torch Check
            inp_x = torch.tensor(sample_test_x, dtype=model.dtype, device=model.device)
            inp_t = torch.tensor(t_star_true, dtype=model.dtype, device=model.device).reshape(-1, 1)
            inp = torch.cat([inp_x, inp_t], dim=1)
            with torch.no_grad():
                pred_raw = model.model(inp) # Direct call to internal model
                pred_mean = pred_raw.mean.cpu().numpy()
        else: # CPU/Numpy Check
            inp = np.hstack([sample_test_x, t_star_true.reshape(-1, 1)])
            pred_mean, _ = model.predict(inp) 
            
        print(f"PRED F (First 5): {pred_mean.flatten()}")
        
        # 4. Simple residual check
        diff = pred_mean.flatten() - f_star_true.flatten()
        print(f"DIFF (Pred - True): {diff}")
        print(f"Calculated MSE (First 5): {np.mean(diff**2):.4f}")
        # ================= [Debug/Validation END] =================

        metrics = evaluate_metrics(model, dataset, X_test)
    
    N = len(current_train_y_np) 
    metrics['N'] = N
    results.append(metrics)

    # 4. Active Learning Loop
    n_rounds = (N_total - N) // B
    
    pbar = tqdm(range(n_rounds), 
                desc=f"Seed {seed} | {sampler_name} | GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')}", 
                leave=False)
    for round_num in pbar: 
        
        # Cholesky Jitter set here for model.predict and model.fit
        with gpytorch.settings.cholesky_jitter(1e-2): 
            
            # Prepare validation context (always pass round number for adaptive strategies)
            validation_context = {'round': round_num, 'N': N}
            if validation_logger:
                validation_context['logger'] = validation_logger
                validation_context['dataset'] = dataset
            
            # Pass validation_context to sampler
            selected_indices, assigned_t = sampler.select_batch(
                model, X_pool, available_indices, B, 
                train_y=current_train_y_np,
                validation_context=validation_context 
            )
            
            if len(selected_indices) == 0:
                pbar.set_description("Pool exhausted.")
                break
                
            batch_x = X_pool[selected_indices]
            batch_y = dataset.observe(batch_x, assigned_t)
            
            available_indices = np.setdiff1d(available_indices, selected_indices)
            batch_train_x_t = np.hstack([batch_x, assigned_t.reshape(-1, 1)])
            
            # --- Optimization ---
            # Call model.update which handles concatenation and training on GPU
            model.update(batch_train_x_t, batch_y)
            
            # Update NumPy copy for EI Sampler
            current_train_y_np = np.concatenate([current_train_y_np, batch_y])

            metrics = evaluate_metrics(model, dataset, X_test)
        
        N = len(current_train_y_np)
        metrics['N'] = N
        results.append(metrics)
        pbar.set_postfix(E2=f"{metrics['E2_error']:.2e}", 
                         SubOpt=f"{metrics['policy_suboptimality']:.2e}")
    
    pbar.close()
    
    # Ensure logs are flushed
    if validation_logger:
        validation_logger.close()
        
    return results