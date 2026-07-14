import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from bias_clease import BinnedBiasPotential
from tqdm import tqdm
from utils_ising import ising2d_ham, ising2d_mag


def mcmc_step(spins, bias_pot, beta, J, h, mod_factor, device):
    """
    Perform one Metropolis-Hastings sweep with Bias Potential updates.
    
    Args:
        spins (Tensor): [B, L*L] current spin configurations {-1, 1}
        bias_pot (BinnedBiasPotential): The bias potential object
        beta (float or Tensor): Inverse temperature
        J (float): Coupling constant
        h (float): External field
        mod_factor (float): The update height (dE) for this step
        device (str): Device
    
    Returns:
        spins (Tensor): Updated spins
        acceptance_rate (float): Fraction of accepted moves
    """
    B, N = spins.shape
    L = int(np.sqrt(N))
    
    # 1. Propose flip
    # Pick random indices for each walker
    # For a full sweep equivalent, we usually do N attempts or vectorized batch updates multiple times.
    # Here we do ONE vectorized update per "step", so effectively 1/N-th of a sweep per walker?
    # No, usually "sweep" means N attempts.
    # To keep it efficient in PyTorch, we can accept/reject a batch of flips.
    
    indices = torch.randint(0, N, (B,), device=device)
    
    # Calculate dH (Standard Ising Energy Change)
    # dH = 2 * s_i * local_field
    # We need to reshape to find neighbors
    spins_grid = spins.view(B, L, L)
    
    rows = indices // L
    cols = indices % L
    
    # Gather spins at indices
    s_i = spins.gather(1, indices.view(-1, 1)).squeeze() # [B]
    
    # Calculate neighbors summation
    # We use gathered indices or roll. Since indices are random per batch, roll is hard.
    # Let's use simple logic: top, bottom, left, right neighbors with PBC
    
    row_up = (rows - 1) % L
    row_down = (rows + 1) % L
    col_left = (cols - 1) % L
    col_right = (cols + 1) % L
    
    # Helper to gather neighbor spins
    # Convert back to flat indices
    idx_up = row_up * L + cols
    idx_down = row_down * L + cols
    idx_left = rows * L + col_left
    idx_right = rows * L + col_right
    
    s_up = spins.gather(1, idx_up.view(-1, 1)).squeeze()
    s_down = spins.gather(1, idx_down.view(-1, 1)).squeeze()
    s_left = spins.gather(1, idx_left.view(-1, 1)).squeeze()
    s_right = spins.gather(1, idx_right.view(-1, 1)).squeeze()
    
    sum_neighbors = s_up + s_down + s_left + s_right
    
    # Energy change from Hamiltonian: H_new - H_old
    # H = -J * sum(si sj). If si flips -> si_new = -si_old.
    # dH = H_new - H_old = (-J * (-si) * neighbors) - (-J * si * neighbors)
    #    = J * si * neighbors + J * si * neighbors = 2 * J * si * neighbors
    # Also field term: -h * si. dH = (-h * (-si)) - (-h * si) = 2 * h * si
    dH = 2.0 * J * s_i * sum_neighbors + 2.0 * h * s_i
    
    # 2. Calculate dV_bias (Bias Potential Change)
    # V(s). We need to calculate CV before and after flip.
    # CV is magnetization M = mean(spins).
    # dM = (sum(s_new) - sum(s_old)) / N = (-s_i - s_i) / N = -2*s_i / N
    
    # Evaluate Bias at OLD state (s)
    # We can optimize this: we don't need to re-evaluate the whole grid?
    # Actually, BinnedBiasPotential.evaluate takes [B] CVs. 
    # Let's compute CVs.
    mag_old = ising2d_mag(spins)
    bias_old = bias_pot.evaluate(mag_old)
    
    # New CV
    mag_new = mag_old - (2.0 * s_i / N)
    bias_new = bias_pot.evaluate(mag_new)
    
    dV = bias_new - bias_old
    
    # Total Energy Change
    dE_total = dH + dV
    
    # 3. Acceptance Rule
    # P_acc = min(1, exp(-beta * dE_total))
    log_p = -beta * dE_total
    probs = torch.exp(log_p)
    rand = torch.rand(B, device=device)
    
    accept_mask = rand < probs
    
    # 4. Update Spins
    # indices to flip where accept_mask is True
    # We create a mask for scatter
    flip_indices = indices[accept_mask]
    
    # Use scatter to update spins (flip signs)
    # spins[b, idx] *= -1 
    # Since spins is [B, N], we can manipulate it flattened. 
    # But we need batch indices.
    batch_indices = torch.arange(B, device=device)[accept_mask]
    
    # Linear indices in the flat buffer
    flat_flip_indices = batch_indices * N + flip_indices
    
    # Create a flat view to modify
    spins_flat = spins.view(-1)
    spins_flat[flat_flip_indices] *= -1.0
    spins = spins_flat.view(B, N)
    
    # 5. Update Bias (Metadynamics Step) at current state
    # Usually updated AFTER the move (so at new state if accepted, old state if rejected)
    current_mag = ising2d_mag(spins)
    bias_pot.update(current_mag, mod_factor)
    
    return spins, accept_mask.float().mean().item()

def run_metadynamics_schedule(args):
    """
    Main loop following the Modulation Schedule.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    L = args.size
    N = L * L
    
    # Temps
    if args.temps:
        temps = args.temps
    else:
        temps = np.linspace(args.temps_min, args.temps_max, args.temps_n)
    
    print(f"Simulating temperatures: {temps}")
    
    # Output Setup
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.out
    
    # Results storage
    final_pots = []
    steps_record = []
    
    # Initialize Spins (Random)
    # Typically we run parallel walkers (Batch Size = B).
    # The user manual script runs ONE walker per temp sequentially?
    # "mc = SGCMonteCarlo(atoms...)" -> Single walker.
    # To be faster, we can run B walkers in parallel and aggregate their visits?
    # Or just run 1 walker to be strictly identical to 'classical' serial simulation.
    # Running 1 walker is VERY slow in Python/Pytorch overhead. 
    # Running B walkers helps converge the histogram faster (collective updates).
    B = 64 # Default batch size for efficiency
    spins = (torch.randint(0, 2, (B, N), device=device) * 2 - 1).float()
    
    for T in temps:
        print(f"\n========================================")
        print(f"Starting T = {T:.4f}")
        print(f"========================================")
        
        beta = 1.0 / T
        
        # Check if result already exists
        expected_prior_path = f"{out_prefix}/ising_{args.size}x{args.size}_pots_T{T:.2f}.npy"
        if Path(expected_prior_path).exists():
            print(f"  Found existing results for T={T} at {expected_prior_path}. Skipping simulation.")
            try:
                vals = np.load(expected_prior_path)
                final_pots.append(vals)
                steps_record.append(0) # Unknown steps for resumed run
                continue
            except Exception as e:
                print(f"  Error loading {expected_prior_path}: {e}. Restarting...")

        # Initialize Bias for this temperature
        # Range of Magnetization is [-1, 1]
        bias_pot = BinnedBiasPotential(cv_min=-1.0, cv_max=1.0, nbins=args.nbins, device=device)
        
        mod_factor = args.mod_start
        total_steps_T = 0
        
        # Restart spins for new temp? Or continue from previous?
        # Continue from previous
        # spins = (torch.randint(0, 2, (B, N), device=device) * 2 - 1).float()

        
        schedule_step = 0
        
        while mod_factor >= args.mod_stop:
            print(f"  [Schedule {schedule_step}] Mod factor: {mod_factor:.6f}")
            
            # Inner Loop: Run until flat
            steps_this_mod = 0
            is_flat = False
            
            pbar = tqdm(total=args.max_sweeps, desc=f"    Sampling (Mod={mod_factor:.1e})", unit="sweeps")
            
            while not is_flat and steps_this_mod < args.max_sweeps:
                # Run a block of steps (e.g., 100 or N)
                # Note: max_sweeps is usually large.
                # Let's run N steps (1 sweep equivalent roughly) per iteration
                
                # Check flatness check interval
                # Run K steps
                K = 100 # Check every 100 sweeps equivalent?
                # Actually, check every 'interval' steps. User script checked frequently.
                
                # Run a batch of updates
                # Since we have B walkers, 1 step updates B samples.
                # 1 sweep = N updates per walker.
                # So we run N calls to mcmc_step() to equal 1 sweep?
                # Yes, technically.
                
                # Doing flatness check is expensive? No, just tensor min/mean.
                # Let's do it every 10 sweeps == 10 * N steps.
                
                check_interval = N # Check every sweep?
                
                for _ in range(check_interval):
                    spins, acc = mcmc_step(spins, bias_pot, beta, args.J, args.field, mod_factor, device)
                
                steps_this_mod += 1 # We count "checks" or "sweeps"? User arg is max_sweeps.
                total_steps_T += 1
                pbar.update(1)
                
                # Check flatness
                if bias_pot.is_flat(args.flatness_limit):
                    is_flat = True
            
            pbar.close()
            
            if is_flat:
                # print(f"    -> Flatness reached after {steps_this_mod} sweeps.")
                pass
            else:
                print(f"    -> Warning: Max sweeps reached without flatness.")
                
            # Reduce factor
            mod_factor /= 2.0
            schedule_step += 1
            
            # Reset histogram
            bias_pot.reset_visits()
        
        # Save results for this T
        print(f"  Finished T={T}. Total sweeps: {total_steps_T}")
        grid, vals = bias_pot.get_bias_grid_np()
        
        # CLEASE Normalization Logic
        # 1. Convert V(s) to Free Energy F(s) = -V(s)
        tmp = -vals
        
        # 2. Subtract linear baseline between endpoints (Mixing Energy)
        # concs from 0 to 1
        concs = np.linspace(0, 1.0, len(vals))
        baseline = concs * tmp[-1] + (1 - concs) * tmp[0]
        new_bias_values = tmp - baseline
        
        # 3. Scale by system size (nbins-1) to get eV/atom
        # (Assuming nbins = N+1)
        norm_factor = len(vals) - 1 if len(vals) > 1 else 1.0
        final_vals = new_bias_values / norm_factor
        
        save_path = f"{out_prefix}/ising_{args.size}x{args.size}_pots_T{T:.2f}.npy"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, final_vals)
        
        final_pots.append(final_vals)
        steps_record.append(total_steps_T)

    # Save aggregated
    save_path_agg = f"{out_prefix}/ising_{args.size}x{args.size}_pots.npy"
    Path(save_path_agg).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path_agg, np.array(final_pots))
    
    save_path_steps = f"{out_prefix}/ising_{args.size}x{args.size}_steps.json"
    with open(save_path_steps, 'w') as f:
        json.dump(steps_record, f)
        
    print(f"\nAll done. Results saved to {out_prefix}*")

def main():
    parser = argparse.ArgumentParser(description="Run classical Metadynamics (Modulation Schedule) on 2D Ising")
    
    # Physics Params
    parser.add_argument("--size", type=int, default=10, help="Lattice size L")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant")
    parser.add_argument("--field", type=float, default=0.0, help="External field")
    
    # Temp Params
    parser.add_argument("--temps", type=float, nargs="+", default=None, help="List of temperatures")
    parser.add_argument("--temps-min", type=float, default=1.0)
    parser.add_argument("--temps-max", type=float, default=5.0)
    parser.add_argument("--temps-n", type=int, default=10)
    
    # Metadynamics Params
    parser.add_argument("--nbins", type=int, default=None, help="Number of bins for CV (Default: L*L + 1)")
    parser.add_argument("--mod-start", type=float, default=0.1, help="Initial bias update height")
    parser.add_argument("--mod-stop", type=float, default=1e-4, help="Stop threshold for update height")
    parser.add_argument("--flatness-limit", type=float, default=0.8, help="Flatness threshold (min/mean)")
    parser.add_argument("--max-sweeps", type=int, default=10000, help="Max sweeps per modulation stage")
    
    # Output
    parser.add_argument("--out", type=str, default="metadyn_results/output", help="Output prefix")
    
    args = parser.parse_args()
    
    # Clean L arg if passed as --L and --size
    # User script used --size nargs=3. We simplify to --size int (L).
    # If user passes 3 ints, we handle it? argparse handles it if type=int nargs='?'.
    # Let's stick to simple scalar L for the benchmark as requested "simpler version".
    
    # Auto-set nbins if not provided
    if args.nbins is None:
        args.nbins = args.size * args.size + 1
        print(f"Auto-setting nbins to {args.nbins} for L={args.size}")
    
    run_metadynamics_schedule(args)

if __name__ == "__main__":
    main()
