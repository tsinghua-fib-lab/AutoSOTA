import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import warnings
from tqdm import tqdm

# Imports
from .hypersphere import Hypersphere
from .scheduler_lib import Geometric
from . import distribution
from .sde import LogBridge_Mixture, ContinuousSDE
from .model.dit import MixeDiT
from .model.ema import ExponentialMovingAverage
from .model import utils as mutils
from .tokenizing.discrete_encoding import convert_samples_to_binary

warnings.filterwarnings("ignore", category=FutureWarning)

inds = ['/res_metrics/avg_drag', '/res_metrics/avg_l', '/res_metrics/comp_weights/total', '/res_metrics/cruise_aoa', '/res_metrics/cruise_fom', '/res_metrics/cruise_l', '/res_metrics/soc']

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt

# ==========================================
# Mixed Sampler (Corrected Time Direction)
# ==========================================
def local_pc_sampler_mixed(model, sde_disc, sde_cont, batch_dims, 
                           continuous_dim, steps=1000, eps=1e-5, device='cuda',
                           fixed_t_cont=None, cond_cont_val=None):
    """
    Sampler integrating from t=0 (Noise/Prior) -> t=1 (Data).
    
    Args:
        fixed_t_cont (float): 
            - 1.0 = Condition on Data (alpha=1, beta=0)
            - 0.0 = Marginalize/Noise (alpha=0, beta=1)
    """
    # 1. Get Drift Function
    drift_fn = mutils.get_drift_fn_mixed(model, sde_disc, sde_cont, train=False, sampling=True)
    
    # 2. Setup Time (0 -> 1)
    timesteps = torch.linspace(0, 1-eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    # 3. Sample Discrete Prior (t=0, Noise/Mask)
    x_disc = sde_disc.prior_sample(batch_dims, device)

    # 4. Setup Continuous Input
    num_samples = batch_dims[0]
    
    if fixed_t_cont is not None:
        # --- FIXED MODE ---
        t_c_scalar = float(fixed_t_cont)
        t_cont = torch.ones(num_samples, device=device) * t_c_scalar
        
        # LOGIC SWAP: In RDLM/Flow Matching, t=1 is DATA, t=0 is NOISE
        if t_c_scalar > 0.5: # t=1 case (Conditioning on Clean Data)
            if cond_cont_val is not None:
                x_cont = cond_cont_val.to(device)
            else:
                # Default clean condition (e.g. zeros)
                x_cont = torch.zeros(num_samples, 1, continuous_dim, device=device)
        else: # t=0 case (Marginalization / Pure Noise)
            # Input is pure noise N(0, I)
            # Because at t=0, alpha=0, beta=1
            x_cont = torch.randn(num_samples, 1, continuous_dim, device=device)
            
    else:
        # --- JOINT SAMPLING MODE ---
        # Initialize x_cont from Prior N(0, I) at t=0
        x_cont = torch.randn(num_samples, 1, continuous_dim, device=device)

    # 5. Sampling Loop
    with torch.no_grad():
        for i in tqdm(range(steps), desc=f'Sampling (t_cont={fixed_t_cont})'):
            t_d = timesteps[i] * torch.ones(num_samples, device=device)
            
            # If joint sampling, continuous time evolves 0 -> 1
            # If fixed, it stays at the conditioning/masking timestep
            if fixed_t_cont is None:
                t_c = timesteps[i] * torch.ones(num_samples, device=device)
            else:
                t_c = t_cont 
            
            # --- Predictor Step (Discrete Only) ---
            z = sde_disc.manifold.random_normal_tangent(base_point=x_disc)
            
            # Calculate drift
            # The model sees x_cont. 
            # If t_c=1, model sees Data. If t_c=0, model sees Noise.
            drift_disc, _ = drift_fn(x_disc, x_cont, t_d, t_c)
            
            diffusion = sde_disc.diffusion(x_disc, t_d)

            # Update discrete state
            tangent_vec = torch.einsum("...,...ij->...ij", diffusion, z) * np.sqrt(np.abs(dt))
            tangent_vec = tangent_vec + drift_disc * dt
            x_disc = sde_disc.manifold.exp(tangent_vec=tangent_vec, base_point=x_disc)
            
            # (Optional) If we were doing joint sampling, we would update x_cont here too
            # x_cont = x_cont + drift_cont * dt ...

    # 6. Decode
    probs = sde_disc.manifold.map_to_simplex(x_disc)
    if sde_disc.add_mask_token:
        probs = probs[..., :-1]
        
    return probs.argmax(dim=-1)

# ==========================================
# Main
# ==========================================
def main(condition_input,
        workdir="./training_data/mixedit/models",
        ckpt_path="/checkpoint_ep1000.pth",
        sde_path="/sde_stats.pkl",
        out_path_cond="/generated_samples_cond_mass_t1.txt",
        out_path_marg="/generated_samples_marg_t0.txt",
        token_path="./training_data/mixedit/data",
        num_samples=10000):

    # Settings
    checkpoint_path = workdir + ckpt_path
    sde_stats_path = workdir + sde_path
    
    # Paths for separate outputs
    output_path_cond = out_path_cond # Conditioned (t=1) #REMOVED WORKDIR
    output_path_marg = out_path_marg # Marginalized (t=0)
    
    continuous_dim = len(condition_input)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # 1. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = load_checkpoint(checkpoint_path, device)
    cfg_dict = ckpt['config']
    
    # 2. Rebuild SDEs
    real_vocab_size = cfg_dict['tokens'] + (1 if cfg_dict['add_mask_token'] else 0)
    manifold = Hypersphere(real_vocab_size - 1)
    
    scheduler = Geometric(beta_0=cfg_dict['beta_0'], beta_f=cfg_dict['beta_f'], weight_type='step')
    batch_dims = (num_samples, cfg_dict['seq_len'], real_vocab_size)
    prior_dist = distribution.Mixture(batch_dims=batch_dims, device=device)
    
    if os.path.exists(sde_stats_path):
        with open(sde_stats_path, "rb") as f:
            alphas, rhos = pickle.load(f)
        preprocessed = (alphas.to(device), rhos.to(device))
    else:
        preprocessed = None
        print("Warning: SDE stats not found. Recalculating...")

    sde_disc = LogBridge_Mixture(
        manifold=manifold,
        scheduler=scheduler,
        prior_dist=prior_dist,
        device=device,
        preprocess_steps=cfg_dict['preprocess_steps'],
        mix_type="step",
        step_thr=cfg_dict['step_thr'],
        dims=batch_dims,
        preprocessed=preprocessed
    )
    
    sde_cont = ContinuousSDE(schedule="vp")

    # 3. Rebuild Model
    model = MixeDiT(
        input_dim=real_vocab_size,
        continuous_dim=continuous_dim,
        output_dim=real_vocab_size,
        hidden_size=cfg_dict['hidden_size'],
        n_heads=cfg_dict['n_heads'],
        cond_dim=cfg_dict['cond_dim'],
        dropout=cfg_dict['dropout'],
        n_blocks=cfg_dict['n_blocks'],
        length=cfg_dict['seq_len']
    ).to(device)
    
    state_dict = ckpt['model']
    new_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg_dict['ema_decay'])
    ema.load_state_dict(ckpt['ema'])
    ema.copy_to(model.parameters())
    model.eval()

    # --- Mode 1: Condition on Continuous (Data is at t=1) ---
    print(f"\nGenerating samples conditioned on Zeros (t_cont=1.0 -> Data)...")
    #zeros_condition = torch.zeros(num_samples, cfg_dict['seq_len'], continuous_dim, device=device)

    #zeros_condition[:,:,2] = 2.0 # Increase total mass

    base = torch.tensor(condition_input, device=device).unsqueeze(0).unsqueeze(0)
    condition = base.expand(num_samples, cfg_dict['seq_len'], continuous_dim)
    
    gen_indices_cond = local_pc_sampler_mixed(
        model=model,
        sde_disc=sde_disc,
        sde_cont=sde_cont,
        batch_dims=batch_dims,
        continuous_dim=continuous_dim,
        steps=1000,
        eps=1e-5,
        device=device,
        fixed_t_cont=1.0,  # t=1 is Data
        cond_cont_val=condition
    )
    np.savetxt(output_path_cond, gen_indices_cond.cpu().numpy(), fmt='%d')
    print(f"Saved conditioned samples to {output_path_cond}")

    # --- Mode 2: Marginalize Continuous (Noise is at t=0) ---
    print(f"\nGenerating samples with marginalized continuous (t_cont=0.0 -> Noise)...")
    gen_indices_marg = local_pc_sampler_mixed(
        model=model,
        sde_disc=sde_disc,
        sde_cont=sde_cont,
        batch_dims=batch_dims,
        continuous_dim=continuous_dim,
        steps=1000,
        eps=1e-5,
        device=device,
        fixed_t_cont=0.0,  # t=0 is Noise
        cond_cont_val=None
    )
    np.savetxt(output_path_marg, gen_indices_marg.cpu().numpy(), fmt='%d')
    print(f"Saved marginalized samples to {output_path_marg}")

    tokenizer = np.load(token_path+"tokenizer.npy",allow_pickle=True).item()
    all_ones_indices = np.load(token_path+"all_ones_indices.npy")
    original_D = np.load(token_path+"original_D.npy")

    recon_data = convert_samples_to_binary(
        gen_indices_cond, 
        tokenizer, 
        all_ones_indices, 
        original_D
        )

    return recon_data

if __name__ == "__main__":
    main()