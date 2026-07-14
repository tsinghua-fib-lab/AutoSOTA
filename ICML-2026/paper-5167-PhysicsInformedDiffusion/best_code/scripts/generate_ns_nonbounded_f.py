import tqdm
import pickle
import numpy as np
import torch
import scipy.io

def random_index(k, grid_size, seq, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size, seq] grid.'''
    np.random.seed(seed)
    mask = torch.zeros((grid_size, grid_size, seq), dtype=torch.float32).to(device)
    for t in range(seq):
        indices = np.random.choice(grid_size**2, k, replace=False)
        xs, ys = np.unravel_index(indices, (grid_size, grid_size))
        for x, y in zip(xs, ys):
            mask[x, y, t] = 1
    return mask


def inverse_components_to_fields(x, keep, full_shape=(128,128)):
    H, W = full_shape
    cx, cy = H // 2, W // 2
    device = x.device
    seq = x.shape[-1] // 2

    # Split into real/imag pairs
    fft_fields = []
    for i in range(seq):
        real = x[:,:,:,2*i]
        imag = x[:,:,:,2*i+1]
        crop = torch.complex(real, imag).to(device) # dim (batch,keep,keep)
        
        # Full padded shape
        padded = torch.zeros((x.shape[0],H,W), dtype=torch.complex128, device=device)

        # Compute insertion slices (centered)
        padded[:,cx-keep//2:cx+keep//2,cy-keep//2:cy+keep//2] = crop

        # Undo fftshift
        padded = torch.fft.ifftshift(padded,  dim=[-2, -1] )
        fft_fields.append(padded)
        
    fft_fields = torch.stack(fft_fields, dim=-1) # (batch,128,128,seq)
    return fft_fields #dimension (batch,128,128,seq)



def get_ns_nonbounded_hat_loss(u_hat, u_GT, mask, nu, T=1.0, n_steps=10, device=torch.device("cuda")):
    """
    u_hat : (B, H, W, T) complex Fourier vorticity
    u_GT  : (B, H, W, T) real physical vorticity
    mask  : (H, W, T) 
    nu    : viscosity
    T     : total time (default 1.0)
    n_steps : number of time snapshots (default 10)
    """
    
    # --------------------------------------------------
    # Shapes & constants
    # --------------------------------------------------
    B, H, W, T_dim = u_hat.shape
    N = H
    L = 1.0
    
    # Time step between recorded snapshots
    dt = T / (n_steps - 1)

    # --------------------------------------------------
    # Wavenumbers
    # --------------------------------------------------
    kx = torch.fft.fftfreq(N, d=L / N).to(device) * 2 * torch.pi
    ky = torch.fft.fftfreq(N, d=L / N).to(device) * 2 * torch.pi
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")      # (H,W)

    k2 = kx**2 + ky**2                                  # (H,W)

    # Safe inverse Laplacian
    inv_k2 = torch.zeros_like(k2)
    inv_k2[k2 != 0] = 1.0 / k2[k2 != 0]

    # --------------------------------------------------
    # Forcing term: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    # --------------------------------------------------
    t = torch.linspace(0, 1, N+1, device=device)[:-1]
    X, Y = torch.meshgrid(t, t, indexing="ij")
    f = 0.1 * (torch.sin(2*torch.pi*(X + Y)) + torch.cos(2*torch.pi*(X + Y)))
    f_hat = torch.fft.fft2(f, norm="ortho")  # Complex (H, W)
    f_hat = f_hat[None, :, :, None].expand(B, H, W, T_dim)

    # --------------------------------------------------
    # grad w in physical space (pseudo-spectral)
    # --------------------------------------------------
    dw_dx = torch.fft.ifft2(
        1j * kx[None, :, :, None] * u_hat,
        dim=(1, 2),
        norm="ortho"
    ).real

    dw_dy = torch.fft.ifft2(
        1j * ky[None, :, :, None] * u_hat,
        dim=(1, 2),
        norm="ortho"
    ).real

    # --------------------------------------------------
    # Velocity from vorticity (Biot–Savart)
    # --------------------------------------------------
    vx_hat = 1j * ky[None, :, :, None] * inv_k2[None, :, :, None] * u_hat
    vy_hat = -1j * kx[None, :, :, None] * inv_k2[None, :, :, None] * u_hat

    # Zero mode (physically correct)
    vx_hat[:, 0, 0, :] = 0.0
    vy_hat[:, 0, 0, :] = 0.0

    vx = torch.fft.ifft2(vx_hat, dim=(1, 2), norm="ortho").real
    vy = torch.fft.ifft2(vy_hat, dim=(1, 2), norm="ortho").real

    # --------------------------------------------------
    # Nonlinear term  v \cdot grad w
    # --------------------------------------------------
    nonlinear = vx * dw_dx + vy * dw_dy
    nonlinear_hat = torch.fft.fft2(
        nonlinear, dim=(1, 2), norm="ortho"
    )

    # --------------------------------------------------
    # Time derivative (second-order accurate, normalized by dt)
    # --------------------------------------------------
    first = (-3*u_hat[...,0] + 4*u_hat[...,1] - u_hat[...,2]).unsqueeze(-1) / (2*dt)
    last  = ( 3*u_hat[...,-1] - 4*u_hat[...,-2] + u_hat[...,-3]).unsqueeze(-1) / (2*dt)
    middle = 0.5 * (u_hat[..., 2:] - u_hat[..., :-2]) / dt

    dt_w_hat = torch.cat([first, middle, last ], dim=-1)

    # --------------------------------------------------
    # Viscous term
    # --------------------------------------------------
    laplace_hat = -k2[None, :, :, None] * u_hat
    laplace_hat[:, 0, 0, :] = 0.0   # optional but clean

    # --------------------------------------------------
    # PDE residual: \partial_t w + v \cdot grad w = ν lap w + f
    # --------------------------------------------------
    rhs_hat = -nonlinear_hat + nu * laplace_hat + f_hat

    residual = dt_w_hat - rhs_hat
    residual[...,0] = 0*residual[...,0] 
    residual[...,-1] = 0*residual[...,-1]
     
    # --------------------------------------------------
    # Observation loss (physical space)
    # --------------------------------------------------
    w = torch.fft.ifft2(u_hat, dim=(1, 2), norm="ortho").real

    if mask.dim() == 3:
        mask = mask[None, :, :, :]

    observation_loss = (w - u_GT) * mask

    # --------------------------------------------------
    # Divergence-free diagnostic (should be almost 0)
    # --------------------------------------------------
    div_hat = 1j * (
        kx[None, :, :, None] * vx_hat +
        ky[None, :, :, None] * vy_hat
    )
    
    return residual.real, div_hat.real, observation_loss



def apply_adam_frequency_aware_fft(x_next, grad, state, step, 
        full_size=(128, 128), 
                                   crop_size=(32, 32),
                                   center=(64, 64),
                                   lr_low=1.0, lr_high=0.004, freq_transition=12,
                                   beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam with frequency-dependent learning rates for FFT-based problems.
    
    For FFT:
    - Frequencies are centered 
    - Low frequencies near center, high frequencies at edges
    - Uses real/imag representation of complex coefficients
    
    Args:
        x_next: Current variable (latent space)
                Shape: (batch, channels*2, crop_h, crop_w)
                channels*2 because real and imaginary parts are separate
        grad: Gradient with same shape as x_next
        state: Dictionary with 'm', 'v'
        step: Step number
        full_size: Size of full FFT domain 
        crop_size: Size of cropped frequency window (keepx, keepy)
        center: Center position (cx, cy) where crop is taken from
        lr_low: Learning rate for low frequencies (near DC)
        lr_high: Learning rate for high frequencies (far from DC)
        freq_transition: Frequency magnitude where transition happens
        beta1, beta2, eps: Adam parameters
    
    Returns:
        updated x_next, updated state
    """
    device = grad.device
    
    if 'm' not in state:
        state['m'] = torch.zeros_like(grad)
        state['v'] = torch.zeros_like(grad)
    
    # Unpack parameters
    H, W = full_size
    keepx, keepy = crop_size
    cx, cy = center
    
    # Create frequency grid for FULL FFT domain (before fftshift)
    # After fftshift, DC is at center (cx, cy)
    kx = torch.arange(H, device=device, dtype=torch.float32)
    ky = torch.arange(W, device=device, dtype=torch.float32)
    
    # Shift so that DC (0,0) is at center
    kx = kx - cx
    ky = ky - cy
    
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    
    # Frequency magnitude (distance from DC)
    k_mag_full = torch.sqrt(KX**2 + KY**2)
    
    # Extract the cropped region that matches your data
    # This corresponds to: f_fft_shifted[slice_x, slice_y]
    slice_x = slice(cx - keepx//2, cx + keepx//2)
    slice_y = slice(cy - keepy//2, cy + keepy//2)
    
    k_mag_crop = k_mag_full[slice_x, slice_y]  # Shape: (keepx, keepy)
    
    # Create frequency-dependent learning rate
    # Smooth transition from lr_low (near DC) to lr_high (far from DC)
    lr_ratio = lr_high / lr_low
    transition_width = 10.0
    freq_weight = lr_ratio + (1 - lr_ratio) / (1 + torch.exp((k_mag_crop - freq_transition) / transition_width))
    
    # Expand to match gradient shape
    # grad shape: (batch, channels*2, keepx, keepy)
    # We want same frequency weight for real and imaginary parts
    num_channels = grad.shape[1] // 2  # Divide by 2 because real/imag are separate
    
    # Replicate for real and imaginary parts
    freq_weight_real_imag = freq_weight.unsqueeze(0).unsqueeze(0).repeat(1, num_channels * 2, 1, 1)
    
    # Ensure it matches grad shape exactly
    freq_weight_expanded = freq_weight_real_imag.expand_as(grad)
    
    # Standard Adam update
    state['m'] = beta1 * state['m'] + (1 - beta1) * grad
    state['v'] = beta2 * state['v'] + (1 - beta2) * (grad ** 2)
    
    # Bias correction
    m_hat = state['m'] / (1 - beta1 ** step)
    v_hat = state['v'] / (1 - beta2 ** step)
    
    # Apply frequency-dependent learning rate
    update = lr_low * freq_weight_expanded * m_hat / (torch.sqrt(v_hat) + eps)
    x_next = x_next - update
    
    return x_next, state

    

def generate_ns_nonbounded_hat(config):
    """Generate non-bounded NS equation."""
    ############################ Load data and network ############################
    keep = 32  # keep x keep central square
    seq = 10 
    
    relative_error_list = []
    state_list = []   
    obs_loss_list = []
    residual_loss_list = []
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = scipy.io.loadmat(datapath)
   
    adam_state = {}
    optimizer_step = 1  # Counter for Adam bias correction
   
    # Adam Frequency-Aware parameters
    lr_low = config["generate"].get("lr_low", 1.0)
    lr_high = config["generate"].get("lr_high", 0.03)
    freq_transition = config["generate"].get("freq_transition", 25)
    beta1 = config["generate"].get("beta1", 0.97)
    beta2 = config["generate"].get("beta2", 0.9875)
    obs_steps = config["generate"]["obs_steps"]
    obs = config["generate"]["obs"]
       
    batch_size = config['generate']['batch_size']
    
    ground_truth = data['u'][offset:offset+batch_size, :, :,0:seq]
    ground_truth = torch.tensor(ground_truth, dtype=torch.float64, device=device)
    assert ground_truth.shape==(batch_size, 128, 128, seq), f"Wrong shape: {ground_truth.shape}"
        
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)
    
    ############################ Set up EDM latent ############################
    print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    assert latents.shape==(batch_size,2*seq,keep,keep), f"Wrong shape: {latents.shape}"
    
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
    
    sigma_min = config['generate']['sigma_min']
    sigma_max = config['generate']['sigma_max']
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    num_steps = config['test']['iterations']
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    rho = config['generate']['rho']
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    save_iters = np.linspace(0, num_steps-1, 10, dtype=int)
    
    x_next = latents.to(torch.float64) * sigma_t_steps[0]
    known_index = random_index(obs, 128, seq, seed=0) # shape (128,128,seq)
    
    not_obs = [x for x in range(0,10) if x not in obs_steps]    
    known_index[:,:,not_obs]=0
    mean = torch.tensor(np.load("mean_and_std/ns-nonbounded_mean.npy")).to(device) 
    std = torch.tensor(np.load("mean_and_std/ns-nonbounded_std.npy")).to(device)  
    zeta_obs = config['generate']['zeta_obs']
    zeta_pde = config['generate']['zeta_pde'] 
    nu = 0.001 

    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step', disable=True): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        
        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        x_N = x_N * (std[None,:,:,:] + 1e-8) + mean[None,:,:,:]
        x_N_full = inverse_components_to_fields(x_N.permute(0,2,3,1), keep, full_shape=(128,128)) 
        pde_loss, divergence_loss, observation_loss = get_ns_nonbounded_hat_loss(x_N_full, ground_truth, known_index, nu, device=device) 
        
        L_pde = torch.sum(torch.abs(pde_loss)**2, dim=[1,2])   # (batch, seq)
        L_divergence = torch.sum(torch.abs(divergence_loss)**2, dim=[1,2])
        L_obs = torch.sum(observation_loss**2, dim=[1,2])
        assert not torch.isnan(L_pde).any(), f"NaN in L_pde at iteration {i}"
        assert not torch.isnan(L_divergence).any(), f"NaN in L_divergence at iteration {i}"
        assert not torch.isnan(L_obs).any(), f"NaN in L_obs at iteration {i}"
        
        grad_x_cur_obs = torch.autograd.grad(outputs=L_obs, inputs=x_cur, grad_outputs=torch.ones_like(L_obs), retain_graph=True)[0]
        grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur, grad_outputs=torch.ones_like(L_pde), retain_graph=True)[0]
        grad_x_cur_div = torch.autograd.grad(outputs=L_divergence, inputs=x_cur, grad_outputs=torch.ones_like(L_divergence), retain_graph=True)[0]

        combined_grad = zeta_pde * grad_x_cur_pde + zeta_obs * grad_x_cur_obs 
        
        x_next, adam_state = apply_adam_frequency_aware_fft(x_next, combined_grad, adam_state, optimizer_step, lr_low=lr_low, lr_high=lr_high, freq_transition=freq_transition, beta1=beta1, beta2=beta2)
        optimizer_step += 1
         
        with torch.no_grad():
            residual_loss_list.append(L_pde.detach().cpu())
            obs_loss_list.append(L_obs.detach().cpu())
            uu_full = torch.fft.ifft2(x_N_full,   dim=[1, 2], norm='ortho').real #(batch,128,128,10)
            relative_error = torch.norm(uu_full - ground_truth[:,:,:,:], 2, dim = [1,2] )/torch.norm(ground_truth, 2, dim = [1,2] )
            relative_error_list.append(relative_error.detach().cpu())
            if i in save_iters:
                x = x_next * (std[None,:,:,:] + 1e-8) + mean[None, :,:,:]
                x = inverse_components_to_fields(x.permute(0,2,3,1), keep, full_shape=(128,128))
                uu = torch.fft.ifft2(x, dim=[1, 2], norm='ortho').real
                state_list.append(uu.detach().cpu())
                
    ############################ Save the data ############################
    x_final = x_next * (std[None,:,:,:] + 1e-8) + mean[None, :,:,:]
    
    x_final = inverse_components_to_fields(x_final.permute(0,2,3,1), keep, full_shape=(128,128)) 
    u = torch.fft.ifft2(x_final,   dim=[1, 2], norm='ortho').real #(batch, 128, 128, 10)
    # Compute relative errors for each time step
    rel_error = torch.norm(u - ground_truth, dim=[1,2], p=2) / torch.norm(ground_truth, dim=[1,2], p=2)  # (batch, 10)
    # Compute relative errors for each time step
    rel_obs_error = torch.norm((u - ground_truth)* known_index, dim=[1,2], p=2) / torch.norm(ground_truth* known_index, dim=[1,2], p=2)  # (batch, 10)
    
    # Print one line per batch
    for b in range(u.shape[0]):
        line = [f"batch:{b}"]
        # rel_err_0 ... rel_err_9
        line += [f"rel_err_{t}:{rel_error[b,t].item():.6e}" for t in range(seq)]
        # rel_obs_err_0 ... rel_obs_err_9
        line += [f"rel_obs_err_{t}:{rel_obs_error[b,t].item():.6e}" for t in range(seq)]
        # L_pde_0 ... L_pde_9
        line += [f"L_pde_{t}:{L_pde[b,t].item():.6e}" for t in range(seq)]
        # L_obs_0 ... L_obs_9
        line += [f"L_obs_{t}:{L_obs[b,t].item():.6e}" for t in range(seq)]
        print(" ".join(line))

    u = u.to('cpu').detach().numpy()
    ground_truth = ground_truth.to("cpu").detach().numpy()
    known_index = known_index.to("cpu").detach().numpy()
    
    scipy.io.savemat('ns_nonbounded_results.mat', {'u': u, 'ground_truth':ground_truth,
                                                   'known_index':known_index, "residual_loss_list": residual_loss_list,
                                                   "obs_loss_list":obs_loss_list, 'relative_error_list' : relative_error_list,
                                                   'state_list' : state_list,
                                                   })
    
