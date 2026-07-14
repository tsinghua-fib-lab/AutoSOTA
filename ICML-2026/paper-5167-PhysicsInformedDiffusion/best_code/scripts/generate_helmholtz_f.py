import tqdm
import pickle
import numpy as np
import torch
import scipy.io
from torch_utils import dst2, idst2, apply_adam_frequency_aware


def random_index(k, grid_size, seed=0, device=torch.device('cuda')):
    '''randomly select k indices from a [grid_size, grid_size] grid.'''
    np.random.seed(seed)
    indices = np.random.choice(grid_size**2, k, replace=False)
    indices_2d = np.unravel_index(indices, (grid_size, grid_size))
    indices_list = list(zip(indices_2d[0], indices_2d[1]))
    mask = torch.zeros((grid_size, grid_size), dtype=torch.float32).to(device)
    for i in indices_list:
        mask[i] = 1
    return mask


def inverse_components_to_fields(components, keep, pad_a, pad_u, full_shape=(128,128)):

    """
    Args:
        components: torch.Tensor of shape (2, keep, keep), on any device
        keep: int, size of cropped frequency square
        full_shape: tuple (H, W), size of full FFT domain

    Returns:
        a_hat: tensor of shape (B, H-2+2*pad_a, W-2+2*pad_a)
        u_hat: tensor of shape (B, H-2+2*pad_u, W-2+2*pad_u)
    """
    H, W = full_shape
    device = components.device
    B = components.shape[0]
    
    
    u_hat_padded = torch.zeros(
        (B, H-2 + 2*pad_u, W-2 + 2*pad_u),
        device=device,
        dtype=components.dtype
    )

    a_hat_padded = torch.zeros(
        (B, H-2 + 2*pad_a, W-2 + 2*pad_a),
        device=device,
        dtype=components.dtype
    )

    # channel extraction (now batched)
    a_hat_cropped = components[:, 0]  # (B, keep, keep)
    u_hat_cropped = components[:, 1]

    u_hat_padded[:, :38, :38] = u_hat_cropped[:, :38, :38]
    a_hat_padded[:, :38, :38] = a_hat_cropped[:, :38, :38]

    for f in range(4):
        u_hat_padded[:, f, 38:2*38] = u_hat_cropped[:, :38, 38+f]
        u_hat_padded[:, 38:2*38, f] = u_hat_cropped[:, 38+f, :38]

        a_hat_padded[:, f, 38:2*38] = a_hat_cropped[:, :38, 38+f]
        a_hat_padded[:, 38:2*38, f] = a_hat_cropped[:, 38+f, :38]

    for f in range(2):
        u_hat_padded[:, f, 2*38:3*38] = u_hat_cropped[:, :38, 42+f]
        u_hat_padded[:, 2*38:3*38, f] = u_hat_cropped[:, 42+f, :38]

        a_hat_padded[:, f, 2*38:3*38] = a_hat_cropped[:, :38, 42+f]
        a_hat_padded[:, 2*38:3*38, f] = a_hat_cropped[:, 42+f, :38]

    
    return a_hat_padded, u_hat_padded
    


def calc_relative_error(x, pad_u, pad_a, u_GT, a_GT, i, mean, std, keep,  known_index_u,  known_index_a):
    with torch.no_grad():
        x = x * (std[None,:,:,:] + 1e-8) + mean[None, :,:,:]
        # Scale the data back
        a_hat_N, u_hat_N = inverse_components_to_fields(x, keep, pad_a, pad_u)
        
        u_final = idst2(u_hat_N, norm='ortho')[0]  # (H-2, W-2)
        a_final = idst2(a_hat_N, norm='ortho')[0] # (H-2, W-2)
        if pad_a>0:
            a_final = a_final[pad_a-1:-pad_a+1,pad_a-1:-pad_a+1]
        if pad_u>0:
            u_final = u_final[pad_u:-pad_u,pad_u:-pad_u]
            
        relative_error_a = torch.norm(a_final - a_GT[0], 2) / torch.norm(a_GT[0], 2)
        relative_error_u = torch.norm(u_final - u_GT[0,1:-1,1:-1], 2) / torch.norm(u_GT[0], 2)
        
        return relative_error_u, relative_error_a, u_final, a_final
        


def get_helmholtz_hat_loss(a_hat, u_hat, a_GT, u_GT, a_mask, u_mask, pad_a, pad_u, device=torch.device('cuda')):
    """Return the loss of the helmholtz equation and the observation loss."""
    
    N = a_GT.shape[-1]  # grid size
    L = 1.0  # physical domain length
    
    kx = torch.arange(1,N-1)
    ky = torch.arange(1,N-1)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    KX = KX.to(device)
    KY = KY.to(device)
    # Laplacian eigenvalues 
    lam = - (torch.pi * (KX ) / L)**2 - (torch.pi * (KY ) / L)**2
    lam = lam.unsqueeze(0)              # (1, 126, 126)
    # Spectral Laplacian
    lap_u_hat = lam * u_hat
    
    a = idst2(a_hat, norm='ortho')
    if pad_a >0:
        a=a[:, pad_a-1:-pad_a+1, pad_a-1:-pad_a+1]   # (B, 128, 128)
    
    u = idst2(u_hat, norm='ortho')        # (B, 126, 126)
    # PDE residual 
    pde_hat_loss = lap_u_hat - dst2(a[:,1:-1,1:-1], norm='ortho') + u_hat
    pde_loss = idst2(lap_u_hat, norm='ortho') - a[:,1:-1,1:-1] + u
    
    # Observation losses 
    u_GT_inner = u_GT[:, 1:-1, 1:-1]
    observation_loss_a = (a - a_GT) * a_mask
    observation_loss_u = (u - u_GT_inner) * u_mask[1:-1,1:-1]
    
    return pde_hat_loss, pde_loss, observation_loss_a, observation_loss_u



def build_freq_weight(size, keep, freq_transition, lr_low, lr_high, device):
    
    kx = torch.arange(size, device=device, dtype=torch.float32)
    ky = torch.arange(size, device=device, dtype=torch.float32)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    k_mag_tot = torch.sqrt(KX**2 + KY**2)

    # map full spectrum → cropped layout (your weird 44×44 layout)
    k_mag = torch.zeros((keep, keep), device=device)

    k_mag[:38, :38] = k_mag_tot[:38, :38]
    for f in range(4):
        k_mag[:38, 38+f] = k_mag_tot[f, 38:2*38]
        k_mag[38+f, :38] = k_mag_tot[38:2*38, f]
    for f in range(2):
        k_mag[:38, 42+f] = k_mag_tot[f, 2*38:3*38]
        k_mag[42+f, :38] = k_mag_tot[2*38:3*38, f]

    lr_ratio = lr_high / lr_low
    transition_width = 10.0
    freq_weight = lr_ratio + (1 - lr_ratio) / (
        1 + torch.exp((k_mag - freq_transition) / transition_width)
    )

    return freq_weight




def generate_helmholtz_hat(config):
    
    keep = 44  
    pad_a = 4
    pad_u = 0
    
    adam_state = {}
    optimizer_step = 1  # Counter for Adam bias correction
    state_list = [[],[]]
    
    # Adam Frequency-Aware parameters
    lr_low = config["generate"]["lr_low"]
    lr_high = config["generate"]["lr_high"]
    freq_transition = config["generate"]["freq_transition"]
    beta1 = config["generate"]["beta1"]
    beta2 = config["generate"]["beta2"]
    
    obs_a = config["generate"]["obs_a"]
    obs_u = config["generate"]["obs_u"]


    """Generate Helmholtz equation."""
    ############################ Load data and network ############################
    datapath = config['data']['datapath']
    offset = config['data']['offset']
    device = config['generate']['device']
    data = scipy.io.loadmat(datapath)
    batch_size = config['generate']['batch_size']
    
    a_GT = data['f_data'][offset:offset+batch_size, :, :]
    a_GT = torch.tensor(a_GT, dtype=torch.float64, device=device)  # size (batch, 128, 128)
    u_GT = data['psi_data'][offset:offset+batch_size, :, :]
    u_GT = torch.tensor(u_GT, dtype=torch.float64, device=device)  # size (batch, 128, 128)
    
    seed = config['generate']['seed']
    torch.manual_seed(seed)
    
    network_pkl = config['test']['pre-trained']
    print(f'Loading networks from "{network_pkl}"...')
    f = open(network_pkl, 'rb')
    net = pickle.load(f)['ema'].to(device)
    
    ############################ Set up EDM latent ############################
    print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
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
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) 
    
    N = 128
    x_next = latents.to(torch.float64) * sigma_t_steps[0]
    known_index_a = random_index(obs_a, N, seed=1)
    known_index_u = random_index(obs_u, N, seed=0)
    
    save_iters = np.linspace(0, num_steps-1, 10, dtype=int)
    error_history_u=[]
    error_history_a=[]
    
    freq_weight = build_freq_weight(N-2, keep, freq_transition, lr_low, lr_high, device)
    mean = torch.tensor(np.load("mean_and_std/helmholtz_mean.npy")).to(device)
    std = torch.tensor(np.load("mean_and_std/helmholtz_std.npy")).to(device)
    
    zeta_obs_a = config['generate']['zeta_obs_a']
    zeta_obs_u = config['generate']['zeta_obs_u']
    zeta_pde = config['generate']['zeta_pde']    
   
    ############################ Sample the data ############################
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step', disable=True): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        sigma_t = net.round_sigma(sigma_t_cur)
        # Euler step
        x_N = net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        x_N = x_N * (std[None,:,:,:] + 1e-8) + mean[None, :,:,:]
        
        # Scale the data back
        a_hat_N, u_hat_N = inverse_components_to_fields(x_N, keep, pad_a, pad_u)
        
        # Compute the loss
        pde_loss, pde_loss_real, observation_loss_a, observation_loss_u = get_helmholtz_hat_loss(a_hat_N, u_hat_N, a_GT, u_GT, known_index_a, known_index_u, pad_a, pad_u, device=device)
        L_pde = torch.sum(pde_loss_real**2, dim=(1, 2))
        L_obs_a = (N * N * torch.sum(observation_loss_a**2, dim=(1, 2)) / torch.sum(known_index_a))  # (B,)
        L_obs_u = (N * N * torch.sum(observation_loss_u**2, dim=(1, 2)) / torch.sum(known_index_u))  # (B,)
        
        assert not torch.isnan(L_pde).any(), f"NaN in L_pde at iteration {i}"
        assert not torch.isnan(L_obs_a).any(), f"NaN in L_obs_a at iteration {i}"
        assert not torch.isnan(L_obs_u).any(), f"NaN in L_obs_u at iteration {i}"
        
        grad_x_cur_obs_a = torch.autograd.grad(outputs=L_obs_a, inputs=x_cur, grad_outputs=torch.ones_like(L_obs_a), retain_graph=True)[0]
        grad_x_cur_obs_u = torch.autograd.grad(outputs=L_obs_u, inputs=x_cur, grad_outputs=torch.ones_like(L_obs_u), retain_graph=True)[0]
        grad_x_cur_pde = torch.autograd.grad(outputs=L_pde, inputs=x_cur, grad_outputs=torch.ones_like(L_pde), retain_graph=True)[0]
        
        combined_grad = (zeta_pde * grad_x_cur_pde + zeta_obs_a * grad_x_cur_obs_a + zeta_obs_u * grad_x_cur_obs_u)
        x_next, adam_state = apply_adam_frequency_aware(x_next, combined_grad, adam_state, optimizer_step, freq_weight, lr_low=lr_low, lr_high=lr_high, beta1=beta1, beta2=beta2)
        optimizer_step += 1
        
        
        u_state_err, a_state_err, u_state, a_state = calc_relative_error(x_next, pad_u, pad_a, u_GT, a_GT,i,mean,std,keep,known_index_u,known_index_a)
        error_history_u.append(u_state_err.detach().cpu().item())
        error_history_a.append(a_state_err.detach().cpu().item())
        
        if i in save_iters:
            state_list[0].append(u_state.detach().cpu())
            state_list[1].append(a_state.detach().cpu())
            
            
    x_final = x_next * (std[None,:,:,:] + 1e-8) + mean[None, :,:,:]
    a_hat_final, u_hat_final = inverse_components_to_fields(x_final, keep, pad_a, pad_u)
    
    u_final = idst2(u_hat_final, norm='ortho')  # (B, H, W)
    a_final = idst2(a_hat_final, norm='ortho')  # (B, H, W)
    if pad_a>0:
        a_final = a_final[:,pad_a-1:-pad_a+1,pad_a-1:-pad_a+1]
    if pad_u>0:
        u_final = u_final[:,pad_u:-pad_u,pad_u:-pad_u]
        
    relative_error_a = torch.norm(a_final - a_GT, 2, dim=(1,2)) / torch.norm(a_GT, 2, dim=(1,2))
    relative_error_u = torch.norm(u_final - u_GT[:,1:-1,1:-1], 2, dim=(1,2)) / torch.norm(u_GT[:,1:-1, 1:-1], 2, dim=(1,2))
    
    u_final_vec = u_final[:, known_index_u[1:-1,1:-1].bool()]
    u_GT_vec = u_GT[:, 1:-1, 1:-1][:, known_index_u[1:-1,1:-1].bool()]  
    a_final_vec = a_final[:, known_index_a.bool()]
    a_GT_vec = a_GT[:, known_index_a.bool()]  

    relative_error_u_obs = torch.norm(u_final_vec - u_GT_vec, p=2, dim=1) / torch.norm(u_GT_vec, p=2, dim=1)
    relative_error_a_obs = torch.norm(a_final_vec - a_GT_vec, p=2, dim=1) / torch.norm(a_GT_vec, p=2, dim=1)
    
    ### CALCULATION OF PDE RESIDUAL WITH FINITE DIFFERENCES
    pde_loss = torch.sum(pde_loss**2, dim=(1, 2))
    B = u_final.shape[0]
    h = 1.0 / (N - 1)
    u_full = torch.zeros(B, 1, N, N, device=u_final.device,dtype=u_final.dtype)
    u_full[:, 0, 1:-1, 1:-1] = u_final
    # Discrete Laplacian (5-point stencil)
    d2u = (u_full[:, :, :-2, 1:-1] + u_full[:, :, 2:,  1:-1] + u_full[:, :, 1:-1, :-2] + u_full[:, :, 1:-1, 2:] - 4.0 * u_full[:, :, 1:-1, 1:-1]) / h**2
    pde_residual = d2u[:,:,:,:] - a_final.view(B, 1, N, N)[:,:,1:-1,1:-1] + u_final.view(B, 1, N-2, N-2)   # (B, 1, 126, 126)
    pde_loss_fd = torch.sum(pde_residual.squeeze(1)**2, dim=(1, 2))
    
    for b in range(relative_error_u.shape[0]):
        print(f"batch:{b} relative_error_u:{relative_error_u[b]:.6e} "
             f"relative_error_a:{relative_error_a[b]:.6e} "
             f"relative_error_u_obs:{relative_error_u_obs[b]:.6e} "
             f"relative_error_a_obs:{relative_error_a_obs[b]:.6e} "
             f"L_pde:{L_pde[b]:.6e} L_obs_u:{L_obs_u[b]:.6e} L_obs_a:{L_obs_a[b]:.6e} "
             f"loss_fourier:{pde_loss[b]:.6e} loss_real:{pde_loss[b]:.6e} loss_fd:{pde_loss_fd[b]:.6e} ")
        
    print(f'Coefficients obs_a: {zeta_obs_a} obs_u: {zeta_obs_u} pde: {zeta_pde}')
    
    a_final = a_final.detach().cpu().numpy()
    u_final = u_final.detach().cpu().numpy()
    a_GT_np = a_GT.detach().cpu().numpy()
    u_GT_np = u_GT.detach().cpu().numpy()
    u_hat_final = u_hat_final.detach().cpu().numpy()
    a_hat_final = a_hat_final.detach().cpu().numpy()
    known_index_a_np =  known_index_a.detach().cpu().numpy()
    known_index_u_np =  known_index_u.detach().cpu().numpy()
    scipy.io.savemat(f'helmholtz_results_{offset}_obs_u{int(np.sum(known_index_u_np))}_obs_a{int(np.sum(known_index_a_np))}.mat',
                     {'a': a_final, 'u': u_final,'u_real': u_GT_np, 'a_real': a_GT_np, 'u_hat': u_hat_final,
                      'a_hat': a_hat_final, 'known_u': known_index_u_np, 'known_a': known_index_a_np,
                      'error_history_u': error_history_u,'error_history_a': error_history_a,
                      'state_list':state_list
                      })
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
