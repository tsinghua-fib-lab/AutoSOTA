import torch
import os
import time
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train
from tabsyn.diffusion_utils import EvaluateError, DiscreteError

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def estimate_discrete_error(args):

    seed_everything(args.seed)
    device = args.device

    train_z, _, _, ckpt_path, _ = get_input_train(args)

    if not os.path.exists(f"{ckpt_path}/model_{args.train_diffusion_model_class}.pt"):
        print("can't find model")

    in_dim = train_z.shape[1] 
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4,
    )

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{ckpt_path}/model_{args.train_diffusion_model_class}.pt"))

    model.eval()
    start_time = time.time()
        
    pbar = tqdm(train_loader, total=len(train_loader))

    all_error, all_sigma = [], []
    sampled_times = 32 
    num_steps = 10

    error_fn = DiscreteError(P_mean=model.loss_fn.P_mean,
                            P_std=model.loss_fn.P_std, sigma_data=model.loss_fn.sigma_data,
                            hid_dim=model.loss_fn.hid_dim, gamma=model.loss_fn.gamma,
                            opts=model.loss_fn.opts)

    sigma_list = sample_discrete_timesteps(model.denoise_fn_D, num_steps=num_steps)
    sigma_list = sigma_list.to(device)

    with torch.no_grad():
        all_error, all_sigma = [], []
        for batch in pbar:
            inputs = batch.float().to(device)

            this_error_with_different_sigma = []
            this_different_sigma = []
            for sampled_time in range(sampled_times):
                error_with_different_sigma, different_sigma = [], []
                for index in range(sigma_list.shape[0]):
                    current_sigma = sigma_list[index]
                    error, sigma = error_fn(model.denoise_fn_D, inputs, current_sigma=current_sigma)
                    error_with_different_sigma.append(error)
                    different_sigma.append(sigma)
                error_with_different_sigma = torch.stack(error_with_different_sigma, dim=1) # (4096, 10, 56)
                different_sigma = torch.stack(different_sigma, dim=1) # (4096, 10)

                this_error_with_different_sigma.append(error_with_different_sigma)
                this_different_sigma.append(different_sigma)

            this_error_with_different_sigma = torch.stack(this_error_with_different_sigma, dim=2) # (4096, 10, sampled_times, 56)
            this_different_sigma = torch.stack(this_different_sigma, dim=2) # (4096, 10, sampled_times)

            # print(this_error_with_different_sigma.shape, this_different_sigma.shape)
            all_error.append(this_error_with_different_sigma)
            all_sigma.append(this_different_sigma)

    all_error = torch.cat(all_error)
    all_sigma = torch.cat(all_sigma)
    store_path = f"./weights_discrete_{num_steps}_{sampled_times}/{args.dataname}" # _{num_steps}_{sampled_times}
    os.makedirs(store_path, exist_ok=True)
    torch.save(all_error, f"{store_path}/unweight_error_{args.train_diffusion_model_class}.pt")
    torch.save(all_sigma, f"{store_path}/sigma_{args.train_diffusion_model_class}.pt")
    
    end_time = time.time()
    print('Time: ', end_time - start_time)


SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

def sample_discrete_timesteps(net, num_steps = 50):

    step_indices = torch.arange(num_steps, dtype=torch.float32)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = net.round_sigma(t_steps)

    return t_steps



