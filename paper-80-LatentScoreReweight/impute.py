import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import warnings

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
import json
import sys
from utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation for the Target Column')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
else:
    args.device = 'cpu'

class_labels=None
randn_like=torch.randn_like

SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1


## One denoising step from t to t-1
def step(net, num_steps, i, t_cur, t_next, x_next):

    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur) 
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    # Euler step.

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


if __name__ == '__main__':
    dataname = args.dataname
    device = args.device
    epoch = args.epoch
    mask_cols = args.cols = [0]
    
    num_trials = 1
    
    data_dir = f'data/{dataname}'

    d_token = 4
    token_bias = True
    device =  args.device
    n_head = 1
    factor = 32

    num_layers = 2

    info_path = f'data/{dataname}/info.json'
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    
    task_type = info['task_type']

    ckpt_dir = f'tabsyn/vae/ckpt/{dataname}' 
    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    for trial in range(50):

        X_num, X_cat, categories, d_numerical = preprocess(data_dir, task_type = info['task_type'])


        X_train_num, X_test_num = X_num
        X_train_cat, X_test_cat = X_cat

        X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
        X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)
        
        mask_idx = 0

        if task_type == 'bin_class':
            unique_values, counts = np.unique(X_train_cat[:, mask_idx], return_counts=True)
            sampled_cat = np.random.choice(unique_values, size=1, p=counts / counts.sum())

            # Replacing the target column with the sampled class
            X_train_cat[:, mask_idx] = torch.tensor(unique_values[sampled_cat[0]]).long()
            X_test_cat[:, mask_idx] = torch.tensor(unique_values[sampled_cat[0]]).long()

        else:
            avg = X_train_num[:, mask_idx].mean(0)
            
            X_train_num[:, mask_idx] = avg
            X_test_num[:, mask_idx] = avg


        model = Model_VAE(num_layers, d_numerical, categories, d_token, n_head = n_head, factor = factor, bias = True)
        model = model.to(device)

        model.load_state_dict(torch.load(f'{ckpt_dir}/model.pt'))

        pre_encoder = Encoder_model(num_layers, d_numerical, categories, d_token, n_head = n_head, factor = factor).to(device)
        pre_decoder = Decoder_model(num_layers, d_numerical, categories, d_token, n_head = n_head, factor = factor)

        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)
        
        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        X_test_num = X_test_num.to(device)
        X_test_cat = X_test_cat.to(device)

        x = pre_encoder(X_test_num, X_test_cat).detach().cpu().numpy()

        embedding_save_path = f'tabsyn/vae/ckpt/{dataname}/train_z.npy'
        train_z = torch.tensor(np.load(embedding_save_path)).float()
        train_z = train_z[:, 1:, :]

        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim

        train_z = train_z.view(B, in_dim)
        mean, std = train_z.mean(0), train_z.std(0)

        x = torch.tensor(x[:, 1:, :]).view(-1, in_dim)

        x = ((x-mean)/2).to(device)    

        denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
        model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

        model.load_state_dict(torch.load(f'tabsyn/ckpt/{dataname}/model.pt'))

        # Define the masking area

        mask_idx = [0]
        if task_type == 'bin_class':
            mask_idx += d_numerical
        
        mask_list = [list(range(i*token_dim, (i+1)*token_dim)) for i in mask_idx]
        mask = torch.zeros(num_tokens * token_dim, dtype=torch.bool)
        mask[mask_list] = True

        ###########################

        num_steps = 50
        N = 20
        net = model.denoise_fn_D

        num_samples, dim = x.shape[0], x.shape[1]
        x_t = torch.randn([num_samples, dim], device='cuda')

        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_t.device)

        sigma_min = max(SIGMA_MIN, net.sigma_min)
        sigma_max = min(SIGMA_MAX, net.sigma_max)

        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        mask = mask.to(torch.int).to(device)
        x_t = x_t.to(torch.float32) * t_steps[0]

        with torch.no_grad():

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                print(i)
                if i < num_steps - 1:
                    for j in range(N):
                        n_curr = torch.randn_like(x).to(device) * t_cur
                        n_prev = torch.randn_like(x).to(device) * t_next

                        x_known_t_prev = x + n_prev
                        x_unknown_t_prev = step(net, num_steps, i, t_cur, t_next, x_t)

                        x_t_prev = x_known_t_prev * (1-mask) + x_unknown_t_prev * mask

                        n = torch.randn_like(x) * (t_cur.pow(2) - t_next.pow(2)).sqrt()

                        if j == N - 1:
                            x_t = x_t_prev                                                # turn to x_{t-1}
                        else:
                            x_t = x_t_prev + n                                            # new x_t

        _, _ , _, _, num_inverse, cat_inverse = preprocess(data_dir, task_type = info['task_type'], inverse = True)
        x_t = x_t * 2 + mean.to(device)

        info['pre_decoder'] = pre_decoder
        info['token_dim'] = token_dim


        syn_data = x_t.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns = idx_name_mapping, inplace=True)

        save_dir = f'impute/{dataname}'
        os.makedirs(save_dir) if not os.path.exists(save_dir) else None

        syn_df.to_csv(f'{save_dir}/{trial}.csv', index = False)