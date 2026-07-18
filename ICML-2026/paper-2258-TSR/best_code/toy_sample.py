import os
import argparse

import matplotlib.pyplot as plt
import torch

from toy_train import load_model_and_data
from TSR_toy.sampler.toy_samplers import * # T, beta, alpha, alpha_bar
from utils.plot_utils import  plot_sample_points


def main():
    parser = argparse.ArgumentParser(description="Training diffusion/flow model for Toy data")
    parser.add_argument("--data", type=str, 
                        choices=["checkerboard", "swissroll"],
                        required=True, help="Type of Supported Toy Data")
    
    parser.add_argument("--save_dir", type=str, 
                        default="./output",
                        help="Directory for saved ckpts and results")
    
    parser.add_argument("--model_type", type=str, 
                        choices=["flow", "diffusion"],
                        help="Type of Supported Models")
    
    parser.add_argument("--k", type=float, 
                        default=10.0,
                        help="The value controls the sampling tempreture. k>1.0 is peaker distribution.")
    
    parser.add_argument("--sigma", type=float, 
                        default=0.1,
                        help="The value controls how early TSR steer the sampling process.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0")
    
    ckpt_folder = os.path.join(args.save_dir, f"{args.data}_{args.model_type}")
    config, model, data_dict, _, _ = load_model_and_data(ckpt_folder, device)
    data = data_dict["data_tensor"]
    data_samples = data_dict["data_samples_tensor"]
    xy_lims = data_dict["xy_lims"]
    dataset_name = data_dict["dataset_name"]
    pred_type = config['pred_type']
   
    
    k_val = args.k
    sigma_val = args.sigma

    if pred_type == 'flow':
        sampler_configs =[
        {"name": f"Euler-ode, k=1", "step_fn": flow_ode_step, "k" : 1.0, "inf_steps" : 1000, "scale_xT": False},
        {"name": f"CNS-sde, k={k_val}", "step_fn": CNS_flow_sde_step, "k" : k_val, "inf_steps" : 1000,  "scale_xT": True},
        {"name": f"TSR-ode, k={k_val}, sigma={sigma_val}", "step_fn": TSR_flow_ode_step, "k" : k_val, "sigma": sigma_val, "inf_steps" : 1000, "scale_xT": False},
    ]
    else:
        sampler_configs =[
            {"name": f"DDPM, k=1", "step_fn": ddpm_step, "k" : 1.0, "inf_steps" : 1000, "scale_xT": False},
            {"name": f"CNS, k={k_val}", "step_fn": CNS_step, "k" : k_val, "inf_steps" : 1000, "scale_xT": True},
            {"name": f"TSR, k={k_val}, sigma={sigma_val}", "step_fn": TSR_ddpm_step, "k" : k_val, "inf_steps" : 1000, "sigma": sigma_val, "scale_xT": False},
        ]

    
    num_samples = 10000
    #Plot Settings
    plt.rcParams.update({'font.size': 16})

    cols = 4
    num_subplots = len(sampler_configs) + 1
    rows = (num_subplots + cols - 1) // cols  # Round up to fit all subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4.4), squeeze=True)
    axs = axs.flatten()

    # First plot the data samples
    data_samples = data[:num_samples]
    plot_sample_points(data_samples, ax=axs[0], xy_lims=xy_lims, s=1, alpha=0.5, color='blue')
    axs[0].set_title("Data sample")

    x_T_common = torch.randn(num_samples, 2, device=device)
    for sid, sampler_config in enumerate(sampler_configs):
        step_fn = sampler_config["step_fn"]
        num_inf_steps = sampler_config["inf_steps"]
        x_T = x_T_common / (k_val**0.5) if sampler_config["scale_xT"] else x_T_common

        if pred_type == 'flow':
            samples, _ = pc_flow_learned_sampling(x_T, model.forward,
                                                    num_inf_steps=num_inf_steps, step_fn=step_fn,
                                                    k=sampler_config["k"], sigma=sampler_config.get("sigma", 1.0))
        else:
            samples, _ = pc_learned_sampling(x_T, model.score_fn, T, 
                                            num_inf_steps=num_inf_steps, step_fn=step_fn, 
                                            k=sampler_config["k"], sigma=sampler_config.get("sigma", 1.0))
        
        plot_sample_points(samples, ax=axs[sid+1], xy_lims=xy_lims, s=1, alpha=0.5, color='blue')
        axs[sid+1].set_title(f"{sampler_config['name']}")

    for ax in axs.flat:
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
           
    fig.tight_layout()
    fig.savefig(f"{ckpt_folder}/samples_k={k_val}.png", bbox_inches='tight', dpi=300)
    
    
    

if __name__ == "__main__":
    main()


