import os
import sys

# Get the directory one level above the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

import argparse
import torch

from sbi.inference import SNLE
from sbi.utils import BoxUniform

def load_data(theta_np, x_np, device="cpu"):
        
    theta = torch.from_numpy(theta_np).to(device).float()
    x = torch.from_numpy(x_np).to(device).float()

    return theta, x

def train_snle(
    theta: torch.Tensor,
    x: torch.Tensor,
    prior,
    device: str = "cpu",
    max_num_epochs: int = 1000,
    batch_size: int = 256,
):
    # Create SNLE inference object
    inference = SNLE(prior=prior) #.to(device)

    # Append simulations
    inference = inference.append_simulations(theta, x)

    # Train likelihood estimator
    likelihood_estimator = inference.train(
        training_batch_size=batch_size,
        max_num_epochs=max_num_epochs
    )

    # Build posterior (likelihood × prior)
    return likelihood_estimator

def sample_likelihood(estimator, theta, xsize, device="cpu"):
    # Ensure theta has batch dimension
    if theta.ndim == 1:
        theta = theta.unsqueeze(0).to(device)

    x_samples = torch.zeros(theta.shape[0],xsize)
    with torch.no_grad():
        for i in range(theta.shape[0]):
            th = theta[i,:].unsqueeze(0)
            x_sample = estimator.sample(
                (1,),
                condition=th
            )
            x_samples[i,:] = x_sample

    return x_samples

def train_and_sample(theta_np,x_np,Nsamples=10,epochs=1000,batch_size=200):

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # 1) Load data
    theta, x = load_data(theta_np, x_np, device=device)

    # 2) Build prior - use a dummy prior
    theta_min = theta.min(dim=0).values
    theta_max = theta.max(dim=0).values
    dummy_prior = BoxUniform(low=theta_min, high=theta_max)

    # 3) Train SNLE
    likelihood_estimator = train_snle(
        theta=theta,
        x=x,
        prior=dummy_prior,
        device=device,
        max_num_epochs=epochs,
        batch_size=batch_size,
    )
    
    # Sample some thetas from uniform prior
    theta_o = dummy_prior.sample((Nsamples,))

    # 4) Sample from posterior
    samples = sample_likelihood(
        estimator=likelihood_estimator,
        theta=theta_o,
        xsize=x.shape[1],
        device=device,
    )

    return samples.detach().cpu().numpy()