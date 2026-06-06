import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys
import os
import warnings
from experiment_utils import ICACholeskyModel, sample_ica, create_strong_cov, HiddenPrints

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from MSM.models import variational_models as var_models  # noqa: E402
import MSM.runners as runners  # noqa: E402
from MSM.utils import NaNModel  # noqa: E402

save_path = os.path.join(
    os.path.pardir,
    os.path.pardir,
    "simulated_data",
    "ICA",
    "10Dim_StrongConnection",
    "VaryDim",
)
os.makedirs(save_path, exist_ok=True)
nrep = int(sys.argv[1])
miss_iter = int(sys.argv[2])
dims = torch.tensor([10, 20, 30, 40, 50])

dim = dims[miss_iter].item()
sample_size = 1000
miss_prob = 0.5


training_args = {
    "nepochs": 1000, "niters": 1000,
    "snapshot_freq": 10, "min_loss_val": -1e3,
    "true_loss": True}
# Construct lists to store results
True_Thetas = []
Thetas = []
Losses = []

for i in tqdm(range(nrep)):
    temp_cov = create_strong_cov(dim)
    true_theta = torch.inverse(temp_cov)
    with HiddenPrints(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Generate corrupted data
        X, sampler = sample_ica(true_theta, sample_size, thin=5, burn=100,
                                use_map=True, epsilon=0.2)
    # Create mask
    mask = torch.bernoulli(torch.ones_like(X) * (1 - miss_prob))
    X_corrupted = X * mask

    # Run MLMC with random start. If model errors randomly choose new starting point
    for attempt in range(100):
        q_theta = ICACholeskyModel(dim, torch.eye(dim))
        theta_optimizer = optim.Adam(q_theta.parameters(), lr=.1)
        theta_scheduler = lr_scheduler.StepLR(theta_optimizer, step_size=100, gamma=0.75)
        p_phi = var_models.ConstantImputer(0.)

        runner = runners.ImputingZerodScoreMatchingRunner(
            q_theta, p_phi, theta_optimizer)

        runner.init_dataset(X_corrupted, mask, batch_size=100)
        runner.set_schedulers(theta_scheduler=theta_scheduler)
        try:
            runner.train(**training_args)
            # If loss too large run again
            if runner.stored_vals['Losses'][-1] > 1e2:
                raise NaNModel
            # Store results in lists
            Thetas.append(q_theta.Theta.detach().clone())
            Losses.append(runner.stored_vals['Losses'][-1])
            True_Thetas.append(true_theta)
            break
        except NaNModel:
            if attempt == 99:
                print(f"Failed {attempt} times")

    # Save results using torch.save
    torch.save({'True Thetas': True_Thetas, 'Est Thetas': Thetas,
                'Losses': Losses, 'Dimensions': dims},
               os.path.join(save_path, f'missdiff_varydim_d={dim}.pt'))
