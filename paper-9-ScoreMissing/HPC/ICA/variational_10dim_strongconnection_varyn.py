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
    "VarySampleSize",
)
os.makedirs(save_path, exist_ok=True)
nrep = int(sys.argv[1])
miss_iter = int(sys.argv[2])
sample_sizes = torch.tensor([400, 600, 800, 1000, 1200, 1400])

dim = 10
miss_prob = 0.5
sample_size = sample_sizes[miss_iter].item()


training_args = {
    "nepochs": 1000, "niters": 1000,
    "snapshot_freq": 10, "min_loss_val": -1e3,
    "true_loss": True}
# Construct lists to store results
True_Thetas = []
Thetas = []
Losses = []
True_Losses = []

for i in tqdm(range(nrep)):
    temp_cov = create_strong_cov(dim)
    true_theta = torch.inverse(temp_cov)
    with HiddenPrints(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Generate corrupted data
        X, sampler = sample_ica(true_theta, sample_size, thin=5, burn=100)
    # Create mask
    mask = torch.bernoulli(torch.ones_like(X) * (1 - miss_prob))
    X_corrupted = X * mask

    # Run MLMC with random start. If model errors randomly choose new starting point
    for attempt in range(100):
        q_theta = ICACholeskyModel(dim, torch.eye(dim))
        theta_optimizer = optim.Adam(q_theta.parameters(), lr=.01)
        theta_scheduler1 = lr_scheduler.StepLR(theta_optimizer, step_size=100, gamma=0.75)
        theta_scheduler2 = lr_scheduler.LinearLR(theta_optimizer, 0.1, 1, 200)
        theta_scheduler = lr_scheduler.ChainedScheduler([theta_scheduler1, theta_scheduler2])
        p_phi = var_models.VariationalMLPConstVarNormal(
            dim=dim, std_lb=0.1, hidden_dims=[40, 40])
        phi_optimizer = optim.Adam(p_phi.parameters(), lr=.01)

        runner = runners.BiLevelMarginal(
            q_theta, p_phi, theta_optimizer, phi_optimizer,
            n_phi_step=5, do_iw=False, ncopies=10, inner_loss="kl")

        runner.init_dataset(X_corrupted, mask, batch_size=sample_size)
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
            True_Losses.append(runner.stored_vals['True_Losses'][-1])
            break
        except NaNModel:
            if attempt == 99:
                print(f"Failed {attempt} times")

    # Save results using torch.save
    torch.save({'True Thetas': True_Thetas, 'Est Thetas': Thetas,
                'Losses': Losses, 'Sample_sizes': sample_sizes},
               os.path.join(save_path, f'variational_varyn_n={sample_size}.pt'))
