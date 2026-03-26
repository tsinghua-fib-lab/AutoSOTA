# %%
import os
import torch
from torch import distributions as dists
from torch.optim import lr_scheduler  # noqa: F401
from tqdm import tqdm
import sys
# Add parent directory of this file to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from MSM.models import density_models as models  # noqa: E402
from MSM.models import variational_models as var_models  # noqa: E402
from MSM import runners  # noqa: E402
from MSM.utils import NaNModel  # noqa: E402
from MSM.utils import data as utils  # noqa: E402
# %%


def custom_trunc(x):
    return utils.my_all((x > -10.) & (x < 10.), dim=-1)


save_path = os.path.join(
    os.path.pardir,
    os.path.pardir,
    "simulated_data",
    "NormalEstimation",
    "10DimStrongConnection",
    "VarySampleSize",
)
os.makedirs(save_path, exist_ok=True)

iter = int(sys.argv[2])
# ####### Experiment 1 #########
dim = 10
sub_dim = dim - 1
missing_prob = 0.2
nrep = int(sys.argv[1])
# Create additional covariance
A = torch.tensor([[1., 0.], [0.5, 0.5]])
nattempts = 100
sample_sizes = torch.tensor([200, 400, 600, 800, 1000])
sample_size = sample_sizes[iter]
print(sample_size)

training_args = {
    "nepochs": 10000,
    "niters": 3000,
    "snapshot_freq": 10,
    "min_loss_val": -1e3,
    "max_loss_val": 1e3,
    "true_loss": False,
}
# Construct lists to store results
Means = []
Precisions = []
Losses = []
True_Means = []
True_Covs = []

for i in tqdm(range(nrep)):
    temp_mean = torch.zeros(dim)+0.5
    sub_dim = dim - 1
    # Create sub covariance
    orthonormal = torch.linalg.qr(torch.randn(sub_dim, sub_dim))[0]
    sub_cov = orthonormal @ torch.diag(torch.rand(sub_dim) + 0.5) @ orthonormal.T
    # Create additional covariance
    sub_cov2 = torch.zeros((dim, dim))
    sub_cov2[:sub_dim, :][:, :sub_dim] = sub_cov
    sub_cov2[sub_dim, sub_dim] = sub_cov[0, 0]
    # Create relationship matrix
    A = torch.eye(dim)
    A[sub_dim, 0] = 0.5
    A[sub_dim, sub_dim] = 0.5
    # Construct full covariance
    temp_cov = A @ sub_cov2 @ A.T

    our_normal = dists.MultivariateNormal(temp_mean, temp_cov)
    True_Means.append(temp_mean)
    True_Covs.append(temp_cov)

    # Generate corrupted data
    X = our_normal.sample((sample_size,))

    # Create mask
    mask = torch.bernoulli(torch.ones_like(X) * (1 - missing_prob))
    X_corrupted = X * mask

    # Run MLMC with random start. If model errors randomly choose new starting point
    for attempt in range(1000):
        start_mean = torch.randn(10)
        start_cov_sqrt = torch.randn(10, 10)
        start_prec = torch.inverse(start_cov_sqrt.T @ start_cov_sqrt)
        q_theta = models.NormalDensity(
            mean=start_mean, Precision=start_prec)
        theta_opt = torch.optim.Adam(
            [{'params': q_theta.mean, 'lr': 0.1},
             {'params': q_theta.Precision, 'lr': 0.01}])

        p_phi = var_models.VariationalMLPTruncConstVarNormal(
            dim, hidden_dims=[20, 20], boundary_func=custom_trunc,
            std_lb=0.1, std_increase=0.1)
        phi_optimizer = torch.optim.Adam(p_phi.parameters(), lr=0.01)

        runner = runners.BiLevelMarginal(
            q_theta, p_phi, theta_opt, phi_optimizer,
            n_phi_step=2, do_iw=False, ncopies=10)

        runner.init_dataset(X_corrupted, mask, batch_size=100)
        runner.set_grad_opts(control_method="clip", max_norm=0.5, norm_type=2)
        try:
            runner.train(**training_args)
            # If loss too large run again
            if runner.stored_vals['Losses'][-1] > 1e2:
                raise NaNModel
            # Store results in lists
            Means.append(q_theta.mean.detach().clone())
            Precisions.append(q_theta.Precision.detach().clone())
            Losses.append(runner.stored_vals['Losses'][-1])
            break
        except NaNModel:
            if attempt == 999:
                print(f"Failed {attempt} times")
        except IndexError:
            if attempt == 999:
                print(f"Failed {attempt} times")

    # Save results using torch.save
    torch.save({'Means': Means, 'Precisions': Precisions, 'Losses': Losses,
                'Sample sizes': sample_sizes, 'True Mean': True_Means, "True Covariance": True_Covs},
               os.path.join(save_path, f'variational_varyn_n={sample_size}.pt'))
