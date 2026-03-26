# %%
from experiment_utils import my_g_weights
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

save_path = os.path.join(
    os.path.pardir,
    os.path.pardir,
    "simulated_data",
    "NormalEstimation",
    "10DimStrongConnectionTruncated",
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

    # Set-up Truncation
    trunc_boundary = torch.stack([temp_mean - 1.65*torch.sqrt(torch.diag(temp_cov)),
                                  temp_mean + 1.65*torch.sqrt(torch.diag(temp_cov))], dim=0)
    dims_to_trunc = 3
    trunc_boundary_inf = trunc_boundary.clone()
    trunc_boundary_inf[0, dims_to_trunc:] = -torch.inf*torch.ones(dim-dims_to_trunc)
    trunc_boundary = trunc_boundary[..., :dims_to_trunc]

    def truncfunc(x: torch.Tensor):
        distances = x[..., :dims_to_trunc]-trunc_boundary[0]
        capped_distances = my_g_weights(
            distances, 0.5)
        return torch.cat((capped_distances, torch.ones((*x.shape[:-1], dim-dims_to_trunc))), dim=-1)

    def truncfunc_masked(x: torch.Tensor, mask: torch.Tensor):
        return truncfunc(x)*(mask)

    def boundary_func(x: torch.Tensor):
        logics = utils.my_all(x[..., :dims_to_trunc] > trunc_boundary[0], dim=-1)
        return logics

    # Generate corrupted data
    X_untrunc = our_normal.sample((sample_size*10,))
    X = X_untrunc[boundary_func(X_untrunc)][:sample_size]
    mask = torch.bernoulli(torch.ones_like(X)-missing_prob)
    X_corrupted = X*mask

    # Run MLMC with random start. If model errors randomly choose new starting point
    for attempt in range(1000):
        start_mean = torch.randn(10)
        start_cov_sqrt = torch.randn(10, 10)
        start_prec = torch.inverse(start_cov_sqrt.T @ start_cov_sqrt)
        q_theta = models.NormalDensity(
            mean=start_mean, Precision=start_prec, boundary_func=boundary_func)
        theta_opt = torch.optim.Adam(q_theta.parameters(), lr=0.01)
        zero_imputer = var_models.ConstantImputer(0.)
        runner = runners.TruncatedZerodImputingScoreMatchingRunner(
            q_theta, zero_imputer, theta_opt, trunc_func=truncfunc_masked, elementwise_trunc=True)

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

    # Save results using torch.save
    torch.save({'Means': Means, 'Precisions': Precisions, 'Losses': Losses,
                'Sample sizes': sample_sizes, 'True Mean': True_Means, "True Covariance": True_Covs},
               os.path.join(save_path, f'missdiff_varyn_n={sample_size}.pt'))
