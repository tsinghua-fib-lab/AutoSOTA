# Make a fully abstracted training SM algorithm

# Libraries
import torch
from functools import partial
from .dsm import compute_loss as compute_loss_dsm
from .edm import compute_loss as compute_loss_edm
from .tsm import compute_loss as compute_loss_tsm
from tqdm import tqdm


def train_sm(dataset, loss_name, time_sampler, sde, net, lr, batch_size, n_epochs, target_score=None,
             exact_scores=None, data_var_scalar=None, antithetic=True, compute_at_T=True, is_particles=False):
    """Train using the EM loss

    Args:
        * dataset (torch.Tensor of shape (dataset_size, *data_shape)): Dataset
        * loss_name (str): Name of the loss function ('dsm', 'edm' or 'tsm')
        * time_sampler (TimeSampler): TimeSampler object
        * sde (LinearSDE): SDE
        * net (torch.nn.Module): Energy-based model
        * lr (float): Learning rate
        * batch_size (int): Batch size
        * n_epochs (int): Number of epochs
        * target_score (function): Score of the target distribution
            (default is None) (only used if loss_name == 'tsm')
        * data_var_scalar (float): Scalar variance of the data (default is None)
            Used if loss_name == 'edm' or 'dsm'. Make sure to have EDM param for denoiser
            in both cases.
        * antithetic (bool): Whether to use the antithetic trick (default is True)
        * compute_at_T (bool): Whether to compute the loss at time T (default is True)
        * is_particles (bool): Whether it is a particle system (default is False)
    """
    # Build the loss function
    if loss_name == 'dsm':
        loss_fn_ = partial(compute_loss_dsm, net=net, sde=sde, antithetic=antithetic)
        if data_var_scalar is not None:
            def loss_fn(ts, data):
                sigma_sq_ = sde.sigma_sq(ts)
                weights = ((sigma_sq_ + data_var_scalar) / (sigma_sq_ * data_var_scalar)).flatten()
                return weights * loss_fn_(ts, data)
        else:
            loss_fn = loss_fn_
    elif loss_name == 'edm':
        loss_fn = partial(compute_loss_edm, net=net, sde=sde,
                          data_var_scalar=data_var_scalar, antithetic=antithetic)
    elif loss_name == 'tsm':
        loss_fn = partial(compute_loss_tsm, net=net, sde=sde, target_score=target_score,
                          antithetic=antithetic)
    else:
        raise NotImplementedError('Loss {} is not implemented.'.format(loss_name))
    # Get the shape of the data
    data_shape_ones = (1,) * (len(dataset.shape)-1)
    # Build an optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Build the dataset
    dataset = torch.utils.data.TensorDataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, drop_last=True)
    # Run the training loop
    for epoch in range(n_epochs):
        loop = tqdm(dataloader, leave=True)
        loop.set_description(f"Epoch {epoch+1}/{n_epochs}")
        for data in loop:
            optimizer.zero_grad()
            ts = time_sampler.sample((data[0].shape[0],), exclude_last_level=not compute_at_T)
            ts = ts.view((-1, *data_shape_ones))
            loss = loss_fn(ts, data[0], is_particles=is_particles)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
