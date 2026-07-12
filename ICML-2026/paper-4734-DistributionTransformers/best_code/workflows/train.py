"""
Main training routine for distribution transformer models
"""
from collections import defaultdict
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except (ImportError, ModuleNotFoundError):
    from contextlib import nullcontext as sdpa_kernel
    from torch.backends.cuda import SDPBackend

from tqdm import tqdm
import time
import itertools
from typing import TypedDict, Optional
from typing_extensions import NotRequired
from os import makedirs
from pathlib import Path

from model.distribution_transformer import DistributionTransformer
from distributions.distributions import CompleteDistribution, GaussianMixtureModel
from distributions.utils import decode_gmm_sample
from workflows.utils import get_openai_lr, get_cosine_schedule_with_warmup
from competitor_methods.pfns import PFN, get_borders_from_prior, RiemannDistribution
from competitor_methods.ace import ACEBaseTransformer, convert_complete_distribution_to_ace_sampler


class TrainKwargs(TypedDict):
    epochs: NotRequired[int]
    warmup_epochs: NotRequired[int]
    steps_per_epoch: NotRequired[int]
    batch_size: NotRequired[int]
    lr: NotRequired[Optional[float]]
    weight_decay: NotRequired[float]
    scheduler: NotRequired[Optional[type[LRScheduler]]]
    gpu_device: NotRequired[str]
    verbose: NotRequired[bool]
    progress_bar: NotRequired[bool]
    save_interval: NotRequired[int]
    save_loss_series: NotRequired[bool]


def train(model: DistributionTransformer,
          complete_distribution: CompleteDistribution,
          epochs: int = 100,
          warmup_epochs: int = 10,
          steps_per_epoch: int = 100,
          batch_size: int = 1000,
          lr: Optional[float] = None,
          weight_decay: float = 0.01,
          max_grad: float = 10000.,
          scheduler: Optional[type[LRScheduler]] = None,
          gpu_device: str = "cuda:0",
          verbose: bool = True,
          progress_bar: bool = True,
          save_interval: Optional[int] = -1,
          save_loss_series: bool = True,
          print_marginals: bool = True,
          use_prior_loss: bool = True,
          _run=None
          ) -> tuple[DistributionTransformer, dict]:
    """
    Training routine for variational transformers. Note that a validation set is not necessary here as the model will
    see each datapoint only once, so overfitting is impossible and a reduction in train error corresponds to an
    improvement to the model.

    Args:
        model: Model to train.
        complete_distribution: Complete data distribution to sample from.
        epochs: Number of epochs.
            Defaults to 100.
        warmup_epochs: Number of epochs for warmup phase of lr scheduler.
            Defaults to 10.
        steps_per_epoch: Number of batches per epoch.
            Defaults to 100.
        batch_size: Number of samples per batch.
            Defaults to 200.
        lr: Learning rate.
            Defaults to the OpenAI method of determining learning rate.
        weight_decay: Weight decay.
            Defaults to 0.01.
        max_grad: Maximum absolute value of gradients during backpropagation.
            Defaults to 10000.
        scheduler: Learning rate scheduler.
            Defaults to cosine annealing with warmup.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        verbose: Whether to display metrics during training.
            Defaults to False.
        progress_bar: Whether to display a progress bar during training.
            Defaults to False.
        save_interval: How frequently to save model weights in units of epochs.
            Defaults to -1, ie save last epoch only.
        save_loss_series: Whether to save loss series.
            Defaults to True.
        _run: Sacred run object.

    Returns:
        Trained model.

    """
    before_training = time.time()
    assert save_interval == -1 or save_interval is None or (save_interval > 0 and isinstance(save_interval, int)), \
        "save_interval must be None (for no saving), -1 (for final epoch saving only) or a strictly positive int"

    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    device = torch.device(device)
    model.to(device)

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -max_grad, max_grad))

    scale_parametrisation = model.component_embedding.scale_parametrisation

    if lr is None:
        lr = get_openai_lr(model) if lr is None else lr
        print(f"Using OpenAI max lr of {lr}")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)\
        if scheduler is None else scheduler(optimizer)

    prior_loss_series = []
    posterior_loss_series = []

    def train_epoch() -> dict:
        model.train()
        total_posterior_loss = 0.
        total_prior_loss = 0.
        total_forward_time = 0.
        total_step_time = 0.
        epoch_prior_loss_series = []
        epoch_posterior_loss_series = []
        marginal_losses = defaultdict(list)
        tqdm_iter = tqdm(range(steps_per_epoch), desc='Training Epoch') if progress_bar else None

        before_get_batch = time.time()

        successful_sample_flag = False
        sample_attempts = 0
        while not successful_sample_flag and sample_attempts < 10:
            try:
                complete_data_sample = complete_distribution.sample((steps_per_epoch, batch_size))
                successful_sample_flag = True
            except Exception as e:
                if sample_attempts >= 10:
                    raise e
                else:
                    sample_attempts += 1

        z = complete_data_sample[2]
        time_to_get_epoch = time.time() - before_get_batch

        for batch, (phi, x) in enumerate(zip(*complete_data_sample[:2])):
            tqdm_iter.update() if tqdm_iter is not None else None
            observations = {key: obs[batch].to(device) for key, obs in z.items()}
            before_forward = time.time()
            phi_in, phi_out = model(phi.to(device), **observations)
            forward_time = time.time() - before_forward
            targets = x.reshape(batch_size, decode_gmm_sample(phi_in, scale_parametrisation)["loc"].shape[-1]
                                ).to(device)

            prior_losses = -GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation)
                                                 ).log_prob(model.sample_space_transform(targets))
            prior_loss = torch.nanmean(prior_losses)

            posterior_losses = -GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation)
                                                     ).log_prob(model.sample_space_transform(targets))
            posterior_loss = torch.nanmean(posterior_losses)
            
            if print_marginals:
                state_size = int((phi_out.shape[-1])**0.5)
                weight = phi_out[:,:,0]
                loc = phi_out[..., 1:state_size+1]
                scale = phi_out[..., -state_size ** 2:].reshape(*phi_out.shape[:-1], state_size, state_size)
                var = torch.diagonal(scale, dim1=-2, dim2=-1)
                
                for var_ix in range(var.shape[-1]):
                    marginal_phi_out = torch.stack([weight, loc[:,:,var_ix], var[:,:,var_ix]], dim=-1)
                    marginal_posterior_losses = -GaussianMixtureModel(**decode_gmm_sample(marginal_phi_out, scale_parametrisation)
                                                     ).log_prob(model.sample_space_transform(targets)[:, [var_ix]])
                    marginal_posterior_loss = torch.nanmean(marginal_posterior_losses)
                    marginal_losses[var_ix].append(marginal_posterior_loss)
            
            if not use_prior_loss:
                total_loss = posterior_loss
            else:
                total_loss = prior_loss + posterior_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            step_time = time.time() - before_forward

            total_prior_loss += prior_loss.cpu().item()
            total_posterior_loss += posterior_loss.cpu().item()
            total_forward_time += forward_time
            total_step_time += step_time
            epoch_prior_loss_series.append(prior_loss.cpu().item())
            epoch_posterior_loss_series.append(posterior_loss.cpu().item())

            if tqdm_iter:
                postfix_dict = {'step time': step_time,
                                'mean prior loss': total_prior_loss / (batch + 1),
                                'mean posterior loss': total_posterior_loss / (batch + 1)}
                tqdm_iter.set_postfix(postfix_dict)

        return {
            "mean_prior_loss": total_prior_loss / steps_per_epoch,
            "mean_posterior_loss": total_posterior_loss / steps_per_epoch,
            "epoch_load_time": time_to_get_epoch,
            "epoch_time": time.time() - before_get_batch,
            "mean_forward_time": total_forward_time / steps_per_epoch,
            "mean_step_time": total_step_time / steps_per_epoch,
            "epoch_prior_loss_series": epoch_prior_loss_series,
            "epoch_posterior_loss_series": epoch_posterior_loss_series,
            "marginal_posterior_loss": marginal_losses
        }

    for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
        epoch_start_time = time.time()
        try:
            with sdpa_kernel(SDPBackend.MATH):
                epoch_metrics = train_epoch()
        except Exception as e:
            print("Invalid epoch encountered, skipping...")
            raise e

        if save_interval is not None and save_interval != -1 and epoch % save_interval == 0 and _run is not None:
            path = Path(_run.observers[0].dir) / Path("state_dicts")
            makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), path / Path(f"model_state_dict_{epoch}.pt"))

        prior_loss_series.append(epoch_metrics["epoch_prior_loss_series"])
        posterior_loss_series.append(epoch_metrics["epoch_posterior_loss_series"])

        if verbose:
            print('\n' + '-' * 179)
            print(
                f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s '
                f'| lr {scheduler.get_last_lr()[0]:5.6f} '
                f'| data time {epoch_metrics["epoch_load_time"]:5.2f} | epoch time {epoch_metrics["epoch_time"]:5.2f} '
                f'| forward time {epoch_metrics["mean_forward_time"]:5.5f} '
                f'| step time {epoch_metrics["mean_step_time"]:5.2f} '
                f'| mean prior loss {epoch_metrics["mean_prior_loss"]:5.2f} '
                f'| mean posterior loss {epoch_metrics["mean_posterior_loss"]:5.2f} |')
        
            if print_marginals:
                for var_ix, marginal_loss in epoch_metrics["marginal_posterior_loss"].items():
                    print(f'| marginal mean posterior loss var {var_ix} {torch.mean(torch.stack(marginal_loss)).item():5.2f} |')
                    
            print('-' * 179)

        scheduler.step()

    if save_interval is not None and _run is not None:
        path = Path(_run.observers[0].dir) / Path("state_dicts")
        makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), path / Path("model_state_dict.pt"))

    if save_loss_series and _run is not None:
        _run.info["epoch_prior_loss_series"] = prior_loss_series
        _run.info["epoch_posterior_loss_series"] = posterior_loss_series

    if _run is not None:
        try:
            _run.info["model_training_time"].append(time.time() - before_training)
        except KeyError:
            _run.info["model_training_time"] = [time.time() - before_training]

    model.eval()
    return model.cpu(), epoch_metrics


def train_pfn(model: PFN,
              complete_distribution: CompleteDistribution,
              epochs: int = 100,
              warmup_epochs: int = 10,
              steps_per_epoch: int = 100,
              batch_size: int = 1000,
              lr: Optional[float] = None,
              weight_decay: float = 0.01,
              max_grad: float = 1000.,
              scheduler: Optional[type[LRScheduler]] = None,
              gpu_device: str = "cuda:0",
              verbose: bool = True,
              progress_bar: bool = True,
              save_interval: Optional[int] = -1,
              save_loss_series: bool = True,
              _run=None
              ) -> tuple[PFN, dict]:
    """
    Training routine for variational transformers. Note that a validation set is not necessary here as the model will
    see each datapoint only once, so overfitting is impossible and a reduction in train error corresponds to an
    improvement to the model.

    Args:
        model: Model to train.
        complete_distribution: Complete data distribution to sample from.
        epochs: Number of epochs.
            Defaults to 100.
        warmup_epochs: Number of epochs for warmup phase of lr scheduler.
            Defaults to 10.
        steps_per_epoch: Number of batches per epoch.
            Defaults to 100.
        batch_size: Number of samples per batch.
            Defaults to 200.
        lr: Learning rate.
            Defaults to the OpenAI method of determining learning rate.
        weight_decay: Weight decay.
            Defaults to 0.01.
        max_grad: Maximum absolute value of gradients during backpropagation.
            Defaults to 10000.
        scheduler: Learning rate scheduler.
            Defaults to cosine annealing with warmup.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        verbose: Whether to display metrics during training.
            Defaults to False.
        progress_bar: Whether to display a progress bar during training.
            Defaults to False.
        save_interval: How frequently to save model weights in units of epochs.
            Defaults to -1, ie save last epoch only.
        save_loss_series: Whether to save loss series.
            Defaults to True.
        _run: Sacred run object.

    Returns:
        Trained model.

    """
    before_training = time.time()
    assert save_interval == -1 or save_interval is None or (save_interval > 0 and isinstance(save_interval, int)), \
        "save_interval must be None (for no saving), -1 (for final epoch saving only) or a strictly positive int"

    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    device = torch.device(device)
    model.to(device)
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -max_grad, max_grad))

    if lr is None:
        lr = get_openai_lr(model) if lr is None else lr
        print(f"Using OpenAI max lr of {lr}")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)\
        if scheduler is None else scheduler(optimizer)

    posterior_loss_series = []

    def train_epoch(update_borders: bool = False) -> dict:
        model.train()
        total_posterior_loss = 0.
        total_forward_time = 0.
        total_step_time = 0.
        epoch_posterior_loss_series = []
        tqdm_iter = tqdm(range(steps_per_epoch), desc='Training Epoch') if progress_bar else None

        before_get_batch = time.time()
        complete_data_sample = complete_distribution.sample((steps_per_epoch, batch_size))

        z = complete_data_sample[2]
        time_to_get_epoch = time.time() - before_get_batch

        for batch, (phi, x) in enumerate(zip(*complete_data_sample[:2])):
            tqdm_iter.update() if tqdm_iter is not None else None
            observations = {key: obs[batch].to(device) for key, obs in z.items()}
            if update_borders and batch == 0:
                with torch.no_grad():
                    borders = get_borders_from_prior(complete_distribution.meta_prior.prior(
                        **complete_distribution.meta_prior.decode_sample(phi)), model.n_buckets, model.infinite_support,
                        model.leftmost_border, model.rightmost_border).mean(dim=0)
                    model.borders = borders.to(device)
            before_forward = time.time()

            phi_out = model(**observations)
            forward_time = time.time() - before_forward
            targets = x.reshape(batch_size).to(device)
            posterior_losses = -RiemannDistribution(phi_out, model.borders, model.infinite_support
                                                    ).log_prob(targets)
            posterior_loss = torch.nanmean(posterior_losses)
            optimizer.zero_grad()
            posterior_loss.backward()
            optimizer.step()
            step_time = time.time() - before_forward

            total_posterior_loss += posterior_loss.cpu().item()
            total_forward_time += forward_time
            total_step_time += step_time
            epoch_posterior_loss_series.append(posterior_loss.cpu().item())

            if tqdm_iter:
                postfix_dict = {'step time': step_time,
                                'mean posterior loss': total_posterior_loss / (batch + 1)}
                tqdm_iter.set_postfix(postfix_dict)

        return {
            "mean_posterior_loss": total_posterior_loss / steps_per_epoch,
            "epoch_load_time": time_to_get_epoch,
            "epoch_time": time.time() - before_get_batch,
            "mean_forward_time": total_forward_time / steps_per_epoch,
            "mean_step_time": total_step_time / steps_per_epoch,
            "epoch_posterior_loss_series": epoch_posterior_loss_series
        }

    epoch_metrics = {}

    for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
        epoch_start_time = time.time()
        epoch_metrics = train_epoch(epoch == 1)

        if save_interval is not None and save_interval != -1 and epoch % save_interval == 0 and _run is not None:
            path = _run.observers[0].dir+"\\state_dicts\\pfns\\"
            makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), path + f"pfn_state_dict_{epoch}.pt")

        posterior_loss_series.append(epoch_metrics["epoch_posterior_loss_series"])

        if verbose:
            print('\n' + '-' * 155)
            print(
                f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s '
                f'| lr {scheduler.get_last_lr()[0]:5.6f} '
                f'| data time {epoch_metrics["epoch_load_time"]:5.2f} | epoch time {epoch_metrics["epoch_time"]:5.2f} '
                f'| forward time {epoch_metrics["mean_forward_time"]:5.5f} '
                f'| step time {epoch_metrics["mean_step_time"]:5.2f} '
                f'| mean posterior loss {epoch_metrics["mean_posterior_loss"]:5.2f} |')
            print('-' * 155)

        scheduler.step()

    if save_interval is not None and _run is not None:
        path = _run.observers[0].dir + "\\state_dicts\\pfns\\"
        makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), _run.observers[0].dir + f"\\state_dicts\\pfns\\pfn_state_dict.pt")

    if save_loss_series and _run is not None:
        _run.info["pfn_epoch_posterior_loss_series"] = posterior_loss_series

    if _run is not None:
        try:
            _run.info["pfn_training_time"].append(time.time() - before_training)
        except KeyError:
            _run.info["pfn_training_time"] = [time.time() - before_training]

    model.eval()
    return model.cpu(), epoch_metrics


def train_ace(model: ACEBaseTransformer,
              complete_distribution: CompleteDistribution,
              epochs: int = 100,
              warmup_epochs: int = 10,
              steps_per_epoch: int = 100,
              batch_size: int = 1000,
              lr: Optional[float] = None,
              weight_decay: float = 0.01,
              max_grad: float = 1000.,
              scheduler: Optional[type[LRScheduler]] = None,
              gpu_device: str = "cuda:0",
              verbose: bool = True,
              progress_bar: bool = True,
              save_interval: Optional[int] = -1,
              save_loss_series: bool = True,
              sample_space_transform=None,
              randomise_target=False,
              _run=None
              ) -> tuple[ACEBaseTransformer, dict]:
    """
    Training routine for variational transformers. Note that a validation set is not necessary here as the model will
    see each datapoint only once, so overfitting is impossible and a reduction in train error corresponds to an
    improvement to the model.

    Args:
        model: Model to train.
        complete_distribution: Complete data distribution to sample from.
        epochs: Number of epochs.
            Defaults to 100.
        warmup_epochs: Number of epochs for warmup phase of lr scheduler.
            Defaults to 10.
        steps_per_epoch: Number of batches per epoch.
            Defaults to 100.
        batch_size: Number of samples per batch.
            Defaults to 200.
        lr: Learning rate.
            Defaults to the OpenAI method of determining learning rate.
        weight_decay: Weight decay.
            Defaults to 0.01.
        max_grad: Maximum absolute value of gradients during backpropagation.
            Defaults to 10000.
        scheduler: Learning rate scheduler.
            Defaults to cosine annealing with warmup.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        verbose: Whether to display metrics during training.
            Defaults to False.
        progress_bar: Whether to display a progress bar during training.
            Defaults to False.
        save_interval: How frequently to save model weights in units of epochs.
            Defaults to -1, ie save last epoch only.
        save_loss_series: Whether to save loss series.
            Defaults to True.
        _run: Sacred run object.

    Returns:
        Trained model.

    """
    before_training = time.time()
    assert save_interval == -1 or save_interval is None or (save_interval > 0 and isinstance(save_interval, int)), \
        "save_interval must be None (for no saving), -1 (for final epoch saving only) or a strictly positive int"

    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    device = torch.device(device)
    model.to(device)
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -max_grad, max_grad))

    if lr is None:
        lr = get_openai_lr(model) if lr is None else lr
        print(f"Using OpenAI max lr of {lr}")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, epochs)\
        if scheduler is None else scheduler(optimizer)

    posterior_loss_series = []
    sampler = convert_complete_distribution_to_ace_sampler(complete_distribution, batch_size, sample_space_transform=sample_space_transform, randomise_target=randomise_target)

    def train_epoch() -> dict:
        model.train()
        total_posterior_loss = 0.
        total_forward_time = 0.
        total_step_time = 0.
        epoch_posterior_loss_series = []
        tqdm_iter = tqdm(range(steps_per_epoch), desc='Training Epoch') if progress_bar else None

        before_get_batch = time.time()
        
        time_to_get_epoch = time.time() - before_get_batch

        for step_number in range(steps_per_epoch):
            tqdm_iter.update() if tqdm_iter is not None else None
        
            optimizer.zero_grad()
            batch = sampler.sample()

            for key, tensor in batch.items():
                batch[key] = tensor.to(device)

            before_forward = time.time()
            outs = model(batch)
            forward_time = time.time() - before_forward
            
            optimizer.zero_grad()
            outs.loss.backward()
            optimizer.step()
            scheduler.step()
            step_time = time.time() - before_forward

            total_posterior_loss += outs.loss.cpu().item()
            total_forward_time += forward_time
            total_step_time += step_time
            epoch_posterior_loss_series.append(outs.loss.cpu().item())

            if tqdm_iter:
                postfix_dict = {'step time': step_time,
                                'mean posterior loss': total_posterior_loss / (step_number + 1)}
                tqdm_iter.set_postfix(postfix_dict)

        return {
            "mean_posterior_loss": total_posterior_loss / steps_per_epoch,
            "epoch_load_time": time_to_get_epoch,
            "epoch_time": time.time() - before_get_batch,
            "mean_forward_time": total_forward_time / steps_per_epoch,
            "mean_step_time": total_step_time / steps_per_epoch,
            "epoch_posterior_loss_series": epoch_posterior_loss_series
        }

    epoch_metrics = {}

    for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
        epoch_start_time = time.time()
        epoch_metrics = train_epoch()

        if save_interval is not None and save_interval != -1 and epoch % save_interval == 0 and _run is not None:
            path = _run.observers[0].dir+"\\state_dicts\\ace\\"
            makedirs(path, exist_ok=True)
            torch.save(model.state_dict(), path + f"ace_state_dict_{epoch}.pt")

        posterior_loss_series.append(epoch_metrics["epoch_posterior_loss_series"])

        if verbose:
            print('\n' + '-' * 155)
            print(
                f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s '
                f'| lr {scheduler.get_last_lr()[0]:5.6f} '
                f'| data time {epoch_metrics["epoch_load_time"]:5.2f} | epoch time {epoch_metrics["epoch_time"]:5.2f} '
                f'| forward time {epoch_metrics["mean_forward_time"]:5.5f} '
                f'| step time {epoch_metrics["mean_step_time"]:5.2f} '
                f'| mean posterior loss {epoch_metrics["mean_posterior_loss"]:5.2f} |')
            print('-' * 155)

        scheduler.step()

    if save_interval is not None and _run is not None:
        path = _run.observers[0].dir + "\\state_dicts\\ace\\"
        makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), _run.observers[0].dir + f"\\state_dicts\\ace\\ace_state_dict.pt")

    if save_loss_series and _run is not None:
        _run.info["ace_epoch_posterior_loss_series"] = posterior_loss_series

    if _run is not None:
        _run.info["ace_training_time"] = time.time() - before_training

    model.eval()
    return model.cpu(), epoch_metrics