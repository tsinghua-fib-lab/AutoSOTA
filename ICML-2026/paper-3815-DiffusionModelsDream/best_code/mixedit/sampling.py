import abc
import torch
from tqdm import tqdm
import torch.distributed as dist
import numpy as np

from .model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    @abc.abstractmethod
    def update_fn(self, drift_model, x, t, dt):
        """One update of the predictor.

        Args:
            drift_model: vector field model
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, drift_model, x, t, dt):
        return x
    
@register_predictor(name="grw")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, drift_model, x, t, dt):
        z = self.sde.manifold.random_normal_tangent(base_point=x)

        drift = drift_model(x, t)
        diffusion = self.sde.diffusion(x, t)

        tangent_vec = torch.einsum("...,...ij->...ij", diffusion, z) * np.sqrt(np.abs(dt))
        tangent_vec = tangent_vec + drift * dt

        x = self.sde.manifold.exp(tangent_vec=tangent_vec, base_point=x)
        return x
    

def get_sde_sampler(
    sde, 
    batch_dims, 
    predictor='grw', 
    steps=1000, 
    eps=1e-5, 
    device='cpu', 
    proj_fn=lambda x: x # used for conditional sampling
):
    predictor = get_predictor(predictor)(sde)

    @torch.no_grad()
    def pc_sampler(model):
        drift_fn = mutils.get_drift_fn(model, sde, train=False, sampling=True)
        timesteps = torch.linspace(0, 1-eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        # Sample from prior distribution
        x = sde.prior_sample(batch_dims, device)

        # Sample from generative process
        for i in tqdm(range(steps), desc='Sampling', position=1, leave=False, disable=(not dist.get_rank()==0)):
            t = timesteps[i] * torch.ones(x.shape[0], device=device)
            x = proj_fn(x)
            x = predictor.update_fn(drift_fn, x, t, dt) # B x model.length x D
        
        # Sample indices
        x = proj_fn(x)
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
        probs = sde.manifold.map_to_simplex(x)

        if sde.add_mask_token:
            # Remove mask token prob
            probs = probs[..., :-1]
            
        return probs.argmax(dim=-1)
    
    return pc_sampler


def get_sampling_fn(config, sde, batch_dims, eps, device, **kwargs):
    
    sampling_fn = get_sde_sampler(
        sde=sde,
        batch_dims=batch_dims,
        predictor=config.sampling.predictor,
        steps=config.sampling.steps,
        eps=eps,
        device=device,
        proj_fn=kwargs.get("proj_fn", lambda x: x)
    )
    
    return sampling_fn