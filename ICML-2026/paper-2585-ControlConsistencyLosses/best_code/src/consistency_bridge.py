'''
Contains the ConsistencyBridge class.

Passes a 'bridge_config' object to the class, which determines the bridge parameterisation:
- base_drift: bool, whether to include the base drift in the controlled SDE
- guiding_type: 'linearised' or 'brownian', type of guiding drift to use
- decay_coeff: bool, whether to use a time-decaying coefficient for the neural network adjustment
- sampler: 'euler', 'heun', 'milstein'. This is the sampler that is differentiated through when calculating the training targets.

Passes a 'train_config' object at training time. Currently two versions of the self-consistency property are implemented:
- standard: the Jacobian is with respect to the base drift
- nodrift: the Jacobian is the identity (taking b_tilde=-b in the auxiliary drift)

Implemented for a general sigma function, which can be a scalar, diagonal, or full matrix, and can depend on (x,t). The code automatically constructs the necessary functions to avoid excessive matrix multiplication when it is not needed.

Also supports STL adjustments when calculating the training targets. This is passed as an argument to the train_config dictionary.  
'''

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import flax
import flax.linen as nn
from functools import partial
from typing import Any, Callable
from tqdm import trange
import wandb

from .samplers import euler_maruyama_sampler, heun_sampler
from .training import _outer_loop_body, _sample_sde_fn, _sample_controlled_sde_fn, _controlled_drift_fn, _control_fn
from .train_utils import _get_sigma_fn

# TrainState dataclass
@flax.struct.dataclass
class TrainState:
    step: int
    params: Any
    ema_params: Any
    opt_state: Any
    ema_grads: Any
    lr: float
    ts: Any
    ema_rate: float = 0.01
    grad_ema_rate: float = 0.01

    

# ==============================================================================
# Consistency Bridge Class
# ==============================================================================

class ConsistencyBridge:

    def __init__(self, shape, x_0, x_T, base_drift_fn, sigma_fn, model, bridge_config=None, T=1.0):
        self.shape = shape
        self.x_0 = x_0
        self.x_T = x_T
        self.model = model
        self.T = T

        self.base_drift_fn = base_drift_fn
        self.reference_drift_fn = lambda x, t: -base_drift_fn(x, t)

        # defaults
        defaults = {
            'base_drift': False,
            'guiding_type': 'brownian',
            'decay_coeff': True,
            'sampler': 'euler',
        }

        # merge defaults with any user-provided values
        if bridge_config is None:
            bridge_config = {}
        self.bridge_config = {**defaults, **bridge_config}

        # Set up the guiding drift function based on the configuration
        if bridge_config['base_drift'] == True:
            coeff = bridge_config.get('base_drift_coeff', 1.0)
            self.optional_base_drift_fn = lambda x, t: coeff * self.base_drift_fn(x, t)
        else:
            self.optional_base_drift_fn = lambda x, t: jnp.zeros_like(x)

        if bridge_config['guiding_type'] == 'linearised':
            raise NotImplementedError("The 'linearised' guiding type is not implemented in this version.")
        elif bridge_config['guiding_type'] == 'brownian':
            def Brownian_bridge_drift_fn(x, t):
                return (x_T - x) / jnp.maximum(T - t, 1e-3)
            self.guiding_drift = Brownian_bridge_drift_fn

        if bridge_config['decay_coeff'] == True:
            self.coeff_fn = lambda t: jnp.sqrt(T - t)
        else:
            self.coeff_fn = lambda t: 1.0

        if bridge_config["sampler"] == "euler":
            self.sampler_fn = euler_maruyama_sampler
        elif bridge_config["sampler"] == "heun":
            self.sampler_fn = heun_sampler

        self.sigma_fn, self.a_inv = _get_sigma_fn(sigma_fn, dim=shape[0])


    def train(self, key, train_config, wandb_config=None, pretrained_params=None):
        # optionally initialize wandb
        if wandb_config is not None:
            wandb.init(
                project=wandb_config.get("project", "consistency_bridges"),
                name=wandb_config.get("name", None),
                config={**train_config, **wandb_config},  # log configs
            )

        # Initialize optimizer and state
        key, init_key = jax.random.split(key)

        if pretrained_params is not None:
            init_params = pretrained_params
        else:
            init_params = self.model.init(init_key, jnp.zeros(self.shape), 0.0)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(train_config.get('grad_clip', 1.0)),
            optax.adam(learning_rate=train_config['lr'],
                          b1=train_config.get('adam_b1', 0.9),
                          b2=train_config.get('adam_b2', 0.999),
            )
        )

        initial_state = TrainState(
            step=0,
            params=init_params,
            ema_params=init_params,
            ema_grads=jax.tree_util.tree_map(jnp.zeros_like, init_params),
            opt_state=optimizer.init(init_params),
            lr=train_config['lr'],
            ema_rate=train_config['ema_rate'],
            grad_ema_rate=train_config.get('grad_ema_rate', 1.0),
            ts=jnp.linspace(0, self.T, train_config['num_steps'] + 1) # initialise at uniform time discretisation
        )
        
        # 2. Bundle all static data for the pure functions
        static_bridge_data = {
            'model': self.model,
            'x_0': self.x_0,
            'x_T': self.x_T,
            'T': self.T,
            'shape': self.shape,
            'base_drift_fn': self.base_drift_fn,
            'optional_base_drift_fn': self.optional_base_drift_fn,
            'guiding_drift_fn': self.guiding_drift,
            'coeff_fn': self.coeff_fn,
            'sigma_fn': self.sigma_fn,
            'a_inv_fn': self.a_inv,
            'sampler_fn': self.sampler_fn,
            'reference_drift_fn': self.reference_drift_fn,
        }

        # 3. JIT compile the outer body (but not scan over it)
        outer_step_fn = jax.jit(
            partial(_outer_loop_body, train_config=train_config, static_bridge_data=static_bridge_data, optimizer=optimizer)
        )

        # 4. Manual outer loop with tqdm
        state = initial_state
        ema_params_list = [state.ema_params]
        mean_losses = []
        ema_grad_norms = []

        outer_loop_keys = jax.random.split(key, train_config['num_outer_iterations'])

        with trange(train_config['num_outer_iterations'], desc="Training", unit="step", ncols=120) as pbar:
            for step in pbar:
                state, outputs = outer_step_fn(state, outer_loop_keys[step], step)
                mean_losses.append(outputs['mean_loss'])

                ema_grad_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(state.ema_grads)))
                ema_grad_norms.append(ema_grad_norm)

                pbar.set_postfix_str(f"loss={float(outputs['mean_loss']):.4e}, ema_grad_norm={float(ema_grad_norm):.4e}")

                # save ema params at intervals
                if (step + 1) % train_config.get('ckpt_freq', 10) == 0:
                    ema_params_list.append(state.ema_params)

                # optional wandb logging
                if wandb_config is not None:
                    if step % wandb_config.get('log_interval', 10) == 0:

                        wandb.log({
                            'train/loss': float(outputs['mean_loss']),
                            'train/ema_grad_norm': float(ema_grad_norm),
                        }, step=step)
                        
                        for plot_fn in wandb_config.get('plot_fn_lst', []):
                            fig = plot_fn(outputs['x_traj'])   # plot_fn should return a matplotlib figure or None
                            if fig is not None:
                                wandb.log({f"plot/{plot_fn.__name__}": wandb.Image(fig)}, step=step)
                                plt.close(fig)

        print("Training finished.")
        return state, ema_params_list, ema_grad_norms

    @partial(jax.jit, static_argnums=(0,2,3))
    def sample_sde(self, key, drift_fn, num_steps):
        ts = jnp.linspace(0, self.T, num_steps + 1)
        return _sample_sde_fn(key, drift_fn, self.sigma_fn, self.sampler_fn, self.x_0, self.shape, ts)
    
    @partial(jax.jit, static_argnums=(0, 3))
    def sample_controlled_sde(self, key, params, num_steps):
        ts = jnp.linspace(0, self.T, num_steps + 1)
        return _sample_controlled_sde_fn(key, params, self.model, self.optional_base_drift_fn, self.guiding_drift, self.coeff_fn, self.sigma_fn, self.sampler_fn, self.x_0, self.shape, ts)
    
    def controlled_drift(self, params, x, t):
        return _controlled_drift_fn(params, x, t, self.model, self.optional_base_drift_fn, self.guiding_drift, self.coeff_fn, self.sigma_fn)

    def control_fn(self, params, xs, ts):
        return _control_fn(params, xs, ts, self.base_drift_fn, self.model, self.optional_base_drift_fn, self.guiding_drift, self.coeff_fn, self.sigma_fn, self.a_inv_fn)