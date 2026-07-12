from typing import List, Callable, Tuple
from functools import partial
import itertools
from tqdm import trange
import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree

KeyArray = random.PRNGKey


def MLP(
    layers: List[int],
    activation: Callable[[ndarray], ndarray] = jax.nn.relu
) -> Tuple[Callable[[KeyArray], Tuple], Callable[[Tuple, ndarray], ndarray]]:
    """MLP Layer for DeepONet"""

    def xavier_init(key: KeyArray, d_in: int, d_out: int) -> Tuple[ndarray, ndarray]:
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    def init(rng_key: KeyArray) -> Tuple:
        U1, b1 =  xavier_init(random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 =  xavier_init(random.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key: KeyArray, d_in: int, d_out: int) -> Tuple[ndarray, ndarray]:
            k1, _ = random.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b

        _, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2) 

    def apply(params: Tuple, inputs: ndarray) -> ndarray:
        params, U1, b1, U2, b2 = params
        U = activation(jnp.dot(inputs, U1) + b1)
        V = activation(jnp.dot(inputs, U2) + b2)

        for W, b in params[:-1]:
            outputs = activation(jnp.dot(inputs, W) + b)
            inputs = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V) 

        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        return outputs

    return init, apply


class PIDeepONet:
    """Physics-informed DeepONet model"""
    def __init__(self, branch_layers: List[int], trunk_layers: List[int]) -> None:    
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=jnp.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=jnp.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key=random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key=random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = \
            optimizers.adam(optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    def operator_net(self, params: Tuple, u: ndarray, t: ndarray, x: ndarray) -> ndarray:
        """Define DeepONet architecture"""
        branch_params, trunk_params = params
        y = jnp.stack([t, x])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = jnp.sum(B * T)
        return outputs

    def s_x_net(self, params: Tuple, u: ndarray, t: ndarray, x: ndarray) -> ndarray:
        """Define ds/dx"""
        s_x = grad(self.operator_net, argnums=3)(params, u, t, x)
        return s_x

    def residual_net(self, params: Tuple, u: ndarray, t: ndarray, x: ndarray) -> ndarray:
        """Define PDE residual"""
        s = self.operator_net(params, u, t, x)
        s_t = grad(self.operator_net, argnums=2)(params, u, t, x)
        s_x = grad(self.operator_net, argnums=3)(params, u, t, x)
        s_xx= grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, t, x)
        res = s_t + s * s_x - 0.01 * s_xx
        return res

    def loss_ics(self, params: Tuple, batch: Tuple[Tuple[ndarray, ndarray], ndarray]) -> ndarray:
        """Define initial loss"""
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        # Compute loss
        loss = jnp.mean((outputs.flatten() - s_pred)**2)
        return loss

    def loss_bcs(self, params: Tuple, batch: Tuple[Tuple[ndarray, ndarray], ndarray]) -> ndarray:
        """Define boundary loss"""
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        s_bc1_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        s_bc2_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:,2], y[:,3])
        s_x_bc1_pred = vmap(self.s_x_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        s_x_bc2_pred = vmap(self.s_x_net, (None, 0, 0, 0))(params, u, y[:,2], y[:,3])
        # Compute loss
        loss_s_bc = jnp.mean((s_bc1_pred - s_bc2_pred)**2)
        loss_s_x_bc = jnp.mean((s_x_bc1_pred - s_x_bc2_pred)**2)
        return loss_s_bc + loss_s_x_bc

    def loss_res(self, params: Tuple, batch: Tuple[Tuple[ndarray, ndarray], ndarray]) -> ndarray:
        """Define residual loss"""
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:,0], y[:,1])
        # Compute loss
        loss = jnp.mean((outputs.flatten() - pred)**2)
        return loss    

    def loss(self, params: Tuple,
        ics_batch: Tuple[Tuple[ndarray, ndarray], ndarray],
        bcs_batch: Tuple[Tuple[ndarray, ndarray], ndarray],
        res_batch: Tuple[Tuple[ndarray, ndarray], ndarray]
    ) -> ndarray:
        """Define total loss"""
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = 20 * loss_ics + loss_bcs + loss_res
        return loss

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        i: int,
        opt_state: object,
        ics_batch: Tuple[Tuple[ndarray, ndarray], ndarray],
        bcs_batch: Tuple[Tuple[ndarray, ndarray], ndarray],
        res_batch: Tuple[Tuple[ndarray, ndarray], ndarray]
    ) -> object:
        """Define a compiled update step"""
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, ics_batch, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    def train(
        self,
        ics_dataset: Tuple[Tuple[ndarray, ndarray], ndarray],
        bcs_dataset: Tuple[Tuple[ndarray, ndarray], ndarray],
        res_dataset: Tuple[Tuple[ndarray, ndarray], ndarray],
        nIter: int = 10000
    ) -> None:
        """Optimize parameters in a loop"""
        ics_data = iter(ics_dataset)
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            ics_batch= next(ics_data)
            bcs_batch= next(bcs_data)
            res_batch = next(res_data)
            self.opt_state = self.step(next(self.itercount), self.opt_state, ics_batch, bcs_batch, res_batch)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
                # Compute losses
                loss_value = self.loss(params, ics_batch, bcs_batch, res_batch)
                loss_ics_value = self.loss_ics(params, ics_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)
                # Store losses
                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                # Print losses
                pbar.set_postfix({'Loss': loss_value, 
                                  'loss_ics' : loss_ics_value,
                                  'loss_bcs' : loss_bcs_value, 
                                  'loss_physics': loss_res_value})

    @partial(jit, static_argnums=(0,))
    def predict_s(
        self,
        params: Tuple,
        U_star: ndarray,
        Y_star: ndarray
    ) -> ndarray:
        """Evaluates predictions at test points"""
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_pred
        # s_pred_list = []
        # for i in range(U_star.shape[0]):
        #     s_pred_i = self.operator_net(params, U_star[i], Y_star[i,0], Y_star[i,1])
        #     s_pred_list.append(s_pred_i)
        # s_pred = np.stack(s_pred_list)
        # return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(
        self,
        params: Tuple,
        U_star: ndarray,
        Y_star: ndarray
    ) -> ndarray:
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:,0], Y_star[:,1])
        return r_pred
