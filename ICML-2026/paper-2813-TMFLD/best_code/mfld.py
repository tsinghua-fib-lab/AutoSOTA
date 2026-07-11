from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, grad, random, lax
from utils.configs import CFG
from utils.kernel import compute_mmd2
from jaxtyping import Array 
from tqdm import tqdm
from goodpoints import compress
from goodpoints.jax.compress import kt_compresspp
from goodpoints.jax.sliceable_points import SliceablePoints


def glorot_normal(key, fan_in, fan_out):
    # std = jnp.sqrt(2.0 / (fan_in + fan_out))
    std = 1.0
    return std * jax.random.normal(key, (fan_in, fan_out))

def initialize(key, d_in, d_hidden, d_out):
    """PyTorch-like initialization for a 2-layer tanh MLP."""
    k1, k2 = random.split(key)
    k3, k4 = random.split(k2)
    W1 = glorot_normal(k1, d_in, d_hidden)
    b1 = jnp.zeros((d_hidden,))
    W2 = glorot_normal(k3, d_hidden, d_out)
    return W1, b1, W2


def uncentered_matern_32_kernel(points_x, points_y, l):
    X, Y = points_x.get("X"), points_y.get("X")  # (N_x, d), (N_y, d)
    # diff = X[:, None, :] - Y[None, :, :]         # (N_x, N_y, d)
    diff = X - Y
    dists = jnp.linalg.norm(diff, axis=-1)       # (N_x, N_y)
    sqrt3_r = jnp.sqrt(3.0) * dists / l
    return (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)   # (N_x, N_y)

@jax.jit
def uncentered_gaussian_kernel(points_x, points_y, l):
    x, y = points_x.get("p"), points_y.get("p")
    k_xy = jnp.exp(-0.5 * jnp.sum((x - y) ** 2, axis=-1) / (l ** 2))
    return k_xy

class MFLDBase(ABC):
    def __init__(self, problem, thinning: str, save_freq: int, cfg: CFG, args):
        self.cfg = cfg
        self.args = args
        self.problem = problem
        self.data = problem.data
        self.seed = cfg.seed
        self.save_freq = save_freq
        self.kernel_type = cfg.kernel
        self.counter = 0

        # Kernel parameters for kt thinning
        if self.kernel_type == "sobolev":
            k_params = np.array([1.0, 2.0, 3.0])
        elif self.kernel_type == "gaussian":
            k_params = np.array([self.cfg.bandwidth])
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")


        if thinning == 'kt':
            # This is the jax version of kt_compresspp, which is very slow.
            #
            # kernel_fn = partial(uncentered_gaussian_kernel, l=float(1.0))
            
            # def thin_fn(x, key):               
            #     points = SliceablePoints({"p": x})  
            #     rng = np.random.default_rng(int(key[0]))
            #     coresets = kt_compresspp(kernel_fn, points, w=np.ones(x.shape[0]) / x.shape[0], 
            #                  rng_gen=rng, inflate_size=x.shape[0], g=0)
            #     return x[coresets, :]

            # This is the cython version which is fast
            if cfg.kt_function == 'compresspp_kt':
                print(f"Using compresspp_kt with skip_swap={self.cfg.skip_swap}")
                def thin_fn(x, rng_key):
                    rng_key, _ = jax.random.split(rng_key)
                    seed = jax.random.randint(rng_key, (), 0, 2**31 - 1).item()
                    x_cpu = np.array(np.asarray(x))
                    coresets = compress.compresspp_kt(x_cpu, kernel_type=self.kernel_type.encode("utf-8"), k_params=k_params, seed=seed, g=self.cfg.g, skip_swap=self.cfg.skip_swap)
                    return jax.device_put(x_cpu[coresets, :])
            elif cfg.kt_function == 'compress_kt':
                print(f"Using compress_kt with skip_swap={self.cfg.skip_swap}")
                def thin_fn(x, rng_key):
                    rng_key, _ = jax.random.split(rng_key)
                    seed = jax.random.randint(rng_key, (), 0, 2**31 - 1).item()
                    x_cpu = np.array(np.asarray(x))
                    coresets = compress.compress_kt(x_cpu, kernel_type=self.kernel_type.encode("utf-8"), k_params=k_params, seed=seed, g=self.cfg.g, skip_swap=self.cfg.skip_swap)
                    return jax.device_put(x_cpu[coresets, :])
            self.thin_fn = thin_fn
        elif thinning == 'random':
            def thin_fn(x, rng_key):
                rng_key, _ = jax.random.split(rng_key)
                N = x.shape[0]
                indices = jax.random.choice(rng_key, N, (int(jnp.sqrt(N)),), replace=False)
                return x[indices, :]
            self.thin_fn = thin_fn
        elif thinning == 'false':
            self.thin_fn = lambda x, rng_key : x
        else:
            raise ValueError(f"Unknown thinning method: {thinning}")

        if self.problem.q1 is not None:
            self._vm_q1 = vmap(vmap(self.problem.q1, in_axes=(0, 0)), in_axes=(0, None))
            self._vm_grad_q1 = vmap(vmap(lambda z, x: jax.jacrev(self.problem.q1, argnums=1)(z, x), in_axes=(0, 0)), in_axes=(0, None))
        
        if self.problem.q2 is not None:
            self._vm_q2 = jax.vmap(
                jax.vmap(self.problem.q2, in_axes=(None, 0, 0)),  # inner: x[j], key[i,j]
                in_axes=(0, None, 0),                             # outer: z[i], keys[i, :]
            )
            self._vm_grad_q2 = jax.vmap(
                jax.vmap(
                    lambda z, x, key: jax.grad(self.problem.q2, argnums=0)(z, x, key),
                    in_axes=(None, 0, 0),
                ),
                in_axes=(0, None, 0),
            )

class MFLD_nn(MFLDBase):
    def __init__(self, problem, thinning, save_freq, cfg: CFG, args):
        super().__init__(problem, thinning, save_freq, cfg, args)

    @partial(jit, static_argnums=0)
    def vector_field(self, x: Array, thinned_x: Array, data: Array) -> Array:
        (Z, y) = data # (n, N, d), (n, )
        thinned_Z, thinned_y = Z[:, :thinned_x.shape[0], :], y[:, :thinned_x.shape[0]]
        s = self._vm_q1(thinned_Z, thinned_x)   # (n, N, d_out)
        coeff = self.problem.R1_prime(s, thinned_y).mean(1)   # (n, d_out)
        term1_vector = self._vm_grad_q1(Z[:, :x.shape[0], :], x)       # (n, N, d_out, d)
        term1_mean = jnp.einsum("na,ncad->cd", coeff, term1_vector) / coeff.shape[0]
        reg = self.cfg.zeta * x
        return term1_mean + reg

    # @partial(jit, static_argnums=0)
    def _step(self, carry, iter):
        x, v, batch, key = carry
        if self.args.thinning in ['kt', 'random', 'false']:
            key, _ = random.split(key)
            thinned_x = self.thin_fn(x, key)
            vec_field = self.vector_field(x, thinned_x, batch)
            # Cosine noise schedule: sigma_t = sigma_end + 0.5*(sigma_start - sigma_end)*(1 + cos(pi*t/T))
            if self.cfg.noise_schedule == "cosine" and iter is not None:
                t, T = iter, self.cfg.steps
                sigma_t = self.cfg.sigma_end + 0.5 * (self.cfg.sigma_start - self.cfg.sigma_end) * (1.0 + jnp.cos(jnp.pi * t / T))
            else:
                sigma_t = self.cfg.sigma
            if self.cfg.alpha > 0:
                # SGHMC momentum: friction-corrected noise + velocity buffer
                if self.cfg.nesterov:
                    # Nesterov: gradient at look-ahead position x + (1-alpha)*v
                    x_lookahead = x + (1 - self.cfg.alpha) * v
                    thinned_x_lookahead = self.thin_fn(x_lookahead, key)
                    vec_field = self.vector_field(x_lookahead, thinned_x_lookahead, batch)
                noise_scale = jnp.sqrt(2.0 * self.cfg.alpha * sigma_t * self.cfg.step_size)
                key, _ = random.split(key)
                noise = noise_scale * random.normal(key, shape=x.shape)
                v_next = (1 - self.cfg.alpha) * v - self.cfg.step_size * vec_field + noise
                x_next = x + v_next
            else:
                noise_scale = jnp.sqrt(2.0 * sigma_t * self.cfg.step_size)
                key, _ = random.split(key)
                noise = noise_scale * random.normal(key, shape=x.shape)
                x_next = x - self.cfg.step_size * vec_field + noise
                v_next = v
        else:
            raise ValueError(f"Unknown thinning method: {self.args.thinning}")
        return (x_next, v_next, key), thinned_x

    def simulate(self, x0: Optional[Array] = None) -> Array:
        key = random.PRNGKey(self.cfg.seed)
        
        x = x0
        v = jnp.zeros_like(x)  # SGHMC velocity buffer
        path = []
        mmd_path = []
        thin_original_mse_path = []
        time_path = []
        # EMA particle averaging (Polyak-Ruppert)
        ema_beta = 0.999  # EMA decay
        x_ema = x  # Initialize EMA with initial particles
        
        for t in tqdm(range(self.cfg.steps)):
            time_start = time.time()
            # for i, (z, y) in enumerate(zip(self.data["Z"], self.data["y"])):
            #     key, subkey = random.split(key)
            #     log_idx = 4
            #     z_ = z.reshape(-1, log_idx, z.shape[-1])
            #     y_ = y.reshape(-1, log_idx, y.shape[-1])
            #     z_ = z_.repeat(self.args.particle_num // log_idx, axis=1)
            #     y_ = y_.repeat(self.args.particle_num // log_idx, axis=1)
            #     perm = jax.random.permutation(key, z_.shape[1])
            #     z, y = z_[:, perm, :], y_[:, perm, :]
            z = self.data["Z"].transpose(1, 0, 2)  # (n, N, d) -> (n, d, N)
            y = self.data["y"].transpose(1, 0, 2)
            perm = jax.random.permutation(key, z.shape[1])
            z, y = z[:, perm, :], y[:, perm, :]
            key, subkey = random.split(key)
            (x, v, key), thinned_x = self._step((x, v, (z, y), subkey), t)
            time_elapsed = time.time() - time_start

            # Debug code compare MMD between x and thinned_x 
            mmd2 = compute_mmd2(x, thinned_x, bandwidth=self.cfg.bandwidth)

            # thinned_output = self._vm_q1(Z, thinned_x).mean(1)
            # original_output = self._vm_q1(Z, x).mean(1)
            # thin_original_mse = jnp.mean((thinned_output - original_output)**2)
            thin_original_mse = 0.0
            ###########################################
            
            # Update EMA (start after 50% of training as burn-in)
            if t >= self.cfg.steps // 2:
                x_ema = ema_beta * x_ema + (1.0 - ema_beta) * x
            
            path.append(x)
            mmd_path.append(mmd2)
            thin_original_mse_path.append(thin_original_mse)
            time_path.append(time_elapsed)

        path = jnp.stack(path, axis=0)          # (steps, N, d)
        mmd_path = jnp.stack(mmd_path, axis=0)  # (steps, )
        thin_original_mse_path = jnp.stack(thin_original_mse_path, axis=0)  # (steps, )
        time_path = jnp.stack(time_path, axis=0)  # (steps, )
        return path, mmd_path, thin_original_mse_path, time_path, x_ema


class MFLD_vlm(MFLDBase):
    def __init__(self, problem, thinning, save_freq, cfg: CFG, args):
        super().__init__(problem, thinning, save_freq, cfg, args)

    # @partial(jit, static_argnums=0)
    def vector_field(self, x: Array, thinned_x: Array, rng_key) -> Array:
        N, M = x.shape[0], thinned_x.shape[0]
        keys_outer = jax.random.split(rng_key, num=N)                # (N, 2)
        keys_mat = jax.vmap(lambda k: jax.random.split(k, num=M))(keys_outer)  # (N, M, 2)

        term1_vector = self._vm_grad_q2(x, thinned_x, keys_mat)
        term1_mean = jnp.mean(term1_vector, axis=1)
        reg = self.cfg.zeta * x
        return term1_mean + reg
    
    # @partial(jit, static_argnums=0)
    def _step(self, carry, iter):
        x, key = carry
        key, _ = random.split(key)

        if self.args.thinning in ['kt', 'random', 'false']:
            thinned_x = self.thin_fn(x, key)
            v = self.vector_field(x, thinned_x, key)
            noise_scale = jnp.sqrt(2.0 * self.cfg.sigma * self.cfg.step_size)
            key, _ = random.split(key)
            noise = noise_scale * random.normal(key, shape=x.shape)
            x_next = x - self.cfg.step_size * v + noise
        else:
            raise ValueError(f"Unknown thinning method: {self.args.thinning}")
        return (x_next, key), thinned_x
    
    def simulate(self, x0: Optional[Array] = None) -> Array:
        key = random.PRNGKey(self.cfg.seed)
        if x0 is None:
            key, sub = random.split(key)
            x0 = 0.05 * random.normal(sub, (self.cfg.N, self.problem.particle_d))

        x = x0
        v = jnp.zeros_like(x)  # SGHMC velocity buffer
        path = []
        mmd_path = []
        thin_original_mse_path = []
        time_path = []
        # EMA particle averaging (Polyak-Ruppert)
        ema_beta = 0.999  # EMA decay
        x_ema = x  # Initialize EMA with initial particles
        
        for t in tqdm(range(self.cfg.steps)):
            time_start = time.time()
            key_, subkey = random.split(key)
            (x, key) , thinned_x = self._step((x, subkey), t)
            time_elapsed = time.time() - time_start

            ###########################################
            # Debug code compare MMD between x and thinned_x 
            mmd2 = compute_mmd2(x, thinned_x, bandwidth=self.cfg.bandwidth)

            # thinned_output = self._vm_grad_q2(x, thinned_x)
            # original_output = self._vm_grad_q2(x, x)
            # thin_original_mse = jnp.mean((thinned_output.mean(1) - original_output.mean(1))**2)
            thin_original_mse = 0.0
            ###########################################
            
            # Update EMA (start after 50% of training as burn-in)
            if t >= self.cfg.steps // 2:
                x_ema = ema_beta * x_ema + (1.0 - ema_beta) * x
            
            path.append(x)
            mmd_path.append(mmd2)
            thin_original_mse_path.append(thin_original_mse)
            time_path.append(time_elapsed)

        path = jnp.stack(path, axis=0)          # (steps, N, d)
        mmd_path = jnp.stack(mmd_path, axis=0)  # (steps, )
        thin_original_mse_path = jnp.stack(thin_original_mse_path, axis=0)  # (steps, )
        time_path = jnp.stack(time_path, axis=0)  # (steps, )
        return path, mmd_path, thin_original_mse_path, time_path


class MFLD_mmd_flow(MFLDBase):
    def __init__(self, problem, thinning, save_freq, cfg: CFG, args):
        super().__init__(problem, thinning, save_freq, cfg, args)
        from utils.kernel import gaussian_kernel
        self.kernel = gaussian_kernel(sigma=args.bandwidth)

    @partial(jit, static_argnums=0)
    def vector_field(self, x: Array, thinned_x: Array, rng_key) -> Array:
        N, M = x.shape[0], thinned_x.shape[0]
        keys_outer = jax.random.split(rng_key, num=N)                # (N, 2)
        keys_mat = jax.vmap(lambda k: jax.random.split(k, num=M))(keys_outer)  # (N, M, 2)

        term1_vector = self._vm_grad_q2(x, thinned_x, keys_mat)
        term1_mean = jnp.mean(term1_vector, axis=1)

        def dummy_mean_embedding(z):
            return self.problem.distribution.mean_embedding(z).sum()
        term2_vector = jax.grad(dummy_mean_embedding)(x)
        reg = self.cfg.zeta * x
        return term1_mean - term2_vector + reg

    # @partial(jit, static_argnums=0)
    def _step(self, carry, iter):
        x, key = carry
        key, _ = random.split(key)

        if self.args.thinning in ['kt', 'random', 'false']:
            thinned_x = self.thin_fn(x, key)
            v = self.vector_field(x, thinned_x, key)
            noise_scale = jnp.sqrt(2.0 * self.cfg.sigma * self.cfg.step_size)
            key, _ = random.split(key)
            noise = noise_scale * random.normal(key, shape=x.shape)
            x_next = x - self.cfg.step_size * v + noise
        else:
            raise ValueError(f"Unknown thinning method: {self.args.thinning}")
        return (x_next, key), thinned_x
    
    def simulate(self, x0: Optional[Array] = None) -> Array:
        key = random.PRNGKey(self.cfg.seed)
        if x0 is None:
            key, sub = random.split(key)
            x0 = 0.05 * random.normal(sub, (self.cfg.N, self.problem.particle_d))

        x = x0
        v = jnp.zeros_like(x)  # SGHMC velocity buffer
        path = []
        mmd_path = []
        thin_original_mse_path = []
        time_path = []
        # EMA particle averaging (Polyak-Ruppert)
        ema_beta = 0.999  # EMA decay
        x_ema = x  # Initialize EMA with initial particles
        
        for t in tqdm(range(self.cfg.steps)):
            time_start = time.time()
            key_, subkey = random.split(key)
            (x, key) , thinned_x = self._step((x, subkey), t)
            time_elapsed = time.time() - time_start

            ###########################################
            # Debug code compare MMD between x and thinned_x 
            mmd2 = compute_mmd2(x, thinned_x, bandwidth=self.cfg.bandwidth)

            # thinned_output = self._vm_grad_q2(x, thinned_x)
            # original_output = self._vm_grad_q2(x, x)
            # thin_original_mse = jnp.mean((thinned_output.mean(1) - original_output.mean(1))**2)
            thin_original_mse = 0.0
            ###########################################
            
            # Update EMA (start after 50% of training as burn-in)
            if t >= self.cfg.steps // 2:
                x_ema = ema_beta * x_ema + (1.0 - ema_beta) * x
            
            path.append(x)
            mmd_path.append(mmd2)
            thin_original_mse_path.append(thin_original_mse)
            time_path.append(time_elapsed)

        path = jnp.stack(path, axis=0)          # (steps, N, d)
        mmd_path = jnp.stack(mmd_path, axis=0)  # (steps, )
        thin_original_mse_path = jnp.stack(thin_original_mse_path, axis=0)  # (steps, )
        time_path = jnp.stack(time_path, axis=0)  # (steps, )
        return path, mmd_path, thin_original_mse_path, time_path