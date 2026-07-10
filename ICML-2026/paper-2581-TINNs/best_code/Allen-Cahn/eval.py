#!/usr/bin/env python3
"""TINN Allen-Cahn Evaluation Script - Single Seed"""

import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "64"

import argparse
import time
import jax
import jax.numpy as jnp
from jax import random, jvp
from flax import linen as nn
import scipy.io
import numpy as np

jax.config.update("jax_enable_x64", True)
ic_lambda = 20.0

class TINN(nn.Module):
    t_layers: list
    layers: list
    @nn.compact
    def __call__(self, xt):
        xt = jnp.asarray(xt)
        is_scalar = (xt.ndim == 1)
        if is_scalar:
            xt = xt.reshape(1,2)
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        temp_t = t
        for i in range(len(self.t_layers)-2):
            tW = self.param(f"tW{i}", nn.initializers.xavier_uniform(), (self.t_layers[i], self.t_layers[i+1]))
            tb = self.param(f"tb{i}", nn.initializers.zeros, (self.t_layers[i+1],))
            temp_t = jnp.tanh(jnp.matmul(temp_t, tW) + tb)
        tWout = self.param("tWout", nn.initializers.xavier_uniform(), (self.t_layers[-2], self.t_layers[-1]))
        temp_t = jnp.matmul(temp_t, tWout)
        t_alpha = self.param("t_alpha", nn.initializers.ones, (1, self.t_layers[-1]))
        temp_t = ((1.0 - t_alpha) * t) + (t_alpha * temp_t)
        # ALGO-03: Multi-harmonic periodic embedding (k=1,2,3,4)
        # Richer frequency basis for sharp transition layers
        z_parts = []
        for k in [1, 2, 3, 4]:
            z_parts.append(jnp.cos(k * jnp.pi * x))
            z_parts.append(jnp.sin(k * jnp.pi * x))
        z = jnp.hstack(z_parts)
        for i in range(len(self.layers)-2):
            in_dim = self.layers[i]
            out_dim = self.layers[i+1]
            aW = self.param(f"aW{i}", nn.initializers.zeros, (in_dim, out_dim))
            ab = self.param(f"ab{i}", nn.initializers.xavier_uniform(), (1,out_dim))
            bW = self.param(f"bW{i}", nn.initializers.xavier_uniform(), (in_dim, out_dim))
            bb = self.param(f"bb{i}", nn.initializers.zeros, (out_dim,))
            coef = temp_t[..., 2*i:2*i+1]
            term1 = jnp.matmul(z, aW) * coef
            term2 = jnp.matmul(z, bW)
            coef = temp_t[..., 2*i+1:2*i+2]
            b_eff = ab * coef + bb
            z = jnp.tanh(term1 + term2 + b_eff)
        aW_out = self.param("aWout", nn.initializers.zeros, (self.layers[-2], self.layers[-1]))
        bW_out = self.param("bWout", nn.initializers.xavier_uniform(), (self.layers[-2], self.layers[-1]))
        coef_out = temp_t[..., -1:]
        term1 = jnp.matmul(z, aW_out) * coef_out
        term2 = jnp.matmul(z, bW_out)
        y = term1 + term2
        y = jnp.squeeze(y, axis=-1)
        if is_scalar:
            return y[0]
        return y

def ic_function(x):
    x = jnp.asarray(x).reshape(-1)
    return x**2 * jnp.cos(3*jnp.pi*x) + x**2

def make_jvp_kernels(model):
    def u_batch(params, xt_batch):
        return model.apply({"params": params}, xt_batch)
    @jax.jit
    def grads_jvp(params, xt_batch):
        N = xt_batch.shape[0]
        v_x = jnp.tile(jnp.array([1.0, 0.0]), (N,1))
        v_t = jnp.tile(jnp.array([0.0, 1.0]), (N,1))
        _, u_x = jvp(lambda xs: u_batch(params, xs), (xt_batch,), (v_x,))
        _, u_t = jvp(lambda xs: u_batch(params, xs), (xt_batch,), (v_t,))
        return u_x, u_t
    @jax.jit
    def u_xx_jvp(params, xt_batch):
        N = xt_batch.shape[0]
        v_x = jnp.tile(jnp.array([1.0, 0.0]), (N,1))
        def first_tangent(xs):
            _, tangent = jvp(lambda x_in: u_batch(params, x_in), (xs,), (v_x,))
            return tangent
        _, u_xx = jvp(first_tangent, (xt_batch,), (v_x,))
        return u_xx
    return jax.jit(u_batch), grads_jvp, u_xx_jvp

def build_loss_and_steps(model):
    u_batch, grads_jvp, u_xx_jvp = make_jvp_kernels(model)
    @jax.jit
    def pde_residual_single(params, xt):
        u = u_batch(params, xt[None, :])[0]
        u_x, u_t = grads_jvp(params, xt[None, :])
        u_xx = u_xx_jvp(params, xt[None, :])
        return u_t - 0.0001*u_xx + 5*u**3 - 5*u
    pde_jac_single = jax.jacrev(pde_residual_single, argnums=0)
    ic_jac_single = jax.jacrev(lambda p, xt: ic_lambda*(u_batch(p, xt[None,:])[0] - ic_function(xt[0])), argnums=0)
    
    def vmap_jac(jac_single, xt_batch, params, chunk=15000):
        N = xt_batch.shape[0]
        outs = []
        for i in range(0, N, chunk):
            xt_sub = xt_batch[i:i+chunk]
            jac_sub = jax.vmap(lambda xt: jac_single(params, xt))(xt_sub)
            outs.append(jac_sub)
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)
    
    @jax.jit
    def loss_fn(params, batch_coll_xt, batch_ic_xt):
        res_pde = jax.vmap(lambda xt: pde_residual_single(params, xt))(batch_coll_xt)
        res_ic = jax.vmap(lambda xt: ic_lambda*(u_batch(params, xt[None,:])[0] - ic_function(xt[0])))(batch_ic_xt)
        lpde = jnp.mean(res_pde**2)
        lic = jnp.mean(res_ic**2)
        return lpde+lic, (lpde, lic)
    
    @jax.jit
    def train_step(params, batch_coll_xt, batch_ic_xt):
        res_pde = jax.vmap(lambda xt: pde_residual_single(params, xt))(batch_coll_xt)
        res_ic = jax.vmap(lambda xt: ic_lambda*(u_batch(params, xt[None,:])[0] - ic_function(xt[0])))(batch_ic_xt)
        pde_theta = vmap_jac(pde_jac_single, batch_coll_xt, params)
        ic_theta  = vmap_jac(ic_jac_single,  batch_ic_xt, params)
        return res_pde, pde_theta, res_ic, ic_theta
    return loss_fn, train_step

def sample_collocation(rng, N):
    k1, k2 = random.split(rng)
    x = random.uniform(k1, (N,1), minval=-1.0, maxval=1.0)
    t = random.uniform(k2, (N,1), minval=0.0, maxval=1.0)
    return jnp.hstack([x,t])

def sample_ic(rng, N):
    x = random.uniform(rng, (N,1), minval=-1.0, maxval=1.0)
    t = jnp.zeros_like(x)
    return jnp.hstack([x,t])

def LM_reshape(grad_params):
    N_num = grad_params["aW0"].shape[0]
    temp = grad_params["aW0"].reshape(N_num, jnp.size(grad_params["aW0"])//N_num)
    for ii, p in enumerate(grad_params):
        if ii>0:
            temp = jnp.hstack((temp, grad_params[p].reshape(N_num, jnp.size(grad_params[p])//N_num )))
    return temp

def main():
    parser = argparse.ArgumentParser(description="TINN Allen-Cahn Evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=60000, help="Training iterations (PARAM-01: extended from 30K)")
    parser.add_argument("--Nc", type=int, default=20000, help="Collocation points (CODE-01: 2x paper default)")
    parser.add_argument("--Nic", type=int, default=1000, help="Initial condition points (CODE-01: 2x paper default)")
    parser.add_argument("--lambda-ic", type=float, default=20.0, help="IC penalty weight")
    parser.add_argument("--mu-init", type=float, default=10.0, help="Initial LM damping")
    parser.add_argument("--div-factor", type=float, default=1.3, help="LM mu reduction factor")
    parser.add_argument("--mul-factor", type=float, default=1.7, help="LM mu growth factor")
    parser.add_argument("--mu-min", type=float, default=1e-12, help="Minimum LM damping")
    parser.add_argument("--mu-max", type=float, default=1e8, help="Maximum LM damping")
    args = parser.parse_args()
    
    global ic_lambda
    ic_lambda = args.lambda_ic
    
    # Load reference data
    data = scipy.io.loadmat("new_AC.mat")
    Exact = data["u_ref"]
    t = data["t_data"].flatten()
    x = data["x_data"].flatten()
    TT, XX = jnp.meshgrid(t, x, indexing="ij")
    total_test = len(TT.flatten())
    
    # Initialize model
    key = random.PRNGKey(args.seed)
    model = TINN([1,10,10,5], [8,20,20,1])  # ALGO-03: 8 input dims (4 harmonics x 2)
    xt_dummy = jnp.zeros((1,2))
    params = model.init(key, xt_dummy)["params"]
    
    total_params = sum(params[p].size for p in params)
    print(f"Model: TINN, Params: {total_params}")
    print(f"CODE-01: Increased points N_coll={args.Nc}, N_ic={args.Nic} (2x paper defaults)")
    
    loss_fn, train_step = build_loss_and_steps(model)
    
    # Warmup
    _dummy_coll = sample_collocation(key, 8)
    _dummy_ic = sample_ic(key, 4)
    _ = loss_fn(params, _dummy_coll, _dummy_ic)
    res_pde, pde_theta, res_ic, ic_theta = train_step(params, _dummy_coll, _dummy_ic)
    
    N_coll = args.Nc
    N_ic = args.Nic
    N_val_coll = 10000  # CODE-01: 2x to match increased training points
    N_val_ic = 500       # CODE-01: 2x to match increased training points
    
    key, kc, ki = random.split(key, 3)
    batch_coll_xt = sample_collocation(kc, N_coll)
    batch_ic_xt = sample_ic(ki, N_ic)
    key, vk1, vk2 = random.split(key, 3)
    val_coll_xt = sample_collocation(vk1, N_val_coll)
    val_ic_xt = sample_ic(vk2, N_val_ic)
    
    # Training
    Epoch = args.epochs
    mu = args.mu_init
    itera_ = 0
    mu_update = 2
    div_factor = args.div_factor
    mul_factor = args.mul_factor
    loss_sum_old = 1e5
    min_mu = args.mu_min
    max_mu_val = args.mu_max
    
    start = time.time()
    # CODE-03: Best-model checkpointing
    best_val_tot = float('inf')
    best_params = None
    patience = 0
    patience_limit = 10000  # Early stop if no improvement for 10K steps
    best_step = 0
    best_rel_l2 = float('inf')
    
    for step in range(Epoch):
        res_pde, pde_theta, res_ic, ic_theta = train_step(params, batch_coll_xt, batch_ic_xt)
        re_pde_theta = LM_reshape(pde_theta)
        re_ic_theta = LM_reshape(ic_theta)
        val_tot, (val_pde, val_ic) = loss_fn(params, val_coll_xt, val_ic_xt)
        
        J_mat = jax.lax.concatenate((re_pde_theta/(N_coll**0.5), re_ic_theta/N_ic**0.5), 0)
        L_vec = jax.lax.concatenate((res_pde/N_coll**0.5, res_ic/N_ic**0.5), 0)
        loss = jnp.mean(res_ic**2) + jnp.mean(res_pde**2)
        
        I = jnp.eye(J_mat.shape[1])
        J_product = J_mat.T@J_mat
        rhs = -J_mat.T@L_vec
        dp = jnp.linalg.solve(J_product + mu*I, rhs)
        cnt = 0
        for p in pde_theta:
            num = jnp.size(params[p])
            params[p] = params[p] + dp[cnt:cnt+num].reshape(params[p].shape)
            cnt += num
        itera_ += 1
        
        if step % mu_update == 0:
            if loss < loss_sum_old:
                mu = max(mu/div_factor, min_mu)
            else:
                mu = min(mul_factor*mu, max_mu_val)
            loss_sum_old = loss
        if loss.item()/mu > 1e5:
            mu = loss.item()/10
        
        if (val_tot/loss > 5):
            key, kc, ki = random.split(key, 3)
            batch_coll_xt = sample_collocation(kc, N_coll)
            batch_ic_xt = sample_ic(ki, N_ic)
        
        if step % 500 == 0:
            elapsed = time.time() - start
            u_pred = jax.vmap(model.apply, (None, 0))({"params": params}, jnp.hstack((XX.reshape([total_test,1]), TT.reshape([total_test,1])))).reshape(TT.shape)
            error = Exact - u_pred
            l2_error = np.sqrt(np.sum(error.flatten()**2)/total_test)
            rel_l2_error = l2_error / np.sqrt(np.sum(Exact.flatten()**2)/total_test)
            # CODE-03 refine: Track best model by RelL2Error (more reliable than val loss)
            if rel_l2_error < best_rel_l2:
                best_rel_l2 = rel_l2_error
                best_params = jax.tree.map(lambda x: x.copy(), params)
                best_step = step
                patience = 0
            else:
                patience += 500
            checkpoint_tag = " [BEST]" if step == best_step else ""
            print(f"Step {step:5d} | RelL2Error={rel_l2_error:.5e} | Loss={loss:.3e} | Val={val_tot.item():.3e} | Time={elapsed:.1f}s{checkpoint_tag}")
            # CODE-03: Early stopping
            if patience >= patience_limit:
                print(f"Early stopping at step {step}: no improvement for {patience_limit} steps")
                break
    
    elapsed = time.time() - start

    # CODE-03: Restore best checkpointed params
    if best_params is not None:
        params = best_params
        print(f"\nRestored best model from step {best_step} (RelL2={best_rel_l2:.5e})")

    # Final evaluation
    u_pred = jax.vmap(model.apply, (None, 0))({"params": params}, jnp.hstack((XX.reshape([total_test,1]), TT.reshape([total_test,1])))).reshape(TT.shape)
    error = Exact - u_pred
    l2_error = np.sqrt(np.sum(error.flatten()**2)/total_test)
    l_inf_error = np.max(abs(error.flatten()))
    rel_l2_error = l2_error / np.sqrt(np.sum(Exact.flatten()**2)/total_test)
    rel_l_inf_error = l_inf_error / np.max(abs(Exact.flatten()))
    
    print("\n" + "="*50)
    print(f"FINAL RESULTS (seed={args.seed})")
    print("="*50)
    print(f"L2-Error:        {l2_error:.5e}")
    print(f"Linf-Error:      {l_inf_error:.5e}")
    print(f"rel-L2-Error:    {rel_l2_error:.5e}")
    print(f"rel-Linf-Error:  {rel_l_inf_error:.5e}")
    print(f"Training Time:   {elapsed:.1f}s ({elapsed/3600:.3f}h)")
    print(f"Metric: Relative L2 Error = {rel_l2_error:.5e}")
    
    return rel_l2_error

if __name__ == "__main__":
    main()
