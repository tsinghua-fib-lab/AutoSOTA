import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "64"

import time
import jax
import jax.numpy as jnp
from jax import jit, random, jvp
import optax
from flax import linen as nn
from flax.training import train_state
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
jax.config.update('jax_enable_x64', True)
nu = jnp.array(0.01 / jnp.pi)
ic_lambda = 2.

class TINN(nn.Module):
    t_layers: list # e.g. [1,10,10,5]
    layers: list # e.g. [1,20,20,1]
    @nn.compact
    def __call__(self, xt):
        xt = jnp.asarray(xt)
        is_scalar = (xt.ndim == 1)
        if is_scalar:
            xt = xt.reshape(1,2)
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        # time encoder
        temp_t = t
        for i in range(len(self.t_layers)-2):
            tW = self.param(f"tW{i}", nn.initializers.xavier_uniform(), (self.t_layers[i], self.t_layers[i+1]))
            tb = self.param(f"tb{i}", nn.initializers.zeros, (self.t_layers[i+1],))
            temp_t = jnp.tanh(jnp.matmul(temp_t, tW) + tb)
        tWout = self.param("tWout", nn.initializers.xavier_uniform(), (self.t_layers[-2], self.t_layers[-1]))
        temp_t = jnp.matmul(temp_t, tWout)
        t_alpha = self.param("t_alpha", nn.initializers.ones, (1, self.t_layers[-1]))
        temp_t = ((1.0 - t_alpha) * t) + (t_alpha * temp_t)
        z = x
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
        # output layer
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
    return -jnp.sin(jnp.pi * x)

def make_jvp_kernels(model):
    def u_batch(params, xt_batch):
        return model.apply({'params': params}, xt_batch)

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
    u_batch_jit = jax.jit(u_batch)
    return u_batch_jit, grads_jvp, u_xx_jvp

def build_loss_and_steps(model):
    u_batch, grads_jvp, u_xx_jvp = make_jvp_kernels(model)
    @jax.jit
    def pde_residual_single(params, xt):
        u = u_batch(params, xt[None, :])[0]
        u_x, u_t = grads_jvp(params, xt[None, :])
        u_xx = u_xx_jvp(params, xt[None, :])
        return u_t + u * u_x - nu * u_xx
    pde_jac_single = jax.jacrev(pde_residual_single, argnums=0)
    ic_jac_single  = jax.jacrev(lambda p, xt: ic_lambda*(u_batch(p, xt[None,:])[0] - ic_function(xt[0])), argnums=0)
    bc_jac_single  = jax.jacrev(lambda p, xt: u_batch(p, xt[None,:])[0], argnums=0)
    def vmap_jac(jac_single, xt_batch, params, chunk=15000):
        N = xt_batch.shape[0]
        outs = []
        for i in range(0, N, chunk):
            xt_sub = xt_batch[i:i+chunk]
            jac_sub = jax.vmap(lambda xt: jac_single(params, xt))(xt_sub)
            outs.append(jac_sub)
        return jax.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *outs)
    
    @jax.jit
    def loss_fn(params, batch_coll_xt, batch_ic_xt, batch_bc_xt):
        res_pde = jax.vmap(lambda xt: pde_residual_single(params, xt))(batch_coll_xt)
        res_ic = jax.vmap(
            lambda xt: ic_lambda*(u_batch(params, xt[None,:])[0] - ic_function(xt[0])))(batch_ic_xt)
        res_bc = jax.vmap(lambda xt: u_batch(params, xt[None,:])[0])(batch_bc_xt)
        lpde = jnp.mean(res_pde**2)
        lic = jnp.mean(res_ic**2)
        lbc = jnp.mean(res_bc**2)
        total = lpde+lic+lbc
        return total, (lpde, lic, lbc)
    
    @jax.jit
    def train_step(params, batch_coll_xt, batch_ic_xt, batch_bc_xt):
        res_pde = jax.vmap(lambda xt: pde_residual_single(params, xt))(batch_coll_xt)
        res_ic = jax.vmap(
            lambda xt: ic_lambda*(u_batch(params, xt[None,:])[0] - ic_function(xt[0])))(batch_ic_xt)
        res_bc = jax.vmap(lambda xt: u_batch(params, xt[None,:])[0])(batch_bc_xt).reshape(len(batch_bc_xt), -1)

        pde_theta = vmap_jac(pde_jac_single, batch_coll_xt, params)
        ic_theta  = vmap_jac(ic_jac_single,  batch_ic_xt, params)
        bc_theta  = vmap_jac(bc_jac_single,  batch_bc_xt, params)
        return res_pde, pde_theta, res_ic, ic_theta, res_bc, bc_theta
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

def sample_bc(rng, N):
    t = random.uniform(rng, (N,1), minval=0.0, maxval=1.0)
    x0 = -jnp.ones_like(t)
    x1 =  jnp.ones_like(t)
    pts0 = jnp.hstack([x0, t])
    pts1 = jnp.hstack([x1, t])
    return jnp.vstack([pts0, pts1])

data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()
x = data['x'].flatten()
Exact = np.real(data['usol']).T
TT, XX = jnp.meshgrid(t, x, indexing="ij")
total_test = len(TT.flatten())

def L2Error():
    u_pred = jax.vmap(model.apply, (None, 0))({'params': params}, jnp.hstack((XX.reshape([total_test,1]), TT.reshape([total_test,1])))).reshape(TT.shape)
    error = Exact-u_pred
    # l2 error
    l2_error = np.sqrt(np.sum(error.flatten()**2)/total_test)
    rel_l2_error = l2_error / np.sqrt(np.sum(Exact.flatten()**2)/total_test)
    return rel_l2_error

key = random.PRNGKey(4)
model = TINN([1,10,10,5], [1,20,20,1])
xt_dummy = jnp.zeros((1,2))
params = model.init(key, xt_dummy)['params']
total_number = 0
for p in params:
    total_number +=params[p].size
print("total parameter:", total_number)

loss_fn, train_step = build_loss_and_steps(model)
_dummy_coll = sample_collocation(key, 8)
_dummy_ic = sample_ic(key, 4)
_dummy_bc = sample_bc(key, 4)
_ = loss_fn(params, _dummy_coll, _dummy_ic, _dummy_bc)
res_pde, pde_theta, res_ic, ic_theta, res_bc, bc_theta = train_step(params, _dummy_coll, _dummy_ic, _dummy_bc)

N_coll = 10000
N_ic   = 500
N_bc   = 200
N_val_coll = 5000
N_val_ic = 250
N_val_bc = 100

key, kc, ki, kb = random.split(key, 4)
batch_coll_xt = sample_collocation(kc, N_coll)
batch_ic_xt   = sample_ic(ki, N_ic)
batch_bc_xt   = sample_bc(kb, N_bc)
key, vk1, vk2, vk3 = random.split(key, 4)
val_coll_xt = sample_collocation(vk1, N_val_coll)
val_ic_xt   = sample_ic(vk2, N_val_ic)
val_bc_xt   = sample_bc(vk3, N_val_bc)
# ic_average = jnp.mean(jnp.sort(abs(ic_function(batch_ic_xt[:,0])))[:300])
# print(1/(ic_average))

def LM_reshape(grad_params):
    N_num = grad_params['aW0'].shape[0]
    temp = grad_params['aW0'].reshape(N_num, jnp.size(grad_params['aW0'])//N_num)
    for ii, p in enumerate(grad_params):
        if ii>0:
            temp = jnp.hstack((temp, grad_params[p].reshape(N_num, jnp.size(grad_params[p])//N_num )))
    return temp
#--------------------------------------
Epoch = 30000
mu = 10**1
itera_ = 0
mu_update = 2 
div_factor = 1.3 
mul_factor = 1.7
loss_sum_old = 10**5
min_mu = 10**-12
loss_LM = []
l2_error_list = []
#--------------------------------------
start = time.time()
for step in range(Epoch):
    res_pde, pde_theta, res_ic, ic_theta, res_bc, bc_theta = train_step(params, batch_coll_xt, batch_ic_xt, batch_bc_xt)
    re_pde_theta = LM_reshape(pde_theta)
    re_ic_theta = LM_reshape(ic_theta)
    re_bc_theta = LM_reshape(bc_theta)

    val_tot, (val_pde, val_ic, val_bc) = loss_fn(params, val_coll_xt, val_ic_xt, val_bc_xt)
    # J_mat
    J_mat = jax.lax.concatenate((re_pde_theta/(N_coll**0.5), re_bc_theta/(2*N_bc)**0.5, re_ic_theta/N_ic**0.5), 0)
    #L_vec
    L_vec = jax.lax.concatenate((res_pde/N_coll**0.5,res_bc/(2*N_bc)**0.5,res_ic/N_ic**0.5), 0)
    loss = jnp.sum(L_vec**2)
    loss_LM.append(loss.item())

    I = jnp.eye((J_mat.shape[1]))
    J_product = J_mat.T@J_mat
    rhs = -J_mat.T@L_vec
    dp = jnp.linalg.solve((J_product+mu*I) , rhs)
    cnt=0
    for p in pde_theta:
        num = jnp.size(params[p])
        params[p] = params[p] + dp[cnt:cnt+num].reshape(params[p].shape)
        cnt+=num
        
    itera_ += 1
    if step % mu_update == 0:
        if loss < loss_sum_old:
            mu = max(mu/div_factor, min_mu)
        else:
            mu = min(mul_factor*mu, 10**(8))
        loss_sum_old = loss
    if loss.item()/mu > 10**5:
        mu = loss.item()/10
    if step%50 == 0:
        lpde = jnp.mean(res_pde**2)
        lic = jnp.mean(res_ic**2)
        lbc = jnp.mean(res_bc**2)
        elapsed = time.time()-start
        temp_error = L2Error().item()
        l2_error_list.append(temp_error)
        print(f"Epoch {itera_:4d} | L2Error={temp_error:.3e} | loss={loss:.3e} | val={val_tot:.3e} | mu={mu:.3e} | pde={lpde:.3e} | ic={lic:.3e} | bc={lbc:.3e} | time={elapsed:.1f}s")
    
    if (val_tot/loss>5):
        print('Validation')
        key, kc, ki, kb = random.split(key, 4)
        batch_coll_xt = sample_collocation(kc, N_coll)
        batch_ic_xt   = sample_ic(ki, N_ic)
        batch_bc_xt   = sample_bc(kb, N_bc)
            
print('the final Loss: %.5e'% (loss))
print('the final valid Loss: %.5e'% (val_tot))
final_time = time.time()
print('total time:', final_time - start)

data = scipy.io.loadmat('burgers_shock.mat')
t = data['t'].flatten()
x = data['x'].flatten()
Exact = np.real(data['usol']).T

TT, XX = jnp.meshgrid(t, x, indexing="ij")
total_test = len(TT.flatten())
u_pred = jax.vmap(model.apply, (None, 0))({'params': params}, jnp.hstack((XX.reshape([total_test,1]), TT.reshape([total_test,1])))).reshape(TT.shape)
error = Exact-u_pred

# l2 error
l2_error = np.sqrt(np.sum(error.flatten()**2)/total_test)
l_int_error = np.max(abs(error.flatten()))
rel_l2_error = l2_error / np.sqrt(np.sum(Exact.flatten()**2)/total_test)
rel_l_inf_error = l_int_error / np.max(abs(Exact.flatten()))

print('L2-Error:%.5e'%l2_error)
print('Linf-Error:%.5e'%l_int_error)
print('rel-L2-Error: %.5e'% rel_l2_error)
print('rel-Linf-Error: %.5e'% rel_l_inf_error)

fig = plt.figure(figsize=(12,2.8))

plt.subplot(1,3,1)
plt.pcolor(TT, XX, Exact, cmap="jet")
plt.xlabel('t')
plt.ylabel('x')
plt.title('Exact u')
plt.colorbar()

plt.subplot(1,3,2)
plt.pcolor(TT, XX, u_pred, cmap="jet")
plt.xlabel('t')
plt.ylabel('x')
plt.title('Predict u')
plt.colorbar()

plt.subplot(1,3,3)
plt.pcolor(TT, XX, abs(error), cmap="jet")
plt.xlabel('t')
plt.ylabel('x')
plt.title('Absolute Error')
plt.colorbar()

plt.tight_layout()
#plt.savefig("TINN_Burgers.png", dpi=300)
plt.show()
