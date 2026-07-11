# experiment: Diffusion — triple-mode 1D diffusion with BEST Rel L2 tracking

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map

from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from jax.nn import tanh as j_tanh

# ---------------------------
# Problem parameters (from your SymPy setup)
# ---------------------------
D = 0.697      # diffusion coefficient
L = 1.397      # spatial length
A = 3.974      # common amplitude
MODES = (1, 3, 5)           # Fourier modes
SIGNS = ( +1.0, -1.0, +1.0) # signs for each mode ( +, -, + )

# ---------------------------
# Domain: [0, L] x [0, 1]
# ---------------------------
domain = RectangularDomainND
domain_init_kwargs = dict(
    xmin=np.array([0.0, 0.0]),
    xmax=np.array([L,   1.0]),
)

# ---------------------------
# Utility: exact triple-mode solution and its terms
# ---------------------------
def _triple_mode_u(x, t):
    """u(x,t) = sum s_k * A * sin(n_k*pi*x/L) * exp(-n_k^2*pi^2*D*t/L^2)."""
    u = 0.0
    for n, s in zip(MODES, SIGNS):
        k = n * jnp.pi / L
        u += s * A * jnp.sin(k * x) * jnp.exp(-(n**2) * (jnp.pi**2) * D * t / (L**2))
    return u

def _u_t(x, t):
    """time derivative of the triple-mode u."""
    ut = 0.0
    for n, s in zip(MODES, SIGNS):
        lam = (n**2) * (jnp.pi**2) * D / (L**2)
        k = n * jnp.pi / L
        ut += s * A * jnp.sin(k * x) * (-lam) * jnp.exp(-lam * t)
    return ut

def _u_xx(x, t):
    """second spatial derivative of the triple-mode u."""
    uxx = 0.0
    for n, s in zip(MODES, SIGNS):
        k = n * jnp.pi / L
        lam_x = (n**2) * (jnp.pi**2) / (L**2)
        uxx += s * A * (-lam_x) * jnp.sin(k * x) * jnp.exp(-(n**2) * (jnp.pi**2) * D * t / (L**2))
    return uxx

# Note: For this canonical diffusion form, u_t - D*u_xx = 0 exactly (homogeneous PDE).
# We still keep the structure flexible in case you want to add a forcing F later.

# ---------------------------
# Problem definition
# ---------------------------
class Diffusion(Problem):
    """Solves u_t = D u_xx with a triple-mode exact solution and zero Dirichlet BCs."""

    @staticmethod
    def init_params(sd=0.1):
        static_params = {"dims": (1, 2), "sd": sd}
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics points
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        # need u_xx and u_t for residual r = u_t - D*u_xx
        required_ujs_phys = (
            (0, (0,0)),  # u_xx
            (0, (1,)),   # u_t
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def loss_fn(all_params, constraints):
        # physics residual: r = u_t - D * u_xx (homogeneous)
        x_batch, u_xx, u_t = constraints[0]
        r = u_t - D * u_xx
        return jnp.mean(r**2)

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Blend the network with exact BCs (x=0, x=L) and IC (t=0) using a smooth mask."""
        x, t = x_batch[:, 0:1], x_batch[:, 1:2]
        sd = all_params["static"]["problem"]["sd"]
        tanh = jnp.tanh

        mask_left   = tanh((x - 0.0) / sd)        # x=0
        mask_right  = tanh((L - x) / sd)          # x=L
        mask_bottom = tanh((t - 0.0) / sd)        # t=0
        mask = mask_left * mask_right * mask_bottom

        u_bc = _triple_mode_u(x, t)               # satisfies BCs & IC
        return mask * u + (1.0 - mask) * u_bc

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        return _triple_mode_u(x, t)

# ===========================
# Eval helpers (relative L2)
# ===========================
def _clone_pytree(tree):
    return tree_map(lambda x: x.copy() if hasattr(x, "copy") else x, tree)

def _rel_l2_percent(u_pred, u_true):
    num = jnp.linalg.norm(u_pred - u_true)
    den = jnp.linalg.norm(u_true) + 1e-12
    return float(100.0 * (num / den))

# Recompute final Rel L2 for PINN using the same internals as the trainer
def eval_pinn_rel_l2(all_params, n_test, domain, decomposition_init_kwargs, network, problem):
    from fbpinns.trainers import PINN_model_jit
    from fbpinns import networks

    mu_, sd_ = decomposition_init_kwargs["unnorm"]
    unnorm_fn = lambda u: networks.unnorm(mu_, sd_, u)
    model_fns = (domain.norm_fn, network.network_fn, unnorm_fn, problem.constraining_fn)

    x_test = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=n_test)
    u_true = problem.exact_solution(all_params=all_params, x_batch=x_test, batch_shape=n_test)
    u_pred, _ = PINN_model_jit(all_params, x_test, model_fns, verbose=False)
    return _rel_l2_percent(u_pred.reshape(-1,1), u_true.reshape(-1,1))

# Recompute final Rel L2 for FBPINN using the same internals as the trainer
def eval_fbpinn_rel_l2(all_params, n_test, domain, decomposition, problem):
    from fbpinns.trainers import FBPINN_model_jit, get_inputs

    x_test = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=n_test)
    active_all = jnp.ones(all_params["static"]["decomposition"]["m"], dtype=int)
    takes, all_ims, cut_all = get_inputs(x_test, active_all, all_params, decomposition)

    model_fns = (decomposition.norm_fn, decomposition.network_fn, decomposition.unnorm_fn,
                 decomposition.window_fn, problem.constraining_fn)

    all_params_cut = {"static":cut_all(all_params["static"]),
                      "trainable":cut_all(all_params["trainable"])}
    u_pred, *_ = FBPINN_model_jit(all_params_cut, x_test, takes, model_fns, verbose=False)
    u_true = problem.exact_solution(all_params=all_params, x_batch=x_test, batch_shape=n_test)
    return _rel_l2_percent(u_pred.reshape(-1,1), u_true.reshape(-1,1))

# ===========================
# Subclass trainers to track BEST Rel L2 during training
# ===========================
from fbpinns.trainers import PINNTrainer as _PINNTrainerBase, FBPINNTrainer as _FBPINNTrainerBase
from fbpinns import plot_trainer

class PINNTrainerBest(_PINNTrainerBase):
    def train(self, *args, **kwargs):
        self.best_rel_l2 = float('inf')   # ratio
        self.best_rel_l2_iter = -1
        self.best_all_params = None
        return super().train(*args, **kwargs)

    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, i, pstep, fstep, start0, all_params, model_fns, problem):
        c, writer = self.c, self.writer
        n_test = c.n_test

        from fbpinns.trainers import PINN_model_jit
        u_test, u_raw_test = PINN_model_jit(all_params, x_batch_test, model_fns, verbose=False)

        # single-field case
        l1 = (jnp.mean(jnp.abs(u_exact-u_test))/jnp.mean(jnp.abs(u_exact))).item()
        print(f"Relative L1 Error: {l1*100:.4f}%")
        l2_scalar = (jnp.sqrt(jnp.mean((u_exact-u_test)**2))/jnp.sqrt(jnp.mean((u_exact)**2))).item()
        print(f"Relative L2 Error: {l2_scalar*100:.4f}%")

        if l2_scalar < self.best_rel_l2:
            self.best_rel_l2 = l2_scalar
            self.best_rel_l2_iter = i
            self.best_all_params = {"static": _clone_pytree(all_params["static"]),
                                    "trainable": _clone_pytree(all_params["trainable"])}
            print(f"[PINN] ✅ New best Relative L2: {self.best_rel_l2*100:.4f}% at step {i}")

        l1_abs = jnp.mean(jnp.abs(u_exact-u_test)).item()
        l1n = l1_abs / u_exact.std().item()
        u_test_losses.append([i, pstep, fstep, time.time()-start0, l1_abs, l1n])
        writer.add_scalar("loss/test/l1_istep", l1_abs, i)

        if i % (c.test_freq * 10) == 0:
            fs = plot_trainer.plot("PINN", all_params["static"]["problem"]["dims"],
                x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test)
            if fs is not None:
                self._save_figs(i, fs)

        return u_test_losses


class FBPINNTrainerBest(_FBPINNTrainerBase):
    def train(self, *args, **kwargs):
        self.best_rel_l2 = float('inf')   # ratio
        self.best_rel_l2_iter = -1
        self.best_all_params = None
        return super().train(*args, **kwargs)

    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, test_inputs, i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition):
        c, writer = self.c, self.writer
        n_test = c.n_test

        from fbpinns.trainers import FBPINN_model_jit
        takes, all_ims, _ = test_inputs
        # reconstruct cut params like base does
        def _cut_all(d):
            from jax.tree_util import tree_map
            def cut_fn(p):
                return p[all_ims] if hasattr(p, "shape") and p.shape[0] >= all_ims.shape[0] else p
            return {cl_k: {k: tree_map(cut_fn, d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                           for k in d[cl_k]}
                    for cl_k in d}

        all_params_cut = {"static":_cut_all(all_params["static"]),
                          "trainable":_cut_all(all_params["trainable"])}

        u_test, wp_test_, us_test_, ws_test_, us_raw_test_ = FBPINN_model_jit(all_params_cut, x_batch_test, takes, model_fns, verbose=False)

        # single-field case
        l1 = (jnp.mean(jnp.abs(u_exact-u_test))/jnp.mean(jnp.abs(u_exact))).item()
        print(f"Relative L1 Error: {l1*100:.4f}%")
        l2_scalar = (jnp.sqrt(jnp.mean((u_exact-u_test)**2))/jnp.sqrt(jnp.mean((u_exact)**2))).item()
        print(f"Relative L2 Error: {l2_scalar*100:.4f}%")

        if l2_scalar < self.best_rel_l2:
            self.best_rel_l2 = l2_scalar
            self.best_rel_l2_iter = i
            self.best_all_params = {"static": _clone_pytree(all_params["static"]),
                                    "trainable": _clone_pytree(all_params["trainable"])}
            print(f"[FBPINN] ✅ New best Relative L2: {self.best_rel_l2*100:.4f}% at step {i}")

        l1_abs = jnp.mean(jnp.abs(u_exact-u_test)).item()
        l1n = l1_abs / u_exact.std().item()
        u_test_losses.append([i, pstep, fstep, time.time()-start0, l1_abs, l1n])
        writer.add_scalar("loss/test/l1_istep", l1_abs, i)

        if i % (c.test_freq * 10) == 0:
            fs = plot_trainer.plot("FBPINN", all_params["static"]["problem"]["dims"],
                x_batch_test, u_exact, u_test, us_test_, ws_test_, us_raw_test_, x_batch, all_params, i, active, decomposition, n_test)
            if fs is not None:
                self._save_figs(i, fs)

        return u_test_losses


# ===========================
# Run
# ===========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fbpinns.networks import FCN
    from fbpinns.decompositions import RectangularDecompositionND
    from fbpinns.constants import Constants

    np.random.seed(0)

    # Optional quick sanity grid (not plotted)
    batch_shape = (32, 32)
    x_batch = RectangularDomainND._rectangle_samplerND(None, "grid",
        np.array([0.0, 0.0]), np.array([L, 1.0]), batch_shape)

    # Shared decomposition
    decomposition = RectangularDecompositionND
    decomposition_init_kwargs = dict(
        subdomain_xs=[np.linspace(0.0, L, 15), np.linspace(0.0, 1.0, 15)],
        subdomain_ws=[0.15 * np.ones((15,)), 0.15 * np.ones((15,))],
        unnorm=(0.0, 1.0),
    )
    start=time.time()
    # ---------------- PINN ----------------
    print("PINNs")
    network = FCN
    network_init_kwargs = dict(layer_sizes=[2, 64, 64, 64, 1])

    c = Constants(
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=Diffusion,
        problem_init_kwargs={},
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        ns=((32, 32),),
        n_test=(32, 32),
        n_steps=100000,
        clear_output=True,
        show_figures=False,
    )
    print(c)

    from fbpinns.trainers import PINNTrainer  # for type consistency only
    pinn_run = PINNTrainerBest(c)
    pinn_all_params = pinn_run.train()

    # Final (recomputed) and Best
    try:
        pinn_final_l2 = eval_pinn_rel_l2(
            all_params=pinn_all_params, n_test=c.n_test,
            domain=domain, decomposition_init_kwargs=decomposition_init_kwargs,
            network=network, problem=Diffusion
        )
        print(f"[PINN] Final Relative L2 (recomputed): {pinn_final_l2:.4f}%")
    except Exception as e:
        print("[PINN] Final evaluation failed:", e)

    if pinn_run.best_rel_l2_iter >= 0:
        print(f"[PINN] Best Relative L2: {pinn_run.best_rel_l2*100:.4f}% at step {pinn_run.best_rel_l2_iter}")
    end=time.time()
    print(f"Setup and PINN time: {end-start:.2f} seconds")
    start=time.time()
    # ---------------- FBPINN ----------------
    print("FBPINNs")
    network_fb = FCN
    network_init_kwargs_fb = dict(layer_sizes=[2, 64, 64, 64, 1])

    c_fb = Constants(
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=Diffusion,
        problem_init_kwargs={},
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network_fb,
        network_init_kwargs=network_init_kwargs_fb,
        ns=((32, 32),),
        n_test=(32, 32),
        n_steps=100000,
        clear_output=True,
        show_figures=False,
    )
    print(c_fb)

    from fbpinns.trainers import FBPINNTrainer  # for type consistency only
    fbpinn_run = FBPINNTrainerBest(c_fb)
    fbpinn_all_params = fbpinn_run.train()

    # Final (recomputed) and Best
    try:
        fbpinn_final_l2 = eval_fbpinn_rel_l2(
            all_params=fbpinn_all_params, n_test=c_fb.n_test,
            domain=domain, decomposition=decomposition, problem=Diffusion
        )
        print(f"[FBPINN] Final Relative L2 (recomputed): {fbpinn_final_l2:.4f}%")
    except Exception as e:
        print("[FBPINN] Final evaluation failed:", e)

    if fbpinn_run.best_rel_l2_iter >= 0:
        print(f"[FBPINN] Best Relative L2: {fbpinn_run.best_rel_l2*100:.4f}% at step {fbpinn_run.best_rel_l2_iter}")
    end=time.time()
    print(f"FBPINN time: {end-start:.2f} seconds")
    # ---------------- Summary ----------------
    print("\n========= SUMMARY =========")
    if pinn_run.best_rel_l2_iter >= 0:
        print(f"PINN   Best: {pinn_run.best_rel_l2*100:.4f}% @ {pinn_run.best_rel_l2_iter}")
    if 'pinn_final_l2' in locals():
        print(f"PINN   Final: {pinn_final_l2:.4f}%")
    if fbpinn_run.best_rel_l2_iter >= 0:
        print(f"FBPINN Best: {fbpinn_run.best_rel_l2*100:.4f}% @ {fbpinn_run.best_rel_l2_iter}")
    if 'fbpinn_final_l2' in locals():
        print(f"FBPINN Final: {fbpinn_final_l2:.4f}%")
    print("===========================")
