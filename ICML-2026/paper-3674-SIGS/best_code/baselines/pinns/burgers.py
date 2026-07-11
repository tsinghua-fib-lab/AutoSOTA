# experiment: Burgers — trains PINN & FBPINN and reports BEST Relative L2 (%)

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
start=time.time()
# ---------------------------
# Domain
# ---------------------------
x_min = -5.0
x_max =  5.0

domain = RectangularDomainND
domain_init_kwargs = dict(
    xmin=np.array([x_min, 0.0]),
    xmax=np.array([x_max, 1.0]),
)

# ---------------------------
# Problem
# ---------------------------
class Burgers(Problem):
    """Solves the Burgers Equation with a manufactured solution."""

    @staticmethod
    def init_params(sd=0.1):
        static_params = {"dims": (1, 2), "sd": sd}
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, ()),      # u
            (0, (0,)),    # u_x
            (0, (0,0)),   # u_xx
            (0, (1,)),    # u_t
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch, u, u_x, u_xx, u_t = constraints[0]
        Lu = -0.01 * u_xx + u * u_x + u_t

        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        tanh = jnp.tanh

        f_u   = 0.6 * tanh(25.8 * t - 30.0 * x + 9.9) + 0.86
        f_ut  = 15.48 - 15.48 * tanh(25.8 * t - 30.0 * x + 9.9) ** 2
        f_ux  = 18.0 * tanh(25.8 * t - 30.0 * x + 9.9) ** 2 - 18.0
        f_uxx = 1080.0 * (tanh(25.8 * t - 30.0 * x + 9.9) ** 2 - 1) * tanh(25.8 * t - 30.0 * x + 9.9)
        f = -0.01 * f_uxx + f_u * f_ux + f_ut

        return jnp.mean((Lu - f) ** 2)

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        x, t = x_batch[:, 0:1], x_batch[:, 1:2]
        sd = all_params["static"]["problem"]["sd"]
        tanh = jnp.tanh

        mask_left   = tanh((x - x_min) / sd)
        mask_right  = tanh((x_max - x) / sd)
        mask_bottom = tanh((t - 0.0) / sd)
        mask = mask_left * mask_right * mask_bottom

        u_bc = 0.6 * tanh(25.8 * t - 30.0 * x + 9.9) + 0.86
        return mask * u + (1.0 - mask) * u_bc

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        tanh = jnp.tanh
        return 0.6 * tanh(25.8 * t - 30.0 * x + 9.9) + 0.86


# ===========================
# Eval helpers
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
    takes, _all_ims, cut_all = get_inputs(x_test, active_all, all_params, decomposition)

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

    # copy parent signature and add tracking
    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, i, pstep, fstep, start0, all_params, model_fns, problem):
        c, writer = self.c, self.writer
        n_test = c.n_test

        # PINN forward
        from fbpinns.trainers import PINN_model_jit
        u_test, u_raw_test = PINN_model_jit(all_params, x_batch_test, model_fns, verbose=False)

        # errors (single-field branch)
        l1 = (jnp.mean(jnp.abs(u_exact-u_test))/jnp.mean(jnp.abs(u_exact))).item()
        print(f"Relative L1 Error: {l1*100:.4f}%")
        l2_scalar = (jnp.sqrt(jnp.mean((u_exact-u_test)**2))/jnp.sqrt(jnp.mean((u_exact)**2))).item()
        print(f"Relative L2 Error: {l2_scalar*100:.4f}%")

        # track best
        if l2_scalar < self.best_rel_l2:
            self.best_rel_l2 = l2_scalar
            self.best_rel_l2_iter = i
            self.best_all_params = {"static": _clone_pytree(all_params["static"]),
                                    "trainable": _clone_pytree(all_params["trainable"])}
            print(f"[PINN] ✅ New best Relative L2: {self.best_rel_l2*100:.4f}% at step {i}")

        # book-keeping (keep same as base)
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

    # copy parent signature and add tracking
    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, test_inputs, i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition):
        c, writer = self.c, self.writer
        n_test = c.n_test

        # FBPINN forward (same as base)
        from fbpinns.trainers import FBPINN_model_jit
        takes, all_ims, _ = test_inputs
        all_params_cut = {"static":self._cut_all(all_params["static"], takes, all_ims),
                          "trainable":self._cut_all(all_params["trainable"], takes, all_ims)}
        u_test, wp_test_, us_test_, ws_test_, us_raw_test_ = FBPINN_model_jit(all_params_cut, x_batch_test, takes, model_fns, verbose=False)

        # single-field case errors & best tracking
        l1 = (jnp.mean(jnp.abs(u_exact-u_test))/jnp.mean(jnp.abs(u_exact))).item()
        print(f"Relative L1 Error: {l1*100:.4f}%")
        l2_scalar = (jnp.sqrt(jnp.mean((u_exact-u_test)**2))/jnp.sqrt(jnp.mean((u_exact)**2))).item()
        print(f"Relative L2 Error: {l2_scalar*100:.4f}%")

        if l2_scalar < self.best_rel_l2:
            self.best_rel_l2 = l2_scalar
            self.best_rel_l2_iter = i
            # params are merged just above in base code before calling _test
            self.best_all_params = {"static": _clone_pytree(all_params["static"]),
                                    "trainable": _clone_pytree(all_params["trainable"])}
            print(f"[FBPINN] ✅ New best Relative L2: {self.best_rel_l2*100:.4f}% at step {i}")

        # log book-keeping (stay compatible)
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

    # helper to mimic base cut_all for test inputs
    @staticmethod
    def _cut_all(d, takes, all_ims):
        from jax.tree_util import tree_map
        m_take, n_take, p_take, np_take, npou = takes
        def cut_fn(p):
            return p[all_ims] if hasattr(p, "shape") and p.shape[0] >= all_ims.shape[0] else p
        return {cl_k: {k: tree_map(cut_fn, d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                       for k in d[cl_k]}
                for cl_k in d}


# ===========================
# Run
# ===========================
if __name__ == "__main__":
    from fbpinns.networks import FCN
    from fbpinns.decompositions import RectangularDecompositionND
    from fbpinns.constants import Constants

    np.random.seed(0)

    # Optional quick sanity grid
    batch_shape = (80,80)
    x_batch = RectangularDomainND._rectangle_samplerND(None, "grid",
        np.array([x_min, 0.0]), np.array([x_max, 1.0]), batch_shape)
    # Note: we don't plot here; trainers will test/print

    # Shared decomposition
    decomposition = RectangularDecompositionND
    decomposition_init_kwargs = dict(
        subdomain_xs=[np.linspace(x_min, x_max, 15), np.linspace(0.0, 1.0, 15)],
        subdomain_ws=[1.5 * np.ones((15,)), 0.15 * np.ones((15,))],
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
        problem=Burgers,
        problem_init_kwargs={},
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        ns=((32,32),),
        n_test=(32,32),
        n_steps=20000,
        clear_output=True,
        show_figures=False,
    )
    print(c)

    pinn_run = PINNTrainerBest(c)
    pinn_all_params = pinn_run.train()
    # Final (recomputed) and Best
    try:
        pinn_final_l2 = eval_pinn_rel_l2(
            all_params=pinn_all_params, n_test=c.n_test,
            domain=domain, decomposition_init_kwargs=decomposition_init_kwargs,
            network=network, problem=Burgers
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
        problem=Burgers,
        problem_init_kwargs={},
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network_fb,
        network_init_kwargs=network_init_kwargs_fb,
        ns=((32,32),),
        n_test=(32,32),
        n_steps=20000,
        clear_output=True,
        show_figures=False,
    )
    print(c_fb)

    fbpinn_run = FBPINNTrainerBest(c_fb)
    fbpinn_all_params = fbpinn_run.train()
    # Final (recomputed) and Best
    try:
        fbpinn_final_l2 = eval_fbpinn_rel_l2(
            all_params=fbpinn_all_params, n_test=c_fb.n_test,
            domain=domain, decomposition=decomposition, problem=Burgers
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
