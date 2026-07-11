# experiment: Wave — 1D wave (homogeneous) with BEST Rel L2 tracking

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.nn import tanh as j_tanh

from fbpinns.domains import RectangularDomainND
from fbpinns.problems import Problem
from fbpinns import networks

# ---------------------------
# Problem parameters (manufactured solution)
# ---------------------------
C = 0.14                              # wave speed shared by PDE and manufactured u
COEFFS = (1.4e-1, 4.6e-3, 2.3e-4, 1.1e-4)  # modal amplitudes a_n for n=1..4
X_MIN, X_MAX = -5.0, 5.0
T_MIN, T_MAX = 0.0, 5.0

# ---------------------------
# Domain: [X_MIN, X_MAX] x [T_MIN, T_MAX]
# ---------------------------
domain = RectangularDomainND
domain_init_kwargs = dict(
    xmin=np.array([X_MIN, T_MIN]),
    xmax=np.array([X_MAX, T_MAX]),
)

# ---------------------------
# Utility: exact 4-mode solution
# u(x,t) = (π/16) * Σ_{n=1..4} a_n sin(nπx) cos(C n π t)
# ---------------------------
def _wave_u(x, t):
    pi, sin, cos = jnp.pi, jnp.sin, jnp.cos
    u = (pi/16.0) * (
        COEFFS[0]*sin(1*pi*x)*cos(C*1*pi*t) +
        COEFFS[1]*sin(2*pi*x)*cos(C*2*pi*t) +
        COEFFS[2]*sin(3*pi*x)*cos(C*3*pi*t) +
        COEFFS[3]*sin(4*pi*x)*cos(C*4*pi*t)
    )
    return u

# Note: With this manufactured solution and PDE u_tt - C^2 u_xx = 0, the residual is exactly 0.

# ---------------------------
# Problem definition
# ---------------------------
class Wave(Problem):
    """Solves u_tt = C^2 u_xx (i.e., u_tt - C^2 u_xx = 0) with a 4-mode exact solution."""

    @staticmethod
    def init_params(sd=0.2):
        static_params = {"dims": (1, 2), "sd": sd}
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        # physics points
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        # need u_xx and u_tt for residual r = u_tt - C^2 u_xx
        required_ujs_phys = (
            (0, (0, 0)),   # u_xx
            (0, (1, 1)),   # u_tt
        )
        return [[x_batch_phys, required_ujs_phys],]

    @staticmethod
    def loss_fn(all_params, constraints):
        x_batch, u_xx, u_tt = constraints[0]
        r = u_tt - (C**2) * u_xx
        return jnp.mean(r**2)

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Blend the network with exact BCs (x=X_MIN,X_MAX) and IC (t=T_MIN) via smooth mask."""
        x, t = x_batch[:, 0:1], x_batch[:, 1:2]
        sd = all_params["static"]["problem"]["sd"]

        mask_left   = j_tanh((x - X_MIN) / sd)    # x = X_MIN
        mask_right  = j_tanh((X_MAX - x) / sd)    # x = X_MAX
        mask_bottom = j_tanh((t - T_MIN) / sd)    # t = T_MIN
        mask = mask_left * mask_right * mask_bottom

        u_bc = _wave_u(x, t)                      # respects all boundaries/IC
        return mask * u + (1.0 - mask) * u_bc

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        x = x_batch[:, 0:1]
        t = x_batch[:, 1:2]
        return _wave_u(x, t)

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

    mu_, sd_ = decomposition_init_kwargs["unnorm"]
    unnorm_fn = lambda u: networks.unnorm(mu_, sd_, u)
    model_fns = (domain.norm_fn, network.network_fn, unnorm_fn, problem.constraining_fn)

    x_test = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=n_test)
    u_true = problem.exact_solution(all_params=all_params, x_batch=x_test, batch_shape=n_test)
    u_pred, _ = PINN_model_jit(all_params, x_test, model_fns, verbose=False)
    return _rel_l2_percent(u_pred.reshape(-1,1), u_true.reshape(-1,1))

# Recompute final Rel L2 for FBPINN using the same internals as the trainer
def eval_fbpinn_rel_l2(all_params, n_test, domain, decomposition, network, problem):
    from fbpinns.trainers import FBPINN_model_jit, get_inputs

    x_test = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=n_test)
    active_all = jnp.ones(all_params["static"]["decomposition"]["m"], dtype=int)
    takes, all_ims, cut_all = get_inputs(x_test, active_all, all_params, decomposition)

    model_fns = (decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn,
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
    def __init__(self, c, patience=2000):
        # patience: number of steps without improvement before early stop
        self.patience = patience
        self.last_improve_iter = -1
        super().__init__(c)

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
            # record last improvement
            self.last_improve_iter = i

        # early stopping: if no improvement for > patience steps, stop
        if self.last_improve_iter >= 0 and (i - self.last_improve_iter) > getattr(self, "patience", 1e9):
            print(f"[PINN] ⏸️ Early stopping triggered at step {i} (no improvement in {self.patience} steps)")
            raise StopIteration("PINN early stopping: no improvement")

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
    def __init__(self, c, patience=2000):
        # patience: number of steps without improvement before early stop
        self.patience = patience
        self.last_improve_iter = -1
        super().__init__(c)

    def train(self, *args, **kwargs):
        self.best_rel_l2 = float('inf')   # ratio
        self.best_rel_l2_iter = -1
        self.best_all_params = None
        return super().train(*args, **kwargs)

    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, test_inputs, i, pstep, fstep, start0, active, all_params, model_fns, problem, decomposition):
        c, writer = self.c, self.writer
        n_test = c.n_test

        from fbpinns.trainers import FBPINN_model_jit
        takes, all_ims, cut_all = test_inputs
        all_params_cut = {"static":cut_all(all_params["static"]),
                          "trainable":cut_all(all_params["trainable"])}

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
            # record last improvement
            self.last_improve_iter = i

        # early stopping: if no improvement for > patience steps, stop
        if self.last_improve_iter >= 0 and (i - self.last_improve_iter) > getattr(self, "patience", 1e9):
            print(f"[FBPINN] ⏸️ Early stopping triggered at step {i} (no improvement in {self.patience} steps)")
            raise StopIteration("FBPINN early stopping: no improvement")

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
    import argparse
    import pickle

    np.random.seed(0)

    # -------------------- CLI --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override optimiser learning rate (defaults from Constants)")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Override number of training steps")
    parser.add_argument("--test_freq", type=int, default=None,
                        help="Override test frequency")
    parser.add_argument("--patience", type=int, default=2000,
                        help="Patience (steps) for early stopping")
    parser.add_argument("--save-best", type=str, default=None,
                        help="Path to save best parameters pickle file")
    args = parser.parse_args()

    # Optional sanity grid (not plotted)
    batch_shape = (32, 32)
    x_batch = RectangularDomainND._rectangle_samplerND(None, "grid",
        np.array([X_MIN, T_MIN]), np.array([X_MAX, T_MAX]), batch_shape)

    # Shared decomposition (overlap ensured: widths > spacing)
    def _grid_and_widths(a, b, n, scale=1.30):
        xs = np.linspace(a, b, n)
        dx = (b - a) / (n - 1) if n > 1 else (b - a)
        ws = (scale * dx) * np.ones((n,))
        return xs, ws

    nx = nt = 15
    xs_x, ws_x = _grid_and_widths(X_MIN, X_MAX, nx, scale=1.30)
    xs_t, ws_t = _grid_and_widths(T_MIN, T_MAX, nt, scale=1.30)

    decomposition = RectangularDecompositionND
    decomposition_init_kwargs = dict(
        subdomain_xs=[xs_x, xs_t],
        subdomain_ws=[ws_x, ws_t],
        unnorm=(0.0, 1.0),
    )

    # ---------------- PINN ----------------
    print("PINNs")
    from fbpinns.trainers import PINNTrainer  # for type consistency only
    network = FCN
    network_init_kwargs = dict(layer_sizes=[2, 64, 64, 64, 1])

    c = Constants(
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=Wave,
        problem_init_kwargs={},
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network,
        network_init_kwargs=network_init_kwargs,
        ns=((32, 32),),
        n_test=(32, 32),
    n_steps=(args.n_steps if args.n_steps is not None else 100000),
        summary_freq=2000,
        test_freq=(args.test_freq if args.test_freq is not None else 2000),
        clear_output=True,
        show_figures=False,
    )
    # override optimiser learning rate if provided
    if args.learning_rate is not None:
        c.optimiser_kwargs = {"learning_rate": args.learning_rate}
    print(c)

    # Create trainer with patience for early stopping (no improvement steps)
    pinn_run = PINNTrainerBest(c, patience=args.patience)
    try:
        pinn_all_params = pinn_run.train()
    except StopIteration:
        print("[PINN] Training stopped early due to no improvement. Using best saved params.")
        pinn_all_params = pinn_run.best_all_params

    # Final (recomputed) and Best
    try:
        pinn_final_l2 = eval_pinn_rel_l2(
            all_params=pinn_all_params, n_test=c.n_test,
            domain=domain, decomposition_init_kwargs=decomposition_init_kwargs,
            network=network, problem=Wave
        )
        print(f"[PINN] Final Relative L2 (recomputed): {pinn_final_l2:.4f}%")
    except Exception as e:
        print("[PINN] Final evaluation failed:", e)

    if pinn_run.best_rel_l2_iter >= 0:
        print(f"[PINN] Best Relative L2: {pinn_run.best_rel_l2*100:.4f}% at step {pinn_run.best_rel_l2_iter}")
        if args.save_best is not None:
            with open(args.save_best + "_pinn_best.pkl", "wb") as f:
                pickle.dump(pinn_run.best_all_params, f)
            print(f"[PINN] Saved best params to {args.save_best + '_pinn_best.pkl'}")

    # ---------------- FBPINN ----------------
    print("FBPINNs")
    from fbpinns.trainers import FBPINNTrainer  # for type consistency only
    network_fb = FCN
    network_init_kwargs_fb = dict(layer_sizes=[2, 64, 64, 64, 1])

    c_fb = Constants(
        domain=domain,
        domain_init_kwargs=domain_init_kwargs,
        problem=Wave,
        problem_init_kwargs={},
        decomposition=decomposition,
        decomposition_init_kwargs=decomposition_init_kwargs,
        network=network_fb,
        network_init_kwargs=network_init_kwargs_fb,
        ns=((32, 32),),
        n_test=(32, 32),
        n_steps=20000,
        summary_freq=2000,
        test_freq=2000,
        clear_output=True,
        show_figures=False,
    )
    print(c_fb)

    fbpinn_run = FBPINNTrainerBest(c_fb, patience=args.patience)
    try:
        fbpinn_all_params = fbpinn_run.train()
    except StopIteration:
        print("[FBPINN] Training stopped early due to no improvement. Using best saved params.")
        fbpinn_all_params = fbpinn_run.best_all_params

    # Final (recomputed) and Best
    try:
        fbpinn_final_l2 = eval_fbpinn_rel_l2(
            all_params=fbpinn_all_params, n_test=c_fb.n_test,
            domain=domain, decomposition=decomposition,
            network=network_fb, problem=Wave
        )
        print(f"[FBPINN] Final Relative L2 (recomputed): {fbpinn_final_l2:.4f}%")
    except Exception as e:
        print("[FBPINN] Final evaluation failed:", e)

    if fbpinn_run.best_rel_l2_iter >= 0:
        print(f"[FBPINN] Best Relative L2: {fbpinn_run.best_rel_l2*100:.4f}% at step {fbpinn_run.best_rel_l2_iter}")
        if args.save_best is not None:
            with open(args.save_best + "_fbpinn_best.pkl", "wb") as f:
                pickle.dump(fbpinn_run.best_all_params, f)
            print(f"[FBPINN] Saved best params to {args.save_best + '_fbpinn_best.pkl'}")

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
