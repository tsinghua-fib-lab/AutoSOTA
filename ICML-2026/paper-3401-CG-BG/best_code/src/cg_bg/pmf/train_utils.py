import pickle
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from cg_bg.utils import get_console, get_progress


@jax.jit
def train_step(state: TrainState, batch: dict) -> tuple[TrainState, float]:

    def loss_wrap(params):
        return loss_fn(params, state.apply_fn, batch)

    loss, grads = jax.value_and_grad(loss_wrap)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def loss_fn(params: FrozenDict, apply_fn: Callable, batch: dict) -> jnp.ndarray:

    def energy_fn(x):
        return apply_fn(params, x)

    x = batch["x"]
    fx_true = batch["force"]
    _, g_vjp = jax.vjp(energy_fn, x)
    fx_pred = -g_vjp(jax.numpy.ones_like(x))[0]
    loss = jnp.mean((fx_pred - fx_true) ** 2)

    return loss


class MBForceMatching:
    def __init__(self, dataloader, model, learning_rate, kT, rng):

        self.dataloader = dataloader
        self.model = model
        self.learning_rate = learning_rate
        self.rng = rng
        self.kT = kT

    def create_train_state(self):
        dummy_batch = next(iter(self.dataloader))["x"]
        dummy_batch = jax.tree.map(jnp.asarray, dummy_batch)
        params = self.model.init({"params": self.rng}, dummy_batch)
        tx = optax.adam(self.learning_rate)
        init_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
        )
        return init_state

    def train(self, epochs):
        state = self.create_train_state()
        epoch_losses = []

        best_loss = float("inf")
        self.best_params = None

        with get_progress() as progress:
            task_id = progress.add_task("Training", total=epochs)
            for _ in range(epochs):
                batch_losses = []

                for batch in self.dataloader:
                    batch = jax.tree.map(jnp.asarray, batch)
                    state, loss = train_step(state, batch)
                    batch_losses.append(float(loss))

                avg_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.best_params = jax.device_get(state.params)

                progress.update(
                    task_id,
                    advance=1,
                    description=f"Training (loss={avg_loss:.6f}, best={best_loss:.6f})",
                )

        self.state = state
        self.epoch_losses = epoch_losses
        self.final_params = state.params

    def save_energy_params(self, file_path, save_format=".pkl", best=False):
        params_to_save = self.best_params if best else self.final_params
        with open(file_path, "wb") as f:
            if save_format == ".pkl":
                pickle.dump(params_to_save, f)
            elif save_format == ".npy":
                jnp.save(f, params_to_save)
            else:
                raise ValueError(f"Unsupported file format: {save_format}")
        get_console().print(f"[green]Energy parameters saved to[/green] {file_path}")

    @staticmethod
    def V(x, y):
        term1 = -17.3 * jnp.exp(-0.0039 * (x - 48) ** 2 - 0.0391 * (y - 8) ** 2)
        term2 = -8.7 * jnp.exp(-0.0039 * (x - 32) ** 2 - 0.0391 * (y - 16) ** 2)
        term3 = -14.7 * jnp.exp(-0.0254 * (x - 24) ** 2 + 0.043 * (x - 24) * (y - 32) - 0.0254 * (y - 32) ** 2)
        term4 = 1.3 * jnp.exp(0.00273 * (x - 16) ** 2 + 0.0023 * (x - 16) * (y - 24) + 0.00273 * (y - 24) ** 2)
        return term1 + term2 + term3 + term4

    def evaluate(self, params, file_path):

        x_vals = jnp.linspace(0, 50, 300)
        y_vals = jnp.linspace(0, 50, 300)

        @jax.jit
        def calculate_true_pmf(xi_vals, yi_vals):
            def V_cg(xi):
                V_y = jax.vmap(lambda y: self.V(xi, y))(yi_vals)
                return -self.kT * jnp.log(jnp.trapezoid(jnp.exp(-V_y / self.kT), yi_vals))

            return jax.vmap(V_cg)(xi_vals)

        V_eff = calculate_true_pmf(x_vals, y_vals)
        V_eff -= jnp.min(V_eff)

        x_input = x_vals.reshape(-1, 1)
        dataset = {"R": x_input}
        energy = self.predict(dataset, params=params)["U"]
        energy -= jnp.min(energy)

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
        ax1.plot(x_vals, energy, lw=2, color="blue", label="Learned PMF")
        ax1.plot(x_vals, V_eff, lw=2, color="red", linestyle="--", label="True PMF")
        ax1.set_xlabel("x")
        ax1.set_ylabel("PMF")
        ax1.set_title("Learned vs True PMF")
        ax1.legend()
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def predict(self, dataset, params=None, batch_size=None):
        dataset = jax.tree.map(jnp.asarray, dataset)
        if params is None:
            params = self.final_params
        if "params" not in params:
            params = {"params": params}
        R = dataset["R"]
        n_samples = R.shape[0]
        if batch_size is None:
            batch_size = n_samples

        def batch_fn(batch_data):
            return self.model.apply(params, batch_data)

        remainder = n_samples % batch_size
        if remainder != 0:
            pad_len = batch_size - remainder
            padding = jnp.zeros((pad_len, *R.shape[1:]), dtype=R.dtype)
            R_padded = jnp.concatenate([R, padding], axis=0)
        else:
            pad_len = 0
            R_padded = R
        num_batches = R_padded.shape[0] // batch_size
        reshaped_inputs = R_padded.reshape(num_batches, batch_size, *R.shape[1:])
        energies_batched = jax.lax.map(batch_fn, reshaped_inputs)
        energies = energies_batched.reshape(-1)
        if pad_len > 0:
            energies = energies[:n_samples]

        return {"U": energies}
