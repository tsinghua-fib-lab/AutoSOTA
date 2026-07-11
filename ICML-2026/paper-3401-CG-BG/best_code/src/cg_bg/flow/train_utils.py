from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.core import FrozenDict
from torch.utils.data import DataLoader

from cg_bg.flow.ema import EMATrainState


def create_train_state(
    rng: jax.random.PRNGKey,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optax.GradientTransformation,
    ema_decay: Optional[float] = None,
) -> EMATrainState:

    dummy_input = next(iter(dataloader))["input"]
    example = jax.tree.map(jnp.array, dummy_input)
    if "features" in example:
        example["training"] = True
    params = model.init({"params": rng}, **example)

    if ema_decay is None:
        ema_decay = 0.0

    return EMATrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        decay=ema_decay,
    )


@jax.jit
def train_step(state: EMATrainState, batch: dict) -> tuple[EMATrainState, float]:

    def loss_wrap(params):
        return loss_fn(params, state.apply_fn, batch)

    loss, grads = jax.value_and_grad(loss_wrap)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def loss_fn(params: FrozenDict, apply_fn: Callable, batch: dict) -> jnp.ndarray:

    vt = batch["vt"]
    if "features" in batch["input"]:
        batch["input"]["training"] = True
    ut = apply_fn(
        params,
        **batch["input"],
    )
    # Conditional flow-matching objective: 0.5*||u_t - v_t||^2 dropping the v_t^2 constant.
    loss = jnp.sum(0.5 * ut**2 - ut * vt)

    return loss


def plot_train_loss(losses, save_path="train_loss.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linestyle="-", label="Train Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
