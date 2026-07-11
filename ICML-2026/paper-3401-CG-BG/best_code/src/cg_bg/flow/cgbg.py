from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from torch.utils.data import DataLoader

from cg_bg.flow.ema import EMATrainState
from cg_bg.flow.train_utils import plot_train_loss, train_step
from cg_bg.utils import get_progress


def train(
    epochs: int,
    dataloader: DataLoader,
    state: EMATrainState,
) -> Tuple[EMATrainState, jnp.ndarray]:

    epoch_losses = []
    with get_progress() as progress:
        task_id = progress.add_task("Training", total=epochs)
        for epoch in range(1, epochs + 1):
            batch_losses = []

            for batch in dataloader:
                batch = jax.tree.map(jnp.asarray, batch)
                state, loss = train_step(state, batch)
                batch_losses.append(loss)

            avg_loss = jnp.mean(jnp.array(batch_losses))
            epoch_losses.append(avg_loss)
            progress.update(task_id, advance=1, description=f"Training (loss={avg_loss:.6f})")
    epoch_losses = jnp.array(epoch_losses)
    plot_train_loss(epoch_losses)

    return state


def sample(
    features: jnp.ndarray,
    sampler: Callable,
    state: EMATrainState,
    species: jnp.ndarray = None,
    box: jnp.ndarray = None,
    std: float = None,
) -> dict:

    x, logp = sampler(
        params=state.params,
        apply_fn=state.apply_fn,
        features=features,
        std=std,
    )
    samples = {"R": x, "logp": logp}

    if species is not None:
        species = jnp.broadcast_to(species, (x.shape[0], *species.shape))
        samples["species"] = species

    if box is not None:
        box = jnp.broadcast_to(box, (x.shape[0], *box.shape))
        samples["box"] = box

    return samples


def energy_evaluate(
    data: dict,
    trainer: Any,
    params: FrozenDict,
) -> dict:
    data_eval = dict(data)
    data_eval["R"] = jnp.squeeze(jnp.asarray(data["R"]))
    data_eval["U"] = jnp.zeros((data_eval["R"].shape[0],))
    if "box" in data_eval:
        data_eval["R"] = data_eval["R"] / data_eval["box"][0, 0, 0]

    predictions = trainer.predict(dataset=data_eval, params=params, batch_size=500)
    energy = predictions["U"]

    return energy


def compute_log_weights(
    kT: float,
    data: dict,
    energy: jnp.ndarray,
) -> jnp.ndarray:

    logp = data["logp"]
    logw = -energy / kT - logp
    logw = logw - jax.scipy.special.logsumexp(logw)

    return logw
