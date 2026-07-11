"""
The optimizer is based on the implementation in ScoreMD:
https://github.com/noegroup/ScoreMD/blob/main/src/scoremd/training/optimizer.py
"""

from typing import Optional

import optax


def get_lr_optimizer(
    epochs: int,
    num_samples: int,
    batch_size: int,
    learning_rate: float,
    min_learning_rate: float = None,
    clip: float = None,
    schedule: str = "constant",
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:

    num_steps = epochs * (num_samples // batch_size)
    if schedule == "constant":
        lr = learning_rate
    elif schedule == "cosine":
        if min_learning_rate is None:
            raise ValueError("min_learning_rate must be provided for cosine schedule")
        lr = optax.cosine_decay_schedule(learning_rate, num_steps, min_learning_rate / learning_rate)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    # inject_hyperparams exposes the (possibly scheduled) learning rate in opt_state so it
    # can be read back for logging via current_learning_rate.
    adamw = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=weight_decay)
    if clip is not None:
        return optax.chain(optax.clip(max_delta=clip), adamw)
    return adamw


def current_learning_rate(opt_state) -> Optional[float]:
    """Read the current learning rate from an inject_hyperparams optimizer state.

    Returns ``None`` if no learning-rate hyperparameter is found (e.g. a different optimizer).
    """
    found: list[float] = []

    def _search(node) -> None:
        hyperparams = getattr(node, "hyperparams", None)
        if isinstance(hyperparams, dict) and "learning_rate" in hyperparams:
            found.append(float(hyperparams["learning_rate"]))
        if isinstance(node, tuple):  # optax states are NamedTuples / chains of them
            for child in node:
                _search(child)

    _search(opt_state)
    return found[0] if found else None
