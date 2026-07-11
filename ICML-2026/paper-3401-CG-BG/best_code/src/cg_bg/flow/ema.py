import optax
from flax.core import FrozenDict
from flax.training import train_state


class EMATrainState(train_state.TrainState):
    ema_params: FrozenDict
    decay: float

    @classmethod
    def create(
        cls,
        *,
        apply_fn,
        params: FrozenDict,
        tx: optax.GradientTransformation,
        decay: float = 0.999,
    ):
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            ema_params=params,
            decay=decay,
        )

    def apply_gradients(self, *, grads, **kwargs):
        new_state = super().apply_gradients(grads=grads, **kwargs)
        new_ema_params = optax.incremental_update(
            new_state.params,
            self.ema_params,
            step_size=1.0 - self.decay,
        )

        return new_state.replace(ema_params=new_ema_params)
