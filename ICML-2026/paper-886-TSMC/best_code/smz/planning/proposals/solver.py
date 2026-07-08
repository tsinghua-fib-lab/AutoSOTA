from typing import Callable, Any

import jax
import jax.numpy as jnp


def bisection_method(
        f: Callable[[jax.Array, dict[str, Any]], jax.Array],
        n: int,
        max_steps: int,
        step_size: float = 0.0
) -> Callable[..., jax.Array]:
    """Compile a bisection method.

    Converges to the root (if it exists) with O(n * sqrt(max_steps))

    Only supports 1-Dimensional search.
    """
    batch_fun = jax.vmap(f, in_axes=(0, None))

    def body(
            carry: tuple[tuple[jax.Array, jax.Array], dict],
            x: None = None
    ) -> tuple[
        tuple[tuple[jax.Array, jax.Array], dict],
        jax.Array | float
    ]:
        prev_bounds, kwargs = carry

        grid = jnp.linspace(*prev_bounds, 2 * n + 1)
        step = grid[1] - grid[0]
        values = batch_fun(grid, kwargs).squeeze()
        values = jnp.nan_to_num(values, nan=1e32, neginf=-1e10, posinf=1e10)

        # Transform values to make the function's roots an attractor.
        # Then argmax/ argmin can find the best bounds without array reshaping.
        best_positive = grid.at[jnp.argmax(1.0 / (values + 1e-32))].get()
        best_negative = grid.at[jnp.argmin(1.0 / (values - 1e-32))].get()

        # Construct new-bounds closest to the function root with some slack.
        new_bounds = (best_negative - step * step_size,
                      best_positive + step * step_size)

        return (new_bounds, kwargs), sum(new_bounds) / 2.0

    def run(
            bounds: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
            kwargs: dict[str, Any]
    ) -> jax.Array:
        _, best = jax.lax.scan(
            body, (bounds, kwargs), xs=None, length=max_steps
        )
        return best[-1]

    return run
