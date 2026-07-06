"""Paper examples — prints outputs used in the paper figures and text."""

import jax
import jax.numpy as jnp
import softjax as sj


jnp.set_printoptions(precision=4, suppress=True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

# ── Soft comparison + where ───────────────────────────────────────────────────

x = jnp.array([0.1, 0.3, 0.7])
y = jnp.array([0.4, 0.2, 0.3])

cond = jnp.greater(x, y)
print("cond:", cond)
z = jnp.where(cond, x, y)
print("z:", z)

soft_cond = sj.greater(x, y)
print("soft_cond:", soft_cond.round(2))
z = sj.where(soft_cond, x, y)
print("soft z:", z.round(2))

# ── Soft argmax + indexing ────────────────────────────────────────────────────

x = jnp.array([0.1, 0.4, 0.8])

idx = jnp.argmax(x)
print("\nHard index:", idx)
y = jax.lax.dynamic_index_in_dim(x, idx)
print("Hard indexed value:", y)

soft_idx = sj.argmax(x)
print("Soft index:", soft_idx.round(3))
y = sj.dynamic_index_in_dim(x, soft_idx)
print("Soft indexed value:", y.round(3))

# ── Soft argsort + take_along_axis ────────────────────────────────────────────

x = jnp.array([0.3, 1.0, -0.5])

ind = jnp.argsort(x)
print("\nHard sort indices:", ind)
values = jnp.take_along_axis(x, ind)
print("Hard sorted values:", values)

soft_idx = sj.argsort(x)
print("Soft sort indices:", soft_idx.round(3))
soft_values = sj.take_along_axis(x, soft_idx)
print("Soft sorted values:", soft_values.round(3))
