import jax
import jax.numpy as jnp


class PlackettLuce():
    def __init__(self, logits):
        self.logits = logits
        self.dim = logits.shape[0]

    def sample(self, key, n_samples):
        logits = self.logits[None, :].repeat(n_samples, axis=0)
        u = jax.random.uniform(key, logits.shape)
        z = logits - jnp.log(-jnp.log(u))
        return jnp.argsort(z, axis=-1, descending=True)

    def log_prob(self, x):
        # x shape: (dim,) or (n_samples, dim)
        logits = self.logits[x] # Shape: (dim,) or (n_samples, dim)
        expanded_logits = jnp.expand_dims(logits, -2).repeat(logits.shape[-1], axis=-2)  # Shape: (n_samples, dim, dim)
        mask = jnp.triu(jnp.ones((logits.shape[-1], logits.shape[-1]), dtype=bool))
        if len(x.shape) == 2:
            mask = mask[None, ...].repeat(x.shape[0], axis=0)  # Shape: (n_samples, dim, dim)
        return (logits - jax.scipy.special.logsumexp(expanded_logits, where=mask, axis=-1)).sum(axis=-1)