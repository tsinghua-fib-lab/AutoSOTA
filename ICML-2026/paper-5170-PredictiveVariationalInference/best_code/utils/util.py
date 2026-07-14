import jax.scipy as jsc
import jax.numpy as jnp

def invgamma_logp(x, a, loc, scale):
    return jsc.stats.gamma.logpdf(1 / x, a, loc, 1/scale) - jnp.log(x) * 2
