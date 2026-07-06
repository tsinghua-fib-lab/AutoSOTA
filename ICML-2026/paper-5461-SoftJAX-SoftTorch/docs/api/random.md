# Random sampling

SoftJAX random sampling functions live under `softjax.random` and mirror the corresponding `jax.random` APIs while returning relaxed samples. Index-valued samplers such as `categorical` return `SoftIndex` objects; Boolean samplers such as `bernoulli` return numeric `SoftBool` values; value-returning samplers such as `choice` and `permutation` apply the relaxed indices to the sampled values.

In `mode="hard"` these functions call the corresponding JAX sampler. Soft modes reuse a matching random perturbation or threshold and replace the hard decision with a soft relaxation. The `_st` variants return hard samples in the forward pass and use the soft relaxation for gradients; when JAX's hard sampler uses a different internal random construction, the straight-through variant uses SoftJAX's internal hard path so the forward and backward branches share the same random perturbation.

!!! note "Shapes"
    `categorical` follows `jax.random.categorical` sample-shape semantics and appends the soft category axis as the last dimension. `choice` follows `jax.random.choice` by inserting the sample shape at the sampled axis. `bernoulli`, `binomial`, and `multinomial` follow the corresponding JAX shape broadcasting rules.

!!! note "Count-valued samplers"
    In soft modes, `binomial` and `multinomial` currently support scalar static counts. They are implemented as sums of independent relaxed Bernoulli or categorical samples.

!!! note "Log probabilities"
    `categorical(..., return_log_probs=True)` returns log relaxed sample weights, not categorical log-likelihoods under the original logits. Use `log_prob_eps` if sparse modes need bounded log values.

::: softjax.random.categorical

::: softjax.random.categorical_st

::: softjax.random.bernoulli

::: softjax.random.bernoulli_st

::: softjax.random.choice

::: softjax.random.choice_st

::: softjax.random.permutation

::: softjax.random.permutation_st

::: softjax.random.rademacher

::: softjax.random.rademacher_st

::: softjax.random.binomial

::: softjax.random.binomial_st

::: softjax.random.multinomial

::: softjax.random.multinomial_st
