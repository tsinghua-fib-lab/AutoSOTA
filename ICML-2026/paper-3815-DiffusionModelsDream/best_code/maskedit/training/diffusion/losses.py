import jax
import jax.numpy as jnp

def denoising_score_matching_loss(key,times,xs_target,loss_mask,*args,model_fn,noise_fn,weight_fn,axis=-2,**kwargs):
    """This function computes the denoising score matching loss.

    Args:
        key: Random generator key.
        times: Time points.
        xs_target: Target distribution.
        loss_mask: Masking for the loss.
        model_fn: Score model model_fn(times, xs_t, *args, **kwargs) -> s_t.
        mean_fn: Mean function of the SDE.
        std_fn: Std function of the SDE.
        weight_fn: Weight function for the loss.
        axis: Axis to sum over. Defaults to -2.
        

    Returns:
        Array: Loss
    """

    xs_t, eps, std_t = noise_fn(key, xs_target, times)
    
    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)
    
    score_pred = model_fn(times, xs_t, *args, **kwargs)
    score_target = -eps / std_t

    loss = (score_pred - score_target) ** 2
    if loss_mask is not None:
        loss = jnp.where(loss_mask, 0.0,loss)

    weight_mask = 1./jnp.sum(loss_mask, 1)
    
    loss = weight_fn(times) * jnp.sum(loss, axis=axis, keepdims=True) * weight_mask
    loss = jnp.mean(loss)

    return loss


def denoising_eps_matching_loss(
    key, times, xs_target, loss_mask, *args,
    model_fn, noise_fn, weight_fn,
    axis=-2,
    use_time_weighting=False,
    likelihood_weighting=False,
    **kwargs
):
    """This function computes the denoising noise matching loss.

    Args:
        key: Random generator key.
        times: Time points.
        xs_target: Target distribution.
        loss_mask: Masking for the loss.
        model_fn: Score model model_fn(times, xs_t, *args, **kwargs) -> s_t.
        noise_fn: Noising function of SDE.
        weight_fn: Weight function for the loss.
        axis: Axis to sum over. Defaults to -2.
        use_time_weighting: DDPM time weighing, default is False.
        likelihood_weighing: VE likelihood weighing, default is False.
        

    Returns:
        Array: Loss
    """

    xs_t, eps, std_t = noise_fn(key, xs_target, times)

    if loss_mask is not None:
        loss_mask = loss_mask.reshape(xs_target.shape)
        xs_t = jnp.where(loss_mask, xs_target, xs_t)

    output = model_fn(times, xs_t, *args, **kwargs)

    if isinstance(output, tuple):
        eps_pred, norm_divider = output
        norm_divider = 1.0
        avg_divider = 1.0
    else:
        eps_pred = output
        norm_divider = 1.0
        avg_divider = 1.0

    loss = (eps_pred - eps) ** 2

    loss = loss / norm_divider

    # Optional likelihood/VE weighting (1 / sigma_t^2)
    if likelihood_weighting:
        loss = loss / (std_t ** 2)

    # Zero-out masked positions + normalize by number of unmasked elements per example
    if loss_mask is not None:
        loss = jnp.where(loss_mask, 0.0, loss)
        denom = jnp.maximum(1.0, jnp.sum(~loss_mask, axis=axis, keepdims=True))
    else:
        denom = jnp.array(1.0)
        
    loss = jnp.sum(loss, axis=axis, keepdims=True) / denom

    # Optional DSM-style time weighting
    if use_time_weighting:
        loss = weight_fn(times) * loss

    return jnp.mean(loss) * avg_divider

