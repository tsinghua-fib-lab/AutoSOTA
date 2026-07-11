import jax
import jax.numpy as jnp

from .losses import denoising_eps_matching_loss
from maskedit.utils.indices import reverse_and_compact
from flax import nnx
import optax

from functools import partial

class DiffusionModel:
    """Diffusion model wrapper.

    Initializes masks, index transforms, EMA tracking, and constructs:
    -- pmap/jit training steps,
    -- pmap/jit validation steps,
    -- SDE sampler with masking.

    `step`, `pstep`, `val`, `pval`, and `sample` are the functions for
    training steps, validation, and sampling, with p indicating pmapped
    functions.
    """

    def __init__(self,diff,train_args, graphdef, tx):
        self.diffeq = diff
        self.T_min = train_args["diffusion"]["T_min"]
        self.batch_size = train_args["batch_size"]
        self.batch_features = train_args["nvals"]
        self.ids = train_args["ids"]
        self.rng = train_args["rng_nnx"]
        train_step, train_pmap_step, val_step, val_pmap_step = self.prepare_step(graphdef, tx)
        self.step, self.pstep, self.val, self.pval = (train_step, train_pmap_step, val_step, val_pmap_step)
        self.sample = self.prepare_sample()
        inds = train_args["indices"]
        self.phi, self.rev_indices = reverse_and_compact(inds)
        results_index = [item for i in train_args["results_index"] for item in inds[i]]
        results_index = [item for i in results_index for item in i]
        self.joint_mask = jnp.zeros((self.batch_size,self.batch_features),dtype=bool)
        self.posterior_mask = self.joint_mask.at[:,results_index].set(True)
        self.likelihood_mask = jnp.logical_not(self.posterior_mask)
        self.ema_handler = optax.ema(decay=0.999)

    def prepare_sample(self):

        @nnx.jit
        def exptopo(mask):
            expanded = mask[:,self.rev_indices,:]
            reduced = (self.phi @ expanded) == 0 # Set to true if no component, as if conditioning if doesn't exist, as just set loss to 0
            return reduced

        @nnx.jit(static_argnames=("shape", "steps"))
        def sample_sde(eps_fn, rng0, rng_nnx, *, shape, condition, node_ids, steps=500):
            """
            eps_fn(params, x, t_broadcast) -> noise pred (same shape as x)
            sde: has diffusion(t) and prior_sample(rng, shape)
            shape: (B, ...) batch-first
            """
            condition_mask, condition_value, topo_mask_comp = condition
            topo_mask = exptopo(topo_mask_comp)
            rng0, sample_key, model_key = jax.random.split(rng0, 3)
            x = self.diffeq.prior_sample(sample_key, shape)
            x = jnp.where(condition_mask, condition_value, x)
            dt = 1.0 / steps # 0 -> 1
            B, T, _ = shape
            
            dummy_t   = jnp.zeros((B, 1, 1), dtype=jnp.float32)
            dummy_x   = jnp.zeros((B, T, 1), dtype=jnp.float32)
            dummy_tmask = jnp.ones((B, T, 1), dtype=bool)
            dummy_ids = jnp.tile(node_ids, (B, 1))
            dummy_cmask = jnp.zeros((B, T, 1), dtype=bool)

            output = eps_fn(dummy_t, dummy_x, dummy_tmask, dummy_ids, dummy_cmask)
            if isinstance(output, tuple):
                eps_pred, norm_divider = output
            else:
                eps_pred = output
            
            graphdef, state = nnx.split(eps_fn)

            def one_step(carry, k):

                def _broadcast_t(t, x):
                    while t.ndim < x.ndim:
                        t = t[..., None]
                    return t
                
                x, state = carry
                mdl = nnx.merge(graphdef, state)
                rng = jax.random.fold_in(rng0, k)
                sample_key, model_key = jax.random.split(rng, 2)
                t_scalar = (k / steps)
                t = jnp.full((shape[0],), t_scalar, dtype=x.dtype)
                t_b = _broadcast_t(t, x)
    
                # (eps-prediction -> score conversion)
                output = mdl(t_b, x, jnp.logical_not(topo_mask), jnp.tile(node_ids,(x.shape[0],1)), condition_mask)#, keys=model_key)
                if isinstance(output, tuple):
                    eps_pred, norm_divider = output
                else:
                    eps_pred = output

                x_next = self.diffeq.step(sample_key, eps_pred, x, t_b, dt, shape)

                # Apply overwrite of conditioned coords
                x = jnp.where(condition_mask, condition_value, x_next)

                # Collect state
                _, state = nnx.split(mdl)

                return (x,state), None
            
            (x, state_out), _ = jax.lax.scan(one_step, (x,state), jnp.arange(steps))
            x = jnp.where(topo_mask, jnp.nan, x)
            return x
        
        return sample_sde


    def prepare_step(self, graphdef, tx):
        B = self.batch_size
        T_min = self.T_min
        ids = self.ids
        diffeq = self.diffeq
        T = self.batch_features

        @nnx.jit
        def exptopo(mask):
            expanded = mask[:,self.rev_indices,:]
            reduced = (self.phi @ expanded) == 0
            return reduced
        
        def gen_cond_mask(rng):
            
            rng_select, rng_cond = jax.random.split(rng)

            # Random mask
            def joint(k): return self.joint_mask
            def post(k): return self.posterior_mask
            def like(k): return self.likelihood_mask
            def rand03(k): return jax.random.bernoulli(k, 0.3, shape=(self.batch_size, self.batch_features))
            def rand07(k): return jax.random.bernoulli(k, 0.7, shape=(self.batch_size, self.batch_features))

            # Choose one for the batch
            kind_idx = jax.random.randint(rng_select, (), 0, 5)
            condition_mask = jax.lax.switch(kind_idx, (joint, post, like, rand03, rand07), rng_cond)

            # Avoid all ones
            condition_mask = jnp.where(jnp.all(condition_mask, axis=-1, keepdims=True),False,condition_mask)
            condition_mask = condition_mask[..., None]

            return condition_mask

        @partial(jax.pmap,
                        axis_name="device",
                        in_axes=(0, 0, 0, 0), # state, rest, batch, data_key
                        out_axes=(None), # Loss is scalar
                        )
        def val_step_pmap(state, rest, batch, rng):
            mdl = nnx.merge(graphdef, state, rest)
            rng_t, rng_noise, rng_cond, rng_model = jax.random.split(rng, 4)

            condition_mask = gen_cond_mask(rng_cond)
            
            topo_mask = exptopo(batch["mask"])

            condition_topo_mask = jnp.logical_or(topo_mask,condition_mask)
            
            times = jax.random.uniform(rng_t, (B,1,1), minval=T_min, maxval=1.0)
            loss = denoising_eps_matching_loss(
                rng_noise, times, batch["x"], condition_topo_mask,
                model_fn=mdl, noise_fn=diffeq.add_noise, weight_fn=diffeq.weight_fn, topo_mask=jnp.logical_not(topo_mask),
                node_ids=jnp.tile(ids,(B,1)), condition_mask=condition_mask)#, keys=rng_model
                # )
            return jax.lax.pmean(loss, axis_name="device")

        @nnx.jit
        def val_step(state, rest, batch, rng):
            mdl = nnx.merge(graphdef, state, rest)
            mdl.eval()
            rng_t, rng_noise, rng_cond, rng_model = jax.random.split(rng, 4)

            condition_mask = gen_cond_mask(rng_cond)
            
            topo_mask = exptopo(batch["mask"])

            condition_topo_mask = jnp.logical_or(topo_mask,condition_mask)
            
            times = jax.random.uniform(rng_t, (B,1,1), minval=T_min, maxval=1.0)
            return denoising_eps_matching_loss(
                rng_noise, times, batch["x"], condition_topo_mask,
                model_fn=mdl, noise_fn=diffeq.add_noise, weight_fn=diffeq.weight_fn, topo_mask=jnp.logical_not(topo_mask),
                node_ids=jnp.tile(ids,(B,1)), condition_mask=condition_topo_mask)
            # )

        @nnx.jit
        def loss_and_state(state, rest, batch, *, rng_key_for_data):
            mdl = nnx.merge(graphdef, state, rest)
            mdl.train()

            rng_t, rng_noise, rng_cond, rng_model = jax.random.split(rng_key_for_data, 4)

            condition_mask = gen_cond_mask(rng_cond)

            topo_mask = exptopo(batch["mask"])

            condition_topo_mask = jnp.logical_or(topo_mask,condition_mask)
            
            times = jax.random.uniform(rng_t, (B,1,1), minval=T_min, maxval=1.0)

            loss = denoising_eps_matching_loss(
                rng_noise, times, batch["x"], condition_topo_mask,
                model_fn=mdl, noise_fn=diffeq.add_noise, weight_fn=diffeq.weight_fn, topo_mask=jnp.logical_not(topo_mask),
                node_ids=jnp.tile(ids,(B,1)), condition_mask=condition_mask)
        
            # Split out updated states
            _, param_state, rest_out = nnx.split(mdl, nnx.Param, ...)
            
            return loss, (param_state, rest_out)

        @nnx.jit
        def train_step(state, rest, opt_state, ema_state, batch, data_key):
            data_key, use_key = jax.random.split(data_key)
            (loss, (param_state, rest_out)), grads = nnx.value_and_grad(
                lambda st: loss_and_state(st, rest, batch, rng_key_for_data=use_key), has_aux=True
            )(state)
        
            updates, new_opt_state = tx.update(grads, opt_state, params=param_state)
            new_params = optax.apply_updates(param_state, updates)
        
            return new_params, rest_out, new_opt_state, loss, data_key

        @partial(jax.pmap,
            axis_name="device",
            in_axes=(0, 0, 0, 0, 0, 0),
            out_axes=(0, 0, 0, 0, None, 0),
            donate_argnums=(0, 1, 2))
        def train_step_pmap(state, rest, opt_state, ema_state, batch, data_key):
            data_key, use_key = jax.random.split(data_key)
    
            (loss, (param_state, rest_out)), grads = nnx.value_and_grad(
                lambda st: loss_and_state(st, rest, batch, rng_key_for_data=use_key),
                has_aux=True
            )(state)
    
            grads = jax.lax.pmean(grads, axis_name="device")
            loss  = jax.lax.pmean(loss,  axis_name="device")
    
            updates, new_opt_state = tx.update(grads, opt_state, params=param_state)
            new_params = optax.apply_updates(param_state, updates)
            _, new_ema_state = self.ema_handler.update(new_params, ema_state)
                
            return new_params, rest_out, new_opt_state, new_ema_state, loss, data_key
                
        return train_step, train_step_pmap, val_step, val_step_pmap