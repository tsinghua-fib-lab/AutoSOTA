strategize_jax_cache_dir <- function() {
  cache_dir <- Sys.getenv("STRATEGIZE_JAX_CACHE_DIR", unset = "")
  if (!nzchar(cache_dir)) {
    cache_dir <- file.path("~", ".cache", "strategize", "jax_compilation_cache")
  }
  path.expand(cache_dir)
}

strategize_configure_jax_compilation_cache <- function(jax = NULL) {
  if (is.null(jax)) {
    jax <- strenv$jax
  }
  if (is.null(jax)) {
    return(invisible(FALSE))
  }

  cache_dir <- strategize_jax_cache_dir()
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(cache_dir)) {
    return(invisible(FALSE))
  }

  configured <- FALSE
  tryCatch({
    jax$config$update("jax_compilation_cache_dir", cache_dir)
    configured <- TRUE
  }, error = function(e) NULL)
  tryCatch({
    jax$config$update("jax_persistent_cache_min_compile_time_secs", 0)
  }, error = function(e) NULL)

  if (!isTRUE(configured)) {
    tryCatch({
      cache_mod <- reticulate::import(
        "jax.experimental.compilation_cache.compilation_cache",
        delay_load = TRUE
      )
      cache_mod$set_cache_dir(cache_dir)
      configured <- TRUE
    }, error = function(e) NULL)
  }

  strenv$jax_compilation_cache_dir <- cache_dir
  invisible(isTRUE(configured))
}

strategize_jax_block_until_ready <- function(x, max_depth = 20L) {
  walk <- function(value, depth) {
    if (is.null(value) || depth > max_depth) {
      return(invisible(value))
    }

    blocked <- tryCatch({
      block_fn <- value$block_until_ready
      if (!is.null(block_fn)) {
        block_fn()
        TRUE
      } else {
        FALSE
      }
    }, error = function(e) FALSE)
    if (isTRUE(blocked)) {
      return(invisible(value))
    }

    if (is.list(value) && !is.data.frame(value)) {
      for (item in value) {
        walk(item, depth + 1L)
      }
      return(invisible(value))
    }

    if (requireNamespace("reticulate", quietly = TRUE)) {
      is_py <- tryCatch(reticulate::is_py_object(value), error = function(e) FALSE)
      if (isTRUE(is_py)) {
        parts <- tryCatch(as.list(value), error = function(e) NULL)
        if (is.list(parts) && length(parts) > 0L) {
          for (item in parts) {
            walk(item, depth + 1L)
          }
        }
      }
    }

    invisible(value)
  }

  walk(x, 0L)
  invisible(x)
}

strategize_register_jax_transformer_helpers <- function() {
  if (!is.null(strenv$jax_transformer_scan_standard)) {
    return(invisible(TRUE))
  }
  helper_code <- paste(
    "import jax",
    "import jax.numpy as jnp",
    "",
    "def _strategize_rms_norm(x, g, eps=1e-6):",
    "    if g is None:",
    "        return x",
    "    g = jnp.asarray(g, dtype=x.dtype)",
    "    mean_sq = jnp.mean(x * x, axis=-1, keepdims=True)",
    "    inv_rms = jnp.reciprocal(jnp.sqrt(mean_sq + eps))",
    "    return x * inv_rms * g",
    "",
    "def _strategize_attention_dtype(dtype_label, backend, original_dtype):",
    "    dtype_label = str(dtype_label or 'auto').lower()",
    "    backend = str(backend or 'xla').lower()",
    "    if dtype_label in ('bf16', 'bfloat16'):",
    "        return jnp.bfloat16",
    "    if dtype_label in ('fp16', 'float16', 'f16', 'half'):",
    "        return jnp.float16",
    "    if dtype_label in ('fp32', 'float32', 'f32'):",
    "        return jnp.float32",
    "    if backend == 'cudnn':",
    "        return jnp.bfloat16",
    "    # 'auto' on non-cudnn resolves to float32, matching the R-side",
    "    # resolver (neural_attention_dtype_object) so the scan and non-scan",
    "    # paths run attention in the same dtype even if the global compute",
    "    # dtype changes (e.g. an x64 pin).",
    "    return jnp.float32",
    "",
    "def _strategize_next_multiple(x, multiple):",
    "    multiple = int(multiple)",
    "    if multiple <= 1:",
    "        return int(x)",
    "    return int(((int(x) + multiple - 1) // multiple) * multiple)",
    "",
    "def _strategize_self_attention(Qh, Kh, Vh, token_mask, model_dims, n_heads, head_dim, attention_backend, attention_dtype, attention_padding_multiple):",
    "    backend = str(attention_backend or 'xla').lower()",
    "    if backend not in ('xla', 'cudnn') or not hasattr(jax.nn, 'dot_product_attention'):",
    "        scale = jnp.sqrt(jnp.asarray(float(head_dim), dtype=Qh.dtype))",
    "        scores = jnp.einsum('nqhd,nkhd->nhqk', Qh, Kh) / scale",
    "        if token_mask is not None:",
    "            mask_use = jnp.reshape((token_mask > 0).astype(scores.dtype),",
    "                                   (token_mask.shape[0], 1, 1, token_mask.shape[1]))",
    "            scores = jnp.where(mask_use > 0, scores, jnp.asarray(jnp.finfo(scores.dtype).min, dtype=scores.dtype))",
    "        attn = jax.nn.softmax(scores, axis=-1)",
    "        return jnp.einsum('nhqk,nkhd->nqhd', attn, Vh)",
    "    original_dtype = Qh.dtype",
    "    seq_len = int(Qh.shape[1])",
    "    seq_use = seq_len",
    "    dtype_use = _strategize_attention_dtype(attention_dtype, backend, original_dtype)",
    "    Q_use = Qh.astype(dtype_use)",
    "    K_use = Kh.astype(dtype_use)",
    "    V_use = Vh.astype(dtype_use)",
    "    if token_mask is None:",
    "        mask_use = jnp.ones((Qh.shape[0], seq_len), dtype=jnp.float32)",
    "    else:",
    "        mask_use = token_mask",
    "    if backend == 'cudnn':",
    "        seq_use = _strategize_next_multiple(seq_len, attention_padding_multiple)",
    "        pad_n = seq_use - seq_len",
    "        if pad_n > 0:",
    "            pad_shape = (Qh.shape[0], pad_n, int(n_heads), int(head_dim))",
    "            Q_use = jnp.concatenate([Q_use, jnp.zeros(pad_shape, dtype=Q_use.dtype)], axis=1)",
    "            K_use = jnp.concatenate([K_use, jnp.zeros(pad_shape, dtype=K_use.dtype)], axis=1)",
    "            V_use = jnp.concatenate([V_use, jnp.zeros(pad_shape, dtype=V_use.dtype)], axis=1)",
    "            mask_use = jnp.concatenate([mask_use, jnp.zeros((Qh.shape[0], pad_n), dtype=mask_use.dtype)], axis=1)",
    "    key_mask = jnp.reshape(mask_use > 0, (Qh.shape[0], 1, 1, seq_use))",
    "    if backend == 'cudnn':",
    "        attn_mask = jnp.broadcast_to(key_mask, (Qh.shape[0], 1, seq_use, seq_use))",
    "    else:",
    "        attn_mask = key_mask",
    "    context_h = jax.nn.dot_product_attention(Q_use, K_use, V_use, mask=attn_mask, implementation=backend)",
    "    if seq_use != seq_len:",
    "        context_h = context_h[:, :seq_len, :, :]",
    "    return context_h.astype(original_dtype)",
    "",
    "def strategize_transformer_scan_standard(",
    "    tokens, token_mask, W_q_layers, W_k_layers, W_v_layers, W_o_layers,",
    "    W_ff1_layers, W_ff2_layers, RMS_attn_layers, RMS_ff_layers,",
    "    RMS_q_layers, RMS_k_layers, alpha_attn_layers, alpha_ff_layers,",
    "    RMS_final, model_dims, n_heads, head_dim,",
    "    attention_backend='xla', attention_dtype='auto', attention_padding_multiple=8",
    "):",
    "    model_dims = int(model_dims)",
    "    n_heads = int(n_heads)",
    "    head_dim = int(head_dim)",
    "    attention_padding_multiple = int(attention_padding_multiple)",
    "    use_qk_norm = RMS_q_layers is not None and RMS_k_layers is not None",
    "    if not use_qk_norm:",
    "        RMS_q_layers = jnp.zeros((W_q_layers.shape[0], head_dim), dtype=tokens.dtype)",
    "        RMS_k_layers = jnp.zeros((W_q_layers.shape[0], head_dim), dtype=tokens.dtype)",
    "",
    "    def body(tokens_carry, layer):",
    "        (Wq, Wk, Wv, Wo, Wff1, Wff2, RMS_attn, RMS_ff,",
    "         RMS_q, RMS_k, alpha_attn, alpha_ff) = layer",
    "        tokens_norm = _strategize_rms_norm(tokens_carry, RMS_attn)",
    "        Q = jnp.einsum('ntm,mk->ntk', tokens_norm, Wq)",
    "        K = jnp.einsum('ntm,mk->ntk', tokens_norm, Wk)",
    "        V = jnp.einsum('ntm,mk->ntk', tokens_norm, Wv)",
    "        Qh = jnp.reshape(Q, (Q.shape[0], Q.shape[1], n_heads, head_dim))",
    "        Kh = jnp.reshape(K, (K.shape[0], K.shape[1], n_heads, head_dim))",
    "        Vh = jnp.reshape(V, (V.shape[0], V.shape[1], n_heads, head_dim))",
    "        if use_qk_norm:",
    "            Qh = _strategize_rms_norm(Qh, RMS_q)",
    "            Kh = _strategize_rms_norm(Kh, RMS_k)",
    "        context_h = _strategize_self_attention(",
    "            Qh, Kh, Vh, token_mask, model_dims, n_heads, head_dim,",
    "            attention_backend, attention_dtype, attention_padding_multiple",
    "        )",
    "        context = jnp.reshape(context_h, (context_h.shape[0], context_h.shape[1], model_dims))",
    "        attn_out = jnp.einsum('ntm,mk->ntk', context, Wo)",
    "        h1 = tokens_carry + alpha_attn * attn_out",
    "        h1_norm = _strategize_rms_norm(h1, RMS_ff)",
    "        ff_pre = jnp.einsum('ntm,mf->ntf', h1_norm, Wff1)",
    "        gate, value = jnp.split(ff_pre, 2, axis=-1)",
    "        ff_act = jax.nn.swish(gate) * value",
    "        ff_out = jnp.einsum('ntf,fm->ntm', ff_act, Wff2)",
    "        return h1 + alpha_ff * ff_out, None",
    "",
    "    xs = (W_q_layers, W_k_layers, W_v_layers, W_o_layers,",
    "          W_ff1_layers, W_ff2_layers, RMS_attn_layers, RMS_ff_layers,",
    "          RMS_q_layers, RMS_k_layers, alpha_attn_layers, alpha_ff_layers)",
    "    tokens_out, _ = jax.lax.scan(body, tokens, xs)",
    "    return _strategize_rms_norm(tokens_out, RMS_final)",
    sep = "\n"
  )
  reticulate::py_run_string(helper_code)
  strenv$jax_transformer_scan_standard <- reticulate::py$strategize_transformer_scan_standard
  invisible(TRUE)
}

strategize_register_jax_svi_helpers <- function() {
  if (!is.null(strenv$jax_svi_update) &&
      !is.null(strenv$jax_svi_update_scan) &&
      !is.null(strenv$jax_svi_update_stable_available) &&
      !is.null(strenv$jax_svi_update_jit_cache_info) &&
      !is.null(strenv$jax_svi_update_jit_cache_clear) &&
      !is.null(strenv$jax_svi_gradient_diagnostics) &&
      !is.null(strenv$jax_svi_gradient_jit_cache_info) &&
      !is.null(strenv$jax_svi_gradient_jit_cache_clear)) {
    return(invisible(TRUE))
  }
  helper_code <- paste(
    "import jax",
    "import jax.numpy as jnp",
    "import numpyro.infer.svi as _svi_mod",
    "",
    "_strategize_svi_update_jit_cache = {}",
    "_strategize_svi_update_jit_compile_count = 0",
    "_strategize_svi_gradient_jit_cache = {}",
    "_strategize_svi_gradient_jit_compile_count = 0",
    "",
    "def _strategize_svi_update_arg_names(batch_args):",
    "    if batch_args is None:",
    "        return tuple()",
    "    if not hasattr(batch_args, 'keys'):",
    "        raise TypeError('SVI batch args must be a mapping with named arguments.')",
    "    return tuple(batch_args.keys())",
    "",
    "def _strategize_svi_update_cache_key(mode, svi, names, forward_mode_differentiation):",
    "    return (mode, id(svi), tuple(names), bool(forward_mode_differentiation))",
    "",
    "def strategize_svi_update_stable_available(svi):",
    "    return hasattr(svi, 'stable_update')",
    "",
    "def _strategize_svi_update_call(svi, state, args_in, forward_mode_differentiation=False):",
    "    update_method = svi.stable_update if hasattr(svi, 'stable_update') else svi.update",
    "    return update_method(",
    "        state,",
    "        forward_mode_differentiation=forward_mode_differentiation,",
    "        **args_in",
    "    )",
    "",
    "def strategize_svi_update_jit(svi, svi_state, batch_args, forward_mode_differentiation=False):",
    "    global _strategize_svi_update_jit_compile_count",
    "    names = _strategize_svi_update_arg_names(batch_args)",
    "    args = {name: batch_args[name] for name in names}",
    "    key = _strategize_svi_update_cache_key('single', svi, names, forward_mode_differentiation)",
    "    fn = _strategize_svi_update_jit_cache.get(key)",
    "    if fn is None:",
    "        _strategize_svi_update_jit_compile_count += 1",
    "        def _compiled(state, args_in):",
    "            return _strategize_svi_update_call(",
    "                svi, state, args_in, forward_mode_differentiation",
    "            )",
    "        fn = jax.jit(_compiled)",
    "        _strategize_svi_update_jit_cache[key] = fn",
    "    return fn(svi_state, args)",
    "",
    "def strategize_svi_update_scan_jit(svi, svi_state, batch_args_chunks, forward_mode_differentiation=False):",
    "    global _strategize_svi_update_jit_compile_count",
    "    names = _strategize_svi_update_arg_names(batch_args_chunks)",
    "    chunks = {name: batch_args_chunks[name] for name in names}",
    "    key = _strategize_svi_update_cache_key('scan', svi, names, forward_mode_differentiation)",
    "    fn = _strategize_svi_update_jit_cache.get(key)",
    "    if fn is None:",
    "        _strategize_svi_update_jit_compile_count += 1",
    "        def _compiled(state, chunks_in):",
    "            def body(step_state, step_args):",
    "                next_state, loss = _strategize_svi_update_call(",
    "                    svi, step_state, step_args, forward_mode_differentiation",
    "                )",
    "                return next_state, loss",
    "            return jax.lax.scan(body, state, chunks_in)",
    "        fn = jax.jit(_compiled)",
    "        _strategize_svi_update_jit_cache[key] = fn",
    "    return fn(svi_state, chunks)",
    "",
    "def _strategize_svi_gradient_cache_key(svi, names):",
    "    return (id(svi), tuple(names))",
    "",
    "def _strategize_svi_gradient_stats(grads):",
    "    leaves = jax.tree_util.tree_leaves(grads)",
    "    sq_sum = jnp.asarray(0.0)",
    "    max_abs = jnp.asarray(0.0)",
    "    n_nonfinite = jnp.asarray(0, dtype=jnp.int32)",
    "    n_elements = jnp.asarray(0, dtype=jnp.int32)",
    "    for leaf in leaves:",
    "        arr = jnp.asarray(leaf)",
    "        n_elements = n_elements + jnp.asarray(arr.size, dtype=jnp.int32)",
    "        if arr.size == 0:",
    "            continue",
    "        finite = jnp.isfinite(arr)",
    "        abs_finite = jnp.where(finite, jnp.abs(arr), 0.0)",
    "        sq_sum = sq_sum + jnp.sum(jnp.square(abs_finite))",
    "        max_abs = jnp.maximum(max_abs, jnp.max(abs_finite))",
    "        n_nonfinite = n_nonfinite + jnp.sum(jnp.logical_not(finite)).astype(jnp.int32)",
    "    grad_l2 = jnp.sqrt(sq_sum)",
    "    grad_rms = jnp.where(",
    "        n_elements > 0,",
    "        jnp.sqrt(sq_sum / n_elements.astype(grad_l2.dtype)),",
    "        jnp.asarray(float('nan'), dtype=grad_l2.dtype),",
    "    )",
    "    return {",
    "        'grad_l2': grad_l2,",
    "        'grad_rms': grad_rms,",
    "        'grad_max_abs': max_abs,",
    "        'grad_n_nonfinite': n_nonfinite,",
    "        'grad_n_elements': n_elements,",
    "    }",
    "",
    "def strategize_svi_gradient_diagnostics_jit(svi, svi_state, batch_args):",
    "    global _strategize_svi_gradient_jit_compile_count",
    "    if not hasattr(_svi_mod, '_make_loss_fn'):",
    "        raise RuntimeError('numpyro.infer.svi._make_loss_fn is unavailable')",
    "    names = _strategize_svi_update_arg_names(batch_args)",
    "    args = {name: batch_args[name] for name in names}",
    "    key = _strategize_svi_gradient_cache_key(svi, names)",
    "    fn = _strategize_svi_gradient_jit_cache.get(key)",
    "    if fn is None:",
    "        _strategize_svi_gradient_jit_compile_count += 1",
    "        def _compiled(state, args_in):",
    "            _, rng_key_eval = jax.random.split(state.rng_key)",
    "            params = svi.optim.get_params(state.optim_state)",
    "            loss_fn = _svi_mod._make_loss_fn(",
    "                svi.loss,",
    "                rng_key_eval,",
    "                svi.constrain_fn,",
    "                svi.model,",
    "                svi.guide,",
    "                tuple(),",
    "                args_in,",
    "                svi.static_kwargs,",
    "                mutable_state=state.mutable_state,",
    "            )",
    "            (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)",
    "            return _strategize_svi_gradient_stats(grads)",
    "        fn = jax.jit(_compiled)",
    "        _strategize_svi_gradient_jit_cache[key] = fn",
    "    stats = jax.device_get(fn(svi_state, args))",
    "    return {",
    "        'grad_l2': float(stats['grad_l2']),",
    "        'grad_rms': float(stats['grad_rms']),",
    "        'grad_max_abs': float(stats['grad_max_abs']),",
    "        'grad_n_nonfinite': int(stats['grad_n_nonfinite']),",
    "        'grad_n_elements': int(stats['grad_n_elements']),",
    "    }",
    "",
    "def strategize_svi_gradient_jit_cache_info():",
    "    return {",
    "        'size': len(_strategize_svi_gradient_jit_cache),",
    "        'compile_count': _strategize_svi_gradient_jit_compile_count,",
    "    }",
    "",
    "def strategize_svi_gradient_jit_cache_clear():",
    "    # Frees the cached jitted closures (which retain the svi object and",
    "    # its dataset arrays); compile_count is lifetime telemetry and is",
    "    # deliberately NOT reset.",
    "    _strategize_svi_gradient_jit_cache.clear()",
    "",
    "def strategize_svi_update_jit_cache_info():",
    "    return {",
    "        'size': len(_strategize_svi_update_jit_cache),",
    "        'compile_count': _strategize_svi_update_jit_compile_count,",
    "    }",
    "",
    "def strategize_svi_update_jit_cache_clear():",
    "    # Frees the cached jitted closures; compile_count is lifetime",
    "    # telemetry and is deliberately NOT reset.",
    "    _strategize_svi_update_jit_cache.clear()",
    "",
    "def strategize_offset_schedule(schedule, offset):",
    "    # Continue an optax schedule from `offset` steps in: used on SVI",
    "    # checkpoint resume so the LR does not re-enter warmup at peak value.",
    "    offset = int(offset)",
    "    def _shifted(step):",
    "        return schedule(step + offset)",
    "    return _shifted",
    sep = "\n"
  )
  reticulate::py_run_string(helper_code)
  strenv$jax_svi_update <- reticulate::py$strategize_svi_update_jit
  strenv$jax_svi_update_scan <- reticulate::py$strategize_svi_update_scan_jit
  strenv$jax_svi_update_stable_available <- reticulate::py$strategize_svi_update_stable_available
  strenv$jax_svi_update_jit_cache_info <- reticulate::py$strategize_svi_update_jit_cache_info
  strenv$jax_svi_update_jit_cache_clear <- reticulate::py$strategize_svi_update_jit_cache_clear
  strenv$jax_svi_gradient_diagnostics <- reticulate::py$strategize_svi_gradient_diagnostics_jit
  strenv$jax_svi_gradient_jit_cache_info <- reticulate::py$strategize_svi_gradient_jit_cache_info
  strenv$jax_svi_gradient_jit_cache_clear <- reticulate::py$strategize_svi_gradient_jit_cache_clear
  strenv$jax_offset_schedule <- reticulate::py$strategize_offset_schedule
  invisible(TRUE)
}

initialize_jax <- function(conda_env = "strategize_env",
                           conda_env_required = TRUE) {
  # reticulate is declared in Imports - use :: syntax
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in strenv
  strenv$jax <- reticulate::import("jax")
  strategize_configure_jax_compilation_cache(strenv$jax)
  strenv$jnp <- reticulate::import("jax.numpy")
  strenv$np  <- reticulate::import("numpy")
  strenv$eq  <- reticulate::import("equinox")
  strenv$py_gc  <- reticulate::import("gc")
  strenv$numpyro  <- reticulate::import("numpyro")
  strenv$optax  <- reticulate::import("optax")
  strenv$orbax_checkpoint <- tryCatch(
    reticulate::import("orbax.checkpoint"),
    error = function(e) NULL
  )
  strategize_register_jax_transformer_helpers()
  strategize_register_jax_svi_helpers()
  
  # setup numerical precisions
  strenv$jaxFloatType <- strenv$jnp$float32
  #strenv$dtj <- strenv$jnp$float64; strenv$jax$config$update("jax_enable_x64", TRUE) # use float64
  strenv$dtj <- strenv$jnp$float32; strenv$jax$config$update("jax_enable_x64", FALSE) # use float32
}
strenv <- new.env( parent = emptyenv() )
