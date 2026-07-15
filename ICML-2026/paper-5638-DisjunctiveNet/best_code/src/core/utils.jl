function _validate_projection_config(config::ProjectionConfig)
    valid_modes = (:dnf_qp, :milp)
    valid_gradients = (:diffopt, :straight_through)
    valid_fallbacks = (:identity, :error)
    valid_formulations = (:dnf, :cnf, :partial_dnf)
    
    config.mode in valid_modes ||
        throw(ArgumentError("Invalid projection mode $(config.mode). Expected one of $(valid_modes)."))

    config.gradient in valid_gradients ||
        throw(ArgumentError("Invalid gradient mode $(config.gradient). Expected one of $(valid_gradients)."))

    config.fallback in valid_fallbacks ||
        throw(ArgumentError("Invalid fallback $(config.fallback). Expected one of $(valid_fallbacks)."))

    config.y_regularization >= 0 ||
        throw(ArgumentError("y_regularization must be nonnegative."))

    config.ycopy_regularization >= 0 ||
        throw(ArgumentError("ycopy_regularization must be nonnegative."))

    config.gamma_regularization >= 0 ||
        throw(ArgumentError("gamma_regularization must be nonnegative."))

    config.anchor_regularization >= 0 ||
        throw(ArgumentError("anchor_regularization must be nonnegative."))
    
    config.num_dnf_rules >= -1 ||
        throw(ArgumentError("num_dnf_rules must be -1 or nonnegative."))
    
    config.formulation in valid_formulations ||
        throw(ArgumentError("Invalid formulation $(config.formulation). Expected one of $(valid_formulations)."))

    if config.mode == :milp && config.gradient == :diffopt
        throw(ArgumentError("MILP mode does not support `gradient = :diffopt`; use `gradient = :straight_through`."))
    end

    if config.mode == :dnf_qp && config.gradient == :straight_through
        # This is allowed, but not the intended default.
        return nothing
    end

    return nothing
end

