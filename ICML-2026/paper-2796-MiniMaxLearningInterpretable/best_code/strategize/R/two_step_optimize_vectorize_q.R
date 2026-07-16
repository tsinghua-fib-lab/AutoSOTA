build_qpop_pair <- function(compile_fxn, getQStar_diff_MultiGroup) {
  compile_fxn(function(t, u,
                       INTERCEPT_ast_, COEFFICIENTS_ast_,
                       INTERCEPT_dag_, COEFFICIENTS_dag_){
    strenv$jnp$take(
      getQStar_diff_MultiGroup(
        t, u,
        INTERCEPT_ast_,  COEFFICIENTS_ast_,
        INTERCEPT_dag_,  COEFFICIENTS_dag_
      ),
      0L
    )
  })
}

build_primary_model_context <- function(model_env, outcome_model_type, GroupsPool = NULL) {
  use_neural <- outcome_model_type == "neural" &&
    exists("neural_model_info_ast_jnp", envir = model_env, inherits = TRUE)
  if (!use_neural) {
    return(list(use_neural = FALSE))
  }
  model_ast <- get("neural_model_info_ast_jnp", envir = model_env, inherits = TRUE)
  model_dag <- if (exists("neural_model_info_dag_jnp", envir = model_env, inherits = TRUE)) {
    get("neural_model_info_dag_jnp", envir = model_env, inherits = TRUE)
  } else {
    model_ast
  }
  model_ast_primary <- if (exists("neural_model_info_ast0_jnp", envir = model_env, inherits = TRUE)) {
    get("neural_model_info_ast0_jnp", envir = model_env, inherits = TRUE)
  } else {
    model_ast
  }
  model_dag_primary <- if (exists("neural_model_info_dag0_jnp", envir = model_env, inherits = TRUE)) {
    get("neural_model_info_dag0_jnp", envir = model_env, inherits = TRUE)
  } else {
    model_dag
  }
  party_label_ast <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 0) {
    GroupsPool[1]
  } else {
    NULL
  }
  party_label_dag <- if (exists("GroupsPool", inherits = TRUE) && length(GroupsPool) > 1) {
    GroupsPool[2]
  } else {
    party_label_ast
  }
  party_idx_ast <- neural_get_party_index(model_ast_primary, party_label_ast)
  party_idx_dag <- neural_get_party_index(model_dag_primary, party_label_dag)
  resp_idx_ast <- neural_get_resp_party_index(model_ast_primary, party_label_ast)
  resp_idx_dag <- neural_get_resp_party_index(model_dag_primary, party_label_dag)
  list(
    use_neural = TRUE,
    model_ast = model_ast,
    model_dag = model_dag,
    model_ast_primary = model_ast_primary,
    model_dag_primary = model_dag_primary,
    party_label_ast = party_label_ast,
    party_label_dag = party_label_dag,
    party_idx_ast = party_idx_ast,
    party_idx_dag = party_idx_dag,
    resp_idx_ast = resp_idx_ast,
    resp_idx_dag = resp_idx_dag
  )
}

build_kappa_helpers <- function(compile_fxn, temp_pushf, ctx,
                                getQStar_diff_SingleGroup,
                                allow_utilities = FALSE) {
  if (!isTRUE(ctx$use_neural)) {
    kappa_pair <- compile_fxn(function(v, v_prime,
                                       INTERCEPT_0_, COEFFICIENTS_0_){
      v <- strenv$jnp$reshape(v, list(-1L, 1L))
      v_prime <- strenv$jnp$reshape(v_prime, list(-1L, 1L))
      strenv$jnp$take(
        getQStar_diff_SingleGroup(
          v,
          v_prime,
          INTERCEPT_0_ * temp_pushf, temp_pushf * COEFFICIENTS_0_,
          INTERCEPT_0_ * temp_pushf, temp_pushf * COEFFICIENTS_0_
        ), 0L)
    })
    return(list(use_neural = FALSE, use_utilities = FALSE, kappa_pair = kappa_pair))
  }

  model_ast_primary <- ctx$model_ast_primary
  model_dag_primary <- ctx$model_dag_primary
  party_idx_ast <- ctx$party_idx_ast
  party_idx_dag <- ctx$party_idx_dag
  resp_idx_ast <- ctx$resp_idx_ast
  resp_idx_dag <- ctx$resp_idx_dag
  stage_idx_ast <- neural_stage_index(party_idx_ast, party_idx_ast, model_ast_primary)
  stage_idx_dag <- neural_stage_index(party_idx_dag, party_idx_dag, model_dag_primary)

  use_utilities <- FALSE
  utility_ast <- utility_dag <- NULL
  kappa_pair_ast <- kappa_pair_dag <- NULL

  if (isTRUE(allow_utilities)) {
    mode_ast <- neural_cross_encoder_mode(model_ast_primary)
    mode_dag <- neural_cross_encoder_mode(model_dag_primary)
    use_utilities <- isTRUE(model_ast_primary$pairwise_mode) &&
      isTRUE(model_dag_primary$pairwise_mode) &&
      !is.null(model_ast_primary$params$W_out) &&
      !is.null(model_dag_primary$params$W_out) &&
      !identical(mode_ast, "full") &&
      !identical(mode_dag, "full")
  }

  if (isTRUE(allow_utilities) && use_utilities) {
    utility_from_logits_ast <- switch(model_ast_primary$likelihood,
                                      categorical = function(logits) {
                                        strenv$jnp$squeeze(strenv$jnp$take(logits, 1L, axis = 1L))
                                      },
                                      function(logits) {
                                        strenv$jnp$squeeze(logits, axis = 1L)
                                      })
    utility_from_logits_dag <- switch(model_dag_primary$likelihood,
                                      categorical = function(logits) {
                                        strenv$jnp$squeeze(strenv$jnp$take(logits, 1L, axis = 1L))
                                      },
                                      function(logits) {
                                        strenv$jnp$squeeze(logits, axis = 1L)
                                      })
    utility_ast <- compile_fxn(function(v, theta_ast0){
      params_ast0 <- neural_params_from_theta(theta_ast0, model_ast_primary)
      matchup_idx <- neural_matchup_index(party_idx_ast, party_idx_ast, model_ast_primary)
      logits <- neural_candidate_utility_soft(
        v,
        party_idx_ast,
        resp_idx_ast,
        stage_idx_ast,
        model_ast_primary,
        params = params_ast0,
        matchup_idx = matchup_idx
      )
      utility_from_logits_ast(logits)
    })
    utility_dag <- compile_fxn(function(v, theta_dag0){
      params_dag0 <- neural_params_from_theta(theta_dag0, model_dag_primary)
      matchup_idx <- neural_matchup_index(party_idx_dag, party_idx_dag, model_dag_primary)
      logits <- neural_candidate_utility_soft(
        v,
        party_idx_dag,
        resp_idx_dag,
        stage_idx_dag,
        model_dag_primary,
        params = params_dag0,
        matchup_idx = matchup_idx
      )
      utility_from_logits_dag(logits)
    })
  } else {
    kappa_pair_ast <- compile_fxn(function(v, v_prime, theta_ast0){
      params_ast0 <- neural_params_from_theta(theta_ast0, model_ast_primary)
      logits <- neural_predict_pair_soft(
        v, v_prime,
        party_idx_ast, party_idx_ast,
        resp_idx_ast, model_ast_primary,
        params = params_ast0,
        return_logits = TRUE
      )
      scaled <- logits * temp_pushf
      prob <- neural_logits_to_q(scaled, model_ast_primary$likelihood)
      strenv$jnp$squeeze(prob)
    })
    kappa_pair_dag <- compile_fxn(function(v, v_prime, theta_dag0){
      params_dag0 <- neural_params_from_theta(theta_dag0, model_dag_primary)
      logits <- neural_predict_pair_soft(
        v, v_prime,
        party_idx_dag, party_idx_dag,
        resp_idx_dag, model_dag_primary,
        params = params_dag0,
        return_logits = TRUE
      )
      scaled <- logits * temp_pushf
      prob <- neural_logits_to_q(scaled, model_dag_primary$likelihood)
      strenv$jnp$squeeze(prob)
    })
  }

  list(
    use_neural = TRUE,
    use_utilities = use_utilities,
    kappa_pair_ast = kappa_pair_ast,
    kappa_pair_dag = kappa_pair_dag,
    utility_ast = utility_ast,
    utility_dag = utility_dag
  )
}

InitializeQMonteFxns <- function(){ 
  # The linearized version exploits:                                                                                                                          
  # 1. Independence of A and B primaries                                                                                                                      
  # 2. Linearity of expectation                                                                                                                               
  # 3. Precomputation of all C combinations 
  
  Qpop_pair <- build_qpop_pair(compile_fxn, getQStar_diff_MultiGroup)
  
  # Helper: primary head-to-head win prob κ_A(t, t') within A's primary
  TEMP_PUSHF <- strenv$primary_strength
  groups_pool <- if (exists("GroupsPool", inherits = TRUE)) GroupsPool else NULL
  ctx <- build_primary_model_context(environment(), outcome_model_type, groups_pool)
  kappa_helpers <- build_kappa_helpers(
    compile_fxn, TEMP_PUSHF, ctx, getQStar_diff_SingleGroup,
    allow_utilities = FALSE
  )
  use_neural <- isTRUE(kappa_helpers$use_neural)
  kappa_pair_ast <- kappa_helpers$kappa_pair_ast
  kappa_pair_dag <- kappa_helpers$kappa_pair_dag
  kappa_pair <- kappa_helpers$kappa_pair
  
  # === Core Monte-Carlo evaluator that respects the push-forward ===
  strenv$QMonteIter_MaxMin <- compile_fxn(function(
    TSAMP_ast, TSAMP_dag,                        # [nA, ·], [nB, ·] entrants
    TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,# [nA',·],[nB',·]  field
    a_i_ast, a_i_dag,                            # (unused here; kept for signature)
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_, P_VEC_FULL_dag_,           # (unused here)
    LAMBDA_, Q_SIGN, SEED_IN_LOOP               # (unused here)
  ){
    
    # ---- Primary κ matrices ----
    # κ_A[i,j] = Pr_A_primary( t_i beats t'_j )
    if (use_neural) {
      kA <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(tp_j){
          kappa_pair_ast(t_i, tp_j, COEFFICIENTS_ast0_)
        }, in_axes = 0L)( strenv$jax$lax$stop_gradient( TSAMP_ast_PrimaryComp ) )
      }, in_axes = 0L)(TSAMP_ast)                             # [nA, nA']
    } else {
      kA <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(tp_j){
          kappa_pair(t_i, tp_j, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
        }, in_axes = 0L)( strenv$jax$lax$stop_gradient( TSAMP_ast_PrimaryComp ) )
      }, in_axes = 0L)(TSAMP_ast)                             # [nA, nA']
    }
    
    # κ_B[s,r] = Pr_B_primary( u_s beats u'_r )
    if (use_neural) {
      kB <- strenv$jax$vmap(function(u_s){
        strenv$jax$vmap(function(up_r){
          kappa_pair_dag(u_s, up_r, COEFFICIENTS_dag0_)
        }, in_axes = 0L)( strenv$jax$lax$stop_gradient(TSAMP_dag_PrimaryComp) )
      }, in_axes = 0L)(TSAMP_dag)                             # [nB, nB']
    } else {
      kB <- strenv$jax$vmap(function(u_s){
        strenv$jax$vmap(function(up_r){
          kappa_pair(u_s, up_r, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
        }, in_axes = 0L)( strenv$jax$lax$stop_gradient(TSAMP_dag_PrimaryComp) )
      }, in_axes = 0L)(TSAMP_dag)                             # [nB, nB']
    }
    
    #sanity checks 
    # hist(strenv$np$array(TSAMP_ast))
    # hist(strenv$np$array(TSAMP_ast_PrimaryComp))
    # causalimages::image2(strenv$np$array(kA))
    # View(strenv$np$array(kA))
    
    # Averages for push-forward weights
    kA_mean_over_field   <- kA$mean(axis=1L)  # [nA]   E_field[κ | entrant]
    kB_mean_over_field   <- kB$mean(axis=1L)  # [nB]
    kA_mean_over_entrant <- kA$mean(axis=0L)  # [nA']  E_entrant[κ | field]
    kB_mean_over_entrant <- kB$mean(axis=0L)  # [nB']
    
    # ---- General-election blocks for the four primary outcomes (2×2) ----
    {
    # C(t_i, u_s)
    C_tu <- strenv$jax$vmap(function(t_i){
      strenv$jax$vmap(function(u_s){
        Qpop_pair(t_i, u_s,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag)
    }, in_axes = 0L)(TSAMP_ast)                           # [nA, nB]
    
    # C(t_i, u'_r)
    C_tu_field <- strenv$jax$vmap(function(t_i){
      strenv$jax$vmap(function(up_r){
        Qpop_pair(t_i, up_r,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
    }, in_axes = 0L)(TSAMP_ast)                    # [nA, nB']
    
    # C(t'_j, u_s)
    C_field_u <- strenv$jax$vmap(function(tp_j){
      strenv$jax$vmap(function(u_s){
        Qpop_pair(tp_j, u_s,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag)
    }, in_axes = 0L)(TSAMP_ast_PrimaryComp)         # [nA', nB]
    
    # C(t'_j, u'_r)
    C_field_field <- strenv$jax$vmap(function(tp_j){
      strenv$jax$vmap(function(up_r){
        Qpop_pair(tp_j, up_r,
                  INTERCEPT_ast_, COEFFICIENTS_ast_,
                  INTERCEPT_dag_, COEFFICIENTS_dag_)
      }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
    }, in_axes = 0L)(TSAMP_ast_PrimaryComp)   # [nA', nB']
    }
    
    # ---- Push-forward mixture weights (profile-specific) ----
    # Eq. (PrimaryPushforward) implies weights depend on each profile's primary win rate.
    one <- strenv$OneTf_flat
    {
      nA <- strenv$jnp$array(kA$shape[[1L]])
      nA_field <- strenv$jnp$array(kA$shape[[2L]])
      nB <- strenv$jnp$array(kB$shape[[1L]])
      nB_field <- strenv$jnp$array(kB$shape[[2L]])

      # Per-sample nominee weights (entrant vs field)
      wA_e <- kA_mean_over_field / nA
      wA_f <- (one - kA_mean_over_entrant) / nA_field
      wB_e <- kB_mean_over_field / nB
      wB_f <- (one - kB_mean_over_entrant) / nB_field

      # Broadcast weights over the four primary outcome blocks
      wA_e_col <- strenv$jnp$expand_dims(wA_e, 1L)
      wA_f_col <- strenv$jnp$expand_dims(wA_f, 1L)
      wB_e_row <- strenv$jnp$expand_dims(wB_e, 0L)
      wB_f_row <- strenv$jnp$expand_dims(wB_f, 0L)

      E1 <- strenv$jnp$sum(C_tu * wA_e_col * wB_e_row)            # A entrant vs B entrant
      E2 <- strenv$jnp$sum(C_tu_field * wA_e_col * wB_f_row)      # A entrant vs B field
      E3 <- strenv$jnp$sum(C_field_u * wA_f_col * wB_e_row)       # A field   vs B entrant
      E4 <- strenv$jnp$sum(C_field_field * wA_f_col * wB_f_row)   # A field   vs B field
    }

    # Exploit linearity of expectation and the mixture structure:
    # the pushforward decomposes into four weighted blocks (E1--E4)
    # corresponding to the primary outcomes (entrant vs. entrant, entrant vs. field, etc.)
    # Weights w1+w2+w3+w4 = 1 by construction, ensuring proper normalization
    q_ast <- E1 + E2 + E3 + E4
    q_dag <- one - q_ast  # zero-sum

    # Wrap scalars in expand_dims for consistency with MC mode's array return
    # This ensures q_ast.mean(0) works correctly in getQPiStar_gd
    return(list("q_ast" = strenv$jnp$expand_dims(q_ast, 0L),
                "q_dag" = strenv$jnp$expand_dims(q_dag, 0L)))
  })

  # Optional pushforward self-test (runs once when enabled)
  if (isTRUE(strenv$UNIT_TEST_PUSHFORWARD) && !isTRUE(strenv$UNIT_TEST_PUSHFORWARD_DONE)) {
    strenv$UNIT_TEST_PUSHFORWARD_DONE <- TRUE

    dim_pi <- if (exists("p_vec_tf", inherits = TRUE)) {
      as.integer(strenv$jnp$shape(p_vec_tf)[[1]])
    } else {
      as.integer(main_indices_i0$size)
    }

    if (dim_pi >= 1L) {
      nA_samp <- 2L
      nB_samp <- 2L
      nA_field_samp <- 2L
      nB_field_samp <- 2L

      make_mat <- function(offset, n_rows) {
        vals <- seq(0.05 + offset, 0.85 + offset, length.out = n_rows * dim_pi)
        strenv$jnp$array(matrix(vals, nrow = n_rows, byrow = TRUE), dtype = strenv$dtj)
      }

      TSAMP_ast <- make_mat(0.00, nA_samp)
      TSAMP_dag <- make_mat(0.02, nB_samp)
      TSAMP_ast_PrimaryComp <- make_mat(0.04, nA_field_samp)
      TSAMP_dag_PrimaryComp <- make_mat(0.06, nB_field_samp)

      coef_len <- if (exists("EST_COEFFICIENTS_tf_ast_jnp", inherits = TRUE)) {
        as.integer(EST_COEFFICIENTS_tf_ast_jnp$size)
      } else {
        as.integer(strenv$np$array(strenv$jnp$max(main_indices_i0))) + 1L
      }
      coef_len <- max(1L, coef_len)

      COEF <- strenv$jnp$reshape(strenv$jnp$linspace(-0.2, 0.2, coef_len),
                                 list(coef_len, 1L))
      INT <- strenv$jnp$array(0.1, dtype = strenv$dtj)

      res <- strenv$QMonteIter_MaxMin(
        TSAMP_ast, TSAMP_dag, TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,
        INT, INT,
        INT, COEF, INT, COEF,
        INT, COEF, INT, COEF,
        INT, INT, INT, 1.0, INT
      )

      # Rebuild kA/kB and the four C blocks
      kA <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(tp_j){
          kappa_pair(t_i, tp_j, INT, COEF)
        }, in_axes = 0L)(TSAMP_ast_PrimaryComp)
      }, in_axes = 0L)(TSAMP_ast)

      kB <- strenv$jax$vmap(function(u_s){
        strenv$jax$vmap(function(up_r){
          kappa_pair(u_s, up_r, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
      }, in_axes = 0L)(TSAMP_dag)

      C_tu <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(u_s){
          Qpop_pair(t_i, u_s, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag)
      }, in_axes = 0L)(TSAMP_ast)

      C_tu_field <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(up_r){
          Qpop_pair(t_i, up_r, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
      }, in_axes = 0L)(TSAMP_ast)

      C_field_u <- strenv$jax$vmap(function(tp_j){
        strenv$jax$vmap(function(u_s){
          Qpop_pair(tp_j, u_s, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag)
      }, in_axes = 0L)(TSAMP_ast_PrimaryComp)

      C_field_field <- strenv$jax$vmap(function(tp_j){
        strenv$jax$vmap(function(up_r){
          Qpop_pair(tp_j, up_r, INT, COEF, INT, COEF)
        }, in_axes = 0L)(TSAMP_dag_PrimaryComp)
      }, in_axes = 0L)(TSAMP_ast_PrimaryComp)

      kA_mean_over_field <- kA$mean(axis=1L)
      kB_mean_over_field <- kB$mean(axis=1L)
      kA_mean_over_entrant <- kA$mean(axis=0L)
      kB_mean_over_entrant <- kB$mean(axis=0L)

      nA <- strenv$jnp$array(kA$shape[[1L]])
      nA_field <- strenv$jnp$array(kA$shape[[2L]])
      nB <- strenv$jnp$array(kB$shape[[1L]])
      nB_field <- strenv$jnp$array(kB$shape[[2L]])

      wA_e <- kA_mean_over_field / nA
      wA_f <- (strenv$OneTf_flat - kA_mean_over_entrant) / nA_field
      wB_e <- kB_mean_over_field / nB
      wB_f <- (strenv$OneTf_flat - kB_mean_over_entrant) / nB_field

      # Explicit 4D expectation over primaries (no aggregation)
      inv_norm <- 1.0 / (nA * nB * nA_field * nB_field)
      kA_4 <- strenv$jnp$reshape(kA, list(nA, 1L, nA_field, 1L))
      kB_4 <- strenv$jnp$reshape(kB, list(1L, nB, 1L, nB_field))
      C_tu_4 <- strenv$jnp$reshape(C_tu, list(nA, nB, 1L, 1L))
      C_tu_field_4 <- strenv$jnp$reshape(C_tu_field, list(nA, 1L, 1L, nB_field))
      C_field_u_4 <- strenv$jnp$reshape(C_field_u, list(1L, nB, nA_field, 1L))
      C_field_field_4 <- strenv$jnp$reshape(C_field_field, list(1L, 1L, nA_field, nB_field))

      q_direct <- inv_norm * strenv$jnp$sum(
        kA_4 * kB_4 * C_tu_4 +
        kA_4 * (strenv$OneTf_flat - kB_4) * C_tu_field_4 +
        (strenv$OneTf_flat - kA_4) * kB_4 * C_field_u_4 +
        (strenv$OneTf_flat - kA_4) * (strenv$OneTf_flat - kB_4) * C_field_field_4
      )

      tol <- 1e-5
      if (!strenv$np$allclose(strenv$np$array(res$q_ast), strenv$np$array(q_direct),
                              rtol = tol, atol = tol)) {
        stop("Pushforward test failed: explicit expectation mismatch")
      }
      if (!strenv$np$allclose(strenv$np$array(strenv$jnp$sum(wA_e) + strenv$jnp$sum(wA_f)),
                              strenv$np$array(strenv$OneTf_flat), rtol = tol, atol = tol) ||
          !strenv$np$allclose(strenv$np$array(strenv$jnp$sum(wB_e) + strenv$jnp$sum(wB_f)),
                              strenv$np$array(strenv$OneTf_flat), rtol = tol, atol = tol)) {
        stop("Pushforward test failed: weight normalization")
      }
      if (!strenv$np$allclose(strenv$np$array(res$q_ast + res$q_dag),
                              strenv$np$array(strenv$OneTf_flat), rtol = tol, atol = tol)) {
        stop("Pushforward test failed: zero-sum")
      }
    }
  }
  
  # Batch wrapper 
  strenv$Vectorized_QMonteIter_MaxMin <- strenv$QMonteIter_MaxMin
  
  # ---- Non-adversarial MC evaluator ----
  strenv$QMonteIter <- function(pi_star_ast_f, 
                                pi_star_dag_f,
                                INTERCEPT_ast_,
                                COEFFICIENTS_ast_,
                                INTERCEPT_dag_,
                                COEFFICIENTS_dag_){
    q_star_ <- QFXN(pi_star_ast_f, 
                    pi_star_dag_f, 
                    INTERCEPT_ast_,  
                    COEFFICIENTS_ast_, 
                    INTERCEPT_dag_,  
                    COEFFICIENTS_dag_)
    return(q_star_)
  }
  strenv$Vectorized_QMonteIter <- compile_fxn(
    strenv$jax$vmap(strenv$QMonteIter,
                    in_axes = list(0L,0L,NULL,NULL,NULL,NULL)))
}

InitializeQMonteFxns_MCSampling <- function(){
  #The default version uses:
  #  1. Monte Carlo integration over profile draws
  #  2. Exact inner expectations over primary outcomes
  
  RelaxedBernoulli <- compile_fxn(function(p, seed, temp){
    eps <- 1e-6
    p <- strenv$jnp$clip(p, eps, 1.0 - eps)
    logit_p <- strenv$jnp$log(p) - strenv$jnp$log1p(-p)
    g1 <- strenv$jax$random$gumbel(seed)
    seed <- strenv$jax$random$split(seed)[[1L]]
    g2 <- strenv$jax$random$gumbel(seed)
    seed <- strenv$jax$random$split(seed)[[1L]]
    z <- strenv$jax$nn$sigmoid((logit_p + g1 - g2) / temp)
    list(z, seed)
  })
  
  Qpop_pair <- build_qpop_pair(compile_fxn, getQStar_diff_MultiGroup)
  
  TEMP_PUSHF <- strenv$primary_strength
  groups_pool <- if (exists("GroupsPool", inherits = TRUE)) GroupsPool else NULL
  ctx <- build_primary_model_context(environment(), outcome_model_type, groups_pool)
  kappa_helpers <- build_kappa_helpers(
    compile_fxn, TEMP_PUSHF, ctx, getQStar_diff_SingleGroup,
    allow_utilities = FALSE
  )
  use_neural <- isTRUE(kappa_helpers$use_neural)
  kappa_pair_ast <- kappa_helpers$kappa_pair_ast
  kappa_pair_dag <- kappa_helpers$kappa_pair_dag
  kappa_pair <- kappa_helpers$kappa_pair
  
  strenv$QMonteIter_MaxMin <- function(
    TSAMP_ast, TSAMP_dag,
    TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,
    a_i_ast,
    a_i_dag,
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_,
    P_VEC_FULL_dag_,
    LAMBDA_,
    Q_SIGN,
    SEED_IN_LOOP){
      if (use_neural) {
        kA <- kappa_pair_ast(TSAMP_ast, TSAMP_ast_PrimaryComp, COEFFICIENTS_ast0_)
        kB <- kappa_pair_dag(TSAMP_dag, TSAMP_dag_PrimaryComp, COEFFICIENTS_dag0_)
      } else {
        kA <- kappa_pair(TSAMP_ast, TSAMP_ast_PrimaryComp, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
        kB <- kappa_pair(TSAMP_dag, TSAMP_dag_PrimaryComp, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
      }
      
      one <- strenv$OneTf_flat

      C_tt <- Qpop_pair(TSAMP_ast, TSAMP_dag,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)
      C_tf <- Qpop_pair(TSAMP_ast, TSAMP_dag_PrimaryComp,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)
      C_ft <- Qpop_pair(TSAMP_ast_PrimaryComp, TSAMP_dag,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)
      C_ff <- Qpop_pair(TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_dag_, COEFFICIENTS_dag_)

      q_ast <- kA * kB * C_tt +
               kA * (one - kB) * C_tf +
               (one - kA) * kB * C_ft +
               (one - kA) * (one - kB) * C_ff
      list("q_ast" = q_ast,
           "q_dag" = one - q_ast)
      }
  strenv$Vectorized_QMonteIter_MaxMin <- compile_fxn(strenv$jax$vmap(
    strenv$QMonteIter_MaxMin,
    in_axes = eval(parse(text = paste("list(0L,0L,0L,0L,", # vectorize over TSAMPs and SEED
                                      paste(rep("NULL,",times = 15-1), collapse=""), "0L",  ")",sep = "") ))))
  
  strenv$QMonteIter <- function(pi_star_ast_f, 
                                pi_star_dag_f,
                                INTERCEPT_ast_,
                                COEFFICIENTS_ast_,
                                INTERCEPT_dag_,
                                COEFFICIENTS_dag_){
    # note: when diff == F, dag is ignored 
    q_star_ <- QFXN(pi_star_ast_f, 
                    pi_star_dag_f, 
                    INTERCEPT_ast_,  
                    COEFFICIENTS_ast_, 
                    INTERCEPT_dag_,  
                    COEFFICIENTS_dag_)
    return( q_star_ )
  }
  strenv$Vectorized_QMonteIter <- compile_fxn(
    strenv$jax$vmap(strenv$QMonteIter, in_axes = list(0L,0L,NULL,NULL,NULL,NULL)))
}

InitializeQMonteFxns_MultiCandidate <- function(){
  # Multi-candidate primaries with nomination probabilities from pairwise win rates

  Qpop_pair <- build_qpop_pair(compile_fxn, getQStar_diff_MultiGroup)

  TEMP_PUSHF <- strenv$primary_strength
  groups_pool <- if (exists("GroupsPool", inherits = TRUE)) GroupsPool else NULL
  ctx <- build_primary_model_context(environment(), outcome_model_type, groups_pool)
  kappa_helpers <- build_kappa_helpers(
    compile_fxn, TEMP_PUSHF, ctx, getQStar_diff_SingleGroup,
    allow_utilities = TRUE
  )
  use_neural <- isTRUE(kappa_helpers$use_neural)
  use_utilities <- isTRUE(kappa_helpers$use_utilities)
  kappa_pair_ast <- kappa_helpers$kappa_pair_ast
  kappa_pair_dag <- kappa_helpers$kappa_pair_dag
  kappa_pair <- kappa_helpers$kappa_pair
  utility_ast <- kappa_helpers$utility_ast
  utility_dag <- kappa_helpers$utility_dag

  strenv$QMonteIter_MaxMin <- function(
    TSAMP_ast, TSAMP_dag,
    TSAMP_ast_PrimaryComp, TSAMP_dag_PrimaryComp,
    a_i_ast,
    a_i_dag,
    INTERCEPT_ast_, COEFFICIENTS_ast_,
    INTERCEPT_dag_, COEFFICIENTS_dag_,
    INTERCEPT_ast0_, COEFFICIENTS_ast0_,
    INTERCEPT_dag0_, COEFFICIENTS_dag0_,
    P_VEC_FULL_ast_,
    P_VEC_FULL_dag_,
    LAMBDA_,
    Q_SIGN,
    SEED_IN_LOOP){

      one <- strenv$OneTf_flat

      # Combine entrants and field candidates into primary pools
      cand_ast <- strenv$jnp$concatenate(list(TSAMP_ast, TSAMP_ast_PrimaryComp), 0L)
      cand_dag <- strenv$jnp$concatenate(list(TSAMP_dag, TSAMP_dag_PrimaryComp), 0L)

      if (use_neural && use_utilities) {
        utilityA <- strenv$jax$vmap(function(t_i){
          utility_ast(t_i, COEFFICIENTS_ast0_)
        }, in_axes = 0L)(cand_ast)

        utilityB <- strenv$jax$vmap(function(u_i){
          utility_dag(u_i, COEFFICIENTS_dag0_)
        }, in_axes = 0L)(cand_dag)

        utilityA <- utilityA * TEMP_PUSHF
        utilityB <- utilityB * TEMP_PUSHF
        pA <- strenv$jax$nn$softmax(utilityA)
        pB <- strenv$jax$nn$softmax(utilityB)
      } else {
        # Pairwise primary win probabilities within each party
        if (use_neural) {
          kA <- strenv$jax$vmap(function(t_i){
            strenv$jax$vmap(function(t_j){
              kappa_pair_ast(t_i, t_j, COEFFICIENTS_ast0_)
            }, in_axes = 0L)(cand_ast)
          }, in_axes = 0L)(cand_ast)

          kB <- strenv$jax$vmap(function(u_i){
            strenv$jax$vmap(function(u_j){
              kappa_pair_dag(u_i, u_j, COEFFICIENTS_dag0_)
            }, in_axes = 0L)(cand_dag)
          }, in_axes = 0L)(cand_dag)
        } else {
          kA <- strenv$jax$vmap(function(t_i){
            strenv$jax$vmap(function(t_j){
              kappa_pair(t_i, t_j, INTERCEPT_ast0_, COEFFICIENTS_ast0_)
            }, in_axes = 0L)(cand_ast)
          }, in_axes = 0L)(cand_ast)

          kB <- strenv$jax$vmap(function(u_i){
            strenv$jax$vmap(function(u_j){
              kappa_pair(u_i, u_j, INTERCEPT_dag0_, COEFFICIENTS_dag0_)
            }, in_axes = 0L)(cand_dag)
          }, in_axes = 0L)(cand_dag)
        }

        # Multinomial logit nomination probabilities based on Bradley-Terry utility
        # Each candidate's utility is their average log-odds of beating other candidates
        nA <- kA$shape[[1L]]
        nB <- kB$shape[[1L]]
        maskA <- strenv$jnp$ones(list(nA, nA), dtype = strenv$dtj) - strenv$jnp$eye(nA, dtype = strenv$dtj)
        maskB <- strenv$jnp$ones(list(nB, nB), dtype = strenv$dtj) - strenv$jnp$eye(nB, dtype = strenv$dtj)

        # Compute log-odds utility for each candidate pair (Bradley-Terry model)
        # kA[i,j] = P(candidate i beats candidate j among same-party voters)
        eps <- strenv$jnp$maximum(
          strenv$jnp$array(1e-8, strenv$dtj),
          strenv$jnp$array(strenv$jnp$finfo(strenv$dtj)$eps, strenv$dtj)
        )
        one_bt <- strenv$jnp$array(1.0, strenv$dtj)
        # Clamp to keep log-odds finite under extreme or non-probability outputs.
        kA_clip <- strenv$jnp$clip(kA, eps, one_bt - eps)
        kB_clip <- strenv$jnp$clip(kB, eps, one_bt - eps)
        logoddsA <- strenv$jnp$log(kA_clip) - strenv$jnp$log(one_bt - kA_clip)
        logoddsB <- strenv$jnp$log(kB_clip) - strenv$jnp$log(one_bt - kB_clip)

        # Mean utility for each candidate (average log-odds vs all candidates)
        denomA <- strenv$jnp$maximum(
          strenv$jnp$array(nA - 1L, strenv$dtj),
          strenv$jnp$array(1L, strenv$dtj)
        )
        denomB <- strenv$jnp$maximum(
          strenv$jnp$array(nB - 1L, strenv$dtj),
          strenv$jnp$array(1L, strenv$dtj)
        )
        utilityA <- (logoddsA * maskA)$sum(axis = 1L) / denomA
        utilityB <- (logoddsB * maskB)$sum(axis = 1L) / denomB

        # Softmax over utilities; primary_strength is already applied in kappa_pair.
        pA <- strenv$jax$nn$softmax(utilityA)
        pB <- strenv$jax$nn$softmax(utilityB)
      }

      # General-election outcomes across all nominee pairs
      C_ab <- strenv$jax$vmap(function(t_i){
        strenv$jax$vmap(function(u_j){
          Qpop_pair(t_i, u_j,
                    INTERCEPT_ast_, COEFFICIENTS_ast_,
                    INTERCEPT_dag_, COEFFICIENTS_dag_)
        }, in_axes = 0L)(cand_dag)
      }, in_axes = 0L)(cand_ast)

      pA_col <- strenv$jnp$expand_dims(pA, 1L)
      pB_row <- strenv$jnp$expand_dims(pB, 0L)
      q_ast <- strenv$jnp$sum(C_ab * pA_col * pB_row)

      list("q_ast" = q_ast,
           "q_dag" = one - q_ast)
      }
  strenv$Vectorized_QMonteIter_MaxMin <- compile_fxn(strenv$jax$vmap(
    strenv$QMonteIter_MaxMin,
    in_axes = eval(parse(text = paste("list(0L,0L,0L,0L,",
                                      paste(rep("NULL,",times = 15-1), collapse=""), "0L",  ")",sep = "") ))))

  # Non-adversarial MC evaluator (same as other implementations)
  strenv$QMonteIter <- function(pi_star_ast_f,
                                pi_star_dag_f,
                                INTERCEPT_ast_,
                                COEFFICIENTS_ast_,
                                INTERCEPT_dag_,
                                COEFFICIENTS_dag_){
    q_star_ <- QFXN(pi_star_ast_f,
                    pi_star_dag_f,
                    INTERCEPT_ast_,
                    COEFFICIENTS_ast_,
                    INTERCEPT_dag_,
                    COEFFICIENTS_dag_)
    return( q_star_ )
  }
  strenv$Vectorized_QMonteIter <- compile_fxn(
    strenv$jax$vmap(strenv$QMonteIter, in_axes = list(0L,0L,NULL,NULL,NULL,NULL)))
}
