build_reinforce_diag <- function(reward_mean,
                                 reward_var,
                                 baseline_prev,
                                 grad_total,
                                 baseline_next = NULL) {
  nonfinite <- strenv$jnp$logical_not(
    strenv$jnp$all(strenv$jnp$isfinite(grad_total))
  )

  diag <- list(
    baseline = strenv$jax$lax$stop_gradient(baseline_prev),
    reward_mean = strenv$jax$lax$stop_gradient(reward_mean),
    reward_var = strenv$jax$lax$stop_gradient(reward_var),
    nonfinite = strenv$jax$lax$stop_gradient(nonfinite)
  )
  if (!is.null(baseline_next)) {
    diag$baseline_next <- strenv$jax$lax$stop_gradient(baseline_next)
  }
  diag
}

getQPiStar_gd <-  function(REGRESSION_PARAMETERS_ast,
                          REGRESSION_PARAMETERS_dag,
                          REGRESSION_PARAMETERS_ast0,
                          REGRESSION_PARAMETERS_dag0,
                          P_VEC_FULL_ast,
                          P_VEC_FULL_dag,
                          SLATE_VEC_ast, 
                          SLATE_VEC_dag,
                          LAMBDA,
                          SEED,
                          functionList,
                          a_i_ast,  # initial value, ast 
                          a_i_dag,  # initial value, dag 
                          functionReturn = T,
                          gd_full_simplex, 
                          quiet = TRUE,
                          optimism = c("none", "ogda", "extragrad", "smp", "rain"),
                          optimism_coef = 1,
                          rain_lambda = 1,
                          rain_gamma = 0.01,
                          rain_L = NULL,
                          rain_eta = 0.001,
                          rain_variant = "alg10_staged",
                          rain_output = "last",
                          force_reinforce = FALSE
                          ){
  optimism <- match.arg(optimism)
  optimism_coef <- as.numeric(optimism_coef)
  if (optimism != "none" && use_optax) {
    stop("optimism/extra-gradient not supported with optax; set use_optax = FALSE.")
  }
  use_smp <- optimism == "smp"
  use_rain <- optimism == "rain"
  use_extragrad <- optimism %in% c("extragrad", "smp", "rain")
  use_joint_extragrad <- adversarial && use_extragrad && !use_optax
  # throw fxns into env 
  dQ_da_ast <- functionList[[1]]
  dQ_da_dag <- functionList[[2]]
  QFXN <- functionList[[3]]

  REGRESSION_PARAMETERS_ast <- gather_fxn(REGRESSION_PARAMETERS_ast)
  INTERCEPT_ast_ <- REGRESSION_PARAMETERS_ast[[1]]
  COEFFICIENTS_ast_ <- REGRESSION_PARAMETERS_ast[[2]]


  INTERCEPT_dag0_ <- INTERCEPT_ast0_ <- INTERCEPT_dag_ <- INTERCEPT_ast_
  COEFFICIENTS_dag0_ <- COEFFICIENTS_ast0_ <- COEFFICIENTS_dag_ <- COEFFICIENTS_ast_
  if( adversarial ){
    REGRESSION_PARAMETERS_dag <- gather_fxn(REGRESSION_PARAMETERS_dag)
    INTERCEPT_dag_ <- REGRESSION_PARAMETERS_dag[[1]]
    COEFFICIENTS_dag_ <- REGRESSION_PARAMETERS_dag[[2]]
  }
  if(adversarial && isTRUE(use_stage_models)){
    REGRESSION_PARAMETERS_ast0 <- gather_fxn(REGRESSION_PARAMETERS_ast0)
    INTERCEPT_ast0_ <- REGRESSION_PARAMETERS_ast0[[1]]
    COEFFICIENTS_ast0_ <- REGRESSION_PARAMETERS_ast0[[2]]

    REGRESSION_PARAMETERS_dag0 <- gather_fxn(REGRESSION_PARAMETERS_dag0)
    INTERCEPT_dag0_ <- REGRESSION_PARAMETERS_dag0[[1]]
    COEFFICIENTS_dag0_ <- REGRESSION_PARAMETERS_dag0[[2]]
  }

  objective_spec <- resolve_q_eval_spec(
    phase = "objective",
    adversarial = adversarial,
    outcome_model_type = outcome_model_type,
    glm_family = glm_family,
    nMonte_Qglm = nMonte_Qglm,
    nMonte_adversarial = nMonte_adversarial,
    ParameterizationType = strenv$ParameterizationType,
    d_locator_use = strenv$jnp$array(strenv$d_locator_use),
    single_party = !isTRUE(diff),
    force_reinforce = force_reinforce
  )
  objective_gradient_mode <- objective_spec$objective_gradient_mode
  strenv$objective_gradient_mode <- objective_gradient_mode
  strenv$reinforce_baseline_ast_vec <- vector("list", length = nSGD)
  strenv$reinforce_baseline_dag_vec <- vector("list", length = nSGD)
  strenv$reinforce_reward_mean_ast_vec <- vector("list", length = nSGD)
  strenv$reinforce_reward_mean_dag_vec <- vector("list", length = nSGD)
  strenv$reinforce_reward_var_ast_vec <- vector("list", length = nSGD)
  strenv$reinforce_reward_var_dag_vec <- vector("list", length = nSGD)
  strenv$reinforce_nonfinite_ast_steps <- 0L
  strenv$reinforce_nonfinite_dag_steps <- 0L
  reinforce_baseline_ast_state <- NULL
  reinforce_baseline_dag_state <- NULL
  assign("reinforce_baseline_ast", NULL, envir = strenv)
  assign("reinforce_baseline_dag", NULL, envir = strenv)
  on.exit({
    assign("reinforce_baseline_ast", NULL, envir = strenv)
    assign("reinforce_baseline_dag", NULL, envir = strenv)
  }, add = TRUE)

  grad_eval_ast <- dQ_da_ast
  grad_eval_dag <- dQ_da_dag
  call_grad_eval_ast <- function(..., BASELINE_PREV = NULL) {
    if (identical(objective_gradient_mode, "reinforce")) {
      return(grad_eval_ast(..., BASELINE_PREV = BASELINE_PREV))
    }
    grad_eval_ast(...)
  }
  call_grad_eval_dag <- function(..., BASELINE_PREV = NULL) {
    if (identical(objective_gradient_mode, "reinforce")) {
      return(grad_eval_dag(..., BASELINE_PREV = BASELINE_PREV))
    }
    grad_eval_dag(...)
  }
  normalize_reinforce_baseline <- function(baseline_prev) {
    if (is.null(baseline_prev)) {
      return(strenv$jnp$array(0., strenv$dtj))
    }
    baseline_prev
  }
  compute_reinforce_baseline_next <- function(baseline_prev, reward_mean) {
    strenv$jnp$add(
      strenv$jnp$multiply(strenv$jnp$array(0.9, strenv$dtj), baseline_prev),
      strenv$jnp$multiply(strenv$jnp$array(0.1, strenv$dtj), reward_mean)
    )
  }
  advance_reinforce_baseline <- function(current_state, grad_result) {
    if (length(grad_result) < 3L || is.null(grad_result[[3L]])) {
      return(current_state)
    }
    diag <- grad_result[[3L]]
    if (is.null(diag$baseline_next)) {
      return(current_state)
    }
    diag$baseline_next
  }

  # Large-support hard draws do not admit the pathwise gradient used by the
  # relaxed/objective code paths, so this branch switches the optimizer to a
  # score-function estimator. The exact penalty gradient is still added
  # directly, and the EMA baseline only reduces variance; it does not change
  # the target objective.
  if (identical(objective_gradient_mode, "reinforce")) {
    index_spec_use <- resolve_multinomial_group_index_spec(
      d_locator_use = strenv$d_locator_use,
      ParameterizationType = strenv$ParameterizationType
    )
    main_comp_mat_use <- strenv$main_comp_mat
    shadow_comp_mat_use <- strenv$shadow_comp_mat
    if (is.null(main_comp_mat_use)) {
      main_comp_mat_use <- strenv$OneTf_flat
    }
    if (is.null(shadow_comp_mat_use)) {
      shadow_comp_mat_use <- strenv$OneTf_flat
    }

    compute_penalty_value <- function(pi_star_full_i,
                                      P_VEC_FULL_,
                                      LAMBDA_) {
      if (penalty_type %in% c("L1", "L2")) {
        PenFxn <- if (penalty_type == "L1") strenv$jnp$abs else strenv$jnp$square
        return(LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn(pi_star_full_i - P_VEC_FULL_))))
      }
      if (penalty_type == "LInfinity") {
        pen_groups <- tapply(1:length(split_vec_full), split_vec_full, function(zer) {
          list(strenv$jnp$max(
            strenv$jnp$take(pi_star_full_i, indices = strenv$jnp$array(ai(zer - 1L)), axis = 0L)
          ))
        })
        names(pen_groups) <- NULL
        return(strenv$jnp$negative(LAMBDA_ * strenv$jnp$sum(strenv$jnp$stack(pen_groups))))
      }

      eps <- 1e-8
      LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(
        P_VEC_FULL_ * (
          strenv$jnp$log(strenv$jnp$clip(P_VEC_FULL_, eps, 1.0)) -
            strenv$jnp$log(strenv$jnp$clip(pi_star_full_i, eps, 1.0))
        )
      ))
    }

    penalty_value_and_grad_ast <- strenv$jax$value_and_grad(function(a_ast, P_VEC_FULL_, LAMBDA_) {
      pi_star_full_i_ast <- getPrettyPi(
        strenv$a2Simplex_diff_use(a_ast),
        strenv$ParameterizationType,
        strenv$d_locator_use,
        main_comp_mat_use,
        shadow_comp_mat_use
      )
      compute_penalty_value(pi_star_full_i_ast, P_VEC_FULL_, LAMBDA_)
    }, argnums = 0L)

    penalty_value_and_grad_dag <- strenv$jax$value_and_grad(function(a_dag, P_VEC_FULL_, LAMBDA_) {
      pi_star_full_i_dag <- getPrettyPi(
        strenv$a2Simplex_diff_use(a_dag),
        strenv$ParameterizationType,
        strenv$d_locator_use,
        main_comp_mat_use,
        shadow_comp_mat_use
      )
      compute_penalty_value(pi_star_full_i_dag, P_VEC_FULL_, LAMBDA_)
    }, argnums = 0L)

    score_value_and_grad_ast <- strenv$jax$value_and_grad(function(a_ast, policy_samples, advantages) {
      pi_star_i_ast <- strenv$a2Simplex_diff_use(a_ast)
      log_probs <- compute_policy_sample_log_probs(
        pi_vec = pi_star_i_ast,
        profiles = policy_samples,
        ParameterizationType = strenv$ParameterizationType,
        index_spec = index_spec_use
      )
      strenv$jnp$mean(strenv$jax$lax$stop_gradient(advantages) * log_probs)
    }, argnums = 0L)

    score_value_and_grad_dag <- strenv$jax$value_and_grad(function(a_dag, policy_samples, advantages) {
      pi_star_i_dag <- strenv$a2Simplex_diff_use(a_dag)
      log_probs <- compute_policy_sample_log_probs(
        pi_vec = pi_star_i_dag,
        profiles = policy_samples,
        ParameterizationType = strenv$ParameterizationType,
        index_spec = index_spec_use
      )
      strenv$jnp$mean(strenv$jax$lax$stop_gradient(advantages) * log_probs)
    }, argnums = 0L)

    update_reinforce_diag <- function(player,
                                      reward_mean,
                                      reward_var,
                                      baseline_prev,
                                      grad_total,
                                      baseline_next = NULL) {
      build_reinforce_diag(
        reward_mean = reward_mean,
        reward_var = reward_var,
        baseline_prev = baseline_prev,
        grad_total = grad_total,
        baseline_next = baseline_next
      )
    }

    grad_eval_ast <- function(a_i_ast, a_i_dag,
                              INTERCEPT_ast_, COEFFICIENTS_ast_,
                              INTERCEPT_dag_, COEFFICIENTS_dag_,
                              INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                              INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                              P_VEC_FULL_ast, P_VEC_FULL_dag,
                              SLATE_VEC_ast, SLATE_VEC_dag,
                              LAMBDA,
                              Q_SIGN_,
                              SEED,
                              BASELINE_PREV = NULL) {
      pi_star_i_ast <- strenv$a2Simplex_diff_use(a_i_ast)
      pi_star_i_dag <- strenv$a2Simplex_diff_use(a_i_dag)
      reinforce_eval <- if (!adversarial) {
        evaluate_average_case_q_reinforce(
          pi_star_ast = pi_star_i_ast,
          pi_star_dag = pi_star_i_dag,
          INTERCEPT_ast_ = INTERCEPT_ast_,
          COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
          INTERCEPT_dag_ = INTERCEPT_dag_,
          COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
          seed_in = SEED,
          outcome_model_type = outcome_model_type,
          glm_family = glm_family,
          nMonte_Qglm = nMonte_Qglm,
          temperature = MNtemp,
          ParameterizationType = strenv$ParameterizationType,
          d_locator_use = strenv$d_locator_use,
          q_fxn = QFXN,
          single_party = !isTRUE(diff),
          force_reinforce = force_reinforce,
          player = "ast"
        )
      } else {
        evaluate_adversarial_q_reinforce(
          pi_star_ast = pi_star_i_ast,
          pi_star_dag = pi_star_i_dag,
          a_i_ast = a_i_ast,
          a_i_dag = a_i_dag,
          INTERCEPT_ast_ = INTERCEPT_ast_,
          COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
          INTERCEPT_dag_ = INTERCEPT_dag_,
          COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
          INTERCEPT_ast0_ = INTERCEPT_ast0_,
          COEFFICIENTS_ast0_ = COEFFICIENTS_ast0_,
          INTERCEPT_dag0_ = INTERCEPT_dag0_,
          COEFFICIENTS_dag0_ = COEFFICIENTS_dag0_,
          P_VEC_FULL_ast_ = P_VEC_FULL_ast,
          P_VEC_FULL_dag_ = P_VEC_FULL_dag,
          SLATE_VEC_ast_ = SLATE_VEC_ast,
          SLATE_VEC_dag_ = SLATE_VEC_dag,
          LAMBDA_ = LAMBDA,
          seed_in = SEED,
          outcome_model_type = outcome_model_type,
          glm_family = glm_family,
          nMonte_Qglm = nMonte_Qglm,
          nMonte_adversarial = nMonte_adversarial,
          primary_pushforward = primary_pushforward,
          primary_n_entrants = primary_n_entrants,
          primary_n_field = primary_n_field,
          temperature = MNtemp,
          ParameterizationType = strenv$ParameterizationType,
          d_locator_use = strenv$d_locator_use,
          player = "ast"
        )
      }

      baseline_prev <- normalize_reinforce_baseline(BASELINE_PREV)
      advantages <- reinforce_eval$reward_draws - baseline_prev
      score_grad <- score_value_and_grad_ast(
        a_i_ast,
        reinforce_eval$policy_samples,
        advantages
      )
      penalty_grad <- penalty_value_and_grad_ast(a_i_ast, P_VEC_FULL_ast, LAMBDA)
      grad_total <- strenv$jnp$add(score_grad[[2]], penalty_grad[[2]])
      reward_mean <- reinforce_eval$reward_mean
      reward_var <- strenv$jnp$mean(strenv$jnp$square(reinforce_eval$reward_draws - reward_mean))
      baseline_next <- compute_reinforce_baseline_next(baseline_prev, reward_mean)
      diag <- update_reinforce_diag(
        player = "ast",
        reward_mean = reward_mean,
        reward_var = reward_var,
        baseline_prev = baseline_prev,
        grad_total = grad_total,
        baseline_next = baseline_next
      )

      list(
        strenv$jnp$add(reward_mean, penalty_grad[[1]]),
        grad_total,
        diag
      )
    }

    grad_eval_dag <- function(a_i_ast, a_i_dag,
                              INTERCEPT_ast_, COEFFICIENTS_ast_,
                              INTERCEPT_dag_, COEFFICIENTS_dag_,
                              INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                              INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                              P_VEC_FULL_ast, P_VEC_FULL_dag,
                              SLATE_VEC_ast, SLATE_VEC_dag,
                              LAMBDA,
                              Q_SIGN_,
                              SEED,
                              BASELINE_PREV = NULL) {
      pi_star_i_ast <- strenv$a2Simplex_diff_use(a_i_ast)
      pi_star_i_dag <- strenv$a2Simplex_diff_use(a_i_dag)
      reinforce_eval <- evaluate_adversarial_q_reinforce(
        pi_star_ast = pi_star_i_ast,
        pi_star_dag = pi_star_i_dag,
        a_i_ast = a_i_ast,
        a_i_dag = a_i_dag,
        INTERCEPT_ast_ = INTERCEPT_ast_,
        COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
        INTERCEPT_dag_ = INTERCEPT_dag_,
        COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
        INTERCEPT_ast0_ = INTERCEPT_ast0_,
        COEFFICIENTS_ast0_ = COEFFICIENTS_ast0_,
        INTERCEPT_dag0_ = INTERCEPT_dag0_,
        COEFFICIENTS_dag0_ = COEFFICIENTS_dag0_,
        P_VEC_FULL_ast_ = P_VEC_FULL_ast,
        P_VEC_FULL_dag_ = P_VEC_FULL_dag,
        SLATE_VEC_ast_ = SLATE_VEC_ast,
        SLATE_VEC_dag_ = SLATE_VEC_dag,
        LAMBDA_ = LAMBDA,
        seed_in = SEED,
        outcome_model_type = outcome_model_type,
        glm_family = glm_family,
        nMonte_Qglm = nMonte_Qglm,
        nMonte_adversarial = nMonte_adversarial,
        primary_pushforward = primary_pushforward,
        primary_n_entrants = primary_n_entrants,
        primary_n_field = primary_n_field,
        temperature = MNtemp,
        ParameterizationType = strenv$ParameterizationType,
        d_locator_use = strenv$d_locator_use,
        player = "dag"
      )

      baseline_prev <- normalize_reinforce_baseline(BASELINE_PREV)
      advantages <- reinforce_eval$reward_draws - baseline_prev
      score_grad <- score_value_and_grad_dag(
        a_i_dag,
        reinforce_eval$policy_samples,
        advantages
      )
      penalty_grad <- penalty_value_and_grad_dag(a_i_dag, P_VEC_FULL_dag, LAMBDA)
      grad_total <- strenv$jnp$add(score_grad[[2]], penalty_grad[[2]])
      reward_mean <- reinforce_eval$reward_mean
      reward_var <- strenv$jnp$mean(strenv$jnp$square(reinforce_eval$reward_draws - reward_mean))
      baseline_next <- compute_reinforce_baseline_next(baseline_prev, reward_mean)
      diag <- update_reinforce_diag(
        player = "dag",
        reward_mean = reward_mean,
        reward_var = reward_var,
        baseline_prev = baseline_prev,
        grad_total = grad_total,
        baseline_next = baseline_next
      )

      list(
        strenv$jnp$add(reward_mean, penalty_grad[[1]]),
        grad_total,
        diag
      )
    }
  }

  record_reinforce_diag <- function(iter_idx, player, grad_result) {
    if (length(grad_result) < 3L || is.null(grad_result[[3L]])) {
      return(invisible(NULL))
    }
    diag <- grad_result[[3L]]
    diag_nonfinite <- FALSE
    if (!is.null(diag$nonfinite)) {
      diag_nonfinite <- tryCatch(
        isTRUE(as.logical(strenv$np$array(diag$nonfinite))),
        error = function(e) isTRUE(diag$nonfinite)
      )
    }
    if (identical(player, "ast")) {
      strenv$reinforce_baseline_ast_vec[[iter_idx]] <- list(diag$baseline)
      strenv$reinforce_reward_mean_ast_vec[[iter_idx]] <- list(diag$reward_mean)
      strenv$reinforce_reward_var_ast_vec[[iter_idx]] <- list(diag$reward_var)
      if (diag_nonfinite) {
        strenv$reinforce_nonfinite_ast_steps <- strenv$reinforce_nonfinite_ast_steps + 1L
      }
    } else {
      strenv$reinforce_baseline_dag_vec[[iter_idx]] <- list(diag$baseline)
      strenv$reinforce_reward_mean_dag_vec[[iter_idx]] <- list(diag$reward_mean)
      strenv$reinforce_reward_var_dag_vec[[iter_idx]] <- list(diag$reward_var)
      if (diag_nonfinite) {
        strenv$reinforce_nonfinite_dag_steps <- strenv$reinforce_nonfinite_dag_steps + 1L
      }
    }
    invisible(NULL)
  }

  # gradient descent iterations
  strenv$grad_mag_ast_vec <- strenv$grad_mag_dag_vec <- rep(NA, times = nSGD)
  strenv$loss_ast_vec <- strenv$loss_dag_vec <- rep(NA, times = nSGD)
  strenv$inv_learning_rate_ast_vec <- strenv$inv_learning_rate_dag_vec <- rep(NA, times = nSGD)
  grad_prev_ast <- grad_prev_dag <- NULL
  if (use_joint_extragrad) {
    strenv$extragrad_eval_points <- vector("list", length = nSGD)
  }
  if (use_rain) {
    strenv$rain_lambda_vec <- vector("list", length = nSGD)
    strenv$rain_lambda_s_vec <- vector("list", length = nSGD)
    strenv$rain_lambda_sum_vec <- vector("list", length = nSGD)
    strenv$rain_stage_idx <- rep(NA_integer_, times = nSGD)
    strenv$rain_anchor_bar_norm_ast <- vector("list", length = nSGD)
    if (adversarial) {
      strenv$rain_anchor_bar_norm_dag <- vector("list", length = nSGD)
    } else {
      strenv$rain_anchor_bar_norm_dag <- NULL
    }
  }
  if (use_smp) {
    smp_sum_weighted_ast <- strenv$jnp$multiply(a_i_ast, strenv$jnp$array(0., strenv$dtj))
    smp_sum_gamma_ast <- strenv$jnp$array(0., strenv$dtj)
    smp_gamma_ast_vec <- vector("list", length = nSGD)
    if (adversarial) {
      smp_sum_weighted_dag <- strenv$jnp$multiply(a_i_dag, strenv$jnp$array(0., strenv$dtj))
      smp_sum_gamma_dag <- strenv$jnp$array(0., strenv$dtj)
      smp_gamma_dag_vec <- vector("list", length = nSGD)
    }
  }
  goOn <- F; i<-0
  INIT_MIN_GRAD_ACCUM <- strenv$jnp$array(0.1)
  if (use_rain) {
    # Stage-wise RAIN (Algorithm 10): recursive anchoring with stage-constant SEG.
    rain_variant <- match.arg(rain_variant, c("alg10_staged", "alg9_single_loop"))
    rain_output <- match.arg(rain_output, c("uniform_half", "last"))
    if (!identical(rain_variant, "alg10_staged")) {
      stop("RAIN variant 'alg9_single_loop' is not implemented yet; use 'alg10_staged'.")
    }

    if (is.null(rain_gamma)) {
      rain_gamma <- 0
    }
    rain_gamma <- as.numeric(rain_gamma)
    if (length(rain_gamma) != 1 || !is.finite(rain_gamma) || rain_gamma < 0) {
      rain_gamma <- 0
    }

    if (is.null(rain_lambda)) {
      rain_lambda <- 0
    }
    rain_lambda <- as.numeric(rain_lambda)
    if (length(rain_lambda) != 1 || !is.finite(rain_lambda) || rain_lambda < 0) {
      rain_lambda <- 0
    }

    rain_L_val <- NULL
    if (!is.null(rain_L)) {
      rain_L_val <- as.numeric(rain_L)
      if (length(rain_L_val) != 1 || !is.finite(rain_L_val) || rain_L_val <= 0) {
        rain_L_val <- NULL
      }
    }

    if (is.null(rain_eta)) {
      if (!is.null(rain_L_val)) {
        rain_eta <- 1 / (8 * rain_L_val)
      } else {
        rain_eta <- as.numeric(learning_rate_max)
      }
    } else {
      rain_eta <- as.numeric(rain_eta)
    }
    if (length(rain_eta) != 1 || !is.finite(rain_eta) || rain_eta <= 0) {
      rain_eta <- 1e-8
    }

    # In the theory, eta = 1/(8L) and stages run for T_s = 16L/lambda_s iterations.
    if (!is.null(rain_L_val)) {
      L_est <- rain_L_val
    } else {
      L_est <- 1 / (8 * rain_eta)
    }
    if (!is.finite(L_est) || L_est <= 0) {
      L_est <- 1 / (8 * 1e-8)
    }

    eta_t <- strenv$jnp$array(rain_eta, strenv$dtj)
    inv_lr_val <- strenv$jnp$reciprocal(eta_t)

    Lambda_sum_num <- 0
    Lambda_sum <- strenv$jnp$array(Lambda_sum_num, strenv$dtj)
    B_ast <- strenv$jnp$multiply(a_i_ast, strenv$jnp$array(0., strenv$dtj))
    if (adversarial) {
      B_dag <- strenv$jnp$multiply(a_i_dag, strenv$jnp$array(0., strenv$dtj))
    } else {
      B_dag <- NULL
    }

    rain_log_step <- function(step_idx) {
      if (step_idx < 5 || step_idx %in% unique(ceiling(c(0.25, 0.5, 0.75, 1) * nSGD))) {
        message(sprintf("SGD Iteration: %s of %s", step_idx, nSGD))
      }
    }

    stage_idx <- 0L
    stage_cap <- Inf
    if (!is.null(rain_L_val) && rain_gamma > 0 && rain_lambda > 0) {
      ratio_val <- rain_L_val / rain_lambda
      if (is.finite(ratio_val) && ratio_val > 0) {
        stage_cap <- ceiling(log(ratio_val) / log(1 + rain_gamma))
        if (!is.finite(stage_cap) || stage_cap < 1) {
          stage_cap <- 1
        }
      }
    }

    lambda_stage_num <- rain_gamma * rain_lambda
    if (!is.finite(lambda_stage_num) || lambda_stage_num < 0) {
      lambda_stage_num <- 0
    }

    use_uniform_half <- identical(rain_output, "uniform_half")

    while (i < nSGD && stage_idx < stage_cap) {
      z_s_ast_start <- a_i_ast
      if (adversarial) {
        z_s_dag_start <- a_i_dag
      }

      has_anchors <- is.finite(Lambda_sum_num) && Lambda_sum_num > 0
      if (has_anchors) {
        zbar_ast <- strenv$jnp$divide(B_ast, Lambda_sum)
        if (adversarial) {
          zbar_dag <- strenv$jnp$divide(B_dag, Lambda_sum)
        }
      } else {
        zbar_ast <- NULL
        zbar_dag <- NULL
      }

      lambda_use_num <- lambda_stage_num
      if (!is.finite(lambda_use_num) || lambda_use_num < 0) {
        lambda_use_num <- 0
      }
      if (lambda_use_num > 0) {
        stage_len_num <- ceiling((16 * L_est) / lambda_use_num)
        if (!is.finite(stage_len_num) || stage_len_num < 1) {
          stage_len_num <- 1
        }
      } else {
        stage_len_num <- nSGD - i
      }
      stage_len <- as.integer(min(stage_len_num, nSGD - i))
      if (!is.finite(stage_len) || stage_len < 1) {
        stage_len <- as.integer(min(1, nSGD - i))
      }

      lambda_use_tf <- strenv$jnp$array(lambda_use_num, strenv$dtj)

      half_count <- 0L
      half_sample_ast <- NULL
      half_sample_dag <- NULL

      for (t in seq_len(stage_len)) {
        i <- i + 1
        rain_log_step(i)

        SEED <- strenv$jax$random$split(SEED)[[1L]]
        seed_base <- SEED

        if (adversarial) {
          base_grad_dag <- call_grad_eval_dag(
            a_i_ast, a_i_dag,
            INTERCEPT_ast_,  COEFFICIENTS_ast_,
            INTERCEPT_dag_,  COEFFICIENTS_dag_,
            INTERCEPT_ast0_, COEFFICIENTS_ast0_,
            INTERCEPT_dag0_, COEFFICIENTS_dag0_,
            P_VEC_FULL_ast, P_VEC_FULL_dag,
            SLATE_VEC_ast, SLATE_VEC_dag,
            LAMBDA,
            Q_SIGN_ <- strenv$jnp$array(-1.),
            seed_base,
            BASELINE_PREV = reinforce_baseline_dag_state
          )
          record_reinforce_diag(i, "dag", base_grad_dag)
          reinforce_baseline_dag_state <- advance_reinforce_baseline(reinforce_baseline_dag_state, base_grad_dag)
          loss_dag_val <- base_grad_dag[[1]]
          grad_dag <- base_grad_dag[[2]]
          if (has_anchors) {
            reg_dag <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_i_dag, zbar_dag))
            grad_dag <- strenv$jnp$subtract(grad_dag, reg_dag)
          }
        }

        base_grad_ast <- call_grad_eval_ast(
          a_i_ast, a_i_dag,
          INTERCEPT_ast_,  COEFFICIENTS_ast_,
          INTERCEPT_dag_,  COEFFICIENTS_dag_,
          INTERCEPT_ast0_, COEFFICIENTS_ast0_,
          INTERCEPT_dag0_, COEFFICIENTS_dag0_,
          P_VEC_FULL_ast, P_VEC_FULL_dag,
          SLATE_VEC_ast, SLATE_VEC_dag,
          LAMBDA,
          Q_SIGN_ <- strenv$jnp$array(1.),
          seed_base,
          BASELINE_PREV = reinforce_baseline_ast_state
        )
        record_reinforce_diag(i, "ast", base_grad_ast)
        reinforce_baseline_ast_state <- advance_reinforce_baseline(reinforce_baseline_ast_state, base_grad_ast)
        loss_ast_val <- base_grad_ast[[1]]
        grad_ast <- base_grad_ast[[2]]
        if (has_anchors) {
          reg_ast <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_i_ast, zbar_ast))
          grad_ast <- strenv$jnp$subtract(grad_ast, reg_ast)
        }

        if (adversarial) {
          a_pred_dag <- strenv$jnp$add(a_i_dag, strenv$jnp$multiply(eta_t, grad_dag))
        } else {
          a_pred_dag <- a_i_dag
        }
        a_pred_ast <- strenv$jnp$add(a_i_ast, strenv$jnp$multiply(eta_t, grad_ast))

        if (use_uniform_half) {
          half_count <- half_count + 1L
          if (stats::runif(1) <= 1 / half_count) {
            half_sample_ast <- a_pred_ast
            if (adversarial) {
              half_sample_dag <- a_pred_dag
            }
          }
        }

        SEED <- strenv$jax$random$split(SEED)[[1L]]
        seed_look <- SEED

        if (adversarial) {
          grad_pred_dag <- call_grad_eval_dag(
            a_pred_ast, a_pred_dag,
            INTERCEPT_ast_,  COEFFICIENTS_ast_,
            INTERCEPT_dag_,  COEFFICIENTS_dag_,
            INTERCEPT_ast0_, COEFFICIENTS_ast0_,
            INTERCEPT_dag0_, COEFFICIENTS_dag0_,
            P_VEC_FULL_ast, P_VEC_FULL_dag,
            SLATE_VEC_ast, SLATE_VEC_dag,
            LAMBDA,
            Q_SIGN_ <- strenv$jnp$array(-1.),
            seed_look,
            BASELINE_PREV = reinforce_baseline_dag_state
          )
          reinforce_baseline_dag_state <- advance_reinforce_baseline(reinforce_baseline_dag_state, grad_pred_dag)
          grad_pred_dag <- grad_pred_dag[[2]]
          if (has_anchors) {
            reg_pred_dag <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_pred_dag, zbar_dag))
            grad_pred_dag <- strenv$jnp$subtract(grad_pred_dag, reg_pred_dag)
          }
        }

        grad_pred_ast <- call_grad_eval_ast(
          a_pred_ast, a_pred_dag,
          INTERCEPT_ast_,  COEFFICIENTS_ast_,
          INTERCEPT_dag_,  COEFFICIENTS_dag_,
          INTERCEPT_ast0_, COEFFICIENTS_ast0_,
          INTERCEPT_dag0_, COEFFICIENTS_dag0_,
          P_VEC_FULL_ast, P_VEC_FULL_dag,
          SLATE_VEC_ast, SLATE_VEC_dag,
          LAMBDA,
          Q_SIGN_ <- strenv$jnp$array(1.),
          seed_look,
          BASELINE_PREV = reinforce_baseline_ast_state
        )
        reinforce_baseline_ast_state <- advance_reinforce_baseline(reinforce_baseline_ast_state, grad_pred_ast)
        grad_pred_ast <- grad_pred_ast[[2]]
        if (has_anchors) {
          reg_pred_ast <- strenv$jnp$multiply(Lambda_sum, strenv$jnp$subtract(a_pred_ast, zbar_ast))
          grad_pred_ast <- strenv$jnp$subtract(grad_pred_ast, reg_pred_ast)
        }

        if (adversarial) {
          a_i_dag <- strenv$jnp$add(a_i_dag, strenv$jnp$multiply(eta_t, grad_pred_dag))
          strenv$grad_mag_dag_vec[i] <- list(strenv$jnp$linalg$norm(grad_pred_dag))
          strenv$loss_dag_vec[i] <- list(loss_dag_val)
        }
        a_i_ast <- strenv$jnp$add(a_i_ast, strenv$jnp$multiply(eta_t, grad_pred_ast))
        strenv$grad_mag_ast_vec[i] <- list(strenv$jnp$linalg$norm(grad_pred_ast))
        strenv$loss_ast_vec[i] <- list(loss_ast_val)

        strenv$inv_learning_rate_ast_vec[i] <- list(inv_lr_val)
        if (adversarial) {
          strenv$inv_learning_rate_dag_vec[i] <- list(inv_lr_val)
        }

        strenv$rain_lambda_vec[i] <- list(lambda_use_tf)
        strenv$rain_lambda_s_vec[i] <- list(lambda_use_tf)
        strenv$rain_lambda_sum_vec[i] <- list(Lambda_sum)
        strenv$rain_stage_idx[i] <- stage_idx
        if (has_anchors) {
          strenv$rain_anchor_bar_norm_ast[i] <- list(
            strenv$jnp$linalg$norm(strenv$jnp$subtract(a_i_ast, zbar_ast))
          )
          if (adversarial) {
            strenv$rain_anchor_bar_norm_dag[i] <- list(
              strenv$jnp$linalg$norm(strenv$jnp$subtract(a_i_dag, zbar_dag))
            )
          }
        } else {
          strenv$rain_anchor_bar_norm_ast[i] <- list(strenv$jnp$array(0., strenv$dtj))
          if (adversarial) {
            strenv$rain_anchor_bar_norm_dag[i] <- list(strenv$jnp$array(0., strenv$dtj))
          }
        }
      }

      if (use_uniform_half && half_count > 0L) {
        a_i_ast <- half_sample_ast
        if (adversarial && !is.null(half_sample_dag)) {
          a_i_dag <- half_sample_dag
        }
      }

      Lambda_sum <- strenv$jnp$add(Lambda_sum, lambda_use_tf)
      Lambda_sum_num <- Lambda_sum_num + lambda_use_num
      B_ast <- strenv$jnp$add(B_ast, strenv$jnp$multiply(lambda_use_tf, z_s_ast_start))
      if (adversarial) {
        B_dag <- strenv$jnp$add(B_dag, strenv$jnp$multiply(lambda_use_tf, z_s_dag_start))
      }
      lambda_stage_num <- lambda_stage_num * (1 + rain_gamma)
      stage_idx <- stage_idx + 1L
    }
  } else {
    while(goOn == F){
      if ((i <- i + 1) < 5 | i %in% unique(ceiling(c(0.25, 0.5, 0.75, 1) * nSGD))) { 
        message(sprintf("SGD Iteration: %s of %s", i, nSGD) ) 
      }

      # do dag updates ("min" step - only do in adversarial mode)
      if( i %% 1 == 0 & adversarial & !use_joint_extragrad ){
      # note: dQ_da_dag built off FullGetQStar_
      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      grad_i_dag <- call_grad_eval_dag(  a_i_ast, a_i_dag,                    #1,2
                                INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                LAMBDA,                              #15
                                Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                SEED,                                #17
                                BASELINE_PREV = reinforce_baseline_dag_state
                                )
      record_reinforce_diag(i, "dag", grad_i_dag)
      reinforce_baseline_dag_state <- advance_reinforce_baseline(reinforce_baseline_dag_state, grad_i_dag)
      strenv$loss_dag_vec[i] <- list(grad_i_dag[[1]]); grad_i_dag <- grad_i_dag[[2]]
      if (use_rain) {
        grad_i_dag <- strenv$jnp$subtract(
          grad_i_dag,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_dag, anchor_dag))
        )
      }

      # choose gradient to apply (OGDA / Extra-Gradient)
      grad_step_dag <- grad_i_dag
      if (!is.null(grad_prev_dag) && optimism == "ogda") {
        grad_step_dag <- strenv$jnp$add(grad_i_dag,
                                        strenv$jnp$multiply(strenv$jnp$array(optimism_coef, strenv$dtj),
                                                            strenv$jnp$subtract(grad_i_dag, grad_prev_dag)))
      }

      if(FALSE){ # sanity view of utilities (debug code) 
        TSAMP1 <- strenv$jax$vmap(function(s_){
          strenv$getMultinomialSamp(SLATE_VEC_ast, MNtemp, s_, strenv$ParameterizationType, strenv$d_locator_use)
        }, in_axes = list(0L))(strenv$jax$random$split(SEED, nMonte_Qglm))
        TSAMP2 <- strenv$jax$vmap(function(s_){
          strenv$getMultinomialSamp(SLATE_VEC_ast, MNtemp, s_, strenv$ParameterizationType, strenv$d_locator_use)
        }, in_axes = list(0L))(strenv$jax$random$split(SEED, nMonte_Qglm))
        ast0_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                                                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        ast_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                        INTERCEPT_ast_, COEFFICIENTS_ast_,
                        INTERCEPT_ast_, COEFFICIENTS_ast_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        dag0_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                                                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        dag_ <- strenv$np$array(strenv$jax$vmap(function(t_,t__){ getQStar_single(t_, t__,
                                                                  INTERCEPT_dag_, COEFFICIENTS_dag_,
                                                                  INTERCEPT_dag_, COEFFICIENTS_dag_)}, in_axes = list(0L,0L))(TSAMP1,TSAMP2))[,1,1]
        cor(cbind(ast0_,ast_,dag0_,dag_))
        plot(ast_,dag_)
      }
      
      if(!use_optax){
        if(i == 1){
          inv_learning_rate_da_dag <- 
            strenv$jax$lax$stop_gradient(
                strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm( grad_i_dag )))
            )
        }
        inv_learning_rate_da_dag <-  strenv$jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_dag, grad_i_dag))
        lr_dag_i <- strenv$jnp$sqrt(inv_learning_rate_da_dag)

        if (use_extragrad) {
          # look-ahead step, then recompute gradient
          a_pred_dag <- GetUpdatedParameters(a_vec = a_i_dag,
                                             grad_i = grad_i_dag,
                                             inv_learning_rate_i = lr_dag_i)
          if (use_smp) {
            gamma_dag <- strenv$jnp$reciprocal(lr_dag_i)
            smp_sum_weighted_dag <- strenv$jnp$add(smp_sum_weighted_dag,
                                                   strenv$jnp$multiply(gamma_dag, a_pred_dag))
            smp_sum_gamma_dag <- strenv$jnp$add(smp_sum_gamma_dag, gamma_dag)
            smp_gamma_dag_vec[[i]] <- gamma_dag
          }
          SEED   <- strenv$jax$random$split(SEED)[[1L]]
          grad_pred_dag <- call_grad_eval_dag(  a_i_ast, a_pred_dag,                 #1,2
                                       INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                       INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                       INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                       INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                       P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                       SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                       LAMBDA,                              #15
                                       Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                       SEED,                                #17
                                       BASELINE_PREV = reinforce_baseline_dag_state
                                       )
          reinforce_baseline_dag_state <- advance_reinforce_baseline(reinforce_baseline_dag_state, grad_pred_dag)
          grad_step_dag <- grad_pred_dag[[2]]
          if (use_rain) {
            grad_step_dag <- strenv$jnp$subtract(
              grad_step_dag,
              strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_dag, anchor_dag))
            )
          }
        }

        a_i_dag <- GetUpdatedParameters(a_vec = a_i_dag, 
                                        grad_i = grad_step_dag,
                                        inv_learning_rate_i = lr_dag_i)
      }
      if(use_optax){
        updates_and_opt_state_dag <- jit_update_dag( updates = grad_i_dag, 
                                                     state = opt_state_dag, 
                                                     params = a_i_dag )
        opt_state_dag <- updates_and_opt_state_dag[[2]]
        a_i_dag <- jit_apply_updates_dag(params = a_i_dag,  
                                         updates = updates_and_opt_state_dag[[1]])
      }

      strenv$grad_mag_dag_vec[i] <- list(strenv$jnp$linalg$norm( grad_step_dag ))
      grad_prev_dag <- grad_i_dag
      if(!use_optax){ strenv$inv_learning_rate_dag_vec[i] <- list( inv_learning_rate_da_dag ) }
    }

    # do updates ("max" step)
    if( (i %% 1 == 0 | (!adversarial)) && !use_joint_extragrad ){
      SEED   <- strenv$jax$random$split(SEED)[[1L]] 
      grad_i_ast <- call_grad_eval_ast( a_i_ast, a_i_dag,
                               INTERCEPT_ast_,  COEFFICIENTS_ast_,
                               INTERCEPT_dag_,  COEFFICIENTS_dag_,
                               INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                               INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                               P_VEC_FULL_ast, P_VEC_FULL_dag,
                               SLATE_VEC_ast, SLATE_VEC_dag,
                               LAMBDA, 
                               Q_SIGN_ <- strenv$jnp$array(1.),
                               SEED,
                               BASELINE_PREV = reinforce_baseline_ast_state
                               )
      record_reinforce_diag(i, "ast", grad_i_ast)
      reinforce_baseline_ast_state <- advance_reinforce_baseline(reinforce_baseline_ast_state, grad_i_ast)
      strenv$loss_ast_vec[i] <- list(grad_i_ast[[1]]); grad_i_ast <- grad_i_ast[[2]]
      if (use_rain) {
        grad_i_ast <- strenv$jnp$subtract(
          grad_i_ast,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_ast, anchor_ast))
        )
      }
      grad_step_ast <- grad_i_ast
      if (!is.null(grad_prev_ast) && optimism == "ogda") {
        grad_step_ast <- strenv$jnp$add(grad_i_ast,
                                        strenv$jnp$multiply(strenv$jnp$array(optimism_coef, strenv$dtj),
                                                            strenv$jnp$subtract(grad_i_ast, grad_prev_ast)))
      }
      
      if(!use_optax){
        if(i==1){ 
          inv_learning_rate_da_ast <- strenv$jax$lax$stop_gradient(
                strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm(grad_i_ast)))  
            )
        }
        inv_learning_rate_da_ast <-  strenv$jax$lax$stop_gradient( GetInvLR(inv_learning_rate_da_ast, grad_i_ast) )
        lr_ast_i <- strenv$jnp$sqrt(inv_learning_rate_da_ast)

        if (use_extragrad) {
          a_pred_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                             grad_i = grad_i_ast,
                                             inv_learning_rate_i = lr_ast_i)
          if (use_smp) {
            gamma_ast <- strenv$jnp$reciprocal(lr_ast_i)
            smp_sum_weighted_ast <- strenv$jnp$add(smp_sum_weighted_ast,
                                                   strenv$jnp$multiply(gamma_ast, a_pred_ast))
            smp_sum_gamma_ast <- strenv$jnp$add(smp_sum_gamma_ast, gamma_ast)
            smp_gamma_ast_vec[[i]] <- gamma_ast
          }
          SEED   <- strenv$jax$random$split(SEED)[[1L]] 
          grad_pred_ast <- call_grad_eval_ast( a_pred_ast, a_i_dag,
                                      INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                      INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                      INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                      INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                      P_VEC_FULL_ast, P_VEC_FULL_dag,
                                      SLATE_VEC_ast, SLATE_VEC_dag,
                                      LAMBDA, 
                                      Q_SIGN_ <- strenv$jnp$array(1.),
                                      SEED,
                                      BASELINE_PREV = reinforce_baseline_ast_state
                                      )
          reinforce_baseline_ast_state <- advance_reinforce_baseline(reinforce_baseline_ast_state, grad_pred_ast)
          grad_step_ast <- grad_pred_ast[[2]]
          if (use_rain) {
            grad_step_ast <- strenv$jnp$subtract(
              grad_step_ast,
              strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_ast, anchor_ast))
            )
          }
        }

        a_i_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                        grad_i = grad_step_ast,
                                        inv_learning_rate_i = lr_ast_i)
      }

      if(use_optax){
        updates_and_opt_state_ast <- jit_update_ast( updates = grad_i_ast, 
                                                     state = opt_state_ast, 
                                                     params = a_i_ast)
        opt_state_ast <- updates_and_opt_state_ast[[2]]
        a_i_ast <- jit_apply_updates_ast(params = a_i_ast, 
                                         updates = updates_and_opt_state_ast[[1]])
      }

      strenv$grad_mag_ast_vec[i] <- list( strenv$jnp$linalg$norm( grad_step_ast ) )
      grad_prev_ast <- grad_i_ast
      if(!use_optax){ strenv$inv_learning_rate_ast_vec[i] <- list( inv_learning_rate_da_ast ) }
    }

    # Joint look-ahead (adversarial mode, non-optax)
    if (use_joint_extragrad) {
      start_ast <- a_i_ast
      start_dag <- a_i_dag

      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      base_grad_dag <- call_grad_eval_dag(  a_i_ast, a_i_dag,                    #1,2
                                   INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                   INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                   INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                   P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                   SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                   LAMBDA,                              #15
                                   Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                   SEED,                                #17
                                   BASELINE_PREV = reinforce_baseline_dag_state
      )
      record_reinforce_diag(i, "dag", base_grad_dag)
      reinforce_baseline_dag_state <- advance_reinforce_baseline(reinforce_baseline_dag_state, base_grad_dag)
      strenv$loss_dag_vec[i] <- list(base_grad_dag[[1]]); base_grad_dag <- base_grad_dag[[2]]
      if (use_rain) {
        base_grad_dag <- strenv$jnp$subtract(
          base_grad_dag,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_dag, anchor_dag))
        )
      }

      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      base_grad_ast <- call_grad_eval_ast( a_i_ast, a_i_dag,
                                  INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                  INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                  INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                  P_VEC_FULL_ast, P_VEC_FULL_dag,
                                  SLATE_VEC_ast, SLATE_VEC_dag,
                                  LAMBDA, 
                                  Q_SIGN_ <- strenv$jnp$array(1.),
                                  SEED,
                                  BASELINE_PREV = reinforce_baseline_ast_state
      )
      record_reinforce_diag(i, "ast", base_grad_ast)
      reinforce_baseline_ast_state <- advance_reinforce_baseline(reinforce_baseline_ast_state, base_grad_ast)
      strenv$loss_ast_vec[i] <- list(base_grad_ast[[1]]); base_grad_ast <- base_grad_ast[[2]]
      if (use_rain) {
        base_grad_ast <- strenv$jnp$subtract(
          base_grad_ast,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_i_ast, anchor_ast))
        )
      }

      if(i == 1){
        inv_learning_rate_da_dag <- 
          strenv$jax$lax$stop_gradient(
            strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm( base_grad_dag )))
          )
        inv_learning_rate_da_ast <- strenv$jax$lax$stop_gradient(
          strenv$jnp$maximum(INIT_MIN_GRAD_ACCUM, 10*strenv$jnp$square(strenv$jnp$linalg$norm(base_grad_ast)))  
        )
      }
      inv_learning_rate_da_dag <-  strenv$jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_dag, base_grad_dag))
      inv_learning_rate_da_ast <-  strenv$jax$lax$stop_gradient(GetInvLR(inv_learning_rate_da_ast, base_grad_ast))
      lr_dag_i <- strenv$jnp$sqrt(inv_learning_rate_da_dag)
      lr_ast_i <- strenv$jnp$sqrt(inv_learning_rate_da_ast)

      # look-ahead for both players
      a_pred_dag <- GetUpdatedParameters(a_vec = a_i_dag,
                                         grad_i = base_grad_dag,
                                         inv_learning_rate_i = lr_dag_i)
      a_pred_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                         grad_i = base_grad_ast,
                                         inv_learning_rate_i = lr_ast_i)
      if (use_smp) {
        gamma_dag <- strenv$jnp$reciprocal(lr_dag_i)
        gamma_ast <- strenv$jnp$reciprocal(lr_ast_i)
        smp_sum_weighted_dag <- strenv$jnp$add(smp_sum_weighted_dag,
                                               strenv$jnp$multiply(gamma_dag, a_pred_dag))
        smp_sum_gamma_dag <- strenv$jnp$add(smp_sum_gamma_dag, gamma_dag)
        smp_gamma_dag_vec[[i]] <- gamma_dag
        smp_sum_weighted_ast <- strenv$jnp$add(smp_sum_weighted_ast,
                                               strenv$jnp$multiply(gamma_ast, a_pred_ast))
        smp_sum_gamma_ast <- strenv$jnp$add(smp_sum_gamma_ast, gamma_ast)
        smp_gamma_ast_vec[[i]] <- gamma_ast
      }

      SEED   <- strenv$jax$random$split(SEED)[[1L]]
      grad_pred_dag <- call_grad_eval_dag(  a_pred_ast, a_pred_dag,               #1,2
                                   INTERCEPT_ast_,  COEFFICIENTS_ast_,  #3,4
                                   INTERCEPT_dag_,  COEFFICIENTS_dag_,  #5,6
                                   INTERCEPT_ast0_, COEFFICIENTS_ast0_, #7,8
                                   INTERCEPT_dag0_, COEFFICIENTS_dag0_, #9,10
                                   P_VEC_FULL_ast, P_VEC_FULL_dag,      #11,12
                                   SLATE_VEC_ast, SLATE_VEC_dag,        #13,14
                                   LAMBDA,                              #15
                                   Q_SIGN_ <- strenv$jnp$array(-1.),    #16
                                   SEED,                                #17
                                   BASELINE_PREV = reinforce_baseline_dag_state
      )
      reinforce_baseline_dag_state <- advance_reinforce_baseline(reinforce_baseline_dag_state, grad_pred_dag)
      grad_pred_dag <- grad_pred_dag[[2]]
      if (use_rain) {
        grad_pred_dag <- strenv$jnp$subtract(
          grad_pred_dag,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_dag, anchor_dag))
        )
      }

      SEED   <- strenv$jax$random$split(SEED)[[1L]] 
      grad_pred_ast <- call_grad_eval_ast( a_pred_ast, a_pred_dag,
                                  INTERCEPT_ast_,  COEFFICIENTS_ast_,
                                  INTERCEPT_dag_,  COEFFICIENTS_dag_,
                                  INTERCEPT_ast0_, COEFFICIENTS_ast0_,
                                  INTERCEPT_dag0_, COEFFICIENTS_dag0_,
                                  P_VEC_FULL_ast, P_VEC_FULL_dag,
                                  SLATE_VEC_ast, SLATE_VEC_dag,
                                  LAMBDA, 
                                  Q_SIGN_ <- strenv$jnp$array(1.),
                                  SEED,
                                  BASELINE_PREV = reinforce_baseline_ast_state
      )
      reinforce_baseline_ast_state <- advance_reinforce_baseline(reinforce_baseline_ast_state, grad_pred_ast)
      grad_pred_ast <- grad_pred_ast[[2]]
      if (use_rain) {
        grad_pred_ast <- strenv$jnp$subtract(
          grad_pred_ast,
          strenv$jnp$multiply(rain_lambda_t, strenv$jnp$subtract(a_pred_ast, anchor_ast))
        )
      }

      a_i_dag <- GetUpdatedParameters(a_vec = a_i_dag, 
                                      grad_i = grad_pred_dag,
                                      inv_learning_rate_i = lr_dag_i)
      a_i_ast <- GetUpdatedParameters(a_vec = a_i_ast, 
                                      grad_i = grad_pred_ast,
                                      inv_learning_rate_i = lr_ast_i)

      strenv$grad_mag_dag_vec[i] <- list(strenv$jnp$linalg$norm( grad_pred_dag ))
      strenv$grad_mag_ast_vec[i] <- list(strenv$jnp$linalg$norm( grad_pred_ast ))
      grad_prev_dag <- base_grad_dag
      grad_prev_ast <- base_grad_ast
      strenv$inv_learning_rate_dag_vec[i] <- list( inv_learning_rate_da_dag )
      strenv$inv_learning_rate_ast_vec[i] <- list( inv_learning_rate_da_ast )
      strenv$extragrad_eval_points[[i]] <- list(
        start = list(a_ast = start_ast, a_dag = start_dag),
        ast = list(a_pred_ast = a_pred_ast, a_pred_dag = a_pred_dag),
        dag = list(a_pred_ast = a_pred_ast, a_pred_dag = a_pred_dag)
      )
    }
      if(i >= nSGD){goOn <- TRUE}
      if (use_rain) {
        anchor_ast <- a_i_ast
        anchor_dag <- a_i_dag
      }
    }
  }

  if (use_smp) {
    a_i_ast <- strenv$jnp$divide(smp_sum_weighted_ast, smp_sum_gamma_ast)
    if (adversarial) {
      a_i_dag <- strenv$jnp$divide(smp_sum_weighted_dag, smp_sum_gamma_dag)
    }
    strenv$smp_sum_gamma_ast <- smp_sum_gamma_ast
    strenv$smp_sum_gamma_dag <- if (adversarial) smp_sum_gamma_dag else NULL
    strenv$smp_avg_ast <- a_i_ast
    strenv$smp_avg_dag <- if (adversarial) a_i_dag else NULL
    strenv$smp_gamma_ast_vec <- smp_gamma_ast_vec
    strenv$smp_gamma_dag_vec <- if (adversarial) smp_gamma_dag_vec else NULL
  }

  message("Saving output from gd [getQPiStar_gd]...")
  {
    pi_star_ast_full_simplex_ <- getPrettyPi( pi_star_ast_<-strenv$a2Simplex_diff_use(a_i_ast),
                                              strenv$ParameterizationType,
                                              strenv$d_locator_use,       
                                              strenv$main_comp_mat,   
                                              strenv$shadow_comp_mat)
    pi_star_dag_full_simplex_ <- getPrettyPi( pi_star_dag_<-strenv$a2Simplex_diff_use(a_i_dag),
                                              strenv$ParameterizationType,
                                              strenv$d_locator_use,       
                                              strenv$main_comp_mat,   
                                              strenv$shadow_comp_mat)

    if (!adversarial) {
      average_case_eval <- evaluate_average_case_q(
        pi_star_ast = pi_star_ast_,
        pi_star_dag = pi_star_dag_,
        INTERCEPT_ast_ = INTERCEPT_ast_,
        COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
        INTERCEPT_dag_ = INTERCEPT_dag_,
        COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
        seed_in = SEED,
        phase = "report",
        outcome_model_type = outcome_model_type,
        glm_family = glm_family,
        nMonte_Qglm = nMonte_Qglm,
        temperature = MNtemp,
        ParameterizationType = strenv$ParameterizationType,
        d_locator_use = strenv$d_locator_use,
        q_fxn = QFXN,
        single_party = !isTRUE(diff),
        force_reinforce = force_reinforce
      )
      q_star_f <- average_case_eval$q_vec
      SEED <- average_case_eval$seed_next
    }
    if (adversarial) {
      adversarial_eval <- evaluate_adversarial_q(
        pi_star_ast = pi_star_ast_,
        pi_star_dag = pi_star_dag_,
        a_i_ast, a_i_dag,
        INTERCEPT_ast_ = INTERCEPT_ast_,
        COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
        INTERCEPT_dag_ = INTERCEPT_dag_,
        COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
        INTERCEPT_ast0_ = INTERCEPT_ast0_,
	        COEFFICIENTS_ast0_ = COEFFICIENTS_ast0_,
	        INTERCEPT_dag0_ = INTERCEPT_dag0_,
	        COEFFICIENTS_dag0_ = COEFFICIENTS_dag0_,
	        P_VEC_FULL_ast_ = P_VEC_FULL_ast,
	        P_VEC_FULL_dag_ = P_VEC_FULL_dag,
	        SLATE_VEC_ast_ = SLATE_VEC_ast,
	        SLATE_VEC_dag_ = SLATE_VEC_dag,
	        LAMBDA_ = LAMBDA,
	        Q_SIGN = strenv$jnp$array(1.),
	        seed_in = SEED,
	        phase = "report",
	        outcome_model_type = outcome_model_type,
	        glm_family = glm_family,
	        nMonte_Qglm = nMonte_Qglm,
	        nMonte_adversarial = nMonte_adversarial,
	        primary_pushforward = primary_pushforward,
	        primary_n_entrants = primary_n_entrants,
	        primary_n_field = primary_n_field,
	        temperature = MNtemp,
	        ParameterizationType = strenv$ParameterizationType,
	        d_locator_use = strenv$d_locator_use
	      )
	      q_star_val <- reshape_scalar_q_value(adversarial_eval$q_ast,
	                                           pi_star_ast_full_simplex_)
	      q_star_f <- strenv$jnp$concatenate(list(q_star_val, q_star_val, q_star_val), 0L)
	      SEED <- adversarial_eval$seed_next
	    }

    # setup results for returning
    ret_array <- strenv$jnp$concatenate(ifelse(gd_full_simplex, 
                                               yes = list(list( q_star_f, 
                                                                pi_star_ast_full_simplex_, 
                                                                pi_star_dag_full_simplex_ )), 
                                               no  = list(list( q_star_f, 
                                                                pi_star_ast_, 
                                                                pi_star_dag_ )))[[1]])
    if( functionReturn ){
                            ret_array <- list(ret_array,
                                               list("dQ_da_ast" = grad_eval_ast,
                                                    "dQ_da_dag" = grad_eval_dag,
                                                    "QFXN" = QFXN,
                                                    "getMultinomialSamp" = strenv$getMultinomialSamp,
                                                    "a_i_ast" = a_i_ast, 
                                                    "a_i_dag" = a_i_dag
                                               ) )
    }
    return( ret_array )
  }
}
