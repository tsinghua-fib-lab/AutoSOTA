make_crossfit_adversarial_fixture <- function(blocks = 4L, seed = 2026L) {
  withr::local_seed(seed)
  n_pairs <- as.integer(blocks) * 4L
  pair_types <- rep(c("cross_A", "cross_B", "same_A", "same_B"),
                    length.out = n_pairs)
  respondent_pair <- ifelse(pair_types %in% c("cross_A", "same_A"), "A", "B")
  left_group <- ifelse(pair_types == "same_B", "B", "A")
  right_group <- ifelse(pair_types == "same_A", "A", "B")
  competition_pair <- ifelse(pair_types %in% c("cross_A", "cross_B"),
                             "Different", "Same")
  W_left <- data.frame(
    female = sample(c(0L, 1L), n_pairs, replace = TRUE),
    stringsAsFactors = FALSE
  )
  W_right <- data.frame(
    female = sample(c(0L, 1L), n_pairs, replace = TRUE),
    stringsAsFactors = FALSE
  )
  y_left <- stats::rbinom(n_pairs, size = 1L, prob = 0.5)

  list(
    Y = c(y_left, 1L - y_left),
    W = rbind(W_left, W_right),
    pair_id = c(seq_len(n_pairs), seq_len(n_pairs)),
    respondent_id = c(seq_len(n_pairs), seq_len(n_pairs)),
    respondent_task_id = c(seq_len(n_pairs), seq_len(n_pairs)),
    profile_order = c(rep(1L, n_pairs), rep(2L, n_pairs)),
    respondent_group = c(respondent_pair, respondent_pair),
    candidate_group = c(left_group, right_group),
    competition_group = c(competition_pair, competition_pair),
    pair_types = pair_types
  )
}

crossfit_adversarial_p_list <- function() {
  list(female = stats::setNames(c(0.5, 0.5), c("0", "1")))
}

test_that("crossfit Q control validates defaults and overrides", {
  defaults <- cs_crossfit_q_default_control()
  expect_equal(defaults$estimators, c("dr_hajek", "dr", "ips", "snips", "model"))
  expect_equal(defaults$headline, "dr_hajek")

  control <- cs_crossfit_q_default_control(list(
    folds = 2,
    estimators = c("ips", "dr_hajek", "ips"),
    headline = "ips",
    return_fold_results = FALSE
  ))

  expect_equal(control$folds, 2L)
  expect_equal(control$estimators, c("ips", "dr_hajek"))
  expect_equal(control$headline, "ips")
  expect_false(control$return_fold_results)
  expect_null(control$perspective_group)

  expect_error(
    cs_crossfit_q_default_control(list(estimators = "unknown")),
    "unknown value"
  )
  expect_error(
    cs_crossfit_q_default_control(list(folds = 1)),
    "integer >= 2"
  )
  expect_error(
    cs_crossfit_q_default_control(list(headline = "snips", estimators = "dr")),
    "must be included"
  )
})

test_that("crossfit Q validation enforces pairwise v1 constraints", {
  W <- data.frame(
    A = c("a", "b", "a", "b"),
    B = c("x", "x", "y", "y"),
    stringsAsFactors = FALSE
  )
  Y <- c(1, 0, 0, 1)
  pair_id <- c(1, 1, 2, 2)
  profile_order <- c(1, 2, 1, 2)
  p_list <- list(
    A = c(a = 0.5, b = 0.5),
    B = c(x = 0.5, y = 0.5)
  )
  control <- cs_crossfit_q_default_control(list(folds = 2))

  validated <- cs_crossfit_q_validate(
    Y = Y,
    W = W,
    pair_id = pair_id,
    profile_order = profile_order,
    p_list = p_list,
    diff = TRUE,
    adversarial = FALSE,
    K = 1,
    outcome_model_type = "glm",
    control = control
  )
  expect_equal(nrow(validated$pair_mat), 2L)

  expect_error(
    cs_crossfit_q_validate(
      Y = Y, W = W, pair_id = pair_id, profile_order = profile_order,
      p_list = p_list, diff = FALSE, adversarial = FALSE, K = 1,
      outcome_model_type = "glm", control = control
    ),
    "diff = TRUE"
  )
  expect_error(
    cs_crossfit_q_validate(
      Y = c(1, 1, 0, 1), W = W, pair_id = pair_id,
      profile_order = profile_order, p_list = p_list, diff = TRUE,
      adversarial = FALSE, K = 1, outcome_model_type = "glm",
      control = control
    ),
    "exactly one selected"
  )
  expect_error(
    cs_crossfit_q_validate(
      Y = Y, W = W, pair_id = pair_id, profile_order = profile_order,
      p_list = p_list, diff = TRUE, adversarial = FALSE, K = 2,
      outcome_model_type = "glm", control = control
    ),
    "requires non-null respondent covariates X"
  )
  k3_validated <- cs_crossfit_q_validate(
    Y = Y,
    W = W,
    X = cbind(Z = c(0, 0, 1, 1)),
    pair_id = pair_id,
    profile_order = profile_order,
    p_list = p_list,
    diff = TRUE,
    adversarial = FALSE,
    K = 3,
    outcome_model_type = "glm",
    respondent_id = c(1, 1, 2, 2),
    control = control
  )
  expect_equal(k3_validated$mode, "covariate_sensitive_pairwise_glm")
  expect_equal(k3_validated$K, 3L)
  expect_error(
    cs_crossfit_q_validate(
      Y = Y, W = W, pair_id = pair_id, profile_order = profile_order,
      p_list = p_list, diff = TRUE, adversarial = FALSE, K = 1,
      outcome_model_type = "glm", force_gaussian = TRUE, control = control
    ),
    "force_gaussian = FALSE"
  )
})

test_that("FactorHet formulas quote non-syntactic profile factor names", {
  W <- data.frame(
    Sex = c("Male", "Female", "Male", "Female"),
    Age = c("45", "60", "30", "45"),
    check.names = FALSE
  )
  W[["Experience in public office"]] <- c("None", "4 years", "8 years", "12 years")
  W[["Salient personal characteristics"]] <- c("Honest", "Intelligent", "Moral", "Strong")
  W[["Favorability rating among the public"]] <- c("41%", "51%", "61%", "71%")
  X <- data.frame(
    `R Age` = c(34, 35, 36, 37),
    `if` = c(0, 1, 0, 1),
    check.names = FALSE
  )
  design <- data.frame(
    Yobs = c(1, 0, 1, 0),
    W,
    X,
    check.names = FALSE
  )

  formulas <- cs_factorhet_formulas(W, X)

  expect_s3_class(formulas$main_only, "formula")
  expect_s3_class(formulas$with_interactions, "formula")
  expect_s3_class(formulas$moderator, "formula")
  expect_match(deparse1(formulas$main_only),
               "`Experience in public office`",
               fixed = TRUE)
  expect_match(deparse1(formulas$with_interactions),
               "`Salient personal characteristics`",
               fixed = TRUE)
  expect_match(deparse1(formulas$moderator), "`R Age`", fixed = TRUE)

  expect_no_error(stats::model.matrix(formulas$main_only, design))
  expect_no_error(stats::model.matrix(formulas$with_interactions, design))
  expect_no_error(stats::model.matrix(formulas$moderator, design))
})

test_that("FactorHet design preserves numeric moderator rank with character respondent ids", {
  resp_ids <- paste0("resp_", seq_len(6))
  respondent_id <- rep(resp_ids, each = 2L)
  respondent_task_id <- rep(seq_len(6), each = 2L)
  profile_order <- rep(c(1L, 2L), times = 6L)
  W <- data.frame(
    treatment = rep(c("status_quo", "new_policy"), times = 6L),
    frame = rep(c("gain", "loss", "neutral"), length.out = 12L),
    stringsAsFactors = FALSE
  )
  ages <- c(29, 34, 45, 53, 67, 74)
  X_resp <- data.frame(
    resp_age = ages,
    age_young = as.integer(ages < 40),
    age_middle = as.integer(ages >= 40 & ages < 60),
    check.names = FALSE
  )
  X <- X_resp[match(respondent_id, resp_ids), , drop = FALSE]

  prepared <- cs_factorhet_prepare_design(
    Y = rep(c(0, 1), times = 6L),
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    W = W,
    X = X
  )
  design <- prepared$design
  formulas <- cs_factorhet_formulas(prepared$W, prepared$X)
  respondent_design <- design[!duplicated(design$respondent_id), , drop = FALSE]
  moderator_matrix <- stats::model.matrix(formulas$moderator, respondent_design)

  expect_true(is.character(design$respondent_id))
  expect_true(is.numeric(design$resp_age))
  expect_true(is.numeric(design$age_young))
  expect_true(is.numeric(design$age_middle))
  expect_false(is.factor(design$resp_age))
  expect_equal(
    colnames(moderator_matrix),
    c("(Intercept)", "resp_age", "age_young", "age_middle")
  )
  expect_equal(qr(moderator_matrix)$rank, ncol(moderator_matrix))
})

test_that("FactorHet design drops rank-deficient moderator columns", {
  resp_ids <- paste0("resp_", seq_len(5))
  respondent_id <- rep(resp_ids, each = 2L)
  respondent_task_id <- rep(seq_len(5), each = 2L)
  profile_order <- rep(c(1L, 2L), times = 5L)
  W <- data.frame(
    treatment = rep(c("status_quo", "new_policy"), times = 5L),
    frame = rep(c("gain", "loss"), length.out = 10L),
    stringsAsFactors = FALSE
  )
  party_a <- c(1, 0, 1, 0, 1)
  X_resp <- data.frame(
    party_a = party_a,
    party_b = 1 - party_a,
    constant = 1,
    check.names = FALSE
  )
  X <- X_resp[match(respondent_id, resp_ids), , drop = FALSE]

  prepared <- cs_factorhet_prepare_design(
    Y = rep(c(0, 1), times = 5L),
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    W = W,
    X = X
  )
  formulas <- cs_factorhet_formulas(prepared$W, prepared$X)
  respondent_design <- prepared$design[
    !duplicated(prepared$design$respondent_id),
    ,
    drop = FALSE
  ]
  moderator_matrix <- stats::model.matrix(formulas$moderator, respondent_design)

  expect_equal(prepared$moderator_columns, "party_a")
  expect_equal(prepared$dropped_moderator_columns, c("party_b", "constant"))
  expect_false("party_b" %in% names(prepared$design))
  expect_equal(qr(moderator_matrix)$rank, ncol(moderator_matrix))
})

test_that("crossfit Q probability weights and diagnostics are computed", {
  W <- data.frame(
    A = c("a", "b", "a", "b"),
    B = c("x", "x", "y", "y"),
    stringsAsFactors = FALSE
  )
  p_list <- list(
    A = c(a = 0.5, b = 0.5),
    B = c(x = 0.5, y = 0.5)
  )
  policy <- list(
    A = c(a = 0.75, b = 0.25),
    B = c(x = 0.4, y = 0.6)
  )

  expect_equal(
    cs_crossfit_q_policy_prob(W, policy, p_list),
    c(0.75 * 0.4, 0.25 * 0.4, 0.75 * 0.6, 0.25 * 0.6)
  )

  diagnostics <- cs_crossfit_q_weight_diagnostics(c(1, 2, 3), c(1, 2, 2))
  expect_equal(diagnostics$n, 3L)
  expect_true(is.finite(diagnostics$ess))
  expect_equal(diagnostics$weight_sum, 5)
  expect_equal(diagnostics$weight_sum_ratio, 5 / 3)
  expect_true(is.finite(diagnostics$p999))
  expect_true(diagnostics$clipped)
})

test_that("crossfit Q scores heldout rows on fitted reduced GLM feature basis", {
  p_list <- list(
    A = c(a = 0.5, b = 0.5),
    B = c(x = 0.5, y = 0.5)
  )
  reduced_feature_info <- list(
    main_info = data.frame(
      d = c(1L, 2L),
      l = c(1L, 1L),
      d_index = c(1L, 2L)
    ),
    interaction_info = data.frame()
  )
  result <- list(
    est_intercept_jnp = 0,
    est_coefficients_jnp = c(0.4, -0.2),
    glm_feature_info = list(overall = reduced_feature_info),
    outcome_model_view = list(
      models = list(overall = list(glm_family = "binomial"))
    ),
    pi_star_point = list(
      A = c(a = 0.6, b = 0.4),
      B = c(x = 0.3, y = 0.7)
    )
  )

  expect_equal(
    cs_crossfit_q_feature_info_ncols(cs_crossfit_q_reconstruct_feature_info(p_list)),
    3L
  )
  expect_equal(cs_crossfit_q_feature_info_ncols(reduced_feature_info), 2L)

  focal_W <- data.frame(A = c("a", "b"), B = c("x", "y"), stringsAsFactors = FALSE)
  opponent_W <- data.frame(A = c("b", "a"), B = c("y", "x"), stringsAsFactors = FALSE)
  pred <- cs_crossfit_q_pair_predict(focal_W, opponent_W, result, p_list)
  expect_length(pred, 2L)
  expect_true(all(is.finite(pred)))

  mu <- cs_crossfit_q_policy_model_mu(
    policy = result$pi_star_point,
    opponent_W = opponent_W,
    result = result,
    p_list = p_list,
    n_draws = 4L,
    seed = 2026L,
    chunk_size = 2L
  )
  expect_length(mu, 2L)
  expect_true(all(is.finite(mu)))
})

test_that("covariate-sensitive policy probabilities mix cluster policies row-wise", {
  p_list <- list(
    A = c(a = 0.5, b = 0.5),
    B = c(x = 0.25, y = 0.75)
  )
  policies <- list(
    k1 = list(A = c(a = 0.8, b = 0.2), B = c(x = 0.1, y = 0.9)),
    k2 = list(A = c(a = 0.2, b = 0.8), B = c(x = 0.6, y = 0.4)),
    k3 = list(A = c(a = 0.5, b = 0.5), B = c(x = 0.3, y = 0.7))
  )
  rho <- rbind(
    c(k1 = 1.0, k2 = 0.0, k3 = 0.0),
    c(k1 = 0.0, k2 = 1.0, k3 = 0.0),
    c(k1 = 0.25, k2 = 0.25, k3 = 0.50)
  )

  row_probs <- cs_crossfit_q_row_soft_policy_probs(policies, rho, p_list)
  expect_equal(row_probs$A[1, ], c(a = 0.8, b = 0.2), tolerance = 1e-8)
  expect_equal(row_probs$A[2, ], c(a = 0.2, b = 0.8), tolerance = 1e-8)
  expect_equal(row_probs$A[3, ], c(a = 0.5, b = 0.5), tolerance = 1e-8)
  expect_equal(row_probs$B[3, ], c(x = 0.325, y = 0.675), tolerance = 1e-8)

  W <- data.frame(A = c("a", "b", "a"), B = c("x", "y", "y"), stringsAsFactors = FALSE)
  expect_equal(
    cs_crossfit_q_row_policy_prob(W, row_probs, p_list),
    c(0.8 * 0.1, 0.8 * 0.4, 0.5 * 0.675),
    tolerance = 1e-8
  )

  diag <- cs_crossfit_q_cluster_diagnostics(
    rho = rho,
    w = c(2, 4, 6),
    w_used = c(2, 3, 3)
  )
  expect_equal(diag$cluster, c("k1", "k2", "k3"))
  expect_equal(diag$n_oriented, c(1, 1, 1))
  expect_true(all(is.finite(diag$membership_mass)))
})

test_that("Hajek-normalized DR is stable under extreme weights and flags zero denominators", {
  y <- c(1, 0)
  m_obs <- c(0.9, 0.9)
  mu_policy <- c(0.5, 0.5)
  w_extreme <- c(100, 0)

  raw_dr <- mean(mu_policy + w_extreme * (y - m_obs))
  hajek_dr <- cs_crossfit_q_dr_hajek(mu_policy, w_extreme, y, m_obs)

  expect_gt(raw_dr, 1)
  expect_true(is.finite(hajek_dr))
  expect_true(hajek_dr >= 0 && hajek_dr <= 1)
  expect_true(cs_crossfit_q_hajek_denominator_ok(w_extreme))

  zero_weight <- c(0, 0)
  expect_true(is.na(cs_crossfit_q_dr_hajek(mu_policy, zero_weight, y, m_obs)))
  zero_diag <- cs_crossfit_q_weight_diagnostics(zero_weight, zero_weight)
  expect_false(zero_diag$hajek_denominator_ok)
})

test_that("adversarial crossfit Q validation canonicalizes contests", {
  data <- make_crossfit_adversarial_fixture(blocks = 4L)
  p_list <- crossfit_adversarial_p_list()
  control <- cs_crossfit_q_default_control(list(
    folds = 2L,
    perspective_group = "B"
  ))

  validated <- cs_crossfit_q_validate(
    Y = data$Y,
    W = data$W,
    pair_id = data$pair_id,
    profile_order = data$profile_order,
    p_list = p_list,
    diff = TRUE,
    adversarial = TRUE,
    K = 1,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    adversarial_model_strategy = "four",
    competing_group_variable_respondent = data$respondent_group,
    competing_group_variable_candidate = data$candidate_group,
    competing_group_competition_variable_candidate = data$competition_group,
    competing_group_variable_respondent_proportions = c(A = 0.4, B = 0.6),
    respondent_id = data$respondent_id,
    control = control
  )

  expect_equal(validated$mode, "adversarial_pairwise_glm")
  expect_equal(validated$groups, c("A", "B"))
  expect_equal(validated$base_group, "A")
  expect_equal(validated$other_group, "B")
  expect_equal(validated$perspective_group, "B")
  expect_equal(validated$opponent_group, "A")
  expect_equal(nrow(validated$contests), 8L)
  expect_equal(validated$contests$Y_perspective,
               data$Y[validated$contests$perspective_idx])
  expect_equal(validated$contests$Y_base,
               data$Y[validated$contests$base_idx])
  expect_equal(validated$rho, c(A = 0.4, B = 0.6), tolerance = 1e-8)

  reversed_pair_mat <- validated$pair_mat
  reversed_pair_mat[validated$contests$pair_row, ] <-
    reversed_pair_mat[validated$contests$pair_row, 2:1]
  reversed_contests <- cs_crossfit_q_build_adversarial_contests(
    Y = data$Y,
    pair_id = data$pair_id,
    pair_mat = reversed_pair_mat,
    respondent_group = validated$respondent_group,
    candidate_group = validated$candidate_group,
    competition_group = validated$competition_group,
    groups = validated$groups,
    perspective_group = validated$perspective_group,
    opponent_group = validated$opponent_group
  )
  original <- validated$contests[order(validated$contests$pair_id), ]
  reversed <- reversed_contests[order(reversed_contests$pair_id), ]
  expect_equal(reversed$base_idx, original$base_idx)
  expect_equal(reversed$other_idx, original$other_idx)
  expect_equal(reversed$Y_perspective, original$Y_perspective)

  fold_y <- cs_crossfit_q_adversarial_fold_strata(
    pair_mat = validated$pair_mat,
    respondent_group = validated$respondent_group,
    candidate_group = validated$candidate_group,
    groups = validated$groups
  )
  fold_obj <- cs_make_stratified_folds(
    n = nrow(validated$pair_mat),
    n_folds = 2L,
    y = fold_y,
    cluster = data$pair_id[validated$pair_mat[, 1]],
    seed = 123L
  )
  expect_silent(cs_crossfit_q_validate_adversarial_folds(
    fold_id = fold_obj$fold_id,
    pair_mat = validated$pair_mat,
    respondent_group = validated$respondent_group,
    candidate_group = validated$candidate_group,
    groups = validated$groups
  ))
  row_folds <- fold_obj$fold_id[match(data$pair_id,
                                      data$pair_id[validated$pair_mat[, 1]])]
  expect_true(all(vapply(split(row_folds, data$pair_id), function(x) {
    length(unique(x)) == 1L
  }, logical(1))))
})

test_that("adversarial crossfit Q validation rejects unsupported cases", {
  data <- make_crossfit_adversarial_fixture(blocks = 4L)
  p_list <- crossfit_adversarial_p_list()
  control <- cs_crossfit_q_default_control(list(folds = 2L))
  validate <- function(Y = data$Y,
                       respondent_group = data$respondent_group,
                       p_list_arg = p_list,
                       strategy = "four") {
    cs_crossfit_q_validate(
      Y = Y,
      W = data$W,
      pair_id = data$pair_id,
      profile_order = data$profile_order,
      p_list = p_list_arg,
      diff = TRUE,
      adversarial = TRUE,
      K = 1,
      outcome_model_type = "glm",
      force_gaussian = FALSE,
      adversarial_model_strategy = strategy,
      competing_group_variable_respondent = respondent_group,
      competing_group_variable_candidate = data$candidate_group,
      competing_group_competition_variable_candidate = data$competition_group,
      respondent_id = data$respondent_id,
      control = control
    )
  }

  expect_error(validate(strategy = "two"), "strategy = 'four'")

  same_a_pair <- which(data$pair_types == "same_A")[[1L]]
  bad_group <- data$respondent_group
  bad_group[data$pair_id == same_a_pair] <- "B"
  expect_error(validate(respondent_group = bad_group), "same-party pairs")

  cross_pair <- which(data$pair_types == "cross_A")[[1L]]
  bad_Y <- data$Y
  rows <- which(data$pair_id == cross_pair)
  bad_Y[rows] <- 1
  expect_error(validate(Y = bad_Y), "exactly one selected")

  zero_level <- as.character(data$W$female[[1L]])
  bad_p <- p_list
  bad_p$female[[zero_level]] <- 0
  expect_error(validate(p_list_arg = bad_p), "positive p_list probability")
})

test_that("adversarial crossfit Q aggregation uses global group weights", {
  control <- cs_crossfit_q_default_control(list(
    folds = 2L,
    weight_clip = 2,
    estimators = c("dr_hajek", "dr", "ips", "snips", "model")
  ))
  records <- data.frame(
    fold = c(1L, 1L, 2L, 2L),
    pair_row = seq_len(4L),
    pair_id = seq_len(4L),
    respondent_group = c("A", "A", "A", "B"),
    Y_perspective = c(1, 0, 1, 0),
    m_obs = c(0.5, 0.5, 0.5, 0.5),
    mu_target = c(0.5, 0.5, 0.5, 0.5),
    mu_reference = c(0.5, 0.5, 0.5, 0.5),
    weight = c(1, 1, 1, 1),
    weight_used = c(1, 1, 1, 1),
    weight_clipped = c(FALSE, FALSE, FALSE, FALSE),
    p_obs = c(0.25, 0.25, 0.25, 0.25),
    pi_obs = c(0.25, 0.25, 0.25, 0.25),
    stringsAsFactors = FALSE
  )

  agg <- cs_crossfit_q_adversarial_aggregate(
    records = records,
    control = control,
    rho = c(A = 0.25, B = 0.75),
    groups = c("A", "B")
  )

  expect_equal(sum(agg$records$a[agg$records$respondent_group == "A"]),
               0.25, tolerance = 1e-8)
  expect_equal(sum(agg$records$a[agg$records$respondent_group == "B"]),
               0.75, tolerance = 1e-8)
  expect_equal(agg$summary$Q_gain_crossfit, rep(0, nrow(agg$summary)),
               tolerance = 1e-8)

  records$weight[[1L]] <- 5
  records$weight_used[[1L]] <- 2
  records$weight_clipped[[1L]] <- TRUE
  clipped <- cs_crossfit_q_adversarial_aggregate(
    records = records,
    control = control,
    rho = c(A = 0.25, B = 0.75),
    groups = c("A", "B")
  )
  expect_true(all(clipped$summary$any_weight_clipped))
  expect_equal(clipped$summary$max_weight, rep(5, nrow(clipped$summary)))

  zero_weight <- records
  zero_weight$weight <- 0
  zero_weight$weight_used <- 0
  zero_weight$weight_clipped <- FALSE
  zero_agg <- cs_crossfit_q_adversarial_aggregate(
    records = zero_weight,
    control = control,
    rho = c(A = 0.25, B = 0.75),
    groups = c("A", "B")
  )
  expect_true(is.na(zero_agg$summary$Q_crossfit[zero_agg$summary$estimator == "dr_hajek"]))
  expect_false(zero_agg$summary$hajek_denominator_ok[zero_agg$summary$estimator == "dr_hajek"])
})

test_that("strategize can return first-class crossfit Q fields", {
  skip_on_cran()
  skip_if_no_jax()

  set.seed(2026)
  n_pairs <- 60L
  sample_profiles <- function(n) {
    data.frame(
      A = sample(c("a", "b"), n, replace = TRUE),
      B = sample(c("x", "y"), n, replace = TRUE),
      stringsAsFactors = FALSE
    )
  }
  W_left <- sample_profiles(n_pairs)
  W_right <- sample_profiles(n_pairs)
  Y_left <- stats::rbinom(n_pairs, size = 1, prob = 0.5)

  W <- rbind(W_left, W_right)
  Y <- c(Y_left, 1 - Y_left)
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))

  plot_sink <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_sink)
  on.exit({
    grDevices::dev.off()
    unlink(plot_sink)
  }, add = TRUE)

  res <- strategize(
    Y = Y,
    W = W,
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    p_list = create_p_list(W, uniform = TRUE),
    K = 1,
    nSGD = 2L,
    diff = TRUE,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    use_regularization = FALSE,
    compute_se = FALSE,
    compute_hessian = FALSE,
    conda_env = "strategize_env",
    conda_env_required = TRUE,
    crossfit_q = TRUE,
    crossfit_q_control = list(
      folds = 2L,
      n_policy_draws = 8L,
      chunk_size = 24L,
      seed = 2026L
    )
  )

  expect_true("Q_crossfit" %in% names(res))
  expect_true("Q_reference_crossfit" %in% names(res))
  expect_true("Q_gain_crossfit" %in% names(res))
  expect_true("Q_crossfit_info" %in% names(res))
  expect_true(is.finite(res$Q_crossfit))
  expect_true(is.finite(res$Q_reference_crossfit))
  expect_true(is.finite(res$Q_gain_crossfit))
  # gain_optimism = in-sample gain minus cross-fitted gain must be computable
  expect_true(is.finite(res$Q_gain_in_sample))
  expect_true(is.finite(as.numeric(res$Q_gain_in_sample) - as.numeric(res$Q_gain_crossfit)))
  expect_s3_class(res$Q_crossfit_info$summary, "data.frame")
  expect_true(all(c("dr_hajek", "dr", "ips", "snips", "model") %in%
                    res$Q_crossfit_info$summary$estimator))
})

test_that("strategize can return K > 1 covariate-sensitive crossfit Q fields", {
  skip_on_cran()
  skip_if_no_jax()
  skip_if_not_installed("FactorHet")

  set.seed(9292)
  n_pairs <- 24L
  sample_profiles <- function(n) {
    data.frame(
      A = sample(c("a", "b"), n, replace = TRUE),
      B = sample(c("x", "y"), n, replace = TRUE),
      stringsAsFactors = FALSE
    )
  }
  W_left <- sample_profiles(n_pairs)
  W_right <- sample_profiles(n_pairs)
  eta <- ifelse(W_left$A == "a", 0.5, -0.2) -
    ifelse(W_right$A == "a", 0.5, -0.2)
  Y_left <- stats::rbinom(n_pairs, size = 1L, prob = stats::plogis(eta))

  W <- rbind(W_left, W_right)
  Y <- c(Y_left, 1L - Y_left)
  pair_id <- c(seq_len(n_pairs), seq_len(n_pairs))
  respondent_id <- pair_id
  respondent_task_id <- pair_id
  profile_order <- c(rep(1L, n_pairs), rep(2L, n_pairs))
  X_pair <- data.frame(
    z1 = stats::rnorm(n_pairs),
    z2 = rep(c(0, 1, 2), length.out = n_pairs)
  )
  X <- X_pair[match(pair_id, seq_len(n_pairs)), , drop = FALSE]

  plot_sink <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_sink)
  on.exit({
    grDevices::dev.off()
    unlink(plot_sink)
  }, add = TRUE)

  res <- strategize(
    Y = Y,
    W = W,
    X = as.matrix(X),
    lambda = 0.1,
    pair_id = pair_id,
    respondent_id = respondent_id,
    respondent_task_id = respondent_task_id,
    profile_order = profile_order,
    p_list = create_p_list(W, uniform = TRUE),
    K = 3,
    nSGD = 1L,
    diff = TRUE,
    outcome_model_type = "glm",
    force_gaussian = FALSE,
    use_regularization = TRUE,
    compute_se = FALSE,
    compute_hessian = FALSE,
    conda_env = "strategize_env",
    conda_env_required = FALSE,
    crossfit_q = TRUE,
    crossfit_q_control = list(
      folds = 2L,
      n_policy_draws = 4L,
      chunk_size = 8L,
      seed = 9292L
    )
  )

  expect_equal(res$Q_crossfit_info$mode, "covariate_sensitive_pairwise_glm")
  expect_equal(res$Q_crossfit_info$K, 3L)
  expect_equal(nrow(res$Q_crossfit_info$cluster_summary), 3L)
  expect_true(all(c("min_cluster_ess_fraction", "cluster_entropy_mean") %in%
                    names(res$Q_crossfit_info$summary)))
})

test_that("strategize can return first-class adversarial crossfit Q fields", {
  skip_on_cran()
  skip_if_no_jax()

  data <- make_crossfit_adversarial_fixture(blocks = 16L, seed = 3030L)
  p_list <- crossfit_adversarial_p_list()

  plot_sink <- tempfile(fileext = ".pdf")
  grDevices::pdf(plot_sink)
  on.exit({
    grDevices::dev.off()
    unlink(plot_sink)
  }, add = TRUE)

  res <- suppressWarnings(
    strategize(
      Y = data$Y,
      W = data$W,
      lambda = 0.1,
      pair_id = data$pair_id,
      respondent_id = data$respondent_id,
      respondent_task_id = data$respondent_task_id,
      profile_order = data$profile_order,
      p_list = p_list,
      competing_group_variable_respondent = data$respondent_group,
      competing_group_variable_candidate = data$candidate_group,
      competing_group_competition_variable_candidate = data$competition_group,
      competing_group_variable_respondent_proportions = c(A = 0.5, B = 0.5),
      K = 1,
      nSGD = 2L,
      diff = TRUE,
      adversarial = TRUE,
      adversarial_model_strategy = "four",
      outcome_model_type = "glm",
      force_gaussian = FALSE,
      use_regularization = FALSE,
      compute_se = FALSE,
      compute_hessian = FALSE,
      conda_env = "strategize_env",
      conda_env_required = TRUE,
      nMonte_adversarial = 4L,
      nMonte_Qglm = 5L,
      crossfit_q = TRUE,
      crossfit_q_control = list(
        folds = 2L,
        n_policy_draws = 4L,
        chunk_size = 8L,
        seed = 2026L,
        perspective_group = "B"
      )
    )
  )

  expect_true("Q_crossfit" %in% names(res))
  expect_true("Q_reference_crossfit" %in% names(res))
  expect_true("Q_gain_crossfit" %in% names(res))
  expect_true("Q_crossfit_info" %in% names(res))
  expect_true(is.finite(res$Q_crossfit))
  expect_true(is.finite(res$Q_reference_crossfit))
  expect_true(is.finite(res$Q_gain_crossfit))
  expect_equal(res$Q_crossfit_info$mode, "adversarial_pairwise_glm")
  expect_equal(res$Q_crossfit_info$perspective_group, "B")
  expect_equal(res$Q_crossfit_info$opponent_group, "A")
  expect_equal(res$Q_crossfit_info$assignment_assumption, "independent_product_p_list")
  expect_s3_class(res$Q_crossfit_info$summary, "data.frame")
  expect_s3_class(res$Q_crossfit_info$target_summary, "data.frame")
  expect_s3_class(res$Q_crossfit_info$reference_summary, "data.frame")
  expect_true(all(c("dr_hajek", "dr", "ips", "snips", "model") %in%
                    res$Q_crossfit_info$summary$estimator))
  expect_true(all(is.finite(res$Q_crossfit_info$summary$Q_crossfit)))
  expect_true(all(is.finite(res$Q_crossfit_info$summary$Q_reference_crossfit)))
  expect_true(all(table(res$Q_crossfit_info$records$respondent_group) > 0L))

  split_info <- res$Q_crossfit_info$split
  row_folds <- split_info$fold_id_by_pair[
    match(data$pair_id, split_info$pair_id)
  ]
  expect_true(all(vapply(split(row_folds, data$pair_id), function(x) {
    length(unique(x)) == 1L
  }, logical(1))))
})
