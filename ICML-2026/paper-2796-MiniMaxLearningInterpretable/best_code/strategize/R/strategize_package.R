#' @keywords internal
"_PACKAGE"

# Base R imports ---------------------------------------------------------------
#' @importFrom graphics hist image
#' @importFrom stats as.formula coef glm lm lowess model.matrix na.omit predict qnorm rnorm runif sd setNames var vcov
#' @importFrom utils combn head modifyList read.csv write.csv
#' @importFrom reticulate %as%
NULL

# Global variables -------------------------------------------------------------
# These are used in dynamically generated code via eval(parse(text=...))
# and need to be declared to avoid R CMD check NOTEs

utils::globalVariables(c(

  # Core parameters
  "K", "LAMBDA_", "LAMBDA_selected", "W", "W_", "X", "Y", "Y_",
  "lambda", "lambda_seq", "n_folds", "nSGD", "nRounds",
  "nMonte_Qglm", "nMonte_adversarial", "nFolds_glm",
  "penalty_type", "adversarial", "p_list", "pi_list",
  "adversarial_model_strategy", "include_stage_interactions",
  "neural_mcmc_control", "outcome_model_type",

  # Model objects
  "ModelList_object", "presaved_outcome_model", "save_outcome_model", "outcome_model_key",
  "vcov_OutcomeModel", "vcov_OutcomeModel_by_k", "vcov_OutcomeModel_general",
  "UsedRegularization",

  # Factor/treatment related
  "FactorsMat_numeric_0Indexed", "factor_levels", "treatment_combs",
  "main_indices_i0", "inter_indices_i0", "main_info", "interaction_info",
  "n_main_params", "n_target",

  # Optimization parameters
  "pi_ast", "pi_dag", "pi_init_list", "pi_init_vec",
  "pi_ast_br_hat", "pi_dag_br",
  "pi_forGrad_minus_mat_list", "pi_forGrad_minusk_mat_vec_comp",
  "pi_forGrad_minuskkt_mat_vec_comp",
  "utility_ast", "utility_dag",

  # JAX/TF compiled objects
  "EST_COEFFICIENTS_tf_ast_jnp", "EST_INTERCEPT_tf_ast_jnp",
  "EST_COEFFICIENTS_tf_general", "EST_INTERCEPT_tf_general",
  "REGRESSION_PARAMS_jax_dag0_jnp",
  "grad_getLoss_jax_unnormalized", "getLoss_tf",
  "getProbRatio_jax", "getProbRatio_tf",
  "getQStar_diff_SingleGroup", "getQStar_diff_MultiGroup",
  "jit_apply_updates_ast", "jit_apply_updates_dag",
  "jit_update_ast", "jit_update_dag",
  "compile_fxn", "gather_fxn", "jit_apply_updates_ast",

  # Function references
  "getClassProb", "getPiList", "getQ_fxn", "GetInvLR", "GetUpdatedParameters",

  # Q-function related
  "QFXN", "Q_DISAGGREGATE", "Qhat", "Qhat_tf",
  "returnWeightsFxn", "log_PrW",

  # Data structures
  "a_structure", "a_structure_leftoutLdminus1",
  "split1_indices", "split2_indices", "splitIndices", "split_vec_full",
  "holdout_indicator", "holdBasis_traintv",
  "trainIndicator_pool", "GroupsPool", "GroupCounter", "indi_",

  # Adversarial mode
  "competing_group_competition_variable_candidate_",
  "competing_group_variable_candidate_", "competing_group_variable_respondent_",
  "varcov_cluster_variable", "varcov_cluster_variable_",
  "pair_id_", "respondent_id", "respondent_task_id",
  "profile_order", "profile_order_",

  # M-estimation
  "ClassProbProj", "ClassProbProjCoefs", "ClassProbProjCoefs_se",
  "ClassProbsXobs", "VarCov_ProbClust", "X_factorized", "X_factorized_complete",
  "projList", "batch_size", "cycle_width",

  # Regularization
  "DD_L2Pen", "regularization_adjust_hash", "CLUSTER_EPSILON",
  "EXPERIMENTAL_SCALING_FACTOR",

  # Misc
  "AVSList", "MNtemp", "Round_",
  "p_vec_sum_prime_use", "p_vec_use",
  "piSEtype", "confLevel", "sg_method",
  "glm_family", "glm_outcome_transform",
  "my_mean_full", "w_orig", "warm_start",
  "p_vec_tf", "use_stage_models", "use_optax", "tape", "tfConst",
  "evaluation_environment",

  # Primary push-forward controls
  "primary_pushforward", "primary_strength",
  "primary_n_entrants", "primary_n_field",

  # Generated code functions
  "generate_ExactSolExplicit", "generate_ExactSolImplicit",

  # TensorFlow/reticulate operators
  "%as%"
))
