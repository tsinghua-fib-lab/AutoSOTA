# Deprecated archive: retained for reference only and not sourced by the active package.
#' Estimate an Optimal (or Adversarial) Stochastic Intervention for Conjoint Analysis Using a One-Step M-estimation Approach
#'
#' @description
#' \code{strategize_onestep} implements a single-step (\dQuote{one-step}) approach to estimating a
#' target quantity of interest in high-dimensional conjoint (or factorial) experiments, such as
#' finding an \emph{optimal stochastic intervention} over factor levels. This method can incorporate
#' adversarial or non-adversarial settings, regularization, multi-stage structures (e.g., primaries
#' followed by a general election), and a variety of user-specified outcome models. It returns
#' estimated distributions of factor levels \emph{(e.g., candidate attributes)} that maximize or
#' minimize an outcome (e.g., vote share), including optional standard errors via M-estimation
#' or the delta method.
#'
#' @usage
#' strategize_onestep(
#'   W,
#'   Y,
#'   X = NULL,
#'   K = 1,
#'   warm_start = FALSE,
#'   automatic_scaling = TRUE,
#'   p_list = NULL,
#'   pi_list = NULL,
#'   pi_init_vec = NULL,
#'   constrain_ub = NULL,
#'   n_lambda = 10,
#'   penalty_type = "LogMaxProb",
#'   test_fraction = 0.5,
#'   log_PrW = NULL,
#'   learning_rate_max = 0.01,
#'   cycle_width  = 50,
#'   cycle_number = 4,
#'   nSGD = 500,
#'   nEpoch = NULL,
#'   X_factorized = NULL,
#'   momentum = 0.99,
#'   n_full_cycles = 1,
#'   optim_method = "tf",
#'   sg_method = NULL,
#'   forceSEs = FALSE,
#'   clip_at = 100000,
#'   adaptive_momentum = FALSE,
#'   known_norm_factor = NULL,
#'   split1_indices = NULL,
#'   split2_indices = NULL,
#'   use_hajek = TRUE,
#'   find_max = TRUE,
#'   quiet = TRUE,
#'   lambda_seq = NULL,
#'   lambda_coef = 0.0001,
#'   n_folds = 3,
#'   batch_size = 50,
#'   confLevel = 0.90,
#'   conda_env = NULL,
#'   conda_env_required = FALSE,
#'   hypothetical_n = NULL
#' )
#'
#' @param W A matrix or data frame of assigned factor levels in the conjoint. For forced-choice designs,
#'   each row generally represents a single profile or a single respondent-task-profile combination.
#'   Must have integer, factor, or character columns representing factor levels.
#' @param Y A numeric or binary outcome vector. Typically \code{Y} is \code{1} if a profile is chosen
#'   over its competitor, and \code{0} otherwise. If \code{find_max = FALSE}, the sign is flipped,
#'   effectively minimizing \code{Y}.
#' @param X Optional matrix or data frame of covariates (or respondent-level features). Can be used
#'   for multi-cluster modeling with \code{K>1}.
#' @param K An integer for the number of mixture components or latent clusters if multi-cluster
#'   modeling is desired. Defaults to \code{1} (no clusters).
#' @param warm_start Logical. If \code{TRUE}, attempts to re-initialize from previous solutions each
#'   time \code{lambda} or other hyperparameters change. Defaults to \code{FALSE}.
#' @param automatic_scaling Logical indicating whether to center or scale \code{X} and \code{Y} automatically.
#'   Defaults to \code{TRUE}.
#' @param p_list A list of baseline factor-level probabilities in the design or assignment mechanism
#'   (e.g., the original random assignment distribution). If \code{NULL}, the function may assume
#'   uniform or empirical distributions.
#' @param pi_list An optional list specifying a counterfactual distribution over factor
#'   levels. If provided, \code{strategize_onestep} directly computes and returns the performance
#'   or value under that distribution instead of estimating a new optimal distribution.
#' @param pi_init_vec A numeric vector for initializing the simplex-based representation of factor-level
#'   probabilities to be optimized. If \code{NULL}, a random initialization is used internally.
#' @param constrain_ub Optional numeric or vector of upper bounds on factor probabilities. If not
#'   \code{NULL}, can help to enforce constraints in optimization.
#' @param n_lambda Integer specifying the number of penalty values considered if cross-validation
#'   is performed. Defaults to 10.
#' @param penalty_type A character specifying the type of penalty for shifting probabilities (e.g.,
#'   \code{"LogMaxProb"}, \code{"L2"}, or \code{"KL"}). This is an additional penalization on top of
#'   \code{penalty_type} for the outcome model. Defaults to \code{"LogMaxProb"}.
#' @param test_fraction Fraction of samples used for holdout in cross-validation. Defaults to 0.5
#'   for a basic split. If \code{NULL}, no split is performed.
#' @param log_PrW Optional numeric vector of log probabilities for each row in \code{W}. If omitted,
#'   the function will compute \code{log_PrW} from \code{p_list} given the assumption of independent
#'   factor-level assignments.
#' @param learning_rate_max Base learning rate for gradient-based optimizers. Defaults to 0.01.
#' @param cycle_width Numeric controlling the frequency of restarts or adaptive learning-rate schedules.
#' @param cycle_number Number of cycles used in the learning-rate schedule.
#' @param nSGD Number of gradient-descent updates. If \code{nEpoch} is provided, that takes precedence.
#' @param nEpoch Number of epochs, each pass including \code{length(availableTrainIndices) / batch_size}
#'   mini-batches. If provided, overrides \code{nSGD}.
#' @param X_factorized An optional matrix or data frame representing factorized (dummy-coded) versions
#'   of \code{X} for advanced modeling. If \code{NULL}, the function may factorize internally.
#' @param momentum Numeric specifying momentum for stochastic gradient descent. Defaults to 0.99.
#' @param n_full_cycles If \code{>1}, repeats training cycles without restarts for a total
#'   number of gradient steps. Useful for stability checks.
#' @param optim_method A character specifying the optimization backend. Defaults to \code{"jax"} if available.
#' @param sg_method A character controlling the type of gradient updates (e.g., \code{"adanorm"}, \code{"wngrad"}).
#'   If \code{NULL}, a default method is chosen.
#' @param forceSEs Logical. If \code{TRUE}, attempts to compute standard errors by M-estimation or the
#'   delta method, even if no cross-validation is done.
#' @param clip_at A large numeric to clip gradient norms if they exceed \code{clip_at}.
#' @param adaptive_momentum Logical. If \code{TRUE}, momentum is adapted automatically as the optimization
#'   proceeds. Defaults to \code{FALSE}.
#' @param known_norm_factor An optional numeric to normalize reweighting for Hajek-based
#'   estimators. If \code{NULL}, it is inferred from the sum of weights.
#' @param split1_indices,split2_indices Optional vectors of indices partitioning the data for
#'   cross-validation or holdout. If \code{NULL}, a random partition is done internally.
#' @param use_hajek Logical. If \code{TRUE}, uses a Hajek-based reweighting in objective
#'   functions for computing the expected outcome under counterfactual probability shifts. Defaults
#'   to \code{TRUE}.
#' @param find_max Logical. If \code{TRUE}, maximizes \code{Y}; if \code{FALSE}, treats \code{Y} as
#'   negative of interest (e.g., adversity minimization).
#' @param quiet Logical controlling the verbosity of printed messages.
#' @param lambda_seq Optional numeric vector of penalty values for cross-validation. If \code{NULL},
#'   the function attempts a default sequence or single value.
#' @param lambda_coef Numeric constant controlling the magnitude of the penalty for the outcome
#'   model. Defaults to 0.0001.
#' @param n_folds Number of folds for cross-validation. Defaults to 3.
#' @param batch_size Positive integer specifying the size of mini-batches in each gradient
#'   iteration. Defaults to 50.
#' @param confLevel Numeric in \((0,1)\), specifying the confidence level for intervals around
#'   estimated probabilities or performance measures. Defaults to 0.90.
#' @param conda_env Optional name of a Conda environment with \pkg{jax}, \pkg{optax}, etc. If
#'   \code{NULL}, attempts a default environment.
#' @param conda_env_required Logical. If \code{TRUE}, errors if the environment \code{conda_env}
#'   cannot be found. Otherwise attempts to proceed gracefully.
#' @param hypothetical_n Optional numeric specifying an alternative \code{n} for certain
#'   variance calculations (e.g., hypothesized population size). If \code{NULL}, uses the observed
#'   sample size.
#'
#' @return A named \code{list} with components, often including:
#' \itemize{
#'   \item \code{pi_star_point}: The estimated optimal or learned distribution(s) over factor levels.
#'         If \code{K>1} or if adversarial competition is considered, may return multiple
#'         distributions (\code{k1}, \code{k2}, etc.).
#'   \item \code{Q_point}: The estimated performance measure under the learned distribution(s).
#'         For example, the average or adversarially optimized outcome. 
#'   \item \code{Q_se_mEst}: If available, standard errors via M-estimation or the delta method.
#'   \item \code{PiStar_lb}, \code{PiStar_ub}: Lower and upper confidence intervals for factor-level
#'         probabilities, if standard errors are computed.
#'   \item \code{CVInfo}: A data frame or list summarizing cross-validation performance for each
#'         candidate \code{lambda}.
#'   \item \code{ClassProbsXobs}, \code{VarCov_ProbClust}, \code{pi_init_next}, \code{optim_max_hajek_list}:
#'         Additional objects storing advanced details of the optimization or M-estimation procedure.
#'   \item \code{Output.Description}: Additional messages describing the run.
#' }
#'
#' @details
#' This function implements a \emph{one-step M-estimation} approach for directly estimating the
#' \dQuote{optimal} probability distributions over high-dimensional factors in conjoint or factorial
#' experiments. Rather than a multi-step procedure of (1) outcome modeling followed by (2) re-optimizing
#' factor distributions, the one-step approach can iteratively re-estimate distribution parameters
#' while simultaneously adjusting the outcome model. This allows regularization or advanced modeling
#' to be integrated into the \emph{same} optimization objective, potentially improving finite-sample
#' performance. Support for adversarial or multiple clusters is also available.
#'
#' By default, \code{strategize_onestep} attempts to find the distribution(s) \eqn{\boldsymbol{\pi}^\ast} that
#' maximizes the average outcome if \code{find_max = TRUE} (e.g., maximizing candidate choice share).
#' In adversarial contexts, each cluster or \dQuote{player} can simultaneously learn a best response.
#' The function is flexible enough to incorporate sub-populations or multiple stages (e.g., primaries
#' plus general elections).
#'
#' If a user-supplied \code{pi_list} is given, the function directly computes \eqn{Q(\boldsymbol{\pi})}
#' for that distribution instead of estimating. This is useful for evaluating the performance of a
#' known or hypothesized distribution (e.g., \dQuote{status quo}).
#'
#' Most users do not need to call \code{strategize_onestep} directly, as this is a lower-level
#' routine. The \code{\link{strategize}} or \code{\link{cv_strategize}} functions may suffice
#' in many typical workflows.
#'
#' @note
#' Advanced arguments like \code{X_factorized}, \code{conda_env}, \code{optim_method}, or specifying
#' \code{adaptive_momentum} are only needed for specialized or larger-scale (GPU-based) computations.
#'
#' @references
#' - Goplerud, M. & Titiunik, R. (2022). \emph{Analysis of High-Dimensional Factorial Experiments:
#'   Estimation of Interactive and Non-Interactive Effects.} ArXiv preprint.
#' - Egami, N. & Imai, K. (2019). \emph{Causal Interaction in Factorial Experiments: Application
#'   to Conjoint Analysis.} Journal of the American Statistical Association, 114(526), 529–540.
#' - Hainmueller, J., Hopkins, D. J., & Yamamoto, T. (2014). \emph{Causal Inference in Conjoint
#'   Analysis: Understanding Multidimensional Choices via Stated Preference Experiments.}
#'   Political Analysis, 22(1), 1–30.
#' - (Paper Reference) A forthcoming or accompanying manuscript describing in detail the methods for
#'   \emph{optimal} or \emph{adversarial} stochastic interventions in conjoint settings.
#'
#' @seealso
#' \code{\link{strategize}} for an approach that first fits an outcome model and then re-optimizes
#' factor-level probabilities.
#' \code{\link{cv_strategize}} for cross-validation across candidate values of \code{lambda}.
#'
#' @examples
#' \dontrun{
#' # ================================================
#' # One-step M-estimation approach
#' # ================================================
#' # This approach simultaneously estimates the outcome model and
#' # optimal distribution, rather than the two-step approach
#'
#' set.seed(123)
#' n <- 400
#'
#' # Generate factor matrix
#' W <- data.frame(
#'   Gender = sample(c("Male", "Female"), n, replace = TRUE),
#'   Age = sample(c("Young", "Middle", "Old"), n, replace = TRUE),
#'   Party = sample(c("Dem", "Rep"), n, replace = TRUE)
#' )
#'
#' # Simulate outcome (Female and Young preferred)
#' latent <- 0.3 * (W$Gender == "Female") + 0.2 * (W$Age == "Young")
#' Y <- rbinom(n, 1, plogis(latent))
#'
#' # Baseline probabilities (uniform)
#' p_list <- list(
#'   Gender = c(Male = 0.5, Female = 0.5),
#'   Age = c(Young = 1/3, Middle = 1/3, Old = 1/3),
#'   Party = c(Dem = 0.5, Rep = 0.5)
#' )
#'
#' # Optional respondent covariates
#' X <- matrix(rnorm(n * 2), n, 2)
#' colnames(X) <- c("Income", "Education")
#'
#' # Run one-step estimation
#' result <- strategize_onestep(
#'   W = W,
#'   Y = Y,
#'   X = X,
#'   p_list = p_list,
#'   nSGD = 100,
#'   penalty_type = "LogMaxProb",
#'   lambda_seq = c(0.01, 0.1),
#'   test_fraction = 0.3,
#'   quiet = TRUE
#' )
#'
#' # View optimal distribution
#' print(result$pi_star_point)
#'
#' # View expected outcome under optimal strategy
#' print(result$Q_point)
#' }
#'
#' @export

strategize_onestep <- function(
                              W,
                              Y,
                              X = NULL,
                              K = 1,
                              warm_start = FALSE,
                              automatic_scaling = TRUE,
                              p_list = NULL,
                              pi_list = NULL,
                              pi_init_vec = NULL,
                              constrain_ub = NULL,
                              n_lambda = 10,
                              penalty_type = "LogMaxProb",
                              test_fraction = 0.5,
                              log_PrW = NULL,
                              learning_rate_max = 0.01,
                              cycle_width  = 50,
                              cycle_number = 4,
                              nSGD = 500,
                              nEpoch = NULL,
                              X_factorized = NULL,
                              momentum = 0.99,
                              n_full_cycles = 1,
                              optim_method = "tf",
                              sg_method = NULL,
                              forceSEs = FALSE,
                              clip_at = 100000,
                              adaptive_momentum = FALSE,
                              known_norm_factor = NULL,
                              split1_indices=NULL,
                              split2_indices=NULL,
                              use_hajek = TRUE,
                              find_max = TRUE, quiet = TRUE,
                              lambda_seq = NULL,
                              lambda_coef = 0.0001,
                              n_folds = 3, 
                              batch_size = 50,
                              confLevel = 0.90,
                              conda_env = NULL,
                              conda_env_required = FALSE,
                              hypothetical_n=NULL){
  # initialize environment
  if(!"jnp" %in% ls(envir = strenv)) {
    initialize_jax(conda_env = conda_env, conda_env_required = conda_env_required) 
  }

  # define evaluation environment
  evaluation_environment <- environment()

  # initial processing
  {
  if(is.null(X)){ X <- cbind(rnorm(length(Y)),rnorm(length(Y))) }
  if(automatic_scaling == T){  X <- scale ( X  )  }

  useHajek <- T
  if(any(unlist(lapply(p_list,class)) == "table")){
    for(ij in 1:length(p_list)){
      n_ <- names(p_list[[ij]] )
      p_list[[ij]] <- as.vector(p_list[[ij]])
      names(p_list[[ij]]) <- n_
    } }
  penaltyProbList_unlisted <- unlist(p_list)
  if(find_max == F){ Y <- -1 * Y }

  # SCALE Y
  mean_Y <- 0; sd_Y <- 1
  if(automatic_scaling == T){
    mean_Y <- mean(Y); sd_Y <- sd(Y)
    Y <-   (Y -  mean_Y) / sd_Y
  }

  enc <- cs_prepare_W_encoding(
    W = W,
    p_list = p_list,
    unknown = "na",
    align = "none"
  )
  FactorsMat_numeric <- enc$W_idx
  }

  ### case 1 - the new multinomial probabilities ARE specified
  if(!is.null(pi_list)){
    Qhat_all <- computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                       Yobs_internal = Y,
                                       log_pr_w_internal = NULL,
                                       knownNormalizationFactor = known_norm_factor,
                                       assignmentProbList_internal = p_list,
                                       hypotheticalProbList_internal = pi_list,
                                       hajek = useHajek)
    SE_Q_all <- computeQse_conjoint(FactorsMat = FactorsMat_numeric,
                                Yobs = Y,
                                # hypotheticalN = NULL,
                                # log_PrW = NULL,
                                # log_treatment_combs = NULL,
                                hajek = useHajek,
                                knownNormalizationFactor = known_norm_factor,
                                assignmentProbList = p_list,
                                returnLog = F,
                                pi_list = pi_list)
    Q_interval_all <- c(Qhat_all$Qest - abs(qnorm((1-confLevel)/2))*SE_Q_all,
                        Qhat_all$Qest + abs(qnorm((1-confLevel)/2))*SE_Q_all)
    return(    list("Q_point_all" = RescaleFxn(Qhat_all$Qest, estMean = mean_Y, estSD = sd_Y),
                    "Q_se_all" = RescaleFxn(SE_Q_all, estMean = mean_Y, estSD = sd_Y,center=F),
                    "Q_interval_all" = RescaleFxn(Q_interval_all, estMean = mean_Y, estSD = sd_Y),
                    "Q_wts_all" = Qhat_all$Q_wts,
                    "Q_wts_raw_sum_all" = Qhat_all$Q_wts_raw_sum,
                    "log_pr_w_new_all"=Qhat_all$log_pr_w_new,
                    "log_pr_w_all"=Qhat_all$log_PrW) )
  }#end !is.null(pi_list)

  #### case 2 - the new multinomial probabilities ARE NOT specified
  if(is.null(pi_list)){

    # setup for m estimation
    forceHajek <- T
    zStar <- abs(qnorm((1-confLevel)/2))
    varHat <- mean( (Y - (muHat <- mean(Y)) )^2   )
    n_target <- ifelse(is.null(hypothetical_n), yes = length(  split1_indices  ), no = hypothetical_n)

    # define number of treatment combinations
    treatment_combs <- exp(log_treatment_combs  <- sum(log(sapply(1:ncol(W),function(ze){ length(p_list[[ze]]) }) )))

    # initialize quantities
        marginalProb_m <- seList <- seList_automatic <- m_se_Q <- seList_manual <- lowerList <- upperList <- NULL
        ClassProbProjCoefs <- ClassProbProjCoefs_se <- ClassProbsXobs <- VarCov_ProbClust <- NULL
        PrXd_vec <- PrXdGivenClust_se <- PrXdGivenClust_mat <- NULL
    if(is.null(split1_indices)){
      split_ <- rep(1,times=length(Y))
      if(is.null(test_fraction)){ test_fraction <- 0.5 }
      split_[sample(1:length(split_), round(length(Y)*test_fraction))] <- 2
      split1_indices = which(split_ == 1); split2_indices = which(split_ == 2)
      if(length(lambda_seq) == 1){
        warning("NO SAMPLE SPLITTING, AS LAMBDA IS FIXED")
        split1_indices <- split2_indices <- 1:length(Y)
      }
    }
    print(c(length(split1_indices),length(split2_indices)))

    # execute splits (drop=FALSE preserves matrix structure for single-column W)
    FactorsMat1 <- W[split1_indices, , drop=FALSE]; FactorsMat1_numeric <- FactorsMat_numeric[split1_indices, , drop=FALSE]
    FactorsMat2 <- W[split2_indices, , drop=FALSE]; FactorsMat2_numeric <- FactorsMat_numeric[split2_indices, , drop=FALSE]

    Yobs_split1 <- Y[split1_indices]
    log_pr_w_split1 <- log_PrW[split1_indices]
    sigma2_hat_split1 <- var( Yobs_split1)

    Yobs_split2 <- Y[split2_indices]
    log_pr_w_split2 <- log_PrW[split2_indices]
    sigma2_hat_split2 <- var( Yobs_split2)

    # obtain pr(w) so we don't need to recompute it at every step
    if( is.null(log_PrW)  ){
      log_PrW    <-   as.vector(computeQ_conjoint_internal(FactorsMat_internal = FactorsMat_numeric,
                                                            Yobs_internal=Y,
                                                            hypotheticalProbList_internal = p_list,
                                                            assignmentProbList_internal = p_list,
                                                            hajek = useHajek)$log_PrW)
    }

    # INITIALIZE M ESTIMATION
    message("Initialize M-Estimation...")
    initialMtext <- paste(deparse(initialize_m),collapse="\n")
    initialMtext <- gsub(initialMtext,pattern="function \\(\\)",replacement="")
    eval(parse( text = initialMtext ),envir = evaluation_environment)

    # get initial pi values
    message("Initialize pi values...")
    {
      if(is.null(pi_init_vec)){
        TARGET_EPSILON_PI <- 0.01  #/ exp( length( p_list ) )
        CLUSTER_EPSILON <- 0.1

        pi_init_type <- "runif"
        pi_init_list = replicate(n_folds+1,replicate(K,lapply(p_list,function(ze){
          systemit_init <- tmp <- ((c(rev(c(compositions::alr(t(rev(ze))))))))
          p_val <- prop.table(exp(ze))
          bounds_seq <- seq(0.001,0.25,length.out=100)
          epsilon_penalty_seq <- sapply(bounds_seq,function(b_){
            max( replicate(100,{
              {
                if(pi_init_type == "runif"){ log_pi <- tmp + runif(length(tmp),min = -b_,  max = b_)}
                if(pi_init_type == "rnorm"){ log_pi <- tmp + rnorm(length(tmp),mean = 0, sd = b_) }
                log_pi <- c(0,log_pi)
                pi_val <- prop.table(exp(log_pi))
              }
              #epsilon_penalty <- max(abs(pi_val  - p_val))
              epsilon_penalty <- max( (pi_val  - p_val) )
              }))})
          b_ <- bounds_seq[ which.min(abs(epsilon_penalty_seq - TARGET_EPSILON_PI)) ]
          if(pi_init_type == "rnorm"){ return_value <- as.matrix(tmp + rnorm(length(tmp), mean = 0, sd = b_))}
          if(pi_init_type == "runif"){ return_value <- as.matrix(tmp + runif(length(tmp), min = -b_, max = b_))}
          attr(return_value,"random_") <- rep(b_,length(return_value))
          attr(return_value,"sys_") <- systemit_init
          return(return_value)
        }), simplify = FALSE), simplify = FALSE)
        # Handle dimension-safe extraction (replicate simplifies differently when K=1)
        pi_init_vec <- unlist(pi_init_list[[1]][[1]])
      }
      names(pi_init_vec) <- NULL
    }

    # get mapped transformations
    {
      probsIndex_mapped <- unlist(p_list)
      probsIndex_mapped[] <- 1:length( probsIndex_mapped )
      probsIndex_mapped <- vec2list_noTransform(probsIndex_mapped)
      FactorsMat1_mapped <- sapply(1:ncol(FactorsMat1_numeric),function(ze){
        probsIndex_mapped[[ze]][ FactorsMat1_numeric[,ze] ]  })
      row.names(FactorsMat1_numeric) <- colnames(FactorsMat1_numeric) <- row.names(FactorsMat1_mapped) <- colnames(FactorsMat1_mapped) <- NULL
    }

    # main cross-validation routine
    optim_max_hajek_list <- sapply(1:(length(lambda_seq)+1),function(er){
      list(matrix(NA,nrow = length( pi_init_vec), ncol = n_folds)) })
    if ( length(lambda_seq) == 1 ){
      LAMBDA_selected <- LAMBDA_ <- lambda_seq;

      FactorsMat_numeric_IN <- FactorsMat1_numeric <- FactorsMat2_numeric <- FactorsMat_numeric
      Yobs_IN <- Yobs_split1 <- Yobs_split2 <- Y
      log_pr_w_IN <- log_pr_w_split1 <- log_pr_w_split2 <- log_PrW
    }
    lambda_seq <- sort(lambda_seq, decreasing = T)

    # optimization
    {
        # helper functions
        if(is.null(X)){X<-matrix(rnorm(length(Y)*max(2,K)),nrow=length(Y),ncol=max(2,K))}
        enc <- cs_prepare_W_encoding(
          W = W,
          p_list = p_list,
          unknown = "na",
          align = "none"
        )
        FactorsMat_numeric <- enc$W_idx
        FactorsMat_numeric_0Indexed <- FactorsMat_numeric - 1L

        performance_matrix_out <- performance_matrix_in <- matrix(NA, ncol = length(lambda_seq), nrow = n_folds)
        holdBasis_traintv <- sample(1:length(split1_indices) %% n_folds+1)
        REGULARIZATION_LAMBDA_SEQ_ORIG <- lambda_seq

        if(length(lambda_seq) > 1){trainIndicator_pool <- c(1,0); if(n_folds == 1){stop("ERROR: SET n_folds > 1!")}}
        if(length(lambda_seq) == 1){warning(sprintf("NO CV SELCTION OF LAMBDA, FORCING LAMBDA = %.5f|",lambda_seq)); trainIndicator_pool <- 0}

        # Build Model
        message("Building...")
        buildText <- paste(deparse(ml_build),collapse="\n")
        buildText <- gsub(buildText,pattern="function \\(\\)",replacement="")
        eval(parse( text = buildText ),envir = evaluation_environment)

        # Train Model + Perform CV
        message("Training...")
        trainText <- paste(deparse(ml_train),collapse="\n")
        trainText <- gsub(trainText,pattern="function \\(\\)",replacement="")
        eval(parse( text = trainText ),envir = evaluation_environment)

        # obtain the pi's
        pi_list <- getPiList( ModelList_object[[1]] );names(pi_list) <- paste("k",1:K,sep="")
        FinalProbList <- pi_list
        optim_max_hajek_full <- na.omit(  unlist(getPiList(ModelList_object[[1]], simplex=F)) )

        ClassProbs <- NULL; if(K > 1){ ClassProbs <- as.array(  getClassProb(X) ) }
        performance_mat <- as.data.frame(
                          cbind("lambda" = REGULARIZATION_LAMBDA_SEQ_ORIG,
                                "Qhat" = colMeans(performance_matrix_out),
                                "Qse" = apply(performance_matrix_out,2,getSE)
                                #"Q_in" = colMeans(performance_matrix_in), "Qse_in" = apply(performance_matrix_in,2,getSE))
                                ))
        LAMBDA_ <- LAMBDA <- LAMBDA_selected
    }

    optim_max_hajek_split1 <- optim_max_hajek_full
    if(length(REGULARIZATION_LAMBDA_SEQ_ORIG)>1){
    qStar <- 1
    par(mar=c(5,5,5,1))
    plot( log(performance_mat$lambda), main = sprintf("%s Fold CV",n_folds),cex.main = 2,
          performance_mat$Q_in,xlab = "log(lambda)", ylab="Value", col="gray",type = 'b',
          ylim = summary(c(performance_mat$Q_in - qStar*performance_mat$Qse_in,
                           performance_mat$Q_in + qStar*performance_mat$Qse_in,
                           performance_mat$Q_out - qStar*performance_mat$Qse_out,
                           performance_mat$Q_out + qStar*performance_mat$Qse_out
                             ))[c(1,6)])
    abline(h=mean(Y),lty = 2, col= "gray",lwd=2)
    for(iaj in 1:nrow(performance_mat)){
      points( c(log(performance_mat$lambda[iaj]),log(performance_mat$lambda[iaj])),
            c(performance_mat$Q_in[iaj] -qStar*performance_mat$Qse_in[iaj],
              performance_mat$Q_in[iaj] +qStar*performance_mat$Qse_in[iaj]), type = "l", lwd = 2,col = "gray" )
    }
    points( log(performance_mat$lambda), performance_mat$Q_out,type = "b",pch = "O")
    for(iaj in 1:nrow(performance_mat)){
      points( c(log(performance_mat$lambda[iaj]),log(performance_mat$lambda[iaj])),
              c(performance_mat$Q_out[iaj] -qStar*performance_mat$Qse_out[iaj],
                performance_mat$Q_out[iaj] +qStar*performance_mat$Qse_out[iaj]),type = "l", lwd = 2,col = "black" )
    }
    abline(v=log(  performance_mat$lambda[which.max(performance_mat$LB_empirical_out)]),lty=2,lwd=0.5 )
    }

    # save and store results
	    {
	      # save results - split 1
	      hypotheticalProbList_full <- hypotheticalProbList_split1 <- pi_list

	      # Get values from tf
	      Qhat_tf <- strenv$jnp$array( getQ_fxn( ModelList_object,  split2_indices  ), strenv$jnp$float32)
	      Qhat <- strenv$np$array( Qhat_tf )
	      seList <- lowerList <- upperList <- NULL
	      m_se_Q <- NULL
	      Qhat_split1 <- Qhat_all <- Q_interval_split2 <- Q_interval_split1 <- Q_se_exact <- NULL
	      Qhat_split2<-list();Qhat_split2$Qest <- strenv$np$array(Qhat_tf)
	    }

	    # get SEs
	    if (isTRUE(forceSEs)) {
	      if (!exists("piSEtype", inherits = TRUE) || length(piSEtype) < 1L || is.null(piSEtype)) {
	        piSEtype <- "automatic"
	      }
	      seText <- paste(deparse(get_se),collapse="\n")
	      seText <- gsub(seText,pattern="function \\(\\)",replacement="")
	      eval(parse( text = seText ),evaluation_environment)
	    }

    #plot(unlist(seList_manual),unlist(seList_automatic));abline(a=0,b=1)
    #plot(VarCov_n_automatic[,1],VarCov_n_manual[,1]);abline(a=0,b=1)
    #plot(VarCov_n_automatic[,5],VarCov_n_manual[,5]);abline(a=0,b=1)
    #image(cor(VarCov_n_automatic,VarCov_n_manual))

    return( list("pi_star_point"=hypotheticalProbList_full,
               "PiStar_se" = seList,
               "ClassProbProjCoefs" = ClassProbProjCoefs,
               "ClassProbProjCoefs_se" = ClassProbProjCoefs_se,
               "PiStar_lb"=lowerList,
               "PiStar_ub"=upperList,
               "Q_point"= RescaleFxn(Qhat_split1$Qest, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_point_mEst"= RescaleFxn(Qhat_split2$Qest, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_se_exact" = RescaleFxn(Q_se_exact, estMean = mean_Y, estSD = sd_Y,center=F),
               "Q_se_mEst" = RescaleFxn(m_se_Q, estMean = mean_Y, estSD = sd_Y,center=F),
               "Q_interval_split1" = RescaleFxn(Q_interval_split1, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_interval_split2" = RescaleFxn(Q_interval_split2, estMean = mean_Y, estSD = sd_Y,center=T),
               "Q_wts" = Qhat_split2$Q_wts,
               "Q_point_split2"=Qhat_split2$Qest,
               "Q_wts_raw_sum" = Qhat_split2$Q_wts_raw_sum,
               "Q_wts_raw_sum_split2" = Qhat_split2$Q_wts_raw_sum,
               "Q_wts_split2" = Qhat_split2$Q_wts,
               "split1_indices" = split1_indices,
               "split2_indices" = split2_indices,
               "ClassProbsXobs" = ClassProbsXobs,
               "VarCov_ProbClust" = VarCov_ProbClust,
               "MarginalClusterProbEvolution"=marginalProb_m,
               "pi_init_next" = optim_max_hajek_full,
               "CVInfo" = performance_mat,
               "optim_max_hajek_list" = optim_max_hajek_list,
               "PrXd_vec" = PrXd_vec,
               "X_factorized_complete"=ifelse(("X_factorized_complete" %in% ls()),yes=list(X_factorized_complete),no=list(NULL)),
               "PrXdGivenClust" = PrXdGivenClust_mat,
               "estimationType" = "OneStep",
               "PrXdGivenClust_se" = PrXdGivenClust_se,
               "Output.Description"=c("")) )
  }
}
