getQStar_single <- function(pi_star_ast, pi_star_dag,
                            EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                            EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){
  if (outcome_model_type == "neural" &&
      exists("neural_model_info_ast_jnp", inherits = TRUE)) {
    return(neural_getQStar_single(pi_star_ast, EST_COEFFICIENTS_tf_ast))
  }
  # note: here, dag ignored 
  pi_star_ast <- strenv$jnp$reshape(pi_star_ast, list(-1L, 1L))
  pi_star_dag <- strenv$jnp$reshape(pi_star_dag, list(-1L, 1L))

  # coef info
  main_coef <- strenv$jnp$reshape(
    strenv$jnp$take(EST_COEFFICIENTS_tf_ast,
                    indices = main_indices_i0,
                    axis = 0L),
    list(-1L, 1L)
  )
  main_term <- strenv$jnp$reshape(
    strenv$jnp$sum(strenv$jnp$reshape(main_coef, list(-1L)) *
                     strenv$jnp$reshape(pi_star_ast, list(-1L))),
    list(1L, 1L)
  )
  if(!is.null(inter_indices_i0)){ 
    inter_coef <- strenv$jnp$reshape(
      strenv$jnp$take(EST_COEFFICIENTS_tf_ast,
                      indices = inter_indices_i0,
                      axis = 0L),
      list(-1L, 1L)
    )
  
    # get interaction info
    pi_dp <- strenv$jnp$take(pi_star_ast, 
                             n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_dpp <- strenv$jnp$take(pi_star_ast, 
                              n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
    inter_term <- strenv$jnp$reshape(
      strenv$jnp$sum(strenv$jnp$reshape(inter_coef, list(-1L)) *
                       strenv$jnp$reshape(pi_dp * pi_dpp, list(-1L))),
      list(1L, 1L)
    )
    Qhat <-  glm_outcome_transform( EST_INTERCEPT_tf_ast +
                                      main_term +
                                      inter_term )
  }

  if(is.null(inter_indices_i0)){ 
    Qhat <-  glm_outcome_transform( EST_INTERCEPT_tf_ast +
                          main_term )
  }
  
  if( length(Qhat$shape) == 3L ) {
    Qhat <- Qhat$squeeze(2L)
  }
  return( strenv$jnp$concatenate( list(Qhat, 
                                       Qhat, 
                                       Qhat), 0L)  ) # to keep sizes consistent with diff case 
}

getQStar_diff_BASE <- function(pi_star_ast, pi_star_dag,
                               EST_INTERCEPT_tf_ast, EST_COEFFICIENTS_tf_ast,
                               EST_INTERCEPT_tf_dag, EST_COEFFICIENTS_tf_dag){

  pi_star_ast <- strenv$jnp$reshape(pi_star_ast, list(-1L, 1L))
  pi_star_dag <- strenv$jnp$reshape(pi_star_dag, list(-1L, 1L))

  if (outcome_model_type == "neural" &&
      exists("neural_model_info_ast_jnp", inherits = TRUE)) {
    return(neural_getQStar_diff_BASE(
      pi_star_ast, pi_star_dag,
      EST_COEFFICIENTS_tf_ast, EST_COEFFICIENTS_tf_dag
    ))
  }

  # coef
  main_coef_ast <- strenv$jnp$reshape(
    strenv$jnp$take(EST_COEFFICIENTS_tf_ast,
                    indices = main_indices_i0,
                    axis = 0L),
    list(-1L, 1L)
  )
  DELTA_pi_star <- strenv$jnp$reshape((pi_star_ast - pi_star_dag), list(-1L))
  main_term_ast <- strenv$jnp$reshape(
    strenv$jnp$sum(strenv$jnp$reshape(main_coef_ast, list(-1L)) * DELTA_pi_star),
    list(1L, 1L)
  )
  
  if(!is.null(inter_indices_i0)){ 
    inter_coef_ast <- strenv$jnp$reshape(
      strenv$jnp$take(EST_COEFFICIENTS_tf_ast,
                      indices = inter_indices_i0,
                      axis = 0L),
      list(-1L, 1L)
    )
  
    # get interaction info
    pi_ast_dp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_ast_dpp <- strenv$jnp$take(pi_star_ast, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
  
    pi_dag_dp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dl_index_adj)-1L), axis=0L)
    pi_dag_dpp <- strenv$jnp$take(pi_star_dag, n2int( ai(interaction_info$dplp_index_adj)-1L), axis=0L)
    DELTA_pi_star_prod <- strenv$jnp$reshape(pi_ast_dp * pi_ast_dpp - pi_dag_dp * pi_dag_dpp,
                                             list(-1L))
    inter_term_ast <- strenv$jnp$reshape(
      strenv$jnp$sum(strenv$jnp$reshape(inter_coef_ast, list(-1L)) * DELTA_pi_star_prod),
      list(1L, 1L)
    )
    
    Qhat_ast_among_ast <- glm_outcome_transform( 
              EST_INTERCEPT_tf_ast + 
              main_term_ast +
              inter_term_ast
            )
  }

  if(is.null(inter_indices_i0)){ 
    Qhat_ast_among_ast <- glm_outcome_transform( 
      EST_INTERCEPT_tf_ast +  main_term_ast )
  }

  if( !Q_DISAGGREGATE ){ Qhat_population <- Qhat_ast_among_dag <- Qhat_ast_among_ast }
  if( Q_DISAGGREGATE ){ # run if DisaggreateQ
    main_coef_dag <- strenv$jnp$reshape(
      strenv$jnp$take(EST_COEFFICIENTS_tf_dag,
                      indices = main_indices_i0,
                      axis = 0L),
      list(-1L, 1L)
    )
    main_term_dag <- strenv$jnp$reshape(
      strenv$jnp$sum(strenv$jnp$reshape(main_coef_dag, list(-1L)) * DELTA_pi_star),
      list(1L, 1L)
    )
    if(!is.null(inter_indices_i0)){ 
      inter_coef_dag <- strenv$jnp$reshape(
        strenv$jnp$take(EST_COEFFICIENTS_tf_dag,
                        indices = inter_indices_i0,
                        axis = 0L),
        list(-1L, 1L)
      )
      inter_term_dag <- strenv$jnp$reshape(
        strenv$jnp$sum(strenv$jnp$reshape(inter_coef_dag, list(-1L)) * DELTA_pi_star_prod),
        list(1L, 1L)
      )
      Qhat_ast_among_dag <- glm_outcome_transform( 
                EST_INTERCEPT_tf_dag + 
                main_term_dag +
                inter_term_dag )
    }
    if(is.null(inter_indices_i0)){ 
      Qhat_ast_among_dag <- glm_outcome_transform( 
        EST_INTERCEPT_tf_dag +  main_term_dag )
    }
  
    # Pr( Ast | Ast Voter) * Pr(Ast Voters) +  Pr( Ast | Dag Voter) * Pr(Dag Voters)
    Qhat_population <- Qhat_ast_among_ast * strenv$jnp$array(strenv$AstProp) +  
                                Qhat_ast_among_dag * strenv$jnp$array(strenv$DagProp)
  }
  return( strenv$jnp$concatenate( list(Qhat_population, 
                                       Qhat_ast_among_ast, 
                                       Qhat_ast_among_dag), 0L)  )
}

FullGetQStar_ <- function(a_i_ast,                                #1 
                          a_i_dag,                                #2 
                          INTERCEPT_ast_, COEFFICIENTS_ast_,      #3,4       
                          INTERCEPT_dag_, COEFFICIENTS_dag_,      #5,6 
                          INTERCEPT_ast0_, COEFFICIENTS_ast0_,    #7,8
                          INTERCEPT_dag0_, COEFFICIENTS_dag0_,    #9,10
                          P_VEC_FULL_ast_, P_VEC_FULL_dag_,       #11,12
                          SLATE_VEC_ast_, SLATE_VEC_dag_,         #13,14
                          LAMBDA_,                                #15
                          Q_SIGN,                                 #16 
                          SEED_IN_LOOP                            #17
){
  
  # Map logits -> simplex (respecting ParameterizationType)
  main_comp_mat_use <- strenv$main_comp_mat
  shadow_comp_mat_use <- strenv$shadow_comp_mat
  if (is.null(main_comp_mat_use)) {
    main_comp_mat_use <- strenv$OneTf_flat
  }
  if (is.null(shadow_comp_mat_use)) {
    shadow_comp_mat_use <- strenv$OneTf_flat
  }
  pi_star_full_i_ast <- strenv$getPrettyPi_diff( pi_star_i_ast<-strenv$a2Simplex_diff_use(a_i_ast), 
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 main_comp_mat_use,   
                                                 shadow_comp_mat_use  )
  pi_star_full_i_dag <- strenv$getPrettyPi_diff( pi_star_i_dag<-strenv$a2Simplex_diff_use(a_i_dag),
                                                 strenv$ParameterizationType,
                                                 strenv$d_locator_use,       
                                                 main_comp_mat_use,   
                                                 shadow_comp_mat_use  )
  
  # Average-case path
  if(!adversarial){
    average_case_eval <- evaluate_average_case_q(
      pi_star_ast = pi_star_i_ast,
      pi_star_dag = pi_star_i_dag,
      INTERCEPT_ast_ = INTERCEPT_ast_,
      COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
      INTERCEPT_dag_ = INTERCEPT_dag_,
      COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
      seed_in = SEED_IN_LOOP,
      phase = "objective",
      outcome_model_type = outcome_model_type,
      glm_family = glm_family,
      nMonte_Qglm = nMonte_Qglm,
      temperature = MNtemp,
      ParameterizationType = strenv$ParameterizationType,
      d_locator_use = strenv$jnp$array(strenv$d_locator_use),
      q_fxn = QFXN,
      single_party = !isTRUE(diff),
      force_reinforce = force_reinforce
    )
    q_max <- average_case_eval$q_max
    SEED_IN_LOOP <- average_case_eval$seed_next
    # In non-adversarial mode, we always optimize the "ast" player
    indicator_UseAst <- 1.0
  }
  
  # Adversarial path: institution-aware push-forward (four-quadrant mixture)
  if(adversarial){
    adversarial_eval <- evaluate_adversarial_q(
      pi_star_ast = pi_star_i_ast,
      pi_star_dag = pi_star_i_dag,
      a_i_ast, a_i_dag,
      INTERCEPT_ast_ = INTERCEPT_ast_,
      COEFFICIENTS_ast_ = COEFFICIENTS_ast_,
      INTERCEPT_dag_ = INTERCEPT_dag_,
      COEFFICIENTS_dag_ = COEFFICIENTS_dag_,
      INTERCEPT_ast0_ = INTERCEPT_ast0_,
      COEFFICIENTS_ast0_ = COEFFICIENTS_ast0_,
      INTERCEPT_dag0_ = INTERCEPT_dag0_,
      COEFFICIENTS_dag0_ = COEFFICIENTS_dag0_,
      P_VEC_FULL_ast_ = P_VEC_FULL_ast_,
      P_VEC_FULL_dag_ = P_VEC_FULL_dag_,
      SLATE_VEC_ast_ = SLATE_VEC_ast_,
      SLATE_VEC_dag_ = SLATE_VEC_dag_,
      LAMBDA_ = LAMBDA_,
      Q_SIGN = Q_SIGN,
      seed_in = SEED_IN_LOOP,
      phase = "objective",
      outcome_model_type = outcome_model_type,
      glm_family = glm_family,
      nMonte_Qglm = nMonte_Qglm,
      nMonte_adversarial = nMonte_adversarial,
      primary_pushforward = primary_pushforward,
      primary_n_entrants = primary_n_entrants,
      primary_n_field = primary_n_field,
      temperature = MNtemp,
      ParameterizationType = strenv$ParameterizationType,
      d_locator_use = strenv$jnp$array(strenv$d_locator_use)
    )
    q_max_ast <- adversarial_eval$q_ast
    q_max_dag <- adversarial_eval$q_dag
    
    # Choose which side we’re optimizing in this call
    indicator_UseAst <- (0.5 * ( 1. + Q_SIGN ))
    q_max <- adversarial_eval$q_max
    SEED_IN_LOOP <- adversarial_eval$seed_next
  }
  
  # ---- Regularization (unchanged), applied to the player being updated ----
  if(penalty_type %in% c("L1","L2")){
    PenFxn <- ifelse(penalty_type == "L1", 
                     yes = list(strenv$jnp$abs),
                     no = list(strenv$jnp$square))[[1]]
    var_pen_ast__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_ast - P_VEC_FULL_ast_ )  ))
    var_pen_dag__ <- LAMBDA_ * strenv$jnp$negative(strenv$jnp$sum(PenFxn( pi_star_full_i_dag - P_VEC_FULL_dag_ )  ))
  } else if(penalty_type == "LInfinity"){
    var_pen_ast__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
      list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_ast, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)) )})
    names(var_pen_ast__)<-NULL ; var_pen_ast__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum(strenv$jnp$stack(var_pen_ast__)))
    var_pen_dag__ <- tapply(1:length(split_vec_full), split_vec_full, function(zer){
      list( strenv$jnp$max(strenv$jnp$take(pi_star_full_i_dag, indices = strenv$jnp$array( ai(zer-1L)),axis = 0L)) )})
    names(var_pen_dag__)<-NULL ; var_pen_dag__ <- strenv$jnp$negative(LAMBDA_*strenv$jnp$sum(strenv$jnp$stack(var_pen_dag__)))
  } else {
    # "KL" default (with epsilon clipping to prevent log(0) = -Inf)
    eps <- 1e-8
    var_pen_ast__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_ast_ * (strenv$jnp$log(strenv$jnp$clip(P_VEC_FULL_ast_, eps, 1.0)) - strenv$jnp$log(strenv$jnp$clip(pi_star_full_i_ast, eps, 1.0)))))
    var_pen_dag__ <- LAMBDA_*strenv$jnp$negative(strenv$jnp$sum(P_VEC_FULL_dag_ * (strenv$jnp$log(strenv$jnp$clip(P_VEC_FULL_dag_, eps, 1.0)) - strenv$jnp$log(strenv$jnp$clip(pi_star_full_i_dag, eps, 1.0)))))
  }
  
  myMaximize <- 
    q_max + ( (indicator_UseAst * var_pen_ast__) 
       + (1.- indicator_UseAst) * var_pen_dag__ )
  
  return( myMaximize )
}
