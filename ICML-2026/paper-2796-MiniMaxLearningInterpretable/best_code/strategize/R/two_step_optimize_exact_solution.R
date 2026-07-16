generate_ExactSol <- function(){

  JaxVectorScatterUpdate <- compile_fxn(function(tensor, updates, indices){
    # replaces  tf.tensor_scatter_nd_update
    tensor <- tensor$at[indices]$set(updates)
  })

  # ParameterizationType == "Implicit" solution
  if(strenv$ParameterizationType == "Implicit"){
  Neg4lambda_diag <-  strenv$jnp$array( rep(-4 * lambda,times=n_main_params))
  Neg4lambda_update <-  strenv$jnp$array(as.matrix(-4*lambda),strenv$dtj)
  Neg2lambda_update <-  strenv$jnp$array(as.matrix(-2*lambda),strenv$dtj)
  Const_4_lambda_pl <-  strenv$jnp$array(as.matrix( 4*lambda*p_vec_use),strenv$dtj)
  Const_2_lambda_plprime <-  strenv$jnp$array(as.matrix( 2*lambda*p_vec_sum_prime_use), dtype = strenv$dtj)

  generate_ExactSolImplicit <- function(EST_COEFFICIENTS_tf){
    main_coef <-  strenv$jnp$take(EST_COEFFICIENTS_tf, indices = main_indices_i0, axis=0L)
    inter_coef <-  strenv$jnp$take(EST_COEFFICIENTS_tf, indices = inter_indices_i0, axis=0L)
    b_vec <-  strenv$jnp$subtract(
                           strenv$jnp$subtract( strenv$jnp$negative( main_coef ), Const_4_lambda_pl),
                          Const_2_lambda_plprime
                          )

    C_mat <- sapply(1:n_main_params,function(main_comp){
      # initialize to 0
      row_ <-  strenv$jnp$zeros(list(n_main_params))

      # update diagonal component
      row_ <- JaxVectorScatterUpdate(row_,
                                     updates = Neg4lambda_update,
                                     indices = n2int(as.matrix(ai(main_comp-1L))))

      # update off-diagonal component (same d)
      same_d_diff_l <- n2int(as.matrix(ai(setdiff(which(main_info$d_adj == main_info[main_comp,]$d_adj),main_comp)-1L)))
      SameDDiffL_update <-  strenv$jnp$multiply(
                                 strenv$jnp$multiply( strenv$jnp$negative(2),lambda),
                                 strenv$jnp$ones(list(same_d_diff_l$size,1L)))
      row_ <- JaxVectorScatterUpdate(row_,
                                     updates = SameDDiffL_update,
                                     indices = same_d_diff_l)

      # update off-diagonal component (different d)
      #interaction_info_red <- interaction_info[interaction_info$dl_index %in% main_comp | interaction_info$dplp_index %in% main_comp,]
      interaction_info_red <- interaction_info[
        (ind1<-(interaction_info$d_adj %in% main_info[main_comp,]$d_adj &
                  interaction_info$l %in% main_info[main_comp,]$l)) |
          (ind2<-(interaction_info$dp_adj %in% main_info[main_comp,]$d_adj &
                    interaction_info$lp %in% main_info[main_comp,]$l ) ),]
      id_d <- apply(interaction_info_red[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
      id_dp <- apply(interaction_info_red[,c("dp_adj","lp")],1,function(zer){paste(zer,collapse="_")})
      id_ <- ifelse(!(interaction_info_red$d_adj %in% main_info[main_comp,]$d_adj),
                    yes = id_d, no = id_dp)
      id_main <- apply(main_info[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
      which_inter <- which(ind1|ind2)
      inter_into_main <- sapply(id_,function(zer){which(id_main %in% zer)})

      #interaction_info_ordering <- interaction_info_red$dl_index * (ind1) + interaction_info_red$dplp_index * (ind2)
      if(nrow(interaction_info_red)>0){
        inter_coef_ <-  strenv$jnp$take(inter_coef,
                                indices = n2int(ai(which_inter-1L)),
                                axis = 0L)
        #if(length(which_inter) == 1){ inter_coef_ <-  strenv$jnp$expand_dims(inter_coef_,0L) }
        row_ <- JaxVectorScatterUpdate(row_,
                                       updates = inter_coef_,
                                       indices = n2int(as.matrix(ai(inter_into_main-1L))))
      }
      return(  strenv$jnp$expand_dims(row_,1L) ) # check - should these be called rows?
    })
    C_mat <-  strenv$jnp$concatenate(C_mat,1L)

    return(  pi_star <-  strenv$jnp$matmul( strenv$jnp$linalg$inv(C_mat), b_vec)  )
  }
  }

  # ParameterizationType == "Full" solution
  if(strenv$ParameterizationType == "Full"){
    Const_2_lambda_pl <-  strenv$jnp$array(as.matrix( 2*lambda*p_vec_use),strenv$dtj)
    Const_2_lambda_plprime <-  strenv$jnp$array(as.matrix( 2*lambda*p_vec_sum_prime_use),strenv$dtj)
    Neg4lambda_update <-  strenv$jnp$array(as.matrix(-4*lambda),strenv$dtj)
    Neg2lambda_update <-  strenv$jnp$array(as.matrix(-2*lambda),strenv$dtj)
    getPiStar_exact <- function(EST_COEFFICIENTS_tf){
      main_coef <-  strenv$jnp$take(EST_COEFFICIENTS_tf,indices = main_indices_i0, axis=0L)
      inter_coef <-  strenv$jnp$take(EST_COEFFICIENTS_tf,indices = inter_indices_i0, axis=0L)
      b_vec <-  strenv$jnp$subtract(   strenv$jnp$negative( main_coef ), Const_2_lambda_pl  )

      C_mat <- sapply(1:n_main_params,function(main_comp){
        # initialize to 0
        row_ <-  strenv$jnp$zeros(  list(n_main_params)  )

        # update diagonal component
        row_ <- JaxVectorScatterUpdate(row_,
                                       updates = Neg2lambda_update,
                                       indices = n2int(as.matrix(ai(main_comp-1L))))

        # update off-diagonal component (same d)
        same_d_diff_l <- n2int(as.matrix(ai(setdiff(which(main_info$d_adj == main_info[main_comp,]$d_adj),main_comp)-1L)))
        # see this on scatter update in jax https://github.com/google/jax/issues/9269

        # check these
        row_ <- JaxVectorScatterUpdate(tensor = row_,
                                      #updates = -2*lambda* strenv$jnp$ones(list(nrow(same_d_diff_l))), # is this right?
                                      updates =  strenv$jnp$zeros(list(same_d_diff_l$size)), # or is this right?
                                      indices = same_d_diff_l)

        # update off-diagonal component (different d)
        #interaction_info_red <- interaction_info[interaction_info$dl_index %in% main_comp | interaction_info$dplp_index %in% main_comp,]
        interaction_info_red <- interaction_info[
          (ind1<-(interaction_info$d_adj %in% main_info[main_comp,]$d_adj &
                    interaction_info$l %in% main_info[main_comp,]$l)) |
            (ind2<-(interaction_info$dp_adj %in% main_info[main_comp,]$d_adj &
                      interaction_info$lp %in% main_info[main_comp,]$l ) ),]
        id_d <- apply(interaction_info_red[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
        id_dp <- apply(interaction_info_red[,c("dp_adj","lp")],1,function(zer){paste(zer,collapse="_")})
        id_ <- ifelse(!(interaction_info_red$d_adj %in% main_info[main_comp,]$d_adj),
                      yes = id_d, no = id_dp)
        id_main <- apply(main_info[,c("d_adj","l")],1,function(zer){paste(zer,collapse="_")})
        which_inter <- which(ind1|ind2)
        inter_into_main <- sapply(id_,function(zer){which(id_main %in% zer)})

        if(nrow(interaction_info_red)>0){
          inter_coef_ <-  strenv$jnp$take(inter_coef,
                                  indices = n2int(ai(which_inter-1L)),
                                  axis = 0L)
          #if(length(which_inter) == 1){ inter_coef_ <-  strenv$jnp$expand_dims(inter_coef_,0L) }
          row_ <- JaxVectorScatterUpdate(row_,
                                         updates = inter_coef_,
                                         indices = n2int(as.matrix(ai(inter_into_main-1L))))
        }
        return(  strenv$jnp$expand_dims(row_,1L) )
      })
      C_mat <-  strenv$jnp$concatenate(C_mat,1L)

      return(  pi_star <-  strenv$jnp$matmul( strenv$jnp$linalg$inv(C_mat),b_vec)  )
    }
  }
}
