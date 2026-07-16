# Deprecated archive: retained for reference only and not sourced by the active package.
initialize_m <- function(){
    ### initializations for fast m estimation
    {
      GRADIENT_FXN_vectorized <- function(FactorsMat_DATA,
                                          Yobs_DATA,
                                          log_pr_w_DATA,
                                          pi_forGrad_simplexListed,
                                          SIGMA2_,
                                          n_in_sum_DATA){
        GradContribInfo <- ( sapply(1:length(pi_forGrad_simplexListed),function(k__){

          # grab data
          pi_forGrad_minusk <- pi_forGrad_simplexListed[-k__]
          pi_forGrad_k      <- pi_forGrad_simplexListed[[k__]]
          FactorsMat_k         <- FactorsMat_DATA[,k__]
          FactorsMat_minusk <- FactorsMat_DATA[,-k__]
          if(c("integer","numeric") %in% all(class(FactorsMat_minusk))){ FactorsMat_minusk <- as.matrix(FactorsMat_minusk) }

          # calculate pr_pi constants
          log_prodConstMinusk_ <- rowSums(  sapply(1:length(pi_forGrad_minusk),function(raa){log(  pi_forGrad_minusk[[raa]][FactorsMat_minusk[,raa]]  )}))
          log_prodConst_ <- log_prodConstMinusk_ + log(  pi_forGrad_k[FactorsMat_k] )

          # softmax gradient
          softMax      <- unlist(pi_forGrad_k[FactorsMat_k])
          softMax_grad <- sapply(2:length(pi_forGrad_k), function(factor_iter){
            softMaxGradComp_ <- ifelse(FactorsMat_k == factor_iter,
                                       yes = pi_forGrad_k[factor_iter]*(1-pi_forGrad_k[factor_iter]),
                                       no =   -pi_forGrad_k[FactorsMat_k]*(pi_forGrad_k[factor_iter]))
          })
          #ifelse(c(T,T,F,F),yes = 10,no = c(1,2,3,4))

          # q gradient
          pr_ratio <- exp( log_pr_ratio <- (log_prodConst_ - log_pr_w_DATA ) )
          pr_ratio_grad <- exp(log_prodConstMinusk_ - log_pr_w_DATA) * softMax_grad
          Q_contrib_grad <- Yobs_DATA * pr_ratio_grad
          Q_contrib <- Yobs_DATA * pr_ratio

          # max prob
          Q2_contrib <- Q2_contrib_grad <- maxProb <- maxProb_grad <- Var_contrib_grad1 <- Var_contrib_grad2 <- lagrangian_contrib_varBound <- NA
          if(penalty_type != "L2"){
            #v variance gradients
            Q2_contrib_grad <- Yobs_DATA^2 * pr_ratio_grad
            Q2_contrib <- Yobs_DATA^2 * pr_ratio

            maxProdConst_minusk  <- prod(  sapply(1:length(pi_forGrad_minusk),function(raa){
              max(pi_forGrad_minusk[[raa]])   }) )
            maxProb_k <- max(pi_forGrad_k);
            maxProb <- maxProdConst_minusk*maxProb_k
            MaxProb_whichmax <- which.max(pi_forGrad_k)
            maxProb_grad <- maxProdConst_minusk * sapply(2:length(pi_forGrad_k), function(factor_iter){
              softMaxGrad_ <- ifelse(pi_forGrad_k[factor_iter] == maxProb_k,
                                     yes = pi_forGrad_k[factor_iter]*(1-pi_forGrad_k[factor_iter]),
                                     no = -pi_forGrad_k[MaxProb_whichmax]*(pi_forGrad_k[factor_iter]) )
            })
            maxProb_grad <- matrix(maxProb_grad, nrow = nrow(Q2_contrib_grad) ,ncol=ncol(Q2_contrib_grad), byrow=T)

            # var
            Var_contrib_grad1 <- -1 / n_in_sum_DATA * SIGMA2_ * treatment_combs * maxProb_grad
            Var_contrib_grad2 <- -1 * treatment_combs * ( maxProb * Q2_contrib_grad +  maxProb_grad *  Q2_contrib)
            lagrangian_contrib_varBound <- LAMBDA_ / n_target * (Var_contrib_grad1 + Var_contrib_grad2)
          }
          #lagrangian_contrib_epBound  <- 1/n_target*( -lagrange_simplexListed[[k__]][1] * softMax_grad + lagrange_simplexListed[[k__]][2] * softMax_grad)
          # LAMBDA * ( 1-ep - pi_k ) AND  LAMBDA * ( pi_k - ep  )

          lagrangian_contrib <- lagrangian_contrib_varBound
          L2_grad <- L2_value <- NULL
          if(penalty_type == "L2"){
            pi_k <- pi_forGrad_k
            p_k  <- p_list[[k__]]
            L2_value <- ( (pi_k - p_k)^2 )[-1]
            L2_grad  <- (2 * ( (pi_k - p_k)*pi_k*(1-pi_k) -
                                 sapply(1:length(pi_k),function(ze){
                                   sum( (pi_k[-ze] -  p_k[-ze])*pi_k[-ze] *  pi_k[ze]) })))[-1]
            lagrangian_contrib <-  - c(LAMBDA_ * L2_grad)
          }
          full_grad <- try(t( t(Q_contrib_grad) + EXPERIMENTAL_SCALING_FACTOR * lagrangian_contrib),T)

          return( list("full_grad"      = full_grad,
                       "pr_ratio"       = pr_ratio,
                       "maxProb"        = maxProb,
                       "maxProb_grad"   = maxProb_grad,
                       "softMax"        = softMax,
                       "softMax_grad"   = softMax_grad,
                       "pr_ratio_grad"  = pr_ratio_grad,
                       "L2_value"       = L2_value,
                       "L2_grad"        = L2_grad,
                       "Q_contrib"      = Q_contrib,
                       "Q_contrib_grad" = Q_contrib_grad,
                       "Q2_contrib"     = Q2_contrib,
                       "Q2_contrib_grad" = Q2_contrib_grad
          ) )
        }) )
        return( GradContribInfo )
      }
      DD_full <- matrix(0, ncol = length( unlist(p_list)) - length(p_list),
                        nrow = length( unlist(p_list)) - length(p_list) )
      DD_full_indices <- c(0,cumsum(unlist(  lapply(p_list,function(zer){(length(zer)-1)}) )))
      INIT_d_d_mat <- DD_full; INIT_d_d_mat[] <- NA#matrix(, nrow = length(p_list), ncol = length(p_list))
      comb1 <- expand.grid(1:nrow(DD_full),1:nrow(DD_full));
      comb2 <- comb1[,2];  comb1 <- comb1[,1]

      HESSIAN_FXN <- function(GradContribInfo,
                              pi_forGrad_simplexListed_outer,
                              FactorsMat_DATA,
                              Yobs_DATA,
                              log_pr_w_DATA,
                              pi_forYVar_outer,
                              n_in_sum_DATA){
        dmaxProb_dakl_TIMES_dmaxProb_daktlt <- dmaxProb_TIMES_dpi_daktlt <- dmaxProb_dakl_TIMES_dwt2_daktlt <- INIT_d_d_mat

        dpi_dakl_TIMES_dpi_daktlt <- INIT_d_d_mat
        dpi_dakl_TIMES_dpi_daktlt[] <- unlist( GradContribInfo["softMax_grad",] )[comb1] *  unlist( GradContribInfo["softMax_grad",] )[comb2]
        dpi_daktlt_TIMES_dpi_dakl <- dpi_dakl_TIMES_dpi_daktlt

        if(penalty_type != "L2"){
          softMax_grad_e <- unlist( GradContribInfo["softMax_grad",] )
          maxProb_grad_e <- unlist( GradContribInfo["maxProb_grad",] )
          dmaxProb_dakl_TIMES_dmaxProb_daktlt[] <- maxProb_grad_e[comb1] * maxProb_grad_e[comb2]
          Q2_contrib_grad_e <- unlist(  GradContribInfo["Q2_contrib_grad",] )
          dmaxProb_dakl_TIMES_dwt2_daktlt[] <- maxProb_grad_e[comb1] * Q2_contrib_grad_e[comb2]
          dmaxProb_daktlt_TIMES_dwt2_dakl <- t( dmaxProb_dakl_TIMES_dwt2_daktlt ) # CHECK THIS
        }

        type_pool <- ifelse(penalty_type == "L2", yes = "ddwt_raw", no = c("ddmax","ddwt_raw"))
        for(type_ in type_pool){
          if(type_ %in% c("ddmax","ddwt_raw")){
            seq_k <- 1:(length(pi_forGrad_simplexListed_outer) - 1)
            seq_kkt <- seq_k[-length( seq_k )]
            ddcomp <- sapply(1:nrow(expanded_kl), function(kkt_entry){
              ze <- expanded_kl[kkt_entry,]
              k_ <- ze[1][[1]];  l_   <- ze[2]
              kt_ <- ze[3][[1]]; lt_  <- ze[4]
              if(k_ != kt_){
                FactorsMat_minuskkt_    <- FactorsMat_DATA[-c(k_,kt_)]
                pi_forGrad_minuskkt_ <- pi_forGrad_simplexListed_outer[-c(k_,kt_)]
                pi_forGrad_kkt_ <- pi_forGrad_simplexListed_outer[c(k_,kt_)]
                if(type_ %in% c("ddwt_raw")){
                  prodConst_  <- ifelse(length(FactorsMat_minuskkt_) > 0,
                                        #yes = prod(  sapply(seq_kkt,function(raa){ pi_forGrad_minuskkt_[[raa]][FactorsMat_minuskkt_[raa]]   }) ),
                                        yes = prod( pi_forGrad_minus_mat_list[[kkt_entry]][pi_forGrad_minuskkt_mat_vec_comp+FactorsMat_minuskkt_]),
                                        no = 1)
                  dd_ <- dpi_dakl_TIMES_dpi_daktlt[kref_vec == k_ & lref_vec == l_, kref_vec == kt_ & lref_vec == lt_]
                }
                if(type_ == "ddmax"){
                  prodConst_  <-   prod( unlist(lapply(pi_forGrad_minuskkt_,max)))
                  dd_ <- dmaxProb_dakl_TIMES_dmaxProb_daktlt[kref_vec == k_ & lref_vec == l_,  kref_vec == kt_ & lref_vec == lt_]
                }
              }
              if(k_ == kt_){
                FactorsMat_minusk_    <- FactorsMat_DATA[-k_]
                pi_forGrad_minusk_ <- pi_forGrad_simplexListed_outer[-k_]
                maxPi_k <- max( pi_forGrad_simplexListed_outer[[k_]] )
                pi_kl   <- pi_forGrad_simplexListed_outer[[k_]][l_]
                pi_ktlt <- pi_forGrad_simplexListed_outer[[kt_]][lt_]
                if(type_ == "ddmax"){
                  prodConst_  <- prod( unlist( lapply(pi_forGrad_minusk_,max) ) )
                  pi_select <- max(  pi_forGrad_simplexListed_outer[[k_]] )
                  cond1 <- pi_kl == maxPi_k & l_ == lt_
                  cond2 <- pi_kl == maxPi_k & l_ != lt_
                  cond3 <- pi_kl != maxPi_k & l_ == lt_
                  cond4 <- pi_kl != maxPi_k & pi_ktlt == maxPi_k
                  cond5 <- pi_kl != maxPi_k & pi_ktlt != maxPi_k & l_ != lt_
                }

                if(type_ %in% c("ddwt_raw")){
                  T_ik <- FactorsMat_DATA[ze[1]]
                  #prodConst_  <- prod(  sapply(seq_k,function(raa){ pi_forGrad_minusk_[[raa]][FactorsMat_minusk_[raa]]   }) )
                  prodConst_ <- prod( pi_forGrad_minus_mat_list[[kkt_entry]][pi_forGrad_minusk_mat_vec_comp+FactorsMat_minusk_])
                  pi_select <- pi_forGrad_simplexListed_outer[[k_]][T_ik]
                  cond1 <- l_ == T_ik & l_  == lt_
                  cond2 <- l_ == T_ik & l_  != lt_
                  cond3 <- l_ != T_ik & lt_ != T_ik & l_ == lt_
                  cond4 <- l_ != T_ik & lt_ == T_ik & l_ != lt_
                  cond5 <- l_ != T_ik & lt_ != T_ik & l_ != lt_
                }
                if(cond1){ dd_ <-   pi_kl*(1-pi_kl)*(1-2*pi_kl) } #checked
                if(cond2){ dd_ <-   -pi_kl*pi_ktlt*(1-2*pi_kl)  } #check
                if(cond3){ dd_ <- (2*pi_kl^2 * pi_select - pi_kl*pi_select) }
                if(cond4){ dd_ <- (2*pi_kl * pi_select^2 - pi_kl*pi_select) }
                if(cond5){ dd_ <-  2*pi_kl*pi_select*pi_ktlt  }
              }
              raw_dd <- dd_ * prodConst_
              return( raw_dd )
            })
          }
          if(type_ == "ddmax"){ddmax <- matrix(ddcomp,nrow=length(pi_init_vec))}
          if(type_ == "ddwt_raw"){ddwt_raw <- matrix(ddcomp / exp(log_pr_w_DATA),nrow=length(pi_init_vec))}
        }

        ddwt <- ddwt_raw * Yobs_DATA
        if(penalty_type != "L2"){
          ddwt2 <- ddwt_raw * Yobs_DATA^2
          TERM1 <-  LAMBDA_ / n_target * pi_forYVar_outer * treatment_combs * ddmax
          TERM2 <- LAMBDA_ / n_target * treatment_combs * (ddmax * GradContribInfo["Q2_contrib",1][[1]] +
                                                             dmaxProb_dakl_TIMES_dwt2_daktlt)
          TERM3 <- LAMBDA_ / n_target  * treatment_combs * (dmaxProb_daktlt_TIMES_dwt2_dakl +
                                                              GradContribInfo["maxProb",1][[1]]*ddwt2)
          HESSIAN <- 1*(ddwt - TERM1 - TERM2 - TERM3)
        }
        if(penalty_type == "L2"){
          HESSIAN <- 1*(ddwt -  LAMBDA_ * DD_L2Pen * EXPERIMENTAL_SCALING_FACTOR)
        }
        return( list(HESSIAN = HESSIAN,
                     ddwt = ddwt,
                     DD_L2Pen = DD_L2Pen) )
      }
    }

    # pre-processed k information
    {
      lref_vec <- unlist(  sapply(1:length(p_list),
                                  function(ze){
                                    l_ <- 2:length(p_list[[ ze ]])  }))
      kref_vec <- unlist(sapply(1:length(p_list),
                                function(ze){ k_ <- p_list[[ ze ]][-1] ;
                                k_[] <- ze; return( k_ )  }))
      names(lref_vec) <- names(kref_vec) <- NULL
      expanded_kl <- expand.grid(1:length(kref_vec), 1:length(kref_vec))
      expanded_kl <- cbind(cbind(kref_vec,lref_vec)[expanded_kl[,1],],
                           cbind(kref_vec,lref_vec)[expanded_kl[,2],])
      GRID_LIST <- lapply(p_list,function(ze){ expand.grid(2:length(ze),2:length(ze)) })

      # bound information (if supplied)
      all_names = unlist(lapply(p_list,function(ze){names(ze)}))

      # simplex generating functions
      splitIndices = as.factor(unlist(sapply(1:length(p_list),function(ze){
        rep(ze,times=length(p_list[[ze]]))  })))
      zeros_vec <- rep(0,times=length(unlist(p_list)));
      nonZero_indices <- lapply(1:length(p_list),function(ze){!duplicated(rep(ze,times=length(p_list[[ze]])))})
      nonZero_indices <- unlist(nonZero_indices)
      nonZero_indices <- which(!nonZero_indices)
      vec2list <- function(vec_){
        zeros_vec[nonZero_indices] <- vec_;
        return( lapply(split(zeros_vec,f = splitIndices),toSimplex) )
      }
      vec2list_noTransform0 <- function(vec_){# adds 0 entries
        zeros_vec[nonZero_indices] <- vec_
        return( split(zeros_vec,f = splitIndices) )   }
      vec2list_noTransform <- function(vec_){
        return( split(vec_,f = splitIndices)) }
      #system.time(replicate(1000,vec2list(pi_init_vec)))
      #lapply(p_list,length)[[1]]
      #vec2list(pi_init_vec)[[1]] - toSimplex(c(0,pi_init_vec[1:6]))
      #sum(abs(unlist(vec2list(pi_init_vec))-unlist(p_list))) # test: should be close to 0
    }
}
