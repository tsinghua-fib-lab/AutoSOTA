# Deprecated archive: retained for reference only and not sourced by the active package.
get_se <- function(){

  # start here - get SEs from M-estimation for marginal mean computation
  # Pr(ClassProb) = X beta + intercept
  VarCov_ProbClust <- ClassProbsXobs <- ClassProbProjCoefs <- ClassProbProjCoefs_se <- NULL
  if((piSEtype  == "automatic" | piSEtype == "both") ){
    {
      tf_LAMBDA_selected <- strenv$jnp$array(LAMBDA_selected,strenv$jnp$float32)
      tf_FALSE <- strenv$jnp$array( returnWeightsFxn(F),strenv$jnp$bool_); tf_TRUE <- strenv$jnp$array(  returnWeightsFxn(T),strenv$jnp$bool_)
      tf_LAMBDA_selected <- strenv$jnp$array( returnWeightsFxn(LAMBDA_selected), strenv$jnp$float32)

      {
        tf_true <- strenv$jnp$array(T,strenv$jnp$bool_); tf_false <- strenv$jnp$array(F,strenv$jnp$bool_)
        nParam_tf <- sum(unlist( lapply(lapply(unlist(ModelList_object),dim),prod)) )
        nParam_total <- strenv$jnp$array( nParam_tf + 1, strenv$jnp$int32)

        # unvectorized
        getJacobianComponents <- function(){
          psi_list <- neg_jacobian_list <- list()
          counter_ji <- 0; for(ji in split2_indices){
            counter_ji <- counter_ji + 1
            if(counter_ji %% 100 == 0){message(sprintf("M Estimation Iteration %i of %i",counter_ji,length(split2_indices)))}

            # jax calc
            {
              # convert
              i_ <- ji
              if(counter_ji == 1){
                for(fn_ in c("getProbRatio_tf", "getLoss_tf_unnormalized")){
                  grad_jax_fxn <- strenv$jax$grad(jax_fxn <- eval(parse(text = fn_)), argnums = 0L)

                  # test the function
                  # i_ <- 1
                  jax_eval <- jax_fxn(
                    ModelList_ = ModelList_object,
                    Y_  = strenv$jnp$array(as.matrix(Y[i_])),
                    X_  = strenv$jnp$array(t(X[i_,])),
                    factorMat_  = strenv$jnp$array(t(FactorsMat_numeric_0Indexed[i_, , drop=FALSE])),
                    logProb_ = strenv$jnp$array(as.matrix(log_PrW[i_])),
                    REGULARIZATION_LAMBDA = strenv$jnp$array(returnWeightsFxn(LAMBDA_selected))
                  )

                  # do some renaming
                  eval(parse(text = sprintf("%s <- jax_fxn", gsub(fn_, pattern="_tf",replacement="_jax"))))
                  eval(parse(text = sprintf("grad_%s <- grad_jax_fxn", gsub(fn_,pattern="_tf",replacement="_jax"))))
                  #eval(parse(text = sprintf("param_set_%s <- ModelList_object", gsub(fn_,pattern="_tf",replace="_jax"))))
                  #eval(parse(text = sprintf("param_set_names_%s <- param_set_names", gsub(fn_,pattern="_tf",replace="_jax"))))
                  rm(jax_fxn, grad_jax_fxn)
                } #end loop setting up dx/dp

                # get psi fxn
                psi_fxn_jax <- strenv$jax$jit( function(
                  # differentiate with respect to these
                  Qhat_tf, ModelList_,

                  # data inputs:
                  Y_, X_, factorMat_, logProb_, REGULARIZATION_LAMBDA
                  ){

                  probRatio_i_jax <- getProbRatio_jax(
                    ModelList_,
                    Y_  = Y_, X_  = X_,
                    factorMat_  = factorMat_, logProb_ = logProb_,
                    REGULARIZATION_LAMBDA = REGULARIZATION_LAMBDA)
                  probRatio_contrib_i <- strenv$jnp$subtract(strenv$jnp$multiply(strenv$jnp$array(Y_), probRatio_i_jax),
                                                      strenv$jnp$multiply(strenv$jnp$array(Qhat_tf), probRatio_i_jax))$flatten()
                  my_grad_i_jax <- grad_getLoss_jax_unnormalized(
                    ModelList_,
                    Y_  = Y_,
                    X_  = X_,
                    factorMat_  = factorMat_,
                    logProb_ = logProb_,
                    REGULARIZATION_LAMBDA = REGULARIZATION_LAMBDA)
                  my_grad_i_jax <- lapply(unlist(my_grad_i_jax), function(zap){zap$flatten()})

                  # set up hessian
                  my_psi_i_jax <- strenv$jnp$concatenate( c(probRatio_contrib_i,
                                                     my_grad_i_jax), 0L)

                  return( my_psi_i_jax )
                } )
                jacobian_psi_fxn_jax <- strenv$jax$jit( strenv$jax$jacobian(psi_fxn_jax, argnums = 0L:1L) )
                #system.time( strenv$jax$jacobian(psi_fxn_jax, argnums = 0L:1L) )
                #system.time( strenv$jax$jacfwd(psi_fxn_jax, argnums = 0L:1L) )
                #system.time( strenv$jax$jacrev(psi_fxn_jax, argnums = 0L:1L) )
              }

              Y__i <- strenv$jnp$array(as.matrix(Y[i_]))
              X__i <- strenv$jnp$array(t(X[i_,]))
              factorMat__i <- strenv$jnp$array(t(FactorsMat_numeric_0Indexed[i_, , drop=FALSE]))
              logProb__i <- strenv$jnp$array(as.matrix(log_PrW[i_]))
              REGULARIZATION_LAMBDA__ <- strenv$jnp$array(returnWeightsFxn(LAMBDA_selected))
              Qhat__ <-  Qhat_tf

              # get psi components
              psi_i <- psi_fxn_jax(
                Qhat_tf = Qhat__,
                ModelList_ = ModelList_object,
                Y_  = Y__i,
                X_  = X__i,
                factorMat_  = factorMat__i,
                logProb_ = logProb__i,
                REGULARIZATION_LAMBDA = REGULARIZATION_LAMBDA__
              )
              psi_i_jacobian <- jacobian_psi_fxn_jax(
                Qhat__,
                ModelList_object,
                Y__i,
                X__i,
                factorMat__i,
                logProb__i,
                REGULARIZATION_LAMBDA__
              )
              psi_i_jacobian <- unlist( psi_i_jacobian )
              names(psi_i_jacobian)[1] <- c("Qhat")
              #cbind(param_set_names,unlist(lapply(tv_trainWith,function(zer){zer$name})))
              #psi_i_jacobian <- psi_i_jacobian[c("Qhat",param_set_names)] # check

              psi_i_jacobian <- sapply(1:length(psi_i_jacobian), function(zer){
                strenv$np$array(psi_i_jacobian[[zer]]$reshape(nParam_total, -1L)) })
              if("list" %in% class(psi_i_jacobian)){
                psi_i_jacobian <- do.call(cbind, psi_i_jacobian)
              }
              psi_i_jacobian <- apply( psi_i_jacobian,2,f2n )

              psi_ji <- strenv$np$array(  psi_i )
              jacobian_ji <- psi_i_jacobian;
            }

            # checks
            if(T == F){
              cbind(strenv$np$array(psi_i)[1:10],psi_ji[1:10])
              plot(strenv$np$array(psi_i),psi_ji); abline(a=0,b=1)
              plot(c(jacobian_ji),c(psi_i_jacobian)); abline(a=0,b=1)
              plot(c(jacobian_ji),c(t(psi_i_jacobian))); abline(a=0,b=1)
              plot(abs(c(jacobian_ji)-c(psi_i_jacobian)))
              abline(a=0, b = 1); abline(a=0, b = -1)
              image(abs(jacobian_ji-psi_i_jacobian))
              plot(c(jacobian_ji[,1]),c(psi_i_jacobian[,1]));abline(a=0,b=1)
              plot(c(jacobian_ji[,2]),c(psi_i_jacobian[,2]));abline(a=0,b=1)
              plot(c(jacobian_ji[,2])-c(psi_i_jacobian[,2]));abline(h=0)
            }

            # sum(is.na(jacobian_ji))
            # save m est results to list holders
            psi_list[[counter_ji]] <- psi_ji
            neg_jacobian_list[[counter_ji]] <-   -1 * jacobian_ji
          }
          return(list("psi_list" = psi_list,
                      "neg_jacobian_list" = neg_jacobian_list))
        }
        JacobianComponents <- getJacobianComponents()
        psi_list <- JacobianComponents$psi_list
        neg_jacobian_list <- JacobianComponents$neg_jacobian_list
        jacob_NAs <- unlist( lapply(neg_jacobian_list,function(l_){is.na(sum(l_))}) )
        # X[which( jacob_NAs ),]
        # FactorsMat_numeric_0Indexed[which( jacob_NAs ),]
        if(sum(jacob_NAs) > 0){ message("Warning: Jacobian contains NAs! Dropping...") }
        psi_list <- psi_list[!jacob_NAs]
        neg_jacobian_list <- neg_jacobian_list[!jacob_NAs]
        rm( JacobianComponents )

        # resources
        #https://www.tensorflow.org/guide/advanced_autodiff
        #https://tensorflow.google.cn/api_docs/python/tf/vectorized_map?hl=zh-cn
        # need to deal with k > 2 case?
      }

      # calculate variance-covariance matrix
      {
        l1_psi <- unlist(lapply(psi_list,function(zer){mean(abs(zer))}))
        l1_neg_jacobian <- unlist(lapply(neg_jacobian_list,function(zer){mean(abs(zer))}))
        l1_check_ref <- split2_indices[which.max(l1_neg_jacobian)]
        try(hist( log(l1_neg_jacobian ), main=sprintf("Max value is %.3f",max(l1_neg_jacobian))),T)

        # calculate variance covariance matrix using A and B
        {
          A_n <- 1/(n_m <- length(psi_list)) * Reduce("+", neg_jacobian_list)
          B_n <- 1/n_m * Reduce("+", lapply(psi_list,function(psi_){ psi_ %*% t(psi_) }))

          # drop components of analysis from K > 2 case
          if(K == 1){
            A_n <- A_n[-nrow(A_n), -ncol(A_n)]
            B_n <- B_n[-nrow(B_n), -ncol(B_n)]
          }

          A_n_inv <- try(solve(A_n), T)
          if(all( "try-error" %in% class(A_n_inv))){warning("Singular A_n -- Forcing Correction"); A_n_inv <- solve(A_n+0.01*diag(nrow(A_n)))}
          VarCov_n_automatic <- 1 / n_m * ( A_n_inv %*% B_n %*% t(A_n_inv) )
          VarCov_names <- c("Qhat")
          for(k_ in 1:K){
            newnames <- paste( paste("k",k_, "av", sep = ""), 1:length(ModelList_object[[1]][[1]]), sep = "")
            newname_lengths <- unlist( lapply(ModelList_object[[1]][[1]], function(zer){ zer$size}) )
            newnames <- sapply(1:length(newnames), function(zer){
              paste(newnames[[zer]], "d", 1:newname_lengths[[zer]], sep = "") })
            VarCov_names <- c(VarCov_names, newnames)
          }
          if(K > 1){ VarCov_names <- c(VarCov_names, paste("ProjKernel", 1:ModelList_object[[2]][[1]]$size, sep=""),
                                                       paste("ProjBias", 1:ModelList_object[[2]][[2]]$size, sep="")) }
        }
      }

      # check this, make sure flattening is done correctly
      Mean_n_automatic <- c("Qhat" = Qhat, unlist(  rrapply::rrapply(ModelList_object, f = function(zer){
        values_ <- strenv$np$array( strenv$jnp$reshape(zer,-1L) )
        #if(prod(dim(zer))>0){names(values_) <- paste(zer$name,1:length(values_),sep="_")}
        return( values_ )
      }) ) )
      if(K == 1){ Mean_n_automatic <- Mean_n_automatic[ -length(Mean_n_automatic) ]  }

      # rename (implement)
      colnames(VarCov_n_automatic) <- row.names( VarCov_n_automatic )  <- VarCov_names

      # obtain uncertainties
      {
        m_se_Q   = sqrt( VarCov_n_automatic[1,1] )
        seList_automatic <- getPiList(SimplexList = ModelList_object[[1]],
                                      VarCov = VarCov_n_automatic,
                                      simplex = T, rename= T,return_SE = T)$FinalSEList

        # projection coefficients + SEs
        if(K > 1){
          ClassProbsXobs <- as.array( getClassProb( X) )
          BinaryCovariateIndicator <- apply(X_factorized, 2, function(zer){names(table(zer))})
          BinaryCovariateIndicator <- unlist( lapply(BinaryCovariateIndicator,function(zer){mean(zer[1:2] == c("0","1") & length(zer)==2)}))
          X_completion <- sapply(which(BinaryCovariateIndicator==0),function(zer){
            zap_ <- X_factorized[,zer]
            if(length(zap_) < 10){
              zap_completion <- eval(parse(text = sprintf("
            model.matrix(~0+factor(%s),data = as.data.frame(X_factorized))",
                                                          colnames(X_factorized)[zer])))
            }
            if(length(zap_) > 10){
              X_factorized_ <- as.data.frame( X_factorized )
              eval(parse(text = sprintf("
                      X_factorized_$%s_quant <- as.numeric( gtools::quantcut(zap_, q = 5) )
                                    ", colnames(X_factorized)[zer])))
              zap_completion <- eval(parse(text = sprintf("
            model.matrix(~0+factor(%s_quant),data = (X_factorized_))",
                                                          colnames(X_factorized_)[zer])))
            }
            return( list(zap_completion ))
          })
          X_completion <- do.call(cbind, X_completion)
          X_factorized_complete <- cbind(
            X_factorized[,which(BinaryCovariateIndicator==1)],
            X_completion)
          which_factor <- unlist( lapply(strsplit(colnames( X_factorized_complete ),split="factor\\("),function(xer){xer[2]}) )
          which_factor_names <- which_factor <- unlist(lapply(strsplit(which_factor,split="\\)"),function(zer){zer[1]}))
          which_factor <- cumsum(!duplicated(which_factor))
          which_factor_mat <- strenv$jnp$array( model.matrix(~0+factor(which_factor)),strenv$jnp$float32)
          # some tests
          {
            #\sum X * Pr(Clust|X) 1/n
            #1  = sum_x Pr(X_d=x | Clust)
            #Pr(X_d=x | Clust) =
            #\sum_x' Pr(X_d=x,X_d'=x'|Clust) =
            #\sum_x' Pr(Clust|X_d=x,X_d'=x') Pr(Clust) / Pr(X_d=x,X_d'=x') =
            #\sum_x' Pr(Clust|X_d=x,X_d'=x') Pr(Clust) / U =
            #1/3*1/3 / (0.5)

            # Pr(Clust|X_d=x,X_d'=x')
            # Pr(Clust) /
            #  sum_x Pr(Clust|X) Pr(Clust)
            #E[X|K=k]
          }
          with(strenv$tf$GradientTape(persistent = T) %as% tape, {
            PrClustGivenX <- getClassProb(strenv$jnp$array(X, strenv$jnp$float32))

            # for checking:
            #PrClustGivenX <- strenv$jnp$array(matrix(rep(c(1/2), times = length( PrClustGivenX )), ncol=2),strenv$jnp$float32)
            #PrClustGivenX <- strenv$jnp$array(matrix(rep(c(0.9,0.1), times = length( PrClustGivenX )/2), ncol=2,byrow=T),strenv$jnp$float32)
            if(T == F){
              # testing
              PrClustGivenX_orig <- PrClustGivenX <- strenv$jnp$array(
                cbind(1*X_factorized_complete[,3] == max(X_factorized_complete[,3]),
                      1*X_factorized_complete[,3] == min(X_factorized_complete[,3])),strenv$jnp$float32)
            }
            PrClust <- strenv$jnp$mean(PrClustGivenX,0L,keepdims=T)
            PrClustGivenX <- strenv$jnp$expand_dims(PrClustGivenX,2L)
            X_factorized_expand <- strenv$jnp$expand_dims(strenv$jnp$array(X_factorized_complete,strenv$jnp$float32),1L)

            #Pr(X=1 | Clust = k) = Pr(Clust = k | X = 1) Pr(X = 1) / Pr(Clust = k)
            TotalNumberWithXd <- strenv$jnp$sum(strenv$jnp$multiply(X_factorized_expand,strenv$jnp$expand_dims(strenv$jnp$transpose(which_factor_mat),0L)),0L:1L)
            # think of PrClustGivenXd as mean(as.matrix(PrClustGivenX_orig)[,2][X_factorized_complete[,3] == 1])
            PrClustGivenXd <- strenv$jnp$divide(strenv$jnp$sum(strenv$jnp$multiply(PrClustGivenX,X_factorized_expand),0L),TotalNumberWithXd)
            PrXd <- strenv$jnp$expand_dims(TotalNumberWithXd / nrow(X_factorized_complete),0L)
            PrXdGivenClust <- strenv$jnp$divide(strenv$jnp$multiply(PrClustGivenXd,PrXd),strenv$jnp$transpose(PrClust))

            #PrXdGivenClust <- strenv$jnp$sum(strenv$jnp$multiply(PrClustGivenX,X_factorized_expand),0L)
            #PrXd <- strenv$jnp$matmul(strenv$jnp$matmul(PrXdGivenClust,which_factor_mat),strenv$jnp$transpose(which_factor_mat))
            #PrXdGivenClust <- PrXdGivenClust / PrXd
            #orig_ <- as.matrix(PrXdGivenClust)
            # plot(c(as.matrix(PrXdGivenClust)),c(as.matrix(orig_)));abline(a=0,b=1)
          })
          #Pr(Xd|Clust) = Pr(Xd AND Clust 1) = Pr(Xd | Clust 1) Pr(Clust 1)
          #E[X|Clust]  = sum_x x Pr(X=x|Clust) = sum_x x (Pr(Clust | X) Pr(X) / Pr(Clust))
          # Pr(X | Clust) = Pr(Clust | X) Pr(X) / Pr(Clust)

          dPrXdGivenClust_dParam_orig <- dPrXdGivenClust_dParam <- tape$jacobian( PrXdGivenClust,
                                                                                  ClassProbProj$trainable_variables )
          dPrXdGivenClust_dParam <- lapply(dPrXdGivenClust_dParam,function(zer){
            if(length(dim(zer))==3){ zer <- strenv$jnp$expand_dims(zer,3L) }
            return( zer)  })
          dPrXdGivenClust_dParam <- lapply(dPrXdGivenClust_dParam,function(zer){
            strenv$jnp$reshape(zer,c(dim(zer)[1:2],-1L))})
          dPrXdGivenClust_dParam <- strenv$jnp$concatenate(dPrXdGivenClust_dParam,2L)
          PrXdGivenClust_mat <- strenv$np$array(PrXdGivenClust)
          row.names(PrXdGivenClust_mat) <- paste("k",1:K,sep="")
          colnames(PrXdGivenClust_mat) <- colnames(X_factorized_complete)
          PrXd_vec <- colMeans(X_factorized_complete)

          kernel_indices <- grep(row.names(VarCov_n_automatic), pattern="ClassProbProj/kernel")
          bias_indices <- grep(row.names(VarCov_n_automatic), pattern="ClassProbProj/bias")
          VarCov_n_ProbClustParam <- VarCov_n_automatic[c(kernel_indices,bias_indices),
                                                        c(kernel_indices,bias_indices)]
          VarCov_n_ProbClustParam <- strenv$jnp$array(VarCov_n_ProbClustParam,strenv$jnp$float32)
          VarCov_n_ProbClustParam <- strenv$jnp$expand_dims(VarCov_n_ProbClustParam,0L)
          PrXdGivenClust_se_tf <- strenv$jnp$matmul(
            strenv$jnp$matmul(dPrXdGivenClust_dParam, VarCov_n_ProbClustParam),
            strenv$jnp$transpose(dPrXdGivenClust_dParam,c(0L,2L,1L)))
          PrXdGivenClust_se_tf <- strenv$jnp$diag(PrXdGivenClust_se_tf)
          PrXdGivenClust_se <- strenv$np$array(PrXdGivenClust_se_tf)
          row.names(PrXdGivenClust_se) <- row.names(PrXdGivenClust_mat)
          colnames(PrXdGivenClust_se) <- colnames(PrXdGivenClust_mat)
        }
      }
      if(any(is.na(unlist(seList_automatic)))){
        warning("NA values encountered in standard error calculations")
      }

      for(k__ in 1:K){
        try(seList_automatic[[k__]] <- sapply(1:length(pi_list[[k__]]),function(d_){
          names(seList_automatic[[k__]][[d_]]) <- names(  pi_list[[k__]][[d_]] )
          list(seList_automatic[[k__]][[d_]]) }), T)
        try(  names(seList_automatic[[k__]]) <- names( pi_list[[k__]]) , T)
      }

      names(hypotheticalProbList_full) <- paste("k",1:length(hypotheticalProbList_full),sep = "")
      names(seList_automatic) <- paste("k",1:length(seList_automatic),sep = "")

      seList <- seList_automatic
    }
  }
  #\ sum_i X_i Pr(Z_k = k)
  # \sum_i X_i

  # get upper and lower lists
  ensure0t1 <- function(ze){ ze[ze<0] <- 0; ze[ze>1] <- 1;ze}
  for(sign_ in c(-1,1)){
    boundsList <- sapply(1:length(hypotheticalProbList_full),function(k_){
      k_res <- sapply(1:length(hypotheticalProbList_full[[k_]]),function(d_){
        tmp_ <- hypotheticalProbList_full[[k_]][[d_]] + sign_*abs(qnorm((1-confLevel)/2))*seList[[k_]][[d_]]
        list( tmp_ )
      })
      names(k_res) <- names( hypotheticalProbList_full[[k_]] )
      return( list(k_res) )
    })
    if(sign_ == -1){lowerList <- boundsList}
    if(sign_ == 1){upperList <- boundsList}
  }
  names(upperList) <- names(lowerList) <-  names( hypotheticalProbList_full )
}
