# Deprecated archive: retained for reference only and not sourced by the active package.
ml_train <- function(){

print("Begin training block...")
device_scalar_to_numeric <- function(x) {
  as.numeric(strenv$np$array(x))
}
device_scalar_list_to_numeric <- function(x_list) {
  vapply(x_list, function(x) {
    if (is.null(x)) {
      NA_real_
    } else {
      device_scalar_to_numeric(x)
    }
  }, numeric(1))
}
batch_indices_to_device <- function(batch_idx) {
  strenv$jnp$array(as.integer(batch_idx - 1L), dtype = strenv$jnp$int32)
}
Y_jax_all <- strenv$jnp$array(as.matrix(Y), strenv$jnp$float32)
X_jax_all <- strenv$jnp$array(as.matrix(X), strenv$jnp$float32)
factorMat_jax_all <- strenv$jnp$array(FactorsMat_numeric_0Indexed, strenv$jnp$int32)
logProb_jax_all <- strenv$jnp$array(as.matrix(log_PrW), strenv$jnp$float32)
for(trainIndicator in trainIndicator_pool){
  # then iterate over folds
  for(fold_ in 1:(n_folds*trainIndicator+1*!trainIndicator)){

  # iterate over lambda FIRST (for warm starts...)

  lambda_counter <- 0; for(REGULARIZATION_LAMBDA in lambda_seq){
        lambda_counter <- lambda_counter + 1

      # training indices pool
      if(trainIndicator == 1){
        availableTrainIndices <- split1_indices[holdBasis_traintv != fold_]
        holdIndices <- split1_indices[holdBasis_traintv == fold_]
      }
      if(trainIndicator == 0){
        if(warm_start){ nSGD <- nSGD * 3 }
        availableTrainIndices <- holdIndices <- split2_indices
      }

      # setup epochs and indices
      if(!is.null(nEpoch) & !is.null(nSGD)){ print("Defaulting to nEpoch over nSGD") }
      if(is.null(nEpoch) & !is.null(nSGD)){
        print("Inferring nEpoch")
        nEpoch <- ceiling(nSGD *  batch_size / length(availableTrainIndices))
      }
      availableTrainIndices_train_seq <- c(replicate(nEpoch,
                tapply(sample(availableTrainIndices),
                            1:length(availableTrainIndices) %%
                               round(length(availableTrainIndices)/batch_size),c) ))
      min_numb <- max(1,min(unlist(lapply(availableTrainIndices_train_seq,length)),na.rm=T))
      availableTrainIndices_train_seq <- lapply(availableTrainIndices_train_seq,function(zer){
        sample(zer,min_numb,replace=F) })
      print(sprintf("N Obs Per Iter: %i",min_numb))
      #table(unlist(lapply(availableTrainIndices_train_seq,length)))

      nSGD <-  length(availableTrainIndices_train_seq)
      print(sprintf("nSGD: %s, nEpoch: %s",nSGD,nEpoch))

      # obtain initialization
      if(fold_ == 1){ selected_bscale <- 1 }
      print(sprintf("Selected bscale: %.4f",selected_bscale))

      # Re-initialize trainable variables right before new training begins
      # IF warmstarts OFF or FOLD == 1 or TESTING
      if(REINIT_BOOL <- (!warm_start | lambda_counter == 1 | trainIndicator == F)){ print("Reinitializing...") }
      if(!REINIT_BOOL){  print("NOT reinitializing variables...")   }

      # start training process
      sg_method_use <- if (length(sg_method) < 1L || is.null(sg_method)) {
        ""
      } else {
        as.character(sg_method[[1L]])
      }
      AdagradLR_vec_eff <- AdagradLR_vec <- MomenetumNextIter_seq <- LR_effective <- InvLR_tracker <- rep(NA, times = nSGD)
      loss_vec_device <- vector("list", length = nSGD)
      L2_norm_squared_vec_device <- vector("list", length = nSGD)
      track_host_grad_metrics <- sg_method_use %in% c("adanorm", "wngrad")
      reg_lambda_jax <- strenv$jnp$array(returnWeightsFxn(REGULARIZATION_LAMBDA), strenv$jnp$float32)
      {
        marginalProb_m <- c()
        i_eff <- 1;
        gc(); strenv$py_gc$collect()
        for(i in 1:nSGD){
          if(i %% 50 == 0L){ gc(); strenv$py_gc$collect() }
          # generate batch indices
          my_batch <- availableTrainIndices_train_seq[[i]]

          # train
          {
            #my_batch_jax <- my_batch[ jax_batch_select <- (1:(length(my_batch) )) ]
            my_batch_jax <- my_batch[ jax_batch_select <- seq_len(max(length(my_batch) - 2L, 1L)) ]

            # define training function via jax
            if(i == 1 & lambda_counter == 1){
              batch_size_jax <- length(my_batch_jax)
              loss_fxn_base <- if (exists("getLoss_tf_raw", inherits = TRUE)) {
                getLoss_tf_raw
              } else {
                getLoss_tf
              }
              v_and_grad_jax_fxn_raw <- strenv$jax$value_and_grad(loss_fxn_base, argnums = 0L)
            }

            # need to reset ModelList_object at i == 1
            if( i == 1 ){
              if(lambda_counter == 1){
                ModelList_object_new_init <- ModelList_object
                #names(ModelList_object_new_init) <- names( ModelList_object )
              }
              ModelList_object <- ModelList_object_new_init;

              # setup optimizer
              optim_type <- "Other"
              if(optim_type == "SecondOrder"){
                hessian_fxn <- strenv$jax$jit(
                  strenv$jax$hessian(loss_fxn_base, argnums = 0L)
                )
                optax_optimizer <-  strenv$optax$chain(
                  strenv$optax$adaptive_grad_clip(1, eps=0.0001),
                  strenv$optax$zero_nans(),
                  strenv$optax$scale(1) )
              }
              if(optim_type == "Other"){
                optax_optimizer <-  strenv$optax$chain(
                  #strenv$optax$sgd(momentum = 0.90, nesterov = T,
                  strenv$optax$adaptive_grad_clip(0.1, eps=0.0001),
                  #strenv$optax$fromage(learning_rate = 0.1)
                  #strenv$optax$clip(1.),
                  strenv$optax$adabelief(learning_rate=0.01)
                  #strenv$optax$scale_by_rss(), strenv$optax$scale(-1)
                  #strenv$optax$scale_by_rss(), strenv$optax$noisy_sgd(learning_rate = 1)
                  )
              }

              # model partition + setup state
              opt_state <- optax_optimizer$init( ModelList_object )
              jit_apply_updates <- strenv$eq$filter_jit(strenv$optax$apply_updates)
              jit_update <- strenv$eq$filter_jit(optax_optimizer$update)
              train_step_jax <- strenv$eq$filter_jit(function(model_list,
                                                              opt_state_in,
                                                              batch_idx,
                                                              reg_lambda_in) {
                Y_batch <- strenv$jnp$take(Y_jax_all, batch_idx, axis = 0L)
                X_batch <- strenv$jnp$take(X_jax_all, batch_idx, axis = 0L)
                factorMat_batch <- strenv$jnp$take(factorMat_jax_all, batch_idx, axis = 0L)
                logProb_batch <- strenv$jnp$take(logProb_jax_all, batch_idx, axis = 0L)
                v_and_grad_eval <- v_and_grad_jax_fxn_raw(
                  model_list,
                  Y_ = Y_batch,
                  X_ = X_batch,
                  factorMat_ = factorMat_batch,
                  logProb_ = logProb_batch,
                  REGULARIZATION_LAMBDA = reg_lambda_in
                )
                grad_set <- v_and_grad_eval[[2]]
                grad_norm_sq <- strenv$jnp$square(strenv$optax$global_norm(grad_set))
                updates_and_opt_state <- optax_optimizer$update(
                  updates = grad_set,
                  state = opt_state_in,
                  params = model_list
                )
                opt_state_out <- updates_and_opt_state[[2]]
                model_out <- strenv$optax$apply_updates(
                  params = model_list,
                  updates = updates_and_opt_state[[1]]
                )
                list(model_out, opt_state_out, v_and_grad_eval[[1]], grad_norm_sq)
              })
            }

            # fix jax training size due to compiled functionality
            my_batch_jax <- my_batch_jax[1:batch_size_jax]
            my_batch_jax[is.na(my_batch_jax)] <- my_batch_jax[sample(1:sum(!is.na(my_batch_jax)),
                                        size = length(my_batch_jax[is.na(my_batch_jax)]), replace = T)]
            my_batch_idx_jax <- batch_indices_to_device(my_batch_jax)

            {
            # updates + derivatives using jax
            if(optim_type == "Other"){
              step_eval <- train_step_jax(
                ModelList_object,
                opt_state,
                my_batch_idx_jax,
                reg_lambda_jax
              )
              ModelList_object <- step_eval[[1]]
              opt_state <- step_eval[[2]]
              currentLossGlobal_tf <- step_eval[[3]]
              L2_norm_squared_tf <- step_eval[[4]]
              loss_vec_device[[i]] <- currentLossGlobal_tf
              L2_norm_squared_vec_device[[i]] <- L2_norm_squared_tf
              if (track_host_grad_metrics) {
                L2_norm_squared <- device_scalar_to_numeric(L2_norm_squared_tf)
              } else {
                L2_norm_squared <- NA_real_
              }
            } else {
              v_and_grad_eval <- v_and_grad_jax_fxn_raw(
                ModelList_object,
                Y_ = strenv$jnp$take(Y_jax_all, my_batch_idx_jax, axis = 0L),
                X_ = strenv$jnp$take(X_jax_all, my_batch_idx_jax, axis = 0L),
                factorMat_ = strenv$jnp$take(factorMat_jax_all, my_batch_idx_jax, axis = 0L),
                logProb_ = strenv$jnp$take(logProb_jax_all, my_batch_idx_jax, axis = 0L),
                REGULARIZATION_LAMBDA = reg_lambda_jax
              )

              currentLossGlobal_tf <- v_and_grad_eval[[1]]
              grad_set <- v_and_grad_eval[[2]] #strenv$jax$grad screws up name orders
              L2_norm_squared_tf <- strenv$jnp$square(strenv$optax$global_norm(grad_set))
              loss_vec_device[[i]] <- currentLossGlobal_tf
              L2_norm_squared_vec_device[[i]] <- L2_norm_squared_tf
              L2_norm_squared <- device_scalar_to_numeric(L2_norm_squared_tf)
            }

            if(optim_type == "SecondOrder"){
              currentLossGlobal <- device_scalar_to_numeric(currentLossGlobal_tf)
              hessian_value <- hessian_fxn(ModelList_object,
                                           Y_ = strenv$jnp$take(Y_jax_all, my_batch_idx_jax, axis = 0L),
                                           X_ = strenv$jnp$take(X_jax_all, my_batch_idx_jax, axis = 0L),
                                           factorMat_ = strenv$jnp$take(factorMat_jax_all, my_batch_idx_jax, axis = 0L),
                                           logProb_ = strenv$jnp$take(logProb_jax_all, my_batch_idx_jax, axis = 0L),
                                           REGULARIZATION_LAMBDA = reg_lambda_jax)
              HessianMat <- matrix(list(),nrow = length(ModelList_object),ncol=length(ModelList_object))
              row.names(HessianMat) <- colnames(HessianMat) <- names(hessian_value)
              for(jaa in names(hessian_value)){ for(ja in names(hessian_value)){
                HessianMat[jaa,ja] <- list(eval(parse(text = sprintf("strenv$jnp$squeeze(strenv$jnp$squeeze(hessian_value$`%s`$`%s`,1L),2L)",jaa,ja))))
              }}
              HessianMat <- apply(HessianMat,1,function(zer){ names(zer) <- NULL;
                              strenv$jnp$concatenate(zer,1L) })
              names(HessianMat) <- NULL; HessianMat <- strenv$jnp$concatenate(HessianMat,0L)

              if(i == 1){ HessianMat_AVE <- strenv$jnp$zeros(HessianMat$shape) }
              {
                  w_i_minus_1 <- (((i-1) + 1)^log((i-1)+1))
                  w_i <- (i + 1)^log(i+1)
                  HessianMomentum <- w_i_minus_1 / w_i
                  HessianMat_AVE <- strenv$jnp$add(strenv$jnp$multiply(HessianMomentum,HessianMat_AVE),
                                                strenv$jnp$multiply(1-HessianMomentum,HessianMat))
              }

              # get flat grad
              grad_vec <- strenv$jax$flatten_util$ravel_pytree( grad_set )

              # get inv hessian times grad
              #image( (strenv$jnp$linalg$inv(HessianMat_AVE)$to_py() ))
              #https://arxiv.org/pdf/2204.09266.pdf
              # armijo condition test
              rho <- 0.98
              NetwornDir <- strenv$jnp$negative( strenv$jnp$matmul(strenv$jnp$linalg$inv(
                strenv$jnp$add(HessianMat_AVE,strenv$jnp$multiply(0.5,strenv$jnp$identity(HessianMat$shape[[1]])))
                ), grad_vec[[1]]) )
              InnerProd_GradNewtonDir <- device_scalar_to_numeric(
                strenv$jnp$sum(strenv$jnp$multiply(NetwornDir, grad_vec[[1]]))
              )
              if(InnerProd_GradNewtonDir > 0){ grad_set <- grad_vec[[2]](strenv$jnp$zeros(NetwornDir$shape))}
              if(InnerProd_GradNewtonDir <= 0){
              armijo_count <- 0; go_on <- F; while(go_on == F){
                armijo_count <- armijo_count + 1
                SecondOrderUpdates <- strenv$jnp$multiply(mu_t <- rho^armijo_count,
                                                   NetwornDir)
                SecondOrderUpdates <-  grad_vec[[2]](SecondOrderUpdates)

                ModelList_object_test <- jit_apply_updates(params = ModelList_object,
                                    updates = SecondOrderUpdates)
                f_x_updated <- getLoss_tf(
                  ModelList_object_test,
                  Y_ = strenv$jnp$take(Y_jax_all, my_batch_idx_jax, axis = 0L),
                  X_ = strenv$jnp$take(X_jax_all, my_batch_idx_jax, axis = 0L),
                  factorMat_ = strenv$jnp$take(factorMat_jax_all, my_batch_idx_jax, axis = 0L),
                  logProb_ = strenv$jnp$take(logProb_jax_all, my_batch_idx_jax, axis = 0L),
                  REGULARIZATION_LAMBDA = reg_lambda_jax
                )
                armijo_cond <- device_scalar_to_numeric(f_x_updated) <=
                  (currentLossGlobal + mu_t*0.25*InnerProd_GradNewtonDir )
                #if(is.na(armijo_cond)){browser()}
                if(armijo_count > 100){go_on <- T; SecondOrderUpdates <- grad_vec[[2]](strenv$jnp$zeros(NetwornDir$shape))}
                if(armijo_cond == T){ go_on <- T }
              }
              grad_set <- SecondOrderUpdates
            }
            }

            # perform model updates
            if(optim_type != "Other"){
              updates_and_opt_state <- jit_update(
                updates = grad_set,
                state = opt_state,
                params = ModelList_object)
              optax_updates <- updates_and_opt_state[[1]]
              opt_state <- updates_and_opt_state[[2]]

              ModelList_object <- jit_apply_updates(params = ModelList_object,
                        updates = optax_updates)
            }
            }
          }

          # report on performance 1 + 4 times during training
          if(i %% max(round( nSGD/4 ), 1) == 0 | i == 1){
            currentLossGlobal <- device_scalar_to_numeric(currentLossGlobal_tf)
            try_ <- try(print(sprintf("[%s] Iter %i of %i - Fold %i - Lambda %i of %i - Current obj: %.3f",
                                      format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                                      i, nSGD, fold_, lambda_counter, length(lambda_seq), currentLossGlobal)),T)
            if("try-error" %in% class(try_)){
              warning("Error encountered while printing progress")
            }
          }

          # cycle info
          if(cycle_width < Inf){if(i %% cycle_width == 0 & i < nSGD / 2){ i_eff <- 1 } }
          if(identical(sg_method_use, "wngrad")){ InvLR_tracker[ i+1 ] <- InvLR_tracker[i] + L2_norm_squared / InvLR_tracker[i] }
          if(identical(sg_method_use, "adanorm")){ InvLR_tracker[ i+1 ] <- InvLR_tracker[i] + L2_norm_squared }
          i_eff <- i_eff + 1;

          # AdaGrad-Norm https://arxiv.org/pdf/1806.01811.pdf
          if(identical(sg_method_use, "adanorm")){ LR_effective[i] <- 1/sqrt( InvLR_tracker[i_eff] ) }

          # WN Grad LR
          if(identical(sg_method_use, "wngrad")){ LR_effective[i] <- 1/InvLR_tracker[i_eff] }

          # cosine LR
          if(identical(sg_method_use, "cosine")){
            LR_effective[i] <- 0.1*abs(cos(i/nSGD*cycle_width))*(i<nSGD/2) +
              NA20(0.1*(i>=nSGD/2)/(i-nSGD/2)^0.2*(i>nSGD/2))
          }
        }

        # figs to figure out dynamics
        {
          loss_vec <- device_scalar_list_to_numeric(loss_vec_device)
          L2_norm_squared_vec <- device_scalar_list_to_numeric(L2_norm_squared_vec_device)
          gc(); strenv$py_gc$collect()
          #try(plot(MomenetumNextIter_seq), T)
          try(plot(LR_effective), T)
          try(plot(L2_norm_squared_vec),T)
          try(plot(  loss_vec   ),T)
          try(points(lowess(loss_vec),type = "l",lwd = 5, col = "red"),T)
          if(K > 1){
              try(plot( as.matrix(getClassProb(tfConst(X[availableTrainIndices,])))[,2] ),T)
          }
        }
      }

      # in sample objective
      #try(plot(unlist( lapply(getPiList(  ModelList_object[[1]]   ),function(zer){ unlist(zer) - unlist(p_list) }) ) ),T)
      getQ_fxn <- function(ModelList_, indices_){
          # broken up indices
          # batch_indices_Q <- tapply(sample(indices_),1:length(indices_) %% round(length(indices_)/batch_size),c)

          # all together indices
          batch_indices_Q <- list( indices_ )
          Qhat_value <- mean( unlist( lapply(batch_indices_Q, function(use_i){
            finalWts_ <- prop.table( strenv$np$array(  getProbRatio_tf(
                                                                 ModelList_ = ModelList_,
                                                                 Y_ = tfConst(as.matrix(Y[use_i])),
                                                                 X_ = tfConst(X[use_i,]),
                                                                 factorMat_ = tfConst(FactorsMat_numeric_0Indexed[use_i, , drop=FALSE],strenv$jnp$int32),
                                                                 logProb_ = tfConst(as.matrix(log_PrW[use_i])),
                                                                 REGULARIZATION_LAMBDA = tfConst(returnWeightsFxn(REGULARIZATION_LAMBDA)))  ) )
            Qhat_ <- sum(as.matrix(Y[use_i])*finalWts_)
          } )))
          return( Qhat_value )
      }

      Qhat_inSamp <- getQ_fxn(  ModelList_object, availableTrainIndices  )
      Qhat_hold  <- getQ_fxn(  ModelList_object, holdIndices  )
      if(trainIndicator == 1){
        print("---Current Q: IN, OUT---");print(c(Qhat_inSamp,Qhat_hold))
        performance_matrix_out[fold_,lambda_counter] <- Qhat_hold
        performance_matrix_in[fold_,lambda_counter] <- Qhat_inSamp
        print( performance_matrix_out )
      }
    }
  }
  if(trainIndicator == 1){
    lowerBound_vec <- colMeans(performance_matrix_out) - 1 * apply(performance_matrix_out,2,function(zer){ sqrt(1/length(zer)*var(zer)) })
    LAMBDA_selected <- lambda_seq <- lambda_seq[which.max(lowerBound_vec)]
  }
}
}
