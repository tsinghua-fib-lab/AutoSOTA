# Deprecated archive: retained for reference only and not sourced by the active package.
ml_build <- function(){
  message("Begin building block...")
  Zero1by1 <- strenv$jnp$array(as.matrix(0.),strenv$jnp$float32)

  # test
  # setup main functions
  useHajekInOptimization_orig <- use_hajek
  if(penalty_type == "LogMaxProb"){
    RegularizationPiAction <- 'strenv$jnp$add(strenv$jnp$max(( (  (log_pidk) ) )),
    strenv$jnp$log(strenv$jnp$squeeze(strenv$jnp$divide(strenv$jnp$take(ClassProbsMarginal, k_ - 1L, axis = 1L),DFactors))))'
  }
  if(penalty_type == "L2"){
    ## if equal weighting of the penalty by cluster
    if(K == 1){ RegularizationPiAction <- 'strenv$jnp$sum(strenv$jnp$square( strenv$jnp$subtract( pd%s, strenv$jnp$exp(log_pidk) ) ))' }

    ## if weighting the penalty by the marginal probability for that factor
    # approach 1: (pi_z - p)^2 * pr(z)
    #if(K > 1){ RegularizationPiAction <- 'strenv$jnp$multiply(strenv$jnp$sum(strenv$jnp$square( strenv$jnp$subtract( pd%s, strenv$jnp$exp(log_pidk) ) )),
               #strenv$jnp$squeeze(strenv$jnp$take(ClassProbsMarginal, k_ - 1L, axis = 1L)))' }

    # approach 2: mean( ( (p_z-pi) * pr(z|x) )^2 )
    if(K > 1){ RegularizationPiAction <- 'strenv$jnp$sum(strenv$jnp$mean(strenv$jnp$square(strenv$jnp$multiply( strenv$jnp$subtract( pd%s, strenv$jnp$exp(log_pidk) ) ,
                                          strenv$jnp$expand_dims(strenv$jnp$take(ClassProbs, k_ - 1L, axis = 1L),0L) )),1L))' }

  }
  for(forMEst in c(T,F)){
    if(forMEst == T){ subtypes <- "probRatio"; use_hajek <- F; return_text <- 'returnThis_ <- minThis_;' }
    if(forMEst == F){ subtypes <- c("probRatio","objToMin") }
  for(subtype in subtypes){
    if(forMEst == F){
      use_hajek <- useHajekInOptimization_orig;
      if(subtype == "probRatio"){ return_text <- 'returnThis_ <- probRatio;' }
      if(subtype == "objToMin"){ return_text <- 'returnThis_ <- minThis_;' }
    }
    if( K > 1 ){
      regularizationContrib <- 'strenv$jnp$add(strenv$jnp$multiply(REGULARIZATION_LAMBDA, RegularizationContribPi),
                                                strenv$jnp$multiply(lambda_coef, strenv$jnp$mean(strenv$jnp$square(ClassProbProj$kernel))) )'
    }
    if( K == 1 ){
      regularizationContrib <- 'strenv$jnp$multiply(REGULARIZATION_LAMBDA, RegularizationContribPi)'
    }
    if(use_hajek == F){
      minThis_text <- sprintf('minThis_ <- strenv$jnp$add(strenv$jnp$negative(strenv$jnp$mean( strenv$jnp$multiply(Y_,probRatio) )), %s);',regularizationContrib)
    }
    if(use_hajek == T){
      minThis_text <- sprintf('minThis_ <- strenv$jnp$add(strenv$jnp$negative(strenv$jnp$sum( strenv$jnp$divide(strenv$jnp$multiply(Y_,probRatio),strenv$jnp$sum(probRatio)))), %s );',regularizationContrib)
    }
    # define main function
    {
      base_tf_function <- eval(parse(text = paste('myLoss_or_returnWeights <- (function(
                                                ModelList_, Y_, X_,factorMat_, logProb_,REGULARIZATION_LAMBDA){
          if(K == 1){
              logClassProbs <- strenv$jnp$zeros(strenv$jnp$shape(Y_),strenv$jnp$float32)
              # Define ClassProbsMarginal for K=1 (single cluster has probability 1.0)
              ClassProbsMarginal <- strenv$jnp$ones(list(1L, 1L), strenv$jnp$float32)
          }
          if(K >  1){
              ProjectionList_ <- ModelList_[[2]]
              logClassProbs <- strenv$jnp$log( ClassProbs <- getClassProb(ProjectionList_,  X_ )  )
              ClassProbsMarginal <- strenv$jnp$mean(ClassProbs, 0L, keepdims=T)
          }

          RegularizationContribPi <- strenv$jnp$zeros( list() )
          logPrWGivenAllClasses <- replicate(  K, list()   )

          # Approach:
          # Pr(W) = sum_k Pr(W, k) = sum_k Pr(W | K = k) Pr(K = k) = sum_k prod_d [Pr(W_d | K=k)] Pr(K=k)

          # iterate over clusters
          for(k_ in 1L:K){

            # log( prod_d [Pr(W_d|K=k)] Pr(K=k) ) =  [sum_d { log(Pr(W_d|K=k)) }] + log( Pr(K=k) )
            logPrW_given_k <- strenv$jnp$expand_dims(strenv$jnp$take(logClassProbs, indices = k_ - 1L, axis=1L),1L)

            # iterate over factors
            for(d__ in 1L:DFactors){
              eval(parse(text=sprintf("av%sk%s <- ModelList_[[1]][[%s]][[%s]]", d__, k_,  k_, d__)))
              log_pidk <- strenv$jax$nn$log_softmax(strenv$jnp$concatenate(list( Zero1by1,
                                         eval(parse(text=sprintf("av%sk%s",d__, k_))) ),0L),axis = 0L)
              received_wd <- strenv$jnp$take(factorMat_, indices = d__ - 1L, axis = 1L)
              log_PrWd_given_k <- strenv$jnp$take(log_pidk, indices = received_wd, axis = 0L)

              logPrW_given_k <- strenv$jnp$add(logPrW_given_k, log_PrWd_given_k)
              if (grepl("%s", RegularizationPiAction, fixed = TRUE)) {
                RegularizationContribPi <- RegularizationContribPi + eval(parse(text = sprintf(RegularizationPiAction, d__)))
              } else {
                RegularizationContribPi <- RegularizationContribPi + eval(parse(text = RegularizationPiAction))
              }
            }
            logPrWGivenAllClasses[[k_]] <- logPrW_given_k
          }
          logPrW_new <- strenv$jax$scipy$special$logsumexp(strenv$jnp$concatenate(logPrWGivenAllClasses, 1L),keepdims = T, axis =  1L)
          probRatio <- strenv$jnp$exp( strenv$jnp$subtract(logPrW_new,  logProb_ ));',
          minThis_text, return_text,'return(   returnThis_    )})', sep = "")))
    }
    if(forMEst == T){
      getLoss_tf_unnormalized_raw <- base_tf_function
      getLoss_tf_unnormalized <- strenv$jax$jit(getLoss_tf_unnormalized_raw)
    }
    if(forMEst == F){
      if(subtype == "probRatio"){
        getProbRatio_tf_raw <- base_tf_function
        getProbRatio_tf <- strenv$jax$jit(getProbRatio_tf_raw)
      }
      if(subtype == "objToMin"){
        getLoss_tf_raw <- base_tf_function
        getLoss_tf <- strenv$jax$jit(getLoss_tf_raw)
      }
    }
  }
  }

## start  building model
{
  # define the class prob projection, the trainable vars, the baseline probs
  KFreeProbProj <- ai(K - 1L)
  b_proj <- 0.1
  ProjectionList <- list(strenv$jnp$array(0.))
  if(KFreeProbProj > 0){
    value_seq <- sapply(b_seq <- 10^seq(-10,1,length.out=100),function(b_){
       value_ <- mean(replicate(25,{
        X_ <- X[sample(1:nrow(X),batch_size),]
        #p_ <- X_ %*% matrix(rnorm(ncol(X_)*KFreeProbProj,mean=0,sd=b_),ncol=KFreeProbProj)
        p_ <- X_ %*% matrix(runif(ncol(X_)*KFreeProbProj,-b_,b_), ncol=KFreeProbProj)
        penalty_val <- mean(apply((apply(p_,1,function(zer){ prop.table(exp(c(0,zer))) })),1,sd))
        penalty_val
      }))
  })
    #value_seq <- value_seq - 1 / K
    b_proj <- b_seq[which.min(abs(value_seq - CLUSTER_EPSILON))]
    message(sprintf("Uniform init bounds for kernel: %.5f",b_proj))

    # projection + bias arrays
    ClassProbKernel <- strenv$jnp$array(matrix(KFreeProbProj*ncol(X), rnorm(KFreeProbProj*ncol(X),
                                                                     mean=0, sd = 1/sqrt(KFreeProbProj)),
                                       ncol = KFreeProbProj), strenv$jnp$float32)
    ClassProbBias <- strenv$jnp$array(t(rep(0,times = KFreeProbProj)), strenv$jnp$float32)

    # initialize
    {
      getClassProb <- strenv$jax$jit(function(projectionList, x){
        ClassProbKernel <- projectionList[[1]][[1]]
        ClassProbBias <- projectionList[[1]][[2]]
        x <- strenv$jnp$add( strenv$jnp$matmul(ClassProbKernel, x), ClassProbBias)
        return(  strenv$jax$nn$softmax( strenv$jnp$concatenate( list(strenv$jnp$zeros(c(nrow = strenv$jnp$shape(x)[[1]],1L),
                                                                        dtype=strenv$jnp$float32),x),1L ) ,1L ) ) })
    }
    projectionList <- list(list(ClassProbKernel,ClassProbBias))
    message(paste("Initial Class Probs: ", paste(round(colMeans(as.array(getClassProb(projList,
                                                                                    strenv$jnp$array(X,strenv$jnp$float32)))),7L),collapse=','),sep=""))
  }

  # initialize pd -  assignment probabilities for penalty and pr(w)
    DFactors <- length( nLevels_vec <- apply(W,2,function(l){length(unique(l))}) )
    for(d_ in 1:(DFactors <- length(nLevels_vec))){
        eval(parse(text=sprintf( "pd%s = strenv$jnp$array(as.matrix(p_list[[d_]]), dtype = strenv$jnp$float32)",d_)) )
    }

  # initialize av- the new probability generators
  initialize_avs <- function(b_SCALE=1){
    AVSList <- replicate(K, list(replicate(DFactors,list())))
    for(k_ in 1:K){ for(d_ in 1:DFactors){
          # Access pi_init_list as nested list: [[fold]][[k]][[d]]
          pi_init_elem <- pi_init_list[[1L]][[k_]][[d_]]
          b_ <- f2n(attr(pi_init_elem,"random_"))[1]
          INIT_systemic <- as.matrix(f2n(attr(pi_init_elem,"sys_")))
          INIT_random <- matrix(runif(length(pi_init_elem),
                                min = -b_SCALE*b_, max = b_SCALE*b_),
                                ncol = dim(pi_init_elem)[2])
          INIT_VALUE <- INIT_systemic + INIT_random
          if(!sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
            eval(parse(text=sprintf( "AVSList[[k_]][[d_]] <-  av%sk%s <- strenv$jnp$array( INIT_VALUE, dtype = strenv$jnp$float32)",d_,k_)) )
            eval(parse(text = sprintf( "av%sk%s_ = function(){ av%sk%s }",d_,k_,d_,k_)))

            # assign to correct environment
            eval(parse(text=sprintf("assign('av%sk%s',av%sk%s, envir = evaluation_environment)",d_,k_,d_,k_)))
            eval(parse(text=sprintf("assign('av%sk%s_',av%sk%s_, envir = evaluation_environment)",d_,k_,d_,k_)))
          }
          #if(sprintf("av%sk%s",d_,k_) %in% ls(envir = evaluation_environment)){
            #eval(parse(text=sprintf( "av%sk%s$assign( INIT_VALUE )",d_,k_)) ) }
    } }
    eval(parse(text=sprintf("assign('AVSList',AVSList, envir = evaluation_environment)")))
  }

  # initialize environment
  environment(initialize_avs) <- evaluation_environment

  # initialize
  initialize_avs()

  getPiDk <- function(AiDk){
    pidk <- strenv$jax$nn$softmax(strenv$jnp$concatenate(list(strenv$jnp$array(as.matrix(0L)),
                                                AiDk ),0L),0L)
  }
  Jacobian_getPiDk <- strenv$jax$jit(strenv$jax$jacobian( getPiDk ))
  getPiList <- function(SimplexList, simplex=T,rename=T,return_SE = F,VarCov = NULL){
    # SimplexList is not on the simplex, but is projected to the simplex
    FinalSEList <- FinalProbList <- eval(parse(text=paste("list(",paste(rep("list(),",times=K-1),collapse=""), "list())",collapse="")))
    for(k_ in 1:K){
     for(d_ in 1:length(nLevels_vec)){
          pidk <- strenv$np$array( getPiDk( SimplexList[[k_]][[d_]] )  )
          #pidk[1] <- NA
         if(simplex == T){
           if(!is.null(VarCov) & return_SE == T){
              dpidk_davdk <- strenv$np$array(strenv$jnp$squeeze(strenv$jnp$squeeze(
                Jacobian_getPiDk( SimplexList[[k_]][[d_]] ),1L), 2L))

              VarCov_subset <- grep(row.names(VarCov),
                              pattern = sprintf("k%sav%sd",k_, d_) )
              VarCov_subset <- as.matrix(VarCov[VarCov_subset,VarCov_subset]) # CHECK ORDER IN Ld > 2 case!!
              # check for lexiconographical order 
              VarCov_transformation <- try((dpidk_davdk) %*% VarCov_subset %*% t(dpidk_davdk), T)
              pidk_se <- sqrt(diag(VarCov_transformation))
           }
           pidk <- strenv$np$array(pidk)
         }

        if(return_SE == T){
          FinalSEList[[k_]][[d_]] <- as.numeric(pidk_se)
        }
        FinalProbList[[k_]][[d_]] <- as.numeric(pidk)
        if(rename == F){
          names(FinalProbList[[k_]][[d_]]) <- paste(sprintf("av%sk%s:0_",d_,k_),1:(length(pidk)),sep="")
          names(FinalProbList[[k_]])[d_] <- paste("k",k_,sep="")

          if(return_SE == T){
            names(FinalSEList[[k_]][[d_]]) <- names(FinalProbList[[k_]][[d_]])
            names(FinalSEList[[k_]])[d_] <- names(FinalProbList[[k_]])[d_]
          }
        }
        if(rename == T){
          tmp_<-try(names(FinalProbList[[k_]][[d_]]) <- names(p_list[[d_]]),T)
          names(FinalProbList[[k_]])[d_] <- names(p_list)[d_]

          if(return_SE == T){
            names(FinalSEList[[k_]][[d_]]) <- names(FinalProbList[[k_]][[d_]])
            names(FinalSEList[[k_]])[d_] <- names(FinalProbList[[k_]])[d_]
          }
        }
      }
    }
    RET <- FinalProbList
    if(return_SE == T){RET <- list("FinalProbList" = FinalProbList, "FinalSEList" = FinalSEList ) }
    return( RET )
  }

  # initial forward pass (with small batch size for speed)
  returnWeightsFxn <- c; regLambdaFxn <- c
  tfConst <- function(x, ty=strenv$jnp$float32){  strenv$jnp$array(x,dtype=ty)  }

  # initial forward pass
  my_batch <- sample(1:length(Y), batch_size,replace=F)
  tmp_loss <- getProbRatio_tf(
                         ModelList_ = list(AVSList, ProjectionList),
                         Y_ = tfConst(as.matrix(Y[my_batch])),
                         X_ = tfConst(X[my_batch,]),
                         factorMat_ = tfConst(FactorsMat_numeric_0Indexed[my_batch, , drop=FALSE],strenv$jnp$int32),
                         logProb_ = tfConst(as.matrix(log_PrW[my_batch])),
                         REGULARIZATION_LAMBDA = tfConst(returnWeightsFxn(lambda_seq[1])))

  gradient_getLoss_tf <- strenv$jax$jit(strenv$jax$grad(getLoss_tf_raw))

  # test gradient
  ModelList_object <- list(AVSList, ProjectionList)
  gradient_init <- gradient_getLoss_tf(
              ModelList_object, # ModelList_ =
              tfConst(as.matrix(Y[my_batch])), # Y_ =
              tfConst(X[my_batch,]), # X_ =
              tfConst(FactorsMat_numeric_0Indexed[my_batch, , drop=FALSE],strenv$jnp$int32), # factorMat_ =
              tfConst(as.matrix(log_PrW[my_batch])), # logProb_ =
              tfConst(returnWeightsFxn(lambda_seq[1])) # REGULARIZATION_LAMBDA =
             )
}

}
