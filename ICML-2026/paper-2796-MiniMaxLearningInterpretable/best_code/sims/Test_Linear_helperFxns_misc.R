getInteractionWts <- function(ze){
  ze_t <- ze
  c(ze_t,ze_t[KChoose2_combs[1,]]*ze_t[KChoose2_combs[2,]])
}
getMaxLogProb = function(theta_){
  maxProb_support = apply(cbind(theta_*1, 1-theta_),1,max)
  maxLogProbValue = sum(log(maxProb_support))
  return( maxLogProbValue )
}
getQ = function(wting){
  if(length(wting) != length(my_beta)){print("getQ error1");browser()}
  if( any(wting < 0) | any(wting > 1) ){print("getQ error2");browser()}
  sum(my_beta*wting)
}
clip2 = function(ze){ze};trim_fxn = function(ze){ze}
L2norm <- function(ze){sqrt(sum((ze)^2))}
marginalVar <- function(beta){
  covFactor_a <- ((treatProb*(1-treatProb)+treatProb^2)*treatProb - treatProb^3)
  covFactor_b <- (treatProb*(1-treatProb)+treatProb^2) * treatProb^2 - treatProb^4
  varTerm <- sum((baselineWeightings*beta)^2)
  covTerm_a <- 2*(sum(beta[1:k_factors][KChoose2_combs[1,]]*beta[-c(1:k_factors)])+
                    sum(beta[1:k_factors][KChoose2_combs[2,]]*beta[-c(1:k_factors)]))*covFactor_a
  covTerm_b <- 2*(sum(beta[1:k_factors][KChoose2_combs[1,]]*beta[-c(1:k_factors)])+
                    sum(beta[1:k_factors][KChoose2_combs[2,]]*beta[-c(1:k_factors)]))*covFactor_b
  var_ <- varTerm + covTerm_a + sigma2
  return( var_ )
}
theoreticalVarBound <- function(theta__,openBrowser = F){
  theta__interacted <- getInteractionWts(theta__)
  if(openBrowser){browser()}
  if(T == F){
    X <- sapply(theta__,function(ze){ rbinom(100000,size=1,prob=ze) })
    X_inter <- apply(KChoose2_combs,2,function(ze){ X[,ze[1]] * X[,ze[2]] })
    Yobs_ = cbind(X,X_inter) %*% my_beta + rnorm(n=nrow(X), sd = sqrt( sigma2 ) )
    #mean(Yobs_);getQ(getInteractionWts(theta__) )
    #Var[Y] + E[y]^2 = E[Y^2]

    #cov(a W[1]*W[2], b W[1])
    #E(a W[1]*W[2] * b W[1]) - E(a W[1]*W[2]) E(b W[1])
    #ab E( W[1]* W[1])W[2] - ab E(W[1]*W[2]) E(W[1])
    #ab (t[1]*(1-t[1])+t[1]^2)*t[2] - ab t[1]^2 t[2]

    #cov(a W[1]*W[2], b W[1] W[3])
    #E(a W[1]*W[2] * b W[1] W[3]) - E(a W[1]*W[2]) E(b W[1] W[3])
    #ab (t[1]*(1-t[1])+t[1]^2)*t[2] t[3] - ab t[1]^2 t[2] t[3]

    #Var[W[1]] = E[W[1]^2] - E[W[1]]^2
    #E[W[1]^2] = Var[W[1]] + E[W[1]]^2
    if(T == F){
      t1 <- 0.5; t2<-0.5; t3 <- 0.5
      W1 <- rbinom(100000,size=1,prob=t1)
      W2 <- rbinom(100000,size=1,prob=t2)
      W3 <- rbinom(100000,size=1,prob=t3)
      cov(W1*W2,W1);(t1*(1-t1)+t1^2)*t2 - t1^2 * t2
      cov(W1*W2,W1*W3);(t1*(1-t1)+t1^2)*t2*t3 - t1^2 * t2*t3
      var(W1*W2);cov(W1*W2,W1*W2); (t1*t2)*(1-(t1*t2))
    }
  }
  # E[ (\sum f_ b_)^2 ] = E[ (\sum f_ b_) ]^2 + Var[ (\sum f_ b_)^2 ]
  mainE2 <-  (sum(my_beta * theta__interacted ))^2
  subVar <- sum(my_beta^2 * theta__interacted * (1-theta__interacted))
  subCov <- sum( apply(expand.grid(1:ncol(AllCombsIncludingMain),
                                   1:ncol(AllCombsIncludingMain)),1,function(er){
                                     i1 <- AllCombsIncludingMain[,er[[1]]]
                                     i2 <- AllCombsIncludingMain[,er[[2]]]
                                     countShared <- sum(unique(i1) %in% unique(i2))
                                     cov_ij <- 0
                                     if(countShared == 1 & !all(i1==i2) ){
                                       #ab (t[1]*(1-t[1])+t[1]^2)*t[2] t[3] - ab t[1]^2 t[2] t[3]
                                       a_ <- my_beta[er[[1]]]; b_ <-  my_beta[er[[2]]];
                                       is_shared <- unique( intersect( i1, i2 ));  is_non_shared <- unique( setdiff( i1, i2 ))
                                       theta_s <- theta__[is_shared]
                                       theta_ns <- theta__[is_shared]
                                       cov_ij <- a_*b_*( (theta_s*(1-theta_s) + theta_s^2) * prod(theta_ns) -
                                                           theta_s^2 * prod(theta_ns) )
                                     }
                                     return( cov_ij )
                                   }) )
  mainVar <- subVar + sigma2 #+ subCov  # (add sigma2 to account for upper bound)
  ECt2 =  mainE2 + mainVar #really, this is an upper bound for EY2
  logECt2 = log( ECt2 )
  Term1 <- (getMaxLogProb(theta__) + logTreatCombs + logMarginalSigma2 - logHypoN )
  Term2 <- (getMaxLogProb(theta__) + logTreatCombs + logECt2 - logHypoN)

  Q_var <- exp(matrixStats::logSumExp(c(Term1,Term2)))
  return( Q_var )
}

PrWGivenPi_fxn = function(doc_indices,pi_mat,terms_posterior,log_=F,
                             doc_indices_u = NULL,d_indices_u = NULL,documents_list_=NULL){
  if(is.null(doc_indices_u)){
    doc_indices_u = unlist(doc_indices,recursive = F)
    d_indices_u = unlist(sapply(1:length(doc_indices),
                                function(se){list(rep(se,length(doc_indices[[se]])))}))
  }
  row.names(terms_posterior) <- colnames( terms_posterior ) <- row.names(pi_mat) <- colnames(pi_mat) <- NULL
  terms_posterior = as.matrix( terms_posterior )
  pr_w_position_given_theta = try(colSums(terms_posterior[,doc_indices_u] * pi_mat[,d_indices_u]),T)
  pr_w_given_theta = c(tapply(1:length(d_indices_u),d_indices_u,function(indi_){ sum(log( pr_w_position_given_theta[indi_] ))}))
  return( pr_w_given_theta)
}

logSEBound_fxn = function(THETA__,INDICES_, DOC_INDICES_U, D_INDICES_U,
                          PI_MAT_INPUT,MARGINAL_BOUNDS,DOC_LIST,
                          MODAL_DOC_LEN,
                          TERMS_MAT_INPUT,LOG_TREATCOMBS,YOBS){
  logW = log(apply( term_mat * c(THETA__),2,sum))
  logW_counter = logW;logW_counter[] <- 0
  my_tmp = c();log_maxProb <- 0;for(fa in 1:MODAL_DOC_LEN){
    logW_counter_select = logW_counter[names(MARGINAL_BOUNDS)]
    which_max = which.max(logW[names(MARGINAL_BOUNDS)][logW_counter_select <= MARGINAL_BOUNDS])
    my_tmp = c(my_tmp, which_max)
    log_maxProb <- log_maxProb + logW[names(which_max)]
    logW_counter[names(which_max)] <- logW_counter[names(which_max)] + 1
  }
  THETA_MAT_ = PI_MAT_INPUT; THETA_MAT_[] <- THETA__
  NUM__ = PrWGivenPi_fxn(doc_indices     = DOC_LIST[INDICES_],
                         pi_mat = THETA_MAT_[,INDICES_],
                         terms_posterior = TERMS_MAT_INPUT,
                         doc_indices_u = DOC_INDICES_U,
                         d_indices_u = D_INDICES_U, log_=T)
  DENOM__ = PrWGivenPi_fxn(doc_indices     = DOC_LIST[INDICES_],
                           pi_mat = PI_MAT_INPUT[,INDICES_],
                           terms_posterior = TERMS_MAT_INPUT,
                           doc_indices_u = DOC_INDICES_U,
                           d_indices_u = D_INDICES_U, log_=T)
  MY_WTS__ = prop.table(exp(NUM__ - DENOM__  ))
  scaleFactor = sum(YOBS[INDICES_]^2 * MY_WTS__)
  upperBound_se_log = 0.5*(log(scaleFactor) + LOG_TREATCOMBS + log_maxProb - log(length(INDICES_)))
  return( upperBound_se_log )
}

hajekNorm = function(WTS, MAX){
  WTS <- WTS/sum(WTS);
  if(!is.null(MAX)){
  ENTROPY_VAL = entropy_fxn(WTS) / entropy_fxn(rep(1/length(WTS),times=length(WTS)))
  counter_ <- 0; go_on = F; while(go_on == F){
    counter_ <- counter_ + 1
    #go_on <- abs(max(WTS) - MAX) < 0.01
    go_on <- (MAX-ENTROPY_VAL) < 0.01
    if(is.na(go_on)){browser()}
    if(class(go_on) != "logical"){browser()}
    if(!go_on){
    #if(counter_ > 1000){go_on <- T}
      #WTS[WTS > MAX] <- MAX; WTS = WTS / sum( WTS )
      UPPER_B = log(counter_+1)/(counter_+1); WTS[WTS > UPPER_B] <- UPPER_B; WTS = WTS / sum( WTS )
      ENTROPY_VAL = entropy_fxn(WTS) / entropy_fxn(rep(1/length(WTS),times=length(WTS)))
    }
  }
  }
  return(  WTS )
}

toSimplex_constrain = function(comb_){
  comb_[comb_ < -20] <- -20; comb_[comb_ >  20] <- 20
  return( toSimplex(comb_) ) }

getPi_foldIn <- function(doc_words_NEW,beta_lda_OLD,pi_OLD){
  INIT_getPI_tmp = rowMeans(pi_OLD)
  pi_OLD_samp = pi_OLD[sample(1:nrow(pi_OLD),min(nrow(pi_OLD),100)),]
  INIT_getPI = rev(c(compositions::alr(t(rev(INIT_getPI_tmp)))))
  pr_topics_maxLike_res <- try(sapply(1:length(doc_words_NEW),function(ze){
    if(ze %% 100 == 0){print(sprintf("getFoldin: %i/%i",ze,length(doc_words_NEW)))}
    # see p. 84 https://digital.library.adelaide.edu.au/dspace/bitstream/2440/115169/2/Glenny2018_MPhil.pdf
    # max likelihood of words given propotion
    beta_lda_subset_ze = beta_lda_OLD[,doc_words_NEW[[ze]]]
    minThis_ = function(theta_){
      theta_ = toSimplex_constrain(c(0,theta_))
      prWords_ze = colSums(beta_lda_subset_ze*theta_)
      #minThis_value = sum(log(prWords_ze)) + 0.0001*klDiv(theta_,INIT_getPI_tmp)
      minThis_value = - sum(log(prWords_ze))# + .1*min(apply(pi_OLD_samp,2,function(pi_old_i){ klDiv(theta_,pi_old_i)}))
      return(minThis_value)#minimize negative of logLike plus penalty
    }
    #pr_topics_maxLike = toSimplex_constrain(c(0,optim(par = INIT_getPI,minThis_,method = "Nelder-Mead")$par))
    #pr_topics_maxLike = toSimplex_constrain(c(0,tmpBest <- Rsolnp::solnp(pars = INIT_getPI,minThis_)$par))
    pr_topics_maxLike = toSimplex_constrain(c(0,tmpBest <- optim(par = INIT_getPI,minThis_)$par))
  }),T)
  if(class(pr_topics_maxLike_res) == "try-error"){browser()}
  return( pr_topics_maxLike_res )
}

myFNN = function(data = NULL, query = NULL, k = 1){
     data_t = as.matrix(t(data))
     nn.index = apply(query,1,function(ze){
        dist2_ze = colSums((data_t - ze)^2)
        which.min(dist2_ze)})
     return(list(nn.index=nn.index))
}

klDiv =function(p_,q_){sum(log(p_/q_)*p_) }#E_p[log p(x)/q(x)].

probGen <- function(piTrain,betaTrain,docWordsTrain,docWordsTest){
  # get out of sample pi's
  {
    piTest = sapply(1:length(docWordsTest),function(ze){
      ze_mat = betaTrain[,docWordsTest[[ze]]]
      if(!class(ze_mat)%in% c("matrix","data.frame")){ze_mat=as.matrix(ze_mat)}
      rowMeans(apply(ze_mat,2,prop.table))})
  }
  PrWGivenPi_train = PrWGivenPi_fxn(doc_indices     = docWordsTrain,
                                    pi_mat          = piTrain,
                                    terms_posterior = betaTrain,log_=T)
  PrWGivenPi_test = PrWGivenPi_fxn(doc_indices     = docWordsTest,
                                   pi_mat          = piTest,
                                   terms_posterior = betaTrain,log_=T)
  return(list(PrWGivenPi_train=PrWGivenPi_train,
              PrWGivenPi_test=PrWGivenPi_test))
}

scale2 = function(ze,scaleRef){
  mean_ = colMeans(scaleRef,na.rm=T)
  sd_ = apply(scaleRef, 2, function(az){sd(az,na.rm=T)})
  sd_[sd_==0] <- 1
  t((t(ze)-mean_)/sd_)}

runLDA <- function(wordList,K=3,allWords, epBeta=10e-10){
  wordList = lapply(wordList,function(ze){as.character(ze)})
  keyList = (table(unlist(wordList)))
  targetVal = quantile(c(keyList)[c(keyList)>0],prob=0.9)
  keyList = ( sapply( names(sample(keyList[keyList>=targetVal],2)),list ) )
  if(is.null(allWords)){allWords = sort(unique(unlist(wordList)))}
  VocabSize = length(allWords)
  alpha_prior = 3/K; scale_beta = 100/VocabSize
  ok_ = F;while(ok_ == F){
    print( ok_ )
    #keyATM_input_ = keyATM::keyATM_read(data.frame(text=do.call(rbind,lapply(wordList,function(ze){paste(ze,collapse=" ")})) ,stringsAsFactors = F),check=F)
    keyATM_input = data.frame(text = do.call(rbind,lapply(wordList,function(ze){ paste(ze,collapse=" ")})),stringsAsFactors = F)
    keyATM_input = keyATM::keyATM_read(keyATM_input,check=T)
    lda_posterior = try(weightedLDA(keyATM_input, model="base",
                                     number_of_topics = K,
                                     options = list(use_weights    = F,
                                                    estimate_alpha = T,
                                                    prune          = F,
                                                    store_theta    = F,
                                                    iterations     = 1000),
                                     priors = list(alpha=rep(alpha_prior,K),
                                                   beta = scale_beta)),T)


    if(class(lda_posterior) != "try-error"){
      beta_lda_temp = try(lda_posterior$phi[,order((colnames(lda_posterior$phi)))],T)
      if(!is.na(sum(beta_lda_temp))){ok_ = T}
    }
  }
  #process
  {
    beta_lda_temp = try(lda_posterior$phi[,order((colnames(lda_posterior$phi)))],T)
    beta_lda <- matrix(NA,nrow = K,ncol=length(allWords)); colnames(beta_lda) <- allWords
    try_ = try(min(beta_lda_temp)/20,T)
    beta_lda[] <- try_
    beta_lda[,colnames(beta_lda_temp)] <- beta_lda_temp
    beta_lda = beta_lda/rowSums(beta_lda)
    ok___ = T;while(ok___==F){
      beta_lda = beta_lda + min(beta_lda)*0.1
      beta_lda = beta_lda/rowSums(beta_lda)
      print(min(c(beta_lda)))
      okTest = try(min(c(beta_lda)) >= epBeta,T)
      if(class(okTest) == "try-error"){browser()}
      if(okTest){ok___ = T}
    }
    if(any(is.na(beta_lda))){browser()}
    pi_lda = try(t(lda_posterior$theta),T)
    return(list(beta_lda=beta_lda,pi_lda=pi_lda))
  }
}

genBeta_simple <- function(targetDist,BiasType="NoBias", L=1238){
  ok_ = F;while(ok_ == F){
  SEQ = matrix(rnorm(L*K),nrow=K)
  {
    beta_SPROUT = matrix(SEQ,nrow=K)
    beta_SPROUT = t(apply(beta_SPROUT,1,function(ze){
      factor_ <- 10^(seq(-5,6,length.out=1000))
      unif_cdf = cumsum(rep(1/L,L))
      my_objective_fxn = sapply(factor_,function(fact_){
        obj_val = abs(entropy_fxn(prop.table(exp(ze*fact_)))/entropy_fxn(rep(1/L,L)) - targetDist)
        #thisCDF = cumsum(sort(prop.table(exp(ze*fact_)))); obj_val = abs(max( abs(thisCDF - unif_cdf)) - targetDist)
        obj_val
      } )
      optimalValue = factor_[which.min(my_objective_fxn)]
      new_distrib = prop.table(exp(ze*optimalValue))
      entropy_ratio = entropy_fxn(new_distrib)/entropy_fxn(rep(1/L,L))

      if(runif(1)<0.01){par(mfrow =c(1,1));
        print(sprintf("Obtained entropy ratio: %s",entropy_ratio))
        plot(cumsum(sort(new_distrib)),
             main = sprintf("Dimensionality = %.0f",L),
             cex = 2,cex.lab = 2.5,cex.main = 2,
             xlab = "Word Index",ylab = "Cumulative Probability",type = "l")
        points( cumsum(rep(1/length(new_distrib),length(new_distrib))),type = "l", lty = 3,col="darkgray")
        }
      new_distrib
    } ))
    myCor_ = cor(t(beta_SPROUT))   ; diag(myCor_) <- 0;
    if(max(abs(myCor_))<0.1){ok_=T}
  }
  }
  colnames(beta_SPROUT) <- 1:ncol(beta_SPROUT)
  return(beta_SPROUT)
}

scaleCoefs_fxn <- function(BETA_MAT_, targetEffect,targetMag,initCoef=NULL){
  L = ncol(BETA_MAT_)
  outerMinBest <- Inf; scaler_best = Inf;
  starting_place_ = (-2)

  if(is.null(initCoef)){frac1 = 0.2;coef_t = (c(rep(1,ceiling(frac1*L)),rep(0,ceiling((1-frac1)*L))))[1:L]}
  if(!is.null(initCoef)){coef_t = initCoef}

  min_this <- function(xa){return( log(abs( abs(QComp(coef_t_=coef_t*xa,
                                                      beta_mat_ = BETA_MAT_,
                                                      theta_ = theta1)) - targetEffect )  ))  }
  SCALER_VEC = 10^(seq(-2,1,length.out = 10))
  SCALER_VEC = SCALER_VEC[SCALER_VEC>0]
  best_i = Inf;best_scale = 0;for( i in 1:length(SCALER_VEC)){
    print(i)
    val_i = min_this(SCALER_VEC[i])
    if(val_i < best_i){best_i = val_i;best_scale = SCALER_VEC[i]  }
  }
  coef_t_best = best_scale * coef_t
  scaler_ = best_scale
  obtained_Q = QComp(coef_t_best,BETA_MAT_,theta1)
  if(obtained_Q<0){ coef_t_best = coef_t_best * -1}
  print(  sprintf("Treat Effect Value: %s", QComp(coef_t_best, BETA_MAT_, theta1) )  )
  coef_t <- coef_t_best
  return( list(coef_t        = coef_t,
               optimalScaler = scaler_))
}
scaleUpCoef <- function(coef_SEED){
  rep_per_word <- L/length(coef_SEED)
  coef_sprout = c(do.call(cbind,sapply(coef_SEED,function(ze){
    list(replicate(rep_per_word,ze)) })))
}
scaleUpBeta <- function(beta_SEED,targetDist = 0.99){
  rep_per_word <- L/ncol(beta_SEED)
  beta_SPROUT = do.call(cbind,do.call(cbind,apply(beta_SEED,2,function(ze){
    list(replicate(rep_per_word,ze)) }))[1,])
  beta_SPROUT = t(apply(beta_SPROUT,1,prop.table))
  colnames(beta_SPROUT) <- 1:ncol(beta_SPROUT)
  if(is.na(targetDist)){
    beta_SPROUT = beta_SPROUT / rowSums(beta_SPROUT)
  }
  if(!is.na(targetDist)){
    beta_SPROUT = t(apply(beta_SPROUT,1,function(ze){
      factor_ <- seq(0.0000001,L*3,length.out=10000)
      my_objective_fxn = sapply(factor_,function(fact_){
        abs(entropy_fxn(prop.table(exp(ze*fact_)))/entropy_fxn(rep(1/L,L)) - targetDist)
        #abs(entropy_fxn(prop.table(ze^fact_))/entropy_fxn(rep(1/L,L)) - targetDist)
      } )
      optimalValue = factor_[which.min(my_objective_fxn)]
      print(sprintf("Optimal Beta Scaling Factor for Dim %i, %s: %.4f",L,BiasType,optimalValue))
      new = prop.table(exp(ze*optimalValue))
      #prop.table(ze^optimalValue)
    } ))
  }
  return( beta_SPROUT )
}

entropy_fxn <- function(prob_){
  prob_ = prop.table(prob_)
  prob_[prob_<1e-50]<- 1e-50;prob_ = prop.table(prob_)
  -sum(prob_*log(prob_))}

toLDA <- function(x){
  myReturn = apply(x, 1, function(row_i){
    wordIndex = which(row_i!=0)
    #my_tab = table( which(row_i!=0)-1 )
    myMat = matrix(NA,nrow=2,ncol=length(wordIndex))
    myMat[1,] <- wordIndex-1
    myMat[2,] <- row_i[wordIndex]
    #my_tab = rbind(f2n(names(my_tab)),my_tab)
    #row.names(my_tab) <- colnames(my_tab) <- NULL
    class(myMat) <- "integer"
    return( list(myMat) )
  })
  myReturn = unlist(myReturn,recursive=F)
  return( myReturn )
}

SmoothSimplex = function(ax,smooth_v=0.001,targ_min=0.01){
  which_dim = which.max(dim(ax))
  oka_ = min(apply(ax,2,min)) >= targ_min
  while(oka_ == F){
    ax = ax + smooth_v / min(dim(ax))
    if(which_dim == 1){ ax = ax / rowSums(ax) }
    if(which_dim == 2){ ax = t(t(ax) / rowSums(t(ax))) }
    oka_ = min(apply(ax,2,min)) >= targ_min
    }
  return(ax)
}

SRatFxn = function(NUM_,DENOM_,maxWt = 1e2){
  la_seq = seq(0,1,length.out=100)
  mean_denom = mean(DENOM_)
  la_pen = sapply(la_seq,function(la_){
    DENOM_s = ((1-la_)*DENOM_ + la_ * DENOM_/mean_denom)
    NUM_s = ((1-la_)*NUM_ + la_ * NUM_/mean_denom)
    wts_ = exp(NUM_s - DENOM_s)
    max(wts_,na.rm=T)
  })
  la_ = min(la_seq[min(which(la_pen<maxWt))])
  if(length(la_) == 0){la_ = 1}
  DENOM_s = ((1-la_)*DENOM_ + la_ * DENOM_/mean_denom)
  NUM_s = ((1-la_)*NUM_ + la_ * NUM_/mean_denom)
  wts_sm = exp(NUM_s - DENOM_s)
  return(wts_sm)
}

cleanText_fxn = function(x){
  x = tolower(x)
  #x <- gsub(x, pattern = "[[:digit:]]+", replace = " <number> ")
  x <- gsub(x, pattern = "[[:digit:]]+", replace = "")
  x <- gsub(x, pattern = "e\\-mail", replace = "email")
  x <- gsub(x, pattern = "e mail", replace = "email")
  x <- gsub(x, pattern = ":", replace = " ")
  x <- gsub(x, pattern = ";", replace = " ")
  x = gsub(x, pattern = "\\(", replace = " ")
  x = gsub(x, pattern = "\\)", replace = " ")
  x <- gsub(x, pattern = "\\’", replace = "'")
  x <- gsub(x, pattern = "\\'s", replace = "")
  x <- gsub(x, pattern = "n\\'t ", replace = " not ")
  x <- gsub(x, pattern = "\\'d ", replace = " would ")
  x <- gsub(x, pattern = "\\'re ", replace = " are ")
  x <- gsub(x, pattern = "\\'ve ", replace = " have ")
  x <- gsub(x, pattern = "\\'ll ", replace = " will ")
  x <- gsub(x, pattern = "\\'m ", replace = " am ")
  x <- gsub(x, pattern = "\\!", replace = " <exclpt> ")
  x <- gsub(x, pattern = "\\?", replace = " ")
  x <- gsub(x, pattern = "\\$", replace = " <dollar> ")
  x <- gsub(x, pattern = "\\.", replace = "")
  x <- gsub(x, pattern = "\\,", replace = "")
  x <- gsub(x, pattern = "\\-", replace = " ")
  x <- gsub(x, pattern = "\\/", replace = " ")
  x <- gsub(x, pattern = "\\—", replace = " ")
  x <- gsub(x, pattern = "\\’", replace = "")
  x <- gsub(x, pattern = "\\'", replace = "")
  x <- gsub(x, pattern = '\\"', replace = "")
  x <- gsub(x, pattern = "\\[applause\\]", replace = "")
  x <- gsub(x, pattern = "\\[laughter\\]", replace = "")
  x <- gsub(x, pattern = "\\”", replace = "")
  x <- gsub(x, pattern = "\\“", replace = "")
  x <- gsub(x, pattern = "\\-", replace = " ")
  x <- gsub(x, pattern = "\\…", replace = " ")
  x <- gsub(x, pattern = "  ", replace = " ")

  return(x)
}

purify <- function(mat_,fillout_){
  mat_[] <- fillout_
  return( mat_ )
}

WtGen = function(pr_num,pr_den,trim_q,dig_keep=20){
  #wts_raw <- Rmpfr::mpfr(pr_num,dig_keep)/Rmpfr::mpfr(pr_den,dig_keep)
  wts_raw <- pr_num/pr_den
  as.numeric(trim_fxn(wts_raw,trim_q))
}

CheckSupport = function(theta_mat, theta0, theta1=NULL,printOut=T,Yobs=NULL,top=3){
  theta1_match = NULL
  theta0 = unlist(theta0)
  theta1 = unlist(theta1)
  #distFx = function(theMat,thet){colSums(abs(theMat - thet),na.rm=T)}
  #distFx = function(theMat,thet){colSums(abs(theMat - thet),na.rm=T)}
  #distFx = function(theMat,thet){colSums(thet * log(thet/theMat),na.rm=T)}
  distFx = function(theMat,thet){#EMD
    OnesMat = matrix(1,nrow = length(thet),ncol = length(thet))
    diag(OnesMat) <- 0
    apply(theMat,2,function(ze){
    Barycenter::Greenkhorn(r = as.matrix(ze), c = as.matrix(thet),
                           costm = OnesMat)$Distance })
  }
  #distFx = function(theMat,thet){colSums(theMat * log(theMat/thet),na.rm=T)}
  #distFx = function(theMat,thet){-colSums(thet * log(theMat),na.rm=T)}

  #mean(Yobs[which(dists0==min(dists0))])

  theta_unique = !duplicated(apply(t(theta_mat),1,function(ae){paste(ae,collapse="_") }))
  unique_theta = t(theta_mat)[theta_unique,]
  theta0_m <- t(unique_theta); theta0_m[] <- theta0; theta0_m <- t(theta0_m)

  devtools::load_all("~/Downloads/WhatIf")
  inHull_theta1 <- inHull_theta0 <- NA
  if(!is.null(theta0)){inHull_theta0 = WhatIf::whatif(data = t(theta_mat),cfact = t(theta0))$in.hull }
  if(!is.null(theta1)){inHull_theta1 = WhatIf::whatif(data = t(theta_mat),cfact = t(theta1))$in.hull }

  dists0 = distFx(theta_mat,theta0)
  dists0_cut = gtools::quantcut(dists0,q = 10)
  theta0_match = theta_mat[,which.min(dists0)]
  theta0_match = t(theta_mat[,!duplicated(dists0)][,order(unique(dists0),decreasing = F)[1:top]])
  meanY0_match = sapply(sort(unique(dists0))[1:top],function(top__)
                          mean( Yobs[which(dists0 == top__)]))

  SummarizeQuantiles = as.data.frame( do.call(rbind,tapply(1:length(dists0_cut),dists0_cut,
                       function(indices_){
                      c( "meanDist_quantiles"=mean( dists0[indices_]),
                         "meanY_quantiles"=mean( Yobs[indices_]))})) )
  #theta0_match = theta_mat[,order(dists0,decreasing = F)[1:top]]
  if(printOut){
    print(paste("Control: " ,
                paste(round(theta0,2),collapse=", "),
                sep = ""))
    print(paste("Match C: " ,
                paste(round(theta0_match,2),collapse=", "),
                sep = ""))
  }
  if(!is.null(theta1)){
  dists1 = distFx(theta_mat,theta1)
  theta1_match = theta_mat[,which.min(dists1)]
  if(printOut){
    print("----------------------")
    print(paste("Treatment: " ,
                paste(round(theta1,2),collapse=", "),
                sep = ""))
    print(paste("Closest T: " ,
                paste(round(theta_mat[,which.min(colSums(abs(theta_mat - theta1),na.rm=T))],2),collapse=", "),
                sep = ""))
  }
  }
  if(!printOut){
    return( list(theta0_match=cbind(theta0_match,meanY0_match),
         theta0_dist=min(dists0),#sum(abs(theta0_match-theta0)),
         inHull_theta0 = inHull_theta0,
         inHull_theta1 = inHull_theta1,
         SummarizeQuantiles = SummarizeQuantiles,
         #dists0[order(unique(dists0),decreasing = F)[1:top],
         Yobs_min0 = mean(Yobs[which(dists0==min(dists0))]),
         theta1_match=theta1_match))
  }
}

postMeans_fxn = function(alpha_mat){ t(t(alpha_mat) / colSums(alpha_mat)) }

toDTM_f <- function(myText,threshold=0.01){
  myText_ = tokenizers::tokenize_word_stems(myText)
  #myText_ = lapply(myText_, function(x){unique(x)})
  myStems_tab = table(unlist(myText_))
  myStems_keep <- names( myStems_tab[myStems_tab > threshold * length(myText)] )
  myText_ = lapply(myText_, function(x){x[x %in% myStems_keep]})
  dfm_ = as.data.frame( matrix(0, nrow = length(myText), ncol = length(myStems_keep)) )
  colnames(dfm_) <- myStems_keep
  for(iaa in 1:length(myText_)){
    doc_i = myText_[[iaa]]
    doc_i = table(doc_i)
    dfm_[iaa,names(doc_i)] <- doc_i
  }
  return( dfm_ )
}


f2n=function(.){as.numeric(as.character(.))}
f2c=function(.){as.character(.)}
fillNAgaps <- function(x, firstBack=FALSE) {
  ## NA's in a vector or factor are replaced with last non-NA values
  ## If firstBack is TRUE, it will fill in leading NA's with the first
  ## non-NA value. If FALSE, it will not change leading NA's.

  # If it's a factor, store the level labels and convert to integer
  lvls <- NULL
  if (is.factor(x)) {
    lvls <- levels(x)
    x    <- as.integer(x)
  }

  goodIdx <- !is.na(x)

  # These are the non-NA values from x only
  # Add a leading NA or take the first good value, depending on firstBack
  if (firstBack)   goodVals <- c(x[goodIdx][1], x[goodIdx])
  else             goodVals <- c(NA,            x[goodIdx])

  # Fill the indices of the output vector with the indices pulled from
  # these offsets of goodVals. Add 1 to avoid indexing to zero.
  fillIdx <- cumsum(goodIdx)+1

  x <- goodVals[fillIdx]

  # If it was originally a factor, convert it back
  if (!is.null(lvls)) {
    x <- factor(x, levels=seq_along(lvls), labels=lvls)
  }

  x
}

f2n=function(.){as.numeric(as.character(.))}
f2c=function(.){as.character(.)}
trim_fxn = function(x,q_,sym=F){
  x_tr = x
  if(q_>1){
    summary_ = summary(x_tr)
    IQR = summary_[5]-summary_[2]
    upper_thres = summary_[5] + q_*IQR
    lower_thres = summary_[2] - q_*IQR
    x_tr[which(x_tr>upper_thres)] <- upper_thres
    if(sym){x_tr[which(x_tr<lower_thres)] <- lower_thres}
  }
  if(q_<1){
    quants_ = quantile(x_tr,c(1-q_,q_),na.rm=T)
    if(sym){x_tr[which(x_tr<quants_[1])] <- quants_[1]}
    x_tr[which(x_tr>quants_[2])] <- quants_[2]
  }
  return( x_tr)
}
normFxn = function(x){(x-mean(x,na.rm=T))/sd(x,na.rm=T)}
colSds <- function (x, center = NULL, dim. = dim(x)){ n <- dim.[1]; x <- x * x; x <- colMeans(x); x <- (x - center^2); sqrt (  x * (n/(n - 1)) )  }
FastScale <- function(x,cm=NULL,csd=NULL){
  if(is.null(cm)){cm = .colMeans(x, m = nrow(x), n = ncol(x))}# Get the column means
  if(is.null(csd)){csd = colSds(x, center = cm)}# Get the column sd
  return( t( (t(x) - cm) / csd ) )
}
dirichlet_prior_fxn = function(my_k){5/my_k}

diffFinder = function(x){
  same = rep(T,times=length(x))
  for(aj in 2:length(x)){
    same[aj] = (x[aj] != x[aj-1])
  }
  return( same )
}

toWlist = function(DTM_,forceUnigrams = F){ apply(DTM_,1,function(row_i){
  times_used = row_i[row_i>0]
  which_used = which(row_i>0)
  row_i = sapply(names(times_used),function(word_i){
    if(!forceUnigrams){va_ = rep(which_used[word_i],times = times_used[word_i])}
    if(forceUnigrams){va_ = rep(which_used[word_i],times = 1)}
    return( va_ )
  })
  unlist(row_i)
})}

wtPen = function(ea){
  return( quantile(ea,prob = 0.90,na.rm=T) )
}


