#!/usr/bin/env Rscript
# install.packages( "~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F);
# try(devtools::install_github( 'cjerzak/strategize-software/strategize',ref="main" ),T)
# https://chatgpt.com/share/67a16997-d684-800f-a352-08bafd4a3bad
{
{
# clear workspace and install necessary packages, load in helper functions
# check k > 10 case for lexoconigraphical ordering issues
rm(list=ls()); options(error = NULL)
use_RCE <- (Sys.info()["sysname"] != "Darwin");
c2p <- function(zer){zer[zer>1]<-1;zer[zer<0]<-0;zer}# clip to prob
f2n<-function(.){as.numeric(as.character(.))}
conda_env <- "strategize"

# set master parameters
COMMAND_ARG_INPUT <- try( as.integer((args <- commandArgs(TRUE))[1]), T)
penaltyType <- "L2"; indicesInDelta <- "1"
ConfLevel <- 0.95

# simulation grid
outer_grid <- expand.grid("kFactors" = (kFactors_seq <- c(5,10,20)),
                          "nObs" = nObs_seq <- c(1000L, 2000L, 5000L, 10000L),
                          
                          #"optimType" = (optimType_seq <- c("twoStep_default","oneStep")),
                          "optimType" = (optimType_seq <-  c("twoStep_default" )),
                          
                          #"case" = c("average", "adversarial"),
                          "case" = c("average"),
                          
                          "monte_i" = 1L:(nMonteCarlo <- 200L)) # set nMonteCarlo to 200L
# select adversarial with two step only and only do adversarial for k = 5
outer_grid <- outer_grid[!(outer_grid$case=="adversarial" & outer_grid$optimType == "oneStep"),]
outer_grid <- outer_grid[ !(outer_grid$case == "adversarial" & outer_grid$kFactors > 5), ]
outer_grid[outer_grid$case=="adversarial",]$nObs <- 2*(2500+outer_grid[outer_grid$case=="adversarial",]$nObs)
print(dim(outer_grid))

# set wd and source 
setwd('~/Dropbox/OptimizingSI/')
source("~/Dropbox/OptimizingSI/Analysis/Simulations/LinearCase_helperFxns_misc.R")

# optimization parameters
{
  GLOBAL_HAJEK_IN_OPTIMIZATION <- T
  GLOBAL_BATCH_SIZE <- 200
  GLOBAL_CLIP_AT <- 1000
  TWOSTEP_NSGD <- 1000L
  #ONESTEP_NSGD <- NULL; GLOBAL_NEPOCH <- 20
  ONESTEP_NSGD <- 200L; GLOBAL_NEPOCH <- NULL
  COEF_LAMBDA_GLOBAL <- 0
  GLOBAL_NFOLDS <- 1
  GLOBAL_KCLUST <- 1
  GLOBAL_MOMENTUM <- 0.0001
  GLOBAL_WARMSTART <- F; GLOBAL_SG_METHOD <- "adanorm"; LEARNING_RATE_BASE_GLOBAL <- 0.001;GLOBAL_CYCLE_WIDTH <- ceiling(1/5*ONESTEP_NSGD)
  ADAPTIVE_MOMENTUM_GLOBAL <- F
  GLOBAL_TESTFRAC <- 1
  kClust_iter <- 1
}

# set up some simulation parameters
{
  SEED_SCALER <- 10L
  set.seed(123456L*SEED_SCALER)
  useHajek = T
  doSE = T; sigma2 <- 0.1^2
  LinearR2_TARGET <- 0.75
  treatProb = 0.5;
  TARGETQ = 1
  LambdaScaling_seq <- 1 / sqrt( nObs_seq )
  keepLambdaFixed <- T # if FALSE, vary lambda_n
}

if(!is.na(COMMAND_ARG_INPUT)){
  OUTER_ITERATION_SEQUENCE <- COMMAND_ARG_INPUT
  print(sprintf("OUTER_ITERATION_SEQUENCE: %s",OUTER_ITERATION_SEQUENCE))
}
if(is.na(COMMAND_ARG_INPUT)){
  OUTER_ITERATION_SEQUENCE <- 1:nrow(outer_grid)
}
library( strategize )

for(out_ in OUTER_ITERATION_SEQUENCE){
    # assign analysis parameter values
    for(name_ in colnames(outer_grid)){eval(parse(text = sprintf("%s <- outer_grid[out_,name_]",name_))) }

    set.seed(  outer_seed <- SEED_SCALER*as.integer(as.numeric(kFactors^2) ))

    # some initial things that are good to know
    logTreatCombs = log(2^kFactors)
    AllCombsIncludingMain <- cbind(matrix(1:kFactors, ncol=kFactors,nrow=2,byrow=T),
                                   (KChoose2_combs <- combn(1:kFactors,m=2)))

    # first, fix true pi*
    if(case == "average"){
    atN_counter = 1; # ?
    findNewLambda <- keepLambdaFixed == F | atN_counter == 1
    if(findNewLambda == F){ LAMBDA_atN <- LAMBDA_master  }
    lambdaSearching <- findNewLambda;  while(lambdaSearching == T){
      baselineWeightings <- getInteractionWts(rep(0.5,times=kFactors))

      ok_ = F;while(ok_ == F){
        # scale the interactions to obtain target R2
        if(atN_counter == 1){
          beta_master = rnorm(max(kFactors_seq)+choose(max(kFactors_seq),2),sd=1)
          my_beta <- my_beta_orig <- beta_master[1:(kFactors+choose(kFactors,2))]

          X = matrix(rbinom(kFactors*10000,size = 1 , prob = treatProb), nrow =10000)
          X_inter <- apply(KChoose2_combs,2,function(ze){ X[,ze[1]] * X[,ze[2]] })
          NormalizedInteractionWts <- my_beta_orig[-(1:kFactors)] / L2norm(my_beta_orig[-(1:kFactors)])
          interactionWts_vec <- 10^seq(-5,5,length.out = 100)
          proposalR2_vec <- sapply(interactionWts_vec, function(ze){
            my_beta_orig[-(1:kFactors)] <- NormalizedInteractionWts*ze
            Yobs_ = cbind(X,X_inter) %*% my_beta_orig
            summary(lm(Yobs_~X))$adj.r.squared
          })
          discrep_proposalR2_from_targetR2 <- abs(proposalR2_vec-LinearR2_TARGET)
          my_beta_orig[-(1:kFactors)] <- NormalizedInteractionWts *
                    interactionWts_vec[ which.min( discrep_proposalR2_from_targetR2 )[1] ]
        }

        # select base lambda
        {
          # generate some data to calibrate
          X = matrix(rbinom(kFactors*10000,size = 1 , prob = treatProb),nrow =10000)
          X_inter <- apply(KChoose2_combs,2,function(ze){ X[,ze[1]] * X[,ze[2]] })
          Yobs = cbind(X,X_inter) %*% my_beta + rnorm(nrow(X), sd = sqrt( sigma2) )
          logMarginalSigma2 = c(log( var( Yobs ) ))#logMarginalSigma2 = log(  marginalVar(my_beta) )

          toSimplexFxn <- function(zer){
            #pi__  =  exp(zer) / (exp(0)+exp(zer))
            #pi__ <- rev(c(compositions::alrInv(as.matrix(rev(zer)))[,2]))
            pi__ <- c(f2n(compositions::ilrInv(as.matrix(zer))[,2]))
            return(  pi__  ) }
          theoreticalMinThisFxn <- function(ze,toSimplex=T){
            Pr1 <- ze
            if(toSimplex == T){ Pr1 <- toSimplexFxn(ze) }
            Q_ <- getQ( getInteractionWts( Pr1 ) )
            #E_w[ (E_y[\beta W])^2 ]
            #E_w[ (\sum_d \beta_d W_d)^2 ] =
            #E_w[ (\sum_d \beta_d W_d) ]^2 + Var_w[ (\sum_d \beta_d W_d) ]
            #Q_varTerm <- ifelse(penaltyType == "L2",yes = sum( (Pr1-pi_gen)^2 ) + sum( ((1-Pr1)-(1-pi_gen) )^2 ),
                                #no = theoreticalVarBound( Pr1 ))
            if(penaltyType == "LogMaxProb"){ minThis_ <-  -1*Q_ + LAMBDA_atN * getMaxLogProb(Pr1)}
            if(penaltyType == "L2"){minThis_ <-  -1*Q_ +
              LAMBDA_atN * sum( c( ( Pr1 - (pi_gen <- rep(0.5,times=kFactors)))^2 ,
                            ((1-Pr1)-(1-pi_gen) )^2) ) }
            return( minThis_ )
          }

          if(atN_counter == 1){ LambdaProposal_seq <- 10^(seq(-3,2,length.out = 500))  }
          if(atN_counter > 1){ LambdaProposal_seq <- LAMBDA_master*LambdaScaling_seq[atN_counter]  }
          SolAtLambdaProposal <- c(); for(LAMBDA_atN in LambdaProposal_seq){
            #pi_star <- optim(par = runif(kFactors,-0.01,0.01),fn = theoreticalMinThisFxn, method = "CG")$par
            # approximate solution to check analytical
            if(T == F){
              pi_star_optimization <- (pi_star_full<-Rsolnp::gosolnp(pars = runif(kFactors,-(runif_init <- 0.01),runif_init),
                        fun = theoreticalMinThisFxn,
                        distr = rep(3,times=kFactors), distr.opt = sapply(1:kFactors,function(ze){ list(list("mean"=0,"sd"=runif_init))  }), LB = rep(-10000000,times=kFactors),UB = rep(100000,times=kFactors),
                        n.restarts = 10, n.sim = 3,
                        control = list(tol=0.0000001,rho=NULL,delta=0.01,outer.iter = 300, inner.iter = 3000,trace=0) ))$pars
              plot(pi_star_analytical, toSimplexFxn(pi_star_optimization))
            }

            ##############################
            # analytical solution
            ##############################
            my_beta_main <- my_beta[1:kFactors]
            my_beta_inter <- my_beta[-c(1:kFactors)]
            zero_vec <- rep(0, times = kFactors)
            COEF_MAT <- sapply(1:kFactors,function(k_){
              zero_vec[k_] <- -4 * LAMBDA_atN
              interindices_ref <- which(KChoose2_combs[1,] == k_ |
                KChoose2_combs[2,] == k_)
              interindices <- sapply(interindices_ref,function(xer){
                KChoose2_combs[,xer][KChoose2_combs[,xer]!=k_]
              })
              zero_vec[interindices] <- my_beta_inter[interindices_ref]
              return(zero_vec)
            })
            B_VEC <- -4*LAMBDA_atN*rep(treatProb,times=kFactors) - my_beta_main
            pi_star_analytical <- pi_star <- c(solve(COEF_MAT,as.matrix(B_VEC)))
            SolAtLambdaProposal <- rbind(SolAtLambdaProposal,  pi_star)
          }
          whichSelected <- whichEligible <- which( apply(SolAtLambdaProposal,1,function(zer){
                            all(zer > 0.15)  & all(zer < 0.85) & sd(zer) > 0.1 }))
          if(length(whichEligible) > 1){
            impliedQ_vec <- apply(SolAtLambdaProposal[whichEligible,],1,function(zap){
                    impliedQ_ <- getQ( getInteractionWts(zap) ) })
            whichSelected <- whichEligible[ which.min(abs(impliedQ_vec-TARGETQ)) ]
          }
          pi_star_true <- SolAtLambdaProposal[whichSelected,]
          LAMBDA_atN <- LAMBDA_base <- LambdaProposal_seq[whichSelected]
          if(atN_counter == 1){ LAMBDA_master <- LAMBDA_base }
        }
        print(sprintf("True Q = %.3f", impliedQ <- getQ(getInteractionWts(pi_star_true))))
        ok_ <- T
      }

      # deal with solutions too close to simplex boundary or too flat
      problem_BoundaryViolate <- any(pi_star_true < 0.10) | any(pi_star_true > 0.90)
      problem_FlatViolate <- sd(pi_star_true) < 0.1
      lambdaSearching <- F
    }
    print(c(nObs,kFactors))
    trueQ <- getQ(getInteractionWts(pi_star_true))
    eval(parse(text=sprintf("pi_star_true%s=pi_star_true",kFactors)))
  
    # set seed
    set.seed( inner_seed <- SEED_SCALER*as.integer((kFactors*nObs*monte_i) ))

    # generate data
    X <- matrix(rbinom(kFactors*nObs,size = 1 , prob = treatProb),nrow =nObs)
    X_inter <- apply(KChoose2_combs,2,function(ze){ X[,ze[1]] * X[,ze[2]] })
    Yobs <- cbind(X,X_inter) %*% my_beta + rnorm(nObs, sd = sqrt( sigma2) )
    print(sprintf("Predicted Mar Var: %.3f, Observed Mar Var: %.3f", exp(logMarginalSigma2), var(Yobs)))

    # generate helper objects
    InitProbList_ <- replicate(ncol(X),{ list(c("0"=x0_<- 1-treatProb+rnorm(1,sd=0.0),"1"=1-x0_)) } )
    TrueProbList_ <- sapply(pi_star_true,function(ze_s){ list(c("0"=1-ze_s,"1"=ze_s)) })
    X_ <- apply(X,2,as.character)
    colnames(X_) <- 1:ncol(X_); names(InitProbList_) <- 1:ncol(X_)
    }
    
    competing_group_variable_respondent <-  respondent_task_id <- profile_order <- pair_id <- NULL
    competing_group_variable_candidate <- competing_group_competition_variable_candidate <- NULL 
    if(case == "adversarial"){
      
      case_ <- "experimental"
      if(case_ == "analytical"){
        ############################################################################
        # XXX: DATA GENERATION + ANALYTICAL SOLUTION
        {
          ############################################################################
          XXX
          ############################################################################
        }
      }
      
      if(case_ == "experimental"){ 
        # respondent info 
        competing_group_variable_respondent <- sample(c("Democrat","Republican"), size = nObs, replace = T, 
                                                      prob = c(Pr_a <- 0.6,
                                                               Pr_b <- (1-Pr_a)))
        competing_group_variable_respondent <- c(competing_group_variable_respondent,competing_group_variable_respondent)
        pair_id <- respondent_id <- c(1:nObs,1:nObs)
        profile_order <- c(rep(1,times=nObs), rep(2,times=nObs)) # assume Democrat/A is 1 and Rep/B is 2
        respondent_task_id <- rep(1, times = 2*nObs)
        
        # candidate info
        competing_group_variable_candidate <- sample(c("Democrat","Republican"), size = 2*nObs, replace = T, prob = c(0.5,0.5))
        competing_group_competition_variable_candidate <- ifelse(competing_group_variable_candidate[1:nObs]==competing_group_variable_candidate[-c(1:nObs)],
                                                                 yes = "Same", no = "Different")
        competing_group_competition_variable_candidate <- c(competing_group_competition_variable_candidate,competing_group_competition_variable_candidate)
        
        table(competing_group_variable_respondent,
              paste0(competing_group_variable_candidate, competing_group_competition_variable_candidate))/2
        
        # coefficients 
        set.seed(999); beta_master = rnorm(max(kFactors_seq)+choose(max(kFactors_seq),2),sd=1)
        
        # set seed
        set.seed( inner_seed <- SEED_SCALER*as.integer((kFactors*nObs*monte_i) ))
        
        # generate data
        my_beta <- my_beta_orig <- beta_master[1:(kFactors+choose(kFactors,2))]
        X <- matrix(rbinom(kFactors*nObs*2,size = 1 , prob = treatProb),nrow =2*nObs)
        X_inter <- apply(KChoose2_combs,2,function(ze){ X[,ze[1]] * X[,ze[2]] })
        Yobs <- cbind(X,X_inter) %*% my_beta + rnorm(nObs*2, sd = 0.5)
        Yobs <- 1*c(Yobs[1:nObs] > Yobs[-c(1:nObs)], Yobs[1:nObs] <= Yobs[-c(1:nObs)])
        l_a <- competing_group_variable_candidate == "Democrat"
        l_b <- competing_group_variable_candidate == "Republican"
        Yobs[] <- 0 
        # setup primary data
        # ap1
        Xcombined <- cbind(X[,1], c(X[((nrow(X)+1)/2):nrow(X),1],
                                    X[(1:(nrow(X)/2)),1] ))
        
        combMat <- expand.grid("C1_"=0:1, "C2_"=0:1,
                               "Competition_"=c("Same","Different"),
                               "CandidateGroup_"=c("Democrat","Republican"),
                               "RespondentGroup_"=c("Democrat","Republican")); jf_ <- 1
        for(jf_ in 1:nrow(combMat)){ 
          for(name_ in colnames(combMat)){eval(parse(text = sprintf("%s <- combMat[jf_,]$%s",name_, name_))) }
          l_stage <- competing_group_competition_variable_candidate == Competition_
          l_candidate_group <- competing_group_variable_candidate == CandidateGroup_ 
          l_respondent_group <- competing_group_variable_respondent == RespondentGroup_ 
          l_sttg  <- which(l_stage & l_candidate_group & l_respondent_group & Xcombined[,1]==C1_ & profile_order == 1)
          l_sttg_ <- which(l_stage & l_candidate_group & l_respondent_group & Xcombined[,1]==C2_ & profile_order == 1)
          Yobs[l_sttg] <- rbinom(length(l_sttg), size = 1, prob = (Pr_p00a <- 0.5)) # enforce 0.5 prob for same 
          Yobs[l_sttg_] <- 1 - Yobs[l_sttg_]
        }
        prop.table(table(Yobs))
        
        # ap2
        l_p11a  <- which(l_p & l_a  & Xcombined[,1]==1 & profile_order == 1)
        l_p11a_ <- which(l_p & l_a  & Xcombined[,2]==1 & profile_order == 2)
        Yobs[l_p11a] <- rbinom(length(l_p11a), size = 1, prob = (Pr_p11a <- 0.5)) # enforce 0.5 prob for same 
        Yobs[l_p11a_] <- 1 - Yobs[l_p11a]
        
        # ap3
        l_p10a  <- which(l_p & l_a  & X[,1]==1 & profile_order == 1)
        l_p10a_ <- which(l_p & l_a  & X[,1]==0 & profile_order == 2)
        Yobs[l_p10a] <- rbinom(length(l_p11a), size = 1, prob = (Pr_p10a <- 0.6)) # enforce 0.5 prob for same 
        Yobs[l_p10a_] <- 1 - Yobs[l_p10a_]
        
        # ap4
        l_p01a  <- which(l_p & l_a  & X[,1]==0 & profile_order == 1)
        l_p01a_ <- which(l_p & l_a  & X[,1]==1 & profile_order == 2)
        Yobs[l_p01a] <- rbinom(length(l_p01a), size = 1, prob = (Pr_p01a <- 1-Pr_p10a)) # enforce 0.5 prob for same 
        Yobs[l_p01a_] <- 1 - Yobs[l_p01a]
        
        Yobs[l_p00a] <- rbinom(length(l_p00a), size = 1, prob = (Pr_p0a <- 0.5)) # enforce 0.5 prob for same 
        Yobs[l_p00a_] <- 1 - Yobs[l_p00a_]
        
        Yobs[l_p1a] <- rbinom(length(l_p1a <- which(l_p & l_a  & X[,1]==1 & profile_order == 1)), 
                              size = 1, prob = (Pr_p1a <- 0.3))
        Yobs[l_p0b] <- rbinom(length(l_p0b <- which(l_p & l_b  & X[,1]==0 & profile_order == 2)), 
                              size = 1, prob = (Pr_p0b <- 0.7))
        Yobs[l_p1b] <- rbinom(length(l_p1b <- which(l_p & l_b  & X[,1]==1 & profile_order == 2)), 
                              size = 1, prob = (Pr_p1b <- 0.3))
        
        # setup generals data 
        Yobs[l_g00a] <- rbinom(length(l_g00a <- which(l_g & l_a  & X[,1]==0 & profile_order == 1)), 
                              size = 1, prob = (Pr_g00a <- 0.6))
        Yobs[l_g10a] <- rbinom(length(l_g1a <- which(l_g & l_a  & X[,1]==1 & profile_order == 1)), 
                              size = 1, prob = (Pr_g1a <- 0.5))
        
        # setup b's data 
        Yobs[l_g0b] <- rbinom(length(l_g0b <- which(l_g & l_b  & X[,1]==0 & profile_order == 2)), 
                              size = 1, prob = (Pr_g0b <- 0.5))
        Yobs[l_g1b] <- rbinom(length(l_g1b <- which(l_g & l_b  & X[,1]==1 & profile_order == 2)), 
                              size = 1, prob = (Pr_g1b <- 0.7))
        
        Pr_mat <- matrix(c(Pr_p0a,Pr_p0a,Pr_p1a,Pr_p1a, # Pr(A|)
                           Pr_p0b,Pr_p0b,Pr_p1b,Pr_p1b,
                           Pr_g0a,Pr_g0a,Pr_g1a,Pr_g1a,
                           Pr_g0b,Pr_g0b,Pr_g1b,Pr_g1a),
                           byrow = T, ncol = 4)
        Pr_A_win <- sum(Pr_mat[1,] * Pr_mat[3,])
        Pr_B_win <- Pr_mat[1,] * Pr_mat[2,]
      }
      if(case_ == "sanity"){ 
      # respondent info 
      competing_group_variable_respondent <- sample(c("Democrat","Republican"), size = nObs, replace = T, prob = c(0.5,0.5))
      competing_group_variable_respondent <- c(competing_group_variable_respondent,competing_group_variable_respondent)
      pair_id <- respondent_id <- c(1:nObs,1:nObs)
      profile_order <- c(rep(1,times=nObs), rep(2,times=nObs))
      respondent_task_id <- rep(1, times = 2*nObs)
      
      # candidate info
      competing_group_variable_candidate <- sample(c("Democrat","Republican"), size = 2*nObs, replace = T, prob = c(0.5,0.5))
      competing_group_competition_variable_candidate <- ifelse(competing_group_variable_candidate[1:nObs]==competing_group_variable_candidate[-c(1:nObs)],
                                                               yes = "Same", no = "Different")
      competing_group_competition_variable_candidate <- c(competing_group_competition_variable_candidate,competing_group_competition_variable_candidate)
      
      table(competing_group_variable_respondent,
            paste0(competing_group_variable_candidate, competing_group_competition_variable_candidate))/2
      
      # coefficients 
      set.seed(999); beta_master = rnorm(max(kFactors_seq)+choose(max(kFactors_seq),2),sd=1)
      
      # set seed
      set.seed( inner_seed <- SEED_SCALER*as.integer((kFactors*nObs*monte_i) ))
      
      # generate data
      my_beta <- my_beta_orig <- beta_master[1:(kFactors+choose(kFactors,2))]
      X <- matrix(rbinom(kFactors*nObs*2,size = 1 , prob = treatProb),nrow =2*nObs)
      X_inter <- apply(KChoose2_combs,2,function(ze){ X[,ze[1]] * X[,ze[2]] })
      Yobs <- cbind(X,X_inter) %*% my_beta + rnorm(nObs*2, sd = 0.5)
      Yobs <- 1*c(Yobs[1:nObs] > Yobs[-c(1:nObs)], 
                  Yobs[1:nObs] <= Yobs[-c(1:nObs)])
      }

      # generate helper objects
      InitProbList_ <- replicate(ncol(X),{ list(c("0"=x0_<- 1-treatProb+rnorm(1,sd=0.0),"1"=1-x0_)) } )
      X_ <- apply(X,2,as.character)
      colnames(X_) <- 1:ncol(X_); names(InitProbList_) <- 1:ncol(X_)
      LAMBDA_atN <- 2;
      
      # generate truth - not clear how to do this 
      TrueProbList_ <- InitProbList_# sapply(pi_star_true,function(ze_s){ list(c("0"=1-ze_s,"1"=ze_s)) })
    }

    #run analyses
    if(optimType == "oneStep"){
          time_elapsed <- system.time(Qoptimized  <-  try( strategize::strategize_onestep(
              automatic_scaling = F,
              penalty_type = penaltyType,
              log_PrW = rep( log(1/2^length(InitProbList_) ),length(Yobs)),
              lambda_seq = LAMBDA_atN,
              Y = Yobs,
              W = X_, # here, X refers to the randomized factors
              K = kClust_iter,
              adaptive_momentum = ADAPTIVE_MOMENTUM_GLOBAL,
              p_list = InitProbList_,
              nFolds = 1,
              test_fraction = GLOBAL_TESTFRAC,
              sg_method = GLOBAL_SG_METHOD,
              lambda_coef = COEF_LAMBDA_GLOBAL,
              nEpoch = GLOBAL_NEPOCH,
              nSGD = ONESTEP_NSGD,
              batch_size = GLOBAL_BATCH_SIZE,
              learning_rate_max = LEARNING_RATE_BASE_GLOBAL,
              use_hajek = GLOBAL_HAJEK_IN_OPTIMIZATION,
              warm_start = GLOBAL_WARMSTART,
              momentum = GLOBAL_MOMENTUM,
              clip_at  = GLOBAL_CLIP_AT,
              cycle_width  = GLOBAL_CYCLE_WIDTH,
              conda_env = conda_env,
              conda_env_required = T ),  T) )
        }
    if(grepl(optimType,pattern = "twoStep")){
          Qoptimized <- { strategize::strategize(
                           Y = Yobs,
                           W = X_,
                           lambda = LAMBDA_atN,
                           conda_env = conda_env,
                           conda_env_required = T,
                           
                           # adversarial parameters 
                           competing_group_variable_respondent = competing_group_variable_respondent,
                           competing_group_variable_candidate = competing_group_variable_candidate,
                           competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
                           pair_id = pair_id,
                           respondent_id = respondent_id,
                           respondent_task_id = respondent_id,
                           profile_order = profile_order,
                           diff = (case == "adversarial"),
                           adversarial = (case == "adversarial"),
                           nMonte_adversarial = 25L,
                           
                           # parameters
                           compute_se = (case == "average"),
                           penalty_type = "L2",
                           use_regularization  = F,
                           use_optax  = F,
                           nSGD = TWOSTEP_NSGD,
                           optim_type = "tryboth",
                           force_gaussian = F,
                           a_init_sd  = 0.001,
                           conf_level = ConfLevel)}
          #plot(unlist(Qoptimized$pi_star_point$k1), unlist(Qoptimized$pi_star_point$k2)); abline(a=0,b=1)
          if(case == "adversarial"){
            trueQ <- 1
            logMarginalSigma2 <- NA
            pi_star_true <- unlist(lapply(Qoptimized$pi_star_point$k1,function(zer){zer[2]}))
          }
    }
    if("try-error" %in% class(Qoptimized)){ print(Qoptimized); stop() }
    try({plot(pi_star_true, tmp_ <- unlist( lapply(Qoptimized$pi_star_point$k1,function(zer){zer[2]}) ));abline(a=0,b=1)},T)
    try({plot(pi_star_true - tmp_, main = round(mean(abs(pi_star_true - tmp_)),5))},T)

    # save information from analysis
    {
      pi_star_hat <- clip2( unlist( lapply(Qoptimized$pi_star_point$k1,function(ze){ze[2]}) ) )
      Q_value_hat_withEstPi_list <- Q_value_hat_withTruePi_list <- list()
      Q_value_hat_withEstPi_list$Q_se <- Q_value_hat_withTruePi_list$Q_se <- NA
      if(T == F){ 
      Q_value_hat_withEstPi_list <- strategize::strategize_onestep(
              W = X_,
              Y = Yobs,
              pi_list = Qoptimized$pi_star_point$k1,
              p_list = InitProbList_)
      Q_value_hat_withTruePi_list <- strategize::strategize_onestep(
              W = X_,
              Y = Yobs,
              hypotheticalProbList = TrueProbList_,
              p_list = InitProbList_)
      }
      dat <- c("monte_i" = monte_i,
                "kFactors" = as.character(kFactors),
                "nObs" = as.character(nObs),
                "penaltyType"=as.character(penaltyType),
                "optimType"=as.character(optimType),

                # analysis of pi*
                "pi_star_true" = list(pi_star_true),
                "pi_star_hat" = list(pi_star_hat),
                "pi_star_hat_se" = list(pi_star_hat_se <- clip2( unlist( lapply(Qoptimized$pi_star_se$k1,function(ze){ze[2]}) ) )),
                "pi_star_lower" = list(pi_star_lower <- c2p(unlist(lapply(Qoptimized$pi_star_lb$k1,function(zer){zer[2]})))),
                "pi_star_upper" = list(pi_star_upper <- c2p(unlist(lapply(Qoptimized$pi_star_ub$k1,function(zer){zer[2]})))),
                "inCI_pi_star" = list(inCI_pi <- 1*(  pi_star_lower <= pi_star_true &
                                                        pi_star_upper >= pi_star_true  )),
                "MaxProbRatio_pi_star" = getMaxLogProb(pi_star_hat),
                "MaxProbRatio_true" = getMaxLogProb(pi_star_true),
               
               # adversarial quantities 
                "case" = as.character(case), 
                "GroupGap" = ifelse(case == "adversarial", yes = mean( (unlist( Qoptimized$pi_star_point$k1 ) - unlist( Qoptimized$pi_star_point$k2 ) )^2 )^0.5, no = NA), 

                # analysis of q with estimated pi
                "SE_Q_exact" = SE_Q_exact <- Q_value_hat_withEstPi_list$Q_se,
                "SE_Q_MEst" = SE_Q_MEst <- Qoptimized$Q_se_mEst,
                "Qest_MEst"  = Qest_MEst <- Qoptimized$Q_point_mEst,
                "Qest_split1" = Qoptimized$Q_point,
                "Qest_split2" =  Qoptimized$Q_point_split,
                "Q_lower_exact" = Q_lower_exact <- Qest_MEst - abs(qnorm((1-ConfLevel)/2))*SE_Q_exact,
                "Q_upper_exact" = Q_upper_exact <- Qest_MEst + abs(qnorm((1-ConfLevel)/2))*SE_Q_exact,
                "Q_upper_mEst" = Q_upper_mEst <- Qest_MEst + abs(qnorm((1-ConfLevel)/2))*Qoptimized$Q_se_mEst,
                "Q_lower_mEst" = Q_lower_mEst <- Qest_MEst - abs(qnorm((1-ConfLevel)/2))*Qoptimized$Q_se_mEst,
                "logHypoN" = logHypoN <- log( nObs ),
                #"theoretical_varB" = theoreticalVarBound( unlist( lapply(Qoptimized$pi_star_point$k1,function(ze){ze[2]}) ) ),
                "theoretical_varB" = NA, 

               # analysis of Q with true pi
               "Q_value_hat_withTruePi" = Q_value_hat_withTruePi_list$Q_point,
               "SE_Q_exact_withTruePi" = Q_value_hat_withTruePi_list$Q_se,

               # record more info about Q
               "estSE_Qhat_exact" = SE_Q_exact,
               "estSE_Qhat_MEst" = SE_Q_MEst,
               "baseQ"     = getQ(getInteractionWts(as.vector(pi_star_hat))),
               "trueQ"     = getQ(getInteractionWts(pi_star_true)),
               "hajekQ" = Qest_MEst,
               "inCIQ_MEst" = 1*(  Q_lower_mEst <= trueQ & Q_upper_mEst >= trueQ  ),
               "inCIQ" = 1*(  Q_lower_exact <= trueQ & Q_upper_exact >= trueQ  ) #exact
                )
      dirname_ <- sprintf("./SavedResults/%s",
                          paste(gsub(Sys.Date(),pattern="-",replace="DASH"), optimType, penaltyType,sep="_"))
      if(!dir.exists(dirname_)){ dir.create(dirname_) }
      save("dat",file = sprintf("%s/K%s_N%s_MONTI%s_OPTIM%s_PEN%s_%sCase.Rdata",
                              dirname_, kFactors,nObs,monte_i,optimType,penaltyType,case ))
    }# end save info
}
}

if( FALSE ){
  # saved analyses
  # optimType <- "twoStep_default"; dirname_ <- sprintf("~/Dropbox/OptimizingSI/Results/2024DASH04DASH16_twoStep_default_L2");

  # use_RCE<-F;ConfLevel <- 0.95; kFactors_seq <- c(5,10,20); indicesInDelta<-1;penaltyType <- "L2"
  print(  sort( list.files(dirname_) ) )
  use_RCE <- F; f2n<-function(.){as.numeric(as.character(.))}
  res_df <- c(); for(file_ in list.files(dirname_)){
    data_ <- load(sprintf("%s/%s",dirname_,file_))
    res_df <- rbind(res_df,eval(parse(text = data_))) 
  }
  res_df <- as.data.frame( res_df )
  nObs_seq <- sort(unique(unlist(res_df$nObs)))

  res_df[res_df$case=="adversarial",]
  t_ <- tapply(unlist(res_df$GroupGap),
         unlist(res_df$nObs),function(zer){mean(zer,na.rm = T)})
  plot( t_[order(f2n(names(t_)))] ) 
  
  sort(  tapply(unlist(res_df$inCIQ_MEst), paste(unlist(res_df$nObs), unlist(res_df$kFactors), sep = "_"), length) )
  sort(  tapply(unlist(res_df$inCIQ_MEst), paste(unlist(res_df$nObs), unlist(res_df$kFactors), sep = "_"), mean) )

  LAMBDA_base <- div_factorQ <- div_factorPi_factorPi <- 1
  name_key <- c(sprintf("VARCONTROL%s_%s_SplitInDelta%s_PenaltyType%s_OptimType%s",gsub(round(LAMBDA_base,3),pattern="\\.",replace="PT"), gsub(Sys.Date(),pattern="-",replace="DASH"), indicesInDelta, penaltyType, optimType))
  FINAL_RESULTS <- F; FiguresLoc <- "~/Dropbox/OptimizingSI/Figures"
  source("~/Dropbox/OptimizingSI/Analysis/Simulations/linearCase_plotResults.R")
}
}
