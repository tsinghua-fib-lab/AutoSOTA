{
  rm(list=ls()); options(error = NULL)  
  # install.packages( "~/Documents/strategize-software/strategize",repos = NULL, type = "source",force = F);
  # try(devtools::install_github( 'cjerzak/strategize-software/strategize',ref="main" ),T)
  
  setwd("~/Dropbox/OptimizingSI")
  
  # global parameters
  ConfirmSanity <- FALSE 
  nObs <- 10000
  lambda <- 0.05  # Regularization parameter for L2 penalty
  tol <- 1e-6
  
  neqr <- c()
  #p_RVoters <- 0.5; { # Proportion of Republican voters
  for(p_RVoters in seq(0.1, 0.9, length.out = 7)){ 
    
    # Nash via grid search
    {
      # Goal: Implement grid search to 
      # find a Nash equilibrium in under two-stage primary
      # where R and D compete, selecting candidate gender.
      
      # Define parameters
      p_DVoters <- 1 - p_RVoters  # Proportion of Democrat voters
      
      # Voter preferences in the general election
      # R voters' probability of voting for R based on candidate genders
      # note: use of +/- values is for estabishing symmetry 
      # so that simple diff approach works out 
      pr_mm_RVoters <- 0.9  # R male vs D male
      pr_mf_RVoters <- 0.9 + 0.05 # R male vs D female
      pr_fm_RVoters <- 0.9 - 0.05  # R female vs D male
      pr_ff_RVoters <- 0.9 + 0.05 - 0.05 # R female vs D female
      
      # D voters' probability of voting for R based on candidate genders
      pr_mm_DVoters <- 0.1  # R male vs D male
      pr_mf_DVoters <- 0.1 - 0.05  # R male vs D female
      pr_fm_DVoters <- 0.1 + 0.05 # R female vs D male
      pr_ff_DVoters <- 0.1 - 0.05 + 0.05  # R female vs D female
      
      # R voters' probability of voting for R in primary based on gender 
      # her (unlike in the general case, the first position 
      # corresponds to the first position)
      p_m_primary <- 0.5
      pd_mm_RVoters_primary <- pr_mm_RVoters_primary <- 0.5
      pd_ff_RVoters_primary <- pr_ff_RVoters_primary <- 0.5
      pr_fm_RVoters_primary <- pd_fm_RVoters_primary <- 0.6
      pr_mf_RVoters_primary <- pd_mf_RVoters_primary <- 1 - pd_fm_RVoters_primary
      pd_m_RVoters_primary <- pr_m_RVoters_primary <- p_m_primary * pr_mm_RVoters_primary + 
        (1 - p_m_primary) * pr_mf_RVoters_primary
      pr_f_RVoters_primary <- pd_f_RVoters_primary <- 1-pd_m_RVoters_primary
      
      # D voters' probability of voting for D in primary based on gender 
      pr_mm_DVoters_primary <- pd_mm_DVoters_primary <- 0.50
      pr_ff_DVoters_primary <- pd_ff_DVoters_primary <- 0.50
      pr_fm_DVoters_primary <- pd_fm_DVoters_primary <- 0.45
      pr_mf_DVoters_primary <- pd_mf_DVoters_primary <- 1-pd_fm_DVoters_primary
      pd_m_DVoters_primary <- pr_m_DVoters_primary <- p_m_primary * pr_mm_DVoters_primary +
        (1 - p_m_primary) * pr_mf_DVoters_primary
      pd_f_DVoters_primary <- pr_f_DVoters_primary <- 1 - pr_m_DVoters_primary
      
      # Generate grid of possible strategies (π_R and π_D)
      pi_R_seq <- seq(0, 1, by = 0.001)
      pi_D_seq <- seq(0, 1, by = 0.001)
      
      # Function to compute expected vote share for R
      compute_vote_share_R <- function(pi_R, pi_D) {
        # note: mf refers to an R "m" and D "f"
        # Comb prob * E[Vote Share Among R] = Comb prob * [Pr(R)*Pr(R|GG)*P(G) + ...]
        term_mm <- pi_R * pi_D             * (p_RVoters * pr_mm_RVoters * pr_m_RVoters_primary + p_DVoters * pr_mm_DVoters * pd_m_DVoters_primary)
        term_mf <- pi_R * (1 - pi_D)       * (p_RVoters * pr_mf_RVoters * pr_m_RVoters_primary + p_DVoters * pr_mf_DVoters * pd_f_DVoters_primary)
        term_fm <- (1 - pi_R) * pi_D       * (p_RVoters * pr_fm_RVoters * pr_f_RVoters_primary + p_DVoters * pr_fm_DVoters * pd_m_DVoters_primary)
        term_ff <- (1 - pi_R) * (1 - pi_D) * (p_RVoters * pr_ff_RVoters * pr_f_RVoters_primary + p_DVoters * pr_ff_DVoters * pd_f_DVoters_primary)
        return( term_mm + term_mf + term_fm + term_ff ) 
      }
      
      # Utility functions incorporating L2 penalty
      utility_R <- function(pi_R, pi_D) {
        vote_share <- compute_vote_share_R(pi_R, pi_D)
        vote_share - lambda * ( (pi_R - 0.5)^2 + ( (1-pi_R) - 0.5)^2 )
      }
      
      utility_D <- function(pi_R, pi_D) {
        vote_share <- 1 - compute_vote_share_R(pi_R, pi_D)
        vote_share - lambda * ( (pi_D - 0.5)^2 + ( (1-pi_D) - 0.5)^2 )
      }
      
      # Compute best responses for each party
      best_response_D <- sapply(pi_R_seq, function(r) {
        utilities <- sapply(pi_D_seq, function(d) utility_D(r, d))
        pi_D_seq[which.max(utilities)]
      })
      
      best_response_R <- sapply(pi_D_seq, function(d) {
        utilities <- sapply(pi_R_seq, function(r) utility_R(r, d))
        pi_R_seq[which.max(utilities)]
      })
      
      # Identify Nash equilibria
      nash_equilibrium <- matrix(FALSE, nrow = length(pi_R_seq), ncol = length(pi_D_seq))
      for (i in seq_along(pi_R_seq)) {
        for (j in seq_along(pi_D_seq)) {
          current_r <- pi_R_seq[i]
          current_d <- pi_D_seq[j]
          
          # is current_r the best response to the given current_d AND
          # is current_d the best response to the given current_r 
          if (abs(current_r - best_response_R[j]) < tol && 
              abs(current_d - best_response_D[i]) < tol ) {
            nash_equilibrium[i, j] <- TRUE
          }
        }
      }
      
      # Extract and print equilibria
      equilibria <- which(nash_equilibrium, arr.ind = TRUE)
      if (nrow(equilibria) > 0) {
        cat("Nash Equilibria Found:\n")
        for (k in 1:nrow(equilibria)) {
          pi_R_grid <- pi_R_seq[equilibria[k, 1]]
          pi_D_grid <- pi_D_seq[equilibria[k, 2]]
          cat(sprintf("π_R: %.2f, π_D: %.2f\n", pi_R_grid, pi_D_grid))
        }
      } else {
        cat("No Nash equilibrium found within the grid.\n")
      }
      
      # Example: create data frame of all (pi_R, pi_D) pairs
      df_R <- expand.grid(pi_R = pi_R_seq, pi_D = pi_D_seq)
      df_R$utility_R <- mapply(utility_R, df_R$pi_R, df_R$pi_D)
      
      df_D <- expand.grid(pi_R = pi_R_seq, pi_D = pi_D_seq)
      df_D$utility_D <- mapply(utility_D, df_R$pi_R, df_R$pi_D)
      
      # best_response_D[r] = best pi_D to use, given pi_R = r
      # best_response_R[d] = best pi_R to use, given pi_D = d
      
      best_response_D <- sapply(pi_R_seq, function(r) {
        utilities <- sapply(pi_D_seq, function(d) utility_D(r, d))
        pi_D_seq[which.max(utilities)]
      })
      best_response_R <- sapply(pi_D_seq, function(d) {
        utilities <- sapply(pi_R_seq, function(r) utility_R(r, d))
        pi_R_seq[which.max(utilities)]
      })
      df_bestD <- data.frame(pi_R = pi_R_seq,
                             pi_D_star = best_response_D)
      
      df_bestR <- data.frame(pi_D = pi_D_seq,
                             pi_R_star = best_response_R)
      
      df_joint <- cbind(df_bestD, df_bestR)
    }
    
    # Nash via iterative optimization
    {
      # We assume 'utility_R' and 'utility_D' are defined above
      # We also assume 'best_response_R' and 'best_response_D' from the grid approach:
      #   best_response_D(r) = argmax_{pi_D in pi_D_seq} utility_D(r, pi_D)
      #   best_response_R(d) = argmax_{pi_R in pi_R_seq} utility_R(pi_R, d)
      #
      # These were computed via:
      #
      #   best_response_D <- sapply(pi_R_seq, function(r) {
      #     utilities <- sapply(pi_D_seq, function(d) utility_D(r, d))
      #     pi_D_seq[which.max(utilities)]
      #   })
      #
      #   best_response_R <- sapply(pi_D_seq, function(d) {
      #     utilities <- sapply(pi_R_seq, function(r) utility_R(r, d))
      #     pi_R_seq[which.max(utilities)]
      #   })
      #
      # But we need them as *functions*, not just as arrays. One quick fix is
      # to define a function best_response_D_grid() that, given a single r, returns
      # the best pi_D in [0,1], and similarly for R. Let’s do that:
      
      # Turn the existing vectors into “lookup” functions.
      # We'll use interpolation on pi_R_seq, or a simpler approach: find the index 
      # of r in pi_R_seq that is closest to the requested r, then return best_response_D. 
      # Because we used a 0.01 step, that is typically close enough for demonstration.
      
      best_response_D_grid <- function(r){
        # find index i where pi_R_seq[i] is closest to r
        i <- which.min(abs(pi_R_seq - r))
        # return the best response stored at best_response_D[i]
        return(best_response_D[i])
      }
      best_response_R_grid <- function(d){
        j <- which.min(abs(pi_D_seq - d))
        return(best_response_R[j])
      }
      
      # Iterative scheme
      max_iter <- 500
      # Initialize pi_R, pi_D at 0.5 for demonstration
      pi_R_current <- 0.50
      pi_D_current <- 0.50
      
      # We’ll keep track of the iteration path
      iteration_history <- data.frame(
        iter = integer(),
        pi_R = numeric(),
        pi_D = numeric()
      )
      
      for(iter in seq_len(max_iter)) {
        
        # 1. Given pi_R_current, find best_response_D
        pi_D_new <- best_response_D_grid(pi_R_current)
        
        # 2. Then, given that pi_D_new, find best_response_R
        pi_R_new <- best_response_R_grid(pi_D_new)
        
        # Record
        iteration_history <- rbind(
          iteration_history,
          data.frame(iter = iter, pi_R = pi_R_new, pi_D = pi_D_new)
        )
        
        # Check convergence
        if(abs(pi_R_new - pi_R_current) < tol && abs(pi_D_new - pi_D_current) < tol) {
          cat(sprintf("Converged at iteration %d:\n", iter))
          cat(sprintf("  pi_R = %.3f, pi_D = %.3f\n", pi_R_new, pi_D_new))
          break
        }
        
        # Update
        pi_R_current <- pi_R_new
        pi_D_current <- pi_D_new
      }
      
      # If no convergence by final iteration
      if(iter == max_iter){
        cat(sprintf("No convergence after %d iterations.\n", max_iter))
        cat(sprintf("Last values: pi_R = %.3f, pi_D = %.3f\n", pi_R_current, pi_D_current))
      }
      pi_R_iterative <- pi_R_current
      pi_D_iterative <- pi_D_current
      
      # Look at the path:
      # head(iteration_history)
      # tail(iteration_history)
    }
    
    # Generate observables  
    {
      #1. Respondent and candidate info
      # -----------------------------
      competing_group_variable_respondent <- sample(c("Democrat","Republican"), 
                                                    size = nObs, replace = TRUE,
                                                    prob = c(p_DVoters, p_RVoters))
      
      # Each respondent sees exactly 2 profiles => forced choice:
      competing_group_variable_respondent <- c(competing_group_variable_respondent,
                                               competing_group_variable_respondent)
      pair_id        <- respondent_id <- c(1:nObs, 1:nObs)
      respondent_id  <- respondent_id <- c(1:nObs, 1:nObs)
      profile_order  <- c(rep(1, nObs), rep(2, nObs))
      respondent_task_id <- rep(1, times = 2*nObs)
      
      # For simplicity, randomly assign candidate group. We then mark “Same” vs. “Different”
      competing_group_variable_candidate <- sample(c("Democrat","Republican"), 
                                                   size = 2*nObs, replace = TRUE,
                                                   prob = c(0.5,0.5))
      # E.g., "Same" means candidate’s_ party matches the respondent’s_ party
      competing_group_competition_variable_candidate <- ifelse(
        competing_group_variable_candidate[1:nObs] == competing_group_variable_candidate[-c(1:nObs)],
        yes = "Same", no = "Different"
      )
      
      # Expand for 2*nObs (long format)
      competing_group_competition_variable_candidate <-
        c(competing_group_competition_variable_candidate,
          competing_group_competition_variable_candidate)
      
      # -----------------------------
      # 2. Generate the design matrix X (binary factors) and outcomes
      # -----------------------------
      # We'll have kFactors columns, each in {0,1}, for 2*nObs profiles
      # THIS IS GENDER 
      X <- as.data.frame(matrix(rbinom(1 * nObs * 2, size = 1, prob = 0.5),
                                nrow = 2*nObs))
      colnames(X) <- "female"
      
      # Suppose 'X' is a vector or matrix of genders for each profile row i=1,...,2*nObs.
      # We can make a vector X_other, which for row i returns the gender of the *other* row
      # in the same pair, by matching on 'pair_id' and ignoring the same row.
      
      other_competing_group_variable_candidate <- X_other <- numeric(length = nrow(X))
      for (p in unique(pair_id)) {
        rows_in_pair <- which(pair_id == p)
        # exactly two rows per pair, e.g. i and j
        i <- rows_in_pair[1]
        j <- rows_in_pair[2]
        X_other[i] <- X$female[j]
        X_other[j] <- X$female[i]
        other_competing_group_variable_candidate[i] <- competing_group_variable_candidate[j]
        other_competing_group_variable_candidate[j] <- competing_group_variable_candidate[i]
      }
      
      Yobs <- rep(NA, times = length(competing_group_competition_variable_candidate))
      i_pool <- c(); for(cand_var in c("Republican","Democrat")){ 
        
        # sanity branching 
        if(ConfirmSanity){f <- function(i, prob){ prob }}
        if(!ConfirmSanity){f <- function(i, prob){ rbinom(length(i), size = 1, prob = prob) }}
        
        # Set s_ based on the candidate's_ party
        if(cand_var == "Republican"){ s_ <- 1 }
        if(cand_var == "Democrat"){ s_ <- -1 }
        
        # Republican primary probabilities if "Same":
        #   pr_m_RVoters_primary (R voter chooses an R male), pr_f_RVoters_primary (R voter chooses an R female)
        # Indices for the "Same" matchup, R respondent, R candidate is male
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 & # R is male 
            X_other == 0 # R is male
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_mm_RVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_mm_RVoters_primary) } 
        
        # indices for mixed 
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 1 &  # R is female 
            X_other == 0 # R is female
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_fm_RVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_fm_RVoters_primary) } 
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 & # R is female 
            X_other == 1 # R is female
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_mf_RVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_mf_RVoters_primary) } 
        
        # Indices for the "Same" matchup, R respondent, R candidate is female
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 1 & # R is female 
            X_other == 1 # R is female
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_ff_RVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_ff_RVoters_primary) } 
        
        # "Different" matchup => R respondent is Republican, candidate is `cand_var`.
        # R is male, D is male
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 &  # R is male
            X_other == 0     # D is male 
        ); i_pool <- c(i_, i_pool)
        Yobs[i_] <- f(i_, 0.5 + s_*(pr_mm_RVoters - 0.5)) 
        
        # R is male, D is female
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 & 
            X_other == 1
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, 0.5 + s_*(pr_fm_RVoters - 0.5)) }
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, 0.5 + s_*(pr_mf_RVoters - 0.5)) }
        
        # R is female, D is male
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 1 & 
            X_other == 0
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, 0.5 + s_*(pr_mf_RVoters - 0.5)) }
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, 0.5 + s_*(pr_fm_RVoters - 0.5)) }
        
        # R is female, D is female
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Republican" & 
            competing_group_variable_candidate == cand_var &
            X$female == 1 &    
            X_other == 1
        ); i_pool <- c(i_, i_pool)
        Yobs[i_] <- f(i_, 0.5 + s_*(pr_ff_RVoters - 0.5)) 
        
        
        ######################################################
        ######################################################
        ######################################################
        # Now, for the Democrat side in a "Same" matchup:
        # D is male
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 & 
            X_other == 0
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_mm_DVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_mm_DVoters_primary) } 
        
        # switch cases 
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var &
            X$female == 1 & 
            X_other == 0
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_fm_DVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_fm_DVoters_primary) } 
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 & 
            X_other == 1 
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_mf_DVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_mf_DVoters_primary) } 
        
        # D is female
        i_ <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var &
            X$female == 1 & 
            X_other == 1
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, pr_ff_DVoters_primary) } 
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, pd_ff_DVoters_primary) } 
        
        # "Different" matchup => respondent is Democrat, candidate is `cand_var`.
        # From the DEMOCRAT's_ perspective
        # D is male, R is male
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 &  # D is male
            X_other == 0     # R is male
        ); i_pool <- c(i_, i_pool)
        Yobs[i_] <- f(i_, 0.5 + s_*(pr_mm_DVoters - 0.5))
        
        # D is male, R is female
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var &
            X$female == 0 & 
            X_other == 1
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Democrat"){ Yobs[i_] <- f(i_, 0.5 + s_*(pr_fm_DVoters - 0.5)) } 
        if(cand_var == "Republican"){ Yobs[i_] <- f(i_, 0.5 + s_*(pr_mf_DVoters - 0.5)) } 
        
        # D is female, R is male
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var & 
            X$female == 1 & 
            X_other == 0
        ); i_pool <- c(i_, i_pool)
        if(cand_var == "Democrat"){  Yobs[i_] <- f(i_, 0.5 + s_*(pr_mf_DVoters - 0.5)) } 
        if(cand_var == "Republican"){  Yobs[i_] <- f(i_, 0.5 + s_*(pr_fm_DVoters - 0.5)) } 
        
        # D is female, R is female
        i_ <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Democrat" & 
            competing_group_variable_candidate == cand_var &
            X$female == 1 & 
            X_other == 1
        ); i_pool <- c(i_, i_pool)
        Yobs[i_] <- f(i_, 0.5 + s_*(pr_ff_DVoters - 0.5))
      }
      
      #plot(Yobs)
      table(Yobs)
      table(table(i_pool))
      
      # sanity checks - force forced choice
      sanity_ <- cbind(Yobs[1:nObs], Yobs[-c(1:nObs)])
      
      
      # values along diagonal should be from primary
      # values off diagonal should be 1-each other
      #plot(sanity_[,1], sanity_[,2], col = as.factor(competing_group_competition_variable_candidate)); abline(a=0,b=1);  abline(a=1,b=-1)
      #plot(sanity_[,1], 1-sanity_[,2] ); abline(a=0,b=1); abline(a=1,b=-1)
      if( ConfirmSanity ){ stop("Confirm Sanity Here") }
      
      #plot(sanity_[,1], 1-sanity_[,2] ); abline(a=0,b=1)
      #plot(sanity_[,1] - sanity_[,2] )
      a <- mean( abs(sanity_[,1] - sanity_[,2] ) < 0.000001)
      b <- mean( abs(sanity_[,1] - (1-sanity_[,2] )) < 0.000001)
      a + b 
      prop.table(table(competing_group_competition_variable_candidate))
      
      # force the force choice structure 
      Yobs[-c(1:nObs)] <- 1 - Yobs[1:nObs]
      
      # run X analysis 
      X_run <- X
      #X_run <- cbind(X, rbinom(nrow(X),size=1,prob=0.1))
      #colnames(X_run) <- c("female", "placeholder1",  "placeholder2")
    }    
    
    {
      ############################################################################
      ##  BEGIN LOGISTIC-BASED GRID SEARCH AND ITERATIVE ESTIMATION
      ############################################################################
      
      {
        ############################################################################
        ## 1) Create subset indices for each logistic regression
        ############################################################################
        X_run_FULL <- as.data.frame(cbind("Yobs" = Yobs, 
                                          "female" = X_run, 
                                          "other_female" = X_other, 
                                          "female_diff" = c(unlist(X_run) - X_other), 
                                          "competing_group_variable_candidate" = competing_group_variable_candidate,
                                          "other_competing_group_variable_candidate" = other_competing_group_variable_candidate  ) )
        
        # Republican primary (R respondents picking the R candidate)
        rep_primary_idx <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Republican" &
            competing_group_variable_candidate == "Republican"
        )
        
        # Democratic primary (D respondents picking the D candidate)
        dem_primary_idx <- which(
          competing_group_competition_variable_candidate == "Same" &
            competing_group_variable_respondent == "Democrat" &
            competing_group_variable_candidate == "Democrat"
        )
        
        # General election among Republican voters (R respondent choosing R candidate in R vs D)
        rep_general_idx <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Republican" &
            competing_group_variable_candidate == "Republican"
        )
        
        # General election among Democratic voters (D respondent choosing R candidate in R vs D)
        dem_general_idx <- which(
          competing_group_competition_variable_candidate == "Different" &
            competing_group_variable_respondent == "Democrat" &
            competing_group_variable_candidate == "Republican"
        )
        
        ############################################################################
        ## 2) Fit logistic regressions
        ############################################################################
        
        ## (a) Republican primary: Probability(R voter chooses R candidate), ~ candidate's gender
        rep_primary_fit <- glm(
          Yobs ~ female,
          data   = X_run_FULL[rep_primary_idx,],
          family = binomial
        )
        #View(cbind(X_run_FULL[rep_primary_idx,]))
        
        # compare with truth 
        pr_fm_RVoters_primary; 
        pr_mf_RVoters_primary; 
        pr_mm_RVoters_primary
        pr_ff_RVoters_primary
        c(pr_m_RVoters_primary, pr_f_RVoters_primary)
        tapply(X_run_FULL$Yobs[rep_primary_idx],
               X_run_FULL$female[rep_primary_idx], mean)
        tapply(rep_primary_fit$fitted.values, 
               rep_primary_fit$model$female, mean)
        c(pr_m_RVoters_primary, pr_f_RVoters_primary)-
          tapply(rep_primary_fit$fitted.values, rep_primary_fit$model$female, mean)
        
        ## (b) Democratic primary: Probability(D voter chooses D candidate), ~ candidate's gender
        dem_primary_fit <- glm(
          Yobs ~ female,
          data   = X_run_FULL[dem_primary_idx,],
          family = binomial
        )
        
        # compare with truth 
        c(pr_m_DVoters_primary, pr_f_DVoters_primary)
        tapply(dem_primary_fit$fitted.values, 
               dem_primary_fit$model$female, mean)
        c(pr_m_DVoters_primary, pr_f_DVoters_primary) - tapply(dem_primary_fit$fitted.values, dem_primary_fit$model$female, mean)
        
        ## (c) General election, R voters: Probability(R voter chooses R candidate),
        ##     ~ R candidate's gender (female) + opponent's gender (X_other)
        rep_general_fit <- glm(
          #Yobs ~ female + other_female + female*other_female,
          Yobs ~ female_diff,
          data   = X_run_FULL[rep_general_idx,],
          family = binomial)
        
        ## (d) General election, D voters: Probability(D voter chooses R candidate),
        ##     ~ R candidate's gender (female) + opponent's gender (X_other)
        dem_general_fit <- glm(
          #Yobs ~ female + other_female + female*other_female,
          Yobs ~ female_diff,
          data   = X_run_FULL[dem_general_idx,],
          family = binomial)
        # View(X_run_FULL[dem_general_idx,])
        # table(dem_general_fit$fitted.values)
        
        ############################################################################
        ## 3) Extract key predicted probabilities for each scenario
        ############################################################################
        
        # --- Primary: R voters picking R candidate if R=male/female
        pr_m_RVoters_primary_l <- predict(rep_primary_fit,
                                          newdata = data.frame(female = 0),
                                          type = "response")
        pr_f_RVoters_primary_l <- predict(rep_primary_fit,
                                          newdata = data.frame(female = 1),
                                          type = "response")
        
        # --- Primary: D voters picking D candidate if D=male/female
        pd_m_DVoters_primary_l <- predict(dem_primary_fit,
                                          newdata = data.frame(female = 0),
                                          type = "response")
        pd_f_DVoters_primary_l <- predict(dem_primary_fit,
                                          newdata = data.frame(female = 1),
                                          type = "response")
        
        # --- General: R voters picking R candidate (vs D) for each (R_female=0/1, D_female=0/1)
        pr_mm_RVoters_l <- predict(rep_general_fit,
                                   newdata = data.frame(female = 0, other_female = 0, female_diff = 0),
                                   type = "response")
        pr_mf_RVoters_l <- predict(rep_general_fit,
                                   newdata = data.frame(female = 0, other_female = 1, female_diff = -1),
                                   type = "response")
        pr_fm_RVoters_l <- predict(rep_general_fit,
                                   newdata = data.frame(female = 1, other_female = 0, female_diff = 1),
                                   type = "response")
        pr_ff_RVoters_l <- predict(rep_general_fit,
                                   newdata = data.frame(female = 1, other_female = 1, female_diff = 0),
                                   type = "response")
        
        # --- General: D voters picking R candidate (vs D) for each (R_female=0/1, D_female=0/1)
        pr_mm_DVoters_l <- predict(dem_general_fit,
                                   newdata = data.frame(female = 0, other_female = 0, female_diff = 0),
                                   type = "response")
        pr_mf_DVoters_l <- predict(dem_general_fit,
                                   newdata = data.frame(female = 0, other_female = 1, female_diff = -1),
                                   type = "response")
        pr_fm_DVoters_l <- predict(dem_general_fit,
                                   newdata = data.frame(female = 1, other_female = 0, female_diff = 1),
                                   type = "response")
        pr_ff_DVoters_l <- predict(dem_general_fit,
                                   newdata = data.frame(female = 1, other_female = 1, female_diff = 0),
                                   type = "response")
        
        
        ############################################################################
        ## 4) Define a new compute_vote_share_R function (logistic-based) and utilities
        ############################################################################
        compute_vote_share_R_logit <- function(pi_R, pi_D) {
          # Weighted by fraction of R vs D voters
          #   and using the logistic predictions we just extracted above:
          
          term_mm <- pi_R * pi_D * (  p_RVoters * pr_mm_RVoters_l * pr_m_RVoters_primary_l +
                                        p_DVoters * pr_mm_DVoters_l * pd_m_DVoters_primary_l )
          term_mf <- pi_R * (1 - pi_D) * (p_RVoters * pr_mf_RVoters_l * pr_m_RVoters_primary_l +
                                            p_DVoters * pr_mf_DVoters_l * pd_f_DVoters_primary_l )
          term_fm <- (1 - pi_R) * pi_D * (p_RVoters * pr_fm_RVoters_l * pr_f_RVoters_primary_l +
                                            p_DVoters * pr_fm_DVoters_l * pd_m_DVoters_primary_l )
          term_ff <- (1 - pi_R) * (1 - pi_D) * (p_RVoters * pr_ff_RVoters_l * pr_f_RVoters_primary_l +
                                                  p_DVoters * pr_ff_DVoters_l * pd_f_DVoters_primary_l )
          return( term_mm + term_mf + term_fm + term_ff ) 
        }
        
        utility_R_logit <- function(pi_R, pi_D) {
          vote_share <- compute_vote_share_R_logit(pi_R, pi_D)
          vote_share - lambda * (
            (pi_R - 0.5)^2 + ((1 - pi_R) - 0.5)^2
          ) }
        
        utility_D_logit <- function(pi_R, pi_D) {
          vote_share_D <- 1 - compute_vote_share_R_logit(pi_R, pi_D)
          vote_share_D - lambda * (
            (pi_D - 0.5)^2 + ((1 - pi_D) - 0.5)^2
          )
        }
        
        # test utility functions 
        x_<-runif(1,0,1);y_<-runif(1,0,1)
        x_ <- y_ <- 0.4
        utility_D(x_,y_) / utility_D_logit(x_,y_)
        utility_R(x_,y_) / utility_R_logit(x_,y_)
        
        
        ############################################################################
        ## 5) Grid search for best responses and approximate Nash equilibrium
        ############################################################################
        
        pi_D_seq2 <- pi_R_seq2 <- seq(0, 1, by = 0.001)
        best_response_D_logit <- sapply(pi_R_seq2, function(r) {
          utilities_D <- sapply(pi_D_seq2, function(d) utility_D_logit(r, d))
          #utilities_D <- sapply(pi_D_seq2, function(d) utility_D(r, d))
          pi_D_seq2[which.max(utilities_D)]
        })
        best_response_R_logit <- sapply(pi_D_seq2, function(d) {
          utilities_R <- sapply(pi_R_seq2, function(r) utility_R_logit(r, d))
          #utilities_R <- sapply(pi_R_seq2, function(r) utility_R(r, d)) # for sanity check 
          pi_R_seq2[which.max(utilities_R)]
        })
        
        # sanity plot 
        plot(sapply(seq(0, 1, by = 0.01), function(r){sapply(seq(0, 1, by = 0.01), function(d) utility_D_logit(r, d)) }),
             sapply(seq(0, 1, by = 0.01), function(r){sapply(seq(0, 1, by = 0.01), function(d) utility_D(r, d)) }))# sanity 
        plot(sapply(seq(0, 1, by = 0.01), function(r){sapply(seq(0, 1, by = 0.01), function(d) utility_R_logit(r, d)) }),
             sapply(seq(0, 1, by = 0.01), function(r){sapply(seq(0, 1, by = 0.01), function(d) utility_R(r, d)) }))# sanity 
        
        nash_equilibrium_logit <- matrix(FALSE,  nrow = length(pi_R_seq2), 
                                         ncol = length(pi_D_seq2))
        for (i in seq_along(pi_R_seq2)) {
          for (j in seq_along(pi_D_seq2)) {
            r_ <- pi_R_seq2[i]
            d_ <- pi_D_seq2[j]
            # Check if r_ is best response to d_ and d_ is best response to r_
            if (
              abs(r_ - best_response_R_logit[j]) < tol &&
              abs(d_ - best_response_D_logit[i]) < tol
            ) {
              nash_equilibrium_logit[i, j] <- TRUE
            }
          }
        }
        
        eqs_logit <- which(nash_equilibrium_logit, arr.ind = TRUE)
        if (nrow(eqs_logit) > 0) {
          cat("Nash Equilibria (logit-based) Found:\n")
          for (k in seq_len(nrow(eqs_logit))) {
            pi_R_grid_est_manual <- pi_R_seq2[eqs_logit[k, 1]]
            pi_D_grid_est_manual <- pi_D_seq2[eqs_logit[k, 2]]
            cat(sprintf("  pi_R: %.3f, pi_D: %.3f\n", pi_R_grid_est_manual, pi_D_grid_est_manual))
            #cat(sprintf("  pi_R: %.3f, pi_D: %.3f\n", pi_R_grid, pi_D_grid))
          }
        } else {
          cat("No logit-based Nash equilibrium found in the grid.\n")
          pi_R_grid_est_manual <- NA
          pi_D_grid_est_manual <- NA
        }
        
        
        ############################################################################
        ## 6) Iterative best response using logistic-based utilities
        ############################################################################
        
        # Convert best_response_D_logit (etc.) from vectors to "lookup" functions
        # (closest grid point approach)
        best_response_D_grid_logit <- function(r) {
          i <- which.min(abs(pi_R_seq2 - r))
          best_response_D_logit[i] }
        
        best_response_R_grid_logit <- function(d) {
          j <- which.min(abs(pi_D_seq2 - d))
          best_response_R_logit[j] }
        
        max_iter_logit <- 500
        pi_R_current_logit <- 0.5
        pi_D_current_logit <- 0.5
        
        for (iter_logit in seq_len(max_iter_logit)) {
          pi_D_new_logit <- best_response_D_grid_logit(pi_R_current_logit)
          pi_R_new_logit <- best_response_R_grid_logit(pi_D_new_logit)
          
          # Check convergence
          if (
            abs(pi_R_new_logit - pi_R_current_logit) < tol &&
            abs(pi_D_new_logit - pi_D_current_logit) < tol
          ) {
            cat(sprintf("Logit-based iterative scheme converged at iteration %d:\n", iter_logit))
            cat(sprintf("  pi_R = %.3f, pi_D = %.3f\n", pi_R_new_logit, pi_D_new_logit))
            break
          }
          
          pi_R_current_logit <- pi_R_new_logit
          pi_D_current_logit <- pi_D_new_logit
        }
        
        if (iter_logit == max_iter_logit) {
          cat(sprintf("Logit-based iteration did NOT converge after %d steps.\n", max_iter_logit))
        }
        
        pi_R_iterative_logit_manual <- pi_R_current_logit
        pi_D_iterative_logit_manual <- pi_D_current_logit
        
        
        ############################################################################
        ## 7) Report final results
        ############################################################################
        
        cat("------------------------------------------------------\n")
        cat("Logit-Based Results:\n")
        cat(sprintf("Grid-search NE:    (pi_R = %.3f, pi_D = %.3f)\n",
                    pi_R_grid_est_manual, pi_D_grid_est_manual))
        cat(sprintf("Iterative Best-R:  (pi_R = %.3f, pi_D = %.3f)\n",
                    pi_R_iterative_logit_manual, pi_D_iterative_logit_manual))
        cat("------------------------------------------------------\n")
      }
      
    }
    
    # Nash via estimation + iterative optimization - package 
    {
      Qoptimized <-  {strategize::strategize(
        Y = Yobs,
        W = X_run,
        lambda = lambda,
        conda_env = "strategize",
        conda_env_required = TRUE,
        
        # Adversarial parameters
        competing_group_variable_respondent = competing_group_variable_respondent,
        competing_group_variable_candidate   = competing_group_variable_candidate,
        competing_group_competition_variable_candidate = competing_group_competition_variable_candidate,
        pair_id         = pair_id,
        respondent_id   = respondent_id,
        respondent_task_id = respondent_task_id,
        profile_order   = profile_order,
        diff           = TRUE,
        adversarial    = TRUE,
        
        # Main parameters
        compute_se       = FALSE,
        penalty_type     = "L2",
        use_regularization= FALSE,
        use_optax        = TRUE, 
        learning_rate_max = 0.001, 
        nSGD             = 500L,
        nMonte_adversarial = 50L, 
        nMonte_Qglm      = 50L, 
        temperature      = 0.3, 
        optim_type       = "tryboth",
        force_gaussian   = FALSE,
        a_init_sd        = 0.01,
        conf_level       = 0.95
      )}
      pi_R_est_iterative <- Qoptimized$pi_star_point$Republican$female["1"]
      pi_D_est_iterative <- Qoptimized$pi_star_point$Democrat$female["1"]
      
      cat(sprintf("Found via grid search: {pi_R: %.3f, pi_D: %.3f}\n", pi_R_grid, pi_D_grid))
      cat(sprintf("Found via iterative: {pi_R: %.3f, pi_D: %.3f}\n", pi_R_iterative, pi_D_iterative))
      cat(sprintf("Found via est iterative: {pi_R: %.3f, pi_D: %.3f}\n", pi_R_est_iterative, pi_D_est_iterative))
    }
    
    {
      library(ggplot2)
      library(gridExtra)  # For arranging plots side by side
      gpub <- {ggplot(df_R, aes(x = pi_R, y = pi_D, fill = utility_R)) +
          geom_tile() +                                
          scale_fill_gradient2(
            midpoint = mean(df_R$utility_R),           
            low = "blue", mid = "white", high = "red"
          ) +
          labs(
            title = "R's Utility Heatmap with Best Response Curves",
            x = expression(pi[R]), y = expression(pi[D]),
            fill = "Utility(R)"
          ) +
          theme_minimal() + 
          # Add best-response lines with inherit.aes = FALSE
          geom_line(data = df_bestD, aes(x = pi_R, y = pi_D_star),
                    color = "blue", linewidth = 1.2, inherit.aes = FALSE) +
          geom_line(data = df_bestR, aes(x = pi_R_star, y = pi_D),
                    color = "red", linewidth = 1.2, inherit.aes = FALSE,
                    arrow = arrow(length = unit(0.3, "cm"), type = "closed")) +
          annotate("point", x = pi_R_grid, y = pi_D_grid, shape = 8, size = 5, color = "black") + 
          annotate("point", x = pi_R_current, y = pi_D_current, shape = 8, size = 5, color = "gray") + 
          annotate("point", x = pi_R_est_iterative, y = pi_D_est_iterative, shape = 8, size = 5, color = "green")
      }
      
      gdem <- {ggplot(df_D, aes(x = pi_R, y = pi_D, fill = utility_D)) +
          geom_tile() +                                
          scale_fill_gradient2(
            midpoint = mean(df_D$utility_D),           
            low = "red", mid = "white", high = "blue"
          ) + 
          labs(
            title = "D's Utility Heatmap with Best Response Curves",
            x = expression(pi[R]), y = expression(pi[D]),
            fill = "Utility(D)"
          ) +
          theme_minimal() + 
          # Add best-response lines with inherit.aes = FALSE
          geom_line(data = df_bestD, aes(x = pi_R, y = pi_D_star),
                    color = "blue", linewidth = 1.2, inherit.aes = FALSE,
                    arrow = arrow(length = unit(0.3, "cm"), type = "closed",ends = "last")) +
          geom_line(data = df_bestR, aes(x = pi_R_star, y = pi_D),
                    color = "red", linewidth = 1.2, inherit.aes = FALSE) +
          annotate("point", x = pi_R_grid, y = pi_D_grid, shape = 8, size = 5, color = "black") + 
          annotate("point", x = pi_R_iterative, y = pi_D_iterative, shape = 8, size = 5, color = "gray") +
          annotate("point", x = pi_R_est_iterative, y = pi_D_est_iterative, shape = 8, size = 5, color = "green")
      }
      
      # grid 
      grid.arrange(gpub, gdem, ncol = 2)
      
      # Export each plot individually as a PDF file using ggsave()
      ggsave("./Figures/gpub.pdf", plot = gpub, device = "pdf", width = 8, height = 6)
      ggsave("./Figures/gdem.pdf", plot = gdem, device = "pdf", width = 8, height = 6)
    }
    
    # append 
    neqr <- rbind(neqr,
                  data.frame(
                    # truth 
                    "pi_R_grid" = pi_R_grid,
                    "pi_D_grid" = pi_D_grid,
                    "pi_R_iterative" = pi_R_iterative,
                    "pi_D_iterative" = pi_D_iterative,
                    
                    # est iterative
                    "pi_R_est_iterative" = pi_R_est_iterative,
                    "pi_D_est_iterative" = pi_D_est_iterative,
                    
                    # est iterative, manual 
                    "pi_R_est_iterative_manual" = pi_R_iterative_logit_manual,
                    "pi_D_est_iterative_manual" = pi_D_iterative_logit_manual,
                    "pi_R_est_grid_manual" = pi_R_grid_est_manual,
                    "pi_D_est_grid_manual" = pi_D_grid_est_manual,
                    
                    "p_RVoters" = p_RVoters
                  ) )
  }
  
  # analyze neqr
  neqr <- as.data.frame( neqr )
  
  # confirm grid vs. iterative
  plot( neqr$pi_R_grid,
        neqr$pi_R_iterative ); abline(a=0,b=1)
  
  # confirm grid vs. est + iterative
  plot( neqr$pi_R_grid,
        neqr$pi_R_est_iterative ); abline(a=0,b=1)
  plot( neqr$pi_D_grid,
        neqr$pi_D_est_iterative ); abline(a=0,b=1)
  
  # confirm grid vs. est grid - manual 
  plot( neqr$pi_R_grid,
        neqr$pi_R_est_grid_manual ); abline(a=0,b=1)
  plot( neqr$pi_D_grid,
        neqr$pi_D_est_grid_manual ); abline(a=0,b=1)
  plot( neqr$pi_R_grid - neqr$pi_R_est_grid_manual, col ="red", pch=19, ylim = c(-0.1,0.1) ); abline(h=0)
  points( neqr$pi_D_grid - neqr$pi_D_est_grid_manual, col = "blue", pch = 19 )
  
  plot( neqr$pi_R_grid,
        neqr$pi_R_est_iterative_manual ); abline(a=0,b=1)
  plot( neqr$pi_D_grid,
        neqr$pi_D_est_iterative_manual ); abline(a=0,b=1)
  plot( neqr$pi_R_grid - neqr$pi_R_est_iterative_manual, col ="red", pch=19, ylim = c(-0.1,0.1) ); abline(h=0)
  points( neqr$pi_D_grid - neqr$pi_D_est_iterative_manual, col = "blue" ,pch=19)
  
  plot(neqr$pi_R_est_iterative_manual, neqr$pi_R_est_grid_manual);abline(a=0,b=1)
  plot(neqr$pi_D_est_iterative_manual, neqr$pi_D_est_grid_manual);abline(a=0,b=1)
  
}

