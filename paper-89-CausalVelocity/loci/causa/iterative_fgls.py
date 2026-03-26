import numpy as np
import statsmodels.api as sm


# Implements iterative fgls for y given a feature map Phi (see https://www.schmidheiny.name/teaching/heteroscedasticity2up.pdf)
# Step 1: fit OLS y~Phi and obtain residuals of the model (res1)
# Step 2: fit OLS log((res1)^2)~Psi (Psi needs to be positive) and obtain predictions h
# Step 3: fit GLS y~Phi with weights 1/exp(h) (the estimated precision)
# If more than 1 iteration:
#   Step 1i: obtain resi (similar to step 1) but now from the GLS fit of Step 3
#   Iterate
def iterative_fgls(y, Phi, Psi=None, n_steps=100, takeLog=False, cutOff=1e-6):
  # track loss
  losses = list()
  
  # If Psi is not provided, use the absolute value of Phi
  if Psi is None and not takeLog:
    Psi = np.abs(Phi)
  elif Psi is None:
    Psi = Phi
  
  # Step 1
  ols_model = sm.OLS(y, Phi).fit_regularized(L1_wt=0.0, alpha=1e-6)
  # Step 2
  hat_u_sq = np.square(y - Phi @ ols_model.params)
  if takeLog:
    hat_u_sq = np.log(hat_u_sq)
  ols_resid = sm.OLS(hat_u_sq, Psi).fit_regularized(L1_wt=0.0, alpha=1e-6)
  # weight_2 = np.clip(ols_resid.params, a_min=0, a_max=None)
  weight_2 = ols_resid.params
  hat_h = Psi @ weight_2
  # Step 3
  if takeLog:
    hat_w = np.clip(np.exp(hat_h), a_min=1e-7, a_max=1e7)
  else:
    hat_w = np.maximum(cutOff, hat_h)
  w_2 = np.sqrt(hat_w) # estimated std
  fgls_model = sm.GLS(y, Phi, sigma=hat_w)
  fgls_model_fit = fgls_model.fit_regularized(L1_wt=0.0, alpha=1e-6)
  losses.append(- fgls_model.loglike(fgls_model_fit.params) / len(y))
  w_1 = fgls_model_fit.params # estimated mean
  
  # iterate
  if n_steps > 1:
    for step in range(n_steps-1):
      # Step 2i
      hat_u_sq = np.square(y - Phi @ w_1)
      if takeLog:
        hat_u_sq = np.log(hat_u_sq)
      ols_resid = sm.OLS(hat_u_sq, Psi).fit_regularized(L1_wt=0.0, alpha=1e-6)
      weight_2 = ols_resid.params
      hat_h = Psi @ weight_2
      # Step 3i
      if takeLog:
        hat_w = np.clip(np.exp(hat_h), a_min=1e-7, a_max=1e7)
      else:
        hat_w = np.maximum(cutOff, hat_h)
      w_2 = np.sqrt(hat_w) # estimated std
      fgls_model = sm.GLS(y, Phi, sigma=hat_w)
      fgls_model_fit = fgls_model.fit_regularized(L1_wt=0.0, alpha=1e-6)
      losses.append(- fgls_model.loglike(fgls_model_fit.params) / len(y))
      w_1 = fgls_model_fit.params # estimated mean
      if abs(losses[-1] - losses[-2]) < 1e-6:
        break

  return w_1, w_2, weight_2, losses, losses[-1]
