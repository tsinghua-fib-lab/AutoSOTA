import numpy as np
import time
from trainable_ot_dro.conic_problem import ConicProblem
from trainable_ot_dro.utils.numerical_utilities import stabilize_L


def optimize_transportation_matrix(L_0,
                                   conic_program,
                                   distributions,
                                   distribution_distance,
                                   constraint_parameters,
                                   optimization_parameters,
                                   use_decreasing_step_size=False):
  """
    Optimize shape of ambiguity set by changing the transport cost matrix L.
  """

  # ========================================
  # Input checks
  # ========================================
  # Initial cost weight matrix:
  assert L_0.shape[0] == L_0.shape[1], "Transport cost matrix is not square."
  assert np.allclose(L_0, np.tril(L_0)), "Transport cost matrix is not lower triangular."
  # Conic program
  assert type(conic_program) == dict, "Conic program has to be provided as a dictionary."
  assert "A" in conic_program, "Constraint matrix A not provided."
  assert "b" in conic_program, "Constraint vector b not provided."
  assert "c" in conic_program, "Objective vector c not provided."
  assert "cone" in conic_program, "Cone not provided."
  assert "get_L" in conic_program, "Function to extract L matrix not provided."
  assert "set_L" in conic_program, "Function to set L matrix not provided."
  assert "get_decision" in conic_program, "Function to extract decision vector not provided."
  # Distributions
  assert type(distributions) == dict, "Distributions have to be provided as a dictionary."
  assert "reference" in distributions, "Reference distribution not provided."
  assert "bootstrap" in distributions, "Bootstrapped distributions not provided."
  assert type(distributions["bootstrap"]) == list, "Bootstrapped distributions have to be provided as a list."
  assert len(distributions["bootstrap"]) > 0, "No bootstrapped distributions provided."
  # Distribution distance
  assert type(distribution_distance) == dict, "Distribution distance has to be provided as a dictionary."
  assert "gradient" in distribution_distance, "Gradient function not provided."
  assert "type" in distribution_distance, "Wasserstein type not provided."
  # Constraint Parameters
  assert type(constraint_parameters) == dict, "Constraint parameters have to be provided as a dictionary."
  assert "eps" in constraint_parameters, "Epsilon (distance upper bound) not provided."
  assert "gamma" in constraint_parameters, "Gamma (confidence level for bootstrap constraint) not provided."
  # Optimization parameters
  assert type(optimization_parameters) == dict, "Optimization parameters have to be provided as a dictionary."
  assert "n_iter_max" in optimization_parameters, "Maximum number of iterations not provided."
  assert "learning_rate" in optimization_parameters, "Learning rate not provided."
  assert "store_every" in optimization_parameters, "Store every not provided."
  assert "stopping_tols" in optimization_parameters, "Stopping tolerances not provided."
  assert type(optimization_parameters["stopping_tols"]) == dict, "Stopping tolerances have to be provided as a dictionary."
  assert "diff_objective" in optimization_parameters["stopping_tols"], "Objective stopping tolerance not provided."
  assert "diff_objective_pen" in optimization_parameters["stopping_tols"], "Penalized objective stopping tolerance not provided."
  assert "diff_decision" in optimization_parameters["stopping_tols"], "Decision stopping tolerance not provided."
  assert "diff_weightmat" in optimization_parameters["stopping_tols"], "Weight matrix stopping tolerance not provided."
  assert "weightmat_gradient" in optimization_parameters["stopping_tols"], "Weight matrix gradient stopping tolerance not provided."
  assert "rel_impr_obj" in optimization_parameters["stopping_tols"], "Relative improvement of objective stopping tolerance not provided."
  assert "rel_diff_obj_pen" in optimization_parameters["stopping_tols"], "Relative difference of penalized objective stopping tolerance not provided."
  assert "penalization" in optimization_parameters, "Penalization parameters not provided."
  assert type(optimization_parameters["penalization"]) == dict, "Penalization parameters have to be provided as a dictionary."
  assert "lambda" in optimization_parameters["penalization"], "Penalization weight not provided."
  assert "eta" in optimization_parameters["penalization"], "Indicator approximation parameter not provided."
  # ========================================

  # ========================================
  # Dimensions
  dim_random = L_0.shape[0]
  n_bootstrap = len(distributions["bootstrap"])
  # ========================================

  # ========================================
  # Constraint parameters
  eps_0 = constraint_parameters["eps"]
  assert eps_0 > 0, "Epsilon has to be positive."
  gamma = constraint_parameters["gamma"]
  # ========================================

  # ========================================
  # Optimization parameters
  n_iter_max = optimization_parameters["n_iter_max"]
  learning_rate = optimization_parameters["learning_rate"]
  stopping_tols = optimization_parameters["stopping_tols"]
  store_every = optimization_parameters["store_every"]
  penalization = optimization_parameters["penalization"]
  lambda_pen = penalization["lambda"]
  eta_pen = penalization["eta"]
  # ========================================

  # ========================================
  # Functions provided
  get_L = conic_program["get_L"]
  set_L = conic_program["set_L"]
  get_decision = conic_program["get_decision"]
  distance_gradient = distribution_distance["gradient"]
  # ========================================

  # ========================================
  # Gradient clipping
  clipval_L_grad_low = -1e4
  clipval_L_grad_high = 1e4
  # ========================================

  # ========================================
  # Indicator approximation function
  def indicator_approx(d):
    return 1 / (1 + np.exp( - eta_pen * (d / eps_0 - 1)))
  # ========================================

  # ========================================
  # Derivative of the indicator approximation function
  def indicator_approx_derivative(d):
    numerator = eta_pen * np.exp(- eta_pen * (d / eps_0 - 1))
    denominator = eps_0 * (1 + np.exp(- eta_pen * (d / eps_0 - 1)))**2
    return numerator / denominator

  # ========================================
  # ADAM parameters
  use_adam = True
  beta1_adam = 0.9
  beta2_adam = 0.999
  eps_adam = 1e-8
  m_adam = np.zeros_like(L_0)
  v_adam = np.zeros_like(L_0)
  t_adam = 0
  # ========================================

  # ========================================
  # Initial cost weight matrix
  L_var = np.copy(L_0)
  # ========================================

  # ========================================
  # DRO as finite conic program
  A_cp = conic_program["A"]
  b_cp = conic_program["b"]
  c_cp = conic_program["c"]
  cone = conic_program["cone"]
  n_primal = A_cp.shape[1]
  m_primal = A_cp.shape[0]
  assert c_cp.shape[0] == n_primal
  assert b_cp.shape[0] == m_primal
  problem = ConicProblem()
  dx = c_cp.reshape(-1, 1)
  dy = np.zeros((m_primal, 1))
  ds = np.zeros((m_primal, 1))
  A_cp = set_L(A_cp, L_var) # just to make sure
  # ========================================

  # ========================================
  # History of the optimization
  L_matrices = []
  gradients_L = []
  objective_values = []
  penalization_values = []
  decisions = []
  gradient_step_times = []
  stopping_criteria = []
  iterations = []
  # ========================================

  # ========================================
  # Temporary variables
  stopping_criteria_tmp = {}
  bootstrap_distances_tmp = np.zeros(n_bootstrap)
  bootstrap_distance_gradients_tmp = np.zeros((n_bootstrap, dim_random, dim_random))
  # ========================================


  # ========================================
  # Optimization Loop for L

  # The optimization problem is
  # min_L f(L) + e(L)
  #
  # where the objective is
  #   f(L) = c^T xsol_cp
  # and the penalization is
  #   e(L) = lambda * max{0, h(L)}**2
  #
  # with h(L) = 1/N_b * sum_i^N_b s(d_i(L); eps_0) - gamma
  # and s(d; eps) = 1 / [ 1 + exp( - eta * [d/eps - 1] ) ]

  for ITERATION in range(n_iter_max):
    if ITERATION % store_every == 0:
      print("iteration: ", ITERATION)
    start_time = time.time()

    # ===
    # Store previous results for convergence check
    # necessary because not every iteration may be stored
    # ===
    if ITERATION > 0:
      objective_prev = objective_tmp
      penalized_obj_prev = penalized_obj_tmp
      decision_prev = decision_tmp

    # ===
    # Solving and differentiating through the lower level problem
    # ===
    lowlev_result = problem.solve_and_derivative(A_cp, b_cp, c_cp, cone, verbose=False)
    dA_S, _, _ = lowlev_result["derivative_adjoint"](dx, dy, ds)
    dLW_obj = get_L(dA_S)

    # conic problem primal solution vector x
    xsol_cp = np.array(lowlev_result["solution"].x).reshape(-1)
    # optimal lower level decision vector
    decision_tmp = get_decision(xsol_cp)
    # objective value without penalization
    objective_tmp = np.dot(c_cp.T, xsol_cp).item()

    # ===
    # Penalization Gradient
    # ===
    for bootstrap_idx in range(n_bootstrap):
      # Gradient and distance of distance REF <--> BOOTSTRAP
      distance_tmp, distance_grad_tmp = distance_gradient(distributions["reference"],
                                            distributions["bootstrap"][bootstrap_idx],
                                            L_var,
                                            wasserstein_type=distribution_distance["type"])
      bootstrap_distances_tmp[bootstrap_idx] = distance_tmp
      # Only keep lower triangular part
      distance_grad_tmp = np.tril(distance_grad_tmp)
      bootstrap_distance_gradients_tmp[bootstrap_idx] = distance_grad_tmp

    # Constraint Violation and Violation Gradient
    s = indicator_approx(bootstrap_distances_tmp)
    violation_tmp = 1/n_bootstrap * np.sum(s) - gamma
    s_grad = indicator_approx_derivative(bootstrap_distances_tmp)
    grad_violation_tmp = 1/n_bootstrap * np.sum(s_grad[:, None, None] * bootstrap_distance_gradients_tmp, axis=0)

    # Penalization Value and Gradient
    penalization_tmp = lambda_pen * np.maximum(0, violation_tmp)**2
    penalized_obj_tmp = objective_tmp + penalization_tmp
    dLW_pen = 2 * lambda_pen * np.maximum(0, violation_tmp) * grad_violation_tmp

    # ===
    # Full Gradient
    # ===
    dLW = dLW_obj + dLW_pen

    if use_adam:
      # ADAM update
      t_adam += 1
      m_adam = beta1_adam * m_adam + (1 - beta1_adam) * dLW
      v_adam = beta2_adam * v_adam + (1 - beta2_adam) * (dLW**2)
      m_hat = m_adam / (1 - beta1_adam**t_adam)
      v_hat = v_adam / (1 - beta2_adam**t_adam)
      dLW = m_hat / (np.sqrt(v_hat) + eps_adam)

    # Gradient clipping
    dLW = np.clip(dLW, clipval_L_grad_low, clipval_L_grad_high)

    # L matrix update
    L_var_prev = np.copy(L_var)
    if not use_decreasing_step_size:
      L_var = L_var - learning_rate * dLW
    else:
      lr_now = learning_rate / (1 + ITERATION/n_iter_max)
      L_var = L_var - lr_now * dLW

    # Ensure L is stable (clip eigenvalues)
    L_var = stabilize_L(L_var)

    # ===
    # PLUG NEW L MATRIX INTO THE CONIC PROBLEM
    # ===
    # Update the constraint matrix of the Conic Problem
    A_cp = set_L(A_cp, L_var)


    # ===
    # Convergence Check
    # ===
    stop_iterating = False
    stop_iter_reasons = []

    # Stopping criteria
    if ITERATION > 0:
      diff_obj = np.abs(objective_prev - objective_tmp)
      diff_pen_obj = np.abs(penalized_obj_prev - penalized_obj_tmp)
      diff_dec = np.linalg.norm(decision_prev - decision_tmp)
      diff_L = np.linalg.norm(L_var_prev - L_var, ord='fro')
      norm_dL = np.linalg.norm(dLW, ord='fro')
      # check for zero division
      if np.abs(objective_prev) < 1e-15:
        relative_impr_obj = diff_obj
      else:
        relative_impr_obj = (objective_prev - objective_tmp) / np.abs(objective_prev)
      if np.abs(penalized_obj_prev) < 1e-15:
        relative_diff_obj_pen = diff_pen_obj
      else:
        relative_diff_obj_pen = diff_pen_obj / np.abs(penalized_obj_prev)

      if diff_dec < stopping_tols["diff_decision"]:
        print("Optimal decision did not change, stopping optimization.")
        stop_iter_reasons.append("Tolerance: Difference in optimal decision.")
        stop_iterating = True

      if diff_obj < stopping_tols["diff_objective"]:
        print("Non-penalized objective value of upper level did not change, stopping optimization.")
        stop_iter_reasons.append("Tolerance: Difference in objective value.\n")
        stop_iterating = True

      if diff_pen_obj < stopping_tols["diff_objective_pen"]:
        print("Penalized objective did not change, stopping optimization.")
        stop_iter_reasons.append("Tolerance: Difference in penalized objective value.\n")
        stop_iterating = True

      if diff_L < stopping_tols["diff_weightmat"]:
        print("Weight matrix did not change, stopping optimization.")
        stop_iter_reasons.append("Tolerance: Difference in weight matrix.\n")
        stop_iterating = True

      if norm_dL < stopping_tols["weightmat_gradient"]:
        print("Weight matrix gradient is small, stopping optimization.")
        stop_iter_reasons.append("Tolerance: Weight matrix gradient.\n")
        stop_iterating = True

      if relative_impr_obj < stopping_tols["rel_impr_obj"]:
        print("Relative improvement of the objective function is small, stopping optimization.")
        stop_iter_reasons.append("Tolerance: Relative improvement of objective function.\n")
        stop_iterating = True

      if relative_diff_obj_pen < stopping_tols["rel_diff_obj_pen"]:
        print("Relative improvement of the penalized objective function is small, stopping optimization.")
        stop_iter_reasons.append("Tolerance: Difference in relative difference of penalized objective function.\n")
        stop_iterating = True

      if ITERATION == n_iter_max - 1:
        stop_iter_reasons.append("Maximum number of iterations reached.")
        print("Maximum number of iterations reached.")
        stop_iterating = True
    else:
      diff_obj = 0.0
      diff_pen_obj = 0.0
      diff_dec = 0.0
      diff_L = 0.0
      norm_dL = 0.0
      relative_impr_obj = 0.0
      relative_diff_obj_pen = 0.0

    # timing and print intermediate results
    end_time = time.time()
    gradient_step_time_seconds = end_time - start_time

    # ===
    # STORE INTERMEDIATE RESULTS
    # ===
    if (ITERATION % store_every == 0 or stop_iterating):

      print(f"Gradient Step took {gradient_step_time_seconds} seconds.")
      print(f"Objective: {objective_tmp}, Penalization: {penalization_tmp}")

      # L matrices
      L_matrices.append(L_var_prev)
      # Gradients of L
      gradients_L.append(dLW)
      # Objective value
      objective_values.append(objective_tmp)
      # Penalization value
      penalization_values.append(penalization_tmp)

      # Decision vector
      decisions.append(decision_tmp)

      # Gradient step time
      gradient_step_times.append(gradient_step_time_seconds)

      # save the stopping criterion values
      stopping_criteria_tmp = {}
      stopping_criteria_tmp["objective"] = diff_obj
      stopping_criteria_tmp["objective_pen"] = diff_pen_obj
      stopping_criteria_tmp["decision"] = diff_dec
      stopping_criteria_tmp["weightmat"] = diff_L
      stopping_criteria_tmp["weightmat_gradient"] = norm_dL
      stopping_criteria_tmp["rel_impr_obj"] = relative_impr_obj
      stopping_criteria_tmp["rel_diff_obj_pen"] = relative_diff_obj_pen
      stopping_criteria.append(stopping_criteria_tmp)
      # print the status of all stopping criteria
      print_tmp = [["Diff. Objective", diff_obj, "Tol", stopping_tols["diff_objective"]],
                   ["Diff. Penalized Objective", diff_pen_obj, "Tol", stopping_tols["diff_objective_pen"]],
                   ["Diff. Decision", diff_dec, "Tol", stopping_tols["diff_decision"]],
                   ["Diff. Weight Matrix", diff_L, "Tol", stopping_tols["diff_weightmat"]],
                   ["Norm Weight Matrix Gradient", norm_dL, "Tol", stopping_tols["weightmat_gradient"]],
                   ["Rel. Impr. Obj.", relative_impr_obj, "Tol", stopping_tols["rel_impr_obj"]],
                   ["Rel. Impr. Obj. Pen.", relative_diff_obj_pen, "Tol", stopping_tols["rel_diff_obj_pen"]]]
      for row in print_tmp:
        print('{:<25} {:<25} {:<5} {:<10}'.format(*row))

      iterations.append(ITERATION)

    if stop_iterating:
      break


  # return a dictionary with all the results
  results_dict = {}
  results_dict["L_matrices"] = L_matrices
  results_dict["gradients_L"] = gradients_L
  results_dict["iterations"] = iterations
  results_dict["objective_values"] = objective_values
  results_dict["penalization_values"] = penalization_values
  results_dict["decisions"] = decisions
  results_dict["gradient_step_times"] = gradient_step_times
  results_dict["stopping_criteria"] = stopping_criteria
  results_dict["stopping_reasons"] = stop_iter_reasons

  return results_dict
