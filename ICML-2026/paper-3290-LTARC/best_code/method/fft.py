import logging
import numpy as np

from method import weights_risk_control
from method import weights_p_bound

from method.create_fft_constraint import FftPolicyConstraint

def get_conformal_risk(n, risk, B):
    """Compute the conformal risk upper bound."""
    return n / (1 + n) * risk + B / (1 + n)

def run_fft(df_train, args, x_names, weighter):
    """Train FFT policies over a grid of constraint values and lookahead depths."""

    fft = FftPolicyConstraint(df_train, weighter, 1, x_names, n_bins=args.n_bins)

    lambda_list = np.linspace(args.lambda_range[1], args.lambda_range[0], args.lambda_n)

    policy_list = []
    for look_ahead in args.lookahead:
        max_constraint = 1000
        for k, lambda_i in enumerate(lambda_list):
            logging.debug(k)
            logging.debug("Max, lambda:", max_constraint, lambda_i)
            logging.debug('-----------------------------')
            if max_constraint > lambda_i:
                logging.debug('running fft')
                fft.constraint_value = lambda_i
                policy, _, _, max_constraint = fft.train_policy(args.n_splits, look_ahead)
                policy_list.append(policy)
    return fft, policy_list

def calibrate_fft_risk(df_test, df_beta, args, x_names, p_a_x, fft, policy_list):
    """Select the best policy per constraint level using conformal risk control."""
    n_constraints = len(args.constraint_values)
    min_obj_list = [1000] * n_constraints
    constr_list = [1000] * n_constraints
    min_policy = [-1] * n_constraints

    weighter_test = weights_risk_control.WeightsDecision(df_test, x_names, args.gamma, p_a_x, args.alpha)
    weighter_beta = weights_risk_control.WeightsDecision(df_beta, x_names, args.gamma, p_a_x, args.alpha)
    for policy in policy_list:
        a_new = fft.get_decisions_policy(policy, df_test)
        obj, constr, n_test = weighter_test.get_obj_and_constr(a_new)
        n = len(policy) - 1
        obj = obj + 0.0002 * n
        if len(policy) == 1 and a_new[0] == 0:
            cf_constr = 0
        else:
            a_new = fft.get_decisions_policy(policy, df_beta)
            max_Z, _ = weighter_beta.get_b(a_new)
            cf_constr = get_conformal_risk(n_test, constr, max_Z)

        for ii, beta in enumerate(args.constraint_values):
            if cf_constr < beta and obj < min_obj_list[ii]:
                min_obj_list[ii] = obj
                constr_list[ii] = constr if cf_constr > 0 else 0
                min_policy[ii] = policy

    return min_policy, min_obj_list, constr_list

def calibrate_fft_synthetic_bound(df_test, args, weighter_test, fft, policy_list):
    """Select the best policy per constraint level using Hoeffding-Bentkus bound (synthetic data)."""
    n_constraints = len(args.constraint_values)
    min_obj_list = [1000] * n_constraints
    constr_list = [1000] * n_constraints
    min_policy = [-1] * n_constraints

    for policy in policy_list[::-1]:
        a_new = fft.get_decisions_policy(policy, df_test)
        obj, constr, _ = weighter_test.get_obj_and_constr(a_new)
        x_0 = fft.get_first_x0_split(policy)

        x = 1 / (1 + np.exp(-(0.5 - x_0)))
        if args.dataset == "synthetic_unmeasured_confounder" or "synthetic_rct":
            x = x / (2 - x)

        max_Z = 10000
        if x_0 > 0:
            max_Z = 1 / (x_0 * x)
        bound = weighter_test.get_hoeffding(a_new, x_0, max_Z)
        if len(policy) == 1 and a_new[0] == 0:
            bound = 0
        for ii, beta in enumerate(args.constraint_values):
            if bound < beta and obj < min_obj_list[ii]:
                min_obj_list[ii] = obj
                constr_list[ii] = constr
                min_policy[ii] = policy

    return min_policy, min_obj_list, constr_list

def calibrate_fft_bound(df_test, args, x_names, p_a_x, fft, policy_list):
    """Select the best policy per constraint level using Hoeffding-Bentkus bound (real data)."""
    n_constraints = len(args.constraint_values)
    min_obj_list = [1000] * n_constraints
    constr_list = [1000] * n_constraints
    min_policy = [-1] * n_constraints

    weighter_test = weights_p_bound.WeightsDecision(df_test, x_names, args.gamma, p_a_x, args.alpha)
    for policy in policy_list:
        a_new = fft.get_decisions_policy(policy, df_test)
        obj, constr, _ = weighter_test.get_obj_and_constr(a_new)
        hoeffding_bound = weighter_test.get_hoeffding(a_new)
        bernstein_bound = weighter_test.get_empirical_bernstein_bound(a_new)
        bound = min(hoeffding_bound, bernstein_bound)
        if len(policy) == 1 and a_new[0] == 0:
            bound = 0
        for ii, beta in enumerate(args.constraint_values):
            if bound < beta and obj < min_obj_list[ii]:
                min_obj_list[ii] = obj
                constr_list[ii] = constr
                min_policy[ii] = policy

    return min_policy, min_obj_list, constr_list