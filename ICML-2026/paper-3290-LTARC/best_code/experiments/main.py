import argparse
import logging
import numpy as np

from utils import save_utils

from method.learn_weights_logistic import LearnProbabilitiesLogistic
from data.data_loader import DataLoader
from utils.results import save_results
from method.fft import run_fft, calibrate_fft_risk, calibrate_fft_synthetic_bound, calibrate_fft_bound
from method.create_fft_constraint import FftPolicyConstraint

from method import weights_risk_control
from method import weights_p_bound

def main_parser():
    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=10603,
                        help='random seed (default: %(default)s)')
    parser.add_argument('--n_splits', default=2, type=int,
                        help='list of number of splits to build a policy for (default: %(default)s)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma value in odds bounds (default: %(default)s)')
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--constraint_values', default=[0.05, 0.1, 0.15, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4], nargs='+',
                        type=float,
                        help='upper limit of constraint (default: %(default)s)')
    parser.add_argument('--lambda_range', default=[0.01, 0.80], nargs='+', type=float,
                        help='upper limit of constraint (default: %(default)s)')
    parser.add_argument('--lambda_n', default=100, type=int,
                        help='upper limit of constraint (default: %(default)s)')
    parser.add_argument('--learn_weights', default="true",
                        choices=["logistic", "true"],
                        help="model to estimate the weights")
    parser.add_argument('--dataset', default="synthetic",
                        choices=["synthetic", "synthetic_unmeasured_confounder", "synthetic_rct", "star", "stroke"],
                        help="model to estimate the weights")
    parser.add_argument('--guarantee', default="p",
                        choices=["p", "average"])
    parser.add_argument('--data_split', default=[0.5, 0.25, 0.25], nargs='+', type=float)
    parser.add_argument('--n_mc', type=int, default=1,
                        help='number of Monte Carlo simulations (default: %(default)s)')
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='number of samples in the synthetic data (default: %(default)s)')
    parser.add_argument('--name', type=str, default="res",)

    parser = save_utils.add_specific_args(parser)
    parser = FftPolicyConstraint.add_model_specific_args(parser)

    args = parser.parse_args()

    return args

def main():
    args = main_parser()
    out_dir = save_utils.save_setup(args)

    true_constraint_all = np.zeros((args.n_mc, len(args.constraint_values)))
    true_obj_all = np.zeros((args.n_mc, len(args.constraint_values)))

    # Load dataset
    data_loader = DataLoader(args.dataset, args.n_samples)

    for i in range(args.n_mc):
        if i % 10 == 0:
            print("MC:", i)
        # Set seed
        seed = args.seed + i
        np.random.seed(seed)

        # Split data
        df_train, df_test, df_beta, x_names = data_loader.get_data(seed, args.data_split)

        if args.learn_weights == "true":
            p_a_x = data_loader.get_func_p_a_x()
        elif args.learn_weights == "logistic":
            weights_data = LearnProbabilitiesLogistic(df_train[x_names], df_train["a"], 0)
            p_a_x = weights_data.get_p_a_x

        if args.guarantee == "average":
            weighter_train = weights_risk_control.WeightsDecision(df_train, x_names, args.gamma, p_a_x, args.alpha)
        elif args.guarantee == "p":
            weighter_train = weights_p_bound.WeightsDecision(df_train, x_names, args.gamma, p_a_x, args.alpha)

        fft, policy_list = run_fft(df_train, args, x_names, weighter_train)

        if args.guarantee == "average":
            min_policy, min_obj_list, constr_list = calibrate_fft_risk(df_test, df_beta, args, x_names, p_a_x, fft, policy_list)
        elif args.guarantee == "p":
            if args.dataset == "synthetic" or args.dataset == "synthetic_unmeasured_confounder":
                weighter_test = weights_p_bound.WeightsDecision(df_test, x_names, args.gamma, p_a_x, args.alpha)
                min_policy, min_obj_list, constr_list = calibrate_fft_synthetic_bound(df_test, args, weighter_test, fft, policy_list)
            elif args.dataset == "synthetic_rct":
                weighter_test = weights_p_bound.WeightsSampling(df_test, x_names, args.gamma, data_loader.get_func_p_s1(), p_a_x, data_loader.get_func_p_s_x(), args.alpha)
                min_policy, min_obj_list, constr_list = calibrate_fft_synthetic_bound(df_test, args, weighter_test, fft,
                                                                                      policy_list)
            else:
                min_policy, min_obj_list, constr_list = calibrate_fft_bound(df_test, args, x_names, p_a_x, fft, policy_list)

        logging.info('Results:')
        for beta, policy, obj, constr in zip(args.constraint_values, min_policy, min_obj_list, constr_list):
            logging.info('beta: %s', beta)
            fft.log_policy(policy)
            logging.info('Objective: %s', obj)
            logging.info('Constraint: %s', constr)

        logging.info('Results tested:')
        if args.dataset == "synthetic" or args.dataset == "synthetic_unmeasured_confounder" or args.dataset == "synthetic_rct":
            for ii, beta in enumerate(args.constraint_values):
                x_split = fft.get_first_x0_split(min_policy[ii])
                if x_split == 0:
                    print("No split found, beta:", beta)
                if args.dataset == "synthetic":
                    true_constraint_all[i, ii] = 0.5 * x_split / 2
                    true_obj_all[i, ii] = 0.5 * x_split * x_split / 2 + (1 - x_split) * 0.8
                elif args.dataset == "synthetic_rct":
                    a_new = fft.get_decisions_policy(min_policy[ii], df_beta)
                    y = data_loader.gen_new_y(df_beta["x0"].to_numpy(), df_beta["u"].to_numpy(), a_new)
                    true_obj_all[i, ii] = np.mean(y)
                    if np.sum(a_new) == 0:
                        true_constraint_all[i, ii] = 0
                    else:
                        true_constraint_all[i, ii] = np.mean(y[a_new == 1])
                else:
                    true_constraint_all[i, ii] = 0.55 * x_split / 2
                    true_obj_all[i, ii] = 0.55 * x_split * x_split / 2 + (1 - x_split) * 0.8
        else:
            if args.guarantee == "average":
                weighter_beta = weights_risk_control.WeightsDecision(df_beta, x_names, args.gamma, p_a_x, args.alpha)
            elif args.guarantee == "p":
                weighter_beta = weights_p_bound.WeightsDecision(df_beta, x_names, args.gamma, p_a_x, args.alpha)
            for ii, beta in enumerate(args.constraint_values):
                a_new = fft.get_decisions_policy(min_policy[ii], df_beta)
                obj, constr, n_test = weighter_beta.get_obj_and_constr(a_new)
                if np.sum(a_new) == 0:
                    constr = 0
                true_constraint_all[i, ii] = constr
                true_obj_all[i, ii] = obj

                logging.info('beta: %s', beta)
                logging.info('Objective: %s', obj)
                logging.info('Constraint: %s', constr)

    mean_true_obj, mean_true_constraint = save_results(out_dir, args, true_obj_all, true_constraint_all)

    print("Samples, list beta: ", "\t", args.n_samples, args.constraint_values)
    print("Mean true obj:", "\t", mean_true_obj)
    print("Mean true constraint:", "\t", mean_true_constraint)

if __name__ == "__main__":
    main()