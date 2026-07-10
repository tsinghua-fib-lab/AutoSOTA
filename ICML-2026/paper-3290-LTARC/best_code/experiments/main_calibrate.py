import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import save_utils
from method.learn_weights_logistic import LearnProbabilitiesLogistic
from data.data_loader import DataLoader
from sklearn.calibration import calibration_curve

def main_parser():
    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=10602,
                        help='random seed (default: %(default)s)')
    parser.add_argument('--learn_weights', default="logistic",
                        choices=["logistic"],
                        help="model to estimate the weights")
    parser.add_argument('--dataset', default="star",
                        choices=["synthetic", "synthetic_unmeasured_confounder", "synthetic_rct", "star", "stroke"],
                        help="model to estimate the weights")
    parser.add_argument('--data_split', default=[0.5, 0.25, 0.25], nargs='+', type=float)
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='number of samples in the synthetic data (default: %(default)s)')
    parser.add_argument('--name', type=str, default="res",)

    parser = save_utils.add_specific_args(parser)

    args = parser.parse_args()

    return args

def main():
    args = main_parser()
    out_dir = save_utils.save_setup(args)

    # Load dataset
    data_loader = DataLoader(args.dataset, args.n_samples)


    # Set seed
    seed = args.seed
    np.random.seed(seed)

    # Split data
    df_train, df_test, df_beta, x_names = data_loader.get_data(seed, args.data_split)

    if args.learn_weights == "logistic":
        weights_data = LearnProbabilitiesLogistic(df_train[x_names], df_train["a"], 0)
    p_a_x_all = weights_data.get_p_a_x(df_test[x_names], 1)
    odds_all = p_a_x_all / (1 - p_a_x_all)
    sort_id = np.argsort(odds_all)
    ecdf_y = np.arange(1, len(sort_id) + 1) / len(sort_id)


    prob_true_test, prob_pred_test = calibration_curve(df_test["a"], p_a_x_all, n_bins=5, strategy='quantile')
    plt.figure()
    plt.plot(prob_pred_test/(1-prob_pred_test), prob_true_test/(1-prob_true_test), marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.show()

    # Save results
    result_df = {
        "odds_true": prob_true_test/(1-prob_true_test),
        "odds_pred": prob_pred_test/(1-prob_pred_test),
    }
    df = pd.DataFrame(result_df)
    df.to_csv(save_utils.get_full_path(out_dir, 'calibration_curve.csv'), index=False)

    df = pd.DataFrame({"ecdf_y": ecdf_y})
    plt.figure()
    for x_names_i in x_names:
        # List without the current feature
        x_names_without_i = [x for x in x_names if x != x_names_i]

        if args.learn_weights == "logistic":
            weights_data_miss = LearnProbabilitiesLogistic(df_train[x_names_without_i], df_train["a"], 0)

        p_a_x_missing = weights_data_miss.get_p_a_x(df_test[x_names_without_i], 1)
        odds_missing = p_a_x_missing / (1 - p_a_x_missing)

        div = odds_all[sort_id] / odds_missing[sort_id]

        div_sorted = np.sort(div)
        df['{}'.format(x_names_i)] = div_sorted

        if np.min(div) < 0.8 or np.max(div) > 1/0.8:
            print(x_names_i)
            #plt.plot(div_sorted, ecdf_y, linestyle="none")
            plt.ecdf(div, complementary=True)

    plt.show()
    df.to_csv(save_utils.get_full_path(out_dir, 'missing_odds.csv'), index=False)
if __name__ == "__main__":
    main()