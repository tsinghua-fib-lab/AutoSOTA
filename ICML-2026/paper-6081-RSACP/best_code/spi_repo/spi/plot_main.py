import argparse
import matplotlib.pyplot as plt
from utils_plot import load_and_plot, plot4paper_w_marginal


def plot_w_marginal(plot_dir, results_dir_marg, results_dir_cond, x, methods2plot, alpha, beta=0.4, **kwargs):
    subset_exp = 'SPI [subset]' in methods2plot
    r_marg,_ = load_and_plot(plot_dir, results_dir_marg, None, methods2plot=methods2plot, conditional=False, only_results_bounds=True,
                             subset_exp=subset_exp, alpha=alpha, beta=beta, **kwargs)
    
    r, b = load_and_plot(plot_dir, results_dir_cond, x, methods2plot=methods2plot, conditional=True, only_results_bounds=True,
                         subset_exp=subset_exp, alpha=alpha, beta=beta, **kwargs)

    plot4paper_w_marginal(plot_dir, r_marg, r, methods2plot=methods2plot, x=x, y='Conditional Coverage', y_marg='Coverage', alpha=alpha, bounds=b,
                            showmeans=True, sharey=False, split_size=False, subset_exp=subset_exp)
    plt.close()
    split_size = alpha == 0.02 or alpha == 0.05
    min_y_upper, max_y_lower = 29.5, 15 
    plot4paper_w_marginal(plot_dir, r_marg, r, methods2plot=methods2plot, x=x, y='Conditional Length', y_marg='Length', alpha=alpha, bounds=b,
                            showmeans=True, sharey=False, split_size=split_size, min_y_upper=min_y_upper, max_y_lower=max_y_lower, subset_exp=subset_exp)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Plot MEPS (as a function of age range) or ImageNet (marginal alongside conditional) experiments.")
    parser.add_argument('--meps', action='store_true', help='Plot MEPS experiment (as a function of age range).')
    parser.add_argument("-p", "--plot_dir", required=True, help="Path to save plots")
    parser.add_argument("--beta", type=float, default=0.4, help="Beta parameter.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Miscoverage level.")
    parser.add_argument('--subset_exp', action='store_true', help='Plot experiment with SPI [subset]. Note: you must pass k={} when using this flag.')
    parser.add_argument('kwargs', nargs="*", help="Additional key-value pairs to filter results when the result directories contain multiple parameter configurations.")
    args, remaining_args = parser.parse_known_args()
    if args.meps:
        parser.add_argument("-r", "--results_dir", required=True, help="Path to the directory containing all age ranges results.")
    else:
        parser.add_argument("-r_cond", "--results_dir_cond", required=True, help="Path to the directory containing class-conditional results.")
        parser.add_argument("-r_marg", "--results_dir_marg", required=True, help="Path to the directory containing marginal results.")

    args = parser.parse_args()
    kwargs = dict(arg.split('=') for arg in args.kwargs if '=' in arg)
    return args, kwargs


if __name__ == "__main__":
    args, kwargs = get_args()
    if args.subset_exp:
        methods2plot = ['SC', 'SPI', 'SPI [subset]']
    else:
        methods2plot = ['SC', 'SC (synth)', 'SPI']
    if args.meps:
        load_and_plot(args.plot_dir, args.results_dir, 'age_range', methods2plot=methods2plot, conditional=False,
                      subset_exp=args.subset_exp, alpha=args.alpha, beta=args.beta, **kwargs)
    else:
        plot_w_marginal(args.plot_dir, args.results_dir_marg, args.results_dir_cond, x='Class', methods2plot=methods2plot, alpha=args.alpha, beta=args.beta, **kwargs)
