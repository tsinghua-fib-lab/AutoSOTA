import argparse
import parse
import os
from experiments.utils import load_config
from main_func import run_comparison_parallel
from utils import get_minority_majority_data, get_run_description, is_classification_dataset
from utils_plot import filter_and_plot


def main(config_path, save_path, **kwargs):
    real_data_params, params = load_config(config_path)
    for i, curr_params in enumerate(params):
        # override parameters
        for k,v in kwargs.items():
            if k in curr_params.keys():
                type_ = type(curr_params[k])
                if type_ is list:
                    type_ = type(curr_params[k][0])
                if isinstance(v, list):
                    v_ = [type_(elem) for elem in v]
                    curr_params[k] = v_
                else:
                    curr_params[k] = type_(v)
            else:
                curr_params[k] = v

        if curr_params['n_cal'] + 1 >= curr_params['n_cal_maj']:
            raise ValueError('The size of synthetic calibration set must be larger than n_cal+1.')
        # cqr can be applied only with one alpha level (correspond to the model)
        if real_data_params['model_name'] == 'cqr':
            # update alpha to the fitted quantile regression level
            args = parse.parse('{dataset_name}_alpha_{alpha}', curr_params['dataset'])
            curr_params['alpha'] = [float(args['alpha'])]
        # results dir
        curr_save_path = os.path.join(save_path, get_run_description(**curr_params, **real_data_params))
        os.makedirs(curr_save_path + '/results/', exist_ok=True)
        if i == 0 or curr_params['dataset'] != params[i-1]['dataset']:
            X_minority, y_minority, X_majority, y_majority = get_minority_majority_data(curr_params['dataset'],
                                                                                        real_data_params['model_name'],
                                                                                        real_data_params['dataset_path'],
                                                                                        real_data_params['dataset_maj_path'])
        results = run_comparison_parallel(X_minority=X_minority, y_minority=y_minority,
                                            X_majority=X_majority, y_majority=y_majority,
                                            save_path=curr_save_path,
                                            **curr_params, **real_data_params)
        # save results
        results.to_pickle(curr_save_path + '/results/results.pkl')
        # plot results
        os.makedirs(curr_save_path + '/plots/', exist_ok=True)
        subset_exp = 'SPI [subset]' in curr_params['methods']
        conditional = curr_params.get('class_conditional', False)
        for alpha_ in curr_params['alpha']:
            os.makedirs(curr_save_path + '/plots/' + f'alpha_{alpha_}/', exist_ok=True)
            filter_and_plot(save_path=curr_save_path + '/plots/' + f'alpha_{alpha_}/', results=results,
                            x='Class' if is_classification_dataset(curr_params['dataset']) and curr_params.get('class_conditional', False) else None, 
                            methods2plot=curr_params['methods'], conditional=conditional,
                            to_table=False, subset_exp=subset_exp,
                            k=curr_params.get('k', None), alpha=alpha_, beta=curr_params.get('beta', None))
            if conditional:  # plot table
                filter_and_plot(save_path=curr_save_path + '/plots/' + f'alpha_{alpha_}/', results=results,
                                x='Class' if is_classification_dataset(curr_params['dataset']) and curr_params.get('class_conditional', False) else None, 
                                methods2plot=curr_params['methods'], conditional=curr_params.get('class_conditional', False),
                                to_table=True, subset_exp=subset_exp,
                                k=curr_params.get('k', None), alpha=alpha_, beta=curr_params.get('beta', None))


def get_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("-c", "--config_path", help="Path to configuration YAML file")
    parser.add_argument("-s", "--save_path", required=True, help="Path to save results")
    parser.add_argument('kwargs', nargs="*", help="Additional key-value pairs (override config)")

    args = parser.parse_args()
    kwargs = dict(arg.split('=') for arg in args.kwargs if '=' in arg)
    if args.config_path is not None and not os.path.exists(args.config_path):
        raise ValueError('Config file does not exist.')
    return args, kwargs


if __name__ == "__main__":
    args, kwargs = get_args()
    main(args.config_path, args.save_path, **kwargs)
