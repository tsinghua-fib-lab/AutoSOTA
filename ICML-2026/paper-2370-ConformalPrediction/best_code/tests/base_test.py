import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
# from copy import deepcopy
import pandas as pd
# from core import standard_weighted_quantile, trailing_window, aci, aci_clipped, quantile, quantile_integrator_log, quantile_integrator_log_scorecaster
from core.methods import ScaleFreeOnlineGradientDescent, KrichevskyTrofimov, UniversalPortfolio, DynamicallyTunedAdaptiveConformalInference
# from core.synthetic_scores import generate_scores

from core.model_scores import generate_forecasts, make_predictions_on_ensemble
from core.volatility import compute_volatility_series
from datasets import load_dataset, generate_synthetic_ensemble
from core.runner import run_conformal_inference, run_conformal_inference_on_ensemble
# from darts import TimeSeries
import yaml
import pickle
# import pdb

# def predict_sets_on_ensemble(dataset, model, conformal_predictor, T_burnin, n_runs):
#     results = []
#     for _ in range(n_runs):
#         # Generate fresh data, model, and predictor for each trial
#         data = generate_synthetic_data(dataset)
        
#         # Reinitialize model and predictor in case they have memory effects
#         fresh_model = deepcopy(model)
#         fresh_conformal_predictor = deepcopy(conformal_predictor)

#         predictions = make_predictions(data, fresh_model)
#         scores = np.abs(data[1] - predictions)
#         # Run single experiment
#         result = run_conformal_inference(scores, fresh_conformal_predictor, T_burnin)
#         result['y'] = data[1]
#         results.append(result)
    
#     # Aggregate results
#     all_predicted_scores = np.array([r["predicted_scores"] for r in results])
#     all_coverages = np.array([r["coverages"] for r in results])
#     all_ys = np.array([r["y"] for r in results])
#     mean_predicted_scores = np.mean(all_predicted_scores, axis=0)
#     mean_coverages = np.mean(all_coverages, axis=0)
#     mean_ys = np.mean(all_ys, axis=0)
#     # mean_sets = [np.array([ mean_ys[i] - mean_predicted_scores[i], mean_ys[i] + mean_predicted_scores[i] ]) for i in range(len(mean_ys))]
#     return pd.DataFrame({
#         'predicted_score': mean_predicted_scores,
#         'coverage': mean_coverages,
#         'y': mean_ys,
#         # 'set': mean_sets,
#     })
 
def aggregate_ensemble(ensemble):
    aggregated = {}
    for key in ensemble[0].keys():
        values_list = [data[key] for data in ensemble]
        aggregated[key] = np.mean(values_list, axis=0)
    return aggregated
   
if __name__ == "__main__":
    json_name = sys.argv[1]
    # json_name = "/Users/liut0c/research/up-for-ocp/tests/configs/changepoint.yaml"
    if len(sys.argv) > 2:
        overwrite = sys.argv[2].split(",")
    else:
        overwrite = []
    args = yaml.safe_load(open(json_name))
    # Set up folder and filename
    foldername = './results/'
    config_name = json_name.split('.')[-2].split('/')[-1]
    filename = foldername + config_name + ".pkl"
    os.makedirs(foldername, exist_ok=True)
    real_data = args['real']
    quantiles_given = args['quantiles_given']
    multiple_series = args['multiple_series']
    score_function_name = args['score_function_name'] 
    model_names = args['sequences'][0]['model_names'] 
    ahead = args['ahead'] if real_data and 'ahead' in args.keys() else 1
    minsize = args['minsize'] if real_data and 'minsize' in args.keys() else 0
    asymmetric = False
    seed = args['seed'] if 'seed' in args.keys() else None
    n_seeds = args['n_seeds'] if 'n_seeds' in args.keys() else 5
    # Try reading in results
    try:
        with open(filename, 'rb') as handle:
            all_results = pickle.load(handle)
    except:
        all_results = {}

    for model_name in model_names:
        try:
            results = all_results[model_name]
        except:
            results = {}
        # Initialize the score function
        score_function_name = args['score_function_name']
        if score_function_name == "absolute-residual":
            def score_function(y, forecast):
                return np.abs(y - forecast)
            def set_function(forecast, q):
                return np.array([forecast - q, forecast + q])
        elif score_function_name == "signed-residual":
            def score_function(y, forecast):
                return np.array([forecast - y, y - forecast])
            def set_function(forecast, q):
                return np.array([forecast - q[0], forecast + q[1]])
            asymmetric = True
        elif score_function_name == "cqr-symmetric":
            def score_function(y, forecasts):
                return np.maximum(forecasts[0] - y, y - forecasts[-1])
            def set_function(forecast, q):
                return np.array([forecast[0] - q, forecast[-1] + q])
        elif score_function_name == "cqr-asymmetric":
            def score_function(y, forecasts):
                return np.array([forecasts[0] - y, y - forecasts[-1]])
            def set_function(forecast, q):
                return np.array([forecast[0] - q[0], forecast[-1] + q[1]])
            asymmetric = True
        elif score_function_name == "volatility-normalized-absolute-residual":
            # Uses same functions as absolute-residual but scores are
            # divided by local volatility in the score computation step
            def score_function(y, forecast):
                return np.abs(y - forecast)
            def set_function(forecast, q):
                return np.array([forecast - q, forecast + q])
        else:
            raise ValueError("Invalid score function name")

        # Pre-compute volatility series for volatility-normalized scores
        vol_normalized = (score_function_name == "volatility-normalized-absolute-residual")
        if vol_normalized:
            vol_half_life = args.get('volatility_half_life', 20)
            _prices_raw = None  # filled after data load

        # Get dataframe and add forecasts and scores to it
        if real_data:
            data = load_dataset(args['sequences'][0]['dataset'])
        else:
            # The only randomness that comes in
            np.random.seed(seed)
            data_ensemble = generate_synthetic_ensemble(config_name, n_seeds)
            data = pd.DataFrame() # Write later
            data['y'] = aggregate_ensemble(data_ensemble)['y']
        # Get the forecasts
        if 'forecasts' not in data.columns:
            os.makedirs('./datasets/processed/', exist_ok=True)
            os.makedirs('./datasets/processed/' + config_name, exist_ok=True)
            args['sequences'][0]['savename'] = './datasets/processed/' + config_name +  '/' + model_name + '.npz'
            args['sequences'][0]['T_burnin'] = args['T_burnin']
            args['sequences'][0]['ahead'] = ahead
            args['sequences'][0]['model_name'] = model_name
            if real_data:
                data['forecasts'] = generate_forecasts(data, **args['sequences'][0])
            else:
                predictions_ensemble = make_predictions_on_ensemble(data_ensemble, **args['sequences'][0])
                data['forecasts'] = aggregate_ensemble(predictions_ensemble)['predictions']
        # Compute scores
        if 'scores' not in data.columns:
            if vol_normalized:
                # Volatility-normalized scores: |y - f| / (price * sigma + eps)
                prices = data['y'].interpolate().to_numpy()
                _volatility = compute_volatility_series(prices, half_life=vol_half_life)
                forecasts_arr = data['forecasts'].interpolate().to_numpy()
                data['scores'] = [
                    np.abs(y - forecast) / max(price * vol + 1e-8, 1e-8)
                    for y, forecast, price, vol in zip(data['y'], forecasts_arr, prices, _volatility)
                ]
            elif real_data:
                data['scores'] = [ score_function(y, forecast) for y, forecast in zip(data['y'], data['forecasts']) ]
            else:
                scores_ensemble = []
                for d, p in zip(data_ensemble, predictions_ensemble):
                    scores_ensemble.append({'scores': [ score_function(y, prediction) for y, prediction in zip(d['y'], p['predictions']) ] })
                data['scores'] = aggregate_ensemble(scores_ensemble)['scores']
            # scores_list = []
            # for key in args['sequences'].keys():
            #     scores_list += [generate_scores(**args['sequences'][key])]
            # scores = np.concatenate(scores_list).astype(float)
            # # Make a pandas dataframe with a datetime index and the scores in their own column called `scores'.
            # data = pd.DataFrame({'scores': scores}, index=pd.date_range(start='1/1/2018', periods=len(scores), freq='D'))
        
        # Loop through each method and learning rate, and compute the results
        for method in args['methods'].keys():
            if (method in results.keys()) and (method not in overwrite):
                continue
            fn = None
            if method == "SFOGD":
                fn = ScaleFreeOnlineGradientDescent
            elif method == "KT":
                fn = KrichevskyTrofimov
                args['methods'][method]['lrs'] = [None]
            elif method == "UP":
                fn = UniversalPortfolio
                args['methods'][method]['lrs'] = [None]
            elif method == "DtACI":
                fn = DynamicallyTunedAdaptiveConformalInference
                args['methods'][method]['lrs'] = [None]
            else:
                raise Exception(f"Method {method} not implemented")
            lrs = args['methods'][method]['lrs']
            # Extract method-specific constructor params BEFORE kwargs gets polluted
            method_params = {k: v for k, v in args['methods'][method].items() if k != 'lrs'}
            kwargs = args['methods'][method]
            kwargs["T_burnin"] = args["T_burnin"]
            kwargs["data"] = data if real_data else None
            kwargs["seasonal_period"] = args["seasonal_period"] if "seasonal_period" in args.keys() else None
            kwargs["config_name"] = config_name
            kwargs["ahead"] = ahead
            # Compute the results
            results[method] = {}
            for lr in lrs:
                if asymmetric:
                    stacked_scores = np.stack(data['scores'].to_list())
                    kwargs['upper'] = False
                    q0 = fn(stacked_scores[:,0], args['alpha']/2, lr, **kwargs)['q']
                    kwargs['upper'] = True
                    q1 = fn(stacked_scores[:,1], args['alpha']/2, lr, **kwargs)['q']
                    q = [ np.array([q0[i], q1[i]]) for i in range(len(q0)) ]
                else:
                    kwargs['upper'] = True
                    # q = fn(data['scores'].to_numpy(), args['alpha'], lr, **kwargs)['q']
                    if lr is None:
                        # For parameter-free methods
                        predictor = fn(alpha=args['alpha'], **method_params)
                    else:
                        predictor = fn(alpha=args['alpha'], lr=lr, **method_params)
                    if real_data:
                        r = run_conformal_inference(data['scores'].to_numpy(), predictor, T_burnin=args['T_burnin'])
                        q = r['predicted_scores']
                        c = r['coverages']
                    else:
                        r_ensemble = run_conformal_inference_on_ensemble(scores_ensemble, predictor, T_burnin=args['T_burnin'])
                        q = aggregate_ensemble(r_ensemble)['predicted_scores']
                        c = aggregate_ensemble(r_ensemble)['coverages'] 
                    
                # Denormalize q for volatility-normalized scores
                if vol_normalized:
                    forecast_arr = data['forecasts'].interpolate().to_numpy()
                    prices_arr = data['y'].interpolate().to_numpy()
                    q = [q[i] * max(prices_arr[i] * _volatility[i], 1e-8) for i in range(len(q))]
                sets = [ set_function(data['forecasts'].interpolate().to_numpy()[i], q[i]) for i in range(len(q)) ]
                # Make sure the set size is at least minsize by setting sets[j][0] = min(sets[j][0], sets[j][1]-minsize) and sets[j][1] = max(sets[j][1], sets[j][1]+minsize)
                sets = [ np.array([np.minimum(sets[j][0], sets[j][1]-minsize), np.maximum(sets[j][1], sets[j][0]+minsize)]) for j in range(len(sets)) ]
                results[method][lr] = { "q": q, "sets": sets, "coverages": c }

        # Save some metadata
        results["scores"] = data['scores']
        results["alpha"] = args['alpha']
        results["T_burnin"] = args['T_burnin']
        results["quantiles_given"] = quantiles_given
        results["multiple_series"] = multiple_series
        results["real_data"] = real_data
        results["score_function_name"] = score_function_name
        results["asymmetric"] = asymmetric

        results["forecasts"] = data['forecasts']
        results["data"] = data
        all_results[model_name] = results

    # Save results
    with open(filename, 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
