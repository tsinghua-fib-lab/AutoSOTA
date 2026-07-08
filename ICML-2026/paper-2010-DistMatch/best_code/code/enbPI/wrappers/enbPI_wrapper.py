import numpy as np
import wandb
from matplotlib import pyplot as plt

from enbPI.wrappers.demo_code_as_py import prediction_interval_with_SPCI
from enbPI.wrappers.enbPI_loader import EnbPI_DATASETS, dataset_depending_model_options
from utils.utils import get_device

class EnbPIBaseWrapper:

    def __init__(self, alpha, model_args, gpu_id=0) -> None:
        super().__init__()
        self.alpha = alpha
        self.fit_sigmaX = model_args['fit_sigmaX']
        self.B = model_args['no_of_bootstrap_batches']
        self.past_window = model_args['past_window']
        self.fit_func = model_args['fit_func']
        self.device = get_device(gpu_id)

    def print_fx_fit_results(self, in_data, fit_results):
        train_len = in_data[2].shape[0]
        test_len = in_data[3].shape[0]

        # Fx All
        #x_predict = np.arange(start=train_len, stop=train_len + test_len)
        #fig, ax = plt.subplots(figsize=(60, 5))
        #ax.plot(torch.concat((in_data[2], in_data[3])).cpu().detach().numpy(),
        #        label=f'Real Y', color='black', linestyle='dashed')
        #ax.plot(x_predict, self.model.Ensemble_pred_interval_centers,
        #        label=f'Fx Ensable', color='red')
        #ax.scatter(np.tile(x_predict, self.B), fit_results[0].reshape(-1,),
        #           c=np.repeat(np.array(range(self.B)), test_len))
        #ax.axvline(in_data[2].shape[0])
        #wandb.log({"All Fx": fig})
        # Fx Test
        fig, ax = plt.subplots(figsize=(60, 5))
        ax.plot(in_data[3].cpu().detach().numpy(), label=f'Real Y', color='black', linestyle='dashed')
        ax.plot(self.model.Ensemble_pred_interval_centers, label=f'Fx Ensemble', color='red')
        ax.scatter(np.tile(np.arange(start=0, stop=test_len), self.B), fit_results[0].reshape(-1,),
                   c=np.repeat(np.array(range(self.B)), test_len))
        wandb.log({"Prediction Fx": wandb.Image(fig)})  # pass as image because incompatibility matlibplot and plotly
        # Show Bootstrap Splitting
        #fig, ax = plt.subplots(figsize=(100, 5))
        #ax.imshow(fit_results[2])
        #wandb.log({"Boostrap Splitting": fig})

    def print_interval(self, in_data, fit_results):
        test_len = in_data[3].shape[0]
        fig, ax = plt.subplots(figsize=(60, 5))
        ax.plot(in_data[3].cpu().detach().numpy(), label=f'Real Y', color='black', linestyle='dashed')
        ax.plot(self.model.Ensemble_pred_interval_centers, label=f'Fx Ensemble', color='red')
        ax.fill_between(np.arange(start=0, stop=test_len), self.model.PIs_Ensemble['lower'],self.model.PIs_Ensemble['upper'])
        wandb.log({"Prediction Interval":  wandb.Image(fig)}) # pass as image because incompatibility matlibplot and plotly


class EnbPIWrapper(EnbPIBaseWrapper):

    def __init__(self, alpha, dataset_name, **kwargs) -> None:
        if dataset_name in EnbPI_DATASETS:
            model_args = dataset_depending_model_options(dataset_name)
        else:
            model_args = kwargs
        super().__init__(alpha, model_args)
        self.model = None

    def fit_and_eval(self, X_train, X_predict, Y_train, Y_predict):
        self.model = prediction_interval_with_SPCI(X_train, X_predict, Y_train, Y_predict, fit_func=self.fit_func)
        fit_results = self.model.fit_bootstrap_models_online(self.B, device=self.device, fit_sigmaX=self.fit_sigmaX)
        self.print_fx_fit_results((X_train, X_predict, Y_train, Y_predict), fit_results)
        self.model.compute_PIs_Ensemble_online(
            self.alpha, smallT=True, past_window=self.past_window, use_quantile_regr=False, quantile_regr='')
        self.print_interval((X_train, X_predict, Y_train, Y_predict), fit_results)

    def get_results(self):
        return self.model.get_results()


class SPICWrapper(EnbPIBaseWrapper):

    def __init__(self, alpha, dataset_name, add_config=None) -> None:
        if dataset_name in EnbPI_DATASETS:
            model_args = dataset_depending_model_options(dataset_name)
        else:
            model_args = add_config
        super().__init__(alpha, model_args)
        self.model = None
        self.use_enbPI_fx = add_config["use_enbPI_fx"]
        self.quantile_reg_retrain = add_config["quantile_reg_retrain"]
        self.quantile_reg_param = add_config.get("quantile_reg_param", {'max_depth': 2, 'random_state': 0})
        self.quantile_with_xt = add_config.get("quantile_with_xt", False)

    def fit_and_eval(self, X_train, X_predict, Y_train, Y_predict):
        self.model = prediction_interval_with_SPCI(X_train, X_predict, Y_train, Y_predict,
                                                   fit_func=self.fit_func if self.use_enbPI_fx else None)
        fit_results = self.model.fit_bootstrap_models_online(self.B, device=self.device, fit_sigmaX=self.fit_sigmaX)
        self.print_fx_fit_results((X_train, X_predict, Y_train, Y_predict), fit_results)
        self.model.compute_PIs_Ensemble_online(
            self.alpha, smallT=False, past_window=self.past_window, use_quantile_regr=True, quantile_regr='RF',
            quantile_reg_retrain=self.quantile_reg_retrain, quantile_reg_param=self.quantile_reg_param,
            quantile_with_Xt=self.quantile_with_xt)
        self.print_interval((X_train, X_predict, Y_train, Y_predict), fit_results)

    def get_results(self):
        return self.model.get_results()
