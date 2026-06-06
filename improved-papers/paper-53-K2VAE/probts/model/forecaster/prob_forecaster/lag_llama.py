# ---------------------------------------------------------------------------------
# Portions of this file are derived from lag-llama
# - Source: https://github.com/time-series-foundation-models/lag-llama
# - Paper: Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import numpy as np
import torch

from gluonts.dataset.common import ListDataset

from probts.model.forecaster import Forecaster
from submodules.lag_llama.lag_llama.gluon.estimator import LagLlamaEstimator


class LagLlama(Forecaster):
    def __init__(
        self,
        use_rope_scaling: bool = True,
        ckpt_path: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # self.ctx_len = kwargs.get('context_length')
        # self.pred_len = kwargs.get('prediction_length')
        
        if type(self.prediction_length) == list:
            self.prediction_length = max(self.prediction_length)
            

        if type(self.context_length) == list:
            self.context_length = max(self.context_length)
            
        self.ctx_len = self.context_length
        self.pred_len = self.prediction_length

        # Load pretrained model
        self.no_training = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location=device)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (self.ctx_len + self.pred_len) / estimator_args["context_length"]), # 32
        }
        # Load model checkpoint
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=self.pred_len,
            context_length=self.ctx_len, # Lag-Llama was trained with a context length of 32, but can work with any context length

            # estimator args
            input_size=estimator_args["input_size"], # 1
            n_layer=estimator_args["n_layer"], # 8
            n_embd_per_head=estimator_args["n_embd_per_head"], # 16
            n_head=estimator_args["n_head"], # 9
            scaling=estimator_args["scaling"], # robust
            time_feat=estimator_args["time_feat"], # True
            rope_scaling=rope_scaling_arguments if use_rope_scaling else None, # long-term set to True

            batch_size=4,
            num_parallel_samples=100,
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self.predictor = estimator.create_predictor(transformation, lightning_module)

    
    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = inputs[:, -self.context_length:]
        datastamps = batch_data.past_time_feat.cpu().numpy().astype('datetime64[s]')

        # for now, we only support batch_size=1
        B, _, K = inputs.shape 
        # past_target = batch_data.past_target_cdf[:, -self.context_length:]
        start_time = datastamps.reshape(-1)[0]
        data = [{"start": start_time, "target": inputs[:,:,i].cpu().squeeze()} for i in range(K)]
        dataset = ListDataset(data, freq='1h')

        forecasts = self.predictor.predict(dataset, num_samples=num_samples)
        samples = [fs.samples for fs in forecasts]
        forecasts = np.array(samples).transpose(1, 2, 0)

        prob_forecast = forecasts[np.newaxis, :, :]
        prob_forecast = torch.tensor(prob_forecast) # shape: b s l k
        
        return prob_forecast
