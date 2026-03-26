# ---------------------------------------------------------------------------------
# Portions of this file are derived from uni2ts
# - Source: https://github.com/SalesforceAIResearch/uni2ts
# - Paper: Unified Training of Universal Time Series Forecasting Transformers
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import Union
from probts.model.forecaster import Forecaster
from einops import rearrange, repeat 
from probts.model.nn.arch.Moirai_backbone import MoiraiBackbone
from uni2ts.model.moirai.module import MoiraiModule
import sys

class Moirai(Forecaster):
    def __init__(
        self,
        variate_mode: str = 'M',
        patch_size: Union[str, int] = 'auto',
        model_size: str = 'base',
        scaling: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variate_mode = variate_mode
        self.patch_size = patch_size if patch_size == 'auto' else int(patch_size)
        
        if type(self.prediction_length) == list:
            self.prediction_length = max(self.prediction_length)

        if type(self.context_length) == list:
            self.context_length = max(self.context_length)
        
        # Load pretrained model
        self.no_training = True
        self.moirai = MoiraiBackbone(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_size}"),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            patch_size=self.patch_size,
            target_dim=self.target_dim if self.variate_mode == 'M' else 1,
            scaling=scaling
        )

    def forecast(self, batch_data, num_samples=None):
        if self.variate_mode == 'M':
            forecasts = self.moirai(
                past_target=batch_data.past_target_cdf,
                past_observed_target=batch_data.past_observed_values,
                past_is_pad=batch_data.past_is_pad,
                num_samples=num_samples
            )
        elif self.variate_mode == 'S':
            B, L, K = batch_data.past_target_cdf.shape
            forecasts = self.moirai(
                past_target=rearrange(batch_data.past_target_cdf, 'b l k -> (b k) l').unsqueeze(-1),
                past_observed_target=rearrange(batch_data.past_observed_values, 'b l k -> (b k) l').unsqueeze(-1),
                past_is_pad=repeat(batch_data.past_is_pad, 'b l -> (b k) l', k=K),
                num_samples=num_samples
            )
            forecasts = forecasts.squeeze(-1)
            forecasts = rearrange(forecasts, '(b k) n l -> b n l k', b=B, k=K)
        else:
            raise ValueError(f"Unknown variate mode: {self.variate_mode}")
        return forecasts
