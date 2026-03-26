import torch
from einops import repeat
from probts.model.forecaster import Forecaster


class MeanForecaster(Forecaster):
    def __init__(
        self,
        global_mean: torch.Tensor,
        mode: str = 'batch',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.global_mean = global_mean
        self.mode = mode
        self.no_training = True

    @property
    def name(self):
        return self.mode + self.__class__.__name__
        
    def forecast(self, batch_data, num_samples=None):
        B = batch_data.past_target_cdf.shape[0]
        if self.mode == 'global':
            outputs = self.global_mean.clone()
        elif self.mode == 'batch':
            outputs = torch.mean(batch_data.past_target_cdf, dim=1)
            outputs = torch.mean(outputs, dim=0)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
            
        outputs = repeat(outputs,'d -> b n l d', b=B, n=1, l=self.prediction_length)
        return outputs
