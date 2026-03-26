# ---------------------------------------------------------------------------------
# Portions of this file are derived from granite-tsfm
# - Source: https://github.com/ibm-granite/granite-tsfm
# - Paper: Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from probts.model.forecaster import Forecaster

from submodules.tsfm.tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction


class TinyTimeMixer(Forecaster):
    """
    TinyTimeMixer from https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb
    prediction length originally 96
    context length originally 512
    changes might cause degradation in performance
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.no_training = True

        # TTM model branch
        # Use main for 512-96 model
        # Use "1024_96_v1" for 1024-96 model
        TTM_MODEL_REVISION = "main"
        
        if (type(self.context_length).__name__=='list'):
            context_length = max(context_length)
            
        if (type(self.prediction_length).__name__=='list'):
            prediction_length = max(prediction_length)

        self.zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm/TTM", revision=TTM_MODEL_REVISION
        )
        

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = inputs[:, -self.context_length:]
        B, _, K = inputs.shape 
        # past_target = batch_data.past_target_cdf[:, -self.context_length:]
        self.zeroshot_model.eval()
        point_forecast = self.zeroshot_model.forward(inputs).prediction_outputs
        return point_forecast.unsqueeze(1)
