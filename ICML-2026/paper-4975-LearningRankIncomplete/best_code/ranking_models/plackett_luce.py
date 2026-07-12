import numpy as np
from .rum import RUMRankingModel

class PlackettLuceRankingModel(RUMRankingModel):
    def __init__(self, pl_params: np.ndarray):
        """
        Initializes a Plackett-Luce ranking model.

        The Plackett-Luce model is a special case of a Random Utility Model (RUM)
        where the utilities are the logarithm of the Plackett-Luce parameters and
        the noise follows a Gumbel distribution.

        Args:
            pl_params: A numpy array representing the Plackett-Luce parameters.
        """
        utilities = np.log(pl_params)
        
        def gumbel_noise(size: int) -> np.ndarray:
            return np.random.gumbel(size=size)
            
        super().__init__(utilities=utilities, noise_distribution=gumbel_noise)
