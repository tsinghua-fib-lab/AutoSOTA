"""
Reference:
    Jha et al., "Enhanced Integrated Gradients:
    improving interpretability of deep learning models using splicing codes as a case study", 2020
"""

from cleanig.explainer.ig import IGExplainer
from cleanig.explainer.path_utils import LatentLinearPathGenerator


class EIGExplainer(IGExplainer):
    """
    Enhanced Integrated Gradients explainer.

    Inherits from IGExplainer but uses LatentLinearPathGenerator.
    The path is computed in latent space (linear interpolation) and decoded to pixel space.
    """

    def __init__(
        self,
        model,
        vae,
        baseline_method='zero',
        num_steps=200,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
    ):
        """
        Initialize EIG explainer.

        Args:
            model: Classifier model to explain
            vae: VAE model with encode() and decode() methods
            baseline_method: Baseline type ('zero')
            num_steps: Number of interpolation steps
            device: Device to run on
            exp_obj: Objective function ('prob' or 'logit')
            preprocess_fn: Function to preprocess inputs for classifier
        """
        self.model = model
        self.vae = vae
        self.baseline_method = baseline_method
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj

        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

        self.path_generator = LatentLinearPathGenerator(
            vae=vae,
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            device=self.device,
            num_steps=self.num_steps,
            use_slerp=False,
        )
