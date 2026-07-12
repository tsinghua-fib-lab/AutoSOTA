"""
Reference:
    Jha et al., "Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution", ICML 2024
"""

from cleanig.explainer.ig import IGExplainer
from cleanig.explainer.path_utils import GeodesicPathGenerator


class MIGExplainer(IGExplainer):
    """
    Manifold Integrated Gradients explainer.

    Inherits from IGExplainer but uses GeodesicPathGenerator.
    The path is computed as a geodesic on the VAE manifold (energy minimization).
    """

    def __init__(
        self,
        model,
        vae,
        baseline_method='zero',
        num_steps=50,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
        # Geodesic specific parameters
        alpha=0.01,
        max_iterations=10,
        epsilon=1e-5,
    ):
        """
        Initialize MIG explainer.

        Args:
            model: Classifier model to explain
            vae: VAE model with encode() and decode() methods
            baseline_method: Baseline type ('zero')
            num_steps: Number of interpolation points (T)
            device: Device to run on
            exp_obj: Objective function ('prob' or 'logit')
            preprocess_fn: Function to preprocess inputs for classifier
            alpha: Learning rate for geodesic optimization
            max_iterations: Maximum iterations for geodesic path optimization
            epsilon: Convergence threshold for energy
        """
        self.model = model
        self.vae = vae
        self.baseline_method = baseline_method
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

        self.path_generator = GeodesicPathGenerator(
            vae=vae,
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            device=self.device,
            num_steps=self.num_steps,
            alpha=self.alpha,
            max_iterations=self.max_iterations,
            epsilon=self.epsilon,
        )
