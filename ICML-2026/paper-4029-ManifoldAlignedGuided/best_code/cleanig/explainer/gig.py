"""
Reference:
    Kapishnikov, et al., "Guided Integrated Gradients: an Adaptive Path Method for Removing Noise", arXiv 2021
"""

from cleanig.explainer.ig import IGExplainer
from cleanig.explainer.path_utils import GuidedPathGenerator


class GIGExplainer(IGExplainer):
    """
    Guided Integrated Gradients explainer.
    
    Inherits from IGExplainer but uses GuidedPathGenerator instead of LinearPathGenerator.
    The path adaptively selects features with lowest gradients at each step.
    """
    
    def __init__(
        self,
        model,
        baseline_method='zero',
        num_steps=200,
        device='cuda',
        exp_obj='prob',
        preprocess_fn=None,
        fraction=0.1,
    ):
        """
        Initialize GIG explainer.
        
        Args:
            model: Classifier model to explain
            baseline_method: Baseline type ('zero')
            num_steps: Number of Riemann sum steps for path integral approximation
            device: Device to run on
            exp_obj: Objective function ('prob' or 'logit')
            preprocess_fn: Function to preprocess inputs for classifier
            fraction: Fraction of features [0, 1] to select at each step (e.g., 0.1 = 10%)
        """
        self.model = model
        self.baseline_method = baseline_method
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj
        self.fraction = fraction
        
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x
        
        self.path_generator = GuidedPathGenerator(
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            model=self.model,
            device=self.device,
            num_steps=self.num_steps,
            fraction=self.fraction,
            exp_obj=self.exp_obj,
        )
    

