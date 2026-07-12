from cleanig.explainer.ig import IGExplainer
from cleanig.explainer.path_utils import LatentGuidedPathGenerator


class LatentGIGExplainer(IGExplainer):
    """
    Latent GIG explainer.
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
        fraction=0.1,
        use_slerp=False,
    ):
        self.model = model
        self.vae = vae
        self.baseline_method = baseline_method
        self.num_steps = num_steps
        self.device = device
        self.exp_obj = exp_obj
        self.fraction = fraction
        self.use_slerp = use_slerp
        
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x
        
        self.path_generator = LatentGuidedPathGenerator(
            vae=vae,
            baseline_method=self.baseline_method,
            preprocess_fn=self.preprocess_fn,
            model=self.model,
            device=self.device,
            num_steps=self.num_steps,
            fraction=self.fraction,
            exp_obj=self.exp_obj,
            use_slerp=self.use_slerp,
        )
    

