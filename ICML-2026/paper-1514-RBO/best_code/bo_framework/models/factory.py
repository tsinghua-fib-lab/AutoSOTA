"""Model factory functions for creating GP models."""

import math

import gpytorch
import torch
from typing import List

from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.models.kernels import CategoricalKernel
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch.optim as optim

from rcgp.models.robust_gp import RobustConjugateGP
from rcgp.models.standard_gp import StandardGP
from rcgp.models.mixed_robust_gp import MixedRobustConjugateGP
from rcgp.models.student_t_process import StudentTProcessModel, StudentTMarginalLogLikelihood
from rcgp.models.a2rcgp import A2RCGP
from rcgp.models.residual_a2rcgp import ResidualA2RCGP, ExperimentalResidualA2RCGP
from rcgp.weighting.plateau_imq import PlateauIMQ
from rcgp.weighting.plateau_cauchy import PlateauCauchy
from rcgp.weighting.plateau_matern32 import PlateauMatern32
from rcgp.weighting.plateau_rbf import PlateauRBF
from rcgp.fitting.rcgp_wloo import calculate_robust_heuristics, create_constant_center_fn

WEIGHTING_CLASSES = {
    "plateau_imq": PlateauIMQ,
    "plateau_cauchy": PlateauCauchy,
    "plateau_matern32": PlateauMatern32,
    "plateau_rbf": PlateauRBF,
}
from rcgp.fitting.wloo_mll import WeightedRobustLeaveOneOutMLL, RobustLeaveOneOutMLL
from rcgp.fitting.scipy_optimizer import optimize_with_scipy_lbfgs
from rcgp.models.student_t_gp import StudentTGP, fit_student_t_gp
from rcgp.models.diagnostic_wrapper import DiagnosticGPWrapper


def print_model_parameters(model, stage_name=""):
    """Print current model parameters for debugging.
    
    Args:
        model: The GP model
        stage_name: Description of current stage (e.g., "After initialization")
    """
    print(f"\n{'='*60}")
    print(f"Model Parameters - {stage_name}")
    
    named_params = model.named_parameters()
    for name, param in named_params:
        print(f"{name}: {param.data}")
    
    plateau_width = model.weighting_function.plateau_width
    c = model.weighting_function.c
    print(f"Plateau width: {plateau_width}")
    print(f"c: {c}")
    print(f"\n{'='*60}")


def create_gp_model(X: torch.Tensor, Y: torch.Tensor, use_botorch_model=True, **kwargs) -> SingleTaskGP:
    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Create outcome transform if requested
    standardize = kwargs.get('standardize', True)
    outcome_transform = Standardize(m=1) if standardize else None
    
    # Create model
    if use_botorch_model:
        model = SingleTaskGP(X, Y, outcome_transform=outcome_transform)
    else:
        model = StandardGP(X, Y, outcome_transform=outcome_transform)
    
    # Fit hyperparameters if requested
    if kwargs.get('fit_hyperparameters', False):
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    
    model.eval()
    return model


def _create_weighting_function(param_dict, Y_data, outcome_transform):
    """
    Helper function to create weighting function from parameter dictionary.
    
    Args:
        param_dict: Parameter handling dictionary
        Y_data: Y data to use for heuristics
        outcome_transform: Outcome transform for data preprocessing
        
    Returns:
        PlateauIMQ weighting function
    """
    try:
        from rcgp.fitting.rcgp_wloo import calculate_robust_heuristics, create_constant_center_fn
    except ImportError:
        raise ImportError("Could not import necessary RCGP utilities.")
    
    # Handle plateau_width
    plateau_width_method = param_dict['plateau_width']['method']
    quantile = param_dict['plateau_width'].get('quantile', 0.95)
    if plateau_width_method == 'manual':
        plateau_width = param_dict['plateau_width']['value']
    elif plateau_width_method == 'heuristics':
        Y_std = outcome_transform(Y_data)[0] if outcome_transform else Y_data
        heuristics = calculate_robust_heuristics(Y_std, quantile=quantile)
        plateau_width = heuristics['plateau_width']
    elif plateau_width_method == 'empirical_std':
        Y_std = outcome_transform(Y_data)[0] if outcome_transform else Y_data
        plateau_width = torch.std(Y_std.squeeze(-1)).clamp(min=0.1).item()
    else:
        raise ValueError(f"Invalid plateau_width_method: {plateau_width_method}")

    # Handle c
    c_method = param_dict['c']['method']
    if c_method == 'manual':
        c = param_dict['c']['value']
    elif c_method == 'heuristics':
        Y_std = outcome_transform(Y_data)[0] if outcome_transform else Y_data
        heuristics = calculate_robust_heuristics(Y_std, quantile=quantile)
        c = heuristics['c']
    elif c_method == 'empirical_std':
        Y_std = outcome_transform(Y_data)[0] if outcome_transform else Y_data
        c = torch.std(Y_std.squeeze(-1)).clamp(min=0.1).item()
    else:
        raise ValueError(f"Invalid c_method: {c_method}")

    # Determine center function
    center_fn = None
    if plateau_width_method == 'heuristics' or c_method in ('heuristics', 'empirical_std'):
        if 'heuristics' not in locals():
            Y_std = outcome_transform(Y_data)[0] if outcome_transform else Y_data
            heuristics = calculate_robust_heuristics(Y_std, quantile=quantile)
        center_fn = create_constant_center_fn(heuristics['center'])

    weighting_type = param_dict.get('weighting_type', 'plateau_imq')
    cls = WEIGHTING_CLASSES.get(weighting_type)
    if cls is None:
        raise ValueError(f"Unknown weighting_type: {weighting_type}. "
                         f"Valid options: {list(WEIGHTING_CLASSES.keys())}")
    return cls(plateau_width=plateau_width, c=c, center_fn=center_fn)


def _create_mixed_covar_module(train_X: torch.Tensor, cat_dims: List[int], cont_kernel_factory=None):
    """Construct a mixed kernel matching MixedRobustConjugateGP."""
    if not cat_dims:
        return None

    cat_dims = sorted(cat_dims)
    d = train_X.shape[-1]
    cont_dims = [i for i in range(d) if i not in cat_dims]

    if cont_kernel_factory is None:
        def cont_kernel_factory(ard_num_dims, active_dims):
            lengthscale_prior = gpytorch.priors.LogNormalPrior(
                loc=math.sqrt(2) + math.log(ard_num_dims) * 0.5,
                scale=math.sqrt(3)
            )
            return RBFKernel(
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(
                    2.5e-2,
                    initial_value=lengthscale_prior.mode,
                    transform=None
                )
            )

    kernel_terms = []

    if cont_dims:
        kernel_terms.append(
            ScaleKernel(
                cont_kernel_factory(
                    ard_num_dims=len(cont_dims),
                    active_dims=cont_dims
                )
            )
        )

    kernel_terms.append(
        ScaleKernel(
            CategoricalKernel(
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(
                    2.5e-2,
                    initial_value=1.0,
                    transform=None
                )
            )
        )
    )

    if cont_dims:
        prod_cont = cont_kernel_factory(
            ard_num_dims=len(cont_dims),
            active_dims=cont_dims
        )
        prod_cat = CategoricalKernel(
            ard_num_dims=len(cat_dims),
            active_dims=cat_dims,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(
                2.5e-2,
                initial_value=1.0,
                transform=None
            )
        )
        kernel_terms.append(ScaleKernel(prod_cont * prod_cat))

    covar_module = kernel_terms[0]
    for term in kernel_terms[1:]:
        covar_module = covar_module + term

    return covar_module


def create_rcgp_model(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> RobustConjugateGP:
    """
    Create a Robust Conjugate GP model using unified fit() method.

    Current recommendations for parameter handling (we usually work with standarised outcomes):
    - plateau_width: use heuristics, or a manual value around 2.0 2.5
    - c: use manual value around 1.0
    - sigma: use fit
    - mean: use fit

    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        **kwargs: Additional arguments including:
            - param_handling_dict: Dictionary of parameter handling methods and values
            - fitting_objective_type: Type of fitting objective (options 'mll', 'loo-cv' or 'wloo-cv')
            - optimizer_type: Type of optimizer (options 'adam', 'lbfgs')
            - optimizer_kwargs: Dictionary of optimizer arguments
            - fit_hyperparameters: Whether to fit hyperparameters (default: True)
            - verbose: Whether to print verbose output
    """
    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Create outcome transform if requested
    standardize = kwargs.get('standardize', True)
    outcome_transform = Standardize(m=1) if standardize else None
    
    # Get parameter handling dictionary
    param_handling_dict = kwargs.get('param_handling_dict', {})
    if not param_handling_dict:
        raise ValueError("param_handling_dict must be provided when creating an RCGP model")

    # Create weighting function from parameter dictionary
    weighting_function = _create_weighting_function(param_handling_dict, Y, outcome_transform)
    
    # Create model
    model = RobustConjugateGP(X, Y, weighting_function=weighting_function, outcome_transform=outcome_transform)
    
    # Initialize parameters using unified method
    model._initialize_parameters(param_handling_dict, verbose=kwargs.get('verbose', False))
    
    # Fit model using unified method (if requested)
    fit_hyperparameters = kwargs.get('fit_hyperparameters', True)
    if fit_hyperparameters:
        # Prepare fit kwargs (exclude param_handling_dict and fit_hyperparameters)
        fit_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['param_handling_dict', 'fit_hyperparameters']}
        if 'fitting_objective_type' in fit_kwargs:
            fit_kwargs['objective_type'] = fit_kwargs.pop('fitting_objective_type')
        model.fit(param_handling_dict, **fit_kwargs)
    else:
        # Skip fitting, just set to eval mode
        model.eval()
    
    return model


def create_mixed_gp_model(X: torch.Tensor, Y: torch.Tensor, cat_dims: List[int], **kwargs) -> MixedSingleTaskGP:
    """Create a mixed variable GP model using BoTorch's MixedSingleTaskGP.
    
    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        cat_dims: List of indices for categorical dimensions
        **kwargs: Additional arguments including:
            - standardize: Whether to standardize outcomes (default: True)
            - fit_hyperparameters: Whether to fit hyperparameters (default: False)
    """
    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Create outcome transform if requested
    standardize = kwargs.get('standardize', True)
    outcome_transform = Standardize(m=1) if standardize else None
    
    # Create model
    model = MixedSingleTaskGP(X, Y, cat_dims=cat_dims, outcome_transform=outcome_transform)
    
    # Fit hyperparameters if requested
    if kwargs.get('fit_hyperparameters', False):
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    
    model.eval()
    return model


def create_mixed_rcgp_model(X: torch.Tensor, Y: torch.Tensor, cat_dims: List[int], **kwargs) -> MixedRobustConjugateGP:
    """Create a mixed variable Robust Conjugate GP model.
    
    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        cat_dims: List of indices for categorical dimensions
        **kwargs: Additional arguments including:
            - param_handling_dict: Dictionary of parameter handling methods and values
            - fitting_objective_type: Type of fitting objective (options 'mll', 'loo-cv' or 'wloo-cv')
            - optimizer_type: Type of optimizer (options 'adam', 'lbfgs')
            - optimizer_kwargs: Dictionary of optimizer arguments
            - verbose: Whether to print verbose output
    """
    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Create outcome transform if requested
    standardize = kwargs.get('standardize', True)
    outcome_transform = Standardize(m=1) if standardize else None
    
    # handle manually set parameters and heuristics
    param_handling_dict = kwargs.get('param_handling_dict', {})
    if not param_handling_dict:
        raise ValueError("param_handling_dict must be provided when creating a mixed RCGP model")

    #################################################
    # Handle creation of the weighting function
    #################################################
    # handle plateau_width and c
    plateau_width_method = param_handling_dict['plateau_width']['method']
    quantile = param_handling_dict['plateau_width'].get('quantile', 0.95)
    if plateau_width_method == 'manual':
        plateau_width = param_handling_dict['plateau_width']['value']
    elif plateau_width_method == 'heuristics':
        Y_std = outcome_transform(Y)[0] if outcome_transform else Y
        heuristics = calculate_robust_heuristics(Y_std, quantile=quantile)
        plateau_width = heuristics['plateau_width']
    else:
        raise ValueError(f"Invalid plateau_width_method: {plateau_width_method}")

    c_method = param_handling_dict['c']['method']
    if c_method == 'manual':
        c = param_handling_dict['c']['value']
    elif c_method == 'heuristics':
        Y_std = outcome_transform(Y)[0] if outcome_transform else Y
        heuristics = calculate_robust_heuristics(Y_std, quantile=quantile)
        c = heuristics['c']
    elif c_method == 'empirical_std':
        Y_std = outcome_transform(Y)[0] if outcome_transform else Y
        c = torch.std(Y_std.squeeze(-1)).clamp(min=0.1).item()
    else:
        raise ValueError(f"Invalid c_method: {c_method}")

    # Determine center function based on whether we have heuristics
    center_fn = None
    if plateau_width_method == 'heuristics' or c_method in ('heuristics', ''):
        # If either parameter uses heuristics, we need to calculate them and use the center
        if 'heuristics' not in locals():
            Y_std = outcome_transform(Y)[0] if outcome_transform else Y
            heuristics = calculate_robust_heuristics(Y_std, quantile=quantile)
        center_fn = create_constant_center_fn(heuristics['center'])

    # create weighting function
    weighting_function = PlateauIMQ(plateau_width=plateau_width, c=c, center_fn=center_fn)

    #################################################
    # Create the model
    #################################################
    model = MixedRobustConjugateGP(
        X, Y, 
        weighting_function=weighting_function, 
        cat_dims=cat_dims,
        outcome_transform=outcome_transform
    )

    #################################################
    # Handle manual values and heuristics for sigma and mean
    #################################################
    sigma_method = param_handling_dict['sigma']['method']
    if sigma_method == 'manual':
        sigma = param_handling_dict['sigma']['value']
    elif sigma_method == 'heuristics':
        Y_std = outcome_transform(Y)[0] if outcome_transform else Y
        heuristics = calculate_robust_heuristics(Y_std)
        sigma = heuristics['noise_estimate']
    elif sigma_method == 'fit':
        sigma = None
    else:
        raise ValueError(f"Invalid sigma_method: {sigma_method}")

    mean_method = param_handling_dict['mean']['method']
    if mean_method == 'manual':
        mean = param_handling_dict['mean']['value']
    elif mean_method == 'heuristics':
        Y_std = outcome_transform(Y)[0] if outcome_transform else Y
        heuristics = calculate_robust_heuristics(Y_std)
        mean = heuristics['center']
    elif mean_method == 'fit':
        mean = None
    else:
        raise ValueError(f"Invalid mean_method: {mean_method}")
    
    if sigma is not None:
        model.likelihood.noise.data.fill_(sigma ** 2)
        model.likelihood.raw_noise.requires_grad_(False)
    if mean is not None:
        model.mean_module.constant.data.fill_(mean)
        model.mean_module.raw_constant.requires_grad_(False)

    #################################################
    # Fit the model
    #################################################
    # Set model to training mode
    model.train()

    fitting_objective_type = kwargs.get('fitting_objective_type', 'mll')
    optimizer_type = kwargs.get('optimizer_type', 'adam')

    if fitting_objective_type == 'mll':
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    elif fitting_objective_type == 'loo-cv':
        mll = RobustLeaveOneOutMLL(model.likelihood, model)
    elif fitting_objective_type == 'wloo-cv':
        mll = WeightedRobustLeaveOneOutMLL(model.likelihood, model)
    else:
        raise ValueError(f"Invalid fitting_objective_type: {fitting_objective_type}")

    # Set mll to training mode as well
    mll.train()

    if optimizer_type == 'lbfgs':
        # lbfgs_backend: "botorch" (default) or "scipy". scipy is only honored
        # for loo-cv/wloo-cv; mll always uses BoTorch (scipy was never wired
        # for the MLL objective in this dispatch).
        lbfgs_backend = kwargs.get('lbfgs_backend', 'botorch')
        if lbfgs_backend not in ('botorch', 'scipy'):
            raise ValueError(
                f"Invalid lbfgs_backend: {lbfgs_backend!r}. "
                "Must be 'botorch' or 'scipy'."
            )
        if fitting_objective_type in ['loo-cv', 'wloo-cv'] and lbfgs_backend == 'scipy':
            max_iterations = kwargs.get('max_iterations', 1000)
            verbose = kwargs.get('verbose', False)
            optimize_with_scipy_lbfgs(mll, model, max_iterations=max_iterations, verbose=verbose)
        else:
            fit_gpytorch_mll(mll)
    elif optimizer_type == 'adam':
        optimizer_kwargs = kwargs.get('optimizer_kwargs', {})
        fit_with_adam(mll, **optimizer_kwargs)
    else:
        raise ValueError(f"Invalid optimizer_type: {optimizer_type}")

    # Set both model and mll to eval mode after fitting
    model.eval()
    mll.eval()
    return model


def fit_with_adam(mll, **kwargs):
    """Fit a model by optimizing the given MLL using Adam optimizer.

    Args:
        mll: The marginal log likelihood to optimize (e.g., ExactMarginalLogLikelihood)
        **kwargs: Optimizer arguments including:
            learning_rate (float): Learning rate for Adam (default: 0.1)
            max_iter (int): Maximum iterations (default: 100)
    """
    learning_rate = kwargs.get('learning_rate', 0.1)
    max_iter = kwargs.get('max_iter', 100)

    optimizer = optim.Adam([
        {'params': mll.parameters()},
    ], lr=learning_rate)

    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Get the model from the MLL
        model = mll.model
        
        # Forward pass: get the function distribution from the model
        output = model(model.train_inputs[0])
        
        # Compute the negative MLL (to minimize for maximization)
        neg_mll = -mll(output, model.train_targets)
        
        # Backward pass
        neg_mll.backward()
        optimizer.step()


def create_student_t_model(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> StudentTProcessModel:
    """
    Create a Student-t Process model for robust GP regression.
    
    The Student-t Process provides robustness to outliers through heavy-tailed
    distributions, controlled by the degrees of freedom parameter (nu).
    
    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        **kwargs: Additional arguments including:
            - nu: Degrees of freedom (default: 3.0). Lower values = heavier tails
            - standardize: Whether to standardize outcomes (default: True)
            - fit_hyperparameters: Whether to fit hyperparameters (default: True)
            - optimizer_type: Type of optimizer ('adam' or 'lbfgs', default: 'lbfgs')
            - optimizer_kwargs: Dictionary of optimizer arguments
            
    Returns:
        StudentTProcessModel: Fitted Student-t Process model
    """
    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Get parameters from kwargs
    nu = kwargs.get('nu', 3.0)
    standardize = kwargs.get('standardize', True)
    fit_hyperparameters = kwargs.get('fit_hyperparameters', True)
    optimizer_type = kwargs.get('optimizer_type', 'lbfgs')
    optimizer_kwargs = kwargs.get('optimizer_kwargs', {})
    
    # Create outcome transform if requested
    outcome_transform = Standardize(m=1) if standardize else None
    
    # Create the Student-t Process model
    model = StudentTProcessModel(
        train_X=X,
        train_Y=Y,
        nu=nu,
        outcome_transform=outcome_transform
    )
    
    # Fit hyperparameters if requested
    if fit_hyperparameters:
        # Set model to training mode
        model.train()
        
        # Create the Student-t MLL
        mll = StudentTMarginalLogLikelihood(model.likelihood, model)
        mll.train()
        
        # Optimize hyperparameters
        if optimizer_type == 'lbfgs':
            fit_gpytorch_mll(mll)
        elif optimizer_type == 'adam':
            fit_with_adam(mll, **optimizer_kwargs)
        else:
            raise ValueError(f"Invalid optimizer_type: {optimizer_type}")
        
        # Set model to eval mode
        mll.eval()
    
    # Set model to eval mode
    model.eval()
    
    return model


def create_a2rcgp_model(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> A2RCGP:
    """
    Create an Adaptive Double Robust Conjugate GP model using unified fit() method.
    
    A2RCGP uses two levels of RCGP models:
    1. Inner RCGP: Uses constant mean, provides base robustness
    2. Outer RCGP: Uses inner RCGP posterior mean as centering function
    
    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        **kwargs: Additional arguments including:
            - inner_param_handling_dict: Parameter handling for inner RCGP
            - outer_param_handling_dict: Parameter handling for outer RCGP
            - fitting_objective_type: Type of fitting objective (options 'mll', 'loo-cv' or 'wloo-cv')
            - optimizer_type: Type of optimizer (options 'adam', 'lbfgs')
            - optimizer_kwargs: Dictionary of optimizer arguments
            - fit_hyperparameters: Whether to fit hyperparameters (default: True)
            - standardize: Whether to standardize outcomes
            - verbose: Whether to print verbose output
    """
    cat_dims = kwargs.pop('cat_dims', None)
    cat_dims = list(cat_dims) if cat_dims else []

    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Create outcome transform if requested
    standardize = kwargs.get('standardize', True)
    outcome_transform = Standardize(m=1) if standardize else None
    
    # Get parameter dictionaries
    inner_param_dict = kwargs.get('inner_param_handling_dict', {})
    outer_param_dict = kwargs.get('outer_param_handling_dict', {})
    if not inner_param_dict or not outer_param_dict:
        raise ValueError("Both inner_param_handling_dict and outer_param_handling_dict must be provided")
    
    # Create weighting functions
    inner_weighting = _create_weighting_function(inner_param_dict, Y, outcome_transform)
    outer_weighting = _create_weighting_function(outer_param_dict, Y, outcome_transform)

    # Create mixed kernels when categorical dimensions are present
    inner_covar_module = None
    outer_covar_module = None
    if cat_dims:
        outer_covar_module = _create_mixed_covar_module(X, cat_dims)
        inner_covar_module = _create_mixed_covar_module(X, cat_dims)
    
    # Create model
    model = A2RCGP(
        X,
        Y,
        inner_weighting,
        outer_weighting,
        inner_covar_module=inner_covar_module,
        outer_covar_module=outer_covar_module,
        outcome_transform=outcome_transform
    )
    
    # Initialize parameters using unified method
    model._initialize_parameters(outer_param_dict, verbose=kwargs.get('verbose', False))
    
    # Initialize inner model parameters
    if model.inner_rcgp.train_inputs[0].numel() > 0:  # Only if inner model has data
        model.inner_rcgp._initialize_parameters(inner_param_dict, verbose=kwargs.get('verbose', False))
    
    # Fit model using unified method (if requested)
    fit_hyperparameters = kwargs.get('fit_hyperparameters', True)
    if fit_hyperparameters:
        # Prepare fit kwargs (exclude param dicts and fit_hyperparameters)
        fit_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['inner_param_handling_dict', 'outer_param_handling_dict', 'fit_hyperparameters']}
        # Map fitting_objective_type to objective_type for backward compatibility
        if 'fitting_objective_type' in fit_kwargs:
            fit_kwargs['objective_type'] = fit_kwargs.pop('fitting_objective_type')
        model.fit(inner_param_dict, outer_param_dict, **fit_kwargs)
    else:
        # Skip fitting, just set to eval mode
        model.eval()
    
    return model


def create_residual_a2rcgp_model(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> ResidualA2RCGP:
    """
    Create a ResidualA2RCGP model using unified fit() method.

    ResidualA2RCGP uses residual-based decomposition:
    1. Inner RCGP: Fits on lagged data (t-1 subset)
    2. Outer RCGP: Fits on residuals (Y - inner_predictions) with zero center
    3. Posterior: Combines inner + outer predictions

    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        **kwargs: Additional arguments including:
            - inner_param_handling_dict: Parameter handling for inner RCGP
            - outer_param_handling_dict: Parameter handling for outer RCGP
            - fitting_objective_type: Type of fitting objective (options 'mll', 'loo-cv' or 'wloo-cv')
            - optimizer_type: Type of optimizer (options 'adam', 'lbfgs')
            - optimizer_kwargs: Dictionary of optimizer arguments
            - fit_hyperparameters: Whether to fit hyperparameters (default: True)
            - standardize: Whether to standardize outcomes
            - verbose: Whether to print verbose output
    """
    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y

    # Create outcome transform if requested
    standardize = kwargs.get('standardize', True)
    outcome_transform = Standardize(m=1) if standardize else None

    # Get parameter dictionaries
    inner_param_dict = kwargs.get('inner_param_handling_dict', {})
    outer_param_dict = kwargs.get('outer_param_handling_dict', {})
    if not inner_param_dict or not outer_param_dict:
        raise ValueError("Both inner_param_handling_dict and outer_param_handling_dict must be provided")

    # Create weighting functions
    inner_weighting = _create_weighting_function(inner_param_dict, Y, outcome_transform)
    outer_weighting = _create_weighting_function(outer_param_dict, Y, outcome_transform)

    # Create model
    model = ResidualA2RCGP(X, Y, inner_weighting, outer_weighting)

    # Fit model using unified method (if requested)
    fit_hyperparameters = kwargs.get('fit_hyperparameters', True)
    if fit_hyperparameters:
        # Prepare fit kwargs (exclude param dicts and fit_hyperparameters)
        fit_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ['inner_param_handling_dict', 'outer_param_handling_dict', 'fit_hyperparameters']}
        # Map fitting_objective_type to objective_type for backward compatibility
        if 'fitting_objective_type' in fit_kwargs:
            fit_kwargs['objective_type'] = fit_kwargs.pop('fitting_objective_type')
        model.fit(inner_param_dict, outer_param_dict, **fit_kwargs)

    return model


def create_experimental_residual_a2rcgp_model(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> ExperimentalResidualA2RCGP:
    """
    Create an ExperimentalResidualA2RCGP model using unified fit() method.

    ExperimentalResidualA2RCGP uses residual-based decomposition with GP MLL optimization:
    1. Inner RCGP: Standard WLOO-CV fitting on lagged data
    2. Outer RCGP: GP MLL optimization on residuals (Y - inner_predictions) with zero center
    3. Posterior: Combines inner + outer predictions

    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        **kwargs: Additional arguments including:
            - inner_param_handling_dict: Parameter handling for inner RCGP
            - outer_param_handling_dict: Parameter handling for outer RCGP
            - fitting_objective_type: Type of fitting objective for inner model (options 'mll', 'loo-cv' or 'wloo-cv')
            - optimizer_type: Type of optimizer (options 'adam', 'lbfgs')
            - optimizer_kwargs: Dictionary of optimizer arguments
            - fit_hyperparameters: Whether to fit hyperparameters (default: True)
            - standardize: Whether to standardize outcomes
            - verbose: Whether to print verbose output
    """
    # Ensure proper tensor format
    X, Y = X.double(), Y.double()
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y

    # Create outcome transform if requested
    standardize = kwargs.get('standardize', True)
    outcome_transform = Standardize(m=1) if standardize else None

    # Get parameter dictionaries
    inner_param_dict = kwargs.get('inner_param_handling_dict', {})
    outer_param_dict = kwargs.get('outer_param_handling_dict', {})
    if not inner_param_dict or not outer_param_dict:
        raise ValueError("Both inner_param_handling_dict and outer_param_handling_dict must be provided")

    # Create weighting functions
    inner_weighting = _create_weighting_function(inner_param_dict, Y, outcome_transform)
    outer_weighting = _create_weighting_function(outer_param_dict, Y, outcome_transform)

    # Create model
    model = ExperimentalResidualA2RCGP(X, Y, inner_weighting, outer_weighting)

    # Fit model using unified method (if requested)
    fit_hyperparameters = kwargs.get('fit_hyperparameters', True)
    if fit_hyperparameters:
        # Prepare fit kwargs (exclude param dicts and fit_hyperparameters)
        fit_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ['inner_param_handling_dict', 'outer_param_handling_dict', 'fit_hyperparameters']}
        # Map fitting_objective_type to objective_type for backward compatibility
        if 'fitting_objective_type' in fit_kwargs:
            fit_kwargs['objective_type'] = fit_kwargs.pop('fitting_objective_type')
        model.fit(inner_param_dict, outer_param_dict, **fit_kwargs)

    return model


def create_student_t_gp_model(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> StudentTGP:
    """
    Create a Student-t GP model with GP prior and Student-t likelihood.
    Uses Variational Inference for approximate inference.
    
    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        **kwargs: Additional arguments including:
            - degrees_of_freedom: Student-t nu parameter (default: 4.0)
            - fit_model: Whether to fit the model (default: True)
            - learning_rate: Learning rate for VI (default: 0.1)
            - num_iterations: Number of VI iterations (default: 100)
            - verbose: Whether to print fitting progress (default: False)
            
    Returns:
        StudentTGP: Fitted Student-t GP model
    """
    # Ensure proper tensor format (use float64 for consistency)
    X = X.to(dtype=torch.float64)
    Y = Y.to(dtype=torch.float64)
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Get parameters
    degrees_of_freedom = kwargs.get('degrees_of_freedom', 4.0)
    fit_model = kwargs.get('fit_model', True)
    
    # Create model
    model = StudentTGP(
        train_X=X,
        train_Y=Y,
        degrees_of_freedom=degrees_of_freedom
    )
    
    # Fit if requested
    if fit_model:
        fit_student_t_gp(
            model=model,
            verbose=kwargs.get('verbose', False)
        )
    
    return model


def create_diagnostic_gp_model(X: torch.Tensor, Y: torch.Tensor, **kwargs) -> DiagnosticGPWrapper:
    """
    Create OD-BO model (Outlier-Diagnostic Bayesian Optimization) with automatic outlier filtering.
    Based on Martinez-Cantin et al. (2018).

    The model periodically diagnoses outliers using a robust Student-t GP (with GPy Laplace
    approximation) and trains a standard GP on the filtered clean data. The filtering is
    transparent to the user - calling posterior() returns the posterior from the clean GP.
    The acquisition function is separate and calls this model's posterior() method.

    Args:
        X: Training inputs [n, d]
        Y: Training targets [n, 1]
        **kwargs: Configuration including:
            - n_init: Minimum points before diagnosis starts (default: 10)
            - n_schedule: Diagnosis frequency (default: 2)
            - nu: Student-t degrees of freedom (default: 4.0)
            - alpha: Outlier threshold (default: 0.05)
            - fitting_kwargs: kwargs for Student-t GP fitting (verbose flag)
            - model_kwargs: kwargs for the underlying GP model

    Returns:
        DiagnosticGPWrapper: Model with automatic outlier filtering
    """
    # Ensure proper tensor format
    X = X.to(dtype=torch.float64)
    Y = Y.to(dtype=torch.float64)
    Y = Y.unsqueeze(-1) if Y.dim() == 1 else Y
    
    # Build configuration
    config = {
        'n_init': kwargs.get('n_init', 10),
        'n_schedule': kwargs.get('n_schedule', 2),
        'nu': kwargs.get('nu', 4.0),
        'alpha': kwargs.get('alpha', 0.05),
        'fitting_kwargs': kwargs.get('fitting_kwargs', {
            'verbose': False
        }),
        'model_kwargs': kwargs.get('model_kwargs', {})
    }
    
    return DiagnosticGPWrapper(X, Y, config)


def main():
    rcgp_kwargs = {
        "param_handling_dict": {
            "plateau_width": {"method": "manual", "value": 0.5},
            "c": {"method": "manual", "value": 1.0},
            "sigma": {"method": "fit"},
            "mean": {"method": "fit"}
        },
        "fitting_objective_type": "wloo-cv", # options 'mll', 'loo-cv' or 'wloo-cv'
        "optimizer_type": "lbfgs",
        "optimizer_kwargs": {"learning_rate": 0.001, "max_iterations": 500},
        "verbose": False
    }
    
    torch.manual_seed(42)
    X = torch.rand(10, 1)
    Y = torch.rand(10, 1)
    rcgp_model = create_rcgp_model(X, Y, **rcgp_kwargs)
    botorch_model = create_gp_model(X, Y, fit_hyperparameters=True, use_botorch_model=True)
    custom_gp_model = create_gp_model(X, Y, fit_hyperparameters=True, use_botorch_model=False)
    
    rcgp_named_params = [(n, p) for n,p in rcgp_model.named_parameters()]
    botorch_named_params = [(n, p) for n,p in botorch_model.named_parameters()]
    custom_gp_named_params = [(n, p) for n,p in custom_gp_model.named_parameters()]

    print('rcgp_named_params')
    print(rcgp_named_params)
    print('botorch_named_params')
    print(botorch_named_params)
    print('custom_gp_named_params')
    print(custom_gp_named_params)

    print('here')

if __name__ == "__main__":
    main()
