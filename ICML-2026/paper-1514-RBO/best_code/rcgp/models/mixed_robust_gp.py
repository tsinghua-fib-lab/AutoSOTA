"""
Mixed Robust Conjugate Gaussian Process for mixed variable types.

This extends RobustConjugateGP to handle categorical and ordinal variables
following BoTorch's MixedSingleTaskGP pattern.
"""

from typing import Optional, List
import math
from torch import Tensor
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from botorch.models.kernels import CategoricalKernel
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.types import _DefaultType, DEFAULT

from .robust_gp import RobustConjugateGP
from ..weighting import WeightingFunction


class MixedRobustConjugateGP(RobustConjugateGP):
    """
    Robust Conjugate GP that supports mixed variable types.
    
    Uses a kernel structure similar to MixedSingleTaskGP:
    K = K_cont_1 + K_cat_1 + K_cont_2 * K_cat_2
    
    where continuous and categorical kernels are combined to handle mixed spaces.
    """
    
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        weighting_function: WeightingFunction,
        cat_dims: Optional[List[int]] = None,
        likelihood: GaussianLikelihood | None = None,
        mean_module: Optional[gpytorch.means.Mean] = None,
        cont_kernel_factory: Optional[callable] = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
    ):
        """
        Initialize Mixed Robust Conjugate GP.
        
        Args:
            train_X: Training inputs [n, d]
            train_Y: Training targets [n, 1] or [n]
            weighting_function: Weighting function for robustness
            cat_dims: List of indices for categorical dimensions
            likelihood: Gaussian likelihood (created if None)
            mean_module: Mean function (defaults to ConstantMean)
            cont_kernel_factory: Factory for continuous kernel (defaults to Matern 2.5)
            outcome_transform: Transform applied to outcomes
        """
        self.cat_dims = cat_dims if cat_dims is not None else []
        self._is_mixed = len(self.cat_dims) > 0
        
        # Create mixed kernel if we have categorical dimensions
        if self._is_mixed:
            covar_module = self._create_mixed_kernel(
                train_X=train_X,
                cat_dims=self.cat_dims,
                cont_kernel_factory=cont_kernel_factory
            )
        else:
            # No categorical dims, use standard kernel
            covar_module = None  # Will use default in parent class
        
        # Initialize parent class with the mixed kernel
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            weighting_function=weighting_function,
            likelihood=likelihood,
            mean_module=mean_module,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
        )
    
    def _create_mixed_kernel(
        self,
        train_X: Tensor,
        cat_dims: List[int],
        cont_kernel_factory: Optional[callable] = None
    ) -> gpytorch.kernels.Kernel:
        """
        Create the mixed kernel: K = K_cont_1 + K_cat_1 + K_cont_2 * K_cat_2
        
        This follows BoTorch's MixedSingleTaskGP kernel structure.
        """
        # Determine continuous and categorical dimensions
        d = train_X.shape[-1]
        cont_dims = [i for i in range(d) if i not in cat_dims]
        
        # Default continuous kernel factory (RBF to match RobustConjugateGP)
        if cont_kernel_factory is None:
            def cont_kernel_factory(ard_num_dims, active_dims):
                # Use the same prior as RobustConjugateGP
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
        
        # Create kernels for sum terms
        if len(cont_dims) > 0:
            # Continuous kernel for sum
            sum_cont_kernel = ScaleKernel(
                cont_kernel_factory(
                    ard_num_dims=len(cont_dims),
                    active_dims=cont_dims
                )
            )
        else:
            sum_cont_kernel = None
        
        if len(cat_dims) > 0:
            # Categorical kernel for sum
            # Use similar constraint setup as continuous kernels
            sum_cat_kernel = ScaleKernel(
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
        else:
            sum_cat_kernel = None
        
        # Create kernels for product term
        if len(cont_dims) > 0 and len(cat_dims) > 0:
            # Product of continuous and categorical
            prod_cont_kernel = cont_kernel_factory(
                ard_num_dims=len(cont_dims),
                active_dims=cont_dims
            )
            prod_cat_kernel = CategoricalKernel(
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(
                    2.5e-2,
                    initial_value=1.0,
                    transform=None
                )
            )
            prod_kernel = ScaleKernel(prod_cont_kernel * prod_cat_kernel)
        else:
            prod_kernel = None
        
        # Combine all kernels
        kernel_terms = []
        if sum_cont_kernel is not None:
            kernel_terms.append(sum_cont_kernel)
        if sum_cat_kernel is not None:
            kernel_terms.append(sum_cat_kernel)
        if prod_kernel is not None:
            kernel_terms.append(prod_kernel)
        
        # Sum all terms
        if len(kernel_terms) > 1:
            covar_module = kernel_terms[0]
            for term in kernel_terms[1:]:
                covar_module = covar_module + term
        else:
            covar_module = kernel_terms[0]
        
        return covar_module