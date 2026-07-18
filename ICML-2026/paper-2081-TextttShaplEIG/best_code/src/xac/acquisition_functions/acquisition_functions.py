from __future__ import annotations

"""Acquisition‑function base class with EIG variants and baseline."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import scipy
import torch
from gpytorch import inv_quad
from linear_operator import to_linear_operator

from xac.applications.applications import BaseApplication
from xac.surrogates.gp_surrogate import GPSurrogate

from linear_operator.utils.cholesky import psd_safe_cholesky

log = logging.getLogger(__name__)

######################
def _stable_inv_quad_diag(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    max_tries: int = 8,
) -> torch.Tensor:
    """Compute diag(rhs^T matrix^{-1} rhs) with PSD-repair fallback."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(matrix.shape)}")
    if rhs.ndim != 2:
        raise ValueError(f"Expected 2D rhs, got shape {tuple(rhs.shape)}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got {tuple(matrix.shape)}")
    if matrix.shape[0] != rhs.shape[0]:
        raise ValueError(
            f"Incompatible shapes: matrix {tuple(matrix.shape)}, rhs {tuple(rhs.shape)}"
        )

    # Remove anti-symmetric noise from subtraction-based covariance assembly.
    matrix_sym = 0.5 * (matrix + matrix.transpose(-2, -1))
    n = matrix_sym.shape[-1]
    eye = torch.eye(n, dtype=matrix_sym.dtype, device=matrix_sym.device)

    diag = matrix_sym.diagonal()
    scale = diag.abs().max().clamp_min(torch.finfo(matrix_sym.dtype).eps)
    jitter = scale * 1e-10

    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(matrix_sym + jitter * eye)
        if int(info.max().item()) == 0:
            sol = torch.cholesky_solve(rhs, L, upper=False)
            q_diag = (rhs * sol).sum(dim=0)
            return q_diag.clamp_min(0.0)
        jitter = jitter * 10.0

    # Fallback for near-singular / indefinite cases:
    # project to PSD cone and use pseudo-inverse spectrum.
    evals, evecs = torch.linalg.eigh(matrix_sym)
    floor = scale * 1e-12
    evals_clamped = evals.clamp_min(floor)
    proj_rhs = evecs.transpose(-2, -1) @ rhs
    weighted = proj_rhs / evals_clamped.unsqueeze(-1)
    sol = evecs @ weighted
    q_diag = (rhs * sol).sum(dim=0)

    return q_diag.clamp_min(0.0)


@torch.no_grad()
def _compute_eig_function_property_naive_Z(
    surrogate: GPSurrogate,
    application: BaseApplication,
) -> torch.Tensor:
    """Naive EIG over all points in Z using the original A @ K @ A^T route."""
    A = application.A
    Z = application.Z
    lazy_covar_fz = surrogate.forward_lazy_covar(Z, observation_noise=False)
    lazy_covar_yz = surrogate.forward_lazy_covar(Z, observation_noise=True)

    covar_yz_diag = lazy_covar_yz.diagonal(dim1=-2, dim2=-1)
    transformed_covar_fz = lazy_covar_fz.matmul(A.T)
    quad_form_covar_fz = A @ transformed_covar_fz
    correction_term = inv_quad(
        input=quad_form_covar_fz,
        inv_quad_rhs=transformed_covar_fz.transpose(-2, -1),
        reduce_inv_quad=False,
    )

    EIG_Z = torch.log(covar_yz_diag) - torch.log(covar_yz_diag - correction_term)
    if EIG_Z.ndim == 2:
        EIG_Z = EIG_Z.mean(dim=0)
    return EIG_Z
###############################

# -----------------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseAcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(
        self,
        X: torch.Tensor,  # (S × D)
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
    ) -> torch.Tensor:  # (S,)
        """Return a utility value for every candidate in *X*.

        Args:
            X (torch.Tensor): A tensor of shape (s, d) representing s candidate configurations in d dimensions.
            surrogate (GPSurrogate): A GP surrogate model that has been trained on the current data archive.

        Returns:
            torch.Tensor: A tensor of shape (s,) containing the utility value for each candidate configuration.
        """
        pass

    @property
    @abstractmethod
    def plot_name(self) -> str:
        """A readable name used for plotting."""
        pass


# -----------------------------------------------------------------------------
# EIG variants
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Random(BaseAcquisitionFunction):
    """Random baseline."""

    def __call__(
        self,
        X: torch.Tensor,  # candidate set
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:
        # same for scalable mode

        utils = torch.zeros(X.shape[0], dtype=X.dtype)
        random_idx = torch.randint(0, X.shape[0], (1,))
        utils[random_idx] = 1.0

        return utils

    @property
    def plot_name(self) -> str:
        return "Random"


@dataclass(frozen=True)
class EIGFunctionProperty(BaseAcquisitionFunction):
    """Efficient implementation of the Shapley-based EIG (ShapEIG; EIG for the function property) for the Shapley values."""
     #NEW

    @torch.no_grad()
    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # In case HPs are not refitted, some qunatities can efficiently be updated
        new_train_x: torch.Tensor | None = None, # Pass selected candidate from previous iteration if HPs are not refitted; this is added to the training data and not part of the candidate set anymore
        prev_max_index: int | None = None, # Pass index of selected candidate from previous iteration if HPs are not refitted as this is no EIG candidate anymore
    ) -> torch.Tensor:
        X_binary = surrogate._model.input_transform(X)

        object.__setattr__(application, "X", X)
        object.__setattr__(application, "X_binary", X_binary)      

        if is_no_refit_step:
            #Currently only works for 1 new point

            assert new_train_x is not None
            assert prev_max_index is not None

            new_train_x_binary = surrogate._model.input_transform(new_train_x.unsqueeze(0)).squeeze(0)

            assert application.prev_X is not None
            assert application.prev_X_binary is not None

            #Ensure that candidate data is not reduced by one point
            assert X.shape[0] == (application.prev_X.shape[0] - 1)
            assert X_binary.shape[0] == (application.prev_X_binary.shape[0] - 1)

            #Ensure that new_train_x_binary is not part of current candidate set anymore
            assert not (new_train_x_binary == X_binary).all(dim=1).any() 

            #Ensure that EIG maximizer from previous iteration is now contained in training data
            assert (surrogate._model.train_inputs[0][-1,:] == new_train_x_binary.squeeze()).all() #Do we need binary of non-binary here?

        # 1: Compute marginal PPV diag(W*E*W) (for all candidates in W)
        lazy_covar_yz_diag = surrogate.forward_lazy_covar(
            X, observation_noise=True
        ).diagonal(dim1=-2, dim2=-1)
        # Requires non-binary encoding here
        #In both cases- HP refit or not - this is computed from scratch on current candidate set
        #(Backlog: One could also apply rank 1 updates here)

        # Only compute A_KZX once
        if is_no_refit_step and hasattr(application, "A_KZX"):
            A_KZX= application.update_A_KZX(surrogate, 
                                            new_train_x_binary= new_train_x_binary)

        else:
            A_KZX = application.compute_A_KZW_new(W= surrogate._model.train_inputs[0],
                                                  surrogate= surrogate)

            #assert torch.allclose(A_KZX_old, A_KZX)


        object.__setattr__(application, "A_KZX", A_KZX)

        AEA = application.compute_AEA_new(
            surrogate,
            scale_by_emp_std=False,
            precomputed_A_KZX=A_KZX, #Properly updated for both cases: HP refit or no HP refit
            is_no_refit_step= is_no_refit_step
        )
        object.__setattr__(application, "AEA_unscaled", AEA)  

        #assert torch.allclose(AEA, AEA_old)  

        # 2.2: Compute B (A*E*W) for all candidates in W)
        B = application.compute_ASigmaW_new(
            X_binary,
            surrogate,
            precomputed_A_KZX=A_KZX,
            is_no_refit_step= is_no_refit_step,
            new_train_x_binary= new_train_x_binary if is_no_refit_step else None, #Properly updated for both cases: HP refit or no HP refit 
            prev_max_index= prev_max_index if is_no_refit_step else None,
        )

        # 2.3: Combine to Q
        
        # Q_diag = inv_quad(input=AEA, inv_quad_rhs=B, reduce_inv_quad=False) #Alternative way to compute the same quantity; check if it is the same as _stable_inv_quad_diag

        # AEA = L L^T
        L_AEA = psd_safe_cholesky(AEA)
        # Solve L Y = B for Y
        Y_AEA = torch.linalg.solve_triangular(L_AEA, B, upper=False)
        # Compute diag(Y^T Y)
        Q_diag= torch.sum(torch.square(Y_AEA), dim=0)

        # Scale by emp. std (only now for numerical stability)
        emp_std = surrogate._model.outcome_transform.stdvs.squeeze()
        Q_diag_scaled = Q_diag * (emp_std**2)

        # Numerically stable reformulation (From log(ppv-q) it follows that ppv must be larger than q)
        R = (Q_diag_scaled / lazy_covar_yz_diag.clamp_min(1e-30)).clamp(
            min=0.0, max=1.0 - 1e-12
        )

        EIG_X = -torch.log1p(-R)
        # EIG_W= torch.log(lazy_covar_yz_diag) - torch.log(lazy_covar_yz_diag - Q_diag)

        object.__setattr__(application, "prev_X", X)
        object.__setattr__(application, "prev_X_binary", X_binary)


        #########
        #Compare to naive computation

        def naive_computation():
            A, Z= application.get_A_Z()
            # A = application.A
            # Z = application.Z

            covar_fz = surrogate.forward_lazy_covar(Z, observation_noise=False)

            covar_yz = surrogate.forward_lazy_covar(Z, observation_noise=True)

            covar_yz_diag = covar_yz.diagonal(dim1=-2, dim2=-1)

            transformed_covar_fz = covar_fz.matmul(A.T)
            quad_form_covar_fz = A @ transformed_covar_fz

            correction_term = inv_quad(
                input=quad_form_covar_fz,
                inv_quad_rhs=transformed_covar_fz.transpose(-2, -1),
                reduce_inv_quad=False,  # Returns diagonal
            )

            EIG_Z_naive = torch.log(covar_yz_diag) - torch.log(
                covar_yz_diag - correction_term
            )

            #For each row in X get the corresponding row in Z and thus EIG value by comparision
            X_row_in_Z= (X[:, None, :] == Z[None, ...]).all(dim=-1).nonzero(as_tuple=False)[:, 1]
            #Z[X_row_in_Z]
            #torch.all(Z[X_row_in_Z] == X)

            return EIG_Z_naive[X_row_in_Z]

        return EIG_X

    @property
    def plot_name(self) -> str:
        return "EIG-FP" #"ShaplEIG"

@dataclass(frozen=True)
class EPIG(BaseAcquisitionFunction):
    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:
        # -----------------------------------------------------------------------------
        # Implementation of the EPIG (expected predictive information gain) acquisition function
        # -----------------------------------------------------------------------------
        # Caution: Not numerically stable yet

        if scalable_mode:
            raise NotImplementedError("EPIG is not implemented for scalable mode yet.")

        Z = application.Z

        # # Assert that all candidates (X) and elements in Z are unique
        # assert X.unique(dim=0).size(0) == X.size(0)
        # assert Z.unique(dim=0).size(0) == Z.size(0)

        lazy_covar_yz = surrogate.forward_lazy_covar(Z, observation_noise=True)

        try:
            covar_yz_diag = lazy_covar_yz.diagonal(dim1=-2, dim2=-1)
            covar_yz_outer = covar_yz_diag.outer(covar_yz_diag)
            covar_yz_squared = lazy_covar_yz.mul(lazy_covar_yz)

            corr_sq = (
                covar_yz_squared.div(covar_yz_outer)
                .to_dense()
                .clamp(min=1e-12, max=1.0 - 1e-12)
            )

            EPIG_Z = -0.5 * (torch.log1p(-corr_sq).mean(dim=0))

            EPIG_X = EPIG_Z[candidate_idx_Z]

            return EPIG_X

        except Exception as e:
            log.info(
                f"CAUTION! Exception occurred in EPIG: {e}. Returning uniform vector to avoid crashes."
            )

            return torch.zeros((X.shape[0]), dtype=X.dtype)

    @property
    def plot_name(self) -> str:
        return "EPIG"


@dataclass(frozen=True)
class EIGExecutionPath(BaseAcquisitionFunction):
    """Expected Information Gain (EIG) for the execution path."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,  # unused but keeps uniform signature
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:

        if scalable_mode:
            lazy_covar_fx = surrogate.forward_lazy_covar(X, observation_noise=False)

            if lazy_covar_fx.ndim == 3:
                raise NotImplementedError()  # is correct, but should not occur
                EIG_X = lazy_covar_fx.mean(dim=0).diag()

            else:
                EIG_X = lazy_covar_fx.diagonal(
                    dim1=-2, dim2=-1
                )  # Extracts the diagonal from the last two dimensions

            if EIG_X.ndim == 2:
                raise ValueError()
                EIG_X = EIG_X.mean(dim=0)

            return EIG_X
            # Z is cont

        else:

            try:
                Z = application.Z
                lazy_covar_fz = surrogate.forward_lazy_covar(Z, observation_noise=False)

                if lazy_covar_fz.ndim == 3:
                    EIG_Z = lazy_covar_fz.mean(dim=0).diag()

                else:
                    EIG_Z = lazy_covar_fz.diagonal(
                        dim1=-2, dim2=-1
                    )  # Extracts the diagonal from the last two dimensions

                if EIG_Z.ndim == 2:
                    EIG_Z = EIG_Z.mean(dim=0)

                return EIG_Z[candidate_idx_Z]

            except Exception as e:
                log.info(
                    f"CAUTION! Exception occurred in EIGExecutionPath: {e}. Returning uniform vector to avoid crashes."
                )

                return torch.zeros((X.shape[0]), dtype=X.dtype)

    @property
    def plot_name(self) -> str:
        return "EIG-EP"


class SHAPIQAcquisitionFunction(BaseAcquisitionFunction):
    """Samples candidates according to SHAP-IQ implementations."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:
        # First entry in candidate_idx_Z is always the next to sample according to SHAP-IQ sampling
        utils_X = torch.zeros(X.shape[0], dtype=X.dtype)
        utils_X[0] = 1

        return None

    @property
    def plot_name(self) -> str:
        return "SHAP-IQ-Sampler"


class KernelSHAPSampler(SHAPIQAcquisitionFunction):
    """Samples candidates according to SHAP-IQ implementation of KernelSHAP."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:

        return None

    @property
    def plot_name(self) -> str:
        return "KernelSHAP"


class LeverageSHAPSampler(SHAPIQAcquisitionFunction):
    """Samples candidates according to SHAP-IQ implementation of LeverageSHAP."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:

        return None

    @property
    def plot_name(self) -> str:
        return "LeverageSHAP"


class SVARMSampler(SHAPIQAcquisitionFunction):
    """Samples candidates according to SHAP-IQ implementation of SVARM."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:

        return None

    @property
    def plot_name(self) -> str:
        return "SVARM"


class PermutationSampler(SHAPIQAcquisitionFunction):
    """Samples candidates according to SHAP-IQ implementation of Permutation Sampling."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:
        return None

    @property
    def plot_name(self) -> str:
        return "Permutation Sampling"


class RegressionMSRSampler(SHAPIQAcquisitionFunction):
    """Samples candidates according to SHAP-IQ implementation of RegressionMSRSampler."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:
        return None

    @property
    def plot_name(self) -> str:
        return "Regression MSR"


class LeverageGPSampler(SHAPIQAcquisitionFunction):
    """Samples candidates according to SHAP-IQ implementation of LeverageSHAP (and fits GP on this)."""

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:
        return None

    @property
    def plot_name(self) -> str:
        return "LeverageSHAP-GP"


class SHAPKernelSampler(BaseAcquisitionFunction):
    """Samples candidates according to SHAP kernel."""

    # https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

    def __call__(
        self,
        X: torch.Tensor,
        candidate_idx_Z: torch.Tensor,
        surrogate: GPSurrogate,
        application: BaseApplication,
        iteration_idx: int,
        scalable_mode: bool = False,
        is_no_refit_step: bool = False, # Ignored, only for uniform signature
        new_train_x: torch.Tensor | None = None, # Ignored, only for uniform signature
        prev_max_index: int | None = None # Ignored, only for uniform signature
    ) -> torch.Tensor:

        if scalable_mode:
            raise NotImplementedError(
                "SHAP Kernel Sampler is not implemented for scalable mode yet."
            )

        Z = application.Z
        Z_binary = surrogate._model.input_transform.transform(Z)

        m = int(math.log2(Z.shape[0]))

        shap_weights = torch.zeros(Z.shape[0], dtype=X.dtype)

        def shap_kernel_weight(m: int, z: int) -> float:
            # According to Theorem 2 in the SHAP paper
            if z == 0 or z == m:
                return 0  # Assign 0 weight to empty set and full set

            else:
                numerator = m - 1
                denominator = scipy.special.comb(m, z) * z * (m - z)

                return numerator / denominator

        # Compute SHAP weights for all rows in X
        for i in range(Z.shape[0]):
            subset_size = torch.sum(Z_binary[i, :]).item()  # Assuming binary features
            shap_weights[i] = shap_kernel_weight(m=m, z=subset_size)

        shap_weights[shap_weights == float("inf")] = 0.0
        shap_weights = shap_weights / shap_weights.sum()  # Normalize to sum to 1

        # Ensure that sampled index is within candidate_idx_Z
        while True:
            sampled_idx = torch.multinomial(shap_weights, num_samples=1)
            if sampled_idx in candidate_idx_Z:
                break

        utils_Z = torch.zeros(Z.shape[0], dtype=X.dtype)
        utils_Z[sampled_idx] = 1.0

        return utils_Z[candidate_idx_Z]  # Return utils for candidates in X only

        # Sanity check: Coalitions with few or many players have higher chance of being sampled, but there are less of them

    @property
    def plot_name(self) -> str:
        return "SHAP-Kernel-Sampler"
