import numpy as np
from ntkunlimited.recursions.symmetric_tensor import SymmetricTensor, TensorSymmetry
from dataclasses import dataclass
import scipy as sp
import logging
from collections.abc import Callable
from numpy.typing import ArrayLike
from functools import partial

logger = logging.getLogger(__name__)

IntegrateableFn = Callable[[ArrayLike], ArrayLike]


@dataclass
class CubatureConf:
    rtol: float
    max_subdiv: int


def normal_expec(fn, pdf, dim, cubature_conf):
    def weighted_integrand(x):
        return fn(x) * pdf(x)

    rule = "genz-malik" if dim > 1 else "gauss-kronrod"
    expectation = sp.integrate.cubature(
        weighted_integrand,
        [-np.inf] * dim,
        [np.inf] * dim,
        rtol=cubature_conf.rtol,
        max_subdivisions=cubature_conf.max_subdiv,
        rule=rule,
    )
    if expectation.status == "not_converged":
        logger.warning(f"Integration of {dim}d integrals did not converge.")
    return expectation.estimate


def group_by_dim(inds: list[tuple[int, ...]]):
    grouped = {}
    for i, ind in enumerate(inds):
        rel_dims, num_appear = np.unique(ind, return_counts=True)
        dim = len(rel_dims)
        if dim not in grouped:
            grouped[dim] = []
        grouped[dim].append(
            {
                "inds": ind,
                "rel_dims": rel_dims,
            }
        )
    return grouped


def comp_fn_normal_expec_all(
    fn, G: np.ndarray, integrand_sym: TensorSymmetry, cubature_conf: CubatureConf
):
    # fn should take (x, i1, i2, ..., in) and return scalars at each point (n_samples)
    inds_by_dim = group_by_dim(integrand_sym.unique_inds)
    result = SymmetricTensor(integrand_sym, partial(np.zeros, dtype=float))

    for dim, ind_group in inds_by_dim.items():
        logger.debug(f"Evaluating {len(ind_group)} {dim}-dimensional integrals")

        sub_covs = [
            G[inds["rel_dims"][:, None], inds["rel_dims"][None, :]]
            for inds in ind_group
        ]
        rvs = [sp.stats.multivariate_normal(cov=sub_cov) for sub_cov in sub_covs]

        expecs = []
        for j, comb in enumerate(ind_group):
            index_remap = {orig_ind: i for i, orig_ind in enumerate(comb["rel_dims"])}
            remapped_indices = [index_remap[i] for i in comb["inds"]]

            def fn_indexed(x, indices):
                # x is of shape (n_evaluation points, dim)
                # transpose is a hacky solution because the lambdified function fn is not
                # compatible with adding batch dimensions
                y = fn(np.transpose(x), *indices)
                return y

            expec = normal_expec(
                partial(fn_indexed, indices=remapped_indices),
                rvs[j].pdf,
                dim,
                cubature_conf,
            )
            expecs.append(expec)
            result[comb["inds"]] = expec
    return result
