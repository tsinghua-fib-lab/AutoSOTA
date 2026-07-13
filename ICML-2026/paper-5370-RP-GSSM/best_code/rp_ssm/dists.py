from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Sequence, Union

import flax.linen as nn

import jax
import jax.numpy as np
import jax.random as jr
from jax import Array, vmap
from jax.scipy.linalg import solve_triangular

from typing_extensions import Self

from rp_ssm import dists_utils, utils
from rp_ssm.actions import ActionMapper
from rp_ssm.types import *

if TYPE_CHECKING:
    from distmaps import DistMap


class NatParam:
    """
    Class for storing and manipulating natural
    parameters of an exponential family distribution.

    Initialize the class by passing in parameters
    with names, e.g. for a Gaussian distribution,
    dist = NatParam(p=..., pwm=...).

    The class then saves the parameters as a dict, e.g.
    in the Gaussian case above,
    dist.params = {'p': ..., 'pwm': ...}.

    When defining a custom NatParam class, the functions that need to
    be specified are lognormalizer, expsuffstat_dot, and flatten
    (and optionally latent_dim and dist_param).

    Every subclass of NatParam is registered as a PyTree node,
    which allows the use of vmap and jit on functions with
    signature Array -> (subclass of NatParam).
    """

    def __init__(self, dist_map: Optional["DistMap"] = None, **kwargs):
        self.dist_map = dist_map
        self.params = kwargs

    def __add__(self, other: Self) -> Self:
        return type(self)(
            self.dist_map, **{k: v + other.params[k] for k, v in self.params.items()}
        )

    def __sub__(self, other: Self) -> Self:
        return type(self)(
            self.dist_map, **{k: v - other.params[k] for k, v in self.params.items()}
        )

    def sum(self, axis: Optional[Union[int, tuple[int]]]) -> Self:
        return type(self)(
            self.dist_map, **{k: np.sum(v, axis=axis) for k, v in self.params.items()}
        )

    @property
    def lognormalizer(self: Self) -> Self:
        ...

    def expsuffstat_dot(self, v: Self) -> Self:
        """Compute the dot product <t(x)>*v, where v is a NatParam of the same type."""
        ...

    def kl(self, other: Self) -> Self:
        return (
            self.expsuffstat_dot(self - other)
            - self.lognormalizer
            + other.lognormalizer
        )
    
    def flatten(self, params: dict[str, Array]) -> Array:
        """Flatten parameters into a single array."""
        ...

    def update(self, params: dict[str, Array]) -> Self:
        assert self.dist_map is not None
        flattened_params = self.flatten(params)
        return self.dist_map(flattened_params)

    @property
    def latent_dim(self) -> int:
        raise NotImplementedError

    @property
    def dist_param(self) -> "DistParam":
        """Turn natural parameters into corresponding distribution parameters"""
        raise NotImplementedError

    ##### register subclasses of NatParam as PyTree nodes for JAX operations
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.dist_map)
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, dist_map = aux_data
        return cls(dist_map, **dict(zip(param_keys, leaves)))
    #####


class DistParam:
    """
    Generic class for standard distribution parameters
    (e.g. mean and covariance for a Gaussian). Contains
    a method self.nat_param that returns the corresponding
    NatParam object.
    """

    def __init__(self, dist_map: Optional["DistMap"] = None, **kwargs):
        self.dist_map = dist_map
        self.params = kwargs

    @property
    def nat_param(self) -> "NatParam":
        ...

    ##### register subclasses of NatParam as PyTree nodes for JAX operations
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node_class(cls)

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.dist_map)
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, dist_map = aux_data
        return cls(dist_map, **dict(zip(param_keys, leaves)))
    #####


class AllParam:
    """
    Class storing both natural and distribution parameters
    of a distribution. In the RP-GSSM code, the prior, factors,
    and posterior all require both natural and distribution
    parameters to compute the loss. This class is initialized
    with either the natural or distribution parameters, and
    automatically computes the other on initialization.
    """

    nat_param: NatParam
    dist_param: DistParam

    def __init__(self, param):
        if isinstance(param, NatParam):
            self.nat_param = param
            self.dist_param = param.dist_param
        elif isinstance(param, DistParam):
            self.nat_param = param.nat_param
            self.dist_param = param
        elif isinstance(param, tuple):
            self.nat_param = param[0]
            self.dist_param = param[1]

    def kl(self, other):
        return self.nat_param.kl(other.nat_param)

    @property
    def lognormalizer(self):
        return self.nat_param.lognormalizer

    def __add__(self, other):
        return AllParam(
            (self.nat_param + other.nat_param, self.dist_param + other.dist_param)
        )


class GaussianNatParam(NatParam):
    """
    Gaussian natural parameters, given by ["p", "pwm"] = [A, A*m],
    where A is the precision matrix and m is the mean. The sufficient
    statistics are [-0.5*xx.T, x].
    """

    @property
    def latent_dim(self):
        return self.params["pwm"].shape[-1]

    @property
    @utils.auto_vmap("pwm", 1)
    def dist_param(self):
        cov = utils.spd_inverse(self.params["p"])
        mean = cov @ self.params["pwm"]
        return GaussianDistParam(
            dist_map=None, mean=mean, cov=cov
        )  # set dist_map=None because it won't be necessary anymore

    @property
    def lognormalizer(self):
        quad, det = utils.inv_quad_form_symmetric(self.params["p"], self.params["pwm"])
        return 0.5 * (quad - det)

    def expsuffstat_dot(self, v):
        term1 = utils.inv_quad_form(
            self.params["p"], self.params["pwm"], v.params["pwm"]
        )
        L = np.linalg.cholesky(self.params["p"])
        first = solve_triangular(
            L.T, solve_triangular(L, v.params["p"], lower=True), lower=False
        )
        second = solve_triangular(
            L.T,
            solve_triangular(
                L, np.outer(self.params["pwm"], self.params["pwm"]), lower=True
            ),
            lower=False,
        )
        term2 = -0.5 * np.trace(first + second @ first)
        return term1 + term2


class GaussianDistParam(DistParam):
    """Contains parameters ["mean", "cov"]"""

    @property
    @utils.auto_vmap("mean", 1)
    def nat_param(self) -> GaussianNatParam:
        p = utils.spd_inverse(self.params["cov"])
        pwm = p @ self.params["mean"]
        return GaussianNatParam(dist_map=None, p=p, pwm=pwm)

    def sample(self, key: Array, shape: Sequence[int]):
        return jr.multivariate_normal(
            key, self.params["mean"], self.params["cov"], shape
        )

    def log_prob(self, x):
        """
        Compute the log-probability of x under the
        Gaussian distribution `self`.
        """
        assert x.shape == self.params["mean"].shape
        if self.params["cov"].ndim == 2:
            inv_quad_form, logdet = utils.inv_quad_form_symmetric(
                self.params["cov"], x - self.params["mean"]
            )
        elif self.params["cov"].ndim == 1:
            logdet = np.sum(np.log(self.params["cov"]))
            inv_quad_form = np.sum((x - self.params["mean"]) ** 2 / self.params["cov"])
        return -0.5 * (
            self.params["mean"].shape[-1] * np.log(2.0 * np.pi) + logdet + inv_quad_form
        )


@jax.tree_util.register_pytree_node_class
class LGStationaryParam:
    """
    Class containing parameters defining a stationary
    (i.e. parameters constant in time)
    linear-Gaussian chain. Parameters are m1,Q1,A,b,Q, where
    p(z_1) = N(m1,Q1) and p(z_t+1|z_t) = N(Az_t+b,Q).
    """

    def __init__(
        self,
        start_from_invariant: bool,
        stay_at_invariant: bool,
        opt_params: list[str],
        Q_dist_map: Optional["DistMap"] = None,
        **kwargs,
    ):
        """
        Set up LGStationaryParam object.

        `opt_params` is a list of parameters to optimize. Must
        contain at least "A".

        If `start_from_invariant`, set p(z_1)=N(0,I) and do not
        learn m1,Q1.

        If Q and/or b are not provided as learnable parameters,
        set them to enforce convergence to the invariant distribution.
        """
        self.params = kwargs
        self.opt_params = opt_params
        self.start_from_invariant = start_from_invariant
        self.stay_at_invariant = stay_at_invariant
        self.Q_dist_map = Q_dist_map
        dim = kwargs["A"].shape[0]

        # save `params` before manipulating them further
        # this is used in `init` to initialize learnable parameters
        self.init_params = deepcopy(self.params)

        # if A is given as a vector, apply sigmoid and reshape to a diagonal matrix
        if kwargs["A"].ndim == 1:
            self.params.update({"A": np.diag(nn.sigmoid(kwargs["A"]))})

        if self.start_from_invariant:
            self.params.update({"m1": np.zeros(dim), "Q1": np.eye(dim)})

        if "Q" in self.opt_params:
            # concatenate a dummy mean to Q and run it through a distmap,
            # then extract the "precision" of the distmap and use that for Q
            assert self.Q_dist_map is not None
            self.params.update(
                {
                    "Q": self.Q_dist_map(
                        np.concatenate([np.zeros(dim), self.params["Q"]])
                    ).params["p"]
                }
            )

        if "b" in self.opt_params:
            assert "b" in self.params

        if "Q" not in kwargs:
            assert self.stay_at_invariant
            self.params.update(
                {"Q": np.eye(dim) - self.params["A"] @ self.params["A"].T}
            )
        if "b" not in kwargs:
            if self.stay_at_invariant:
                self.params.update(
                    {"b": (np.eye(dim) - self.params["A"]) @ self.params["m1"]}
                )
            else:
                self.params.update({"b": np.zeros(dim)})

    def init(self, key, actions):
        """Initialize optimizable parameters."""
        return {k: self.init_params[k] for k in self.opt_params}

    @property
    def latent_dim(self):
        return self.params["A"].shape[0]

    def update(self, params):
        """Update an arbitrary number of parameters."""
        return LGStationaryParam(
            start_from_invariant=self.start_from_invariant,
            stay_at_invariant=self.stay_at_invariant,
            opt_params=self.opt_params,
            Q_dist_map=self.Q_dist_map,
            **params,
        )

    def to_chain(self, num_timesteps, actions, num_samples=0, key=None):
        return dists_utils.transitions_to_marginals(
            self.params,
            num_timesteps,
            invariant=self.stay_at_invariant,
            num_samples=num_samples,
            key=key,
        ), self.params

    def tree_flatten(self):
        leaves = list(self.params.values())
        aux_data = (list(self.params.keys()), self.start_from_invariant)
        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        param_keys, start_from_invariant = aux_data
        return cls(
            start_from_invariant=start_from_invariant, **dict(zip(param_keys, leaves))
        )


class ACLGParam(LGStationaryParam):
    """
    Class containing parameters defining an action-conditioned
    linear-Gaussian chain:
    p(z_t+1|z_t,a_t) = N(A(a_t)z_t + b(a_t), Q).

    WLOG Assumes p(z_1) = N(0,I).

    For now assumes Q is not action-dependent.

    `transition_matrix`:
        'constant': z_t+1 = Az_t + ...
        'additive': z_t+1 = (A + C(a_t))z_t + ...
        'multiplicative': z_t+1 = (A @ C(a_t))z_t + ...

    `constant_transition_bias`:
        constant: b is constant (and learned as in LGStationaryParam)
        action_dependent: b is action-dependent, i.e. b(a_t)
        none: b = 0
    """

    def __init__(
        self,
        action_mapper: ActionMapper,
        transition_matrix: TransitionMatrixType,
        transition_bias: TransitionBiasType,
        Q_dist_map: Optional["DistMap"],
        opt_params=["A", "Q"],
        start_from_invariant=True,
        **kwargs,
    ):

        if transition_bias == "constant":
            opt_params.append("b")

        super().__init__(
            start_from_invariant=start_from_invariant,
            stay_at_invariant=False,
            opt_params=opt_params,
            Q_dist_map=Q_dist_map,
            **kwargs,
        )

        self.action_mapper = action_mapper
        self.transition_matrix = transition_matrix
        self.transition_bias = transition_bias

    def init(self, key: Array, actions: Array) -> dict[str, Array]:
        key1, key2 = jr.split(key)
        non_action_params = super().init(key1, actions)

        action_params = self.action_mapper.init(
            key2, actions
        )  # init C, c differently based on transition_matrix and transition_bias

        self.params = {**self.params, **action_params}

        return {**non_action_params, **action_params}

    def update(self, params: dict[str, Array]) -> Self:
        """Update an arbitrary number of parameters."""
        new_params = {**self.params, **params} # jax-compatible update
        return type(self)(
            action_mapper=self.action_mapper,
            transition_matrix=self.transition_matrix,
            transition_bias=self.transition_bias,
            Q_dist_map=self.Q_dist_map,
            **new_params,
        )

    def to_chain(self, num_timesteps: int, actions: Array) -> tuple["LGChainDistParam", dict[str, Array]]:
        """
        Return entire prior chain as well as params
        necessary for inference, i.e., As and bs, if
        applicable.
        """

        def _to_chain_single(act):

            Cs, cs = self.action_mapper.apply(act, self.params)

            if self.transition_matrix == "multiplicative":
                As = self.params["A"] @ Cs
            else:
                As = self.params["A"] + Cs

            bs = self.params["b"] + cs

            params = {**self.params, "As": As, "bs": bs}

            return dists_utils.transitions_to_marginals(
                params, num_timesteps=num_timesteps, invariant=False
            ), params

        out, params = vmap(_to_chain_single)(actions)
        return out, params


class LGChainDistParam(DistParam):
    """
    Class containing the full distribution of a linear-Gaussian chain.

    Contains parameters ['means', 'covs', 'cross_covs'].

    Is generated from LGStationaryParam.to_chain.

    Contains methods to compute KL divergences between chains and
    compute a corresponding AllParam object.
    """

    def kl(self, other):
        """Compute KL(self||other)"""
        return dists_utils.chain_kl(self.params, other.params)

    @property
    def all_param(self):
        dist_param = GaussianDistParam(
            dist_map=None, mean=self.params["means"], cov=self.params["covs"]
        )
        return AllParam(dist_param)

    @property
    def mean_field(self):
        """Strip of all cross-covariances"""
        return GaussianDistParam(mean=self.params["means"], cov=self.params["covs"])

    @property
    def moment_match(self):
        """
        Given a batched LGChainDistParam, i.e. with means
        of shape BxTxK, moment-match the mixture of Gaussians
        1/B * self.sum(axis=0) to a single unbatched Gaussian,
        i.e. LGChainDistParam with means of shape TxK.
        """
        B = self.params["means"].shape[0]
        mean = 1 / B * np.sum(self.params["means"], axis=0)
        mean_diffs = self.params["means"] - mean
        outers = vmap(vmap(lambda x: np.outer(x, x)))(mean_diffs)
        cov = 1 / B * np.sum(self.params["covs"] + outers, axis=0)
        return LGChainDistParam(means=mean, covs=cov, cross_covs=np.zeros_like(cov)[1:])
