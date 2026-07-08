from .interface import PolicyObjective
from .objectives import (
    Greedy, Uniform, Prior,
    VariationalKullbackLeibler,
    ExPropKullbackLeibler, SquaredHellinger,
    Jeffrey, JensenShannon,
    TotalVariationL2,
    Muesli
)

supported = {
    Greedy.__name__,
    Uniform.__name__,
    Prior.__name__,
    VariationalKullbackLeibler.__name__,
    ExPropKullbackLeibler.__name__,
    SquaredHellinger.__name__,
    Muesli.__name__,
    Jeffrey.__name__,
    JensenShannon.__name__,
    TotalVariationL2.__name__
}


class FullyAnalytical:
    """Fully analytically solvable proposal distributions"""
    greedy: PolicyObjective = Greedy
    prior: PolicyObjective = Prior
    uniform: PolicyObjective = Uniform


class AnalyticNormalizable:
    """Proposal distributions with analytic solutions and normalizers.

    Only the trust-region constraint needs to be solved for numerically.
    """
    variational_kl: PolicyObjective = VariationalKullbackLeibler


class AnalyticTrustRegion:
    """Proposal distributions with analytic solutions and normalizers.

    Only the trust-region constraint needs to be solved for numerically.
    """
    exprop_kl: PolicyObjective = ExPropKullbackLeibler
    hellinger: PolicyObjective = SquaredHellinger
    muesli: PolicyObjective = Muesli


class AnalyticExpressable:
    """Proposal distributions with analytic solutions and normalizers.

    Both the trust-region constraint and normalizer needs to be numerically
    solved.
    """
    jensen_shannon: PolicyObjective = JensenShannon
    jeffrey: PolicyObjective = Jeffrey


class FullyNumerical:
    """Proposal distributions that need to be fully numerically estimated.

    """
    total_variation: PolicyObjective = TotalVariationL2
