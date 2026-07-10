import torch


class IndependenceCopula:
    """
    Independence copula implemented with pytorch.
    """

    def __init__(self):
        pass

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        u : torch.Tensor (float)
            tensor of shape [batch_size, 2]. Each element should be in [0, 1].

        Returns
        -------
        probability : torch.Tensor (float)
            tensor of shape [batch_size].
        """

        return torch.prod(u, dim=1)


class ClaytonCopula:
    """
    Clayton copula implemented with pytorch.
    """

    def __init__(self, theta: torch.Tensor):
        """
        Parameters
        ----------
        theta: parameter of Clayton copula.
        """

        self.theta = theta

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        u : torch.Tensor (float)
            Each element should be in [0, 1].

        Returns
        -------
        probability : torch.Tensor (float)
        """

        # if u.dim() == 3:
        #    temp_0 = u[:, 0, :] ** (-self.theta)
        #    temp_1 = u[:, 1, :] ** (-self.theta)
        # else:
        temp_0 = u[:, 0] ** (-self.theta)
        temp_1 = u[:, 1] ** (-self.theta)
        return torch.clamp(temp_0 + temp_1 - 1.0, max=0.0) ** (-1.0 / self.theta)


class FrankCopula:
    """
    Frank copula implemented with pytorch.
    """

    def __init__(self, theta: torch.Tensor):
        """
        Parameters
        ----------
        theta: parameter of Frank copula.
        """

        self.theta = theta

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        # if u.dim() == 3:
        #    temp_0 = torch.exp(-self.theta * u[:, 0, :]) - 1
        #    temp_1 = torch.exp(-self.theta * u[:, 1, :]) - 1
        # else:
        temp_0 = torch.exp(-self.theta * u[:, 0]) - 1
        temp_1 = torch.exp(-self.theta * u[:, 1]) - 1
        denominator = torch.exp(-self.theta) - 1.0
        return -torch.log(1 + temp_0 * temp_1 / denominator) / self.theta


class GumbelCopula:
    """
    Gumbel copula implemented with pytorch.
    """

    def __init__(self, theta: torch.Tensor):
        """
        Parameters
        ----------
        theta: parameter of Gumbel copula.
        """

        self.theta = theta

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        temp_0 = (-torch.log(u[:, 0])) ** self.theta
        temp_1 = (-torch.log(u[:, 1])) ** self.theta
        return torch.exp(-((temp_0 + temp_1) ** (1.0 / self.theta)))


class SurvivalCopula:
    """
    Survival copula implemented with pytorch.
    """

    def __init__(self, copula):
        self.copula = copula

    def cdf(self, u: torch.Tensor) -> torch.Tensor:
        if u.dim() != 2:
            raise ValueError("u must be 2-dimensional tensor.")
        return u[:, 0] + u[:, 1] - 1 + self.copula.cdf(1.0 - u)


def create(name: str, theta: float = 0.0):
    """
    Create a copula object based on the name and theta parameter.
    Parameters
    ----------
    name : str
        Name of the copula. Options are "independence" and "frank".
    theta : float
        Parameter for the copula. Default is 0.0.
    Returns
    -------
    copula
        An instance of the Copula class.
    """

    if name == "independence":
        return IndependenceCopula()
    elif name == "frank":
        return FrankCopula(torch.tensor(theta))
    else:
        raise ValueError(f"Invalid copula name: {name}")
