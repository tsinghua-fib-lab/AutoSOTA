"""The OTP-FM training curriculum."""

import math


class Curriculum:
    """
    OTP-FM training curriculum, see Sec. 4.3 of :cite:t:`kansal2026otpfm`.

    The ``alpha(i)`` parameter controls the transition from vanilla flow matching (``alpha(i)=0``)
    to full OTP-FM (``alpha(i)=1``) as a function of the training iteration ``i``.

    We observe the sigmoid schedule to be most stable. The default hyperparameters are tuned for the W2Inf potential.

    Args:
        total_iterations: Total number of training iterations
        schedule: Schedule type. Options:
            - "sigmoid": Smooth sigmoid transition (recommended)
            - "linear": Linear ramp from 0 to 1
            - "constant" or "1": Always returns 1.0 (full OTP-FM)
            - "0": Always returns 0.0 (vanilla flow matching)
        slope: Steepness of sigmoid transition (default: 6.0)
        midpoint: Normalized time when sigmoid reaches 0.5 (0.5 means halfway through the training) (default: 0.5)

    Example:
        >>> from otpfm import Curriculum
        >>> n_epochs, iterations_per_epoch = 100, 50
        >>> alpha_schedule = Curriculum(total_iterations=n_epochs * iterations_per_epoch)
        >>> for epoch in range(n_epochs):
        ...     for batch_idx, batch in enumerate(dataloader):
        ...         iteration = epoch * iterations_per_epoch + batch_idx
        ...         otp_alpha = alpha_schedule(iteration)
        ...         loss = model.forward_with_loss(batch, otp_alpha=otp_alpha)
    """

    def __init__(
        self,
        total_iterations: int,
        schedule: str = "sigmoid",
        slope: float = 6.0,
        midpoint: float = 0.5,
    ):
        self.total_iterations = total_iterations
        self.schedule = schedule
        self.slope = slope
        self.midpoint = midpoint

        if schedule not in ("sigmoid", "linear", "constant", "0", "1"):
            raise ValueError(
                f"Unknown schedule: {schedule}. Use 'sigmoid', 'linear', 'constant', '0', or '1'."
            )

    def __call__(self, iteration: int) -> float:
        """
        Get ``alpha(i)`` for the given training iteration.

        Args:
            iteration: Current training iteration

        Returns:
            ``alpha(i)`` value in [0, 1]
        """
        if self.total_iterations <= 0:
            return 1.0

        progress = iteration / self.total_iterations

        match self.schedule:
            case "sigmoid":
                return 1.0 / (1.0 + math.exp(-self.slope * (progress - self.midpoint)))
            case "linear":
                return min(1.0, max(0.0, progress))
            case "0":
                return 0.0
            case "1":
                return 1.0
            case "constant":
                return 1.0
            case _:
                raise ValueError(
                    f"Unknown schedule: {self.schedule}. Use 'sigmoid', 'linear', 'constant', '0', or '1'."
                )

    def __repr__(self) -> str:
        return (
            f"Curriculum(total_iterations={self.total_iterations}, "
            f"schedule='{self.schedule}', slope={self.slope}, midpoint={self.midpoint})"
        )
