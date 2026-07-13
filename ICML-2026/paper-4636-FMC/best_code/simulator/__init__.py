from simulator.base import (
    Simulator,
    generate_simulation_dataset,
    generate_calibration_dataset,
)
from simulator.pendulum import Pendulum
from simulator.pure_gaussian import PureGaussian
from simulator.adaptive_gaussian import AdaptiveGaussian
from simulator.gaussian import Gaussian
from simulator.sir import SIR
from simulator.ou_process import OUProcess
from simulator.light_tunnel import LightTunnel
from simulator.js import JS
from simulator.wind_tunnel import WindTunnel


def get_simulator(name: str, **kwargs) -> Simulator:
    """Return the simulator object."""
    if name == "pendulum":
        return Pendulum(**kwargs)
    elif name in ["high_dim_gaussian", "pure_gaussian"]:
        return PureGaussian(**kwargs)
    elif name in ["adaptive_gaussian", "high_dim_adaptive_gaussian"]:
        return AdaptiveGaussian(**kwargs)
    elif name == "gaussian":
        return Gaussian(**kwargs)
    elif name == "sir":
        return SIR(**kwargs)
    elif name == "ou_process":
        return OUProcess(**kwargs)
    elif name == "light_tunnel":
        return LightTunnel(**kwargs)
    elif name == "js":
        return JS(**kwargs)
    elif name == "wind_tunnel":
        return WindTunnel(**kwargs)
    else:
        raise ValueError(f"Unknown simulator {name}")


__all__ = [
    "Simulator",
    "get_simulator",
    "generate_simulation_dataset",
    "generate_calibration_dataset",
]
