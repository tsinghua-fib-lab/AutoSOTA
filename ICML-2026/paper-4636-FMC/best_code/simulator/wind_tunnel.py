from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import causalchamber
import numpy as np
import pandas as pd
import torch
from causalchamber.models import simulator_a2_c3, model_a1
from causalchamber.simulators import Simulator as CausalSimulator
from pyro import distributions as dist

from simulator.base import Simulator
from utils.transform import LogitBoxTransform

# Placeholder for your actual model
# from your_model_module import WindA2C3

# For models A1 and B1
C_MIN = 0.166  # estimated above
C_MAX = 0.27
L_MIN = 0.1
OMEGA_MAX = 3000 * np.pi / 30

# For model A2
I = 0.5 * 0.059**2 * 0.02  # modelled as a solid disk: I = 1/2 * r^2 * m
T = 0.05


# Torque function \tau(L) defined in model A2, Appendix IV
def tau(L, C_min=C_MIN, C_max=C_MAX, L_min=L_MIN, T=T):
    L = np.atleast_1d(L)
    torques = T * (C_min + np.maximum(L_min, L) ** 3 * (C_max - C_min) - C_min)
    torques[L == 0] = 0
    return torques if len(L) > 1 else torques[0]


# def tau(L, C_min=C_MIN, C_max=C_MAX, L_min=L_MIN, T=T):
#    return T * (C_min + np.maximum(L_min, L) ** 3 * (C_max - C_min) - C_min)

C = tau(1) / OMEGA_MAX**2

# For models C2 and C3 (see below)
Q_MAX = 186.7 / 3600  # m3 / s
S_MAX = 74.82473949999999  # pascals
BETA = 0.15
r = 0.75
r_0 = 0.75
LOAD_IN = np.array([0.01] * 5 + [1.0] * 20 + [0.01] * 25)
LOAD_OUT = np.array([0.5] * 50)
OMEGA_IN_0 = model_a1(LOAD_IN[0], L_MIN, OMEGA_MAX)
OMEGA_OUT_0 = model_a1(LOAD_OUT[0], L_MIN, OMEGA_MAX)
TIMESTAMPS = np.linspace(0, 7.58541261, 50)
P_AMB = 95735.53125
BAROMETER_ERROR = 0
BAROMETER_PRECISION = 0.2


class WindA2C3(CausalSimulator):
    """Simulator for the pressure measurements from the downwind barometer
    in the wind tunnel.

    The derivation of the simulator, including inputs, outputs and
    parameters, is described under Models A2 and C3 in the paper
    "Causal chambers as a real-world physical testbed for AI
    Methodology" (2025) by Gamella et al.

    Link for direct access:
    https://arxiv.org/pdf/2404.11341#page=28&zoom=100,57,332

    """

    # Class variables: variable names of inputs, parameters, and outputs
    inputs_names = ["hatch"]
    outputs_names = ["pressure_downwind", "omega_in", "omega_out"]

    def __init__(
        self,
        load_in,
        load_out,
        pressure_ambient,
        # Parameters for model A2
        I,
        tau,
        K,
        timestamp,
        omega_in_0,
        omega_out_0,
        # Parameters for model C3
        S_max,
        omega_max,
        Q_max,
        r_0,
        beta,
        # Sensor noise
        barometer_error,  # The barometer offset
        barometer_precision,  # The std. of the barometer sensor noise
        random_state=42,
        # For the ODE solver of model A2
        simulation_steps=100,
    ):
        """Initializes the simulator with the given parameters. See the
        docstring for `_simulate` for a full description of the simulator
        parameters."""
        super(WindA2C3, self).__init__()

        # Parameters for models A2 and C3
        self.load_in = load_in
        self.load_out = load_out
        self.pressure_ambient = pressure_ambient
        self.I = I
        self.tau = tau
        self.K = K
        self.timestamp = timestamp
        self.omega_in_0 = omega_in_0
        self.omega_out_0 = omega_out_0
        self.S_max = S_max
        self.omega_max = omega_max
        self.Q_max = Q_max
        self.r_0 = r_0
        self.beta = beta

        # Sensor noise parameters
        self.barometer_error = barometer_error
        self.barometer_precision = barometer_precision
        self.random_state = random_state

        # ODE solver parameter
        self.simulation_steps = simulation_steps

    def parameters(self):
        """Return a dictionary with the simulator parameters and their values."""
        return {
            "load_in": self.load_in,
            "load_out": self.load_out,
            "pressure_ambient": self.pressure_ambient,
            "I": self.I,
            "tau": self.tau,
            "K": self.K,
            "timestamp": self.timestamp,
            "omega_in_0": self.omega_in_0,
            "omega_out_0": self.omega_out_0,
            "S_max": self.S_max,
            "omega_max": self.omega_max,
            "Q_max": self.Q_max,
            "r_0": self.r_0,
            "beta": self.beta,
            "barometer_error": self.barometer_error,
            "barometer_precision": self.barometer_precision,
            "random_state": self.random_state,
            "simulation_steps": self.simulation_steps,
        }

    def _simulate(
        self,
        # Inputs
        load_in,
        load_out,
        hatch,
        pressure_ambient,
        timestamp,
        # Parameters for model A2
        I,
        tau,
        K,
        omega_in_0,
        omega_out_0,
        # Parameters for model C3
        S_max,
        omega_max,
        Q_max,
        r_0,
        beta,
        # Parameters for noise simulation
        barometer_error,
        barometer_precision,
        random_state,
        simulation_steps,
    ):
        """
        Simulate dynamic wind tunnel behavior using Models A2 and C3.

        This function combines the dynamic fan speed simulation from Model A2 with the static
        pressure model that accounts for hatch position (Model C3). It computes the evolution
        of the downwind static pressure along with the intake and exhaust fan speeds over time.
        The fan speeds are internally calculated in radians per second and then converted to
        revolutions per minute (rpm) using the conversion factor (30 / π).

        The inputs (load_in, load_out, hatch, pressure_ambient and timestamp) are time-series.

        Parameters
        ----------
        load_in : float or array-like
            The load applied to the intake fan.
        load_out : float or array-like
            The load applied to the exhaust fan.
        hatch : float or array-like
            The hatch position affecting the system impedance.
        pressure_ambient : float
            The ambient static pressure (Pa) outside the wind tunnel.
        timestamp : array-like
            Time stamps (in seconds) at which the simulation is evaluated.
        I : float
            The moment of inertia of the fan (kg·m²), used in Model A2.
        tau : function
            A function that computes the motor torque as a function of load for Model A2.
        K : float
            The drag constant in the torque-balance equation of Model A2.
        omega_in_0 : float
            The initial intake fan speed (in rpm; note that internal computations use rad/s).
        omega_out_0 : float
            The initial exhaust fan speed (in rpm; note that internal computations use rad/s).
        S_max : float
            The maximum static pressure produced by the fan at full speed (Pa) in Model C3.
        omega_max : float
            The maximum fan speed (in rpm) in Model C3.
        Q_max : float
            The maximum airflow produced by the fan (m³/s) in Model C3.
        r_0 : float
            The baseline airflow ratio when the hatch is closed in Model C3.
        beta : float
            The coefficient representing the linear effect of the hatch position on the airflow ratio.
        barometer_error : float
            The offset error of the barometer sensor.
        barometer_precision : float
            The standard deviation of the barometer sensor noise.
        random_state : int or RandomState
            Seed or random state for simulating sensor noise.
        simulation_steps : int
            The number of simulation steps used by the internal ODE solver (Euler's method).

        Returns
        -------
        tuple
            A tuple containing:
                - pressure_downwind (float or array-like): The simulated downwind static pressure (Pa).
                - rpm_in (float or array-like): The simulated intake fan speed in rpm.
                - rpm_out (float or array-like): The simulated exhaust fan speed in rpm.

        Notes
        -----
        Fan speeds are converted from rad/s to rpm using the factor (30 / π).
        This method is invoked by the class method `simulate_from_inputs`.
        """
        pressure_downwind, omega_in, omega_out = simulator_a2_c3(
            load_in=load_in,
            load_out=load_out,
            hatch=hatch,
            P_amb=pressure_ambient,
            timestamps=timestamp,
            I=I,
            tau=tau,
            C=K,
            omega_in_0=omega_in_0,  # convert to rad/s
            omega_out_0=omega_out_0,
            S_max=S_max,
            omega_max=omega_max,
            Q_max=Q_max,
            r_0=r_0,
            beta=beta,
            barometer_error=barometer_error,
            barometer_precision=barometer_precision,
            random_state=random_state,
            simulation_steps=simulation_steps,
        )
        rpm_in = omega_in / np.pi * 30
        rpm_out = omega_out / np.pi * 30
        return pressure_downwind - pressure_downwind[0], rpm_in, rpm_out


class WindTunnel(Simulator):
    def __init__(
        self,
        theta_dim: int = 1,
        obs_dim: int = 50,  # just y: (time steps,)
        data_dir: str = "/home/pruhlman/data/ropefm",  # Base data directory
        exp_name: str = "load_out_0.5_osr_downwind_4",
        no_misspecification: bool = False,
    ):
        theta_dim = int(theta_dim)
        obs_dim = int(obs_dim)
        super().__init__(obs_dim=obs_dim, theta_dim=theta_dim, name="wind_tunnel")

        # Construct data path from data_dir and task name
        self.data_path = Path(data_dir) / "wind_tunnel"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.exp_name = exp_name
        self.no_misspecification = no_misspecification
        self.callable_simulator = True
        self.callable_dgp = no_misspecification  # When no misspec, DGP = simulator
        self.supported_generation = ["independent"]

        # Download data
        self.dataset = causalchamber.datasets.Dataset(
            "wt_intake_impulse_v1", download=True, root=self.data_path
        )

        # Load experiment
        self.experiment = self.dataset.get_experiment(self.exp_name)

        # Extract dataframe once
        df = self.experiment.as_pandas_dataframe()

        # Compute min/max of pressure_downwind for rescaling
        self.y_min = torch.tensor(df["pressure_downwind"].min(), dtype=torch.float32)
        self.y_max = torch.tensor(df["pressure_downwind"].max(), dtype=torch.float32)
        # self.y_max = 1.0
        # self.y_min = 0.0

        # Prior: Uniform over [0, 45]
        self.prior_params = {
            "low": torch.tensor([0.0]),
            "high": torch.tensor([45.0]),
        }
        self.prior_dist = dist.Independent(
            dist.Uniform(self.prior_params["low"], self.prior_params["high"]), 1
        )
        self.prior_dist.set_default_validate_args(False)

        # Transform for theta
        self.transform = LogitBoxTransform(a=self.prior_params["low"], b=self.prior_params["high"])

        # Initialize model
        self.model = WindA2C3(
            load_in=LOAD_IN,
            load_out=LOAD_OUT,
            pressure_ambient=P_AMB,
            I=I,
            tau=tau,
            K=C,
            timestamp=TIMESTAMPS,
            omega_in_0=OMEGA_IN_0,
            omega_out_0=OMEGA_OUT_0,
            S_max=S_MAX,
            omega_max=OMEGA_MAX,
            Q_max=Q_MAX,
            r_0=r_0,
            beta=BETA,
            barometer_error=BAROMETER_ERROR,
            barometer_precision=BAROMETER_PRECISION,
        )

    def get_simulator(self, misspecified: bool):
        def simulator(theta: torch.Tensor) -> torch.Tensor:
            """
            Simulate the wind tunnel trajectory x given theta (hatch).
            Theta is repeated along the time dimension to form a constant
            time series of shape (N, 50).

            Args:
                theta: (N, 1) tensor of hatch values

            Returns:
                x: (N, 50) tensor of simulated values
            """
            if not misspecified and not self.no_misspecification:
                raise ValueError("Misspecified must be True for WindTunnel simulator.")

            # Repeat theta along time dimension → (N, 50)
            theta_series = theta.repeat(1, 50)

            # Convert each series into a DataFrame expected by the model
            inputs = [pd.DataFrame({"hatch": ts.cpu().numpy()}) for ts in theta_series]

            # Simulate with the model
            outputs = [self.model.simulate_from_inputs(inp)[0] for inp in inputs]
            outputs = torch.stack([torch.as_tensor(out, dtype=torch.float32) for out in outputs])

            return outputs  # (N, 50)

        return simulator

    def get_total_observations(self) -> int:
        """Get total number of observation series available in dataset."""
        df = self.experiment.as_pandas_dataframe()
        return len(df) // 50  # Each observation is a 50-step time series

    def obs_from_files(
        self, n: int, mode: str = "training", indices: Optional[Tuple[int, ...]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load n observations (theta, y) from dataset.
        Each observation is a time series of length 50.

        Args:
            n: number of samples
            mode: "training" or "testing" (deprecated, use indices instead)
            indices: Explicit indices to load. These are observation indices (0 to total_obs-1),
                    not row indices. Each observation index i corresponds to rows [i*50, (i+1)*50).
                    If provided, mode is ignored.

        Returns:
            theta: (n, 1)
            y:     (n, 50)
        """
        df = self.experiment.as_pandas_dataframe()
        total_obs = len(df) // 50  # Number of 50-step observation series

        if n > total_obs:
            raise ValueError(f"Requested {n} series but only {total_obs} available.")

        if indices is not None:
            # Use explicit indices
            if len(indices) != n:
                raise ValueError(f"indices length ({len(indices)}) must match n ({n})")
            chosen_obs_indices = list(indices)
        else:
            # Fallback to old behavior (not recommended - use DataSplitter instead)
            import warnings
            warnings.warn(
                "obs_from_files called without explicit indices. "
                "Use DataSplitter for reproducible train/test splits.",
                DeprecationWarning,
            )
            chosen_obs_indices = list(range(n))

        # Convert observation indices to row indices and load data
        y_series, theta_vals = [], []
        for obs_idx in chosen_obs_indices:
            row_start = obs_idx * 50
            block = df.iloc[row_start : row_start + 50]
            y_block = torch.tensor(block["pressure_downwind"].values, dtype=torch.float32)
            theta_val = torch.tensor(block["hatch"].iloc[0], dtype=torch.float32)
            # Normalize: subtract first value
            y_block = y_block - y_block[0]

            y_series.append(y_block)
            theta_vals.append(theta_val)

        # Stack results
        y = torch.stack(y_series, dim=0)  # (n, 50)
        theta = torch.stack(theta_vals).unsqueeze(1)  # (n, 1)

        return theta, y
