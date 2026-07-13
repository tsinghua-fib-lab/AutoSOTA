from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils import causalchamber_offline

causalchamber_offline.enable()

from causalchamber.datasets import ImageExperiment
import pandas as pd
import math
from simulator.base import Simulator
import numpy as np
import torch
from pyro import distributions as dist
import causalchamber
import causalchamber.simulators.lt as lt
from causalchamber.simulators import Simulator as CausalSimulator

from utils.transform import CosineSquaredAngleDiff, LogitBoxTransform, RGBAlpha


class LightTunnel(Simulator):
    def __init__(
        self,
        theta_dim: int = 5,
        obs_dim: Tuple[int, int, int] = (3, 64, 64),
        data_dir: str = "/home/pruhlman/data/ropefm",  # Base data directory
        exp_name: str = "uniform_ap_1.8_iso_500.0_ss_0.005",
        model_config: Dict[str, Any] = {"": {}},
    ):
        theta_dim = int(theta_dim)
        obs_dim = (int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2]))
        model_config["params"]["image_size"] = obs_dim[1]  # Ensure image size matches model config
        super().__init__(obs_dim=obs_dim, theta_dim=theta_dim, name="light_tunnel")
        self.image_size = obs_dim[1]
        # Construct data path from data_dir and task name
        self.data_path = str(Path(data_dir) / "light_tunnel")
        self.exp_name = exp_name
        self.callable_simulator = True
        self.callable_dgp = False
        self.supported_generation = ["independent"]

        # Download data
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        self.dataset = causalchamber.datasets.Dataset(
            "lt_camera_v1", root=self.data_path, download=True
        )
        self.experiment: ImageExperiment = self.dataset.get_experiment(self.exp_name)  # type: ignore
        assert self.exp_name in self.dataset.available_experiments(), (
            f"Experiment not found in dataset.\nSupported experiments: {self.dataset.available_experiments()}"
        )
        assert str(self.image_size) in self.experiment.available_sizes(), (
            f"Image size {self.image_size} not supported for experiment {self.exp_name}.\nAvailable sizes: {self.experiment.available_sizes()}"
        )

        # Prior parameters
        if model_config["name"] == "rope_f1":
            self.prior_params = {
                "low": torch.Tensor([0.0, 0.0, 0.0, 0.0]),
                "high": torch.Tensor([255.0, 255.0, 255.0, 1.0]),
            }
            self.prior_dist = RGBAlpha()
        else:
            self.prior_params = {
                "low": torch.Tensor([0.0, 0.0, 0.0, -180.0, -180.0]),
                "high": torch.Tensor([255.0, 255.0, 255.0, 180.0, 180.0]),
            }
            # self.uniform_dist = dist.Independent(dist.Uniform(**self.prior_params), 1)
            # self.prior_dist = CosineSquaredAngleDiff(self.uniform_dist, device="cpu")
            self.prior_dist = dist.Independent(dist.Uniform(**self.prior_params), 1)
        self.prior_dist.set_default_validate_args(False)

        # Transform for theta
        self.transform = LogitBoxTransform(a=self.prior_params["low"], b=self.prior_params["high"])

        # Load model
        if model_config["name"] == "rope_f1":
            self.model = ModelRope_F1(**model_config["params"])
        else:
            model_builder = getattr(lt, "Model" + model_config["name"].upper(), None)
            if model_builder is None:
                raise ValueError(f"Model {model_config['name']} not found in lt module.")
            self.model = model_builder(**model_config["params"])
        print(f"Successfully initialized LightTunnel simulator with model {model_config['name']}.")
        print(
            f"Total number of images in experiment '{self.exp_name}': {len(self.experiment.as_pandas_dataframe())}"
        )

    def get_simulator(self, misspecified: bool):
        def simulator(theta: torch.Tensor) -> torch.Tensor:
            """
            Simulate the light tunnel data based on the provided theta parameters.
            """
            if not misspecified:
                raise ValueError("Misspecified must be True for LightTunnel simulator.")
            if isinstance(self.model, ModelRope_F1):
                R, G, B, alpha = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
                inputs = pd.DataFrame(
                    {
                        "red": R.cpu().numpy(),
                        "green": G.cpu().numpy(),
                        "blue": B.cpu().numpy(),
                        "alpha": alpha.cpu().numpy(),
                    }
                )
            else:
                R, G, B, pol_1, pol_2 = (
                    theta[:, 0],
                    theta[:, 1],
                    theta[:, 2],
                    theta[:, 3],
                    theta[:, 4],
                )
                inputs = pd.DataFrame(
                    {
                        "red": R.cpu().numpy(),
                        "green": G.cpu().numpy(),
                        "blue": B.cpu().numpy(),
                        "pol_1": pol_1.cpu().numpy(),
                        "pol_2": pol_2.cpu().numpy(),
                    }
                )
            output = self.model.simulate_from_inputs(inputs)
            output = torch.from_numpy(output).float()  # Convert to torch tensor and float32
            output = output.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            return output

        return simulator

    def get_total_observations(self) -> int:
        """Get total number of observations available in dataset."""
        return len(self.experiment.as_pandas_dataframe())

    def obs_from_files(
        self, n: int, mode: str = "training", indices: Optional[Tuple[int, ...]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load observations from files.

        Args:
            n: Number of samples to load
            mode: "training" or "testing" (deprecated, use indices instead)
            indices: Explicit indices to load. If provided, mode is ignored.
                    Should be a tuple/array of integer indices into the dataset.

        Returns:
            Tuple of (thetas, images) tensors
        """
        total_obs = len(self.experiment.as_pandas_dataframe())

        if n > total_obs:
            raise ValueError(f"Requested {n} samples but only {total_obs} available.")

        if indices is not None:
            # Use explicit indices
            if len(indices) != n:
                raise ValueError(f"indices length ({len(indices)}) must match n ({n})")
            selected_indices = np.array(indices)
        else:
            # Fallback to old behavior (not recommended - use DataSplitter instead)
            import warnings
            warnings.warn(
                "obs_from_files called without explicit indices. "
                "Use DataSplitter for reproducible train/test splits.",
                DeprecationWarning,
            )
            # Default: use sequential indices starting from 0
            selected_indices = np.arange(n)

        # Load images and thetas at selected indices
        all_images = self.experiment.as_image_array(str(self.image_size))
        images = all_images[selected_indices]

        all_thetas = self.experiment.as_pandas_dataframe()[
            ["red", "green", "blue", "pol_1", "pol_2"]
        ].values
        thetas = all_thetas[selected_indices]
        thetas = np.asarray(thetas)

        # Rescale images to [0, 1]
        images = images / 255.0
        images = images.astype("float32")
        # Convert to torch tensor (N, C, H, W)
        images = torch.from_numpy(images).permute(0, 3, 1, 2)

        # Handle ModelRope_F1 theta transformation
        if isinstance(self.model, ModelRope_F1):
            alphas = np.cos(np.deg2rad(thetas[:, 3] - thetas[:, 4])) ** 2
            alphas = np.asarray(alphas)
            thetas = np.concatenate([thetas[:, :3], alphas[:, np.newaxis]], axis=1)

        thetas = torch.from_numpy(thetas).float()
        return thetas, images


class ModelRope_F1(CausalSimulator):
    """Simulator of the images produced by the light tunnel.

    The derivation of the simulator, including inputs, outputs and
    parameters, is described under Model F1 in Appendix IV.2.2 of the
    paper "Causal chambers as a real-world physical testbed for AI
    Methodology" (2025) by Gamella et al.

    Link for direct access:
    https://arxiv.org/pdf/2404.11341#page=32&zoom=100,57,670

    """

    inputs_names = ["red", "green", "blue", "alpha"]
    outputs_names = ["image"]

    def __init__(
        self,
        center_x,
        center_y,
        radius,
        offset,
        image_size,
    ):
        """
        Initialize the simulator by storing its parameters.
        """
        super(ModelRope_F1, self).__init__()
        # Store the simulator's parameters
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.offset = offset
        self.image_size = image_size

    def parameters(self):
        """
        Return a dictionary with the simulator parameters and their values.
        """
        params = {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "offset": self.offset,
            "image_size": self.image_size,
        }
        return params

    def _simulate(
        self,
        red,
        green,
        blue,
        alpha,
        center_x,
        center_y,
        radius,
        offset,
        image_size,
    ):
        """Simulates a synthetic image using model_f1, generating a colored
        hexagon over a black background.

        Parameters
        ----------
        red : np.ndarray
            Brightness of the red LEDs of the light source.
        green : np.ndarray
            Brightness of the green LEDs of the light source.
        blue : np.ndarray
            Brightness of the blue LEDs of the light source.
        alpha:
            Dimming factor cos(pol_1 - pol_2)^2
        center_x : float
            X-coordinate of the hexagon's center in the image.
        center_y : float
            Y-coordinate of the hexagon's center in the image.
        radius : float
            Radius of the circumference encompassing the hexagon.
        offset : float
            Rotation of the hexagon in degrees.
        image_size : int
            Size of the synthetic image in pixels (i.e., image_size x image_size pixels).

        Returns
        -------
        np.ndarray
            Generated synthetic images with dimensions [n_images,
            image_size, image_size, 3], where n_images is the length of
            the inputs (i.e., red, green, blue, pol_1, pol_2).

        """
        N = len(red)
        images = np.ones((N, image_size, image_size, 3))
        # Color
        malus_factor = alpha
        red = red / 255 * malus_factor
        green = green / 255 * malus_factor
        blue = blue / 255 * malus_factor
        images[:, :, :, 0] *= red[:, np.newaxis, np.newaxis]
        images[:, :, :, 1] *= green[:, np.newaxis, np.newaxis]
        images[:, :, :, 2] *= blue[:, np.newaxis, np.newaxis]

        # Apply hexagon mask
        mask = hexagon_mask(center_x, center_y, radius, offset, image_size)
        images *= mask[np.newaxis, :, :, np.newaxis]
        return clip(images)


def clip(images):
    """Clip the pixels of the image so they are always in the range [0,1],
    e.g., 1.2 becomes 1, and -0.1 becomes 0.

    """
    images = np.maximum(images, 0)
    images = np.minimum(images, 1)
    return images


def hexagon_mask(center_x, center_y, radius, offset, image_size):
    """Produce the hexagon mask given its center, radius, (angle) offset
    and the size (in pixels) of the image.

    """
    mask = np.zeros((image_size, image_size))
    image_points = coord_grid(image_size)
    vertices = hexagon_vertices(center_x, center_y, radius, offset) * image_size
    # Compute cross products for a segment and all points
    cross_prods = []
    for i, vertex in enumerate(vertices):
        segment = vertex - vertices[(i + 1) % len(vertices)]
        vertex_to_points = image_points - vertex
        cross = np.cross(vertex_to_points, segment)
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    all_neg = (cross_prods <= 0).all(axis=0)
    all_pos = (cross_prods > 0).all(axis=0)
    mask = np.logical_or(all_neg, all_pos)
    return mask


def hexagon_vertices(center_x, center_y, radius, offset):
    """Given the center, radius and (angle) offset of the hexagon,
    compute the location of its six vertices.

    """
    vertices = []
    for angle in np.arange(0, 2 * np.pi, np.pi / 3):
        x = center_x + radius * np.cos(angle + offset)
        y = center_y + radius * np.sin(angle + offset)
        vertices.append([x, y])
    return np.array(vertices)


def coord_grid(image_size):
    """
    Make a coordinate grid (in pixels) given the image size
    """
    X = np.tile(np.arange(image_size), (image_size, 1))
    Y = X.T[::-1, :]
    grid = np.array([X, Y])
    return np.transpose(grid, (1, 2, 0))
