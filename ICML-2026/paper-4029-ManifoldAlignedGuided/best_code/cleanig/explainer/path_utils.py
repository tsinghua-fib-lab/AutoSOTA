import math
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


def slerp(t, v0, v1, dot_threshold=0.9995):
    """
    Spherical linear interpolation between two vectors.
    Falls back to lerp when vectors are nearly parallel.
    """
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()

    norm0 = torch.norm(v0_flat)
    norm1 = torch.norm(v1_flat)

    if norm0 < 1e-9 or norm1 < 1e-9:
        return v0 * (1 - t) + v1 * t

    v0_unit = v0_flat / norm0
    v1_unit = v1_flat / norm1

    dot = torch.clamp(torch.sum(v0_unit * v1_unit), -1.0, 1.0)

    if torch.abs(dot) > dot_threshold:
        return v0 * (1 - t) + v1 * t

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t

    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0

    result = s0 * v0 + s1 * v1
    return result


class PathGenerator(ABC):
    """
    Abstract class for generating paths.
    [!] Assume that the input tensor is already preprocessed.
    """

    def __init__(self, baseline_method=None, preprocess_fn=None, device="cuda"):
        self.baseline_method = baseline_method
        self.device = device
        if preprocess_fn is not None:
            self.preprocess_fn = preprocess_fn
        else:
            self.preprocess_fn = lambda x: x

    def get_baselines(self, input_tensor):
        if self.baseline_method == "zero":
            baselines = self.preprocess_fn(
                torch.zeros_like(input_tensor).float().to(self.device)
            )
        else:
            raise ValueError(f"Invalid baseline method: {self.baseline_method}")
        return baselines

    @abstractmethod
    def get_paths(self, inputs, labels=None):
        pass


class LinearPathGenerator(PathGenerator):
    """
    Generate linear paths between the input and the baseline.
    [!] Assume that the input tensor is already preprocessed.
    """

    def __init__(self, baseline_method, preprocess_fn, device, num_steps):
        super().__init__(baseline_method, preprocess_fn, device)

        self.num_steps = num_steps

    def get_paths(self, inputs, labels=None):
        baselines = self.get_baselines(inputs)

        batch_size = inputs.shape[0]
        input_dims = list(inputs.size())[1:]
        num_input_dims = len(input_dims)

        baselines = baselines.unsqueeze(1).repeat(
            1, self.num_steps, *[1] * num_input_dims
        )
        if self.num_steps == 1:
            alpha = torch.cat([torch.Tensor([1.0]) for _ in range(batch_size)]).to(
                self.device
            )
        else:
            alpha = torch.cat(
                [torch.linspace(0, 1, self.num_steps) for _ in range(batch_size)]
            ).to(self.device)

        shape = [batch_size, self.num_steps] + [1] * num_input_dims
        interp_coef = alpha.view(*shape).to(self.device)

        end_point_baselines = (1.0 - interp_coef) * baselines
        inputs_expand_mult = inputs.unsqueeze(1)
        end_point_inputs = interp_coef * inputs_expand_mult

        paths = end_point_inputs + end_point_baselines.to(self.device)
        return paths


class GuidedPathGenerator(PathGenerator):
    """
    Generate guided paths based on Guided Integrated Gradients algorithm.
    The path adaptively selects features with lowest gradients at each step.
    [!] Assume that the input tensor is already preprocessed.
    """

    def __init__(
        self,
        baseline_method,
        preprocess_fn,
        model,
        device,
        num_steps,
        fraction=0.1,
        exp_obj="prob",
    ):
        """
        Initialize Guided Path Generator.

        Args:
            baseline_method: Method for generating baselines
            preprocess_fn: Preprocessing function
            device: Device to run on
            num_steps: Number of integration steps
            fraction: Fraction of features to select at each step
            max_dist: Maximum relative L1 distance from straight path
            model: Model for computing gradients
            exp_obj: Objective function ('prob' or 'logit')
        """
        super().__init__(baseline_method, preprocess_fn, device)
        self.num_steps = num_steps
        self.fraction = fraction
        self.model = model
        self.exp_obj = exp_obj

    def _l1_distance(self, x1, x2):
        """Returns L1 distance between two tensors."""
        return torch.abs(x1 - x2).sum()

    def _get_gradients(self, x, labels=None):
        """Compute gradients for current position."""
        x = x.clone().detach().requires_grad_(True)

        output = self.model(x)
        if labels is None:
            labels = output.max(1, keepdim=False)[1]

        if self.exp_obj == "logit":
            output = output[torch.arange(output.shape[0]), labels]
        elif self.exp_obj == "prob":
            output = torch.softmax(output, dim=-1)
            output = output[torch.arange(output.shape[0]), labels]
        else:
            raise ValueError(f"Invalid objective function: {self.exp_obj}")

        grad = torch.autograd.grad(output.sum(), x)[0].detach()
        return grad

    def _translate_alpha_to_x(self, alpha, x_input, x_baseline):
        """Translates alpha to point coordinates within interval."""
        return x_baseline + (x_input - x_baseline) * alpha

    def _translate_x_to_alpha(self, x, x_input, x_baseline):
        """Translates a point on path to its corresponding alpha value."""
        # Avoid division by zero
        diff = x_input - x_baseline
        alpha = torch.where(diff != 0, (x - x_baseline) / diff, torch.zeros_like(x))
        return alpha

    def get_paths(self, inputs, labels=None):
        # EPSILON for numerical stability
        EPSILON = 1e-9

        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b]
            x_baseline = self.get_baselines(inputs[b : b + 1]).squeeze(0)

            # Initialize
            x = x_baseline.clone()
            l1_total = self._l1_distance(x_input, x_baseline)

            paths = []

            for step in range(self.num_steps):
                # Store current position in path
                if step == self.num_steps - 1:
                    x = x_input.clone()
                    paths.append(x)
                    break
                else:
                    paths.append(x.clone())

                # Get gradients at current position
                grad_actual = self._get_gradients(x[None], labels)[0]
                grad = grad_actual.clone()

                # Unbounded GIG
                alpha_min, alpha_max = 0.0, 1.0
                x_min, x_max = x_baseline, x_input

                # Target L1 distance for this step
                l1_target = l1_total * (1 - (step + 1) / self.num_steps)

                gamma = np.inf
                while gamma > 1.0:
                    # Translate current x to alpha space
                    x_alpha = self._translate_x_to_alpha(x, x_input, x_baseline)

                    # Handle NaN values (when x_input == x_baseline for some features)
                    # These features should be set to alpha_max
                    x_alpha = torch.where(
                        torch.isnan(x_alpha),
                        torch.tensor(alpha_max).to(self.device),
                        x_alpha,
                    )

                    # Ensure x stays within bounds - features behind should catch up
                    # x = torch.where(x_alpha < alpha_min, x_min, x)
                    debug = x_alpha < alpha_min
                    assert debug.sum() < 1
                    # print(debug.sum())

                    # Calculate current L1 distance
                    l1_current = self._l1_distance(x, x_input)

                    # Check if we're close enough to target
                    close_enough = torch.isclose(
                        l1_target, l1_current, rtol=EPSILON, atol=EPSILON
                    )
                    if close_enough:
                        break

                    # Features that reached `x_max` should not be included in the selection.
                    # Assign very high gradients to them so they are excluded.
                    at_max = torch.abs(x - x_max) < EPSILON
                    grad = torch.where(
                        at_max, torch.tensor(float("inf")).to(self.device), grad
                    )

                    abs_grad = grad.abs()
                    # Cosine fraction schedule: start broad (0.10), anneal to narrow (0.02)
                    cosine_frac = 0.02 + 0.08 * 0.5 * (1 + math.cos(math.pi * step / self.num_steps))
                    threshold = torch.quantile(
                        abs_grad.reshape(-1), cosine_frac, interpolation="lower"
                    )

                    # Select features with gradients below threshold
                    s = (torch.abs(grad) <= threshold) & (grad != float("inf"))

                    # Compute how much we can move selected features
                    l1_s = (torch.abs(x - x_max) * s).sum()

                    # Calculate ratio `gamma` that show how much the selected features should
                    # be changed toward `x_max` to close the gap between current L1 and target
                    # L1.
                    if l1_s > 0:
                        gamma = (l1_current - l1_target) / l1_s
                    else:
                        gamma = np.inf

                    if gamma > 1.0:
                        # Move selected features as much as possible toward target
                        x = torch.where(s, x_max, x)
                    else:
                        # Move selected features by gamma fraction toward target
                        assert gamma > 0, f"Gamma should be positive, got {gamma}"
                        # x_new = x + gamma * (x_max - x
                        x_new = self._translate_alpha_to_x(
                            torch.tensor(gamma).to(self.device), x_max, x
                        )
                        x = torch.where(s, x_new, x)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        # Stack paths
        all_paths = torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]

        return all_paths


class LatentGuidedPathGenerator(GuidedPathGenerator):
    """
    Generate guided paths in VAE latent space.
    Similar to GuidedPathGenerator but operates in latent space and decodes to pixel space.
    """

    def __init__(
        self,
        vae,
        baseline_method,
        preprocess_fn,
        model,
        device,
        num_steps,
        fraction=0.1,
        exp_obj="prob",
        use_slerp=False,
    ):
        super().__init__(
            baseline_method, preprocess_fn, model, device, num_steps, fraction, exp_obj
        )
        self.vae = vae
        self.use_slerp = use_slerp

    def _slerp_update(self, z_current, z_target, gamma, selection_mask):
        z_new = z_current.clone()

        if selection_mask.sum() == 0:
            return z_new

        z_sel = z_current[selection_mask]
        z_tgt = z_target[selection_mask]

        z_new[selection_mask] = slerp(gamma, z_sel, z_tgt)

        return z_new

    def _get_latent_gradients(self, z, labels=None):
        """Compute gradients with respect to latent space."""
        z = z.clone().detach().requires_grad_(True)

        # Decode latent to image space
        x = self.vae.decode(z)

        # Get model output
        output = self.model(x)
        if labels is None:
            labels = output.max(1, keepdim=False)[1]

        if self.exp_obj == "logit":
            output = output[torch.arange(output.shape[0]), labels]
        elif self.exp_obj == "prob":
            output = torch.softmax(output, dim=-1)
            output = output[torch.arange(output.shape[0]), labels]
        else:
            raise ValueError(f"Invalid objective function: {self.exp_obj}")

        # Compute gradient with respect to latent z
        grad = torch.autograd.grad(output.sum(), z)[0].detach()
        return grad

    def get_paths(self, inputs, labels=None):
        # EPSILON for numerical stability
        EPSILON = 1e-9

        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b : b + 1]  # Keep batch dimension for VAE
            x_baseline = self.get_baselines(inputs[b : b + 1])

            # Encode to latent space
            z_input = self.vae.encode(x_input).squeeze(0)
            z_baseline = self.vae.encode(x_baseline).squeeze(0)

            # Raw image-space endpoints (used to anchor the path so that
            # paths[0] and paths[-1] do not depend on VAE reconstruction error).
            x_input_raw = x_input.squeeze(0)
            x_baseline_raw = x_baseline.squeeze(0)

            # Initialize in latent space
            z = z_baseline.clone()
            l1_total = self._l1_distance(z_input, z_baseline)

            paths = []

            for step in range(self.num_steps):
                # Store current position in path (decoded to image space).
                # VAE-consistent endpoints: decode latent at ALL steps including
                # endpoints to eliminate pixel/VAE-space discontinuity in deltas.
                if step == self.num_steps - 1:
                    z = z_input.clone()
                    x = self.vae.decode(z.unsqueeze(0)).squeeze(0)
                    paths.append(x.clone())
                    break
                else:
                    x = self.vae.decode(z.unsqueeze(0)).squeeze(0)
                    paths.append(x.clone())

                # Get gradients in latent space
                grad_actual = self._get_latent_gradients(z[None], labels)[0]
                grad = grad_actual.clone()

                # Unbounded GIG in latent space
                alpha_min, alpha_max = 0.0, 1.0
                z_min, z_max = z_baseline, z_input

                # Target L1 distance for this step (in latent space)
                l1_target = l1_total * (1 - (step + 1) / self.num_steps)

                gamma = np.inf
                while gamma > 1.0:
                    # Translate current z to alpha space
                    z_alpha = self._translate_x_to_alpha(z, z_input, z_baseline)

                    # Handle NaN values (when z_input == z_baseline for some features)
                    z_alpha = torch.where(
                        torch.isnan(z_alpha),
                        torch.tensor(alpha_max).to(self.device),
                        z_alpha,
                    )

                    # # Ensure z stays within bounds
                    # debug = z_alpha < alpha_min
                    # assert debug.sum() < 1

                    # Calculate current L1 distance in latent space
                    l1_current = self._l1_distance(z, z_input)

                    # Check if we're close enough to target
                    close_enough = torch.isclose(
                        l1_target, l1_current, rtol=EPSILON, atol=EPSILON
                    )
                    if close_enough:
                        break

                    # Features that reached z_max should not be included in the selection
                    at_max = torch.abs(z - z_max) < EPSILON
                    grad = torch.where(
                        at_max, torch.tensor(float("inf")).to(self.device), grad
                    )

                    abs_grad = grad.abs()
                    # Cosine fraction schedule: start broad (0.10), anneal to narrow (0.02)
                    cosine_frac = 0.02 + 0.08 * 0.5 * (1 + math.cos(math.pi * step / self.num_steps))
                    threshold = torch.quantile(
                        abs_grad.reshape(-1), cosine_frac, interpolation="lower"
                    )

                    # Select features with gradients below threshold
                    s = (torch.abs(grad) <= threshold) & (grad != float("inf"))

                    # Compute how much we can move selected features in latent space
                    l1_s = (torch.abs(z - z_max) * s).sum()

                    # Calculate ratio gamma
                    if l1_s > 0:
                        gamma = (l1_current - l1_target) / l1_s
                    else:
                        gamma = np.inf

                    if gamma > 1.0:
                        # Move selected features fully to z_max
                        z = torch.where(s, z_max, z)
                    else:
                        # Move selected features by gamma fraction toward target
                        # assert gamma > 0, f"Gamma should be positive, got {gamma}"
                        if self.use_slerp:
                            z = self._slerp_update(z, z_max, gamma, s)
                        else:
                            z_new = self._translate_alpha_to_x(
                                torch.tensor(gamma).to(self.device), z_max, z
                            )
                            z = torch.where(s, z_new, z)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        # Stack paths
        all_paths = torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]

        return all_paths


class LatentLinearPathGenerator(PathGenerator):
    """
    Generate linear paths in VAE latent space, then decode to pixel space.
    Path: baseline_latent → input_latent (linear interpolation) → decode each step

    This is used for Enhanced Integrated Gradients (EIG).
    """

    def __init__(
        self,
        vae,
        baseline_method,
        preprocess_fn,
        device,
        num_steps,
        use_slerp=True,
    ):
        """
        Initialize Latent Linear Path Generator.

        Args:
            vae: VAE model with encode() and decode() methods
            baseline_method: Method for generating baselines ('zero')
            preprocess_fn: Preprocessing function
            device: Device to run on
            num_steps: Number of interpolation steps
        """
        super().__init__(baseline_method, preprocess_fn, device)
        self.vae = vae
        self.num_steps = num_steps
        self.use_slerp = use_slerp

    def get_paths(self, inputs, labels=None):
        """
        Generate linear path in latent space and decode to pixel space.

        1. Encode inputs and baselines to latent space
        2. Linear interpolation in latent space
        3. Decode each latent point to pixel space

        Args:
            inputs: Input images [B, C, H, W]
            labels: Target labels [B] (not used, for API compatibility)

        Returns:
            paths: Tensor [B, num_steps, C, H, W]
        """
        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b : b + 1]
            x_baseline = self.get_baselines(x_input)

            # Encode to latent space
            z_input = self.vae.encode(x_input)
            z_baseline = self.vae.encode(x_baseline)

            paths = []
            for i in range(self.num_steps):
                alpha = i / (self.num_steps - 1) if self.num_steps > 1 else 1.0
                if self.use_slerp:
                    z = slerp(alpha, z_baseline, z_input)
                else:
                    z = z_baseline + alpha * (z_input - z_baseline)
                x = self.vae.decode(z).squeeze(0)
                paths.append(x)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        return torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]


class GeodesicPathGenerator(PathGenerator):
    """
    Generate geodesic paths in VAE latent space using energy minimization.
    The geodesic path minimizes the path energy on the VAE manifold.

    Reference: Jha et al., "Manifold Integrated Gradients", ICML 2024

    The algorithm finds a geodesic curve γ(t) on the data manifold by minimizing
    the path energy E[γ] = ∫||γ''(t)||² dt, where the acceleration is computed
    using the Jacobian of the decoder.
    """

    def __init__(
        self,
        vae,
        baseline_method,
        preprocess_fn,
        device,
        num_steps,
        alpha=0.01,
        max_iterations=10,
        epsilon=1e-5,
    ):
        """
        Initialize Geodesic Path Generator.

        Args:
            vae: VAE model with encode() and decode() methods
            baseline_method: Method for generating baselines ('zero')
            preprocess_fn: Preprocessing function
            device: Device to run on
            num_steps: Number of interpolation points (T)
            alpha: Learning rate for geodesic optimization
            max_iterations: Maximum iterations for geodesic path optimization
            epsilon: Convergence threshold for energy
        """
        super().__init__(baseline_method, preprocess_fn, device)
        self.vae = vae
        self.num_steps = num_steps
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def compute_etta(self, z, z_minus, z_plus, dt):
        """
        Compute acceleration (etta) in latent space using VJP of decoder.

        This computes: etta_i = -J^T @ (g(z+) - 2*g(z) + g(z-)) / dt

        where J is the Jacobian of the decoder at z, and the finite difference
        approximates the second derivative of the decoded path in image space.

        Args:
            z: Current latent point [1, D] or [1, C, H, W]
            z_minus: Previous latent point
            z_plus: Next latent point
            dt: Time step (1/(T-1) where T is num_steps)

        Returns:
            etta: Acceleration vector in latent space, same shape as z
        """
        # Compute decoded images
        g_minus = self.vae.decode(z_minus)
        g = self.vae.decode(z)
        g_plus = self.vae.decode(z_plus)

        # Finite difference approximation of second derivative in image space
        # This represents the "acceleration" of the path in image space
        finite_diff = (g_plus - 2 * g + g_minus) / dt

        # Compute VJP (Vector-Jacobian Product): J^T @ finite_diff
        # This projects the image-space acceleration back to latent space
        vjp_result = torch.autograd.functional.vjp(self.vae.decode, z, finite_diff)
        # vjp_result[0] is the forward pass output (g(z))
        # vjp_result[1] is J^T @ finite_diff
        etta = -vjp_result[1]

        # Clean up intermediate tensors
        del g_minus, g, g_plus, finite_diff, vjp_result
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return etta

    def compute_energy(self, z_collection, dt):
        """
        Compute total path energy (sum of squared etta norms).

        Energy E = Σ ||etta_i||² measures the total "curvature" of the path.
        A geodesic minimizes this energy.

        Args:
            z_collection: List of latent points [z_0, z_1, ..., z_{T-1}]
            dt: Time step

        Returns:
            Total energy (float)
        """
        energy = 0.0
        for j in range(1, len(z_collection) - 1):
            etta_j = self.compute_etta(
                z_collection[j], z_collection[j - 1], z_collection[j + 1], dt
            )
            energy += etta_j.norm().pow(2).item()
            del etta_j

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return energy

    def geodesic_path_algorithm(self, z_collection):
        """
        Optimize path to minimize energy (find geodesic).

        Uses gradient descent on the path energy. At each iteration,
        each intermediate point z_i is updated by moving in the direction
        that reduces the local curvature.

        Args:
            z_collection: List of latent points (initial linear interpolation)

        Returns:
            Optimized z_collection (geodesic path)
        """
        T = len(z_collection)
        dt = 1.0 / (T - 1) if T > 1 else 1.0

        for iteration in range(self.max_iterations):
            # Compute current energy
            energy = self.compute_energy(z_collection, dt)

            # Check convergence
            if energy < self.epsilon:
                break

            # Update each intermediate point (not endpoints)
            for i in range(1, T - 1):
                etta_i = self.compute_etta(
                    z_collection[i], z_collection[i - 1], z_collection[i + 1], dt
                )
                # Gradient descent step: move in negative gradient direction
                z_collection[i] = z_collection[i] - self.alpha * etta_i
                del etta_i

            if self.device == "cuda":
                torch.cuda.empty_cache()

        return z_collection

    def get_paths(self, inputs, labels=None):
        """
        Generate geodesic path in latent space and decode to pixel space.

        Process:
        1. Encode input and baseline to latent space
        2. Initialize path with linear interpolation in latent space
        3. Optimize path using geodesic algorithm (energy minimization)
        4. Decode optimized latent path to pixel space

        Args:
            inputs: Input images [B, C, H, W]
            labels: Target labels [B] (not used, for API compatibility)

        Returns:
            paths: Tensor [B, num_steps, C, H, W]
        """
        batch_size = inputs.shape[0]
        all_paths = []

        for b in range(batch_size):
            x_input = inputs[b : b + 1]
            x_baseline = self.get_baselines(x_input)

            # Encode to latent space
            z_input = self.vae.encode(x_input)
            z_baseline = self.vae.encode(x_baseline)

            # Initialize with linear interpolation in latent space
            z_collection = []
            for i in range(self.num_steps):
                t = i / (self.num_steps - 1) if self.num_steps > 1 else 1.0
                z = z_baseline + t * (z_input - z_baseline)
                z_collection.append(z.clone())

            # Optimize to find geodesic path
            z_collection = self.geodesic_path_algorithm(z_collection)

            # Decode to pixel space
            paths = []
            for z in z_collection:
                x = self.vae.decode(z).squeeze(0)
                paths.append(x)

            paths = torch.stack(paths, dim=0)  # [num_steps, C, H, W]
            all_paths.append(paths)

        return torch.stack(all_paths, dim=0)  # [B, num_steps, C, H, W]
