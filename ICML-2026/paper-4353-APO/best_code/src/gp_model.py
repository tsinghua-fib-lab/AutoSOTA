import copy
from typing import Iterable, Tuple

import gpytorch
import numpy as np
import torch
from linear_operator.utils.errors import NotPSDError

class ExactGPModel(gpytorch.models.ExactGP):
    """
    GP Model Definition
    """

    def __init__(self, train_x, train_y, likelihood, dim_x):
        super().__init__(train_x, train_y, likelihood)
        self.dim_x = dim_x

        self.mean_module = gpytorch.means.ConstantMean()

        self.k_x = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=dim_x,
            )
        )
        self.k_t = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=1,
            )
        )
        self.covar_module = self.k_x * self.k_t

        self.reset_hyperparameters()

    def forward(self, inputs):
        mean_x = self.mean_module(inputs)
        x_spatial = inputs[..., : self.dim_x]
        x_time = inputs[..., self.dim_x :]

        covar_spatial = self.k_x(x_spatial)
        covar_time = self.k_t(x_time)
        covar = covar_spatial * covar_time

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

    def reset_hyperparameters(self):
        device = self.mean_module.constant.device
        dtype = self.mean_module.constant.dtype

        with torch.no_grad():
            self.mean_module.constant.fill_(0.0)
            self.k_x.base_kernel.lengthscale = torch.full(
                (1, self.dim_x), 0.6, device=device, dtype=dtype
            )
            self.k_t.base_kernel.lengthscale = torch.full(
                (1, 1), 0.3, device=device, dtype=dtype
            )
            self.k_x.outputscale = torch.tensor(1.2, device=device, dtype=dtype)
            self.k_t.outputscale = torch.tensor(1.0, device=device, dtype=dtype)


class GPModel:
    """GP Model Wrapper (Optimized data flow based on ExactGP, supports normalization)"""

    def __init__(self, dim_x: int, device: torch.device, lock=None, normalize_x: bool = False, normalize_y: bool = False):
        self.dim_x = dim_x
        self.device = device
        
        self.dtype = torch.float64
        
        # --- Normalization Config ---
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        
        # Store normalization statistics (calculated in update)
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        # ----------------------

        noise_constraint = gpytorch.constraints.GreaterThan(5e-5)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint
        ).to(self.device, dtype=self.dtype)
        self.likelihood.noise = torch.tensor(5e-3, device=self.device, dtype=self.dtype)

        self.model = None
        
        # Data is always stored as GPU tensors; stores raw data.
        self.train_x = torch.empty(0, self.dim_x + 1, device=self.device, dtype=self.dtype)
        self.train_y = torch.empty(0, device=self.device, dtype=self.dtype)

        # Training Config
        self.training_steps = 90
        self.training_lr = 0.06
        self.training_jitter_values: Tuple[float, ...] = (
            1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,
        )
        self.rescue_jitter = 5e-2
        self.grad_clip_norm = 10.0

    # ------------------------------------------------------------------ #
    # Public Interface
    # ------------------------------------------------------------------ #
    
    def _fit_model(self):
        """
        (Internal) Fits the model using self.train_x and self.train_y.
        Handles normalization logic by feeding transformed data into the model.
        """
        if self.train_x.shape[0] == 0:
            print("Warning: _fit_model() called, but no training data is available.")
            return

        # 1. Prepare training data (Apply transformation if normalization is enabled)
        train_x_use = self.train_x
        train_y_use = self.train_y

        if self.normalize_x and self.x_mean is not None and self.x_std is not None:
            train_x_use = (self.train_x - self.x_mean) / self.x_std
        
        if self.normalize_y and self.y_mean is not None and self.y_std is not None:
            train_y_use = (self.train_y - self.y_mean) / self.y_std

        # 2. Initialize or update the model
        if self.model is None:
            self.model = ExactGPModel(
                train_x_use, train_y_use, self.likelihood, self.dim_x
            ).to(self.device, dtype=self.dtype)
        else:
            self.model.set_train_data(train_x_use, train_y_use, strict=False)

        baseline_state = self._capture_state()

        last_error = None
        # 3. Training loop (Note: passing train_x_use, train_y_use)
        for jitter in self.training_jitter_values:
            self._restore_state(baseline_state)
            try:
                self._run_training_loop(jitter, train_x_use, train_y_use)
                self.model.eval()
                self.likelihood.eval()
                return
            except (RuntimeError, NotPSDError) as err:
                last_error = err
                continue

        # Rescue training attempt
        self.model.reset_hyperparameters()
        with torch.no_grad():
            self.likelihood.noise = torch.clamp(
                self.likelihood.noise, min=5e-4, max=0.5
            )
        try:
            self._run_training_loop(self.rescue_jitter, train_x_use, train_y_use)
            self.model.eval()
            self.likelihood.eval()
            return
        except (RuntimeError, NotPSDError) as err:
            raise RuntimeError(
                "GP training failed after rescue attempt. "
                f"Last error: {err} (previous {last_error})"
            ) from err

    def update(self, new_x_np: np.ndarray, new_y_np: np.ndarray):
        """
        (Public Interface) Adds new data to the model and re-fits.
        """
        if new_x_np.ndim == 1:
            new_x_np = new_x_np.reshape(1, -1)
        if new_y_np.ndim == 0:
            new_y_np = new_y_np.reshape(1)

        # 1. Convert new data chunk to GPU tensor
        new_x_torch = torch.tensor(new_x_np, dtype=self.dtype, device=self.device)
        new_y_torch = torch.tensor(new_y_np, dtype=self.dtype, device=self.device)
        
        # 2. Concatenate on GPU directly (raw data)
        self.train_x = torch.cat([self.train_x, new_x_torch], dim=0)
        self.train_y = torch.cat([self.train_y, new_y_torch], dim=0)

        # 3. Update normalization stats based on current total raw data
        if self.train_x.shape[0] > 1:
            if self.normalize_x:
                self.x_mean = self.train_x.mean(dim=0, keepdim=True)
                self.x_std = self.train_x.std(dim=0, keepdim=True)
                # Prevent division by zero for constant features
                self.x_std[self.x_std < 1e-6] = 1.0
            
            if self.normalize_y:
                self.y_mean = self.train_y.mean()
                self.y_std = self.train_y.std()
                # Prevent division by zero
                if self.y_std < 1e-6:
                    self.y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            # Fallback for single data point
            if self.normalize_x:
                self.x_mean = self.train_x.clone()
                self.x_std = torch.ones_like(self.x_mean)
            if self.normalize_y:
                self.y_mean = self.train_y.clone()
                self.y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        # 4. Call internal fit
        self._fit_model()

    def predict(self, test_x_np: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        self.model.eval()
        self.likelihood.eval()

        test_x = torch.tensor(
            test_x_np, dtype=self.dtype, device=self.device
        )

        # --- Normalize Input ---
        if self.normalize_x and self.x_mean is not None and self.x_std is not None:
            test_x = (test_x - self.x_mean) / self.x_std
        # -----------------

        # Use LOVE for accelerated predictions
        with torch.no_grad():
            means = []
            variances = []
            for x_chunk in torch.split(test_x, 1024):
                pred_dist = self.likelihood(self.model(x_chunk))
                means.append(pred_dist.mean.detach())
                variances.append(pred_dist.variance.detach())
        
        mean_full = torch.cat(means, dim=0)
        variance_full = torch.cat(variances, dim=0)

        # --- Denormalize Output ---
        if self.normalize_y and self.y_mean is not None and self.y_std is not None:
            mean_full = mean_full * self.y_std + self.y_mean
            variance_full = variance_full * (self.y_std ** 2)
        # -------------------

        return mean_full.cpu().numpy(), variance_full.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Internal Functions
    # ------------------------------------------------------------------ #
    def _run_training_loop(self, jitter: float, train_x: torch.Tensor, train_y: torch.Tensor):
        """
        Executes training loop.
        Note: Receives train_x and train_y directly as parameters as they might be pre-normalized.
        """
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_lr,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        with gpytorch.settings.cholesky_jitter(jitter), \
             gpytorch.settings.max_preconditioner_size(0), \
             gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False, solves=False):

            for _ in range(self.training_steps):
                optimizer.zero_grad(set_to_none=True)

                # Use passed-in data
                output = self.model(train_x)
                loss = -mll(output, train_y)
                
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Encountered non-finite loss (value={loss.item()}, jitter={jitter})."
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )
                optimizer.step()
                
                if _ % 20 == 0 or _ == self.training_steps - 1:
                    pass
                self._soft_parameter_projection()

    def _soft_parameter_projection(self):
        """Gentle correction of hyperparameters in extreme cases"""
        with torch.no_grad():
            ls_x = self.model.k_x.base_kernel.lengthscale
            ls_t = self.model.k_t.base_kernel.lengthscale
            ls_x.clamp_(min=1e-3, max=30.0)
            ls_t.clamp_(min=5e-4, max=10.0)

            os_x = self.model.k_x.outputscale
            os_t = self.model.k_t.outputscale
            self.model.k_x.outputscale = os_x.clamp(2e-3, 50.0)
            self.model.k_t.outputscale = os_t.clamp(2e-3, 50.0)

            self.likelihood.noise = self.likelihood.noise.clamp(5e-5, 0.6)

    def _capture_state(self):
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "likelihood": copy.deepcopy(self.likelihood.state_dict()),
        }

    def _restore_state(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.likelihood.load_state_dict(state_dict["likelihood"])

    def __call__(self, inputs: torch.Tensor):
        """
        Makes GPModel callable.
        Modified:
        1. Automatically normalizes input X (Pre-processing).
        2. Automatically denormalizes output distribution (Post-processing).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        self.model.eval()

        # 1. Input Normalization
        x_input = inputs
        if self.normalize_x and self.x_mean is not None and self.x_std is not None:
            x_input = (inputs - self.x_mean) / self.x_std

        # 2. Forward Pass
        # Returns normalized distribution
        dist_norm = self.model(x_input)

        # 3. Output Denormalization
        if self.normalize_y and self.y_mean is not None and self.y_std is not None:
            # Restore mean: mean_orig = mean_norm * std + mean
            mean_orig = dist_norm.mean * self.y_std + self.y_mean
            
            # Restore covariance: cov_orig = cov_norm * (std^2)
            cov_orig = dist_norm.covariance_matrix * (self.y_std.pow(2))
            
            return gpytorch.distributions.MultivariateNormal(mean_orig, cov_orig)
        
        return dist_norm