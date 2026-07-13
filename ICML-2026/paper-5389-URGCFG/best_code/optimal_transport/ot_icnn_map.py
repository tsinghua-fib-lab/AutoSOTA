"""ICNN-based optimal transport solver with alternating f/g steps."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimal_transport.ot import OT
from tools.feedback import logger

def count_icnn_params(dim, hidden_sizes):
    """
    Count parameters of an ICNN given layer widths.

    Parameters
    ----------
    dim : int
        Input dimensionality.
    hidden_sizes : Sequence[int]
        Widths for each hidden layer (final width usually 1 for scalar output).

    Returns
    -------
    int
        Total parameter count for A_k, b_k, and W_k matrices.
    """
    total = 0
    # A_k and b_k
    for h in hidden_sizes:
        total += dim * h     # A_k
        total += h           # b_k (1×h)
    # W_k for k>=2
    for h_prev, h_next in zip(hidden_sizes[:-1], hidden_sizes[1:]):
        total += h_prev * h_next
    return total

def choose_icnn_architecture(dim, n_params_target, 
                             depth_range=(2, 8), 
                             width_range=(4, 2048)):
    """
    Search for ICNN architecture that closely matches a target parameter budget.

    Parameters
    ----------
    dim : int
        Input dimensionality of the ICNN.
    n_params_target : int
        Desired total number of parameters.
    depth_range : tuple[int, int]
        Inclusive search range for the number of layers.
    width_range : tuple[int, int]
        Range of widths to try per hidden layer.

    Returns
    -------
    tuple[int, list[int], int]
        (depth, hidden_sizes, actual_param_count) for the best matching architecture.
    """

    best = None
    best_diff = float("inf")

    for depth in range(depth_range[0], depth_range[1] + 1):
        # We will search width for layers 1..(depth-1), last layer is output=1
        for width in range(width_range[0], width_range[1] + 1):
            hidden_sizes = [width] * (depth - 1) + [1]
            count = count_icnn_params(dim, hidden_sizes)
            diff = abs(count - n_params_target)

            if diff < best_diff:
                best_diff = diff
                best = (depth, hidden_sizes, count)

            # exact match possible
            if diff == 0:
                return best

    return best


class KantorovichPotential(nn.Module):
    """
    Modelling the Kantorovich potential as an Input Convex Neural Network (ICNN).

    input:  y (batch_size, input_size)
    output: z = h_L (batch_size, 1) if hidden_size_list ends with 1

    Architecture:
        h_1     = ReLU^2(A_0 y + b_0)
        h_{l+1} = LeakyReLU(A_l y + b_l + W_{l-1} h_l),  for l >= 1

    Constraint: W_l >= 0 (enforced by projection and/or penalty).
    """

    def __init__(self, input_size, hidden_size_list):
        """
        Initialize trainable ICNN parameters for the Kantorovich potential.

        Parameters
        ----------
        input_size : int
            Dimensionality of each input sample.
        hidden_size_list : Sequence[int]
            Width of each hidden layer (last entry should be 1 for scalar output).
        """
        super().__init__()
        # hidden_size_list always contains 1 at the end (scalar output)
        self.input_size = input_size
        self.num_hidden_layers = len(hidden_size_list)

        # A_k: matrices that interact with inputs (input_size x hidden_k)
        self.A = nn.ParameterList([
            nn.Parameter(torch.empty(input_size, hidden_size_list[k]))
            for k in range(self.num_hidden_layers)
        ])

        # b_k: bias vectors (1 x hidden_k)
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_size_list[k]))
            for k in range(self.num_hidden_layers)
        ])

        # W_k: matrices between consecutive layers (hidden_{k-1} x hidden_k)
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size_list[k - 1], hidden_size_list[k]))
            for k in range(1, self.num_hidden_layers)
        ])

        # Initialization similar to tf.random_uniform(maxval=0.1)
        for A_k in self.A:
            nn.init.uniform_(A_k, a=0.0, b=0.1)
        for W_k in self.W:
            nn.init.uniform_(W_k, a=0.0, b=0.1)

    @property
    def positive_constraint_loss(self):
        """
        Equivalent to:
            tf.add_n([tf.nn.l2_loss(tf.nn.relu(-w)) for w in self.W])

        tf.nn.l2_loss(x) = sum(x^2) / 2
        """
        loss = 0.0
        for w in self.W:
            neg_part = F.relu(-w)          # relu(-w) >= 0 where w < 0
            loss = loss + 0.5 * (neg_part ** 2).sum()
        return loss

    @torch.no_grad()
    def proj(self):
        """
        Projection step to enforce W_l >= 0:
            w.assign(tf.nn.relu(w)) in TF.
        """
        for w in self.W:
            w.clamp_(min=0.0)

    def forward(self, input_y):
        """
        input_y: (batch_size, input_size)
        returns: (batch_size, hidden_size_list[-1]) (usually (batch_size, 1))
        """
        # First layer: leaky ReLU then squared
        z = input_y @ self.A[0] + self.b[0]             # (B, hidden_0)
        z = F.leaky_relu(z, negative_slope=0.2)
        z = z * z                                       # ReLU^2 as in TF code

        # Subsequent layers: leaky ReLU(A_k y + b_k + W_{k-1} z)
        for k in range(1, self.num_hidden_layers):
            z = input_y @ self.A[k] + self.b[k] + z @ self.W[k - 1]
            z = F.leaky_relu(z, negative_slope=0.2)

        return z

    def get_hidden_size(self):
        """
        Return the per-layer widths (including the final scalar output).
        """
        return [A.shape[1] for A in self.A]   # hidden sizes from A matrices


class ICNNOT(OT):
    """
    PyTorch version of the TensorFlow ComputeOT driver.

    Supports:
      - alternating f/g optimization steps
      - transport_X_to_Y, transport_Y_to_X
      - compute_W2
      - architecture auto-selection via initialize_right_architecture
    """

    def __init__(self, input_dim, f_model, g_model, lr,
                 lambda_reg=1.0, betas=(0.5, 0.9), device="cpu"):
        """
        Initialize alternating ICNN optimal transport solver.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input data.
        f_model : nn.Module | None
            Model representing u potential.
        g_model : nn.Module | None
            Model representing v potential.
        lr : float
            Learning rate for inner Adam optimizers.
        lambda_reg : float
            Coefficient for positivity penalty on g_model.
        betas : tuple[float, float]
            Adam beta parameters.
        device : str
            Device identifier (\"cpu\", \"cuda\", ...).
        """

        self.device  = torch.device(device)
        self.lr      = lr
        self.betas   = betas
        self.lambda_reg = lambda_reg
        self.input_dim  = input_dim

        # Models
        self.f_model = f_model.to(self.device) if f_model is not None else None
        self.g_model = g_model.to(self.device) if g_model is not None else None

        # Optimizers (only if models exist)
        if self.f_model is not None:
            self.f_optimizer = torch.optim.Adam(
                self.f_model.parameters(), lr=lr, betas=betas
            )
        if self.g_model is not None:
            self.g_optimizer = torch.optim.Adam(
                self.g_model.parameters(), lr=lr, betas=betas
            )


    # ----------------------------------------------------------------------
    #  Architecture Auto-Initialization
    # ----------------------------------------------------------------------
    @staticmethod
    def initialize_right_architecture(dim,
                                      n_params_target,
                                      depth_range=(2, 8),
                                      width_range=(4, 2048),
                                      device="cpu",
                                      *args,
                                      **kwargs):
        """
        Determine suitable ICNN depth/width and instantiate f/g models.

        Parameters
        ----------
        dim : int
            Input dimension of the transport maps.
        n_params_target : int
            Target budget for trainable parameters.
        depth_range : tuple[int, int]
            Inclusive range of depths to search.
        width_range : tuple[int, int]
            Width range for hidden layers.
        device : str
            Device label for the initialized models.

        Returns
        -------
        ICNNOT
            Fully configured OT solver built from auto-selected ICNNs.
        """
        # local imports or replace with your file paths
        # Search for architecture
        depth, hidden_sizes, actual = choose_icnn_architecture(
            dim,
            n_params_target,
            depth_range=depth_range,
            width_range=width_range
        )

        logger.info(f"[ARCH] dim={dim}, target={n_params_target}")
        logger.info(f"       depth        = {depth}")
        logger.info(f"       hidden_sizes = {hidden_sizes}")
        logger.info(f"       n_params     = {actual}")

        # Build two ICNN models
        f_model = KantorovichPotential(dim, hidden_sizes).to(device)
        g_model = KantorovichPotential(dim, hidden_sizes).to(device)
        return ICNNOT(
            dim,
            f_model,
            g_model,
            lr=1e-3,
            lambda_reg=1.0,
            betas=(0.5, 0.9),
            device=device
        )
    # ----------------------------------------------------------------------
    #  Core Computations
    # ----------------------------------------------------------------------
    def _compute_core(self, x, y, create_graph=True):
        """
        Evaluate potentials, gradients, and loss terms for a batch.

        Parameters
        ----------
        x : torch.Tensor
            Batch of source samples.
        y : torch.Tensor
            Batch of target samples.
        create_graph : bool
            Whether to retain gradient graph for higher-order autodiff.

        Returns
        -------
        dict
            Contains fx, gy, gradients, losses, and dual W2 objective scalars.
        """

        x = x.to(self.device)
        y = y.to(self.device)
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Evaluate u and v potentials on the current minibatch
        fx = self.f_model(x)         # (B,1)
        gy = self.g_model(y)         # (B,1)

        # ∇f(x)
        grad_fx = torch.autograd.grad(
            fx.sum(), x, create_graph=create_graph
        )[0]

        # ∇g(y)
        grad_gy = torch.autograd.grad(
            gy.sum(), y, create_graph=create_graph
        )[0]

        # f(∇g(y))
        f_grad_gy = self.f_model(grad_gy)

        # <y, ∇g(y)>
        y_dot_grad_gy = (y * grad_gy).sum(dim=1, keepdim=True)

        # Square norms for regularization terms
        x_squared = (x * x).sum(dim=1, keepdim=True)
        y_squared = (y * y).sum(dim=1, keepdim=True)

        # Loss components driving alternating optimization
        f_loss = (fx - f_grad_gy).mean()
        g_loss = (f_grad_gy - y_dot_grad_gy).mean()

        # Dual W₂ objective
        W2 = (f_grad_gy - fx - y_dot_grad_gy
              + 0.5 * x_squared + 0.5 * y_squared).mean()

        return {
            "fx": fx,
            "gy": gy,
            "grad_fx": grad_fx,
            "grad_gy": grad_gy,
            "f_grad_gy": f_grad_gy,
            "y_dot_grad_gy": y_dot_grad_gy,
            "x_squared": x_squared,
            "y_squared": y_squared,
            "f_loss": f_loss,
            "g_loss": g_loss,
            "W2": W2,
        }


    # ----------------------------------------------------------------------
    #  Training Steps
    # ----------------------------------------------------------------------
    def step_g(self, x_batch, y_batch):
        """
        g-step: minimize g_loss + λ * positivity_penalty
        """

        self.g_optimizer.zero_grad()
        out = self._compute_core(x_batch, y_batch, create_graph=True)

        g_loss = out["g_loss"]
        if self.lambda_reg > 0:
            g_loss = g_loss + self.g_model.positive_constraint_loss

        g_loss.backward()
        self.g_optimizer.step()

        if self.lambda_reg <= 0:
            self.g_model.proj()   # TF mirrors this case

        return g_loss.item()


    def step(self, x_batch, y_batch):
        """
        f-step: minimize f_loss, always project W≥0 afterwards (like TF)
        """

        self.f_optimizer.zero_grad()
        out = self._compute_core(x_batch, y_batch, create_graph=True)

        f_loss = out["f_loss"]
        f_loss.backward()
        self.f_optimizer.step()

        # Always project
        self.f_model.proj()

        return f_loss.item()


    def transport_X_to_Y(self, X):
        """
        Return gradient map ∇f that transports X → Y.
        """
        X = X.to(self.device)
        X.requires_grad_(True)
        fx = self.f_model(X)
        grad_fx = torch.autograd.grad(fx.sum(), X)[0]
        return grad_fx

    def transport_Y_to_X(self, Y):
        """
        Return gradient map ∇g that transports Y → X.
        """
        Y = Y.to(self.device)
        Y.requires_grad_(True)
        gy = self.g_model(Y)
        grad_gy = torch.autograd.grad(gy.sum(), Y)[0]
        return grad_gy

    def debug_losses(self, X, Y):
        """
        Return loss diagnostics (f_loss, g_loss, W2) without gradient tracking.
        """
        out = self._compute_core(X, Y, create_graph=False)
        return [out["f_loss"].item(),
                out["g_loss"].item(),
                out["W2"].item()]
    
    def _fit(self,
            X,
            Y,
            iters_done=0,
            iters=10000,
            inner_steps=5,
            print_every=50,
            callback=None,
            convergence_tol=1e-4,
            convergence_patience=50):
        """
        Full-batch ICNN-OT training loop with early stopping by W2 convergence.

        Notes
        -----
        - `inner_steps` controls how many g-updates precede each f-update.
        - `lambda_reg`>0 uses an L2 penalty on negative W (no projection);
          `lambda_reg`<=0 projects W≥0 after each g-step.
        - Caching / checkpoint resume are provided by the OT base class; this
          method just runs the raw alternating updates for the given data.
        
        Returns
        -------
        dict
            Logs for f/g losses and W2 over the training run.
        """
        # --------------------------------------------------------
        # Begin training (fresh or resumed)
        # --------------------------------------------------------
        logs = {"f_losses": [], "g_losses": [], "W2": []}

        # Use parent's helper to determine active inner_steps
        active_inner_steps = self._get_active_inner_steps(inner_steps)

        prev_w2 = None
        convergence_counter = 0

        # MAIN LOOP
        for it in range(iters_done, iters):

            # ---------------------------
            # g steps
            # ---------------------------
            for _ in range(active_inner_steps):
                self.step_g(X, Y)

            # ---------------------------
            # f step
            # ---------------------------
            self.step(X, Y)

            # ---------------------------
            # Compute losses (required for convergence)
            # ---------------------------
            f_loss, g_loss, w2 = self.debug_losses(X, Y)

            # ---------------------------
            # Logging
            # ---------------------------
            if it % print_every == 0:
                logs["f_losses"].append(f_loss)
                logs["g_losses"].append(g_loss)
                logs["W2"].append(w2)
                logger.debug(f"[Iter {it}] f={f_loss:.4f}, g={g_loss:.4f}, W2={w2:.6f}")

            if callback is not None:
                callback(it)

            # ---------------------------
            # Early stopping by convergence
            # ---------------------------
            if prev_w2 is not None:
                rel_change = abs(w2 - prev_w2) / (abs(prev_w2) + 1e-12)

                if rel_change < convergence_tol:
                    convergence_counter += 1
                else:
                    convergence_counter = 0

                if convergence_counter >= convergence_patience:
                    logger.info(f"[CONVERGED] Relative W2 change < {convergence_tol} "
                        f"for {convergence_patience} iterations.")
                    iters = it + 1
                    break

            prev_w2 = w2

        return logs
    
    def save(self, address, iters_done):
        """Serialize ICNN states and iterations completed."""
        torch.save(
            {
                "f_model_state_dict": self.f_model.state_dict(),
                "g_model_state_dict": self.g_model.state_dict(),
                "iters_done": iters_done,
            },
            address
        )

    def load(self, address):
        """Restore saved ICNN states and return iterations completed."""
        data = torch.load(address, map_location=self.device)
        self.f_model.load_state_dict(data["f_model_state_dict"])
        self.g_model.load_state_dict(data["g_model_state_dict"])
        iters_done = data.get("iters_done", 0)
        return iters_done
