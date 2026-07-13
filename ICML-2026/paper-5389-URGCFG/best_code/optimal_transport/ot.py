"""Base classes and utilities for optimal transport solvers."""

import logging
import os
from typing import Any

import torch
from config import WRITING_ROOT
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from tools.feedback import LiveOrJupyter, logger, make_status_panel
from tools.utils import hash_dict


class OT:
    """Abstract optimal transport solver providing common training utilities."""
    def __init__(self,
                 outer_lr: float | None = None,
                 inner_lr: float | None = None) -> None:
        """Base OT class.

        Parameters
        ----------
        outer_lr : float | None
            Learning rate for outer optimization if applicable (subclasses may use).
        inner_lr : float | None
            Learning rate for inner/nested optimization (used by subclasses implementing
            dual potentials with inner transforms). Stored for unified interface only.
        """
        self.input_dim: int = 0
        self.device = torch.device("cpu")
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        # Placeholder attributes so base class satisfies attribute checks in _fit.
        self.model = None              # Subclasses provide nn.Module
        self.optimizer = None          # Subclasses provide optimizer
        self.inner_steps = 5           # Default fallback
        self.inner_optimizer = None    # e.g. 'lbfgs', 'adam'
        self.inner_tol = None
        self.inner_patience = None
        self.inner_lam = None

    @staticmethod
    def initialize_right_architecture(dim,
                                      n_params_target,
                                      *args,
                                      outer_lr: float | None = None,
                                      inner_lr: float | None = None,
                                      **kwargs):
        """Factory for base OT (no parametrized model). Included for API symmetry.

        Parameters mirror FCOT for consistency; outer_lr/inner_lr stored but unused.
        """
        return OT(outer_lr=outer_lr, inner_lr=inner_lr, *args, **kwargs)

    def step(self, x_batch, y_batch, *args, **kwargs) -> Any:
        """Placeholder training step that subclasses override.

        Parameters
        ----------
        x_batch : torch.Tensor
            Batch of source samples.
        y_batch : torch.Tensor
            Batch of target samples.
        *args, **kwargs :
            Additional arguments (used by subclasses such as FCOT).

        Returns
        -------
        Any
            Placeholder stats dictionary expected by _fit.
        """
        return {}

    def _dual_objective(self, *args, **kwargs):
        """Compute dual objective for the current solver state.

        Raises
        ------
        NotImplementedError
            Always raised for the base class; concrete OT subclasses implement this.
        """
        raise NotImplementedError("_dual_objective is not implemented for base OT.")
    
    def transport_X_to_Y(self, X):
        """Identity mapping for base OT (no transport defined)."""
        return X
    
    def debug_losses(self, X, Y):
        """Return diagnostic loss information for the current state."""
        return []
    
    def get_hash(self, X, Y):
        """Compute deterministic hash for model+data configuration.

        Parameters
        ----------
        X, Y : torch.Tensor
            Training datasets used to seed the hash.
        """
        info = {'X': X,
                'Y': Y,
                **self.__dict__}
        return hash_dict(info)

    def get_address(self, X, Y, dir=WRITING_ROOT):
        """Return checkpoint filepath based on model/data hash and metadata.

        Parameters
        ----------
        X, Y : torch.Tensor
            Reference datasets used to compute the hash.
        dir : str
            Directory to place the checkpoint file.
        """
        h = self.get_hash(X, Y)
        classname = self.__class__.__name__

        # Try to extract a human-readable kernel name (if available)
        kernel_name = None

        # 1) Direct attributes on the solver
        for attr in ("kernel", "kernel_fn", "cost", "cost_fn"):
            fn = getattr(self, attr, None)
            if callable(fn):
                kernel_name = getattr(fn, "__name__", fn.__class__.__name__)
                break

        # 2) Attributes on the underlying model (e.g., FiniteModel / FiniteSeparableModel)
        if kernel_name is None:
            model = getattr(self, "model", None)
            if model is not None:
                for attr in ("kernel_fn", "kernel", "cost_fn"):
                    fn = getattr(model, attr, None)
                    if callable(fn):
                        kernel_name = getattr(fn, "__name__", fn.__class__.__name__)
                        break

        extra = ""
        if kernel_name is not None:
            # Sanitize for filenames
            safe = kernel_name.replace("<", "").replace(">", "")
            safe = safe.replace(" ", "_").replace(".", "-")
            extra = f"_kernel-{safe}"

        filename = f"{classname}_dim{self.input_dim}{extra}_{h}.pt"
        return os.path.join(dir, filename)

    def save(self, address, iters_done):
        """Persist solver state to disk (overridden by subclasses)."""
        return None

    def load(self, address):
        """Restore saved solver state (returns iterations done)."""
        return 0
    
    def _get_active_inner_steps(self, inner_steps):
        """
        Helper to determine which inner_steps value to use.
        
        Parameters
        ----------
        inner_steps : int | None
            Hint provided by the caller. Fallback defaults to `self.inner_steps`.
        
        Returns
        -------
        int
            Inner loop iterations that should be executed.
        """
        if inner_steps is not None:
            return inner_steps
        # Fall back to instance attribute if it exists
        if hasattr(self, 'inner_steps'):
            return self.inner_steps
        # Default fallback
        return 5
    
    def _fit(self,
            X,
            Y,
            iters_done=0,
            iters=10000,
            inner_steps=5,
            print_every=50,
            callback=None,
            convergence_tol=1e-4,
            convergence_patience=50,
            batch_size=None,
            eval_every=100,
            warmup_steps=None,
            warmup_until_converged=False,
            log_every=None,
            log_level="info",
        ):
        """
        Internal fit function with rich logging and progress tracking.
        
        This method handles:
        - Live status panel updates with training metrics
        - Periodic full-data evaluation
        - Convergence detection
        - Gradient monitoring
        - Inner optimization tracking (for applicable subclasses)
        
        Args:
            X, Y: Training data
            iters_done: Number of iterations already completed
            iters: Total iterations to run
            inner_steps: Number of steps for inner optimization loops.
                         Subclasses may use this to override their default
                         inner_steps value when not None.
            print_every: How often to update the Rich status panel (iterations)
            eval_every: How often to evaluate on full dataset (iterations)
            callback: Optional callback(iteration) called each iteration
            convergence_tol: Relative tolerance for convergence detection
            convergence_patience: Number of checks before declaring convergence
            batch_size: Mini-batch size for stochastic training. If None, uses full batch (no sampling).
            warmup_steps: Number of inner optimization steps for initial warm-up pass.
                         If None, uses max(10, active_inner_steps). Set to 0 to disable warm-up.
                         Ignored if warmup_until_converged=True.
            warmup_until_converged: If True, run warm-up until all points converge (ignores warmup_steps).
                                   Uses adaptive steps with progress tracking. Good for ensuring all
                                   warm starts are properly initialized before training.
            log_every: How often to write debug logs (iterations). If None, debug logging is disabled.
                      Set to an integer (e.g., 1, 10, 100) to enable periodic debug output.
            log_level: String log level ("debug", "info", ...). Applied to both the feedback logger
                       and the root logger when periodic logging is enabled.
            
        Returns:
            logs: dict with training metrics including dual_obj, converged, etc.
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        nX, nY = X.shape[0], Y.shape[0]
        effective_log_level = (log_level or "info").lower()

        # Precompute full index tensor for Y once (reused for warm-up and full-batch)
        idx_y_full = torch.arange(nY, device=self.device)
        
        # Determine active inner_steps for inner optimization
        active_inner_steps = self._get_active_inner_steps(inner_steps)
        
        # Initialize logging dictionaries
        logs = {
            "dual_obj": [],           # Dual objective on minibatch
            "dual_obj_full": [],      # Dual objective on full dataset
            "converged": [],
            "inner_converged": [],    # Per-iteration inner convergence
            "inner_failures": 0,
            "grad_norms": [],         # Gradient norms
        }
        
        prev_dual = None
        patience = 0
        converged_flag = False
        it = 0  # Initialize to avoid unbound variable
        
        # Warm-up pass: Initialize all warm starts on full dataset
        # This is crucial for efficient training - ensures all samples
        # have good starting points for inner optimization
        if (
            iters_done == 0
            and hasattr(self, '_dual_objective')
            and not getattr(self, "_skip_warmup", False)
        ):
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
            
            if warmup_until_converged:
                # Adaptive warm-up: Run until all points converge
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Warm-up (until converged):"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TextColumn("[cyan]{task.fields[status]}"),
                ) as progress:
                    # Start with unknown total, will update as we learn convergence rate
                    task = progress.add_task(
                        "Initializing...",
                        total=100,  # Initial estimate
                        status=f"Full dataset (nY={nY})"
                    )
                    
                    prev_level = logger.level
                    logger.setLevel(logging.ERROR)  # Suppress warnings during warm-up
                    
                    try:
                        # Run in batches of steps, checking convergence
                        total_steps = 0
                        converged_count = 0
                        batch_steps = max(10, active_inner_steps)
                        
                        while converged_count < nY:
                            # Run a batch of inner optimization steps
                            progress.update(task, status=f"Step {total_steps}, {converged_count}/{nY} converged")
                            _, _, _, inner_converged_batch = self._dual_objective(X, Y, idx_y_full, batch_steps)
                            total_steps += batch_steps
                            
                            # Count how many points have converged
                            # inner_converged_batch is either a bool or array of bools
                            if isinstance(inner_converged_batch, bool):
                                converged_count = nY if inner_converged_batch else 0
                            else:
                                converged_count = int(inner_converged_batch.sum().item())
                            
                            # Update progress
                            progress.update(
                                task,
                                completed=converged_count,
                                total=nY,
                                status=f"Step {total_steps}, {converged_count}/{nY} converged"
                            )
                            
                            # Safety: Stop after too many steps
                            if total_steps > active_inner_steps * 20:
                                progress.update(task, status=f"Stopped at {total_steps} steps (max reached)")
                                logger.warning(f"[WARMUP] Stopped after {total_steps} steps with {converged_count}/{nY} converged")
                                break
                        
                        progress.update(task, completed=nY, status=f"Complete! ({total_steps} steps)")
                        
                    finally:
                        logger.setLevel(prev_level)
                        
            else:
                # Fixed warm-up steps - run in batches to show progress
                if warmup_steps is None:
                    actual_warmup_steps = max(10, active_inner_steps)
                else:
                    actual_warmup_steps = warmup_steps
                
                # Only do warm-up if warmup_steps > 0
                if actual_warmup_steps > 0:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]Warm-up:"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeRemainingColumn(),
                        TextColumn("•"),
                        TextColumn("[cyan]{task.fields[status]}"),
                    ) as progress:
                        task = progress.add_task(
                            "Initializing...",
                            total=actual_warmup_steps,
                            status=f"Full dataset (nY={nY})"
                        )
                        
                        prev_level = logger.level
                        logger.setLevel(logging.ERROR)  # Suppress warnings during warm-up
                        
                        try:
                            # Run warm-up in batches to show progress
                            batch_size = min(10, active_inner_steps)  # Steps per batch
                            steps_done = 0
                            
                            while steps_done < actual_warmup_steps:
                                steps_remaining = actual_warmup_steps - steps_done
                                steps_this_batch = min(batch_size, steps_remaining)
                                
                                progress.update(task, completed=steps_done, 
                                              status=f"Step {steps_done}/{actual_warmup_steps}")
                                
                                # Run a batch of inner optimization steps
                                _, _, _, _ = self._dual_objective(X, Y, idx_y_full, steps_this_batch)
                                steps_done += steps_this_batch
                            
                            progress.update(task, completed=actual_warmup_steps, status="Complete!")
                            
                        except Exception as e:
                            progress.update(task, status=f"Failed: {e}")
                            logger.warning(f"[WARMUP] Failed to initialize warm starts: {e}")
                        finally:
                            logger.setLevel(prev_level)
        
        # Rich logging with live panel
        with LiveOrJupyter() as live:
            for it in range(iters):
                # Sample minibatches or use full batch
                if batch_size is None:
                    # Full batch training - use all data and full index for warm starts
                    Xb = X
                    Yb = Y
                    idx_y = idx_y_full
                else:
                    # Mini-batch training - random sampling
                    idx_x = torch.randint(0, nX, (batch_size,), device=self.device)
                    idx_y = torch.randint(0, nY, (batch_size,), device=self.device)
                    
                    Xb = X[idx_x]
                    Yb = Y[idx_y]
                
                # Perform one training step (subclass-specific)
                # Pass additional args only if method accepts them (FCOT does)
                if getattr(self.step, "__code__", None) and self.step.__code__.co_argcount >= 5:
                    # Signature supports (self, x_batch, y_batch, idx_y, active_inner_steps)
                    stats = self.step(Xb, Yb, idx_y, active_inner_steps)
                else:
                    stats = self.step(Xb, Yb)
                
                # Extract metrics from step (with defaults)
                dual_batch = stats.get("dual", 0.0) if isinstance(stats, dict) else 0.0
                grad_norm = stats.get("grad_norm", 0.0) if isinstance(stats, dict) else 0.0
                u_mean = stats.get("u_mean", 0.0) if isinstance(stats, dict) else 0.0
                uc_mean = stats.get("uc_mean", 0.0) if isinstance(stats, dict) else 0.0
                inner_converged = stats.get("inner_converged", True) if isinstance(stats, dict) else True
                
                # Track inner convergence
                logs["inner_converged"].append(inner_converged)
                if not inner_converged:
                    logs["inner_failures"] += 1
                
                # Compute gradient norm (if model has parameters)
                # grad_norm = 0.0
                # model_ref = getattr(self, 'model', None)
                # if model_ref is not None and hasattr(model_ref, 'parameters'):
                #     grad_norm = sum(
                #         p.grad.norm().item() ** 2
                #         for p in model_ref.parameters()
                #         if p.grad is not None
                #     ) ** 0.5
                logs["grad_norms"].append(grad_norm)
                
                # Convergence check (every iteration)
                if prev_dual is not None:
                    rel_change = abs(dual_batch - prev_dual) / (abs(prev_dual) + 1e-12)
                    if rel_change < convergence_tol:
                        patience += 1
                    else:
                        patience = 0
                    
                    if patience >= convergence_patience:
                        logger.info(
                            f"[CONVERGED] Dual rel change < {convergence_tol:.2e} "
                            f"for {convergence_patience} consecutive steps."
                        )
                        converged_flag = True
                        logs["dual_obj"].append(dual_batch)
                        logs["converged"].append(converged_flag)
                        break
                
                prev_dual = dual_batch
                
                # Periodic full dataset evaluation
                dual_full = None
                if it % eval_every == 0 and it > 0:
                    # Evaluate dual objective on full dataset (no gradient computation)
                    with torch.no_grad():
                        # For FCOT: just compute dual objective without backward
                        if hasattr(self, '_dual_objective'):
                            try:
                                D_full, u_full, uc_full, _ = self._dual_objective(X, Y, None, active_inner_steps)
                                dual_full = float(D_full.item())
                            except:
                                dual_full = None
                        logs["dual_obj_full"].append(dual_full if dual_full is not None else dual_batch)
                
                # Update panel every iteration for live feedback
                # Compute running statistics
                recent_inner_conv = logs["inner_converged"][-min(100, len(logs["inner_converged"])):]  
                inner_conv_rate = sum(recent_inner_conv) / len(recent_inner_conv) if recent_inner_conv else 1.0
                
                recent_grads = logs["grad_norms"][-min(100, len(logs["grad_norms"])):]  
                avg_grad_norm = sum(recent_grads) / len(recent_grads) if recent_grads else 0.0
                
                # Build panel data
                panel_data = {
                    "iteration": it + iters_done,
                    "dual (batch)": dual_batch,
                    "u(X) mean": u_mean,
                    "u^c(Y) mean": uc_mean,
                    "grad_norm": grad_norm,
                    "avg_grad_norm": avg_grad_norm,
                    "inner_conv_rate": f"{inner_conv_rate:.1%}",
                    "inner_failures": logs["inner_failures"],
                    "convergence": f"{patience}/{convergence_patience}",
                    "batch_size": batch_size if batch_size is not None else "full",
                }
                temp_val = getattr(getattr(self, "model", None), "temp", None)
                if temp_val is not None:
                    panel_data["temp"] = temp_val
                active_frac = stats.get("intercept_active_frac") if isinstance(stats, dict) else None
                if active_frac is not None:
                    panel_data["active_intercepts"] = f"{active_frac*100:.1f}%"
                
                # Add full dataset metrics if available
                if dual_full is not None:
                    panel_data["dual (full)"] = dual_full
                
                # Add optimizer-specific info (if available)
                opt_ref = getattr(self, 'optimizer', None)
                if opt_ref is not None and hasattr(opt_ref, 'param_groups'):
                    panel_data["lr"] = opt_ref.param_groups[0]['lr']

                if getattr(self, 'inner_optimizer', None) is not None:
                    panel_data["inner_opt"] = self.inner_optimizer
                    panel_data["inner_steps"] = active_inner_steps
                    if getattr(self, 'inner_tol', None) is not None:
                        panel_data["inner_tol"] = self.inner_tol
                    if getattr(self, 'inner_lr', None) is not None:
                        panel_data["inner_lr"] = self.inner_lr
                
                # Create and update panel every iteration
                panel_data["desc"] = f"{self.__class__.__name__} Training"
                panel = make_status_panel(panel_data)
                live.update(panel)
                
                # Periodic logging (save to logs and optionally to debug output)
                if it % print_every == 0:
                    logs["dual_obj"].append(dual_batch)
                    logs["converged"].append(converged_flag)
                
                # Debug/logging at user-specified frequency
                if log_every is not None and log_every > 0 and it % log_every == 0:
                    msg = (
                        f"[Iter {it + iters_done}] dual={dual_batch:.6f} "
                        f"u={u_mean:.4f} uc={uc_mean:.4f} grad={grad_norm:.2e} "
                        f"inner_conv={inner_conv_rate:.1%} failures={logs['inner_failures']}"
                    )
                    if temp_val is not None:
                        msg += f" temp={temp_val:.2f}"
                    if active_frac is not None:
                        msg += f" active={active_frac*100:.1f}%"
                    level_name = effective_log_level
                    level_value = {
                        "debug": logging.DEBUG,
                        "info": logging.INFO,
                        "warning": logging.WARNING,
                        "error": logging.ERROR,
                    }.get(level_name, logging.INFO)
                    if logger.level > level_value:
                        logger.setLevel(level_value)
                    root_logger = logging.getLogger()
                    if root_logger.level > level_value:
                        root_logger.setLevel(level_value)
                    logger.log(level_value, msg)
                
                # Callback
                if callback is not None:
                    callback(it)
        
        # Final convergence status
        if logs["converged"]:
            logs["converged"][-1] = converged_flag
        
        # Report summary statistics
        total_iters = it + 1
        failure_rate = (logs["inner_failures"] / total_iters) * 100 if total_iters > 0 else 0.0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total iterations: {total_iters}")
        logger.info(f"Converged: {converged_flag}")
        if prev_dual is not None:
            logger.info(f"Final dual objective: {prev_dual:.6e}")
        
        if hasattr(self, 'inner_optimizer'):
            logger.info(f"Inner convergence failures: {logs['inner_failures']} ({failure_rate:.2f}%)")
            if failure_rate > 10:
                logger.warning(
                    f"High inner optimization failure rate ({failure_rate:.1f}%). "
                    f"Consider: (1) increasing inner_steps, (2) relaxing inner_tol, "
                    f"or (3) trying different inner_optimizer."
                )
        
        logger.info(f"{'='*70}\n")
        
        return logs
        
    def fit(self,
            X,
            Y,
            iters=10000,
            inner_steps=5,
            print_every=50,
            log_every=None,
            log_level="info",
            callback=None,
            force_retrain=False,
            convergence_tol=1e-4,
            convergence_patience=50,
            batch_size=None,
            warmup_steps=None,
            warmup_until_converged=False):
        """
        Full-batch ICNN-OT training with:
        - caching by architecture+data hash
        - checkpoint resume
        - early stopping by W2 convergence
        
        Args:
            X, Y: Training data (numpy arrays or tensors)
            iters: Number of training iterations
            inner_steps: Steps for inner optimization. If provided, overrides
                         the default value set during initialization.
            print_every: How often to update Rich visual display (default: 50)
            log_every: How often to write debug logs. If None, logs are disabled.
                      Set to an integer to log every N iterations.
            callback: Optional function(iter) called each iteration
            force_retrain: If True, ignore cached checkpoints
            convergence_tol: Relative change threshold for convergence
            convergence_patience: Number of checks before stopping
            batch_size: Mini-batch size for training. If None, uses full batch (len(X))
            warmup_steps: Number of inner optimization steps for initial warm-up pass.
                         If None, uses max(10, inner_steps). Set to 0 to disable warm-up.
                         Ignored if warmup_until_converged=True.
            warmup_until_converged: If True, run warm-up until all points converge (adaptive).
                                   Good for ensuring high-quality warm starts before training.
        Returns
        -------
        dict
            Logs collected during training (dual objectives, convergence flags, etc.).
        """

        # --------------------------------------------------------
        # Convert data to tensors
        # --------------------------------------------------------
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        if not torch.is_tensor(Y):
            Y = torch.from_numpy(Y).float()

        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Note: batch_size can be None for full-batch training
        # Don't replace None with full size here - let _fit handle it
        
        # --------------------------------------------------------
        # Compute save location
        # --------------------------------------------------------
        address = self.get_address(X, Y)
        os.makedirs(os.path.dirname(address), exist_ok=True)

        # --------------------------------------------------------
        # Load checkpoint if exists
        # --------------------------------------------------------
        iters_done = 0
        if os.path.exists(address) and not force_retrain:
            logger.info(f"[CACHE] Found saved ICNN-OT model:\n  {address}")
            iters_done = self.load(address)             
            logger.info(f"[CACHE] Model previously trained for {iters_done} iterations.")
            # If we already reached or exceeded desired iters → return
            if iters_done >= iters:
                logger.info("[CACHE] Requested iterations already satisfied. Skipping training.")
                return {"f_losses": [], "g_losses": [], "W2": []}
            logger.info(f"[RESUME] Resuming training for {iters - iters_done} more iterations.")
        else:
            if force_retrain:
                logger.info("[FORCE] Forced retrain from scratch.")
            else:
                logger.info(f"[TRAIN] No checkpoint found at `{address}`. Training from scratch.")

        logs = self._fit(X, Y, iters_done, iters, inner_steps, print_every, callback, convergence_tol, convergence_patience, batch_size, warmup_steps=warmup_steps, warmup_until_converged=warmup_until_converged, log_every=log_every, log_level=log_level)
        # --------------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------------
        logger.info(f"[SAVE] Writing checkpoint to:\n  {address}")
        self.save(address, iters)
        return logs
