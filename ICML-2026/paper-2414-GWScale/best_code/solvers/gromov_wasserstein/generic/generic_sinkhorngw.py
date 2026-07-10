"""
Generic Sinkhorn-Based Gromov-Wasserstein Solver

This module provides a generic framework for solving Gromov-Wasserstein problems using Sinkhorn iterations.
It includes monitoring capabilities and adaptive scheduling for the inner Sinkhorn solver.
"""

import logging
import pandas as pd
import time
import torch

from solvers.eot.sinkhorn import SinkhornSolver


class GromovSolverMonitor:
    """
    Monitor and record statistics during Gromov-Wasserstein optimization.

    This class tracks various metrics during the optimization process including energies, errors, computation time,
    transport plans, and potentials. It can export the collected data as a pandas DataFrame for analysis.

        Attributes:
        solver: The Gromov-Wasserstein solver being monitored
        verbose (bool): Whether to log iteration information
        record (bool): Whether to activate statistics recording
        record_energies (bool): Whether to record energy values
        record_errors (bool): Whether to record convergence errors
        record_sink_errors (bool): Whether to record Sinkhorn marginal errors
        record_sink_iters (bool): Whether to record Sinkhorn iteration counts
        record_time (bool): Whether to record cumulative time
        record_transport_plans (bool): Whether to store transport plans
        record_potentials (bool): Whether to store potential matrices
        approx_energies (bool): Whether to use approximate energy computation
        include_divergence (bool): Whether to include KL divergence in energy
        record_list (list): List storing recorded statistics per iteration
        logger: Logger instance for the solver
        cum_time (float): Cumulative computation time
        last_timer (float): Timestamp of last measurement
    """

    def __init__(self,
                 solver,
                 verbose=False,
                 record=None,
                 record_losses=False,
                 record_errors=False,
                 record_sink_errors: bool = False,
                 record_sink_iters: bool = False,
                 record_time: bool = False,
                 record_transport_plans: bool = False,
                 record_potentials: bool = False,
                 approx_losses=False,
                 include_divergence=True):
        """
        Initialize the solver monitor.

        Args:
            solver: Gromov-Wasserstein solver instance to monitor
            verbose: If True, log iteration information
            record: If True, enable recording. If None, auto-enabled if any record_* flag is True
            record_losses: Record loss values at each iteration
            record_errors: Record convergence errors at each iteration
            record_sink_errors: Record Sinkhorn marginal errors
            record_sink_iters: Record number of Sinkhorn iterations performed
            record_time: Record cumulative computation time
            record_transport_plans: Store transport plans (memory intensive)
            record_potentials: Store potential matrices (memory intensive)
            approx_losses: Use approximate loss computation for speed
            include_divergence: Include KL divergence term in loss calculation
        """
        self.solver = solver

        self.verbose = verbose

        if record is None:
            record = record_losses or record_errors or record_sink_errors or record_sink_iters or record_transport_plans or record_potentials
        self.record = record

        self.record_energies = record_losses
        self.record_errors = record_errors
        self.record_sink_errors = record_sink_errors
        self.record_sink_iters = record_sink_iters
        self.record_time = record_time
        self.record_transport_plans = record_transport_plans
        self.record_potentials = record_potentials

        self.approx_energies = approx_losses
        self.include_divergence = include_divergence

        self.record_list = []
        self.logger = logging.getLogger(self.solver.__class__.__name__)

        self.last_timer = time.time()
        self.cum_time = 0.

    def _record_statistics(self, iteration):
        """
        Record statistics for the current iteration.

        Args:
            iteration: Current iteration number
            time_elapsed: Time elapsed since last measurement
        """
        new_record = {'iteration': iteration, 'solver': str(self.solver.__class__.__name__), **self.solver.parameters()}

        if self.record_energies:
            loss = self.solver.loss(approx=self.approx_energies, include_divergence=self.include_divergence).item()
            new_record['loss'] = loss

        if self.record_errors:  # TODO vérifier
            new_record['error'] = self.solver.variation_measurement()

        if self.record_sink_errors:
            sink_errors = self.solver.sinkhorn_solver.marginal_errors()
            new_record['sink_error_x'] = sink_errors[0].item()
            new_record['sink_error_y'] = sink_errors[1].item()

        if self.record_sink_iters:  # TODO vérifier
            new_record['sink_iter'] = self.solver.sinkhorn_solver.last_numIters

        if self.record_time:
            new_record['time'] = self.cum_time

        if self.record_transport_plans:
            new_record['transport_plan'] = self.solver.transport_plan(lazy=False).cpu().numpy()

        if self.record_potentials:
            new_record['potentials'] = self.solver.Z.cpu().numpy().copy()

        self.record_list.append(new_record)

    def monitor_step(self, iteration=None):
        """
        Monitor a single solver iteration.

        Args:
            iteration: Current iteration number
        """
        time_elapsed = time.time() - self.last_timer
        self.cum_time += time_elapsed

        if self.record:
            self._record_statistics(iteration)

        if self.verbose:
            error = self.solver.variation_measurement()

            msg = f"Iteration {iteration}: variation = {error:.4G}"
            self.logger.info(msg)

            self.last_timer = time.time()

    def monitor_global_step(self, global_iteration=None, loss=None, best_loss=None):
        """
        Monitor a global optimization step (used in multi-start optimization).

        Args:
            global_iteration: Global iteration/sample number
            loss: Current loss value
            best_loss: Best loss found so far
        """
        if self.verbose:
            msg = f"===> Sample {global_iteration}: cost = {loss:.4g} (best cost = {best_loss:.4g})"
            self.logger.info(msg)

    def export_record(self):
        """
        Export recorded statistics as a pandas DataFrame.

        Returns:
            pandas.DataFrame or None: DataFrame containing all recorded statistics, or None if recording was disabled
        """
        if self.record:
            record_data = pd.DataFrame(self.record_list)

            if self.record_sink_iters:
                record_data['sink_time'] = record_data['sink_iter'].cumsum()

            return record_data
        else:
            return None


class SinkhornBasedGW:
    """
    Base class for Sinkhorn-based Gromov-Wasserstein solvers.

    This class provides the generic framework for solving Gromov-Wasserstein problems by alternating between updating
    a cost matrix and solving an optimal transport problem with Sinkhorn iterations.

    In this framework, cost matrices are parametrized by a potential Z. For matrix-based solver, this potential is the
    cost matrix C directly, but for memory-efficient implementations, Z provides a low-dimensional parametrisation of C
    that avoids to store C explicitly.

    Attributes:
        a (torch.Tensor): Source distribution, shape (N,)
        b (torch.Tensor): Target distribution, shape (M,)
        Z (torch.Tensor): Current potential/cost matrix
        Z_new (torch.Tensor): Updated potential/cost matrix
        lazy (bool): Whether to use lazy tensor operations
        eps (float): Entropic regularization parameter
        numItermax (int): Maximum number of outer iterations
        sink_budget (int or None): Total Sinkhorn iteration budget
        sink_adaptive_schedule (bool): Whether to adaptively increase Sinkhorn iterations
        sink_max_numItermax (int or None): Maximum Sinkhorn iterations per step
        stopThr (float or None): Stopping threshold for convergence
        stop_criterion (str): Convergence criterion ('potential' or 'loss')
        initialization_mode: Mode for initializing the potential
        sinkhorn_solver (SinkhornSolver): Inner Sinkhorn solver instance
        loss_val (float): Previous energy value
        new_loss_val (float): Current energy value
    """

    def __init__(self,
                 a: torch.Tensor,
                 b: torch.Tensor,
                 Z: torch.Tensor | None = None,
                 lazy: bool = False,
                 eps: float = 0.1,
                 numItermax: int = 1000,
                 sink_budget: int | None = None,
                 sink_adaptive_schedule: bool = False,
                 sink_max_numItermax: int = None,
                 stopThr: float | None = None,
                 stop_criterion: str = 'potential',
                 initialization_mode=None,
                 SINK_ARGS: dict | None = None,
                  sink_progressive_start: int = 0
                 ):
        """
        Initialize the Gromov-Wasserstein solver.

        Args:
            a: Source distribution (must sum to 1), shape (N,)
            b: Target distribution (must sum to 1), shape (M,)
            Z: Initial potential matrix. If None, initialized internally
            lazy: If True, use lazy tensor implementation
            eps: Entropic regularization parameter
            numItermax: Maximum number of outer GW iterations
            sink_budget: Total budget for Sinkhorn iterations across all GW iterations
            sink_adaptive_schedule: If True, double Sinkhorn iterations when loss increases
            sink_max_numItermax: Cap for Sinkhorn iterations in adaptive schedule
            stopThr: Convergence threshold (meaning depends on stop_criterion)
            stop_criterion: 'potential' for potential change, 'loss' for loss change
            initialization_mode: Mode for potential initialization (subclass-specific)
            sink_progressive_start: If > 0, start Sinkhorn with this many iterations and linearly increase to numItermax over outer iterations (coarse-to-fine). Saves early-iteration cost.
            SINK_ARGS: Additional arguments passed to SinkhornSolver
        """
        self.a = a
        self.b = b

        self.initialization_mode = initialization_mode
        self.Z = Z if Z is not None else self.initialize_potential()
        self.Z_new = self.Z

        self.lazy = lazy
        self.eps = eps

        self.numItermax = numItermax
        self.sink_budget = sink_budget

        self.sink_adaptive_schedule = sink_adaptive_schedule
        self.sink_max_numItermax = sink_max_numItermax

        self.stopThr = stopThr
        self.stop_criterion = stop_criterion

        if SINK_ARGS is None:
            SINK_ARGS = {}
        self.sink_progressive_start = sink_progressive_start
        self._initial_sink_numItermax = SINK_ARGS.get("numItermax", 1000) if SINK_ARGS else 1000
        self.sinkhorn_solver = SinkhornSolver(a=a, b=b, lazy=lazy, eps=eps, **SINK_ARGS)

        self.loss_val, self.new_loss_val = float('inf'), float('inf')

    def parameters(self):
        """
        Get solver parameters as a dictionary.

        Returns:
            dict: Dictionary containing all solver configuration parameters
        """
        params = {'solver': self.__class__.__name__,
                  'eps': self.eps,
                  'numItermax': self.numItermax,
                  'sink_budget': self.sink_budget,
                  'stopThr': self.stopThr,
                  'stop_criterion': self.stop_criterion,
                  'sink_adaptive_schedule': self.sink_adaptive_schedule,
                  'sink_max_numItermax': self.sink_max_numItermax
                  }
        sink_params = self.sinkhorn_solver.parameters()
        for k in sink_params:
            if k != 'eps':
                params['sink_' + k] = sink_params[k]

        return params

    def to(self, device):
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., 'cuda', 'cpu')
        """
        self.sinkhorn_solver.to(device)
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.Z = self.Z.to(device)
        self.Z_new = self.Z_new.to(device)

    def clear(self):
        """Reset the solver to initial state (reinitialize potential and Sinkhorn)."""
        self.Z = self.initialize_potential()
        self.Z_new = self.Z

        self.loss_val, self.new_loss_val = float('inf'), float('inf')

        self.sinkhorn_solver.clear()

    def initialize_potential(self):
        """
        Initialize the potential matrix. Must be implemented by subclasses.

        Returns:
            torch.Tensor: Initial potential matrix Z
        """
        pass

    def update_potential(self):
        """
        Update the potential matrix based on current transport plan. Must be implemented by subclasses.

        Returns:
            torch.Tensor: Updated potential matrix
        """
        pass

    def cost_matrix(self, lazy=None):
        """
        Compute the cost matrix for the Sinkhorn subproblem. Must be implemented by subclasses.

        Args:
            lazy: Whether to return lazy tensor. If None, uses self.lazy

        Returns:
            torch.Tensor or LazyTensor: Cost matrix for optimal transport
        """
        pass

    def solver_step(self, logger=None):
        """
        Perform one step of the Gromov-Wasserstein algorithm.

        This consists of:
            1. Update Z to Z_new from previous step
            2. Compute cost matrix from current Z
            3. Solve Sinkhorn problem with this cost matrix
            4. Update potential based on the resulting transport plan
        """
        self.Z = self.Z_new

        self.sinkhorn_solver.C = self.cost_matrix()
        self.sinkhorn_solver.solve(logger=logger)
        self.Z_new = self.update_potential()

    def potential_variation(self):
        """
        Compute the variation between current and new potential.

        Returns:
            float: Mean absolute difference between Z and Z_new

        """
        return (self.Z - self.Z_new).abs().mean().item()

    def loss_variation(self):
        """
        Compute the change in loss between last steps.

        Returns:
            float: Absolute difference between previous and current losses
        """
        return abs(self.loss_val - self.new_loss_val)

    def variation_measurement(self, variation_type=None):
        """
        Compute the amount of change between last steps, in terms of the chosen criterion.

        Args:
            variation_type: Which metric to output. If None, uses self.stop_criterion

        Returns:
            float: Variation measurement between current and last steps
        """
        variation_type = self.stop_criterion if variation_type is None else variation_type
        if variation_type == 'potential':
            return self.potential_variation()
        else:
            return self.loss_variation()

    def early_stopping(self):
        """
        Check if convergence criterion is met.

        Returns:
            bool: True if the solver should stop, False otherwise
        """
        return self.variation_measurement() < self.stopThr if self.stopThr is not None else False

    def solve(self, solver_monitor=None, reset_sinkhorn_numItermax=True, **kwargs):
        """
        Solve the Gromov-Wasserstein problem.

        Args:
            solver_monitor: GromovSolverMonitor instance for tracking. If None, creates one
            reset_sinkhorn_numItermax: If True, reset Sinkhorn iterations to initial value after solve
            **kwargs: Additional arguments passed to GromovSolverMonitor if created

        Returns:
            pandas.DataFrame or None: Recorded statistics if monitoring was enabled
        """
        with torch.no_grad():

            if solver_monitor is None:
                solver_monitor = GromovSolverMonitor(self, **kwargs)

            init_sink_numItermax = self.sinkhorn_solver.numItermax
            cum_sinkIters = 0.

            for it in range(self.numItermax):
                ### Progressive Sinkhorn budget: start small, increase toward full ###
                if self.sink_progressive_start > 0:
                    frac = min((it + 1) / self.numItermax, 1.0)
                    target_iters = int(self.sink_progressive_start + (self._initial_sink_numItermax - self.sink_progressive_start) * frac)
                    self.sinkhorn_solver.numItermax = target_iters

                ### Apply one solving step ###
                self.solver_step(logger=solver_monitor.logger)

                ### Compute new loss value, if needed ###
                if self.sink_adaptive_schedule or self.stop_criterion == 'energy':
                    self.new_loss_val = self.loss(approx=True)

                ### Monitor the current solving step ###
                solver_monitor.monitor_step(iteration=it + 1)

                if self.early_stopping() or (self.sink_budget is not None and cum_sinkIters > self.sink_budget):
                    break

                cum_sinkIters += self.sinkhorn_solver.last_numIters

                ### Increase the number of Sinkhorn steps if energy has increased ###
                if self.sink_adaptive_schedule:
                    if self.loss_val < self.new_loss_val:
                        self.sinkhorn_solver.numItermax = 2 * self.sinkhorn_solver.numItermax
                        if self.sink_max_numItermax is not None:
                            self.sinkhorn_solver.numItermax = min(self.sinkhorn_solver.numItermax,
                                                                  self.sink_max_numItermax)

                self.loss_val = self.new_loss_val

            if (self.sink_adaptive_schedule or self.sink_progressive_start > 0) and reset_sinkhorn_numItermax:
                self.sinkhorn_solver.numItermax = init_sink_numItermax

        return solver_monitor.export_record()

    def solve_global(self, numSamples=100, solver_monitor=None, **kwargs):
        """
        Solve with multiple random initializations and keep the best solution.

        This is useful to explore the different local minima of the problem.

        Args:
            numSamples: Number of random initializations to try
            solver_monitor: GromovSolverMonitor instance for tracking. If None, creates one
            **kwargs: Additional arguments passed to solve() for monitoring
        """
        if solver_monitor is None:
            solver_monitor = GromovSolverMonitor(self, **kwargs)

        best_loss, best_Z, best_f, best_g = None, None, None, None
        for it in range(numSamples):
            ### Solve EGW on a new initialization ###
            self.clear()
            self.solve(solver_monitor=solver_monitor)

            ### Replace the best candidate with the current solution if it yields a better loss ###
            loss = self.loss(approx=True)
            if best_loss is None or loss < best_loss:
                best_loss, best_Z = loss, self.Z.clone()
                best_f, best_g = self.sinkhorn_solver.f.clone(), self.sinkhorn_solver.g.clone()

            ### Monitor the current global step ###
            solver_monitor.monitor_global_step(global_iteration=it, loss=loss, best_loss=best_loss)

        ### Set the current parameters to the best one ###
        self.Z, self.Z_new = best_Z, best_Z
        self.sinkhorn_solver.f, self.sinkhorn_solver.g = best_f, best_g
        self.sinkhorn_solver.C = self.cost_matrix()

    def transport_plan(self, lazy=None, device=None):
        """
        Get the optimal transport plan.

        Args:
            lazy: Whether to output a lazy tensor. If None, uses self.lazy
            device: Target device for non-lazy operations

        Returns:
            torch.Tensor or LazyTensor: Optimal transport plan, shape (N, M)
        """
        if lazy is None:
            lazy = self.lazy

        if not lazy:
            P = self.sinkhorn_solver.transport_plan(C=self.cost_matrix(lazy=False), lazy=False, device=device)
        else:
            P = self.sinkhorn_solver.transport_plan(lazy=True, device=device)

        return P

    def loss(self, **kwargs):
        """
        Compute the Gromov-Wasserstein loss. Must be implemented by subclasses.

        Args:
            **kwargs: Implementation-specific arguments

        Returns:
            torch.Tensor: Scalar loss value
        """
        pass
