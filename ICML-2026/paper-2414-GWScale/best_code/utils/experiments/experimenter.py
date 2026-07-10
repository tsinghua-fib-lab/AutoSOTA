"""
Solver Experimentation Framework

Utilities for running and comparing multiple solvers across different parameter configurations.
"""

import itertools
import time

import pandas as pd


class SolverExperimenter:
    """
    Run multiple solvers and aggregate results.
    
    Executes a list of configured solvers and collects their outputs into a single DataFrame.
    
    Attributes:
        solvers (list): List of solver instances to run
        verbose (bool): If True, print progress messages
    """
    
    def __init__(self, solvers, verbose=False):
        """
        Initialize the experimenter.
        
        Args:
            solvers: List of solver instances (already configured)
            verbose: If True, print solver names as they run
        """
        self.solvers = solvers
        self.verbose = verbose

    def compute(self, **kwargs):
    	"""
        Run all solvers and collect results.
        
        Args:
            **kwargs: Arguments passed to each solver's solve() method
        
        Returns:
            pandas.DataFrame: Concatenated results from all solvers
        """
        results_data = []
        for solver in self.solvers:
            if self.verbose:
                print(f"\t Computing {solver}...")

            results_data.append(solver.solve(**kwargs))

        return pd.concat(results_data)


class ParametersExperimenter:
    """
    Run experiments across multiple parameter configurations for multiple solvers.
    
    Creates solver instances with all combinations of specified parameter values
    and runs experiments systematically. Useful for hyperparameter tuning and
    comparing solver variants.
    
    Attributes:
        solvers_classes (list): Solver class constructors
        solvers_args (list): Base arguments for each solver class
        sink_variable_names (list): Names of Sinkhorn-specific parameters to vary
        sink_variable_lists (list): Values for each Sinkhorn parameter
        variable_names (list): Names of main parameters to vary
        variable_lists (list): Values for each main parameter
        verbose (bool): If True, print progress
    """
    def __init__(self, solvers_generators, experimented_variables, verbose=False):
    	"""
        Initialize parameter grid experimenter.
        
        Args:
            solvers_generators: List of tuples (SolverClass, base_args_dict)
            experimented_variables: Dictionary of parameter names to lists of values.
                Special key 'SINK_ARGS' contains sub-dictionary for Sinkhorn parameters
            verbose: If True, print experiment progress
        
        Example:
            >>> exp = ParametersExperimenter(
            ...     solvers_generators=[
            ...         (QuadraticGW, {'X': X, 'b': X}),
            ...         (EntropicGW, {'Dx': Dx, 'Dy': Dy})
            ...     ],
            ...     experimented_variables={
            ...         'eps': [0.01, 0.1, 1.0],
            ...         'SINK_ARGS': {'numItermax': [100, 500, 1000]}
            ...     }
            ... )
        """
        experimented_variables = experimented_variables.copy()

        self.solvers_classes = [c[0] for c in solvers_generators]
        self.solvers_args = [c[1].copy() for c in solvers_generators]

        for v in self.solvers_args:
            if 'SINK_ARGS' not in v:
                v['SINK_ARGS'] = {}

        self.sink_variable_names, self.sink_variable_lists = [], []
        if 'SINK_ARGS' in experimented_variables:
            self.sink_variable_names, self.sink_variable_lists = zip(*experimented_variables.pop('SINK_ARGS').items())

        self.variable_names, self.variable_lists = [], []
        if experimented_variables:
            self.variable_names, self.variable_lists = zip(*experimented_variables.items())

        self.verbose = verbose

    def create_solvers(self, variable_values, sink_variables_values):
    	"""
        Create solver instances with specified parameter values.
        
        Args:
            variable_values: Tuple of values for main parameters
            sink_variables_values: Tuple of values for Sinkhorn parameters
        
        Returns:
            list: Configured solver instances
        """
        variables = {k: v for k, v in zip(self.variable_names, variable_values)}

        solvers = []
        for solver_class, solver_args in zip(self.solvers_classes, self.solvers_args):
            current_args = {**solver_args, **variables}

            for i, k in enumerate(self.sink_variable_names):
                current_args['SINK_ARGS'][k] = sink_variables_values[i]

            solvers.append(solver_class(**current_args))

        return solvers

    def compute(self, **kwargs):
	"""
	Run experiments defined at initialization.
	
	Args:
            **kwargs: Arguments passed to each solver's solve() method
        
        Returns:
            pandas.DataFrame: Concatenated results from all solvers and all experiments
            
	"""
        results_data = []

        experiment_id = 0
        start_time = time.time()

        for current_values in itertools.product(*self.variable_lists):
            for current_sink_values in itertools.product(*self.sink_variable_lists):
                if self.verbose:
                    print(
                        f"===== EXPERIMENT {experiment_id}, \t elapsed time = {time.time() - start_time} s \t ({current_values}, "
                        f"{current_sink_values}) =====")

                solvers = self.create_solvers(current_values, current_sink_values)

                results = SolverExperimenter(solvers=solvers, verbose=self.verbose).compute(**kwargs)
                results['experiment_id'] = experiment_id

                results_data.append(results)
                experiment_id += 1

        return pd.concat(results_data)
