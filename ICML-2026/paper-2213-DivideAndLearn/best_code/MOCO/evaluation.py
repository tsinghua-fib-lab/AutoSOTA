import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import json
from datetime import datetime
import os
from scipy import stats

class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(EnhancedJSONEncoder, self).default(obj)

@dataclass
class RunResult:
    """Store results of a single algorithm run"""
    runtime: float
    hypervolume: float
    num_nondominated: int
    objectives: List[List[float]]
    spread: float = 0.0  # Added spread metric


@dataclass
class AlgorithmResult:
    """Store results of an algorithm evaluation including individual runs"""
    def __init__(self, 
                 algorithm_name: str,
                 problem_name: str,
                 runtime: float,  # Average runtime
                 hypervolume: float,  # Final hypervolume
                 num_nondominated: int,  # Final number of non-dominated solutions
                 objectives: List[List[float]],  # Final combined objectives
                 parameters: Dict[str, Any],
                 timestamp: str,
                 problem_size: Dict[str, int],
                 individual_runs: List[RunResult] = None,  # Individual runs
                 statistics: Dict[str, Dict[str, float]] = None):  # Added statistics
        self.algorithm_name = algorithm_name
        self.problem_name = problem_name
        self.runtime = runtime
        self.hypervolume = hypervolume
        self.num_nondominated = num_nondominated
        self.objectives = objectives
        self.parameters = parameters
        self.timestamp = timestamp
        self.problem_size = problem_size
        self.individual_runs = individual_runs or []
        self.statistics = statistics or {}

    def to_dict(self) -> Dict:
        """Convert result to dictionary for JSON serialization"""
        base_dict = {
            'algorithm_name': self.algorithm_name,
            'problem_name': self.problem_name,
            'avg_runtime': float(self.runtime),
            'final_hypervolume': float(self.hypervolume),
            'final_num_nondominated': int(self.num_nondominated),
            'final_objectives': [[float(v) for v in obj] for obj in self.objectives],
            'parameters': self._convert_to_serializable(self.parameters),
            'timestamp': self.timestamp,
            'problem_size': {k: int(v) for k, v in self.problem_size.items()}
        }
        
        # Add individual run data
        if self.individual_runs:
            base_dict['individual_runs'] = [
                {
                    'runtime': float(run.runtime),
                    'hypervolume': float(run.hypervolume),
                    'num_nondominated': int(run.num_nondominated),
                    'objectives': [[float(v) for v in obj] for obj in run.objectives],
                    'spread': float(run.spread) if hasattr(run, 'spread') else 0.0
                }
                for run in self.individual_runs
            ]
        
        # Add statistics data
        if self.statistics:
            base_dict['statistics'] = self._convert_to_serializable(self.statistics)
        
        return base_dict

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(v) for v in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @classmethod
    def from_dict(cls, data: Dict) -> 'AlgorithmResult':
        """Create AlgorithmResult from dictionary"""
        # Extract individual runs if available
        individual_runs = None
        if 'individual_runs' in data:
            individual_runs = [
                RunResult(
                    runtime=run['runtime'],
                    hypervolume=run['hypervolume'],
                    num_nondominated=run['num_nondominated'],
                    objectives=run['objectives'],
                    spread=run.get('spread', 0.0)
                )
                for run in data['individual_runs']
            ]
            del data['individual_runs']  # Remove to avoid duplicate argument
        
        # Extract statistics if available
        statistics = data.pop('statistics', None)
        
        # Create AlgorithmResult with rest of data
        result = cls(**data)
        
        # Set individual_runs and statistics if available
        if individual_runs:
            result.individual_runs = individual_runs
        if statistics:
            result.statistics = statistics
        
        return result


class MOCOEvaluator:
    """Extended MOCOEvaluator with statistical measures and reliability scores"""
    
    def __init__(self, reference_point: Tuple[float, ...], confidence_level: float = 0.95, results_dir: str = "results_MOCO_April"):
        self.reference_point = reference_point
        self.confidence_level = confidence_level
        self.results: List[AlgorithmResult] = []
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def calculate_statistics(self, metric_values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a metric across multiple runs"""
        values = np.array(metric_values)
        n = len(values)
        
        if n == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'cv': 0.0,
                'min': 0.0,
                'max': 0.0,
                'range': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'confidence_level': self.confidence_level,
                'iqr': 0.0,
                'q1': 0.0,
                'q3': 0.0,
                'n_samples': 0
            }
        
        # Basic statistics
        mean = np.mean(values)
        median = np.median(values)
        std_dev = np.std(values, ddof=1) if n > 1 else 0.0  # Sample standard deviation
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Coefficient of variation (relative standard deviation)
        cv = std_dev / mean if mean != 0 else float('inf')
        
        # Confidence interval
        if n > 1:
            t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n-1)
            margin_of_error = t_critical * (std_dev / np.sqrt(n))
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
        else:
            ci_lower = mean
            ci_upper = mean
        
        # Interquartile range
        if n > 1:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
        else:
            q1 = values[0]
            q3 = values[0]
            iqr = 0.0
        
        return {
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'cv': cv,  # Coefficient of variation
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': self.confidence_level,
            'iqr': iqr,
            'q1': q1,
            'q3': q3,
            'n_samples': n
        }
    
    def evaluate_algorithm(self, 
                     algorithm_class: Any,
                     problem_class: Any,
                     algorithm_name: str,
                     parameters: Dict[str, Any],
                     problem_params: Dict[str, Any],
                     num_runs: int = 1) -> AlgorithmResult:
        """Evaluate algorithm on multiple random problem instances"""
        # Determine problem type and size
        problem_type = None
        problem_size = None
        
        if problem_class.__name__ == 'BiObjectiveTSP':
            problem_type = 'BiTSP'
            problem_size = problem_params.get('n_cities', 0)
        elif problem_class.__name__ == 'MultiObjectiveKnapsack' and problem_params.get('n_objectives', 0) == 2:
            problem_type = 'BiKP'
            problem_size = problem_params.get('n_items', 0)
        elif problem_class.__name__ == 'TriObjectiveTSP' or problem_params.get('n_objectives', 0) == 3:
            problem_type = 'TriTSP'
            problem_size = problem_params.get('n_cities', 0)
        elif problem_class.__name__ == 'BiObjectiveCVRP':  # No need to check n_objectives
            problem_type = 'BiCVRP'
            problem_size = problem_params.get('n_customers', 0)  # Fixed typo: n_customerrs -> n_customers
        
        print(f"DEBUG ##### in evaluate_algorithm: problem_type={problem_type}, problem_size={problem_size}")
        print(f"DEBUG: ##### problem_params={problem_params}")

        total_runtime = 0
        all_solutions = []
        individual_runs = []
        
        # Lists to store metrics from each run
        hypervolumes = []
        runtimes = []
        solution_counts = []
        spread_metrics = []
        
        # Run algorithm multiple times with different problem instances
        for run in range(num_runs):
            # Create new problem instance with different seed
            seed = run + 1
            np.random.seed(seed)
            
            # Check if the problem class accepts a seed parameter
            from inspect import signature
            problem_sig_params = signature(problem_class.__init__).parameters
            run_problem_params = problem_params.copy()
            
            # Only add seed parameter if the problem class accepts it
            if 'seed' in problem_sig_params:
                run_problem_params['seed'] = seed if 'seed' not in run_problem_params else run_problem_params['seed']
            
            problem = problem_class(**run_problem_params)
            
            # Create new algorithm instance
            algorithm = algorithm_class(problem, **parameters)
            
            start_time = time.time()
            solutions = algorithm.run()
            end_time = time.time()
            
            run_time = end_time - start_time
            total_runtime += run_time
            
            # Get non-dominated solutions for this run
            nondom_solutions = self._get_nondominated(solutions)
            run_objectives = [[float(v) for v in sol[1]] for sol in nondom_solutions]
            
            # Calculate metrics for this run
            # Use the enhanced _calculate_hypervolume_metrics with problem type and size
            run_metrics = self._calculate_hypervolume_metrics(nondom_solutions, problem_type, problem_size)
            run_hv = run_metrics['hv_ratio']  # Use the normalized hypervolume ratio
            run_spread = self._calculate_spread(nondom_solutions)
            
            # Store metrics for statistical analysis
            hypervolumes.append(run_hv)
            runtimes.append(run_time)
            solution_counts.append(len(nondom_solutions))
            spread_metrics.append(run_spread)
            
            # Store individual run results
            individual_runs.append(RunResult(
                runtime=run_time,
                hypervolume=run_hv,
                num_nondominated=len(nondom_solutions),
                objectives=run_objectives,
                spread=run_spread
            ))
            
            # Add solutions to overall set
            all_solutions.extend(solutions)
        
        # Get final non-dominated set from all runs
        nondom_solutions = self._get_nondominated(all_solutions)
        
        # Calculate final metrics using enhanced hypervolume calculation
        final_metrics = self._calculate_hypervolume_metrics(nondom_solutions, problem_type, problem_size)
        
        # Calculate final metrics
        avg_runtime = total_runtime / num_runs
        final_hv = final_metrics['hv_ratio']  # Use the normalized hypervolume ratio
        final_num_nondom = len(nondom_solutions)
        
        # Convert objectives to list of floats
        final_objectives = [[float(v) for v in sol[1]] for sol in nondom_solutions]
        
        # Calculate statistics for each metric
        hv_stats = self.calculate_statistics(hypervolumes)
        runtime_stats = self.calculate_statistics(runtimes)
        solution_count_stats = self.calculate_statistics(solution_counts)
        spread_stats = self.calculate_statistics(spread_metrics)
        
        # Combine all statistics
        statistics = {
            'hypervolume': hv_stats,
            'runtime': runtime_stats,
            'num_nondominated': solution_count_stats,
            'spread': spread_stats
        }
        
        # Calculate reliability scores
        reliability_scores = self._calculate_reliability_scores(statistics)
        statistics['reliability_scores'] = reliability_scores
        
        result = AlgorithmResult(
            algorithm_name=algorithm_name,
            problem_name=problem_class.__name__,
            runtime=avg_runtime,
            hypervolume=final_hv,
            num_nondominated=final_num_nondom,
            objectives=final_objectives,
            parameters={"algorithm": parameters, "problem": problem_params},
            timestamp=datetime.now().isoformat(),
            problem_size=self._get_problem_size(problem),
            individual_runs=individual_runs,
            statistics=statistics
        )
        
        #  MINIMAL FIX: Add convergence stats if they exist
        if hasattr(algorithm, 'final_convergence_stats'):
            result.convergence_stats = algorithm.final_convergence_stats

        self.results.append(result)
        self._save_result(result)
        # After calculating final metrics
        print(f"\nDEBUG - Final metrics for {algorithm_name} on {problem_class.__name__}:")
        print(f"  Using hv_ratio: {final_hv:.6f} for hypervolume field")
        return result

    def _calculate_reliability_scores(self, statistics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate reliability scores based on coefficient of variation"""
        reliability_scores = {}
        
        for metric, stats in statistics.items():
            if metric != 'reliability_scores':  # Skip if it's already a reliability score
                cv = stats.get('cv', float('inf'))
                # Transform CV to a 0-100 reliability score (inversely related)
                # CV < 0.05 (5%) is excellent, CV > 0.3 (30%) is poor
                reliability = max(0, min(100, 100 * (1 - cv / 0.3)))
                reliability_scores[metric] = reliability
        
        return reliability_scores
    
    def _get_problem_size(self, problem: Any) -> Dict[str, int]:
        """Extract problem size information"""
        size_info = {}
        if hasattr(problem, 'n_cities'):
            size_info['n_cities'] = int(problem.n_cities)
        if hasattr(problem, 'n_items'):
            size_info['n_items'] = int(problem.n_items)
        if hasattr(problem, 'n_objectives'):
            size_info['n_objectives'] = int(problem.n_objectives)
        if hasattr(problem, 'capacity'):
            size_info['capacity'] = float(problem.capacity)
        if hasattr(problem, 'n_customers'):
            size_info['n_customers'] = int(problem.n_customers)
        if hasattr(problem, 'n_vehicles'):
            size_info['n_vehicles'] = int(problem.n_vehicles)
        return size_info
    
    def _save_result(self, result: AlgorithmResult):
        """Save single result to JSON file"""
        filename = f"{result.algorithm_name}_{result.problem_name}_{result.timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, cls=EnhancedJSONEncoder)
    
    def save_all_results(self, filename: str):
        """Save all results to a single JSON file"""
        filepath = os.path.join(self.results_dir, filename)
        results_data = [result.to_dict() for result in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def load_results(self, filename: str):
        """Load results from JSON file"""
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            self.results = [AlgorithmResult.from_dict(result_data) 
                          for result_data in data]
        else:
            self.results = [AlgorithmResult.from_dict(data)]
    
    def generate_report(self, filename: str = "report.json") -> str:
        """Generate detailed report including individual runs and statistics"""
        report_dict = {
            'algorithms': {}
        }
        
        report_text = "Multi-Objective Optimization Results\n"
        report_text += "===================================\n\n"
        
        for result in self.results:
            algo_report = {
                'avg_runtime': float(result.runtime),
                'final_hypervolume': float(result.hypervolume),
                'final_num_nondominated': int(result.num_nondominated),
                'problem_name': str(result.problem_name),
                'timestamp': str(result.timestamp),
                'individual_runs': [],
                'statistics': result.statistics
            }
            
            # Add to text report
            report_text += f"Algorithm: {result.algorithm_name}\n"
            report_text += f"Problem: {result.problem_name}\n"
            report_text += f"Parameters: {result.parameters}\n\n"
            
            # Add statistics to text report if available
            if result.statistics:
                # Hypervolume statistics
                if 'hypervolume' in result.statistics:
                    hv = result.statistics['hypervolume']
                    report_text += "Hypervolume Statistics:\n"
                    report_text += f"  Mean: {hv['mean']:.4f}\n"
                    report_text += f"  Standard Deviation: {hv['std_dev']:.4f} (CV: {hv['cv']:.2%})\n"
                    report_text += f"  {hv['confidence_level']*100:.0f}% Confidence Interval: [{hv['ci_lower']:.4f}, {hv['ci_upper']:.4f}]\n"
                    report_text += f"  Range: [{hv['min']:.4f}, {hv['max']:.4f}]\n"
                    report_text += f"  Median: {hv['median']:.4f} (IQR: {hv['iqr']:.4f})\n\n"
                
                # Runtime statistics
                if 'runtime' in result.statistics:
                    rt = result.statistics['runtime']
                    report_text += "Runtime Statistics (seconds):\n"
                    report_text += f"  Mean: {rt['mean']:.2f}\n"
                    report_text += f"  Standard Deviation: {rt['std_dev']:.2f} (CV: {rt['cv']:.2%})\n"
                    report_text += f"  Range: [{rt['min']:.2f}, {rt['max']:.2f}]\n\n"
                
                # Solution count statistics
                if 'num_nondominated' in result.statistics:
                    sc = result.statistics['num_nondominated']
                    report_text += "Non-dominated Solution Count Statistics:\n"
                    report_text += f"  Mean: {sc['mean']:.1f}\n"
                    report_text += f"  Standard Deviation: {sc['std_dev']:.1f} (CV: {sc['cv']:.2%})\n"
                    report_text += f"  Range: [{sc['min']:.0f}, {sc['max']:.0f}]\n\n"
                
                # Spread statistics
                if 'spread' in result.statistics:
                    sp = result.statistics['spread']
                    report_text += "Solution Spread Statistics:\n"
                    report_text += f"  Mean: {sp['mean']:.4f}\n"
                    report_text += f"  Standard Deviation: {sp['std_dev']:.4f} (CV: {sp['cv']:.2%})\n"
                    report_text += f"  Range: [{sp['min']:.4f}, {sp['max']:.4f}]\n\n"
                
                # Reliability scores
                if 'reliability_scores' in result.statistics:
                    report_text += "Reliability Scores (0-100, higher is better):\n"
                    rs = result.statistics['reliability_scores']
                    for metric, score in rs.items():
                        reliability_category = "Excellent" if score > 90 else "Good" if score > 75 else "Fair" if score > 50 else "Poor"
                        report_text += f"  {metric}: {score:.1f}/100 ({reliability_category})\n"
                    report_text += "\n"
            
            # Add individual run data
            if result.individual_runs:
                for i, run in enumerate(result.individual_runs, 1):
                    algo_report['individual_runs'].append({
                        'run_number': i,
                        'runtime': float(run.runtime),
                        'hypervolume': float(run.hypervolume),
                        'num_nondominated': int(run.num_nondominated),
                        'spread': float(run.spread) if hasattr(run, 'spread') else 0.0
                    })
            
            report_dict['algorithms'][result.algorithm_name] = algo_report
            report_text += "-----------------------------------\n\n"
        
        # Save to JSON file
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, cls=EnhancedJSONEncoder)
        
        # Save text report
        text_filepath = os.path.join(self.results_dir, "report.txt")
        with open(text_filepath, 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def plot_performance_profiles(self, metric: str = 'hypervolume', save_path: str = None):
        """Plot performance profiles with error bars for comparing algorithms"""
        if not self.results or len(self.results) < 1:
            print("No results to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        
        algorithms = []
        values = []
        errors = []
        
        for result in self.results:
            if result.statistics and metric in result.statistics:
                algorithms.append(result.algorithm_name)
                stats = result.statistics[metric]
                values.append(stats['mean'])
                errors.append(stats['std_dev'])
        
        if not algorithms:
            print(f"No statistical data available for metric: {metric}")
            return
        
        x = np.arange(len(algorithms))
        plt.bar(x, values, yerr=errors, align='center', alpha=0.7, ecolor='black', capsize=10)
        plt.xticks(x, algorithms)
        plt.ylabel(f'{metric.capitalize()}')
        plt.title(f'Performance Comparison ({metric})')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + errors[i], f"{v:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            
            plt.show()

    def generate_reliability_scores(self) -> Dict[str, Dict[str, float]]:
        """Generate reliability scores for each metric across all algorithms"""
        reliability_scores = {}
        
        for result in self.results:
            algorithm_name = result.algorithm_name
            problem_name = result.problem_name
            key = f"{algorithm_name}_{problem_name}"
            
            reliability_scores[key] = {}
            
            # For each metric, extract reliability score if available
            if result.statistics and 'reliability_scores' in result.statistics:
                reliability_scores[key] = result.statistics['reliability_scores']
            else:
                # Calculate on the fly if not available
                for metric in ['hypervolume', 'runtime', 'num_nondominated', 'spread']:
                    if result.individual_runs and len(result.individual_runs) > 1:
                        values = [getattr(run, metric) for run in result.individual_runs]
                        stats = self.calculate_statistics(values)
                        cv = stats['cv']
                        reliability = max(0, min(100, 100 * (1 - cv / 0.3)))
                        reliability_scores[key][metric] = reliability
                    else:
                        reliability_scores[key][metric] = 0.0  # Not enough runs for reliability
        
        return reliability_scores
    
    def _get_nondominated(self, solutions: List[Tuple[Any, Tuple[float, ...]]]) -> List[Tuple[Any, Tuple[float, ...]]]:
        """Extract non-dominated solutions"""
        nondominated = []
        for sol1 in solutions:
            dominated = False
            for sol2 in solutions:
                if sol1 != sol2 and self._dominates(sol2[1], sol1[1]):
                    dominated = True
                    break
            if not dominated:
                nondominated.append(sol1)
        return nondominated
    
    def _dominates(self, obj1: Tuple[float, ...], obj2: Tuple[float, ...]) -> bool:
        """Check if obj1 dominates obj2"""
        better_in_one = False
        for v1, v2 in zip(obj1, obj2):
            if v1 > v2:  # Assuming minimization
                return False
            if v1 < v2:
                better_in_one = True
        return better_in_one
    
    def _calculate_hypervolume(self, solutions: List[Tuple[Any, Tuple[float, ...]]]) -> float:
        """
        Calculate hypervolume indicator with support for >2 objectives.
        This is the main method to be used by the evaluate_algorithm function.
        
        Args:
            solutions: List of tuples (solution, objectives)
            
        Returns:
            Hypervolume ratio (normalized hypervolume)
        """
        # Get hypervolume metrics
        metrics = self._calculate_hypervolume_metrics(solutions)
        
        # Return the hypervolume ratio as the preferred metric
        return metrics["hv_ratio"]
    
    def _is_maximization_problem(self, problem_type: str) -> bool:
        """Determine if a problem type is maximization"""
        # Knapsack problems are maximization
        if 'BiKP' in problem_type or 'Knapsack' in problem_type:
            return True
        # TSP and VRP problems are minimization
        return False

    def _calculate_hypervolume_metrics(self, solutions: List[Tuple[Any, Tuple[float, ...]]], problem_type=None, problem_size=None) -> Dict[str, float]:
        """
        Calculate multiple hypervolume metrics using standard reference and ideal points if available.
        
        Args:
            solutions: List of tuples (solution, objectives)
            problem_type: Type of problem (BiTSP, BiKP, etc.)
            problem_size: Size of the problem
            
        Returns:
            Dictionary containing different hypervolume metrics
        """
        # Get standard points if problem type and size are provided
        standard_points = None
        if problem_type and problem_size:
            standard_points = self.get_standard_points(problem_type, problem_size)
        
        if not solutions:
            return {
                "raw_hypervolume": 0.0, 
                "original_normalized_hv": 0.0, 
                "hv_ratio": 0.0,
                "ideal_point": [],
                "nadir_point": [],
                "reference_point": self.reference_point if hasattr(self, 'reference_point') else [],
                "dimensions": 0
            }
        
        # Extract objective vectors
        points = np.array([list(sol[1]) for sol in solutions])
        
        # If no points or empty objectives, return 0
        if len(points) == 0 or len(points[0]) == 0:
            return {
                "raw_hypervolume": 0.0, 
                "original_normalized_hv": 0.0, 
                "hv_ratio": 0.0,
                "ideal_point": [],
                "nadir_point": [],
                "reference_point": self.reference_point if hasattr(self, 'reference_point') else [],
                "dimensions": 0
            }
        # Get number of objectives (dimensions)
        n_objectives = len(points[0])
        
        # # Get ideal point (minimum in each dimension)
        # ideal_point = np.min(points, axis=0)
        
        # # Get nadir point (maximum in each dimension)
        nadir_point = np.max(points, axis=0)
        
        # # Use reference point if provided, otherwise create one
        # if hasattr(self, 'reference_point') and self.reference_point is not None:
        #     reference_point = np.array(self.reference_point)
            
        #     # Ensure reference point has correct dimensions
        #     if len(reference_point) != n_objectives:
        #         print(f"Warning: Reference point has {len(reference_point)} dimensions, but problem has {n_objectives} objectives")
        #         # Create a new reference point with correct dimensions
        #         reference_point = nadir_point * 1.1  # 10% larger than nadir
        # else:
        #     # Create reference point that dominates all points with a margin
        #     reference_point = nadir_point * 1.1  # 10% larger than nadir
        # Use standard ideal point if available, otherwise compute it
        if standard_points and standard_points['ideal'] is not None:
            ideal_point = np.array(standard_points['ideal'])
        else:
            ideal_point = np.zeros_like(self.reference_point)
        
        # Use standard reference point if available, otherwise use provided or compute it
        if standard_points and standard_points['reference'] is not None:
            reference_point = np.array(standard_points['reference'])
        elif hasattr(self, 'reference_point') and self.reference_point is not None:
            reference_point = np.array(self.reference_point)
        else:
            reference_point = nadir_point * 1.1  # 10% larger than nadir


        print(f"DEBUG ##### in _calculate_hypervolume_metrics: problem_type={problem_type}, problem_size={problem_size}")

        # Calculate raw hypervolume
        raw_hv = self._compute_raw_hypervolume(points, reference_point)
        
        # Calculate original normalized hypervolume (for backward compatibility)
        original_hv = self._original_calculate_hypervolume(solutions)
        
        # Calculate hypervolume ratio
        # HV'ᵣ(F) = HVᵣ(F) / ∏ᴹᵢ₌₁|rᵢ - zᵢ|
        volume_normalization = np.prod(reference_point - ideal_point)
        
        if volume_normalization <= 0:
            hv_ratio = 0.0  # Avoid division by zero
        else:
            hv_ratio = raw_hv / volume_normalization
        
        # Add debug statements
        n_dimensions = len(points[0]) if len(points) > 0 else 0
        print(f"\nDEBUG - Hypervolume calculation for {problem_type} (dim={n_dimensions}):")
        print(f"  Reference point: {reference_point}")
        print(f"  Ideal point: {ideal_point}")
        print(f"  Raw hypervolume: {raw_hv:.6f}")
        print(f"  Original (2D) normalized HV: {original_hv:.6f}")
        print(f"  Normalization factor: {volume_normalization:.6f}")
        print(f"  Final HV ratio: {hv_ratio:.6f}")


        return {
            "raw_hypervolume": float(raw_hv),
            "original_normalized_hv": float(original_hv),
            "hv_ratio": float(hv_ratio),
            "ideal_point": ideal_point.tolist(),
            "nadir_point": nadir_point.tolist(),
            "reference_point": reference_point.tolist(),
            "dimensions": n_objectives
        }
    
    def _compute_raw_hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Compute raw hypervolume value based on the number of dimensions.
        This method handles the accurate calculation for different dimensionalities.
        
        Args:
            points: Array of objective vectors
            reference_point: Reference point for hypervolume calculation
        
        Returns:
            Raw hypervolume value
        """
        # Get number of dimensions
        n_objectives = points.shape[1]
        
        # First, identify non-dominated points
        non_dominated_mask = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        non_dominated_mask[i] = False
                        break
        
        non_dominated_points = points[non_dominated_mask]
        
        # If no non-dominated points (shouldn't happen), return 0
        if len(non_dominated_points) == 0:
            return 0.0
        
        # Calculate hypervolume based on number of dimensions
        if n_objectives == 1:
            # For 1D, hypervolume is just the difference between reference and best point
            return reference_point[0] - np.min(non_dominated_points[:, 0])
        
        elif n_objectives == 2:
            # # For 2D, use the corrected exclusive hypervolume calculation
            # return self._compute_2d_hypervolume(non_dominated_points, reference_point)

            # AUTO-DETECT maximization vs minimization
            is_maximization = np.all(points >= reference_point - 1e-6)
            
            if is_maximization:
                # Use your algorithm's method for maximization
                return self._calculate_2d_hypervolume_maximization(points, reference_point)
            else:
                # Use existing method for minimization
                return self._compute_2d_hypervolume(non_dominated_points, reference_point)
            
        else:
            # For higher dimensions, use a library if available, otherwise use recursive method
            try:
                # Try to import and use pygmo (preferred) or pymoo
                import pygmo as pg
                hv_obj = pg.hypervolume(non_dominated_points)
                return hv_obj.compute(reference_point)
            except ImportError:
                try:
                    from pymoo.indicators.hv import HV
                    hv_calculator = HV(ref_point=reference_point)
                    return hv_calculator.do(non_dominated_points)
                except ImportError:
                    # Fallback to recursive method (may be slow for high dimensions)
                    return self._recursive_hypervolume(non_dominated_points, reference_point, 0)
    
    def _calculate_2d_hypervolume_maximization(self, points, ref_point):
        """2D hypervolume for maximization (copied from your algorithm)"""
        sorted_indices = np.argsort(-points[:, 0])
        sorted_points = points[sorted_indices]
        
        hv = 0.0
        for i in range(len(sorted_points)):
            curr_point = sorted_points[i]
            
            if i < len(sorted_points) - 1:
                next_x = sorted_points[i + 1][0]
            else:
                next_x = ref_point[0]
            
            width = curr_point[0] - next_x
            height = curr_point[1] - ref_point[1]
            
            if width > 0 and height > 0:
                hv += width * height
        
        return hv


    def _compute_2d_hypervolume(self, points: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Compute 2D hypervolume accurately using the lebesgue measure (area calculation).
        
        Args:
            points: Array of non-dominated points
            reference_point: Reference point
        
        Returns:
            Hypervolume (area)
        """
        # Sort points by first objective (ascending)
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        # Initialize hypervolume
        hypervolume = 0.0
        
        # Add contribution of each point
        prev_x = reference_point[0]
        prev_y = reference_point[1]
        
        # Process points from right to left
        for i in range(len(sorted_points)-1, -1, -1):
            point = sorted_points[i]
            
            # Calculate the area of the rectangle
            width = prev_x - point[0]
            height = reference_point[1] - point[1]
            
            # Add area to hypervolume
            hypervolume += width * height
            
            # Update previous x coordinate
            prev_x = point[0]
            
            # Update previous y coordinate if this point has a higher y
            if point[1] < prev_y:
                prev_y = point[1]
        
        return hypervolume
    
    def _recursive_hypervolume(self, points: np.ndarray, reference_point: np.ndarray, depth: int) -> float:
        """
        Calculate hypervolume recursively for higher dimensions.
        
        Args:
            points: Array of points
            reference_point: Reference point
            depth: Current recursion depth (dimension index)
        
        Returns:
            Hypervolume value
        """
        # Base case: no points
        if len(points) == 0:
            return 0.0
        
        # Base case: 1D - return difference between reference and best point
        if depth == len(reference_point) - 1:
            return reference_point[depth] - np.min(points[:, depth])
        
        # Sort points by current dimension
        sorted_indices = np.argsort(points[:, depth])
        sorted_points = points[sorted_indices]
        
        # Initialize hypervolume
        hypervolume = 0.0
        
        # Process each point
        prev_point = None
        prev_hypervolume = 0.0
        
        for i in range(len(sorted_points)):
            current_point = sorted_points[i]
            
            # Create a new reference point for the recursive call
            next_reference = reference_point.copy()
            next_reference[depth] = current_point[depth]
            
            # Find points to consider in the next recursion
            if i < len(sorted_points) - 1:
                # All points to the right that are not dominated in deeper dimensions
                next_points = sorted_points[i+1:]
                
                # Filter dominated points
                mask = np.ones(len(next_points), dtype=bool)
                for j in range(len(next_points)):
                    dominated = True
                    for d in range(depth+1, len(reference_point)):
                        if next_points[j, d] < current_point[d]:
                            dominated = False
                            break
                    if dominated:
                        mask[j] = False
                
                next_points = next_points[mask]
            else:
                next_points = np.empty((0, len(reference_point)))
            
            # Calculate hypervolume at next level
            next_hypervolume = self._recursive_hypervolume(next_points, next_reference, depth + 1)
            
            # Add contribution to total hypervolume
            if prev_point is not None:
                layer_height = current_point[depth] - prev_point[depth]
                hypervolume += layer_height * prev_hypervolume
            
            # Update for next iteration
            prev_point = current_point
            prev_hypervolume = next_hypervolume
        
        # Add contribution of last slice
        if prev_point is not None:
            layer_height = reference_point[depth] - prev_point[depth]
            hypervolume += layer_height * prev_hypervolume
        
        return hypervolume
    
    def _original_calculate_hypervolume(self, solutions: List[Tuple[Any, Tuple[float, ...]]]) -> float:
        """
        Original hypervolume calculation method for backward compatibility.
        Only works for 2D problems.
        
        Args:
            solutions: List of tuples (solution, objectives)
        
        Returns:
            Normalized hypervolume value using the original algorithm
        """
        if not solutions:
            return 0.0
        
        # Extract objective vectors
        points = np.array([list(sol[1]) for sol in solutions])
        
        # Normalize all points to [0,1] range using min-max normalization
        if len(points) > 0 and len(points[0]) == 2:  # Bi-objective case
            # Apply min-max normalization to each objective
            normalized_points = np.zeros_like(points, dtype=float)
            for i in range(2):
                min_val = np.min(points[:, i])
                max_val = np.max(points[:, i])
                if max_val > min_val:
                    normalized_points[:, i] = (points[:, i] - min_val) / (max_val - min_val)
                else:
                    normalized_points[:, i] = points[:, i] / max_val if max_val != 0 else points[:, i]
            
            # Sort normalized points by first objective
            sorted_points = normalized_points[normalized_points[:, 0].argsort()]
            
            # Calculate hypervolume
            hv = 0.0
            for i in range(len(sorted_points)-1):
                width = sorted_points[i+1][0] - sorted_points[i][0]
                height = sorted_points[i][1]
                hv += width * height
            
            # Add last rectangle
            if len(sorted_points) > 0:
                width = 1.0 - sorted_points[-1][0]
                height = sorted_points[-1][1]
                hv += width * height
            
            return hv
        else:
            return 0.0  # Handle non-2D cases

    def _calculate_spread(self, solutions: List[Tuple[Any, Tuple[float, ...]]]) -> float:
        """Calculate spread metric (diversity) of solutions"""
        if len(solutions) < 2:
            return 0.0
        
        # Extract objective vectors
        obj_vectors = [obj for _, obj in solutions]
        
        # Calculate Euclidean distances between consecutive points
        distances = []
        sorted_vectors = sorted(obj_vectors, key=lambda x: x[0])
        
        for i in range(len(sorted_vectors) - 1):
            p1 = sorted_vectors[i]
            p2 = sorted_vectors[i + 1]
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            distances.append(dist)
        
        # Calculate mean distance
        mean_dist = np.mean(distances) if distances else 0.0
        
        # Calculate spread (standard deviation of distances)
        if len(distances) < 2:
            return 0.0
        
        spread = np.std(distances, ddof=1) / mean_dist if mean_dist > 0 else 0.0
        return spread
    
    def plot_pareto_front(self, result_index: int = -1, show_all: bool = False):
        """Plot Pareto front for bi-objective problems"""
        if not self.results:
            print("No results available.")
            return
        
        if result_index >= len(self.results) or result_index < -len(self.results):
            print(f"Invalid result index: {result_index}. Must be between {-len(self.results)} and {len(self.results)-1}.")
            return
        
        result = self.results[result_index]
        
        if len(result.objectives[0]) != 2:
            print("Plotting only supported for bi-objective problems")
            return
        
        objectives = np.array(result.objectives)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(objectives[:, 0], objectives[:, 1], label=f"Final Pareto Front", s=80, alpha=0.7)
        
        # Plot individual runs if requested
        if show_all and result.individual_runs:
            for i, run in enumerate(result.individual_runs):
                run_objectives = np.array(run.objectives)
                if len(run_objectives) > 0:
                    plt.scatter(run_objectives[:, 0], run_objectives[:, 1], label=f"Run {i+1}", alpha=0.3, s=30)
        
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front - {result.algorithm_name} on {result.problem_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{result.problem_name}_pareto_front.png")
    
    def compare_algorithms(self) -> Dict[str, Dict[str, float]]:
        """Compare metrics across all evaluated algorithms"""
        comparison = {}
        
        for result in self.results:
            comparison[result.algorithm_name] = {
                'Runtime': result.runtime,
                'Hypervolume': result.hypervolume,
                'Num_Nondominated': result.num_nondominated
            }
            
            # Add reliability scores if available
            if result.statistics and 'reliability_scores' in result.statistics:
                for metric, score in result.statistics['reliability_scores'].items():
                    comparison[result.algorithm_name][f"{metric}_reliability"] = score
        
        return comparison
    
    def plot_comparison(self, metrics: List[str] = None):
        """Plot comparison of metrics across algorithms"""
        if not metrics:
            metrics = ['Runtime', 'Hypervolume', 'Num_Nondominated']
        
        algorithms = [r.algorithm_name for r in self.results]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]  # Ensure axes is iterable
        
        for i, metric in enumerate(metrics):
            if metric.lower() in ['runtime', 'hypervolume', 'num_nondominated']:
                values = [getattr(r, metric.lower()) for r in self.results]
                
                # Get error bars if available
                errors = []
                for r in self.results:
                    if r.statistics and metric.lower() in r.statistics:
                        errors.append(r.statistics[metric.lower()]['std_dev'])
                    else:
                        errors.append(0)
                
                axes[i].bar(algorithms, values, yerr=errors, align='center', alpha=0.7, 
                     ecolor='black', capsize=5)
                axes[i].set_title(metric)
                axes[i].set_xticklabels(algorithms, rotation=45)
                axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels on top of bars
                for j, v in enumerate(values):
                    axes[i].text(j, v + (errors[j] if errors[j] > 0 else 0.05*v), 
                              f"{v:.2f}", ha='center', va='bottom')
                
        plt.tight_layout()
        plt.savefig(f"metric_comparison.png")

    def plot_multi_metric_profiles(self, metrics=None, save_path=None):
        """
        Plot performance profiles with multiple metrics side by side for comparing algorithms
        
        Args:
            metrics: List of metrics to plot (default: ['hypervolume', 'runtime', 'num_nondominated'])
            save_path: Path to save the figure (if None, the figure is displayed)
        
        Returns:
            Matplotlib figure object
        """
        if not self.results or len(self.results) < 1:
            print("No results to plot.")
            return None
        
        if metrics is None:
            metrics = ['hypervolume', 'runtime', 'num_nondominated']
        
        metrics_names = {
            'hypervolume': 'Hypervolume', 
            'runtime': 'Runtime (s)', 
            'num_nondominated': 'Non-dominated Solutions',
            'spread': 'Spread'
        }
        
        # Get all algorithm names
        algorithms = [result.algorithm_name for result in self.results]
        
        # Create a figure with one subplot per metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        
        # Make sure axes is always a list even with a single subplot
        if len(metrics) == 1:
            axes = [axes]
        
        # For each metric
        for i, metric in enumerate(metrics):
            values = []
            errors = []
            
            for result in self.results:
                if hasattr(result, 'statistics') and result.statistics and metric in result.statistics:
                    stats = result.statistics[metric]
                    values.append(stats['mean'])
                    errors.append(stats['std_dev'])
                else:
                    # If statistics aren't available, use the single value
                    values.append(getattr(result, metric.lower()) if hasattr(result, metric.lower()) else 0)
                    errors.append(0)
            
            # Get metric name or use metric itself if not in the dictionary
            metric_name = metrics_names.get(metric, metric.capitalize())
            
            # Plot bars
            axes[i].bar(algorithms, values, yerr=errors, align='center', alpha=0.7, 
                    ecolor='black', capsize=10)
            axes[i].set_title(metric_name)
            axes[i].set_ylabel(metric_name)
            
            # Set x-tick labels with rotation
            axes[i].set_xticks(range(len(algorithms)))
            axes[i].set_xticklabels(algorithms, rotation=45, ha='right')
            
            # Add grid
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Invert y-axis for runtime (lower is better)
            if metric == 'runtime' and values:
                max_val = max([v + e for v, e in zip(values, errors)]) * 1.2  # 20% margin
                axes[i].set_ylim(0, max_val)
                
                # Add annotation that lower is better for runtime
                axes[i].annotate('Lower is better', xy=(0.5, 0.95), xycoords='axes fraction',
                            ha='center', va='top', fontsize=10, style='italic')
            else:
                # Add annotation that higher is better for other metrics
                axes[i].annotate('Higher is better', xy=(0.5, 0.95), xycoords='axes fraction',
                            ha='center', va='top', fontsize=10, style='italic')
            
            # Add value labels on top of bars
            for j, v in enumerate(values):
                if values and errors[j] > 0:
                    axes[i].text(j, v + errors[j] + (max(values) * 0.01 if values else 0), 
                            f"{v:.2f}", ha='center', va='bottom', fontsize=9)
                else:
                    axes[i].text(j, v + (max(values) * 0.01 if values else 0), 
                            f"{v:.2f}", ha='center', va='bottom', fontsize=9)
        
        # Add super title
        if hasattr(self.results[0], 'problem_name'):
            plt.suptitle(f'Performance Profile: {self.results[0].problem_name}', fontsize=16)
        else:
            plt.suptitle('Performance Profile', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig

    def get_standard_points(self, problem_type, problem_size):
        """Get standard reference and ideal points based on problem type and size"""
        standard_points = {
            'BiTSP': {
                20: {'reference': (20, 20), 'ideal': (0, 0)},
                50: {'reference': (35, 35), 'ideal': (0, 0)},
                100: {'reference': (65, 65), 'ideal': (0, 0)},
                150: {'reference': (85, 85), 'ideal': (0, 0)},
                200: {'reference': (115, 115), 'ideal': (0, 0)}
            },
            'BiCVRP': {
                20: {'reference': (30, 4), 'ideal': (0, 0)},
                50: {'reference': (45, 4), 'ideal': (0, 0)},
                100: {'reference': (80, 4), 'ideal': (0, 0)} # 60, 4
            },
            # 'BiCVRP': {
            #     20: {'reference': (15, 3), 'ideal': (0, 0)},
            #     50: {'reference': (40, 3), 'ideal': (0, 0)},
            #     100: {'reference': (60, 4), 'ideal': (0, 0)}
            # },
            'BiKP': {
                50: {'reference': (5, 5), 'ideal': (30, 30)},
                100: {'reference': (20, 20), 'ideal': (50, 50)},#100: {'reference': (20, 20), 'ideal': (50, 50)},# 100: {'reference': (20, 20), 'ideal': (100, 100)},
                200: {'reference': (30, 30), 'ideal': (75, 75)}
            },
            'TriTSP': {
                20: {'reference': (20, 20, 20), 'ideal': (0, 0, 0)},
                50: {'reference': (35, 35, 35), 'ideal': (0, 0, 0)},
                100: {'reference': (65, 65, 65), 'ideal': (0, 0, 0)}
            },
            # Add entries for TriKP (3-objective knapsack problem)
            'TriKP': {
                50: {'reference': (500, 500, 500), 'ideal': (0, 0, 0)},
                100: {'reference': (500, 500, 500), 'ideal': (0, 0, 0)},
                200: {'reference': (500, 500, 500), 'ideal': (0, 0, 0)}
            }
        }
        
        # If problem type is not in the dictionary, return a default
        if problem_type not in standard_points:
            print(f"Warning: Unknown problem type '{problem_type}'. Using default reference and ideal points.")
            # Determine dimensionality based on problem type prefix
            if problem_type.startswith('Tri'):
                # 3D problem
                return {'reference': (500, 500, 500), 'ideal': (0, 0, 0)}
            else:
                # Default to 2D problem
                return {'reference': (100, 100), 'ideal': (0, 0)}
        
        # Find closest problem size if exact match not found
        if problem_size not in standard_points.get(problem_type, {}):
            sizes = sorted(standard_points.get(problem_type, {}).keys())
            if not sizes:
                print(f"Warning: No sizes defined for problem type '{problem_type}'. Using default reference and ideal points.")
                # Determine dimensionality based on problem type prefix
                if problem_type.startswith('Tri'):
                    # 3D problem
                    return {'reference': (500, 500, 500), 'ideal': (0, 0, 0)}
                else:
                    # Default to 2D problem
                    return {'reference': (100, 100), 'ideal': (0, 0)}
            
            # Find closest size
            closest_size = min(sizes, key=lambda x: abs(x - problem_size))
            print(f"Using reference and ideal points for size {closest_size} (closest to requested size {problem_size})")
            return standard_points[problem_type][closest_size]
        
        return standard_points.get(problem_type, {}).get(problem_size)

def test_evaluator_with_json():
    """Example usage with JSON support"""
    # Create problem instances
    bitsp = BiObjectiveTSP(n_cities=20)
    mokp = MultiObjectiveKnapsack(n_items=20, n_objectives=2, capacity=10.0)
    
    # Create algorithm instances with parameters
    wslkh_params = {'num_weights': 10, 'improvement_type': '2-opt'}
    wsdp_params = {'num_weights': 10, 'scale_factor': 10}
    
    wslkh = WSLKH(bitsp, **wslkh_params)
    wsdp = WSDP(mokp, **wsdp_params)
    
    # Initialize evaluator with normalized reference points
    evaluator = MOCOEvaluator(reference_point=(1.2, 1.2))  # Slightly larger than normalized range
    
    # Evaluate algorithms
    print("\nEvaluating and saving results:")
    
    # Evaluate TSP
    tsp_result = evaluator.evaluate_algorithm(
        wslkh, bitsp, "WS-LKH", 
        parameters=wslkh_params,
        num_runs=3
    )
    
    # Evaluate Knapsack
    knapsack_result = evaluator.evaluate_algorithm(
        wsdp, mokp, "WS-DP",
        parameters=wsdp_params,
        num_runs=3
    )
    
    # Save all results
    evaluator.save_all_results("all_results.json")
    
    # Generate report
    evaluator.generate_report()
    
    # # Load and verify results
    # evaluator = MOCOEvaluator(reference_point=(1.2, 1.2))  # Slightly larger than normalized range
    # new_evaluator.load_results("all_results.json")
    
    print("\nLoaded results:")
    for result in evaluator.results:
        print(f"\n{result.algorithm_name} on {result.problem_name}:")
        print(f"Runtime: {result.runtime:.2f} seconds")
        print(f"Hypervolume: {result.hypervolume:.4f}")
        print(f"Non-dominated solutions: {result.num_nondominated}")
        print(f"Parameters used: {result.parameters}")
        print(f"Problem size: {result.problem_size}")

if __name__ == "__main__":
    test_evaluator_with_json()