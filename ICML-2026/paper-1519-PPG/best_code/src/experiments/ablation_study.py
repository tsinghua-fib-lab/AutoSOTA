from abc import ABC, abstractmethod
from statistics import mean, stdev

from joblib import Parallel, delayed
from tqdm import tqdm

from src.envs.gridworld import Gridworld
from src.performative_pepg import PePG
from src.reg_pepg import RegPePG


class AblationStudy(ABC):
    """Base class for ablation studies."""

    def __init__(self, env_params, common_params, seeds, n_jobs):
        self.env_params = env_params
        self.common_params = common_params
        self.seeds = seeds
        self.n_jobs = min(n_jobs, len(seeds))

    @abstractmethod
    def get_parameter_values(self):
        """Return list of parameter values to test."""
        pass

    @abstractmethod
    def get_parameter_name(self):
        """Return parameter name being tested."""
        pass

    @abstractmethod
    def run_single_experiment(self, param_value, seed):
        """Run single experiment with given parameter and seed."""
        pass

    @abstractmethod
    def get_algorithm_name(self):
        """Return algorithm name."""
        pass

    def run(self):
        """Execute ablation study across all parameter values."""
        results = []
        param_values = self.get_parameter_values()
        param_name = self.get_parameter_name()

        for param_value in param_values:
            print(f"\n[{self.get_algorithm_name()}] Testing {param_name}={param_value}")

            results_across_seeds = Parallel(n_jobs=self.n_jobs)(
                delayed(self.run_single_experiment)(param_value, seed)
                for seed in tqdm(self.seeds, desc=f"{param_name}={param_value}")
            )

            d_diffs = [result["d_diff"] for result in results_across_seeds]
            v_values_list = [
                result["v_values"]
                for result in results_across_seeds
                if result["v_values"]
            ]

            result_entry = self._create_result_entry(param_value, param_name)

            if d_diffs:
                result_entry["d_diff_mean"] = list(map(mean, zip(*d_diffs)))
                result_entry["d_diff_std"] = list(map(stdev, zip(*d_diffs)))
                print(
                    f"  d_diff: {result_entry['d_diff_mean'][-1]:.6f} ± {result_entry['d_diff_std'][-1]:.6f}"
                )

            if v_values_list:
                result_entry["v_values_mean"] = list(map(mean, zip(*v_values_list)))
                result_entry["v_values_std"] = list(map(stdev, zip(*v_values_list)))
                print(
                    f"  value:  {result_entry['v_values_mean'][-1]:.4f} ± {result_entry['v_values_std'][-1]:.4f}"
                )

            results.append(result_entry)

        return {self.get_algorithm_name().lower(): results}

    def _create_result_entry(self, param_value, param_name):
        return {
            "beta": self.env_params["beta"],
            "lamda": self.common_params["lamda"],
            "gamma": self.env_params["gamma"],
            "reg": self.common_params["reg"],
            "eta": self.common_params.get("eta"),
            "nu": self.common_params.get("nu"),
            "warmup": self.common_params.get("warmup"),
            param_name: param_value,
            "algorithm": self.get_algorithm_name(),
        }


class EntropyAblationStudy(AblationStudy):
    """Entropy regularization ablation study."""

    def __init__(self, env_params, common_params, entropy_lambdas, seeds, n_jobs):
        super().__init__(env_params, common_params, seeds, n_jobs)
        self.entropy_lambdas = entropy_lambdas

    def get_parameter_values(self):
        return self.entropy_lambdas

    def get_parameter_name(self):
        return "entropy_reg_lambda"

    def get_algorithm_name(self):
        return "RegPePG"

    def run_single_experiment(self, entropy_lambda, seed):
        env = Gridworld(**self.env_params)
        params = self.common_params.copy()
        params["entropy_reg_lambda"] = entropy_lambda
        params["seed"] = seed

        regpepg = RegPePG(env, **params)
        regpepg.execute()

        return {
            "d_diff": regpepg.d_diff,
            "v_values": regpepg.v_values if regpepg.v_values else [],
        }


class LearningRateAblationStudy(AblationStudy):
    """Learning rate ablation study."""

    def __init__(self, env_params, common_params, learning_rates, seeds, n_jobs):
        super().__init__(env_params, common_params, seeds, n_jobs)
        self.learning_rates = learning_rates

    def get_parameter_values(self):
        return self.learning_rates

    def get_parameter_name(self):
        return "eta"

    def get_algorithm_name(self):
        return "PePG"

    def run_single_experiment(self, eta, seed):
        env = Gridworld(**self.env_params)
        params = self.common_params.copy()
        params["eta"] = eta
        params["seed"] = seed

        pepg = PePG(env, **params)
        pepg.execute()

        return {
            "d_diff": pepg.d_diff,
            "v_values": pepg.v_values if pepg.v_values else [],
        }
