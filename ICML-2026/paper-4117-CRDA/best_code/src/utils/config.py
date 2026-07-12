import os
from typing import Optional, List, Union, Dict, Any
import time

class Config:
    """
    Configuration class for counterfactual data augmentation experiments.
    Generates a unique name for each experiment run and stores experiment parameters.
    """
    def __init__(
        self,
        # Dataset parameters
        dataset_path: str = "./data/HousePrice.csv",
        dataset_dir_path: str = None,

        # Model parameters
        baseline: str = "mlp",  # Options: "mlp", "xgboost"
        
        # Data augmentation parameters
        aug_data_size_factor: float = 1.0, # 1.0 means we generate the same number of augmentedsamples as the original data
        max_n_features_to_perturb: int = 5,  # Maximum number of features to perturb
        max_perturb_percent: float = 0.1,  # Maximum percentage to perturb with (0.1 = 10%)
        min_perturb_percent: float = -0.1,  # Minimum percentage to perturb with (0.1 = 10%)
        aug_data_weight: float = 0.9,  # Weight of the augmented data

        # Causal parameters
        alpha: float = 0.05,  # Significance level for causal tests
        r_corr_threshold: float = 0.1,  # Significance threshold for pearson correlation coefficient
        p_corr_threshold: float = 0.05,  # Significance threshold for pearson p value
        
        # Experiment parameters
        test_size: float = 0.2,  # Test split ratio
        random_seed: int = 0,  # Random seed for reproducibility
        num_seeds: int = 15,  # Number of seeds to run as trials
        p_wilcoxon_threshold: float = 0.05,  # Significance threshold for wilcoxon test
        hyperparam_tune: bool = False, # if true, the hyperparameters and augmentation parameters will be tuned
        method_param_tune: bool = False, # if true, the method parameters will be tuned
        sample_sizes: Optional[list[int]] = None, # if None, the entire dataset will be used, otherwise a list of sample sizes will be used
        ignore_filter: bool = False, # if true, the filter will be ignored so execution will continue to the data augmentation step
        n_crda_iterations: int = 1, # number of CRDA iterative refinement rounds
        
        # Output parameters
        results_dir: str = "./runs",
        save_plots: bool = False, # if true, the plots (correlation, causal) will be saved
        save_models: bool = False, # if true, the models will be saved
        save_params: bool = False, # if true, the parameters will be saved
        
        # Additional parameters that can be passed as a dictionary
        **kwargs: Any
    ):
        # Generate a unique name for this experiment run
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.experiment_name = f"{baseline}_{self.timestamp}"
        
        # Dataset parameters
        self.dataset_path = dataset_path
        self.dataset_dir_path = dataset_dir_path
        
        # Model parameters
        self.baseline = baseline
        
        # Data generation parameters
        self.aug_data_size_factor = aug_data_size_factor
        self.max_n_features_to_perturb = max_n_features_to_perturb
        self.max_perturb_percent = max_perturb_percent
        self.min_perturb_percent = min_perturb_percent
        self.aug_data_weight = aug_data_weight
        
        # Causal parameters
        self.alpha = alpha
        self.r_corr_threshold = r_corr_threshold
        self.p_corr_threshold = p_corr_threshold
        
        # Experiment parameters
        self.test_size = test_size
        self.random_seed = random_seed
        self.num_seeds = num_seeds
        self.p_wilcoxon_threshold = p_wilcoxon_threshold
        self.hyperparam_tune = hyperparam_tune
        self.method_param_tune = method_param_tune
        self.sample_sizes = sample_sizes
        self.ignore_filter = ignore_filter
        self.n_crda_iterations = n_crda_iterations

        # Output parameters
        self.results_dir = results_dir
        self.save_plots = save_plots
        self.save_models = save_models
        self.save_params = save_params

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Create an experiment directory
        self.experiment_dir = self.get_experiment_dir()
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_experiment_dir(self) -> str:
        """
        Returns the directory path for this experiment's results.
        """
        exp_dir = os.path.join(self.results_dir, f"{self.experiment_name}")
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the config to a dictionary for serialization.
        """
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create a Config instance from a dictionary.
        """
        config = cls(**config_dict)
        return config
    
    def __str__(self) -> str:
        """
        Returns a string representation of the config.
        """
        return f"Experiment Config (Name: {self.experiment_name})\n" + "\n".join(
            f"  {k}: {v}" for k, v in self.__dict__.items() if k != 'experiment_name'
        )
