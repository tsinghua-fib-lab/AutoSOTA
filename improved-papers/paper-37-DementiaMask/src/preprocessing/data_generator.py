"""
Generate confounding shift data

Handles:
- Synthetic data creation
- Train/test splitting
- Distribution validation
- Dataset serialization
"""

from pathlib import Path
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from .utils import (
    generate_experiment_configurations, 
    load_data, 
    create_stratified_mixture,
)

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    p_pos_train_z0: Union[float, List[float]]
    p_pos_train_z1: Union[float, List[float]]
    alpha_test: Union[float, List[float]]
    mix_probs_z1: Union[float, List[float]] = (0.5,)
    test_size: int = 150
    min_samples: int = 10

class CfDataGenerator:
    """Handles dataset creation with controlled confounding effects"""
    
    def __init__(self, data_name: str, data_seed: int, config: DataConfig, save: bool,
                 base_data: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None) -> None:
        self.data_name = data_name
        self.config = config
        self.seed = data_seed
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.save = save

        if base_data:
            self.base_data = base_data
        
    def generate_dataset(self, save_dir: Path = Path("data"), verbose: bool =True) -> None:
        np.random.seed(self.seed)
        """Generate and save dataset with specified characteristics"""
        base_data, label, confounder = load_data(self.data_name)
        settings = self._create_settings()
        self.label = label
        self.confounder = confounder
                
        if not settings:
            raise ValueError("No valid configurations found")
        
        # replace with user defined base data
        if hasattr(self, 'base_data'):
            base_data = self.base_data

        # pick one valid setting
        sett = np.random.choice(settings)
        
        z0_df = base_data[0].rename(columns={self.label: "target", self.confounder: "confounder"})
        z1_df = base_data[1].rename(columns={self.label: "target", self.confounder: "confounder"})
        

        self.datasets = create_stratified_mixture(
            z0_df=z0_df,
            z1_df=z1_df,
            target_col='target',
            setting=sett,
            sample=True,
            seed=self.seed
        )
        
        self._validate_distribution(verbose)
        if self.save:
            self._save_data(save_dir)

    def _create_settings(self) -> List[Dict]:
        """Generate valid dataset configurations"""
        return generate_experiment_configurations(
            train_pos_probs_z0=self.config.p_pos_train_z0,
            train_pos_probs_z1=self.config.p_pos_train_z1,
            alpha_tests=self.config.alpha_test,
            mix_probs_z1=self.config.mix_probs_z1,
            test_size=self.config.test_size,
            min_samples=self.config.min_samples
        )

    
    def _validate_distribution(self, verbose, tol: float = 0.03) -> None:
        """Validate dataset meets distribution requirements"""
        train_df = self.datasets["train"]
        test_df = self.datasets["test"]
        
        # Calculate empirical probabilities
        p_z1_train = self._calc_conditional_prob(train_df, 1)
        p_z0_train = self._calc_conditional_prob(train_df, 0)
        p_z1_test = self._calc_conditional_prob(test_df, 1)
        p_z0_test = self._calc_conditional_prob(test_df, 0)

        if verbose:
            print("\n" + "Empirical Probabilities of Y=1 given Z")
            print("-"*25)
            print("| P(Y=1|Z) | Z=0 |  Z=1  |")
            print("-"*25)
            print(f"| Train    | {p_z0_train:.2f} | {p_z1_train:.2f} |")
            print(f"| Test     | {p_z0_test:.2f} | {p_z1_test:.2f} |")
            print("-"*25)

        if not (np.isclose(p_z1_train, self.config.p_pos_train_z1, atol=tol, rtol=tol) and
                np.isclose(p_z0_train, self.config.p_pos_train_z0, atol=tol, rtol=tol) and
                np.isclose(p_z1_test/p_z0_test, self.config.alpha_test, atol=tol, rtol=tol)):
            raise ValueError("Generated data distribution validation failed")

    def _calc_conditional_prob(self, df: pd.DataFrame, z_value: int) -> float:
        """Calculate P(Y=1|Z=z)"""
        subset = df[df['confounder'] == z_value]
        return len(subset[subset['target'] == 1]) / len(subset)

    
    def _save_data(self, save_dir: Path) -> None:
        """Save generated datasets to disk"""
        save_dir.mkdir(parents=True, exist_ok=True)
        self.datasets["train"].to_csv(save_dir / "train.csv", index=False)
        self.datasets["test"].to_csv(save_dir / "test.csv", index=False)
        logger.info(f"Datasets saved to {save_dir}")


def cf_train_test_split(
    data_name: str, 
    train_pos_z0: Union[float, List[float]],
    train_pos_z1: Union[float, List[float]],
    alpha_test: Union[float, List[float]],
    test_size: int, 
    pz: float = 0.5,
    validation_size: Optional[int] = None,
    random_state: int = 2025,
    min_samples: int = 10,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets"""
    config = DataConfig(
        p_pos_train_z0=train_pos_z0,
        p_pos_train_z1=train_pos_z1,
        alpha_test=alpha_test,
        mix_probs_z1=pz,
        test_size=test_size,
        min_samples=min_samples
        )
    generator = CfDataGenerator(data_name, random_state, config, save=False)
    if verbose:
        print("Splitting train and test datasets...")
    generator.generate_dataset(verbose=verbose)
    if validation_size:
        # create a validation set with the same distribution as the train set
        _config = DataConfig(
            p_pos_train_z0=train_pos_z0,
            p_pos_train_z1=train_pos_z1,
            alpha_test=train_pos_z1/train_pos_z0,
            mix_probs_z1=pz,
            test_size=validation_size,
            min_samples=min_samples
        )
        if verbose:
            print("Splitting train and validation datasets...")
        train_data = generator.datasets['train']
        base_data = (train_data[train_data['confounder'] == 0], train_data[train_data['confounder'] == 1])
        val_generator = CfDataGenerator(data_name, random_state, _config, save=False, base_data=base_data)
        val_generator.generate_dataset(verbose=verbose)
        return val_generator.datasets["train"], val_generator.datasets["test"], generator.datasets["test"]
    
    return generator.datasets["train"], generator.datasets["test"]


if __name__ == "__main__":
    config = DataConfig(
        p_pos_train_z0=0.2,
        p_pos_train_z1=0.8,
        alpha_test=0.25,
        test_size=150,
    )
    
    generator = CfDataGenerator("pitts", config)
    generator.generate_dataset()