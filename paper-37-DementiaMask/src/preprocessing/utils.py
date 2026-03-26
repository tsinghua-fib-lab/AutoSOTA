from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import itertools
import warnings
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import os 

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
data_dir = project_root / "data"


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SplitParameters:
    """Container for split configuration parameters"""
    train_pos_prob_z0: float
    train_pos_prob_z1: float
    test_pos_prob_z0: float
    test_pos_prob_z1: float
    mix_prob_z0: float
    mix_prob_z1: float
    alpha_train: float
    alpha_test: float
    marginal_y_prob: float
    marginal_z_prob: float


@dataclass
class SampleRequirements:
    """Container for sample size requirements"""
    train_total: int
    test_total: int
    z0_pos_train: int
    z0_neg_train: int
    z1_pos_train: int
    z1_neg_train: int
    z0_pos_test: int
    z0_neg_test: int
    z1_pos_test: int
    z1_neg_test: int
    parameters: SplitParameters



def calculate_split_parameters(
    train_pos_prob_z0: float,
    train_pos_prob_z1: float,
    mix_prob_z1: float,
    alpha_test: float
) -> SplitParameters:
    """Calculate dataset split parameters with probability constraints.
    
    Args:
        train_pos_prob_z0: P(Y=1|Z=0) in training set
        train_pos_prob_z1: P(Y=1|Z=1) in training set
        mix_prob_z1: P(Z=1) in both train/test sets
        alpha_test: Ratio P(Y=1|Z=1)/P(Y=1|Z=0) in test set
        
    Returns:
        SplitParameters: Structured split configuration
    """
    # check if probabilities are in [0,1] range
    for prob in [train_pos_prob_z0, train_pos_prob_z1, mix_prob_z1]:
        if not 0 <= prob <= 1:
            raise ValueError(f"Probability {prob} out of [0,1] range")
    # check if alpha_test is non-negative
    if alpha_test < 0:
        raise ValueError("alpha_test must be non-negative")

    mix_prob_z0 = 1 - mix_prob_z1
    marginal_y_prob = mix_prob_z0 * train_pos_prob_z0 + mix_prob_z1 * train_pos_prob_z1

    try:
        test_pos_prob_z0 = marginal_y_prob / (1 - (1 - alpha_test) * mix_prob_z1)
        test_pos_prob_z1 = alpha_test * test_pos_prob_z0
    except ZeroDivisionError as e:
        raise ValueError("Invalid parameter combination causing division by zero") from e

    return SplitParameters(
        train_pos_prob_z0=train_pos_prob_z0,
        train_pos_prob_z1=train_pos_prob_z1,
        test_pos_prob_z0=test_pos_prob_z0,
        test_pos_prob_z1=test_pos_prob_z1,
        mix_prob_z0=mix_prob_z0,
        mix_prob_z1=mix_prob_z1,
        alpha_train=train_pos_prob_z1 / train_pos_prob_z0,
        alpha_test=alpha_test,
        marginal_y_prob=marginal_y_prob,
        marginal_z_prob=mix_prob_z1,
    )



def calculate_sample_requirements(
    train_pos_prob_z0: float,
    train_pos_prob_z1: float,
    mix_prob_z1: float,
    alpha_test: float,
    train_test_ratio: int = 4,
    test_size: int = 100,
    min_samples: int = 10,
    verbose: bool = True
) -> Optional[SampleRequirements]:
    """Calculate required sample sizes for valid experimental setup."""
    params = calculate_split_parameters(
        train_pos_prob_z0,
        train_pos_prob_z1,
        mix_prob_z1,
        alpha_test
    )
    train_size = test_size * train_test_ratio
    z1_train = round(train_size * params.marginal_z_prob)
    z0_train = train_size - z1_train

    z1_test = round(test_size * params.marginal_z_prob)
    z0_test = test_size - z1_test

    # Calculate positive/negative counts for each group
    counts = {
        'z0_pos_train': round(z0_train * params.train_pos_prob_z0),
        'z0_neg_train': z0_train - round(z0_train * params.train_pos_prob_z0),
        'z1_pos_train': round(z1_train * params.train_pos_prob_z1),
        'z1_neg_train': z1_train - round(z1_train * params.train_pos_prob_z1),
        'z0_pos_test': round(z0_test * params.test_pos_prob_z0),
        'z0_neg_test': z0_test - round(z0_test * params.test_pos_prob_z0),
        'z1_pos_test': round(z1_test * params.test_pos_prob_z1),
        'z1_neg_test': z1_test - round(z1_test * params.test_pos_prob_z1),
    }

    if any(v < min_samples for v in counts.values()):
        if verbose:
            logger.warning(f"Insufficient samples: {', '.join(f'{k}={v}' for k,v in counts.items() if v < min_samples)}")
        return None

    return SampleRequirements(
        train_total=train_size,
        test_total=test_size,
        parameters=params,
        **counts
    )

def generate_experiment_configurations(
    train_pos_probs_z0: Union[float, List[float]],
    train_pos_probs_z1: Union[float, List[float]],
    alpha_tests: Union[float, List[float]],
    mix_probs_z1: Union[float, List[float]] = (0.5,),
    test_size: int = 150,
    min_samples: int = 10
) -> List[SampleRequirements]:
    """Generate valid experimental configurations.
    
    Args:
        train_pos_probs_z0: P(Y=1|Z=0) in train (single or list)
        train_pos_probs_z1: P(Y=1|Z=1) in train (single or list)
        alpha_tests: Test set alpha ratios (single or list)
        mix_probs_z1: Mixing probabilities for Z=1 (default: 0.5)
        test_size: Base test set size
        min_samples: Minimum samples per group
        
    Returns:
        List of valid SampleRequirements configurations
    """
    params = {
        'p0': _ensure_list(train_pos_probs_z0),
        'p1': _ensure_list(train_pos_probs_z1),
        'alpha': _ensure_list(alpha_tests),
        'mix_z1': _ensure_list(mix_probs_z1)
    }

    valid_configs = []
    for p0, p1, alpha, mix_z1 in itertools.product(*params.values()):
        config = calculate_sample_requirements(
            train_pos_prob_z0=p0,
            train_pos_prob_z1=p1,
            mix_prob_z1=mix_z1,
            alpha_test=alpha,
            test_size=test_size,
            min_samples=min_samples,
            verbose=True
        )
        if config:
            valid_configs.append(config)
    
    logger.info(f"Generated {len(valid_configs)} valid configurations")
    return valid_configs

def _ensure_list(param: Union[float, List[float]]) -> List[float]:
    """Ensure parameter is a list"""
    return param if isinstance(param, (list, tuple)) else [param]




def load_data(data_name: str = 'pitts') -> Tuple[pd.DataFrame, pd.DataFrame]:
    if data_name == 'pitts':
        df_pitts = pd.read_csv(data_dir/"processed_pitts_last.csv")
        gender_map = {'female': 1, 'male': 0} # code female as 1
        df_pitts['gender'] = df_pitts['gender'].replace(gender_map)
        # split original dataset into two
        df_pitts_male = df_pitts.query("gender==0").reset_index(drop=True)
        df_pitts_female = df_pitts.query("gender==1").reset_index(drop=True)
        
        return (df_pitts_male, df_pitts_female), 'label', 'gender'

    
    elif data_name == 'wls':
        df_wls = pd.read_csv(data_dir/"processed_wls.csv", index_col = 0)

        # split original dataset into two
        df_wls_male = df_wls[df_wls['gender']==0]
        df_wls_female = df_wls[df_wls['gender']==1]
    
        return (df_wls_male, df_wls_female), 'label', 'gender'
    
    elif data_name == 'ccc':
        df_ccc = pd.read_csv(data_dir/"processed_ccc.csv", index_col = 0)

        # split original dataset into two
        df_ccc_male = df_ccc[df_ccc['gender']==0]
        df_ccc_female = df_ccc[df_ccc['gender']==1]

        return (df_ccc_male, df_ccc_female), 'label', 'gender'
    

def id_aware_split(
        df: pd.DataFrame,
        n_train: int,
        n_test: int,
        sample: bool = True,
        seed: int = 2023,
        id_col: str = "id",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """  
    Split DataFrame into train/test sets while maintaining ID integrity
    
    Args:
        df: Input DataFrame containing multiple instances per ID
        n_train: Target number of training instances
        n_test: Target number of test instances
        sample: Allow sampling with replacement if needed
        seed: Random seed for reproducibility
        id_col: Name of the ID column
        
    Returns:
        Tuple of (train_df, test_df) or (None, None) if invalid
    """
    if n_train + n_test == 0:
        return pd.DataFrame(), pd.DataFrame()

    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame")
    
    unique_ids = df[id_col].drop_duplicates().values
    n_target = n_train + n_test

    # edge case: single id
    if len(unique_ids) == 1:
        if sample:
            df_train, df_test = train_test_split(
                df, 
                train_size=n_train, 
                test_size=n_test, 
                random_state=seed,
                shuffle=True,
            )
        return df_train, df_test

    # case: number of ids > number of target samples
    # here we 
    # 1. split number of ids that are equal to the target number of samples (isolate train and test ids)
    # 2. include all the instances for each id
    # 3. we again sample from all the instances to get the ideal number
    if len(unique_ids) >= n_target:
        train_ids, test_ids = train_test_split(unique_ids,
                    train_size=n_train,
                    test_size=n_test,
                    shuffle = True, 
                    random_state=seed)

                # Get all instances for selected IDs
        df_train = df[df[id_col].isin(train_ids)]
        df_test = df[df[id_col].isin(test_ids)]

    # case number of ids < number of target samples
    # here we 
    # 1. split the ids proportionally to the target number of samples
    # 2. if the target number of samples is not reached, we sample with replacement
    else:
        # Split IDs proportionally
        train_ids, test_ids = train_test_split(
            unique_ids,
            test_size=n_test/(n_train + n_test),
            random_state=seed,
            shuffle=True
        )
        
        # Get base instances
        df_train = df[df[id_col].isin(train_ids)]
        df_test = df[df[id_col].isin(test_ids)]

    # resize to target sizes
    if sample and (len(df_train) != n_train or len(df_test) != n_test):
        # Sample to reach target sizes
        df_train = _sample_to_size(df_train, n_train, random_state=seed)
        df_test = _sample_to_size(df_test, n_test, random_state=seed)

    assert len(df_train) == n_train, f"Train size mismatch: {len(df_train)} != {n_train}"
    assert len(df_test) == n_test, f"Test size mismatch: {len(df_test)} != {n_test}"    
    
    return df_train, df_test
    

            

def _sample_to_size(
    df: pd.DataFrame,
    target_size: int,
    random_state: int
    ) -> pd.DataFrame:
    """Helper to sample DataFrame to exact size"""
    if len(df) == 0:
        return pd.DataFrame()
    
    if len(df) >= target_size:
        return df.sample(n=target_size, random_state=random_state, replace=False)
    
    # Sample with replacement if needed
    n_samples = target_size - len(df)
    extra = df.sample(n=n_samples, replace=True, random_state=random_state)
    return pd.concat([df, extra], ignore_index=True) 

def check_mutual_exclusive(train_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "id") -> None:
    """Check if IDs are mutually exclusive in train and test sets"""
    if not set(train_df[id_col]).isdisjoint(test_df[id_col]) and (len(set(train_df[id_col]))!=1 or len(set(test_df[id_col]))!=1):
        raise ValueError("Train and test sets are not mutually exclusive")
    
def create_stratified_mixture(
    z0_df: pd.DataFrame,
    z1_df: pd.DataFrame,
    target_col: str,
    setting: Dict[str, int],
    sample: bool = True,
    seed: int = 2023
) -> Dict[str, pd.DataFrame]:
    """Create stratified mixture dataset from two sources with ID-aware splitting.
    
    Args:
        z0_df: Data from source Z=0
        z1_df: Data from source Z=1
        target_col: Name of target column
        settings: Dictionary containing split sizes
        sample: Allow sampling with replacement
        seed: Random seed
        
    Returns:
        Dictionary with 'train' and 'test' DataFrames
    """
    np.random.seed(seed)
    pd.util.hash_pandas_object = pd.util.hash_pandas_object  # Fix for pandas 2.0+

    def process_group(df_group: pd.DataFrame, prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pos_df = df_group[df_group[target_col] == 1]
        neg_df = df_group[df_group[target_col] == 0]
        pos_train, pos_test = id_aware_split(
            pos_df, 
            n_train = getattr(setting, f'{prefix}_pos_train'),
            n_test  = getattr(setting, f'{prefix}_pos_test'),
            sample=sample,
            seed=seed
        )
        
        neg_train, neg_test = id_aware_split(
            neg_df,
            n_train = getattr(setting, f'{prefix}_neg_train'),
            n_test  = getattr(setting, f'{prefix}_neg_test'),
            sample=sample,
            seed=seed
        )
             # check if id are mutually exclusive in train and test
        check_mutual_exclusive(pos_train, pos_test)
        check_mutual_exclusive(neg_train, neg_test)
        
        return (
            pd.concat([pos_train, neg_train]),
            pd.concat([pos_test, neg_test])
        )

    z0_train, z0_test = process_group(z0_df, 'z0')
    z1_train, z1_test = process_group(z1_df, 'z1')

    return {
        'train': pd.concat([z0_train, z1_train]).sample(frac=1, random_state=seed).reset_index(drop=True),
        'test': pd.concat([z0_test, z1_test]).sample(frac=1, random_state=seed).reset_index(drop=True),
        'settings': setting
    }