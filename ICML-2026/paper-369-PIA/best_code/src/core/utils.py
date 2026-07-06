"""
General utility functions
"""
import os
import logging
import pandas as pd
from typing import Optional, Dict, Any


# Global debug flag
_DEBUG_MODE = False


def set_debug_mode(debug: bool):
    """Set global debug mode for logging"""
    global _DEBUG_MODE
    _DEBUG_MODE = debug
    # Update all existing loggers
    log_level = get_log_level()
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


def get_log_level() -> int:
    """Get current log level based on debug mode"""
    return logging.DEBUG if _DEBUG_MODE else logging.INFO


def setup_logger(name: str, log_file: Optional[str] = None, log_level: int = None) -> logging.Logger:
    """
    Setup logger

    Args:
        name: Logger name
        log_file: Log file path (optional)
        log_level: Logging level (default: uses global debug mode)

    Returns:
        Configured Logger object
    """
    logger = logging.getLogger(name)

    # Use global debug mode if no level specified
    if log_level is None:
        log_level = get_log_level()

    logger.setLevel(log_level)

    # Avoid duplicate handler addition
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File output
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def safe_save_csv(df: pd.DataFrame, filepath: str, mode: str = 'w') -> bool:
    """
    Safely save DataFrame to CSV (with error handling)

    Args:
        df: DataFrame to save
        filepath: File path
        mode: Write mode 'w' overwrite or 'a' append

    Returns:
        Success status
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Append mode needs to determine if header is needed
        header = not (mode == 'a' and os.path.exists(filepath))

        df.to_csv(filepath, mode=mode, header=header, index=False)
        return True
    except Exception as e:
        print(f"Failed to save CSV: {e}")
        return False


def load_csv_with_validation(filepath: str, required_columns: Optional[list] = None) -> Optional[pd.DataFrame]:
    """
    Load CSV and validate column structure

    Args:
        filepath: CSV file path
        required_columns: List of required column names

    Returns:
        DataFrame or None (if failed)
    """
    try:
        if not os.path.exists(filepath):
            print(f"File does not exist: {filepath}")
            return None

        df = pd.read_csv(filepath)

        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                print(f"Missing required columns: {missing}")
                return None

        return df
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return None


def filter_valid_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter valid action data

    Args:
        df: Original data

    Returns:
        Filtered data
    """
    if 'action' not in df.columns:
        raise ValueError("DataFrame must contain 'action' column")

    return df[df['action'].isin(['Compliance', 'Refusal'])].copy()


def prepare_for_fitting(df: pd.DataFrame, filters: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Prepare data for cognitive model fitting

    Args:
        df: Original experiment data
        filters: Dictionary of filter settings

    Returns:
        Preprocessed data
    """
    if filters is None:
        filters = {}

    # Filter valid actions
    df = filter_valid_actions(df)

    # Apply custom action filters from config
    exclude_actions = filters.get('exclude_actions', ['ParseFail'])
    if exclude_actions:
        df = df[~df['action'].isin(exclude_actions)].copy()

    # Action mapping: 0=Refusal, 1=Compliance
    df['a_idx'] = df['action'].apply(lambda x: 1 if x == 'Compliance' else 0)

    # Reward column processing
    if 'reward' in df.columns:
        df['reward'] = pd.to_numeric(df['reward'], errors='coerce').fillna(0.0)

    # Handle negative reinforcement for Punishment and Optimism-Neg
    neg_mask = ((df['group'] == 'Punishment') | (df['group'] == 'Optimism-Neg')) & (df['reward'] == 0)
    if neg_mask.any():
        df.loc[neg_mask, 'reward'] = -1.0

    # is_full_feedback processing
    if 'is_full_feedback' in df.columns:
        df['is_full_feedback'] = df['is_full_feedback'].apply(
            lambda x: str(x).lower() in ['true', '1', 'yes']
        )
    else:
        df['is_full_feedback'] = False

    # forgone_reward processing
    if 'forgone_reward' not in df.columns:
        df['forgone_reward'] = 0.0

    # file_id processing (for session reset)
    if 'file_id' not in df.columns:
        df['file_id'] = 1

    # Filter by minimum trials per group
    min_trials = filters.get('min_trials_per_group', 5)
    if min_trials > 0:
        group_counts = df['group'].value_counts()
        valid_groups = group_counts[group_counts >= min_trials].index
        df = df[df['group'].isin(valid_groups)].copy()

    return df


def format_results_report(results: Dict) -> str:
    """
    Format fitting result report (all 9 parameters always shown)

    Args:
        results: FitResult dictionary

    Returns:
        Formatted string
    """
    if not results:
        return "No results to display"

    header = f"{'GROUP':<15} | {'α+':<6} | {'α-':<6} | {'ρ':<6} | {'R_p':<6} | {'θ':<6} | {'λ':<6} | {'φ':<6} | {'BIC':<8} | {'NLL':<8} | {'H':<8}"
    lines = [header, "-" * len(header)]

    for name, result in results.items():
        p = result.params
        row = (
            f"{name:<15} | "
            f"{p['alpha_pos']:<6.3f} | "
            f"{p['alpha_neg']:<6.3f} | "
            f"{p['rho']:<6.3f} | "
            f"{p['R_perc']:<6.1f} | "
            f"{p['theta']:<6.3f} | "
            f"{p['lambda']:<6.3f} | "
            f"{p['phi']:<6.3f} | "
            f"{result.bic:<8.1f} | "
            f"{result.nll:<8.1f} | "
            f"{result.entropy:<8.1f}"
        )
        lines.append(row)

    return "\n".join(lines)