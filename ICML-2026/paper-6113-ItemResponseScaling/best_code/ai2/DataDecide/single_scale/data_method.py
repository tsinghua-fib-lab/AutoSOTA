import hashlib
import pickle
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from algorithms.scaling_law import ScalingLaw

def exponential_smoothing(values, alpha) -> list[float]:
    # Remove NaN values
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return [np.nan]  # Return NaN if all values are missing

    # Init with first value in series
    smoothed_values = [values[0]]
    for val in values[1:]:
        smoothed_value = alpha * val + (1 - alpha) * smoothed_values[-1]
        smoothed_values.append(smoothed_value)

    # # Calculate residuals
    # residuals = [values[i] - smoothed_values[i] for i in range(len(values))]
    # variance = np.var(residuals)
    # std = np.std(variance)
    return smoothed_values

def process_task(group_key, sub_df, task_metric, subset_idx, transform_fn):
    """
    Process a single task unit: a specific group, metric, and compute subset index.
    """
    # Sort by compute
    sub_df = sub_df.sort_values(by="compute")
    compute_values = sub_df["compute"].tolist()
    tokens_values = sub_df["tokens"].tolist()
    model_values = sub_df["model"].tolist()
    group, seed = group_key[-2], group_key[-1]

    # Incrementally compute subsets
    subset_compute = compute_values[:subset_idx + 1]
    subset_tokens = tokens_values[:subset_idx + 1]
    subset_models = model_values[:subset_idx + 1]
    subset_values = sub_df[task_metric].tolist()[:subset_idx + 1]

    # Transform values
    transformed_value = transform_fn(
        values=subset_values,
        models=subset_models,
        tokens=subset_tokens,
    )

    # Return result
    return {
        "model": subset_models[-1],
        "group": group,
        "seed": seed,
        "metric": task_metric,
        "models": subset_models,
        "compute_latest": subset_compute[-1],
        "token_latest": subset_tokens[-1],
        "raw_values": subset_values,
        "value": transformed_value,
    }

class BaseMethod:
    def __init__(self, method_args=None):
        self.method_args = method_args or {}
        self.cache = {}

    def _generate_cache_key(self, **kwargs):
        key_data = pickle.dumps(kwargs)
        return hashlib.md5(key_data).hexdigest()

    def transform(self, values, **kwargs):
        raise NotImplementedError("Subclass must implement transform method")

    # def __call__(self, df: pd.DataFrame, task_metrics: list[str]) -> pd.DataFrame:
    #     """
    #     Applies transform method and returns a DataFrame with transformed values.
    #     """
    #     transformed_rows = []
    #     # Determine groupby keys, since ScalingLaw takes data points from different model scales
    #     groupby_keys = ["group", "seed"] if isinstance(self, ScalingLawMethod) else ["model", "group", "seed"]

    #     for group_key, sub_df in tqdm(df.groupby(groupby_keys)):
    #         # Sort by compute
    #         sub_df = sub_df.sort_values(by="compute")

    #         compute_values = sub_df["compute"].tolist()
    #         tokens_values = sub_df["tokens"].tolist()
    #         model_values = sub_df["model"].tolist()
    #         group, seed = group_key[-2], group_key[-1]

    #         for metric in task_metrics:
    #             values = sub_df[metric].tolist()

    #             # Transform values with incrementally increasing compute
    #             for i in range(1, len(compute_values) + 1):
    #                 subset_compute = compute_values[:i]
    #                 subset_models = model_values[:i]
    #                 subset_tokens = tokens_values[:i]
    #                 subset_values = values[:i]

    #                 transformed_value = self.transform(
    #                     values=subset_values,
    #                     models=subset_models,
    #                     tokens=subset_tokens,
    #                 )

    #                 transformed_rows.append({
    #                     "group": group,
    #                     "seed": seed,
    #                     "metric": metric,
    #                     "models": subset_models,
    #                     "compute_latest": subset_compute[-1],
    #                     "token_latest": subset_tokens[-1],
    #                     "raw_values": subset_values,
    #                     "value": transformed_value,
    #                 })

    #     return pd.DataFrame(transformed_rows)

    def __call__(self, df: pd.DataFrame, task_metrics: list[str]) -> pd.DataFrame:
        """
        Applies the transform method and returns a DataFrame with transformed values, parallelized for all loops.
        """
        # Determine groupby keys
        groupby_keys = ["group", "seed"] if isinstance(self, ScalingLawMethod) else ["model", "group", "seed"]

        # Prepare tasks for parallel processing
        tasks = []
        for group_key, sub_df in df.groupby(groupby_keys):
            compute_count = len(sub_df["compute"])
            for task_metric in task_metrics:
                for subset_idx in range(compute_count):
                    tasks.append((group_key, sub_df, task_metric, subset_idx, self.transform))

        # Process tasks in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_task, *task) for task in tasks]

            # Gather results
            for future in tqdm(futures, desc="Processing metric transfomrs"):
                results.append(future.result())

        return pd.DataFrame(results)

class Raw(BaseMethod):
    def transform(self, values, **kwargs) -> float:
        return values[-1] # last raw value

class ExponentialSmoothing(BaseMethod):
    def __init__(self, method_args=None):
        super().__init__(method_args)
        self.alpha = self.method_args.get("alpha", 0.1)

    def transform(self, values, **kwargs) -> float:
        cache_key = self._generate_cache_key(values=values, alpha=self.alpha)
        if cache_key in self.cache:
            return self.cache[cache_key]

        transformed_values = exponential_smoothing(values, self.alpha)
        # Get last weighted value
        last_value = transformed_values[-1]
        self.cache[cache_key] = last_value
        return last_value

class ScalingLawMethod(BaseMethod):
    def __init__(self, method_args=None, device="cpu"):
        super().__init__(method_args)
        self.device = device
        self.scaling_law = ScalingLaw(
            lin_space=self.method_args.get("lin_space", 4),
            device=self.device
        )
        self.max_compute = None # Compute we will try to predict

    def transform(self, values, models, tokens,**kwargs):
        """
        Predict target scale value using scaling laws, incrementally fitted for increasing compute.
        Bulk of work is in param fit search.
        """
        metric_df = pd.DataFrame({
            "param_count": [self.scaling_law.map_parameters(model) for model in models],
            "tokens": tokens,
            "values": values,
        }).dropna()
        # Handle zero tokens
        if (metric_df["tokens"] == 0).all():
            logging.warning("All tokens are zero. Returning raw value.")
            return values[-1]

        # Fit parameters for the current subset
        params = self.scaling_law.fit_scaling_law(metric_df)
        logging.info(f"Best params: {params}")
        if params is None:
            logging.warning("Since params is None, returning raw value.")
            return values[-1]

        # Predict target value
        predicted_value = self.scaling_law.compute_predicted_value(*params, self.max_compute)
        return predicted_value

    def __call__(self, df: pd.DataFrame, task_metrics: list[str]) -> pd.DataFrame:
        """
        Apply transformation to the DataFrame for all task metrics and groups.
        """
        self.max_compute = max(df["compute"])

        # Call BaseMethod to perform transformations
        return super().__call__(df, task_metrics)
