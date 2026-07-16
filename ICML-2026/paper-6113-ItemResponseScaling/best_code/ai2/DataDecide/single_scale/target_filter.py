from itertools import combinations
from algorithms.log_fit import LogFit
from collections import defaultdict
from utils import find_common_checkpoints
import logging


class TargetFilter:
    def __init__(self, target_model, method=None, method_args=None):
        """Initialize the TargetFilter with specified method and parameters."""
        self.target_model = target_model
        self.method = method
        self.method_args = method_args or {}
        if self.method == "log_fit":
            self.fit_method = LogFit(threshold=self.method_args.get("threshold", 1.0))
            self.target_keep_fn = self.fit_method.target_keep_fn
        else:
            self.fit_method = None
            self.target_keep_fn = None

    def _get_primary_values(self, df, primary_metric="primary_metric") -> dict:
        """
        Compute primary values for the chosen target model.
        Return dict of primary values for each group.
        """
        # Filter for chosen target model
        df_target_model = df[df['model'] == self.target_model]
        if df_target_model.empty:
            raise ValueError(f"No data found for target model '{self.target_model}'")

        primary_values = defaultdict(lambda: defaultdict(list))
        for (group, seed), sub_df in df_target_model.groupby(['group', 'seed']):
            sub_df = sub_df.sort_values(by="compute")
            primary_values[group][seed] = sub_df[primary_metric].tolist()

        return primary_values

    def _filter_pairs(self, primary_values):
        """Filter target pairs based on the chosen method."""
        mixes = set(primary_values.keys())
        pairs = list(combinations(mixes, 2))
        target_pairs_kept = []

        for mix1, mix2 in pairs:
            if self.method == "log_fit":
                # Currently default seed 0 unless method == "seed"
                default_seed = 0
                values1 = primary_values[mix1][default_seed]
                values2 = primary_values[mix2][default_seed]

                # Align checkpoints
                common_checkpoints = find_common_checkpoints(values1, values2)
                values1_common = [values1[i] for i in common_checkpoints]
                values2_common = [values2[i] for i in common_checkpoints]

                # Ian's log fit and pseudo errors
                mean_fit1, std1 = self.fit_method.fit(values1_common)
                mean_fit2, std2 = self.fit_method.fit(values2_common)

                # Use the target_keep_fn from the fit method
                if self.target_keep_fn(mean_fit1[-1], mean_fit2[-1], std1[-1], std2[-1]):
                    target_pairs_kept.append((mix1, mix2))

            elif self.method == "seed":
                # Check if the condition is true for all seeds
                all_seeds_satisfy = all(
                    primary_values[mix1][seed][-1] > primary_values[mix2][seed][-1] 
                    for seed in primary_values[mix1].keys() if seed in primary_values[mix2]
                ) or all(
                    primary_values[mix1][seed][-1] < primary_values[mix2][seed][-1] 
                    for seed in primary_values[mix1].keys() if seed in primary_values[mix2]
                )

                if all_seeds_satisfy:
                    target_pairs_kept.append((mix1, mix2))

            elif self.method is None:
                target_pairs_kept.append((mix1, mix2))

            else:
                raise NotImplementedError(f"Filtering method '{self.method}' not implemented yet")

        logging.info(f"Target pairs kept: {len(target_pairs_kept)}/{len(pairs)} ({len(mixes)} choose 2)")
        return target_pairs_kept

    def apply(self, df, primary_metric="primary_metric") -> list[tuple]:
        """Main function to apply the filtering process"""
        # # TODO: Will need fix later
        # if self.method != "seed":
        #     assert 0 in df["seed"].unique()
        #     df = df[df["seed"] == 0]
        # Get primary values for the target model
        primary_values = self._get_primary_values(df, primary_metric)
        # Filter pairs based on the chosen method
        return self._filter_pairs(primary_values)
