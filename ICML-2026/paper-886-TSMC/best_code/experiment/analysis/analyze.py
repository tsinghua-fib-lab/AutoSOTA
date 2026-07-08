from __future__ import annotations
import os
from dataclasses import dataclass, replace

import tqdm

import pandas as pd

from experiment.analysis import export
from experiment.analysis import utils


@dataclass
class DataManipulator:
    # Utility class for manipulating data for visualization
    projects: list[str]
    envs: list[str]

    configs: dict[str, pd.DataFrame]
    data: dict[str, pd.DataFrame]

    @property
    def codes(self) -> list[str]:
        return [f'{p}/{e}' for p in self.projects for e in self.envs]

    @classmethod
    def create(cls, path: str, entity: str, projects: list[str]):
        # Load in configs annotated as 'project/env_code'
        configs = {
            p: {
                f.removesuffix('.csv'): pd.read_csv(
                    os.path.join(path, export.CACHE_PATH, entity, p, f))
                for f in
                tqdm.tqdm(os.listdir(os.path.join(path, 'cache', entity, p)),
                          desc=f'Loading configs {p}') if f.endswith('.csv')
            }
            for p in projects
        }
        flat_configs = utils.flatten_dict(configs, separator='/')

        dataframes = {
            k: {r: pd.read_csv(
                os.path.join(path, export.DATA_PATH, r + '.csv'), index_col=0)
                for r in v.run}
            for k, v in
            tqdm.tqdm(flat_configs.items(), desc='Loading dataframes')
        }
        dataframes = {k: pd.concat(v, names=['run', 'index'])
                      for k, v in dataframes.items()}

        # Store 'env_codes' for utility
        env_codes = list({k.split('/')[-1] for k in flat_configs.keys()})

        return cls(projects, env_codes, flat_configs, dataframes)

    def select(
            self, projects: list[str] | None, envs: list[str] | None
    ) -> DataManipulator:
        # Extract specific projects and environments

        projects = self.projects if projects is None else projects
        envs = self.envs if envs is None else envs

        cfgs, dfs = {}, {}
        for p in projects:
            for e in envs:
                c = f'{p}/{e}'

                cfgs[c] = self.configs[c]
                dfs[c] = self.data[c]

        return DataManipulator(projects, envs, cfgs, dfs)

    def filter(self, spec: list[tuple[str, set]]) -> DataManipulator:
        # Filter out runs based on config value

        outer_cfg, outer_data = {}, {}
        for code in self.codes:

            cfg = self.configs[code]
            dfs = self.data[code]

            for key, inclusion_set in spec:
                if key in cfg.columns:
                    # Get mask from filter
                    mask = cfg[key].isin(inclusion_set)

                    # Apply filter
                    cfg = cfg[mask]
                    dfs = dfs.loc[cfg.run, :]

            outer_cfg[code] = cfg
            outer_data[code] = dfs

        return replace(self, configs=outer_cfg, data=outer_data)

    def cap_seeds(
            self,
            spec: list[tuple[str, set]],
            max_num_seeds: int,
    ) -> "DataManipulator":
        """
        Caps the number of 'runs' (seeds) for each unique combination of
        specified configuration keys within the DataManipulator's datasets.

        For each unique combination of values across the specified 'spec' keys,
        this function ensures that no more than 'max_num_seeds' runs are retained.
        Rows that do not match the values in 'spec' are initially filtered out.

        Args:
            spec: A list of tuples, where each tuple contains a configuration key (str)
                  and a set of allowed values for that key. Only runs where the
                  config values for these keys are within the specified sets
                  will be considered for capping.
            max_num_seeds: The maximum number of runs to keep for each unique
                           combination of the 'spec' keys.

        Returns:
            A new DataManipulator instance with the capped configurations and data.
        """
        capped_cfg, capped_data = {}, {}

        for code in self.codes:
            # Get the configuration and data DataFrames for the current code
            cfg = self.configs[code].copy() # Work on a copy to avoid modifying original
            dfs = self.data[code].copy()   # Work on a copy

            # 1. Initial Filtering: Keep only rows where config values match any of the spec values
            # Create a boolean mask, initially True for all rows
            mask = pd.Series(True, index=cfg.index)
            for key, values in spec:
                if key in cfg.columns:
                    # If the key exists in the config, update the mask
                    # to include only rows where the key's value is in the 'values' set
                    mask &= cfg[key].isin(values)
                else:
                    # Optional: Print a warning if a spec key is not found in the config.
                    # The current behavior correctly ignores it for filtering and grouping.
                    print(f"Warning: Spec key '{key}' from 'spec' not found in config for code '{code}'. "
                          "It will not be used for filtering or grouping.")

            filtered_cfg = cfg[mask]

            if filtered_cfg.empty:
                # If no runs remain after initial filtering, skip this code
                # and ensure empty DataFrames are assigned to maintain structure.
                capped_cfg[code] = pd.DataFrame(columns=cfg.columns)
                # Create an empty DataFrame for data with the same structure (e.g., columns, index levels)
                # by slicing with 0 rows
                capped_data[code] = dfs.iloc[0:0]
                continue

            # 2. Identify Grouping Keys:
            # Only group by keys from 'spec' that are actually present in the filtered config's columns.
            group_keys = [key for key, _ in spec if key in filtered_cfg.columns]

            if not group_keys:
                # If no relevant grouping keys are found from 'spec' within this config,
                # treat the entire filtered_cfg as a single group for capping.
                print(f"Warning: No valid grouping keys found from 'spec' in config for code '{code}'. "
                      f"Capping will apply to the entire filtered dataset for this code.")
                # Create a pseudo-group to allow uniform processing
                grouped = [('no_group_key', filtered_cfg)]
            else:
                # Group the filtered configuration DataFrame by the identified keys
                grouped = filtered_cfg.groupby(group_keys)

            selected_rows = []

            # 3. Sample from each group:
            for group_name, group in grouped:
                # Add a print statement to show group size before sampling
                print(f"Code: {code}, Group: {group_name}, Original Group Size: {len(group)}, Capping to: {min(max_num_seeds, len(group))} runs.")
                # Select up to max_num_seeds rows from the current group.
                # If the group has fewer than max_num_seeds, all rows are selected.
                # 'random_state=0' ensures reproducibility of the sampling.
                selected_rows.append(group.sample(n=min(max_num_seeds, len(group)), random_state=0))

            # Concatenate all selected rows from different groups into a single DataFrame
            capped_cfg_df = pd.concat(selected_rows)

            # Get the unique 'run' IDs from the capped configuration DataFrame.
            # These are the runs that should be retained in the data DataFrame.
            if 'run' not in capped_cfg_df.columns:
                print(f"Error: 'run' column not found in the capped config for code '{code}'. "
                      "Cannot link data to configurations. Data will be empty for this code.")
                capped_data[code] = dfs.iloc[0:0] # Assign empty DataFrame if 'run' is missing
            else:
                capped_runs = capped_cfg_df["run"].unique()

                # Filter the original data DataFrame to only include these selected runs.
                # This relies on 'dfs' having 'run' as the first level of its MultiIndex.
                # `get_level_values('run')` is a robust way to access values from a specific level.
                dfs_filtered = dfs[dfs.index.get_level_values('run').isin(capped_runs)]
                capped_data[code] = dfs_filtered

            # Store the capped configuration and data for the current code
            capped_cfg[code] = capped_cfg_df

        # Return a new DataManipulator instance with the modified configurations and data
        return replace(self, configs=capped_cfg, data=capped_data)


@dataclass
class Group:
    groups: list[str | None]
    values: list[pd.DataFrame]
    data: list[pd.DataFrame]

    names: list[str | None]
    aggregated: list[str]

    aggr_data: pd.DataFrame | None = None

    @classmethod
    def create(cls, config, data, spec: list[str]):
        cfg = config.columns.tolist()

        group_keys = utils.ListAsSet.intersect(cfg, spec)
        aggr_keys = utils.ListAsSet.diff(cfg, spec)

        if not group_keys:
            return cls([None, ], [config], [data], [None], aggr_keys)

        values, subconfigs, subdata = [], [], []
        for g, sc in config.groupby(group_keys):
            values.append(g)
            subconfigs.append(sc)
            subdata.append(data.loc[sc.run, :])
            # subdata.append(data.iloc[sc.index, :])

        return cls(values, subconfigs, subdata, group_keys, aggr_keys)

    def aggregate(self, x: str, y: str, method):

        ref = self.data[0].loc[:, x]
        ref_idx = ref.index.get_level_values(0)[0]
        ref = ref.loc[ref_idx].values

        aggr = {g: pd.DataFrame(
            {'x': ref} | method(df.loc[:, y].unstack(level=0).values)) for
            g, df in zip(self.groups, self.data)}

        return replace(
            self,
            aggr_data=pd.concat(aggr, names=self.names)  # type: ignore
        )
