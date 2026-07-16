import os
from typing import List
import pandas as pd
import numpy as np
from datasets import load_dataset


def create_or_load_large_language_monkeys_mini_f2f_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_mini_f2f_individual_outcomes_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_mini_f2f_individual_outcomes.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_mini_f2f_individual_outcomes_df_path
    ):
        print(
            f"Creating {large_language_monkeys_mini_f2f_individual_outcomes_df_path} anew..."
        )

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_original_dfs_list = []
        subsets = [
            "MiniF2F-MATH_Llama-3-8B-Instruct",
            "MiniF2F-MATH_Llama-3-70B-Instruct",
        ]
        for subset in subsets:
            benchmark, model = subset.split("_")
            ds = load_dataset("ScalingIntelligence/monkey_business", subset)["test"]
            correct: List[List[bool]] = ds["is_corrects"]
            # Shape: (128, 10000)
            wide_df = pd.DataFrame(
                correct,
                columns=1 + np.arange(10000),
                dtype=np.float16,
            )
            # Convert to floats.
            wide_df = wide_df.astype(np.float16)
            wide_df["Problem Idx"] = ds["orig_dset_idx"]
            df = wide_df.melt(
                id_vars=["Problem Idx"],
                var_name="Attempt Idx",
                value_name="Score",
            )

            df["Benchmark"] = benchmark
            # Convert, e.g., "Pythia-1.4B" to "Pythia 1.4B".
            df["Model"] = model.replace("-", " ")
            large_language_monkeys_original_dfs_list.append(df)

        large_language_monkeys_original_individual_outcomes_df = pd.concat(
            large_language_monkeys_original_dfs_list,
        )
        large_language_monkeys_original_individual_outcomes_df[
            "Attempt Idx"
        ] = pd.to_numeric(
            large_language_monkeys_original_individual_outcomes_df["Attempt Idx"]
        )

        large_language_monkeys_original_individual_outcomes_df.to_parquet(
            large_language_monkeys_mini_f2f_individual_outcomes_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_mini_f2f_individual_outcomes_df_path} to disk."
        )
        del large_language_monkeys_original_individual_outcomes_df

    large_language_monkeys_original_individual_outcomes_df = pd.read_parquet(
        large_language_monkeys_mini_f2f_individual_outcomes_df_path
    )
    print(
        f"Loaded {large_language_monkeys_mini_f2f_individual_outcomes_df_path} with shape: ",
        large_language_monkeys_original_individual_outcomes_df.shape,
    )
    return large_language_monkeys_original_individual_outcomes_df


def create_or_load_large_language_monkeys_code_contests_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_code_contests_individual_outcomes_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_code_contests_individual_outcomes.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_code_contests_individual_outcomes_df_path
    ):
        print(
            f"Creating {large_language_monkeys_code_contests_individual_outcomes_df_path} anew..."
        )

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_original_dfs_list = []
        subsets = [
            "CodeContests_Llama-3-8B",
            "CodeContests_Llama-3-8B-Instruct",
            "CodeContests_Llama-3-70B-Instruct",
            "CodeContests_Gemma-2B",
            "CodeContests_Gemma-7B",
        ]
        for subset in subsets:
            benchmark, model = subset.split("_")
            ds = load_dataset("ScalingIntelligence/monkey_business", subset)["test"]
            correct: List[List[bool]] = ds["is_corrects"]
            # Shape: (128, 10000)
            wide_df = pd.DataFrame(
                correct,
                columns=1 + np.arange(10000),
                dtype=np.float16,
            )
            # Convert to floats.
            wide_df = wide_df.astype(np.float16)
            wide_df["Problem Idx"] = ds["orig_dset_idx"]
            df = wide_df.melt(
                id_vars=["Problem Idx"],
                var_name="Attempt Idx",
                value_name="Score",
            )

            df["Benchmark"] = benchmark
            # Convert, e.g., "Pythia-1.4B" to "Pythia 1.4B".
            df["Model"] = model.replace("-", " ")
            large_language_monkeys_original_dfs_list.append(df)

        large_language_monkeys_original_individual_outcomes_df = pd.concat(
            large_language_monkeys_original_dfs_list,
        )
        large_language_monkeys_original_individual_outcomes_df[
            "Attempt Idx"
        ] = pd.to_numeric(
            large_language_monkeys_original_individual_outcomes_df["Attempt Idx"]
        )

        large_language_monkeys_original_individual_outcomes_df.to_parquet(
            large_language_monkeys_code_contests_individual_outcomes_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_code_contests_individual_outcomes_df_path} to disk."
        )
        del large_language_monkeys_original_individual_outcomes_df

    large_language_monkeys_original_individual_outcomes_df = pd.read_parquet(
        large_language_monkeys_code_contests_individual_outcomes_df_path
    )
    print(
        f"Loaded {large_language_monkeys_code_contests_individual_outcomes_df_path} with shape: ",
        large_language_monkeys_original_individual_outcomes_df.shape,
    )
    return large_language_monkeys_original_individual_outcomes_df


def create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df(
    raw_data_dir=f"{os.getcwd()}/data/raw_data",
    processed_data_dir=f"{os.getcwd()}/data/processed_data",
    refresh: bool = False,
) -> pd.DataFrame:
    large_language_monkeys_pythia_math_individual_outcomes_df_path = os.path.join(
        processed_data_dir,
        "large_language_monkeys_pythia_math_individual_outcomes.parquet",
    )

    if refresh or not os.path.exists(
        large_language_monkeys_pythia_math_individual_outcomes_df_path
    ):
        print(
            f"Creating {large_language_monkeys_pythia_math_individual_outcomes_df_path} anew..."
        )

        os.makedirs(processed_data_dir, exist_ok=True)
        large_language_monkeys_pythia_math_dfs_list = []
        subsets = [
            "MATH_Pythia-70M",
            "MATH_Pythia-160M",
            "MATH_Pythia-410M",
            "MATH_Pythia-1B",
            "MATH_Pythia-2.8B",
            "MATH_Pythia-6.9B",
            "MATH_Pythia-12B",
        ]
        for subset in subsets:
            benchmark, model = subset.split("_")
            ds = load_dataset("ScalingIntelligence/monkey_business", subset)["test"]
            correct: List[List[bool]] = ds["is_corrects"]
            # Shape: (128, 10000)
            wide_df = pd.DataFrame(
                correct,
                columns=1 + np.arange(10000),
                dtype=np.float16,
            )
            # Convert to floats.
            wide_df = wide_df.astype(np.float16)
            wide_df["Problem Idx"] = ds["orig_dset_idx"]
            df = wide_df.melt(
                id_vars=["Problem Idx"],
                var_name="Attempt Idx",
                value_name="Score",
            )

            df["Benchmark"] = benchmark
            # Convert, e.g., "Pythia-1.4B" to "Pythia 1.4B".
            df["Model"] = model.replace("-", " ")
            large_language_monkeys_pythia_math_dfs_list.append(df)

        large_language_monkeys_pythia_math_individual_outcomes_df = pd.concat(
            large_language_monkeys_pythia_math_dfs_list,
        )
        large_language_monkeys_pythia_math_individual_outcomes_df[
            "Attempt Idx"
        ] = pd.to_numeric(
            large_language_monkeys_pythia_math_individual_outcomes_df["Attempt Idx"]
        )

        large_language_monkeys_pythia_math_individual_outcomes_df.to_parquet(
            large_language_monkeys_pythia_math_individual_outcomes_df_path,
            index=False,
        )

        print(
            f"Wrote {large_language_monkeys_pythia_math_individual_outcomes_df_path} to disk."
        )
        del large_language_monkeys_pythia_math_individual_outcomes_df

    large_language_monkeys_pythia_math_individual_outcomes_df = pd.read_parquet(
        large_language_monkeys_pythia_math_individual_outcomes_df_path
    )
    print(
        f"Loaded {large_language_monkeys_pythia_math_individual_outcomes_df_path} with shape: ",
        large_language_monkeys_pythia_math_individual_outcomes_df.shape,
    )
    return large_language_monkeys_pythia_math_individual_outcomes_df

if __name__ == "__main__":
    mini_f2f = create_or_load_large_language_monkeys_mini_f2f_individual_outcomes_df()
    contests = create_or_load_large_language_monkeys_code_contests_individual_outcomes_df()
    pythia_math = create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df()

    # breakpoint()
    # number of unique "Problem Idx"
    # print(pythia_math["Model"].nunique())
    # print(mini_f2f.head())
