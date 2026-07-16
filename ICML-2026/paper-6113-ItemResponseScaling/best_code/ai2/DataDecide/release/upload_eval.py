import os
import argparse
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi

def process_and_upload_results(csv_path, repo_id, token, private=True):
    """
    Processes a CSV file and uploads the results as a dataset to the Hugging Face Hub.

    Args:
        csv_path (str): Path to the CSV file containing the results.
        repo_id (str): Repository ID on Hugging Face Hub (e.g., "organization/repo_name").
        token (str): Hugging Face authentication token.
        private (bool): Whether the repository should be private. Defaults to True.
        branch (str): Branch to upload the dataset to. Defaults to "main".
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Rename columns
    df.rename(columns={"model": "params", "group": "data"}, inplace=True)

    # Map data names
    DATA_NAME_CLEAN = {
        "dolma17": "Dolma1.7",
        "no_code": "Dolma1.7 (no code)", 
        "no_math_no_code": "Dolma1.7 (no math, code)",
        "no_reddit": "Dolma1.7 (no Reddit)",
        "no_flan": "Dolma1.7 (no Flan)",
        "dolma-v1-6-and-sources-baseline": "Dolma1.6++",
        "c4": "C4",
        "prox_fineweb_pro": "FineWeb-Pro",
        "fineweb_edu_dedup": "FineWeb-Edu", 
        "falcon": "Falcon",
        "falcon_and_cc": "Falcon+CC",
        "falcon_and_cc_eli5_oh_top10p": "Falcon+CC (QC 10%)",
        "falcon_and_cc_eli5_oh_top20p": "Falcon+CC (QC 20%)",
        "falcon_and_cc_og_eli5_oh_top10p": "Falcon+CC (QC Orig 10%)",
        "falcon_and_cc_tulu_qc_top10": "Falcon+CC (QC Tulu 10%)",
        "DCLM-baseline": "DCLM-Baseline",
        "dolma17-75p-DCLM-baseline-25p": "DCLM-Baseline 25% / Dolma 75%",
        "dolma17-50p-DCLM-baseline-50p": "DCLM-Baseline 50% / Dolma 50%", 
        "dolma17-25p-DCLM-baseline-75p": "DCLM-Baseline 75% / Dolma 25%",
        "dclm_ft7percentile_fw2": "DCLM-Baseline (QC 7%, FW2)",
        "dclm_ft7percentile_fw3": "DCLM-Baseline (QC 7%, FW3)",
        "dclm_fw_top10": "DCLM-Baseline (QC FW 10%)",
        "dclm_fw_top3": "DCLM-Baseline (QC FW 3%)",
        "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "DCLM-Baseline (QC 10%)",
        "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "DCLM-Baseline (QC 20%)",
    }
    df['data'] = df['data'].map(DATA_NAME_CLEAN)

    # Map seed values
    SEED_MAPPING = {
        6198: "default",
        14: "small aux 2",
        15: "small aux 3",
        4: "large aux 2",
        5: "large aux 3"
    }
    df['seed'] = df['seed'].map(SEED_MAPPING)

    # Convert the dataframe to a Hugging Face dataset
    hf_dataset = Dataset.from_pandas(df)

    # Push the dataset to the Hugging Face Hub
    hf_dataset.push_to_hub(repo_id, private=private, token=token)
    print(f"Successfully uploaded dataset to {repo_id}.")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process and upload results to the Hugging Face Hub.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../results/processed_data/results_ladder_5xC_seeds_cleaned_correct_params.csv",
        help="Path to the CSV file containing the results. Default is '../results/processed_data/results_ladder_5xC_seeds_cleaned_correct_params.csv'."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="allenai/DataDecide-eval-results",
        help="Repository ID on Hugging Face Hub (e.g., 'organization/repo_name'). Default is 'allenai/DataDecide-eval-results'."
    )
    parser.add_argument("--private", action="store_true", help="Whether the repository should be private.")

    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set as an environment variable
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."

    process_and_upload_results(
        csv_path=args.csv_path,
        repo_id=args.repo_id,
        token=hf_token,
        private=args.private,
    )
