import os
import argparse
import json
import tempfile
from huggingface_hub import HfApi, Repository
from pprint import pprint
import tqdm
from transformers import AutoTokenizer
from hf_olmo import OLMoForCausalLM

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_FILE = os.path.join(SCRIPT_DIR, "../checkpoints/weka_paths.jsonl")

SEED_MAPPING = {
        6198: "default",
        14: "small aux 2",
        15: "small aux 3",
        4: "large aux 2",
        5: "large aux 3"
    }

WEKA_PATH = "/data/input/"

# Mapping for model names to repo names
recipe_display_name = {
    "dolma17": "dolma1_7",
    "no_code": "dolma1_7-no-code", 
    "no_math_no_code": "dolma1_7-no-math-code",
    "no_reddit": "dolma1_7-no-reddit",
    "no_flan": "dolma1_7-no-flan",
    "dolma-v1-6-and-sources-baseline": "dolma1_6plus",
    "c4": "c4",
    "prox_fineweb_pro": "fineweb-pro",
    "fineweb_edu_dedup": "fineweb-edu", 
    "falcon": "falcon",
    "falcon_and_cc": "falcon-and-cc",
    "falcon_and_cc_eli5_oh_top10p": "falcon-and-cc-qc-10p",
    "falcon_and_cc_eli5_oh_top20p": "falcon-and-cc-qc-20p",
    "falcon_and_cc_og_eli5_oh_top10p": "falcon-and-cc-qc-orig-10p",
    "falcon_and_cc_tulu_qc_top10": "falcon-and-cc-qc-tulu-10p",
    "DCLM-baseline": "dclm-baseline",
    "dolma17-75p-DCLM-baseline-25p": "dclm-baseline-25p-dolma1.7-75p",
    "dolma17-50p-DCLM-baseline-50p": "dclm-baseline-50p-dolma1.7-50p", 
    "dolma17-25p-DCLM-baseline-75p": "dclm-baseline-75p-dolma1.7-25p",
    "dclm_ft7percentile_fw2": "dclm-baseline-qc-7p-fw2",
    "dclm_ft7percentile_fw3": "dclm-baseline-qc-7p-fw3",
    "dclm_fw_top10": "dclm-baseline-qc-fw-10p",
    "dclm_fw_top3": "dclm-baseline-qc-fw-3p",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "dclm-baseline-qc-10p",
    "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "dclm-baseline-qc-20p",
}

def model_name_to_repo_name(model_name):
    seed = int(model_name.split('-')[-1])
    assert seed in [6198, 4, 5, 14, 15], f"Unknown seed {seed}"
    size = model_name.split('-')[-2]
    assert size.endswith('M') or size.endswith('B'), f"Unknown size {size}"
    recipe = '-'.join(model_name.split('-')[:-2])
    if recipe == 'baseline':
        recipe = 'dolma17'
    if recipe in ['DCLM-baseline-25p', 'DCLM-baseline-50p', 'DCLM-baseline-75p']:
        return None
    assert recipe in recipe_display_name, f"Unknown recipe {recipe}"
    recipe = recipe_display_name[recipe]\
    
    return f"DataDecide-{recipe}-{size}"

def get_checkpoints(repo_name):
    """Find all intermediate checkpoints for the given repo name."""
    checkpoints = []

    with open(CHECKPOINTS_FILE, "r") as file:
        for line in file:
            obj = json.loads(line)
            model_name = obj["model_name"]
            model_name = model_name_to_repo_name(model_name)
            if model_name is None:
                continue

            if model_name.startswith(repo_name):
                checkpoints.append(obj)


    seeds = [int(model['model_name'].split('-')[-1]) for model in checkpoints]
    assert all(seed in SEED_MAPPING for seed in seeds), f"Unknown seed {seeds}"
    if len(checkpoints) == 5:
        assert repo_name.split('-')[-1] == "1B", f"1B model should have 5 checkpoints, got {len(checkpoints)}"
    else:
        assert len(checkpoints) == 3, f"Expected 3 or 5 checkpoints, got {len(checkpoints)}"

    return checkpoints

def upload_checkpoint(repo, checkpoint_path, branch_name, temp_dir):
    """Upload a checkpoint to the Hugging Face Hub."""

    # Push to the specified branch
    repo.git_checkout(branch_name, create_branch_ok=True)
    # Load the model and tokenizer
    model = OLMoForCausalLM.from_pretrained(checkpoint_path)
    # Save model and tokenizer to the temporary directory
    model.save_pretrained(temp_dir)
    repo.push_to_hub(commit_message=f"Pushing model to {branch_name} branch")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Upload checkpoints to a Hugging Face repository.")
    parser.add_argument("--repo_name", type=str, required=True, help="The Hugging Face repo name (e.g., DataDecide-falcon-and-cc-qc-tulu-10p-60M)")
    parser.add_argument("--org_name", type=str, default="allenai", help="The organization name on Hugging Face Hub.")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set as an environment variable
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."

    # Get checkpoints
    repo_name = args.repo_name
    checkpoints = get_checkpoints(repo_name)
    repo_name = f"{args.org_name}/{repo_name}"

    # Initialize Hugging Face API
    api = HfApi()

    # Check if the repository exists
    try:
        api.repo_info(repo_id=repo_name)
        print(f"Repository '{repo_name}' exists.")
    except Exception as e:
        raise ValueError(f"Repository '{repo_name}' does not exist or is inaccessible. Please create it first.") from e

    # Create a persistent temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize the repository
        repo = Repository(local_dir=temp_dir, clone_from=repo_name, token=hf_token)
        # Get the list of branches from the repository
        git_refs = api.list_repo_refs(repo_id=repo_name)
        branches = [branch.name for branch in git_refs.branches]
        # Upload checkpoints
        for checkpoint in checkpoints:
            seed = int(checkpoint["model_name"].split("-")[-1])
            seed = SEED_MAPPING[seed].replace(" ", "-")
            progress_bar = tqdm.tqdm(total=len(checkpoint["revisions"]), desc="Uploading checkpoints")
            location = checkpoint['checkpoints_location'].replace("weka://oe-eval-default/", WEKA_PATH)
            for step in checkpoint["revisions"]:
                branch_name = f"{step.replace('-unsharded-hf','')}-seed-{seed}"
                if branch_name in branches:
                    print(f"Branch {branch_name} already exists, skipping upload.")
                    progress_bar.update(1)
                    continue
                progress_bar.set_postfix_str(branch_name)
                progress_bar.update(1)
                checkpoint_path = os.path.join(location, step)
                upload_checkpoint(repo, checkpoint_path, branch_name, temp_dir)
            progress_bar.close()

if __name__ == "__main__":
    main()