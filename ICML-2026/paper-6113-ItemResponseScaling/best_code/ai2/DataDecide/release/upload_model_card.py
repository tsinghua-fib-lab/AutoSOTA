import os
import argparse
from huggingface_hub import HfApi

def upload_model_card(repo_id, token, local_path="model_card.md", repo_path="README.md", branch="main"):
    """
    Uploads a model card to the specified repository on Hugging Face Hub.

    Args:
        repo_id (str): Repository ID on Hugging Face Hub (e.g., "username/repo_name").
        token (str): Hugging Face authentication token.
        local_path (str): Path to the local model card file. Defaults to "model_card.md".
        repo_path (str): Path in the repository where the file will be uploaded. Defaults to "README.md".
        branch (str): Branch to upload the file to. Defaults to "main".
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"The file {local_path} does not exist.")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        token=token,
        revision=branch,
    )
    print(f"Successfully uploaded {local_path} to {repo_id}/{repo_path} on branch {branch}.")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Upload a model card to the Hugging Face Hub.")
    parser.add_argument("--local_path", type=str, default="model_card.md", help="Path to the local model card file. Default is 'model_card.md'.")
    parser.add_argument("--repo_path", type=str, default="README.md", help="Path in the repository where the file will be uploaded. Default is 'README.md'.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID on Hugging Face Hub (e.g., 'username/repo_name').")
    parser.add_argument("--branch", type=str, default="main", help="Branch to upload the file to. Default is 'main'.")

    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set as an environment variable
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."

    upload_model_card(
        repo_id=args.repo_id,
        token=hf_token,
        local_path=args.local_path,
        repo_path=args.repo_path,
        branch=args.branch,
    )