import os
import argparse
from huggingface_hub import HfApi

def update_repo_visibility(repo_id, token, visibility="public"):
    """
    Updates the visibility of a repository on Hugging Face Hub.

    Args:
        repo_id (str): Repository ID on Hugging Face Hub (e.g., "username/repo_name").
        token (str): Hugging Face authentication token.
        visibility (str): Visibility of the repository ("public" or "private"). Defaults to "public".
    """
    api = HfApi()
    api.update_repo_visibility(repo_id=repo_id, private=(visibility == "private"), token=token)
    print(f"Repository '{repo_id}' is now {visibility}.")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Update the visibility of a repository on the Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID on Hugging Face Hub (e.g., 'username/repo_name').")
    parser.add_argument("--visibility", type=str, default="public", choices=["public", "private"], help="Visibility of the repository ('public' or 'private'). Default is 'public'.")

    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set as an environment variable
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."

    update_repo_visibility(
        repo_id=args.repo_id,
        token=hf_token,
        visibility=args.visibility,
    )
