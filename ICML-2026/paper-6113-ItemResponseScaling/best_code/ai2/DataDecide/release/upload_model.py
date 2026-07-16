from hf_olmo import OLMoForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import Repository, HfApi, add_collection_item
import tempfile
import os
import argparse

def upload_model(model_path, repo_name, organization, hf_token, branch="main", private=True, collection_slug=None):
    """
    Uploads a model and tokenizer to the Hugging Face Hub and optionally adds it to a collection.

    Args:
        model_path (str): Path to the model and tokenizer directory.
        repo_name (str): Name of the repository to create or use.
        organization (str): Organization name on Hugging Face Hub.
        hf_token (str): Hugging Face authentication token.
        branch (str): Branch to push the model to. Default is "main".
        private (bool): Whether the repository should be private. Default is True.
        collection_slug (str): Slug of the collection to add the repository to. Default is None.
    """
    # Load the model and tokenizer
    model = OLMoForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create a new repository
    api = HfApi()
    repo_id = f"{organization}/{repo_name}"
    api.create_repo(repo_id=repo_id, private=private, token=hf_token, exist_ok=False)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize the repository
        repo = Repository(local_dir=temp_dir, clone_from=repo_id, token=hf_token)

        if branch != "main":
            # Specify the branch to push to
            repo.git_checkout(branch, create_branch_ok=True)

        # Save model and tokenizer to the temporary directory
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        # Add model card as README.md
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_card_path = os.path.join(script_dir, "model_card.md")
        readme_path = os.path.join(temp_dir, "README.md")
        if os.path.exists(model_card_path):
            with open(model_card_path, "r") as model_card_file:
                with open(readme_path, "w") as readme_file:
                    readme_file.write(model_card_file.read())
        else:
            raise  FileNotFoundError(f"Model card file not found at {model_card_path}. Please provide a valid path.")
        

        # Push to the specified branch
        repo.push_to_hub(commit_message=f"Pushing model and tokenizer to {branch} branch")
        print(f"Model and tokenizer pushed to {repo_id} on branch '{branch}'.")

    # Optionally add the repository to a collection
    if collection_slug:
        add_collection_item(
            collection_slug=collection_slug,
            item_id=repo_id,
            item_type="model",
            token=hf_token
        )
        print(f"Repository '{repo_id}' has been added to the collection '{collection_slug}'.")

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Upload a model and tokenizer to the Hugging Face Hub.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model and tokenizer directory.")
    parser.add_argument("--repo_name", type=str, required=True, help="Name of the repository to create or use.")
    parser.add_argument("--organization", type=str, default="allenai", help="Organization name on Hugging Face Hub.")
    parser.add_argument("--branch", type=str, default="main", help="Branch to push the model to. Default is 'main'.")
    parser.add_argument("--private", action="store_true", help="Whether the repository should be private. Default is True.")
    parser.add_argument("--collection_slug", type=str, default="allenai/datadecide-67edb1d2bacba40b5d3ed633", help="Slug of the collection to add the repository to.")

    args = parser.parse_args()

    args.collection_slug = None if args.collection_slug == "" else args.collection_slug

    hf_token = os.getenv("HF_TOKEN")  # Ensure your Hugging Face token is set as an environment variable
    assert hf_token is not None, "Please set the HF_TOKEN environment variable."

    upload_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        organization=args.organization,
        hf_token=hf_token,
        branch=args.branch,
        private=args.private,
        collection_slug=args.collection_slug,
    )