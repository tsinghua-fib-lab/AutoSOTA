from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    folder_path=".",
    repo_id="yuhengtu/irsl_datadecide_results",
    repo_type="dataset",
    ignore_patterns=["backup/**"],
)
