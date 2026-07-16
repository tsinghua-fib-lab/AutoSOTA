from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yuhengtu/irsl_datadecide_data",
    repo_type="dataset",
    local_dir="data",
)

snapshot_download(
    repo_id="yuhengtu/irsl_datadecide_results",
    repo_type="dataset",
    local_dir="results",
)

print(f"Downloaded `yuhengtu/irsl_datadecide_data` into `./data`")
print(f"Downloaded `yuhengtu/irsl_datadecide_results` into `./results`")