import sys, os
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi, login, hf_hub_download
from utils.constants import DATA_DIR

def convert_csv_to_parquet(csv_file_path):
    parquet_file_path = csv_file_path.replace(".csv", ".parquet")
    print(f"Converting '{csv_file_path}' -> '{parquet_file_path}'")
    df = pd.read_csv(csv_file_path, encoding='utf-8')

    # Remove fake added index
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')
    
    df.to_parquet(parquet_file_path, index=False)
    return parquet_file_path


def push_parquet_to_hf(parquet_file_path, hf_dataset_name, split_name='main', private=True, overwrite=False):
    """ Push a .parquet file to HuggingFace """
    if not isinstance(parquet_file_path, str):
        parquet_file_path = str(parquet_file_path)
    
    if parquet_file_path.endswith(".csv"):
        parquet_file_path = convert_csv_to_parquet(parquet_file_path)
        parquet_file_path = parquet_file_path.replace('.csv', '.parquet')

    file_name = os.path.basename(parquet_file_path)

    import pyarrow.parquet as pq
    print('Loading sanity check...')
    df = pq.read_table(parquet_file_path).slice(0, 100).to_pandas()
    pd.set_option('display.max_columns', None)
    print(df)

    login(new_session=False)
    api = HfApi()
    
    # Check if the repo exists; create it if not
    try:
        api.repo_info(repo_id=hf_dataset_name, repo_type="dataset")
    except Exception as e:
        api.create_repo(repo_id=hf_dataset_name, private=private, repo_type="dataset", exist_ok=True)

    # Determine the target file path in the repository
    path_in_repo = os.path.join('data', f'{split_name}-00000-of-00001.parquet') # https://huggingface.co/docs/hub/en/datasets-file-names-and-splits

    # Check if the file exists in the repository
    repo_files = api.list_repo_files(repo_id=hf_dataset_name, repo_type="dataset")
    if path_in_repo in repo_files:
        if not overwrite:
            print(f"File '{path_in_repo}' already exists in '{hf_dataset_name}'. Skipping upload.")
            return
        print(f"File '{path_in_repo}' exists and will be overwritten.")

    print(f"Uploading '{parquet_file_path}' -> '{path_in_repo}' to hf dataset '{hf_dataset_name}'")

    # Upload the file to the repository
    api.upload_file(
        path_or_fileobj=parquet_file_path,
        path_in_repo=path_in_repo,
        repo_id=hf_dataset_name,
        repo_type="dataset"
    )
    print(f"File '{path_in_repo}' uploaded to '{hf_dataset_name}'.")


def download_parquet_from_hf(hf_dataset_name, file_name, local_path):
    print(f'Downloading {file_name} -> {local_path}')
    file_path = hf_hub_download(
        repo_id=hf_dataset_name,
        filename=file_name,
        repo_type="dataset",
        local_dir=local_path
    )
    return file_path


def pull_predictions_from_hf(repo_id, split_name, local_path=DATA_DIR):
    """ Pull predictions files from Huggingface, merge them and return the merged file path """
    output_dir = os.path.join(local_path, 'hf', repo_id.replace('/', '_'))
    output_path = os.path.join(output_dir, f'{split_name}.parquet')

    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping download")
        return output_path

    file_pattern = f'data/{split_name}-*.parquet'
    api = HfApi()
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.startswith(f'data/{split_name}-')]
    
    if not files:
        raise ValueError(f"No parquet files found matching pattern {file_pattern}")
        
    dfs = []
    for file_name in files:
        file_path = download_parquet_from_hf(repo_id, file_name, output_dir)
        df = pd.read_parquet(file_path)
        dfs.append(df)
        
    # Merge all DFs (0000X-of-0000X.parquet)
    df = pd.concat(dfs, ignore_index=True)

    # Expand metrics json string into columns
    if 'metrics' in df.columns:
        print('Expanding metrics col...')

        metrics_dict = []
        for metric in tqdm(df['metrics'], total=len(df)):
            try:
                metrics_dict.append(eval(metric))
            except Exception as e:
                print(f"Failed to parse metrics entry: {metric}")
                print(f"Error: {e}")
                raise
        metrics_dict = pd.Series(metrics_dict)
        metrics_df = pd.json_normalize(metrics_dict)
        df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    
    return output_path


def main():
    org_name = 'allenai'

    push_parquet_to_hf(
        parquet_file_path='data/hf/consistent_ranking_metrics.parquet',
        hf_dataset_name=f'{org_name}/DataDecide-eval-results',
        split_name='benchmarks',
        overwrite=True,
        private=True,
    )


if __name__ == '__main__': main()