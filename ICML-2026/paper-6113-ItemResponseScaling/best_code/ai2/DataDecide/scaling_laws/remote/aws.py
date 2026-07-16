import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import boto3

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import DATA_DIR

EXCLUDED_FILE_NAMES = [
    'requests.jsonl',
    'recorded-inputs.jsonl',
    'metrics-all.jsonl'
]


def download_file(s3_client, bucket_name, key, local_dir, excluded_file_names):
    local_path = os.path.join(local_dir, key)

    if any(f in key.split('/')[-1] for f in excluded_file_names):
        return # Skip download if there are any str matches with EXCLUDED_FILE_NAMES
    
    if os.path.exists(local_path):
        s3_head = s3_client.head_object(Bucket=bucket_name, Key=key)
        s3_file_size = s3_head['ContentLength']
        local_file_size = os.path.getsize(local_path)
        if s3_file_size == local_file_size:
            return  # Skip download if the file already exists and has the same size

    # Manual override for some AWS paths
    local_path.replace('all_olmes_rc_tasks/', '')
    local_path.replace('all_olmes_paraphrase_tasks/', '')

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket_name, key, local_path)


def fetch_page(page):
    return [obj['Key'] for obj in page.get('Contents', [])]


def fetch_keys_for_prefix(bucket_name, prefix):
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    for page in pages:
        keys.extend(fetch_page(page))
    return keys


def mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100, excluded_file_names=EXCLUDED_FILE_NAMES, s3_prefix_list=None):
    """ Recursively download an S3 folder to mirror remote """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    keys = []

    if s3_prefix_list is not None:
        with open(s3_prefix_list, 'r') as file:
            s3_prefixes = [line.strip() for line in file.readlines() if line.strip()]
    elif not isinstance(s3_prefix, list):
        s3_prefixes = [s3_prefix]
    else:
        s3_prefixes = s3_prefix

    print(f'Searching through S3 prefixes: {s3_prefixes}')

    with ThreadPoolExecutor(max_workers=100) as executor:
        if s3_prefix_list is not None or len(s3_prefix) > 1:
            future_to_prefix = {}
            with tqdm(total=len(s3_prefixes), desc="Submitting S3 prefix tasks", unit="prefix") as submit_pbar:
                for prefix in s3_prefixes:
                    future = executor.submit(fetch_keys_for_prefix, bucket_name, prefix)
                    future_to_prefix[future] = prefix
                    submit_pbar.update(1)

            with tqdm(total=len(future_to_prefix), desc="Fetching keys from S3 prefixes", unit="prefix") as pbar:
                for future in as_completed(future_to_prefix):
                    try:
                        keys.extend(future.result())
                    except Exception as e:
                        print(f"Error processing prefix {future_to_prefix[future]}: {e}")
                    pbar.update(1)
        else:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
            keys = []
            for result in tqdm(executor.map(fetch_page, pages), desc="Listing S3 entries"):
                keys.extend(result)

            for s3_prefix in tqdm(s3_prefixes, desc="Processing S3 prefixes", unit="prefix"):
                pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)
                for result in executor.map(fetch_page, pages):
                    keys.extend(result)

    if max_threads > 1:
        # ProcessPoolExecutor seems not to work with AWS, so we use ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(download_file, s3_client, bucket_name, key, local_dir, excluded_file_names): key for key in keys}
            with tqdm(total=len(futures), desc="Syncing download folder from S3", unit="file") as pbar:
                for _ in as_completed(futures):
                    pbar.update(1)
    else:
        # Sequential implmentation
        for key in tqdm(keys, desc="Syncing download folder from S3", unit="file"):
            download_file(s3_client, bucket_name, key, local_dir, excluded_file_names)


def main():
    """ Mirror AWS bucket to a local folder """
    # Get requests for BPB calculation
    # bucket_name = 'ai2-llm'
    # s3_prefix = 'eval-results/downstream/eval-for-consistent-ranking/baseline-150M-5xC-2/step38157-unsharded-hf/'
    # folder_name = 'consistent_ranking'
    # local_dir = f'{DATA_DIR}/{folder_name}'
    # mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100, excluded_file_names=[])

    bucket_name = 'ai2-llm'
    s3_prefix = [
        'eval-results/downstream/eval-for-consistent-ranking/', 
        'eval-results/downstream/eval-for-consistent-ranking-small/', 
        'eval-results/downstream/eval-for-consistent-ranking-small-seeds-extra/']
    folder_name = 'consistent_ranking'

    local_dir = f'{DATA_DIR}/{folder_name}'

    mirror_s3_to_local(bucket_name, s3_prefix, local_dir, max_threads=100)

    # Pre-process result folder into .parquet file
    from preprocess import main
    main(folder_name, file_type='metrics')
    main(folder_name, file_type='predictions')
    main(folder_name, file_type='questions')

    # Push .parquet file to AWS
    from hf import main
    main()

if __name__ == '__main__':
    main()
