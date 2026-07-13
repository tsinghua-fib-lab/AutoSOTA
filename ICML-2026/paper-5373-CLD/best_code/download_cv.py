import argparse
import os
import re
import shutil
import tarfile
import tempfile

import requests
from dotenv import load_dotenv

BASE_URL = "https://datacollective.mozillafoundation.org/api/datasets/{dataset_id}/download"

def process_dataset(api_key, dataset_id, language_id, base_output_dir):
    output_dir = os.path.join(base_output_dir, language_id)
    
    # Step 1: Create download session
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(BASE_URL.format(dataset_id=dataset_id), headers=headers)
    response.raise_for_status()
    data = response.json()
    download_token = data.get("downloadToken")
    if not download_token:
        raise ValueError("Download token not found in response")

    # Step 2: Construct download URL and download the file
    download_url = f"{BASE_URL.format(dataset_id=dataset_id)}/{download_token}"
    response = requests.get(download_url, headers={"Authorization": f"Bearer {api_key}"}, stream=True)
    response.raise_for_status()

    # Get filename from Content-Disposition or default
    disposition = response.headers.get("content-disposition")
    if disposition:
        filename_match = re.findall(r'filename="?(.+?)"?', disposition)
        filename = filename_match[0] if filename_match else "dataset.tar.gz"
    else:
        filename = "dataset.tar.gz"

    # Save the file
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Extract and move contents
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(temp_dir)

        # Assume single top-level directory in the archive
        extracted_items = os.listdir(temp_dir)
        if len(extracted_items) != 1 or not os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
            raise ValueError("Unexpected archive structure: expected single top-level directory")

        top_level_dir = os.path.join(temp_dir, extracted_items[0])

        # Assume single inner directory
        inner_dirs = [item for item in os.listdir(top_level_dir) if os.path.isdir(os.path.join(top_level_dir, item))]
        if len(inner_dirs) != 1:
            raise ValueError("Unexpected inner structure: expected exactly one inner directory")



        source_dir = os.path.join(top_level_dir, inner_dirs[0])

        # Do not create output_dir here to allow direct move/rename
        # Assume base_output_dir exists; move will create the language_id dir
        shutil.move(source_dir, output_dir)

    # Cleanup downloaded file
    os.remove(filename)

    print(f"Dataset {dataset_id} downloaded, extracted, and contents moved to {output_dir}")

def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in .env file")

    parser = argparse.ArgumentParser(description="Process multiple datasets from a txt file")
    parser.add_argument("input_file", help="Path to txt file with lines like dataset_id,language_id")
    parser.add_argument("base_output_dir", help="Base output directory where language_id subdirs will be created")
    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)

    with open(args.input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue
            dataset_id = parts[0].strip()
            language_id = parts[1].strip()
            try:
                process_dataset(api_key, dataset_id, language_id, args.base_output_dir)
            except Exception as e:
                print(f"Error processing {dataset_id}: {str(e)}")

if __name__ == "__main__":
    main()