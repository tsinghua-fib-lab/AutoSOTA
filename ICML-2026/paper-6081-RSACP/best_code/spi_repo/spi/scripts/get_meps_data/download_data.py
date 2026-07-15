#BSD
#!/usr/bin/env python3

import os
import sys
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

try:
    import pyreadstat
except ImportError:
    print("Installing pyreadstat...")
    os.system('pip install pyreadstat')
    import pyreadstat

# Usage note and user acknowledgment
usage_note = """
By using this script you acknowledge the responsibility for reading and
abiding by any copyright/usage rules and restrictions as stated on the
MEPS web site (https://meps.ahrq.gov/data_stats/data_use.jsp).

Continue [y/n]? > """
answer = input(usage_note).strip().lower()
if answer != 'y':
    sys.exit("Aborted by user.")

def convert(ssp_file, csv_file):
    print(f"Loading dataframe from file: {ssp_file}")
    df, meta = pyreadstat.read_xport(ssp_file)
    print(f"Exporting dataframe to file: {csv_file}")
    df.to_csv(csv_file, index=False)

datasets = ["h181", "h192"]
base_url = "https://meps.ahrq.gov/mepsweb/data_files/pufs"

for dataset in datasets:
    zip_file = f"{dataset}ssp.zip"
    ssp_file = f"{dataset}.ssp"
    csv_file = f"{dataset}.csv"
    url = f"{base_url}/{zip_file}"

    # Skip if CSV already exists
    if Path(csv_file).exists():
        print(f"{csv_file} already exists")
        continue

    # Download the zip if not already downloaded
    if not Path(zip_file).exists():
        print(f"Downloading {zip_file} from {url}")
        urllib.request.urlretrieve(url, zip_file)

    # Unzip and convert
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    convert(ssp_file, csv_file)

    # Clean up if conversion succeeded
    if Path(csv_file).exists():
        os.remove(zip_file)
        os.remove(ssp_file)
