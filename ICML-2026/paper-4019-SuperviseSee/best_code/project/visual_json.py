import os
import argparse
import multiprocessing as mp
from utils import save_instance
from tqdm import tqdm

def _map(job):

    idx_png, input_dir = job
    idx = os.path.splitext(os.path.basename(idx_png))[0]  # '1234'
    subfolder = os.path.join(input_dir, idx)
    if not os.path.isdir(subfolder):
        return 

    soft_json = os.path.join(subfolder, "soft_masks.json")
    if os.path.exists(soft_json):
        try:
            save_instance(soft_json, subfolder, "init")
        except Exception:
            pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Process subfolders of RLE JSONs and save masks within each subfolder.'
    )
    parser.add_argument(
        '--input_dir', required=True,
        help='Path to the folder containing subfolders with instances_init.json and instances_refine.json'
    )
    parser.add_argument(
        "--index_file", required=True,
        help="Text file listing image indices ending with .png (one per line)"
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of worker processes'
    )
    args = parser.parse_args()

    with open(args.index_file, "r") as f:
        indices = [ln.strip() for ln in f if ln.strip()]
    jobs = [(fname, args.input_dir) for fname in indices]

    cpus = args.num_workers

    with mp.Pool(processes=cpus) as pool:
        for _ in tqdm(pool.imap_unordered(_map, jobs), total=len(jobs)):
            pass 
    
