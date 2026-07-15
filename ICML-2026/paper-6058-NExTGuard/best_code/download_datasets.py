import os
import glob
import yaml
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# ==============================================================================
# 0. Configuration Integration
# ==============================================================================
# Simulate the loading logic of src/config.py, but adapt to the current script location (project root directory)
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"🌲 Loaded environment from: {ENV_PATH}")

DATASET_ROOT = os.getenv("DATASET_ROOT")
HF_TOKEN = os.getenv("HF_TOKEN")

# Default configuration directory
DEFAULT_CONFIG_DIR = os.path.join(BASE_DIR, "configs/datasets")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. Dataset Mapping (Dataset Name -> HF ID)
# ==============================================================================

from huggingface_hub import snapshot_download  # New: for original file download

DATASET_HUB_IDS = {
    "Aegis1.0": "nvidia/Aegis-AI-Content-Safety-Dataset-1.0",
    "Aegis2.0": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
    "ToxicChat": "lmsys/toxic-chat",
    "WildGuardTest": "walledai/WildGuardTest",
    "HarmBench": "walledai/HarmBench",
    "HarmBenchResponse": "walledai/HarmBench",
    "OpenAIMod": "walledai/openai-moderation-dataset",
    "SimpleSafetyTest": "walledai/SimpleSafetyTests",
    "StrongREJECT": "walledai/StrongREJECT",
    "WildJailbreak": "walledai/WildJailbreak",
    "WildGuardMix": "allenai/wildguardmix",
    "XSTestResponse": "allenai/xstest-response",
    "SafeRLHF": "PKU-Alignment/PKU-SafeRLHF",
    "BeaverTails": "PKU-Alignment/BeaverTails"
}

def process_single_config(config_path, data_root, check_mode):
    """Process a single YAML configuration file"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return

    logger.info(f"📄 Processing config: {os.path.basename(config_path)}")
    with open(config_path, 'r', encoding='utf-8') as f:
        # Get the dataset list from the YAML file
        datasets = yaml.safe_load(f).get('datasets', [])

    if not datasets:
        logger.warning(f"   No datasets found in {os.path.basename(config_path)}")
        return

    for item in datasets:
        name, folder = item.get('name'), item.get('folder')
        if not name or not folder: continue
        
        # Path processing
        if os.path.isabs(folder):
            local_path = folder
        else:
            local_path = os.path.join(data_root, folder)
            
        hf_id = DATASET_HUB_IDS.get(name)

        # --- Check mode ---
        if check_mode:
            status, note = "❌ Missing", "Directory not found"
            if os.path.exists(local_path) and os.listdir(local_path):
                # Check if the directory contains files (not checking Arrow format)
                files = os.listdir(local_path)
                status, note = "✅ Ready", f"Files: {len(files)}"
            
            print(f"{name:<25} | {status:<12} | {note:<25} | {local_path}")

        # --- Download mode (using snapshot_download) ---
        else:
            if not hf_id:
                logger.warning(f"   ⚠️  Unknown dataset: {name} (Skipping)")
                continue
            
            # If the directory is not empty, skip the download
            if os.path.exists(local_path) and os.listdir(local_path):
                logger.info(f"   ✅ [{name}] Already exists.")
                continue

            logger.info(f"   ⬇️  [{name}] Downloading snapshot from {hf_id}...")
            try:
                # Use snapshot_download to download the original files
                # repo_type="dataset" ensures the pointer to the dataset repository
                # local_dir_use_symlinks=False downloads the files directly rather than links
                snapshot_download(
                    repo_id=hf_id,
                    repo_type="dataset",
                    local_dir=local_path,
                    token=HF_TOKEN,
                    local_dir_use_symlinks=False
                )
                logger.info(f"   🎉 [{name}] Snapshot saved to {local_path}")
            except Exception as e:
                logger.error(f"   ❌ [{name}] Failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch download datasets using HF Snapshot.")
    parser.add_argument("--config_dir", type=str, default=DEFAULT_CONFIG_DIR, help="Config directory/file")
    parser.add_argument("--data_root", type=str, default=DATASET_ROOT, help="Data root path")
    parser.add_argument("--check", action="store_true", help="Check only")
    args = parser.parse_args()

    if not args.check:
        if not HF_TOKEN:
            logger.warning("⚠️  HF_TOKEN missing in .env. Gated datasets might fail.")
        os.makedirs(args.data_root, exist_ok=True)

    config_files = []
    if os.path.isfile(args.config_dir):
        config_files = [args.config_dir]
    elif os.path.isdir(args.config_dir):
        config_files = glob.glob(os.path.join(args.config_dir, "*.yaml"))
        config_files.sort()
    
    if args.check:
        print(f"\n{'Dataset':<25} | {'Status':<12} | {'Note':<25} | {'Path'}")
        print(f"{'-'*100}")

    for config_file in config_files:
        process_single_config(config_file, args.data_root, args.check)

if __name__ == "__main__":
    main()