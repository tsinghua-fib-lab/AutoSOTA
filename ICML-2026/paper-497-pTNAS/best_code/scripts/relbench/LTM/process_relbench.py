import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from LTM import get_embeddings
from utils.data import DatabaseFactory


TASKS = [
    ("hm", "user-churn", "rel-hm"),
    ("event", "user-repeat", "rel-event"),
    ("trial", "study-outcome", "rel-trial"),
    ("ratebeer", "user-active", "ratebeer"),
    ("avito", "user-clicks", "rel-avito"),

    ("hm", "item-sales", "rel-hm"),
    ("event", "user-attendance", "rel-event"),
    ("trial", "site-success", "rel-trial"),
    ("ratebeer", "beer-positive", "ratebeer"),
    ("avito", "ad-ctr", "rel-avito")
]

pretrain_dir = os.environ.get("TPBERTA_PRETRAIN_DIR", "third_party/tp-berta/checkpoints/tp-joint")
cache_dir_root = os.environ.get("RELBENCH_CACHE_DIR", "datasets/relbench_cache/")


def main():
    """Main function to switch between different embedding models."""
    parser = argparse.ArgumentParser(
        description="Convert relational data to embeddings using different models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tpberta",
        choices=["tpberta", "nomic", "bge"],
        help="Embedding model to use: tpberta, nomic, or bge"
    )
    
    args = parser.parse_args()
    model = args.model
    
    print(f"Using embedding model: {model}")
    print()
    
    for ele in tqdm(TASKS, desc="Processing tasks"):
        db_name = ele[0]
        task_name = ele[1]
        cache_dir_db_name = ele[2]
        cache_dir = cache_dir_root + cache_dir_db_name
        
        try:
            print(f"Processing {db_name}:{task_name}")

            db = DatabaseFactory.get_db(db_name,
                                        cache_dir=cache_dir,
                                        upto_test_timestamp=False,
                                        with_text_compress=True)
            dataset = DatabaseFactory.get_dataset(db_name, cache_dir)
            task = DatabaseFactory.get_task(db_name, task_name, dataset)
            entity_table = task.entity_table
            used_df = db.table_dict[entity_table].df

            print(f"  DataFrame shape: {used_df.shape}")

            # Get embeddings using specified model
            embeddings = get_embeddings(
                df=used_df,
                model=model,
                dataset_name=f"{db_name}_{task_name}",
                pretrain_dir=pretrain_dir if model == "tpberta" else None,
                has_label=False,
                device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            )

            print(f"  Embeddings shape: {embeddings.shape}")

            # Save to model-specific folder: {output_root}/{model}/{db_name}_{task_name}_data.npy
            output_dir = Path("run_outputs/data/relbench/baselines/ltm/tpberta_relbench") / model
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{db_name}_{task_name}_data.npy"
            output_path = output_dir / output_filename
            np.save(output_path, embeddings)
            print(f"  Saved to: {output_path}")
            print()
        except Exception as e:
            print(f"  ❌ Error processing {db_name}:{task_name}: {str(e)}")
            print(f"  Continuing to next dataset...")
            print()
            continue


if __name__ == "__main__":
    main()
