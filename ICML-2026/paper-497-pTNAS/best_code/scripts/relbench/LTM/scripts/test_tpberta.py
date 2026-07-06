import os
from pathlib import Path

import pandas as pd
import numpy as np
from LTM.models.tpberta import get_tpberta_embeddings

# Set pretrain directory
pretrain_dir = "third_party/tp-berta/checkpoints/tp-joint"

# Create test DataFrame (no label column)
test_df = pd.DataFrame({
    'age': [25, 30, 35],
    'gender': ['male', 'female', 'male'],
    'occupation': ['engineer', 'teacher', 'doctor'],
    'salary': [50000, 45000, 80000]
})

print(f"Input DataFrame shape: {test_df.shape}")
print(f"Input DataFrame columns: {test_df.columns.tolist()}")
print(f"\nInput DataFrame:\n{test_df}")

# Get embeddings (has_label=False since no label column)
embeddings = get_tpberta_embeddings(
    df=test_df,
    pretrain_dir=pretrain_dir,
    has_label=False,  # No label column
    device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
)

print(f"\nOutput embeddings shape: {embeddings.shape}")
print(f"Output embeddings dtype: {embeddings.dtype}")

PROJECT_ROOT = Path(__file__).resolve().parents[5]
output_dir = PROJECT_ROOT / "run_outputs" / "data" / "relbench" / "baselines" / "ltm" / "test_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Save to numpy array file
output_path = output_dir / "test_embeddings.npy"
np.save(output_path, embeddings)
print(f"\nEmbeddings saved to: {output_path}")

# Optionally, also save as CSV for inspection
output_csv = output_dir / "test_embeddings.csv"
pd.DataFrame(embeddings).to_csv(output_csv, index=False)
print(f"Embeddings also saved as CSV to: {output_csv}")
