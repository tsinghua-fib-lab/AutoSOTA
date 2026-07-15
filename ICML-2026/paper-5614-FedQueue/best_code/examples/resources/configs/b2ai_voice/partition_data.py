"""
Partition the spectrogram dataset into federated learning clients
based on disease cohort.

Client assignment (by participant_id):
  0 - Voice Disorders only        (eligible_studies___1, single-cohort)
  1 - Neurological only           (eligible_studies___2, single-cohort)
  2 - Mood / Psychiatric only     (eligible_studies___3, single-cohort)
  3 - Respiratory only            (eligible_studies___4, single-cohort)
  4 - Multi-cohort & controls     (eligible for 2+ cohorts)

Output: datasets/RawData/b2ai-voice/partitioned_data/client_<N>/data.npz
  each file contains arrays: X (recordings x 402), y (labels), pids (participant_ids)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "datasets", "RawData", "b2ai-voice"
)
CACHE_FILE = os.path.join(DATA_DIR, "spectrogram_pooled_cache.npz")
OUT_DIR = os.path.join(DATA_DIR, "partitioned_data")

# ---------------------------------------------------------------------------
# 1. Load labels (per participant)
# ---------------------------------------------------------------------------
print("Loading phenotype labels...")
phenotype = pd.read_csv(
    os.path.join(DATA_DIR, "phenotype.tsv"),
    sep="\t",
    usecols=[
        "participant_id",
        "sex_at_birth",
        "eligible_studies___1",
        "eligible_studies___2",
        "eligible_studies___3",
        "eligible_studies___4",
    ],
)
phenotype = phenotype.dropna(subset=["sex_at_birth"])
label_map = {"Female": 0, "Male": 1}
pid_to_label = {
    row.participant_id: label_map[row.sex_at_birth]
    for row in phenotype.itertuples()
    if row.sex_at_birth in label_map
}

# ---------------------------------------------------------------------------
# 2. Assign each participant to a client based on cohort membership
# ---------------------------------------------------------------------------
cohort_cols = [
    "eligible_studies___1",
    "eligible_studies___2",
    "eligible_studies___3",
    "eligible_studies___4",
]

pid_to_client = {}
for row in phenotype.itertuples():
    if row.participant_id not in pid_to_label:
        continue
    memberships = [not pd.isna(getattr(row, c)) for c in cohort_cols]
    n_cohorts = sum(memberships)
    if n_cohorts == 1:
        pid_to_client[row.participant_id] = memberships.index(True)  # 0-3
    else:
        pid_to_client[row.participant_id] = 4  # multi-cohort

client_names = {
    0: "Voice Disorders",
    1: "Neurological",
    2: "Mood / Psychiatric",
    3: "Respiratory",
    4: "Multi-cohort / Controls",
}
for cid, name in client_names.items():
    n = sum(v == cid for v in pid_to_client.values())
    print(f"  Client {cid} ({name}): {n} participants")

# ---------------------------------------------------------------------------
# 3. Load or build pooled spectrogram features (reuse cache from train_gender)
# ---------------------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"\nLoading pooled features from cache: {CACHE_FILE}")
    cache = np.load(CACHE_FILE)
    X = cache["X"]
    y = cache["y"]
    pids = cache["pids"]
else:
    print(
        "\nCache not found. Loading spectrograms from parquet (this may take a minute)..."
    )
    ds = pd.read_parquet(os.path.join(DATA_DIR, "spectrogram.parquet"))
    features, labels, participant_ids = [], [], []
    for rec in ds.itertuples(index=False):
        pid = rec.participant_id
        if pid not in pid_to_label:
            continue
        spec = np.asarray(rec.spectrogram, dtype=np.float32)  # (201, N)
        pooled = np.concatenate([spec.mean(axis=1), spec.std(axis=1)])  # (402,)
        features.append(pooled)
        labels.append(pid_to_label[pid])
        participant_ids.append(pid)
    X = np.stack(features)
    y = np.array(labels)
    pids = np.array(participant_ids)
    np.savez(CACHE_FILE, X=X, y=y, pids=pids)
    print(f"  Cache saved to: {CACHE_FILE}")

print(f"  Total recordings: {len(X)}")

# ---------------------------------------------------------------------------
# 4. Partition and save
# ---------------------------------------------------------------------------
print()
os.makedirs(OUT_DIR, exist_ok=True)

for cid, name in client_names.items():
    mask = np.array([pid_to_client.get(p, -1) == cid for p in pids])
    X_c, y_c, pids_c = X[mask], y[mask], pids[mask]

    client_dir = os.path.join(OUT_DIR, f"client_{cid}")
    os.makedirs(client_dir, exist_ok=True)
    out_path = os.path.join(client_dir, "data.npz")
    np.savez(out_path, X=X_c, y=y_c, pids=pids_c)

    n_participants = len(np.unique(pids_c))
    n_female = int((y_c == 0).sum())
    n_male = int((y_c == 1).sum())
    print(f"  client_{cid}/ ({name})")
    print(
        f"    participants={n_participants}  recordings={len(X_c)}"
        f"  Female={n_female}  Male={n_male}"
    )
    print(f"    -> {out_path}")

print("\nPartitioning complete.")
