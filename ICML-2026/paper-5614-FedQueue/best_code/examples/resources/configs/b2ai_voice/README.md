# B2AI-Voice Federated Learning Experiment

Sex-at-birth prediction from voice spectrograms using the [Bridge2AI Voice v2.0.1](https://physionet.org/content/b2ai-voice/2.0.1) dataset, trained with **FedAvg** across four disease-cohort clients via gRPC.

---

## Dataset

**Source:** Bridge2AI Voice v2.0.1 — [https://physionet.org/content/b2ai-voice/2.0.1](https://physionet.org/content/b2ai-voice/2.0.1)

Download and place the raw dataset files under `examples/datasets/RawData/b2ai-voice/`. The two required files are:
- `phenotype.tsv` — participant metadata (sex, cohort eligibility)
- `spectrogram.parquet` — per-recording spectrograms

Then run the partitioning script (from the `examples/` directory) to split the data into per-client `.npz` files:

```bash
python resources/configs/b2ai_voice/partition_data.py
```

This produces `examples/datasets/RawData/b2ai-voice/partitioned_data/client_<N>/data.npz` for clients 0–4. A pooled-feature cache (`spectrogram_pooled_cache.npz`) is saved alongside the raw data on the first run to speed up re-partitioning.

---

## Task

- **Input:** 402-dimensional feature vector — per-recording mean and standard deviation pooled from a 201-frequency spectrogram over time.
- **Label:** `sex_at_birth` (0 = Female, 1 = Male) from `phenotype.tsv`.
- **Metric:** Classification accuracy on a shared held-out validation set.

---

## Data Partitioning

Each client holds recordings from a single disease cohort. Training normalization (mean/std) is computed locally and applied to both the local training split and the shared validation set.

| Config | Client ID | Cohort | Participants | Recordings |
|---|---|---|---|---|
| `client_1.yaml` | Client0 | Voice Disorders | 94 | 3,261 |
| `client_2.yaml` | Client1 | Neurological | 69 | 3,498 |
| `client_3.yaml` | Client2 | Mood / Psychiatric | 22 | 670 |
| `client_4.yaml` | Client3 | Respiratory | 77 | 2,289 |
| *(validation)* | — | Multi-cohort / Controls | 175 | 6,863 |

The shared validation set (`client_4/data.npz`) is used by every client after each round but is never used for training.

**Expected data layout:**

```
examples/datasets/RawData/b2ai-voice/partitioned_data/
├── client_0/data.npz   # Voice Disorders
├── client_1/data.npz   # Neurological
├── client_2/data.npz   # Mood / Psychiatric
├── client_3/data.npz   # Respiratory
└── client_4/data.npz   # Shared validation (Multi-cohort / Controls)
```

Each `.npz` file contains:
- `X`: float32 array of shape `(N, 402)` — spectrogram features
- `y`: int64 array of shape `(N,)` — binary sex-at-birth label

---

## Model

`GenderMLP` (`examples/resources/model/gender_mlp.py`):

```
Linear(402 → 256) → BatchNorm1d → ReLU → Dropout(0.3)
Linear(256 → 64)  → BatchNorm1d → ReLU → Dropout(0.3)
Linear(64 → 2)
```

---

## FL Configuration

| Parameter | Value |
|---|---|
| Algorithm | FedAvg (synchronous) |
| Global rounds | 20 |
| Local trainer | VanillaTrainer (step mode) |
| Local steps per round | 200 |
| Optimizer | Adam (lr = 0.001) |
| Batch size | 16 (drop last to avoid single-sample BatchNorm batches) |
| Client weighting | Proportional to number of training samples |
| Communication | gRPC, `localhost:50051`, no SSL |

---

## Running the Experiment

All commands are run from the `examples/` directory.

### 1. Start the server

```bash
python grpc/run_server.py --config ./resources/configs/b2ai_voice/server_fedavg.yaml
```

### 2. Start the four clients (in separate terminals or as background jobs)

```bash
python grpc/run_client.py --config ./resources/configs/b2ai_voice/client_1.yaml &
python grpc/run_client.py --config ./resources/configs/b2ai_voice/client_2.yaml &
python grpc/run_client.py --config ./resources/configs/b2ai_voice/client_3.yaml &
python grpc/run_client.py --config ./resources/configs/b2ai_voice/client_4.yaml
```

Training logs are written to `examples/output/` and run event logs to `examples/runs/`.
