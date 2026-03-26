#!/usr/bin/env python3
"""
GSTransform (paper-303) full evaluation pipeline for AmzCF dataset.
Uses oracle labels (ground-truth is_counterfactual) from the dataset.
Outputs: FINAL V-MEASURE: <score>
"""
import os
import sys
import json
import pickle
import shutil
import logging
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_PATH = "/datasets"
MODELS_PATH  = "/models"
UAE_MODEL_PATH = "/repo/uae_model"
CACHE_DIR    = "/repo/cache"
EMBED_CACHE  = os.path.join(CACHE_DIR, "embeddings.pkl")
CLASSIF_JSON = os.path.join(CACHE_DIR, "classifications_xxx.json")
MODEL_PTH    = os.path.join(CACHE_DIR, "model.pth")
N_TRAIN_SAMPLES = 5000
RANDOM_SEED  = 42
EMBEDDING_DIM = 128           # GSTransform output dim
BATCH_SIZE    = 512          # training batch size
NUM_EPOCHS    = 500
LEARNING_RATE = 0.0001
MARGIN        = 1.0
PATIENCE      = 50

# ─── Step 0: Setup UAE model path ─────────────────────────────────────────────
def setup_uae_model():
    os.makedirs(UAE_MODEL_PATH, exist_ok=True)
    # Check if tokenizer.json is already present
    if os.path.exists(os.path.join(UAE_MODEL_PATH, "tokenizer.json")):
        logger.info("UAE model already set up")
        return
    logger.info("Setting up UAE model...")
    # Copy model files from /models (read-only mount)
    for fname in os.listdir(MODELS_PATH):
        src = os.path.join(MODELS_PATH, fname)
        dst = os.path.join(UAE_MODEL_PATH, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    # Copy 1_Pooling directory
    pooling_src = os.path.join(MODELS_PATH, "1_Pooling")
    pooling_dst = os.path.join(UAE_MODEL_PATH, "1_Pooling")
    if os.path.isdir(pooling_src) and not os.path.isdir(pooling_dst):
        shutil.copytree(pooling_src, pooling_dst)
    # Download tokenizer from HF mirror
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = hf_endpoint
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.txt",
                      "special_tokens_map.json"]:
            dst = os.path.join(UAE_MODEL_PATH, fname)
            if not os.path.exists(dst):
                path = hf_hub_download(
                    repo_id="WhereIsAI/UAE-Large-V1",
                    filename=fname,
                    local_dir=UAE_MODEL_PATH,
                    token=os.environ.get("HUGGING_FACE_HUB_TOKEN", "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"),
                )
                logger.info(f"Downloaded {fname}")
    except Exception as e:
        logger.warning(f"Tokenizer download failed: {e}")
        # Try alternate: copy from /tmp/uae_model if it exists
        tmp_uae = "/tmp/uae_model"
        if os.path.isdir(tmp_uae):
            for fname in os.listdir(tmp_uae):
                dst = os.path.join(UAE_MODEL_PATH, fname)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(tmp_uae, fname), dst)

# ─── Step 1: Generate embeddings ──────────────────────────────────────────────
def generate_embeddings_for_dataset():
    if os.path.exists(EMBED_CACHE):
        logger.info(f"Embeddings cache found at {EMBED_CACHE}")
        with open(EMBED_CACHE, "rb") as f:
            return pickle.load(f), None  # embeddings, labels (loaded later)

    logger.info("Generating embeddings from scratch...")
    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds = load_dataset(DATASET_PATH)
    # Concatenate all splits
    all_sentences = []
    all_labels = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            for row in ds[split]:
                all_sentences.append(row["sentence"])
                all_labels.append(row["is_counterfactual"])

    logger.info(f"Total samples: {len(all_sentences)}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(UAE_MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(UAE_MODEL_PATH, trust_remote_code=True)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))
        eff_batch = 128 * n_gpus
    else:
        eff_batch = 128
    model = model.to(device)
    model.eval()

    embeddings = []
    for i in tqdm(range(0, len(all_sentences), eff_batch)):
        batch = all_sentences[i:i + eff_batch]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Embeddings saved: {embeddings.shape}")
    return embeddings, all_labels

# ─── Step 2: Create oracle labels JSON ────────────────────────────────────────
def create_oracle_labels(all_labels, n_samples=N_TRAIN_SAMPLES, seed=RANDOM_SEED):
    if os.path.exists(CLASSIF_JSON):
        logger.info(f"Classifications cache found at {CLASSIF_JSON}")
        return

    logger.info(f"Creating oracle labels (n={n_samples})...")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_labels), size=min(n_samples, len(all_labels)), replace=False)

    classifications = {}
    for idx in indices:
        lbl = all_labels[int(idx)]
        classifications[str(idx)] = "Counterfactual" if lbl == 1 else "Non-counterfactual"

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CLASSIF_JSON, "w") as f:
        json.dump(classifications, f)
    logger.info(f"Saved {len(classifications)} oracle labels")

# ─── Step 3: Train GSTransform ────────────────────────────────────────────────
def train_model():
    if os.path.exists(MODEL_PTH):
        logger.info(f"Model cache found at {MODEL_PTH}")
        return

    logger.info("Training GSTransform model...")
    import train as train_module
    train_module.main()

# ─── Step 4: Evaluate V-measure ───────────────────────────────────────────────
def evaluate():
    from evaluate import load_trained_model, get_transformed_embeddings

    # Load embeddings and labels
    logger.info("Loading embeddings...")
    with open(EMBED_CACHE, "rb") as f:
        embeddings = pickle.load(f)

    # Reload labels from dataset
    from datasets import load_dataset
    ds = load_dataset(DATASET_PATH)
    all_labels = []
    for split in ["train", "validation", "test"]:
        if split in ds:
            for row in ds[split]:
                all_labels.append(int(row["is_counterfactual"]))

    labels = np.array(all_labels)
    n_clusters = len(np.unique(labels))

    # Base NMI
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    base_clusters = kmeans.fit_predict(embeddings)
    base_nmi = normalized_mutual_info_score(labels, base_clusters)
    logger.info(f"Base NMI: {base_nmi:.4f}")

    # Transformed NMI
    checkpoint = torch.load(MODEL_PTH)
    model = load_trained_model(MODEL_PTH, embeddings.shape[1],
                               checkpoint.get("embedding_dim", EMBEDDING_DIM))
    transformed = get_transformed_embeddings(model, embeddings)

    # Run KMeans multiple times, pick best by inertia (avoids unlucky initialization)
    best_inertia = float('inf')
    best_clusters = None
    for seed in range(20):
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=1)
        clust = km.fit_predict(transformed)
        if km.inertia_ < best_inertia:
            best_inertia = km.inertia_
            best_clusters = clust
    transformed_clusters = best_clusters
    nmi = normalized_mutual_info_score(labels, transformed_clusters)
    logger.info(f"Transformed NMI (V-measure): {nmi:.4f}")

    return nmi * 100.0  # return as percentage

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.chdir("/repo")
    os.makedirs(CACHE_DIR, exist_ok=True)

    setup_uae_model()

    embeddings, all_labels = generate_embeddings_for_dataset()

    if all_labels is None:
        # Load labels from dataset (embeddings were cached)
        from datasets import load_dataset
        ds = load_dataset(DATASET_PATH)
        all_labels = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for row in ds[split]:
                    all_labels.append(int(row["is_counterfactual"]))

    create_oracle_labels(all_labels)
    train_model()
    v_measure = evaluate()

    print(f"FINAL V-MEASURE: {v_measure:.2f}")

if __name__ == "__main__":
    main()
