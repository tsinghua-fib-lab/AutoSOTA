# ============================================================================
# CGBoost Explainability Pipeline (LLM-Optimized)
# ============================================================================

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import Pool

# ============================ CONFIG ========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_FILE = "data/test.csv"
STORE_DIR = "feature_store"
MODEL_DIR = "model"
OUTPUT_DIR = "cgboost_explanations"
PREDICTION_THRESHOLD = 0.30

TRANSFORMER_PATH = "transformer.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "explanations"), exist_ok=True)

# ============================ LOAD DATA ======================================

print("Loading test data and models...")

df_test = pd.read_csv(TEST_FILE)

sol_raw = pd.read_parquet(os.path.join(STORE_DIR, "solute_raw.parquet")).set_index("SMILES_KEY")
solv_raw = pd.read_parquet(os.path.join(STORE_DIR, "solvent_raw.parquet")).set_index("SMILES_KEY")
sol_council = pd.read_parquet(os.path.join(STORE_DIR, "solute_council.parquet")).set_index("SMILES_KEY")
solv_council = pd.read_parquet(os.path.join(STORE_DIR, "solvent_council.parquet")).set_index("SMILES_KEY")

from train_transformer import InteractionTransformer

transformer = InteractionTransformer().to(DEVICE)
transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
transformer.eval()

catboost_model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
selector = joblib.load(os.path.join(MODEL_DIR, "selector.joblib"))

council_feature_names = sol_council.columns.tolist()

# ============================ FEATURE MAP ===================================

raw_solute_names = [f"Solute_{c}" for c in sol_raw.columns]
raw_solvent_names = [f"Solvent_{c}" for c in solv_raw.columns]

interact_names = [f"Interact_{c}" for c in council_feature_names]
thermo_names = ["pred_Tm", "T_red", "T", "T_inv"]

full_feature_names = (
    raw_solute_names +
    raw_solvent_names +
    interact_names +
    thermo_names
)

trained_feature_names = np.array(full_feature_names)[selector.get_support()]

pd.DataFrame({
    "Index": range(len(trained_feature_names)),
    "Feature": trained_feature_names
}).to_csv(os.path.join(OUTPUT_DIR, "trained_feature_map.csv"), index=False)

# ============================ FEATURE GENERATION =============================

def generate_features(df):
    X_sol = sol_council.loc[df["Solute"]].values.astype(np.float32)
    X_solv = solv_council.loc[df["Solvent"]].values.astype(np.float32)

    embeds, attns = [], []

    with torch.no_grad():
        for i in range(len(df)):
            sol = torch.tensor(X_sol[i:i+1]).to(DEVICE)
            solv = torch.tensor(X_solv[i:i+1]).to(DEVICE)
            _, feats, attn = transformer(sol, solv)
            embeds.append(feats.cpu().numpy())
            attns.append(attn.cpu().numpy())

    X_embed = np.vstack(embeds)
    attention = np.vstack(attns)

    T = df["Temperature"].values.reshape(-1, 1)
    T_inv = (1000 / df["Temperature"]).values.reshape(-1, 1)
    Tm = sol_raw.loc[df["Solute"], "pred_Tm"].values.reshape(-1, 1)
    T_red = T / Tm

    X_reshaped = X_embed.reshape(-1, 24, 32)
    X_mod = np.linalg.norm(X_reshaped, axis=2)
    X_sign = np.sign(X_reshaped.mean(axis=2))
    X_interact = (X_sign * X_mod) * T_inv

    X_raw = np.hstack([
        sol_raw.loc[df["Solute"]].values,
        solv_raw.loc[df["Solvent"]].values
    ])

    X_full = np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])
    return X_full, attention

X_full, attention_weights = generate_features(df_test)
X_pruned = selector.transform(X_full)
preds = catboost_model.predict(X_pruned)

errors = np.abs(df_test["LogS"].values - preds)
good_idx = np.where(errors < PREDICTION_THRESHOLD)[0]

# ============================ SHAP ===========================================

pool = Pool(X_pruned[good_idx], feature_names=trained_feature_names.tolist())
shap_vals = catboost_model.get_feature_importance(pool, type="ShapValues")
shap_vals = np.array(shap_vals)[:, :-1]
leaf_paths = catboost_model.calc_leaf_indexes(pool)

# ============================ LLM-READY EXTRACTION ============================

def summarize_attention(attn):
    score = attn.sum(axis=0) + attn.sum(axis=1)
    pairs = [
        (i, j, float(attn[i, j]))
        for i in range(24) for j in range(24)
    ]
    top_pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)[:8]
    return score.tolist(), top_pairs

def group_shap(shap_row):
    out = {"Solute":0, "Solvent":0, "Interact":0, "Thermo":0}
    for v, n in zip(shap_row, trained_feature_names):
        if n.startswith("Solute_"): out["Solute"] += v
        elif n.startswith("Solvent_"): out["Solvent"] += v
        elif n.startswith("Interact_"): out["Interact"] += v
        else: out["Thermo"] += v
    return {k: float(v) for k, v in out.items()}

llm_rows = []

for i, idx in enumerate(good_idx):
    row = df_test.iloc[idx]
    attn = attention_weights[idx]

    council_score, top_pairs = summarize_attention(attn)
    shap_group = group_shap(shap_vals[i])

    top_features = sorted(
        zip(trained_feature_names, shap_vals[i]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    explanation = {
        "solute": row["Solute"],
        "solvent": row["Solvent"],
        "temperature": float(row["Temperature"]),
        "y_true": float(row["LogS"]),
        "y_pred": float(preds[idx]),
        "abs_error": float(errors[idx]),

        "interaction_reasoning": {
            "council_importance": dict(zip(council_feature_names, council_score)),
            "top_interactions": [
                {
                    "council_i": council_feature_names[i],
                    "council_j": council_feature_names[j],
                    "weight": w
                } for i, j, w in top_pairs
            ]
        },

        "decision_reasoning": {
            "top_features": [
                {"name": f, "shap": float(v)} for f, v in top_features
            ],
            "group_contributions": shap_group,
            "leaf_path": leaf_paths[i].tolist()
        }
    }

    with open(os.path.join(OUTPUT_DIR, "explanations", f"sample_{idx}.json"), "w") as f:
        json.dump(explanation, f, indent=2)

    llm_rows.append({
        "solute": row["Solute"],
        "solvent": row["Solvent"],
        "dominant_interaction": council_feature_names[np.argmax(council_score)],
        "dominant_feature_group": max(shap_group, key=shap_group.get),
        "abs_error": errors[idx]
    })

pd.DataFrame(llm_rows).to_csv(
    os.path.join(OUTPUT_DIR, "llm_table.csv"),
    index=False
)

print("✓ LLM-ready explainability pipeline complete")
