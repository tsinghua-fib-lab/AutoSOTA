import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# config
STORE_DIR, DATA_DIR = "feature_store", "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
OOF_OUTPUT = "train_embeddings.csv"
MODEL_OUTPUT = "transformer.pth"
HEATMAP_OUTPUT = "interaction_heatmap.png"

# Hyperparameters
COUNCIL_SIZE = 24
EMBED_DIM = 32
NUM_HEADS = 4
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001
N_FOLDS = 5
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class InteractionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_proj = nn.Parameter(torch.randn(COUNCIL_SIZE, EMBED_DIM))
        self.feat_bias = nn.Parameter(torch.zeros(COUNCIL_SIZE, EMBED_DIM))
        self.type_emb = nn.Parameter(torch.randn(1, COUNCIL_SIZE, EMBED_DIM))
        self.cross_attn = nn.MultiheadAttention(EMBED_DIM, num_heads=NUM_HEADS, batch_first=True)
        self.head = nn.Sequential(nn.Linear(COUNCIL_SIZE * EMBED_DIM, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, sol, solv):
        sol_emb = (sol.unsqueeze(-1) * self.feat_proj) + self.feat_bias + self.type_emb
        solv_emb = (solv.unsqueeze(-1) * self.feat_proj) + self.feat_bias + self.type_emb
        enriched, attn = self.cross_attn(query=sol_emb, key=solv_emb, value=solv_emb)
        return self.head(enriched.reshape(enriched.size(0), -1)), enriched.reshape(enriched.size(0), -1), attn

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    sol_store = pd.read_parquet(os.path.join(STORE_DIR, "solute_council.parquet")).set_index("SMILES_KEY")
    solv_store = pd.read_parquet(os.path.join(STORE_DIR, "solvent_council.parquet")).set_index("SMILES_KEY")
    X_sol = sol_store.loc[df["Solute"]].values.astype(np.float32)
    X_solv = solv_store.loc[df["Solvent"]].values.astype(np.float32)
    return df, X_sol, X_solv, df["LogS"].values.astype(np.float32), sol_store.columns.tolist()

def run_training():
    df_train, X_sol, X_solv, y, feat_names = load_data(TRAIN_FILE)
    oof_feats = np.zeros((len(df_train), COUNCIL_SIZE * EMBED_DIM))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"Starting {N_FOLDS}-Fold OOF Training on {DEVICE}...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_sol)):
        model = InteractionTransformer().to(DEVICE)
        optimizer, criterion = optim.Adam(model.parameters(), lr=LR), nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_sol[train_idx]).to(DEVICE), 
                          torch.tensor(X_solv[train_idx]).to(DEVICE), 
                          torch.tensor(y[train_idx]).view(-1, 1).to(DEVICE)), 
            batch_size=BATCH_SIZE, shuffle=True
        )

        model.train()
        for epoch in tqdm(range(EPOCHS), desc=f"Training on Fold {fold + 1}"):
            for b_sol, b_solv, b_y in loader:
                optimizer.zero_grad()
                pred, _, _ = model(b_sol, b_solv)
                criterion(pred, b_y).backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            _, feats, _ = model(torch.tensor(X_sol[val_idx]).to(DEVICE), torch.tensor(X_solv[val_idx]).to(DEVICE))
            oof_feats[val_idx] = feats.cpu().numpy()
        print(f"  Fold {fold+1} complete.")

    pd.DataFrame(oof_feats, columns=[f"Learned_{i}" for i in range(oof_feats.shape[1])]).assign(LogS=df_train["LogS"]).to_csv(OOF_OUTPUT, index=False)
    
    print("Training Final Production Model...")
    final_model = InteractionTransformer().to(DEVICE)
    optimizer, criterion = optim.Adam(final_model.parameters(), lr=LR), nn.MSELoss()
    all_loader = DataLoader(
        TensorDataset(torch.tensor(X_sol).to(DEVICE), 
                      torch.tensor(X_solv).to(DEVICE), 
                      torch.tensor(y).view(-1, 1).to(DEVICE)), 
        batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in tqdm(range(EPOCHS), desc="Final Pass"):
        for b_sol, b_solv, b_y in all_loader:
            optimizer.zero_grad()
            pred, _, _ = final_model(b_sol, b_solv)
            criterion(pred, b_y).backward()
            optimizer.step()

    # Generate Interaction Heatmap
    final_model.eval()
    with torch.no_grad():
        _, _, attn = final_model(torch.tensor(X_sol[:100]).to(DEVICE), torch.tensor(X_solv[:100]).to(DEVICE))
        plt.figure(figsize=(12, 10))
        sns.heatmap(attn.mean(dim=0).cpu().numpy(), xticklabels=feat_names, yticklabels=feat_names, cmap="magma")
        plt.title("Physical Interaction Heatmap (Expanded Council)")
        plt.savefig(HEATMAP_OUTPUT, dpi=300)
        plt.close()

    torch.save(final_model.state_dict(), MODEL_OUTPUT)
    print(f"Success: {OOF_OUTPUT}, {MODEL_OUTPUT}, and {HEATMAP_OUTPUT} saved.")

if __name__ == "__main__":
    run_training()
