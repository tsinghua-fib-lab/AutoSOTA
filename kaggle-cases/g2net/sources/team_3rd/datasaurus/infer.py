import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import COMP_NAME, INPUT_PATH, MODEL_CACHE, OUTPUT_PATH
from src.datasets import GWDataModule
from src.models import GWModel
from src.utils import prepare_args

torch.hub.set_dir(MODEL_CACHE)


def infer(model, loader, device="cuda", desc=None):
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    loader = tqdm(loader, desc=desc)

    # Permuatation for TTA
    perm = torch.tensor([1, 0, 2]).to(device)

    with torch.no_grad():
        predictions = []
        for x, _ in loader:
            x = x.to(device)
            p1 = model(x)
            p2 = model(x[:, perm])
            predictions.append(((p1 + p2) / 2).cpu())
    return torch.cat(predictions, 0).numpy()


def make_predictions(args, ckpt_suffix="auc"):
    mpaths = list((OUTPUT_PATH / args.timestamp).rglob(f"*{ckpt_suffix}.ckpt"))
    mpaths.sort()

    if len(mpaths) == 0:
        print("No checkpoints found")
        return

    sub = pd.read_csv(INPUT_PATH / "sample_submission.csv")
    oofs = pd.read_csv(INPUT_PATH / "training_labels.csv", index_col="id")
    oofs["prediction"] = 0
    oofs["fold"] = 0
    test_preds = []
    fold_scores = []

    for fold, p in enumerate(mpaths):
        model = GWModel.load_from_checkpoint(p)
        dm = GWDataModule().from_argparse_args(args)

        # Make OOF preds
        dm.setup("fit", fold)  # Apprently this can only be called once
        val_df = dm.df.query(f"fold == {fold}").copy()
        preds = infer(model, dm.val_dataloader(), desc=f"Fold {fold} OOFs")
        oofs.loc[val_df["id"], "prediction"] = preds
        oofs.loc[val_df["id"], "fold"] = fold
        fold_scores.append(roc_auc_score(val_df["target"], preds))

        # Make test preds
        dm.setup("test")
        test_preds.append(infer(model, dm.test_dataloader(), desc=f"Fold {fold} Test"))

    for i, s in enumerate(fold_scores):
        print(f"Fold {i}: {s:0.5f}")

    score = roc_auc_score(oofs["target"], oofs["prediction"])
    print(f"Mean AUC: {np.mean(fold_scores):0.5f}")
    print(f"OOF AUC: {score:0.5f}")
    string = f"{args.timestamp}_CV-{np.mean(fold_scores):0.5f}"

    # Generate test predictions per fold
    for fold, t in enumerate(test_preds):
        sub[f"fold_{fold}"] = t

    sub["target"] = np.stack(test_preds).mean(0)

    sub.to_csv(OUTPUT_PATH / args.timestamp / f"test_{string}.csv", index=False)
    oofs.to_csv(OUTPUT_PATH / args.timestamp / f"oofs_{string}.csv")
    sub[["id", "target"]].to_csv(
        OUTPUT_PATH / args.timestamp / f"sub_{string}.csv", index=False
    )

    if args.submit:
        print("Submitting to Kaggle")
        submit_parts = [
            "kaggle",
            "competitions",
            "submit",
            COMP_NAME,
            "-f",
            OUTPUT_PATH / args.timestamp / f"sub_{string}.csv",
            "-m",
            f"CV: {np.mean(fold_scores):0.5f}",
        ]
        subprocess.call(submit_parts)


if __name__ == "__main__":
    args = prepare_args()
    make_predictions(args)
