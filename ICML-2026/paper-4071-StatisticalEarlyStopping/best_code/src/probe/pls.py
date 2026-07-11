import pickle
from src.probe.train import MODEL_LAYER_DEFAULTS
import argparse
# load principle component regression model
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# load dataset
def load_dataset(model_name, layer_idx):
    with open(f"probe/{model_name}/layer{layer_idx}.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--layer_idx', type=int, default=None)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    return parser

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if args.layer_idx is None:
        args.layer_idx = MODEL_LAYER_DEFAULTS[args.model_name]
    dataset = load_dataset(args.model_name, args.layer_idx)
    X = np.stack([item["x"] for item in dataset])
    y = np.array([item["label"] for item in dataset])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )
    pls = PLSRegression(n_components=1)
    pls.fit(X_train, y_train)
    y_pred = pls.predict(X_test)
    # make sure to cap the predictions to be between 0 and 1
    y_pred = np.clip(y_pred, 0, 1)
    score = roc_auc_score(y_test, y_pred)
    with open(f"probe/{args.model_name}/pls_auc_score_layer{args.layer_idx}.json", "w") as f:
        f.write(f'{{"auc_score": {score}}}')
    print(f"Model: {args.model_name}, Layer: {args.layer_idx}, AUC Score: {score}")
    with open(f"probe/{args.model_name}/pls_model_layer{args.layer_idx}.pkl", "wb") as f:
        pickle.dump(pls, f)

if __name__ == "__main__":
    main()