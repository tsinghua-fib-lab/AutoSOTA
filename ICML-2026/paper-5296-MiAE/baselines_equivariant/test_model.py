import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=(
        "The TorchScript type system doesn't support instance-level annotations "
        "on empty non-base types in `__init__`.*"
    ),
    category=UserWarning,
)

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm

from data import (
    batches_to_protein_graphs,
    build_graph_batches,
    resolve_existing_path,
)
from formatting import format_table


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./datasets/afdb_FS_plddt80")
    parser.add_argument("--dataset_name", type=str, default="ted")
    parser.add_argument("--dataset_batch_size", type=int, default=512)
    parser.add_argument(
        "--tedbench_path",
        type=str,
        default=None,
        help="Optional path containing the tedbench package for raw TED preprocessing.",
    )

    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--feature_mode", type=str, default="constant", choices=["constant", "token"])
    parser.add_argument("--cutoff", type=float, default=8.0)

    parser.add_argument(
        "--model",
        type=str,
        default="bige3nn",
        choices=["bige3nn", "mace", "gotennet"],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_graphs_path", type=str, default="test_graphs.pt")
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--allow_partial_load", action="store_true")

    parser.add_argument("--external_dataset_root", type=str, default=None)
    parser.add_argument("--external_dataset_name", type=str, default="cath4.4")
    parser.add_argument("--external_split", type=str, default="test")

    return parser.parse_args()


def add_tedbench_path(args):
    candidates = []
    if args.tedbench_path:
        candidates.append(Path(args.tedbench_path).expanduser())
    candidates.extend([SCRIPT_DIR.parent, Path.cwd()])

    for candidate in candidates:
        if (candidate / "tedbench").exists():
            sys.path.insert(0, str(candidate))
            return


def _strip_prefix(state_dict, prefix):
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(state):
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        state = _strip_prefix(state, "model.")
        state = _strip_prefix(state, "module.")
    return state


def infer_gotennet_basis(state_dict):
    if not isinstance(state_dict, dict):
        return None
    for key in (
        "model.node_init.A_nbr.weight",
        "node_init.A_nbr.weight",
        "model.mlp.0.weight",
        "mlp.0.weight",
    ):
        if key in state_dict:
            return state_dict[key].shape[1]
    return None


def infer_num_classes(state_dict):
    if not isinstance(state_dict, dict):
        return None
    for key in (
        "mlp.0.weight",
        "model.mlp.0.weight",
        "mlp.2.weight",
        "model.mlp.2.weight",
        "readout.6.weight",
        "model.readout.6.weight",
        "readout.4.weight",
        "model.readout.4.weight",
    ):
        if key in state_dict:
            return state_dict[key].shape[0]
    return None


def load_model_weights(model, path, state=None, allow_partial=False):
    if state is None:
        state = torch.load(path, map_location=device, weights_only=False)
    state = extract_state_dict(state)

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError as exc:
        model_state = model.state_dict()
        missing = [key for key in model_state if key not in state]
        unexpected = [key for key in state if key not in model_state]
        shape_mismatch = [
            key
            for key in state.keys() & model_state.keys()
            if hasattr(state[key], "shape")
            and hasattr(model_state[key], "shape")
            and state[key].shape != model_state[key].shape
        ]

        if not missing and not shape_mismatch:
            filtered_state = {key: value for key, value in state.items() if key in model_state}
            model.load_state_dict(filtered_state, strict=True)
            if unexpected:
                print(f"Ignored {len(unexpected)} unused checkpoint keys.")
            return

        if not allow_partial:
            raise RuntimeError(
                "Checkpoint does not exactly match the model. "
                "Pass --allow_partial_load to load with strict=False."
            ) from exc
        print("Strict checkpoint load failed; retrying with strict=False.")
        model.load_state_dict(state, strict=False)


def build_model(model_name, F_in, n_classes, cutoff, state_dict):
    if model_name == "bige3nn":
        from models.e3nn import BigE3NN_NoSH

        return BigE3NN_NoSH(F_in=F_in, n_classes=n_classes, cutoff=cutoff)
    if model_name == "mace":
        from models.mace import MACEProteinNet

        return MACEProteinNet(F_in=F_in, n_classes=n_classes, cutoff=cutoff)
    if model_name == "gotennet":
        from models.gotennet import GotenNetProtein

        inferred_basis = infer_gotennet_basis(state_dict) or 96
        return GotenNetProtein(
            n_atom_basis=inferred_basis,
            n_classes=n_classes,
            cutoff=cutoff,
            n_interactions=4,
        )
    raise ValueError(f"Unknown model type: {model_name}")


def get_raw_test_loader(args):
    add_tedbench_path(args)
    try:
        from tedbench.data import LightningStructureDataset, TEDLightningDataset
    except ImportError as exc:
        raise RuntimeError(
            "Could not import tedbench.data. Use a precomputed test graph file or pass "
            "--tedbench_path pointing to the original TEDBench repo."
        ) from exc

    if args.external_dataset_root:
        dataset = LightningStructureDataset(
            root=args.external_dataset_root,
            dataset_name=args.external_dataset_name,
            batch_size=args.dataset_batch_size,
        )
        split = args.external_split.lower()
        stage = "fit" if split == "train" else split
        dataset.setup(stage)

        if split == "train" and hasattr(dataset, "train_dataloader"):
            return dataset.train_dataloader(shuffle=False)
        if split == "val" and hasattr(dataset, "val_dataloader"):
            return dataset.val_dataloader()
        if split == "test" and hasattr(dataset, "test_dataloader"):
            return dataset.test_dataloader()
        raise ValueError(f"No dataloader available for split '{split}'.")

    dataset = TEDLightningDataset(
        root=args.dataset_root,
        dataset_name=args.dataset_name,
        batch_size=args.dataset_batch_size,
    )
    dataset.setup()
    return dataset.test_dataloader()


def load_or_build_test_graphs(args):
    test_path = args.test_graphs_path
    if args.external_dataset_root and test_path == "test_graphs.pt":
        test_path = f"test_graphs_{args.external_dataset_name}_{args.external_split}.pt"

    resolved_path = resolve_existing_path(test_path, SCRIPT_DIR)
    if resolved_path.exists():
        test_graphs = torch.load(resolved_path, map_location="cpu", weights_only=False)
        print(f"Loaded precomputed test graphs: {len(test_graphs)} from {resolved_path}")
        return test_graphs

    print("Precomputing test graphs...")
    raw_loader = get_raw_test_loader(args)
    test_batches = build_graph_batches(
        raw_loader,
        feature_mode=args.feature_mode,
        cutoff=args.cutoff,
    )
    test_graphs = batches_to_protein_graphs(test_batches)
    torch.save(test_graphs, test_path)
    print(f"Saved test graphs to {test_path}")
    return test_graphs


def run_inference(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = []
    y_pred = []
    unpack_data = model.unpack_data if hasattr(model, "unpack_data") else True

    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", unit="batch"):
            data = data.to(device)
            y = data.y.view(-1).long()

            if unpack_data:
                pos = data.pos
                x = data.x
                edge_index = data.edge_index
                batch_idx = data.batch

                row, col = edge_index
                edge_vec = pos[col] - pos[row]
                dist = edge_vec.norm(dim=1)

                logits, _ = model(
                    pos=pos,
                    x=x,
                    edge_index=edge_index,
                    edge_vec=edge_vec,
                    dist=dist,
                    batch_idx=batch_idx,
                )
            else:
                data.z = data.x
                logits, _ = model(data)

            loss = criterion(logits, y)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            y_true.append(y.detach().cpu())
            y_pred.append(preds.detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return total_loss / total_samples, total_correct / total_samples, y_true, y_pred


def main():
    args = parse_args()
    print("Using device:", device)

    model_path = resolve_existing_path(args.model_path, SCRIPT_DIR)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {args.model_path}")
    print("Loading model weights from:", model_path)

    state = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(state)
    n_classes = args.n_classes or infer_num_classes(state_dict)
    if n_classes is None:
        raise ValueError("Could not infer n_classes; pass --n_classes explicitly.")
    print("Num classes:", n_classes)

    test_graphs = load_or_build_test_graphs(args)
    test_loader = PygDataLoader(
        test_graphs,
        batch_size=args.test_batch_size,
        shuffle=False,
    )

    F_in = test_graphs[0].x.shape[1]
    model = build_model(args.model, F_in, n_classes, args.cutoff, state_dict).to(device)
    load_model_weights(model, model_path, state=state, allow_partial=args.allow_partial_load)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = run_inference(model, test_loader, criterion)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print("\nEvaluation")
    print(
        format_table(
            ["metric", "value"],
            [
                ["test_loss", f"{test_loss:.4f}"],
                ["accuracy", f"{test_acc:.4f}"],
                ["balanced_accuracy", f"{bal_acc:.4f}"],
                ["f1_macro", f"{f1:.4f}"],
            ],
            right_align={1},
        )
    )


if __name__ == "__main__":
    main()
