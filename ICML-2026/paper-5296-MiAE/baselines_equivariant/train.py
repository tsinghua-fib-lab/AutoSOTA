import argparse
import copy
import os
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
from torch import nn
from torch_geometric.loader import DataLoader as PygDataLoader
from tqdm import tqdm

from data import (
    batches_to_protein_graphs,
    build_graph_batches,
    infer_num_classes_from_graphs,
    resolve_existing_path,
)


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

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--true_batch_size", type=int, default=64)

    parser.add_argument("--feature_mode", type=str, default="constant", choices=["constant", "token"])
    parser.add_argument("--cutoff", type=float, default=8.0)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--scheduler_factor", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=5)

    parser.add_argument(
        "--model",
        type=str,
        default="bige3nn",
        choices=["bige3nn", "mace", "gotennet"],
    )
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"])

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subset_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_classes", type=int, default=None)

    parser.add_argument("--train_graphs_path", type=str, default="train_graphs.pt")
    parser.add_argument("--val_graphs_path", type=str, default="val_graphs.pt")
    parser.add_argument("--test_graphs_path", type=str, default="test_graphs.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--skip_final_test", action="store_true")

    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--disable_wandb", action="store_true")

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


def load_raw_ted_graphs(args):
    add_tedbench_path(args)
    try:
        from tedbench.data import TEDLightningDataset
    except ImportError as exc:
        raise RuntimeError(
            "Could not import tedbench.data. Use precomputed graph files or pass "
            "--tedbench_path pointing to the original TEDBench repo."
        ) from exc

    dataset = TEDLightningDataset(
        root=args.dataset_root,
        dataset_name=args.dataset_name,
        batch_size=args.dataset_batch_size,
    )
    dataset.setup()

    print("Precomputing train graphs...")
    train_batches = build_graph_batches(
        dataset.train_dataloader(shuffle=True),
        feature_mode=args.feature_mode,
        cutoff=args.cutoff,
    )
    print("Precomputing val graphs...")
    val_batches = build_graph_batches(
        dataset.val_dataloader(),
        feature_mode=args.feature_mode,
        cutoff=args.cutoff,
    )

    train_graphs = batches_to_protein_graphs(train_batches)
    val_graphs = batches_to_protein_graphs(val_batches)

    torch.save(train_graphs, args.train_graphs_path)
    torch.save(val_graphs, args.val_graphs_path)
    print(f"Saved graphs to {args.train_graphs_path} and {args.val_graphs_path}")
    return train_graphs, val_graphs, dataset.train_dataset.num_classes


def load_or_build_graphs(args):
    train_path = resolve_existing_path(args.train_graphs_path, SCRIPT_DIR)
    val_path = resolve_existing_path(args.val_graphs_path, SCRIPT_DIR)

    if train_path.exists() and val_path.exists():
        train_graphs = torch.load(train_path, map_location="cpu", weights_only=False)
        val_graphs = torch.load(val_path, map_location="cpu", weights_only=False)
        print(
            f"Loaded precomputed graphs: train={len(train_graphs)} from {train_path}, "
            f"val={len(val_graphs)} from {val_path}"
        )
        n_classes = args.n_classes or infer_num_classes_from_graphs(train_graphs + val_graphs)
        return train_graphs, val_graphs, n_classes

    return load_raw_ted_graphs(args)


def load_or_build_test_graphs(args):
    test_path = resolve_existing_path(args.test_graphs_path, SCRIPT_DIR)
    if test_path.exists():
        test_graphs = torch.load(test_path, map_location="cpu", weights_only=False)
        print(f"Loaded precomputed test graphs: test={len(test_graphs)} from {test_path}")
        return test_graphs

    add_tedbench_path(args)
    try:
        from tedbench.data import TEDLightningDataset
    except ImportError as exc:
        raise RuntimeError(
            "Could not import tedbench.data to run final test eval. "
            "Provide --test_graphs_path or pass --skip_final_test."
        ) from exc

    dataset = TEDLightningDataset(
        root=args.dataset_root,
        dataset_name=args.dataset_name,
        batch_size=args.dataset_batch_size,
    )
    dataset.setup()

    print("Precomputing test graphs...")
    test_batches = build_graph_batches(
        dataset.test_dataloader(),
        feature_mode=args.feature_mode,
        cutoff=args.cutoff,
    )
    test_graphs = batches_to_protein_graphs(test_batches)
    torch.save(test_graphs, args.test_graphs_path)
    print(f"Saved test graphs to {args.test_graphs_path}")
    return test_graphs


def build_model(model_name, F_in, n_classes, cutoff):
    if model_name == "bige3nn":
        from models.e3nn import BigE3NN_NoSH

        return BigE3NN_NoSH(F_in=F_in, n_classes=n_classes, cutoff=cutoff)
    if model_name == "mace":
        from models.mace import MACEProteinNet

        return MACEProteinNet(F_in=F_in, n_classes=n_classes, cutoff=cutoff)
    if model_name == "gotennet":
        from models.gotennet import GotenNetProtein

        return GotenNetProtein(
            n_atom_basis=96,
            n_classes=n_classes,
            cutoff=cutoff,
            n_interactions=4,
        )
    raise ValueError(f"Unknown model type: {model_name}")


class CombinedOptimizer:
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()


class SchedulerWrapper:
    def __init__(self, *schedulers):
        self.schedulers = schedulers

    def step(self, metric):
        for scheduler in self.schedulers:
            scheduler.step(metric)


def make_optimizer_and_scheduler(model, args):
    if args.optimizer == "adamw" or not hasattr(torch.optim, "Muon"):
        if args.optimizer == "muon":
            print("torch.optim.Muon is unavailable; falling back to AdamW.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
        )
        return optimizer, scheduler

    param_2d = []
    param_rest = []
    for param in model.parameters():
        if param.dim() == 2:
            param_2d.append(param)
        else:
            param_rest.append(param)

    optimizers = []
    schedulers = []
    if param_rest:
        optimizer_1d = torch.optim.AdamW(
            param_rest,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        optimizers.append(optimizer_1d)
        schedulers.append(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_1d,
                mode="min",
                patience=args.scheduler_patience,
                factor=args.scheduler_factor,
            )
        )
    if param_2d:
        optimizer_2d = torch.optim.Muon(
            param_2d,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        optimizers.append(optimizer_2d)
        schedulers.append(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_2d,
                mode="min",
                patience=args.scheduler_patience,
                factor=args.scheduler_factor,
            )
        )

    return CombinedOptimizer(*optimizers), SchedulerWrapper(*schedulers)


def run_epoch(model, loader, criterion, optimizer=None, accum_steps=1, max_grad_norm=1.0):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    unpack_data = model.unpack_data if hasattr(model, "unpack_data") else True

    if train:
        optimizer.zero_grad()

    last_step = 0
    for step, data in enumerate(loader, start=1):
        last_step = step
        data = data.to(device)
        y = data.y.view(-1).long()

        with torch.set_grad_enabled(train):
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

            raw_loss = criterion(logits, y)
            loss = raw_loss / accum_steps

            if train:
                loss.backward()
                if step % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

        with torch.no_grad():
            total_loss += raw_loss.item() * y.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    if train and last_step % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_samples, total_correct / total_samples


def maybe_debug_subset(train_graphs, val_graphs, args):
    if not args.debug:
        return train_graphs, val_graphs

    generator = torch.Generator().manual_seed(args.seed)

    train_size = max(1, int(len(train_graphs) * args.subset_fraction))
    train_idx = torch.randperm(len(train_graphs), generator=generator)[:train_size].tolist()

    val_size = max(1, int(len(val_graphs) * args.subset_fraction))
    val_idx = torch.randperm(len(val_graphs), generator=generator)[:val_size].tolist()

    print(f"DEBUG mode: train={train_size}/{len(train_graphs)}, val={val_size}/{len(val_graphs)}")
    return [train_graphs[i] for i in train_idx], [val_graphs[i] for i in val_idx]


def init_wandb(args, model, F_in, n_classes, true_bs, total_params):
    if args.disable_wandb or not args.wandb_project:
        return None

    import wandb

    run_name = args.wandb_run_name or (
        f"{args.model}_bs{true_bs}_lr{args.lr}_cut{args.cutoff}_"
        f"seed{args.seed}_params{total_params}"
    )
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "model_type": type(model).__name__,
            "F_in": F_in,
            "n_classes": n_classes,
            "total_params": total_params,
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "true_batch_size": true_bs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "cutoff": args.cutoff,
            "epochs": args.epochs,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_factor": args.scheduler_factor,
            "debug": args.debug,
            "subset_fraction": args.subset_fraction,
            "seed": args.seed,
        },
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    print("Using device:", device)

    train_graphs, val_graphs, n_classes = load_or_build_graphs(args)
    if n_classes is None:
        raise ValueError("Could not infer n_classes; pass --n_classes explicitly.")

    train_graphs, val_graphs = maybe_debug_subset(train_graphs, val_graphs, args)
    print(f"#train proteins: {len(train_graphs)}, #val proteins: {len(val_graphs)}")
    print("Num classes:", n_classes)

    train_batch_size = args.train_batch_size
    true_bs = max(args.true_batch_size, train_batch_size)
    accum_steps = max(1, true_bs // train_batch_size)
    print(f"Using gradient accumulation steps: {accum_steps} to achieve true bs={true_bs}")

    train_loader = PygDataLoader(train_graphs, batch_size=train_batch_size, shuffle=True)
    val_loader = PygDataLoader(val_graphs, batch_size=args.val_batch_size, shuffle=False)

    F_in = train_graphs[0].x.shape[1]
    model = build_model(args.model, F_in, n_classes, args.cutoff).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = make_optimizer_and_scheduler(model, args)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    wandb_run = init_wandb(args, model, F_in, n_classes, true_bs, total_params)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"best_model_{args.model}_seed{args.seed}.pt"

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            accum_steps=accum_steps,
            max_grad_norm=args.max_grad_norm,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "trn_loss": train_loss,
                    "trn_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": max(best_val_acc, val_acc),
                },
                step=epoch,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(best_state, checkpoint_path)
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch:04d} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f} | "
            f"best val acc {best_val_acc:.4f}"
        )

        if epochs_no_improve >= args.early_stop_patience:
            print(
                "Early stopping: no val acc improvement for "
                f"{args.early_stop_patience} epochs (best epoch {best_epoch})."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (val acc = {best_val_acc})")
        print(f"Saved best checkpoint to {checkpoint_path}")

        if not args.skip_final_test:
            test_graphs = load_or_build_test_graphs(args)
            test_loader = PygDataLoader(
                test_graphs,
                batch_size=args.val_batch_size,
                shuffle=False,
            )
            test_loss, test_acc = run_epoch(model, test_loader, criterion)
            print(f"Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                    }
                )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
