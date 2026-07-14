import warnings
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from blip.posterior import PosteriorModelORB as PosteriorModel
from blip.bayes import BayesianModelWrapper, KL

from orb_models.forcefield import base
from orb_models.forcefield import pretrained

from utils import (
    directory_handler,
    set_seed_precision,
    run,
    load_checkpoint,
    choose_activation,
)

from silica.dataset import SilicaDataset
from silica.args import parse_args

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


@torch.no_grad()
def compute_output_sample(model, batch, mc_steps, device, test_loader):
    energies = []
    forces = []
    for _ in range(mc_steps):
        outputs = []
        for batch in test_loader:
            batch = batch.to(device)
            out = model.predict(batch=batch)
            energy = out["energy"].cpu()
            forces_per_batch = torch.stack(
                torch.split(out["forces"].cpu(), [699] * len(energy))
            )
            outputs.append({"energy": energy, "forces": forces_per_batch})
        energies.append(torch.cat([out["energy"] for out in outputs], dim=0))
        forces.append(torch.cat([out["forces"] for out in outputs], dim=0))
    energies = torch.stack(energies, dim=0)
    forces = torch.stack(forces, dim=0)
    return energies, forces


@torch.no_grad()
def compute_output_map(model, batch, mc_steps, device, test_loader):
    model.eval()
    energies = []
    forces = []
    outputs = []
    for batch in test_loader:
        batch = batch.to(device)
        out = model.predict(batch=batch)
        energy = out["energy"].cpu()
        forces_per_batch = torch.stack(
            torch.split(out["forces"].cpu(), [699] * len(energy))
        )
        outputs.append({"energy": energy, "forces": forces_per_batch})
    energies = torch.cat([out["energy"] for out in outputs], dim=0)
    forces = torch.cat([out["forces"] for out in outputs], dim=0)
    return energies, forces


def main():
    args = parse_args()
    set_seed_precision(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    directory_handler(["./logs", "./ckpt", "./data"])

    # 1. Download the dataset for training
    train_dataset = SilicaDataset(partition="train", train_size=args.train_size)
    train_loader = DataLoader(
        train_dataset, batch_size=8, generator=g, collate_fn=base.batch_graphs
    )
    val_dataset = SilicaDataset(partition="val")
    val_loader = DataLoader(
        val_dataset, batch_size=8, generator=g, collate_fn=base.batch_graphs
    )
    test_dataset = SilicaDataset(partition="test")
    test_loader = DataLoader(
        test_dataset, batch_size=8, generator=g, collate_fn=base.batch_graphs
    )

    # 2. Download ORB pretrained-model
    model = getattr(pretrained, "orb_v3_direct_20_omat")(
        device=args.device, compile=False, precision="float32-high"
    )
    model.heads.pop("stress")  # remove stress head
    model.heads.pop("confidence")

    # set the bayesian finetuing
    if args.finetune_bayesian:
        model = BayesianModelWrapper(
            model=model,
            kl=KL(args.prior_probability, args.beta),
        )
        track_data = next(iter(train_loader)).to(args.device)
        model.warm_up(
            num_nodes=track_data.n_node.sum(),
            num_edges=track_data.n_edge.sum(),
            batch=track_data,
            regex_pattern=[
                r"^model\.gnn_stacks\.\d+\._(node_mlp|edge_mlp)\.mlp\.NN-\d+$"
            ],  # apply Bayesian on GNN
        )
        posterior = PosteriorModel(
            node_dim=track_data.node_features["atomic_numbers_embedding"].shape[-1],
            edge_dim=23,
            num_alphas_edge=model.num_alphas_edge,
            num_alphas_node=model.num_alphas_node,
            hidden_dim=args.hidden_dim_posterior,
            n_node_layers=args.n_node_layers,
            n_edge_layers=args.n_edge_layers,
            activation=choose_activation(args.activation_posterior),
        )
        model.add_posterior_net(posterior)
    ## 3. Train/Test
    model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad = True
    # set optimizer/scheduler
    lr = 1e-5
    num_epochs = 50
    if args.finetune_bayesian:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.model.parameters(), "lr": lr},
                {"params": model.posterior.parameters(), "lr": 1e-4},
            ],
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = torch.nn.L1Loss()
    name = "BayesFinetunedORB" if args.finetune_bayesian else "FinetunedORB"

    if args.run_type == "train":
        writer = SummaryWriter(
            log_dir=f"logs/silica/{args.seed}/{name}_{args.train_size}_logs/"
        )
        run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=args.device,
            writer=writer,
            ckpt_file=f"ckpt/silica/{args.seed}/{name}_{args.train_size}_ckpt.pth",
            run_type=args.run_type,
            max_epochs=num_epochs,
            loss="mlff_orb",
        )
    else:
        load_checkpoint(
            f"ckpt/silica/{args.seed}/{name}_{args.train_size}_ckpt.pth",
            model,
            args.device,
            optimizer,
            scheduler,
        )

        # get the data
        true_energies = []
        true_forces = []
        for batch in test_loader:
            true_energies.append(batch.system_targets["energy"].cpu())
            true_forces.append(batch.node_targets["forces"].cpu())
        true_energies = torch.cat(true_energies, dim=0)
        true_forces = torch.cat(true_forces, dim=0)
        true_forces = torch.stack(torch.split(true_forces, [699] * len(test_dataset)))

        # compute output
        if name == "FinetunedORB":
            compute_output_fn = compute_output_map
        if name == "BayesFinetunedORB":
            if args.mc_steps == 0:
                compute_output_fn = compute_output_map
                name = "BayesFinetunedORB_MAP"
            else:
                compute_output_fn = compute_output_sample
        writer = SummaryWriter(
            log_dir=f"logs/silica/{args.seed}/{name}_{args.train_size}_logs/"
        )
        energies, forces = compute_output_fn(
            model, batch, args.mc_steps, args.device, test_loader
        )

        # Convert eV to kcal/mol (just for reporting consistent results)
        true_forces = true_forces * 23.0605
        forces = forces * 23.0605
        true_energies = true_energies * 23.0605
        energies = energies * 23.0605

        # Save Results
        torch.save(
            {
                "energies": energies,
                "forces": forces,
                "true_energies": true_energies,
                "true_forces": true_forces,
            },
            f"silica_results_{name}.pt",
        )


if __name__ == "__main__":
    main()
