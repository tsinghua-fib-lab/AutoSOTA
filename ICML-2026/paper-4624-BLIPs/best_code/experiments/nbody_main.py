import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from blip.model import MessagePassingNetwork, EquivariantGraphNeuralNetwork
from blip.posterior import PosteriorModel
from blip.bayes import BayesianModelWrapper, KL

from utils import (
    directory_handler,
    set_seed_precision,
    run,
    choose_activation,
    load_checkpoint,
)

from nbody.dataset import NBodyGeometricDataset
from nbody.args import parse_args

warnings.filterwarnings("ignore")


class RegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch=None):
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear_out(x)
        return x


def main():
    args = parse_args()
    set_seed_precision(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    directory_handler(["./logs", "./ckpt", "./data"])

    # 0. Download the dataset for training
    model_id = "GNN" if args.model_id in ["GNN", "DropGNN", "BayesGNN"] else "EGNN"
    train_dataset = NBodyGeometricDataset(
        partition="train",
        dataset_name=args.dataset,
        max_samples=args.max_training_samples,
        model_id=model_id,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
    )
    val_dataset = NBodyGeometricDataset(
        partition="val", dataset_name=args.dataset, model_id=model_id
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        generator=g,
    )
    test_dataset = NBodyGeometricDataset(
        partition="test", dataset_name=args.dataset, model_id=model_id
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        generator=g,
    )
    # 2. Build the Network
    if model_id == "GNN":
        node_dim = 6
        edge_dim = 1
        Module = MessagePassingNetwork
        regression_head = RegressionHead(
            args.hidden_dim, args.hidden_dim_regression_head, 3
        )
    elif model_id == "EGNN":
        node_dim = 1
        edge_dim = 2
        Module = EquivariantGraphNeuralNetwork
        regression_head = None
    # extra kwargs
    kwargs = {}
    if args.model_id == "GNN" or args.model_id == "EGNN":
        args.mc_steps = 0
    if args.model_id == "DropGNN" or args.model_id == "DropEGNN":
        kwargs["dropout"] = args.dropout
    else:
        kwargs["dropout"] = None
    # build model
    model = Module(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        n_message_layers=args.n_message_layers,
        n_update_layers=args.n_update_layers,
        num_layers=args.num_layers,
        activation=choose_activation(args.activation),
        regression_head=regression_head,
        **kwargs,
    )
    # if Bayesian
    if args.model_id == "BayesGNN" or args.model_id == "BayesEGNN":
        track_data = next(iter(val_loader))
        model = BayesianModelWrapper(
            model,
            kl=KL(args.prior_probability, args.beta),
        )
        model.warm_up(
            num_nodes=track_data.num_nodes,
            num_edges=track_data.num_edges,
            batch=track_data,
            regex_pattern=[r"^blocks\."],
        )
        posterior = PosteriorModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_alphas_edge=model.num_alphas_edge,
            num_alphas_node=model.num_alphas_node,
            hidden_dim=args.hidden_dim_posterior,
            n_node_layers=args.n_node_layers,
            n_edge_layers=args.n_edge_layers,
            activation=choose_activation(args.activation_posterior),
        )
        model.add_posterior_net(posterior)

    ## 5. Train/Test
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-12)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    criterion = torch.nn.MSELoss()
    writer = SummaryWriter(
        log_dir=f"logs/nbody/{args.seed}/{args.model_id}_{args.dataset}_logs/"
    )

    if args.run_type == "train":
        run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=args.device,
            writer=writer,
            ckpt_file=f"ckpt/nbody/{args.seed}/{args.model_id}_nbody_ckpt.pth",
            run_type=args.run_type,
            max_epochs=args.max_epochs,
        )
    else:
        load_checkpoint(
            f"ckpt/nbody/{args.seed}/{args.model_id}_nbody_ckpt.pth",
            model,
            args.device,
            optimizer,
            scheduler,
        )
        stats = {
            "true_positions": [],
            "mean_positions": [],
            "var_positions": [],
        }
        for data in test_loader:
            data = data.to(args.device)
            if args.mc_steps > 0:
                total_output = []
                for _ in range(args.mc_steps):
                    out = model(batch=data)
                    total_output.append(out.cpu().detach())
                total_output = torch.stack(total_output, dim=0)
                mean = total_output.mean(dim=0)
                var = total_output.var(dim=0)
            else:
                model.eval()
                mean = model(batch=data)
                var = None
            # append output
            stats["true_positions"].append(data.y.cpu())
            stats["mean_positions"].append(mean.cpu())
            if var is not None:
                stats["var_positions"].append(var.cpu())
        # aggregate and compute extra metrics
        stats["true_positions"] = torch.cat(stats["true_positions"], dim=0)
        stats["mean_positions"] = torch.cat(stats["mean_positions"], dim=0)
        stats["var_positions"] = (
            torch.cat(stats["var_positions"], dim=0) if var is not None else None
        )

        # Save Results
        torch.save(stats, f"nbody_results_{args.model_id}_{args.seed}.pt")


if __name__ == "__main__":
    main()
