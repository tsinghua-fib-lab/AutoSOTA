import warnings
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from blip.posterior import PosteriorModelPaiNN as PosteriorModel
from blip.bayes import BayesianModelWrapper, KL
from blip.painn import PaiNN

from utils import (
    directory_handler,
    set_seed_precision,
    run,
    load_checkpoint,
    choose_activation,
)

from ammonia.dataset import AmmoniaDataset, collate_fn
from ammonia.args import parse_args


warnings.filterwarnings("ignore")


def compute_output(model, batch):
    out = model(batch=batch)
    energy = out["energy"]
    energy_grad = torch.stack(torch.split(out["energy_grad"], [4] * len(energy)))
    return energy.detach(), energy_grad.detach()

def main():
    args = parse_args()
    set_seed_precision(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    directory_handler(["./logs", "./ckpt", "./data"])

    # 1. Download the dataset for training
    train_dataset = AmmoniaDataset(partition="train")
    train_loader = DataLoader(
        train_dataset, batch_size=64, generator=g, collate_fn=collate_fn
    )
    val_dataset = AmmoniaDataset(partition="val")
    val_loader = DataLoader(
        val_dataset, batch_size=64, generator=g, collate_fn=collate_fn
    )
    test_dataset = AmmoniaDataset(partition="test")
    test_loader = DataLoader(
        test_dataset, batch_size=10, generator=g, collate_fn=collate_fn
    )

    # 2. Build the Network (same model parameters as https://www.nature.com/articles/s41524-023-01180-8)
    modelparams = {
        "n_atom_basis": 128,
        "n_filters": 128,
        "n_gaussians": 16,
        "n_convolutions": 3,
        "cutoff": 5.0,
        "trainable_gauss": False,
        "dropout_rate": 0.0,
        "activation": "shifted_softplus",
        "num_readout_layer": {
            "energy": 1,
        },
        "pool_dic": {
            "energy": {
                "name": "sum",
                "param": {},
            }
        },
        "output_keys": ["energy"],
        "grad_keys": ["energy_grad"],
    }
    model = BayesianModelWrapper(
        model=PaiNN(modelparams),
        kl=KL(args.prior_probability, args.beta),
    )
    track_data = next(iter(train_loader))
    model.warm_up(
        num_nodes=len(track_data["nxyz"]),
        num_edges=2 * len(track_data["nbr_list"]),
        batch=track_data,
        regex_pattern=[r"^(?!.*\.(u_mat|v_mat)(\.|$)).*$"],
    )
    posterior = PosteriorModel(
        node_dim=1,
        edge_dim=1,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=30, threshold=0.0001, min_lr=1e-7
    )
    criterion = torch.nn.L1Loss()
    writer = SummaryWriter(log_dir=f"logs/ammonia/{args.seed}/BayesPaiNN_logs/")

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
            ckpt_file=f"ckpt/ammonia/{args.seed}/BayesPaiNN_ammonia_ckpt.pth",
            run_type=args.run_type,
            max_epochs=500,
            loss="mlff",
        )
    else:
        load_checkpoint(
            f"ckpt/ammonia/{args.seed}/BayesPaiNN_ammonia_ckpt.pth",
            model,
            args.device,
            optimizer,
            scheduler,
        )
        # compute output
        energies = []
        forces = []
        true_energies = []
        true_forces = []
        for _ in range(args.mc_steps):
            outputs = []
            for batch in test_loader:
                batch = batch.to(args.device)
                energy, energy_grad = compute_output(model, batch)
                outputs.append({"energy": energy.cpu(), "energy_grad": energy_grad.cpu()})
            energies.append(torch.cat([out["energy"] for out in outputs], dim=0))
            forces.append(torch.cat([-out["energy_grad"] for out in outputs], dim=0))
        for batch in test_loader:
            true_energies.append(batch["energy"].cpu())
            true_forces.append(-batch["energy_grad"].cpu())
        energies = torch.stack(energies, dim=0)
        forces = torch.stack(forces, dim=0)
        true_energies = torch.cat(true_energies, dim=0)
        true_forces = torch.cat(true_forces, dim=0)
        true_forces = torch.stack(torch.split(true_forces, [4] * len(test_dataset)))
        # Save Results
        torch.save(
            {
                'energies' : energies,
                'forces' : forces,
                'true_energies' : true_energies,
                'true_forces' : true_forces
            },
            'ammonia_results.pt'
        )
        

if __name__ == "__main__":
    main()
