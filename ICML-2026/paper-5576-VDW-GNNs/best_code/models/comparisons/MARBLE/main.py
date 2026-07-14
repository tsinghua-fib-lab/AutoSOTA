"""Main network"""

import glob
import os
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as opt
import yaml
from torch import nn
from torch_geometric.nn import MLP
from tqdm import tqdm

from . import dataloader
from . import geometry
from . import layers
from . import utils


class net(nn.Module):
    """MARBLE neural network.

    The possible parameters and their default values are described below,
    and can be accessed via the `params` dictionnary in this class constructor.

    Args:
        batch_size: batch size (default=64)
        epochs: optimisation epochs (default=20)
        lr: iniital learning rate (default=0.01)
        momentum: momentum (default=0.9)
        diffusion: set to True to use diffusion layer before gradient computation (default=False)
        include_positions: include positions as features (warning: this is untested) (default=False)
        include_self: include vector at the center of feature (default=True)
        order: order to which to compute the directional derivatives (default=2)
        inner_product_features: transform gradient features to inner product features (default=True)
        frac_sampled_nb: fraction of neighbours to sample for gradient computation
            (if -1 then all neighbours) (default=-1)
        dropout: dropout in the MLP (default=0.)
        hidden_channels: number of hidden channels (default=16). If list, then adds multiple layers.
        out_channels: number of output channels (if null, then =hidden_channels) (default=3)
        bias: learn bias parameters in MLP (default=True)
        vec_norm: normalise features at each derivative order to unit length (default=False)
        emb_norm: normalise MLP output to unit length (default=False)
        batch_norm: batch normalisation (default=True)
        seed: seed for reproducibility
    """

    def __init__(self, data, loadpath=None, params=None, verbose=True):
        """
        Constructor of the MARBLE net.

        Args:
            data: PyG data
            loadpath: path to a model file, or a directory with models (best model will be used)
            params: dict with parameters to overwrite default params or a path to a yaml file
            verbose: run in verbose mode
        """
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if loadpath is not None:
            if Path(loadpath).is_dir():
                loadpath = max(glob.glob(f"{loadpath}/best_model*"))
            self.params = torch.load(loadpath, map_location=device)["params"]
        else:
            if params is not None:
                self.params = params
            else:
                self.params = {}

        self._epoch = 0  # to resume optimisation
        self.parse_parameters(data)
        self.check_parameters(data)
        self.setup_layers()
        self.loss = loss_fun()
        self.reset_parameters()
        self.timestamp = None

        if verbose:
            utils.print_settings(self)

        if loadpath is not None:
            self.load_model(loadpath)

    def parse_parameters(self, data):
        """Load default parameters and merge with user specified parameters"""

        file = os.path.dirname(__file__) + "/default_params.yaml"
        with open(file, "rb") as f:
            params = yaml.safe_load(f)

        params["dim_signal"] = data.x.shape[1]
        params["dim_emb"] = data.pos.shape[1]

        if hasattr(data, "dim_man"):
            params["dim_man"] = data.dim_man

        # merge dictionaries without duplications
        for key in params.keys():
            if key not in self.params.keys():
                self.params[key] = params[key]

        if params["frac_sampled_nb"] != -1:
            self.params["n_sampled_nb"] = int(data.degree * params["frac_sampled_nb"])
        else:
            self.params["n_sampled_nb"] = -1

        if self.params["batch_norm"]:
            self.params["batch_norm"] = "batch_norm"
        else:
            self.params["batch_norm"] = None

    def check_parameters(self, data):
        """Check parameter validity"""

        assert self.params["order"] > 0, "Derivative order must be at least 1!"

        if self.params["vec_norm"]:
            assert data.x.shape[1] > 1, "Using vec_norm=True is not permitted for scalar signals"

        if self.params["diffusion"]:
            assert hasattr(data, "L"), "No Laplacian found. Compute it in preprocessing()!"

        pars = [
            "batch_size",
            "epochs",
            "lr",
            "momentum",
            "order",
            "inner_product_features",
            "dim_signal",
            "dim_emb",
            "frac_sampled_nb",
            "dropout",
            "diffusion",
            "hidden_channels",
            "out_channels",
            "bias",
            "batch_norm",
            "vec_norm",
            "emb_norm",
            "seed",
            "include_positions",
            "include_self",
        ]

        for p in pars:
            assert p in list(self.params.keys()), f"Parameter {p} is not specified!"

    def reset_parameters(self):
        """reset parmaeters."""
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def setup_layers(self):
        """Setup layers."""

        s, d, o = self.params["dim_signal"], self.params["dim_emb"], self.params["order"]
        if "dim_man" in self.params.keys():
            s = d = self.params["dim_man"]

        # diffusion
        self.diffusion = layers.Diffusion()

        # gradient features
        self.grad = nn.ModuleList(layers.AnisoConv() for i in range(o))

        # cumulated number of channels after gradient features
        cum_channels = s * (1 - d ** (o + 1)) // (1 - d)
        if not self.params["include_self"]:
            cum_channels -= s

        if self.params["inner_product_features"]:
            cum_channels //= s
            if s == 1:
                cum_channels = o + 1

            self.inner_products = layers.InnerProductFeatures(cum_channels, s)
        else:
            self.inner_products = None

        if self.params["include_positions"]:
            cum_channels += d

        # encoder
        if not isinstance(self.params["hidden_channels"], list):
            self.params["hidden_channels"] = [self.params["hidden_channels"]]

        channel_list = (
            [cum_channels] + self.params["hidden_channels"] + [self.params["out_channels"]]
        )

        self.enc = MLP(
            channel_list=channel_list,
            dropout=self.params["dropout"],
            bias=self.params["bias"],
            norm=self.params["batch_norm"],
        )

    def forward(self, data, n_id, adjs=None):
        """Forward pass.
        Messages are passed to a set target nodes (current batch) from source
        nodes. The source nodes and target nodes form a bipartite graph to
        simplify message passing. By convention, the first size[1] entries of x
        are the target nodes, i.e, x = concat[x_target, x_other]."""

        x = data.x
        n, d = x.shape[0], data.gauges.shape[2]
        mask = data.mask

        # diffusion
        if self.params["diffusion"]:
            if hasattr(data, "Lc"):
                x = geometry.global_to_local_frame(x, data.gauges)
                x = self.diffusion(x, data.L, Lc=data.Lc, method="spectral")
                x = geometry.global_to_local_frame(x, data.gauges, reverse=True)
            else:
                x = self.diffusion(x, data.L, method="spectral")

        # local gauges
        if self.params["inner_product_features"]:
            x = geometry.global_to_local_frame(x, data.gauges)

        # restrict to current batch
        x = x[n_id]
        mask = mask[n_id]
        if data.kernels[0].size(0) == n * d:
            n_id = utils.expand_index(n_id, d)
        else:
            d = 1

        if self.params["vec_norm"]:
            x = F.normalize(x, dim=-1, p=2)

        # gradients
        if self.params["include_self"]:
            out = [x]
        else:
            out = []
        for i, (_, _, size) in enumerate(adjs):
            kernels = [K[n_id[: size[1] * d], :][:, n_id[: size[0] * d]] for K in data.kernels]

            x = self.grad[i](x, kernels)

            if self.params["vec_norm"]:
                x = F.normalize(x, dim=-1, p=2)

            out.append(x)

        last_size = adjs[-1][2]
        # take target nodes
        out = [o[: last_size[1]] for o in out]

        # inner products
        if self.params["inner_product_features"]:
            out = self.inner_products(out)
        else:
            out = torch.cat(out, axis=1)

        if self.params["include_positions"]:
            out = torch.hstack([data.pos[n_id[: last_size[1]]], out])

        target_device = next(self.enc.parameters()).device
        out = out.to(target_device)
        emb = self.enc(out)

        if self.params["emb_norm"]:  # spherical output
            emb = F.normalize(emb)

        return emb, mask[: last_size[1]]

    def evaluate(self, data):
        """Evaluate."""
        warnings.warn("MARBLE.evaluate() is deprecated. Use MARBLE.transform() instead.")
        return self.transform(data)

    def transform(self, data):
        """Forward pass @ evaluation (no minibatches)"""
        with torch.no_grad():
            size = (data.x.shape[0], data.x.shape[0])
            adjs = utils.EdgeIndex(data.edge_index, torch.arange(data.edge_index.shape[1]), size)
            adjs = utils.to_list(adjs) * self.params["order"]

            try:
                data.kernels = [
                    utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()).t()
                    for K in utils.to_list(data.kernels)
                ]
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            _, data, adjs = utils.move_to_gpu(self, data, adjs)
            out, _ = self.forward(data, torch.arange(len(data.x)), adjs)
            utils.detach_from_gpu(self, data, adjs)

            data.emb = out.detach().cpu()

            return data

    def batch_loss(self, data, loader, train=False, verbose=False, optimizer=None):
        """Loop over minibatches provided by loader function.

        Args:
            x : (nxdim) feature matrix
            loader : dataloader object from dataloader.py

        """

        if train:  # training mode (enables dropout in MLP)
            self.train()

        if verbose:
            print("\n")

        cum_loss = 0
        for batch in tqdm(loader, disable=not verbose):
            _, n_id, adjs = batch
            adjs = [adj.to(data.x.device) for adj in utils.to_list(adjs)]
            emb, mask = self.forward(data, n_id, adjs)
            loss = self.loss(emb, mask)
            cum_loss += float(loss)

            if optimizer is not None:
                optimizer.zero_grad()  # zero gradients, otherwise accumulates
                loss.backward()  # backprop
                optimizer.step()

        self.eval()

        return cum_loss / len(loader), optimizer

    def run_training(self, data, outdir=None, verbose=False):
        """Run training."""
        warnings.warn("MARBLE.run_training() is deprecated. Use MARBLE.fit() instead.")

        self.fit(data, outdir=outdir, verbose=verbose)

    def fit(self, data, outdir=None, verbose=False):
        """Network training.

        Args:
            data: PyG data
            outdir: folder to save intermediate models
            verbose: run in verbose mode
        """

        print("\n---- Training network ...")

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        print(f"\n---- Timestamp: {self.timestamp}")

        # load to gpu (if possible)
        # pylint: disable=self-cls-assignment
        self, data, _ = utils.move_to_gpu(self, data)

        # data loader
        train_loader, val_loader, test_loader = dataloader.loaders(data, self.params)
        optimizer = opt.SGD(
            self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"]
        )
        if hasattr(self, "optimizer_state_dict"):
            optimizer.load_state_dict(self.optimizer_state_dict)

        # training scheduler
        scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer)

        best_loss = -1
        self.losses = {"train_loss": [], "val_loss": [], "test_loss": []}
        for epoch in range(
            self.params.get("epoch", 0), self.params.get("epoch", 0) + self.params["epochs"]
        ):
            self._epoch = epoch

            train_loss, optimizer = self.batch_loss(
                data, train_loader, train=True, verbose=verbose, optimizer=optimizer
            )
            val_loss, _ = self.batch_loss(data, val_loader, verbose=verbose)
            scheduler.step(train_loss)

            print(
                f"\nEpoch: {self._epoch}, Training loss: {train_loss:4f}, Validation loss: {val_loss:.4f}, lr: {scheduler._last_lr[0]:.4f}",  # noqa, pylint: disable=line-too-long,protected-access
                end="",
            )

            if best_loss == -1 or (val_loss < best_loss):
                outdir = self.save_model(
                    optimizer, self.losses, outdir=outdir, best=True, timestamp=self.timestamp
                )
                best_loss = val_loss
                print(" *", end="")

            self.losses["train_loss"].append(train_loss)
            self.losses["val_loss"].append(val_loss)

        test_loss, _ = self.batch_loss(data, test_loader)
        print(f"\nFinal test loss: {test_loss:.4f}")

        self.losses["test_loss"].append(test_loss)

        self.save_model(optimizer, self.losses, outdir=outdir, best=False, timestamp=self.timestamp)
        self.load_model(os.path.join(outdir, f"best_model_{self.timestamp}.pth"))

    def load_model(self, loadpath):
        """Load model.

        Args:
            loadpath: directory with models to load best model, or specific model path
        """
        checkpoint = torch.load(
            loadpath, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._epoch = checkpoint["epoch"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        if hasattr(self, "losses"):
            self.losses = checkpoint["losses"]

    def save_model(self, optimizer, losses, outdir=None, best=False, timestamp=""):
        """Save model."""
        if outdir is None:
            outdir = "./outputs/"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        checkpoint = {
            "epoch": self._epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "time": timestamp,
            "params": self.params,
            "losses": losses,
        }

        if best:
            fname = "best_model_"
        else:
            fname = "last_model_"

        fname += timestamp
        fname += ".pth"

        if best:
            torch.save(checkpoint, os.path.join(outdir, fname))
        else:
            torch.save(checkpoint, os.path.join(outdir, fname))

        return outdir


class loss_fun(nn.Module):
    """Loss function."""

    def forward(self, out, mask=None):
        """forward."""
        z, z_pos, z_neg = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((z * z_pos).sum(-1)).mean()  # pylint: disable=not-callable
        neg_loss = F.logsigmoid(-(z * z_neg).sum(-1)).mean()  # pylint: disable=not-callable

        coagulation_loss = 0.0
        if mask is not None:
            z_mask = out[mask]
            coagulation_loss = (z_mask - z_mask.mean(dim=0)).norm(dim=1).sum()

        return -pos_loss - neg_loss + torch.sigmoid(coagulation_loss) - 0.5
