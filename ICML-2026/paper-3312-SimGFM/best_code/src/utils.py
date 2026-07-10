import os
import time
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs("graphs")
        os.makedirs("chains")
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs("graphs/" + args.general.name)
        os.makedirs("chains/" + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = X * norm_values[0] + norm_biases[0]
    E = E * norm_values[1] + norm_biases[1]
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def symmetrize_and_mask_diag(E):
    # symmetrize the edge matrix
    upper_triangular_mask = torch.zeros_like(E)
    indices = torch.triu_indices(row=E.size(1), col=E.size(2), offset=1)
    if len(E.shape) == 4:
        upper_triangular_mask[:, indices[0], indices[1], :] = 1
    else:
        upper_triangular_mask[:, indices[0], indices[1]] = 1
    E = E * upper_triangular_mask
    E = E + torch.transpose(E, 1, 2)
    # mask the diagonal
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag] = 0

    return E


def pyg_data_to_place_holder(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    """
    X.shape: [B, N, F]
    node_mask.shape: [B, N]
    """
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(
        edge_index, edge_attr
    )
    max_num_nodes = X.size(1)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = add_no_edge_flag_in_edge_attr(E)
    """
    E.shape: [B, N, N, F]
    """
    return PlaceHolder(X=X, E=E, y=None), node_mask


def add_no_edge_flag_in_edge_attr(E):
    """
    Performs 2 operations:
    1. Where there are no edges, fill edge_attr[...,0] with 1
    2. Fill diagonal with edge_attr[...,:]  with 0, removing self-loops but edge_attr[...,0] remains 0
    """
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge_mask = torch.sum(E, dim=3) == 0
    edge_flag_in_edge_attr = E[:, :, :, 0]
    edge_flag_in_edge_attr[no_edge_mask] = 1
    E[:, :, :, 0] = edge_flag_in_edge_attr
    diag_mask = (
        torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    )
    E[diag_mask] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        """
        X.shape: [B, N, F]
        E.shape: [B, N, N, F]
        y.shape: [B, F]
        node_mask.shape: [B, N]
        """
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def to_device(self, device):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device) if self.y is not None else None
        return self

    def mask(self, node_mask, one_hot_to_index=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if one_hot_to_index:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def __repr__(self):
        return (
            f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def split(self, node_mask):
        """
        Split a PlaceHolder representing a batch into
        a list of placeholders representing individual graphs."""
        graph_list = []
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            n = torch.sum(node_mask[i], dim=0)
            x = self.X[i, :n]
            e = self.E[i, :n, :n]
            y = self.y[i] if self.y is not None else None
            graph_list.append(PlaceHolder(X=x, E=e, y=y))
        return graph_list

    # def cat(self, other: PlaceHolder) -> PlaceHolder:
    #     X = torch.cat(self.X, other.X, dim=-1)
    #     E = torch.cat(self.E, other.E, dim=-1)
    #     y = torch.cat(self.y, other.y, dim=-1)
    #     return PlaceHolder(X=X, E=E, y=y)


def setup_wandb(cfg):
    # Ensure a valid wandb directory exists to avoid FileNotFoundError for logs
    wandb_dir = os.path.join(os.getcwd(), "wandb")
    print(f"wandb_dir: {wandb_dir}")
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    kwargs = {
        "name": cfg.wandb.name,
        "project": cfg.wandb.project,
        "group": cfg.wandb.group,
        "entity": cfg.wandb.entity,
        "config": config_dict,
        "settings": wandb.Settings(
            _disable_stats=True,
            log_internal=os.path.join(wandb_dir, "debug-internal.log"),
            log_user=os.path.join(wandb_dir, "debug.log"),
        ),
        "reinit": True,
        "mode": cfg.wandb.mode,
        "dir": wandb_dir,
    }
    config_dict["general"]["local_dir"] = os.getcwd()
    wandb.init(**kwargs)
    wandb.save("*.txt")  # Upload all txt files in the same directory as wandb


def make_wandb_logger(cfg):
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    # Ensure a valid wandb directory exists to avoid FileNotFoundError for logs
    wandb_dir = os.path.join(os.getcwd(), "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    print(f"wandb_dir: {wandb_dir}")
    os.environ["WANDB_DIR"] = wandb_dir
    from pytorch_lightning.loggers import WandbLogger
    from src.wandb_resume_callback import get_wandb_id_from_checkpoint
    
    config_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    kwargs = {
        "name": cfg.wandb.name,
        "project": cfg.wandb.project,
        "group": cfg.wandb.group,
        "entity": cfg.wandb.entity,
        "config": config_dict,
        "settings": wandb.Settings(
            _disable_stats=True,
            log_internal=os.path.join(wandb_dir, "debug-internal.log"),
            log_user=os.path.join(wandb_dir, "debug.log"),
        ),
        "reinit": True,
        "mode": cfg.wandb.mode,
    }
    
    # Auto-resume: extract wandb_id from checkpoint if resuming
    if cfg.general.resume is not None:
        wandb_id = get_wandb_id_from_checkpoint(cfg.general.resume)
        if wandb_id:
            kwargs["id"] = wandb_id
            kwargs["resume"] = "must"
            print(f"Auto-resuming wandb run: {wandb_id}")
        else:
            kwargs["id"] = cfg.wandb.wandb_id
            kwargs["resume"] = "allow"
            print(f"Resuming wandb run: {cfg.wandb.wandb_id}")
    config_dict["general"]["local_dir"] = os.getcwd()
    # Ensure artifacts/logs land under a stable path
    kwargs["save_dir"] = wandb_dir
    return WandbLogger(**kwargs)


from src.ema import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn
class EMAWeightAveraging(WeightAveraging):
    def __init__(self,decay):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))


class IntervalTimer:
    """A simple resettable timer that measures a single time interval."""

    def __init__(self):
        self._start_time = None
        self._value = None

    def start(self) -> None:
        """Starts the timer."""
        if self._start_time is None:
            self._start_time = time.time()

    def pause(self) -> None:
        """
        Stops the timer and records the elapsed time.
        Raises a ValueError if the timer is not running.
        """
        if self._start_time is not None:
            self._value = time.time() - self._start_time
            self._start_time = None
        else:
            self._value = None


    def get_value(self) -> float:
        """
        Returns the measured time interval.
        If the timer is still running, it is paused first.
        """
        if self._value is None:
            # If get_value() is called on a running timer, pause it to get the current time.
            if self._start_time is not None:
                self.pause()
        return self._value

    def reset(self) -> None:
        """Resets the timer to its initial state."""
        self._start_time = None
        self._value = None