import torch
import pathlib
import importlib
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from typing import Any, Union, List, Dict, Optional
from appfl.algorithm.aggregator import BaseAggregator
from appfl.misc.utils import create_instance_from_file, run_function_from_file
from appfl.misc.memory_utils import optimize_memory_cleanup


class VFLAggregator(BaseAggregator):
    """
    VFLAggregator:
        Aggregator for vertical federated learning, which takes in local embeddings from clients,
        concatenates them, and trains a model on the concatenated embeddings. The aggregator then
        sends back the gradient of the loss with respect to the concatenated embeddings to the clients
        for them to update their local embedding models.
    """

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        aggregator_configs: DictConfig = DictConfig({}),
        logger: Any | None = None,
    ):
        self.round = 0
        self.model = model
        self.logger = logger
        self.aggregator_configs = aggregator_configs

        # Check for optimize_memory in aggregator_configs, default to True
        self.optimize_memory = getattr(aggregator_configs, "optimize_memory", True)

        self.device = self.aggregator_configs.get("device", "cpu")
        self.model.to(self.device)
        optim_module = importlib.import_module("torch.optim")
        assert hasattr(optim_module, self.aggregator_configs.optim), (
            f"Optimizer {self.aggregator_configs.optim} not found in torch.optim"
        )
        self.optimizer = getattr(optim_module, self.aggregator_configs.optim)(
            self.model.parameters(), **self.aggregator_configs.optim_args
        )
        self._load_loss()
        self._load_data()
        self.train_losses, self.val_losses = [], []

    def aggregate(self, local_embeddings: Union[List[Dict], Dict[str, Dict]], **kwargs):
        # Prepare concatenated embeddings
        embedding_lengths = []
        if isinstance(local_embeddings, dict):
            client_id_order = []
            train_embedding_list = []
            val_embedding_list = []
            for client_id, emb in local_embeddings.items():
                client_id_order.append(client_id)
                embedding_lengths.append(emb["train_embedding"].shape[1])
                train_embedding_list.append(emb["train_embedding"])
                val_embedding_list.append(emb["val_embedding"])
            train_embedding = torch.cat(train_embedding_list, dim=1)
            val_embedding = torch.cat(val_embedding_list, dim=1)
        else:
            embedding_lengths = [
                emb["train_embedding"].shape[1] for emb in local_embeddings
            ]
            train_embedding = torch.cat(
                [emb["train_embedding"] for emb in local_embeddings], dim=1
            )
            val_embedding = torch.cat(
                [emb["val_embedding"] for emb in local_embeddings], dim=1
            )
        if self.optimize_memory:
            with torch.no_grad():
                train_embedding = (
                    train_embedding.detach().requires_grad_().to(self.device)
                )
                val_embedding = val_embedding.to(self.device)
                # Clean up intermediate lists
                optimize_memory_cleanup(
                    train_embedding_list, val_embedding_list, force_gc=False
                )
        else:
            train_embedding = train_embedding.detach().requires_grad_().to(self.device)
            val_embedding = val_embedding.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        y_train_pred = self.model(train_embedding)
        train_loss = self.loss_fn(y_train_pred, self.train_labels)
        train_loss.backward()
        self.optimizer.step()
        self.train_losses.append(train_loss.item())
        if self.validation:
            with torch.no_grad():
                self.model.eval()
                y_val_pred = self.model(val_embedding)
                val_loss = self.loss_fn(y_val_pred, self.val_labels)
                self.val_losses.append(val_loss.item())

        self.round += 1
        validation_interval = self.aggregator_configs.get("validation_interval", 50)
        if (self.round % validation_interval) == 0:
            if self.logger:
                if self.validation:
                    self.logger.info(
                        f"Round {self.round}: Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}"
                    )
                else:
                    self.logger.info(
                        f"Round {self.round}: Training Loss: {train_loss.item()}"
                    )
            else:
                if self.validation:
                    print(
                        f"Round {self.round}: Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}"
                    )
                else:
                    print(f"Round {self.round}: Training Loss: {train_loss.item()}")

        if (
            hasattr(self.aggregator_configs, "plot_epoch")
            and self.round == self.aggregator_configs.plot_epoch
        ):
            plot_file_path = getattr(
                self.aggregator_configs, "plot_file_path", "loss.pdf"
            )
            plot_file_dir = pathlib.Path(plot_file_path).parent
            plot_file_dir.mkdir(parents=True, exist_ok=True)
            self.plot_loss(plot_file_path)

        grads = train_embedding.grad.split(embedding_lengths, dim=1)

        # Memory optimization: Clean up intermediate tensors
        if self.optimize_memory:
            optimize_memory_cleanup(
                train_embedding, val_embedding, y_train_pred, force_gc=True
            )

        if isinstance(local_embeddings, dict):
            ret = {}
            for i, client_id in enumerate(client_id_order):
                ret[client_id] = {"client_grad": grads[i]}
        else:
            ret = [{"client_grad": grad} for grad in grads]
        return ret

    def get_parameters(self, **kwargs):
        raise NotImplementedError(
            "get_parameters method should not be called for VFLAggregator"
        )

    def plot_loss(self, save_path: Optional[str] = None):
        if save_path:
            import matplotlib

            matplotlib.use("Agg")
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training MSE")
        if self.validation:
            plt.plot(self.val_losses, label="Validation MSE")
            plt.title("Vertical Federated Learing: Training and Validation MSE")
        else:
            plt.title("Vertical Federated Learing: Training MSE")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error (MSE)")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def _load_loss(self):
        if hasattr(self.aggregator_configs, "loss_fn_path") and hasattr(
            self.aggregator_configs, "loss_fn_name"
        ):
            kwargs = self.aggregator_configs.get("loss_fn_kwargs", {})
            self.loss_fn = create_instance_from_file(
                self.aggregator_configs.loss_fn_path,
                self.aggregator_configs.loss_fn_name,
                **kwargs,
            )
        elif hasattr(self.aggregator_configs, "loss_fn"):
            kwargs = self.aggregator_configs.get("loss_fn_kwargs", {})
            if hasattr(torch.nn, self.aggregator_configs.loss_fn):
                self.loss_fn = getattr(torch.nn, self.aggregator_configs.loss_fn)(
                    **kwargs
                )
            else:
                self.loss_fn = None
        else:
            self.loss_fn = None

    def _load_data(self) -> None:
        """Get train and validation dataloaders from local dataloader file."""
        labels = run_function_from_file(
            self.aggregator_configs.server_label_path,
            self.aggregator_configs.server_label_fn,
            **(
                self.aggregator_configs.server_label_fn_kwargs
                if hasattr(self.aggregator_configs, "server_label_fn_kwargs")
                else {}
            ),
        )
        if isinstance(labels, tuple):
            self.train_labels, self.val_labels = labels
            self.validation = self.aggregator_configs.get("validation", True)
        else:
            self.train_labels = labels
            self.val_labels = None
            self.validation = False
