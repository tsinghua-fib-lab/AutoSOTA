import time
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
 
import pytorch_lightning as pl
from src.models.transformer_model import GraphTransformer
from src.metrics.train_metrics import TrainLossDiscrete
from src import utils
from src.flow_matching.kappa_scheduler import KappaScheduler
from src.flow_matching.step_controller import StepController
from src.flow_matching.probability_projector import (
    RowStochasticProjector,
    ResidualAddProjector,
)
from src.flow_matching.rate_registry import build_rate_registry

from src.flow_matching.tail_postprocessor import TailPostProcessor
from src.flow_matching.noise_model import NoiseModel
from src.flow_matching.sampler import Sampler
from src.flow_matching.class_free_masker import ClassFreeMasker



class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        sampling_metrics,
        extra_features,
        input_output_dims,
        node_dist,
        noise_dist,
        test_labels=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        self.conditional = cfg.general.conditional
        self.input_dims = input_output_dims[0]
        self.output_dims = input_output_dims[1]
        self.node_dist = node_dist
        self.sampling_metrics = sampling_metrics
        self.extra_features = extra_features
        # self.domain_features = domain_features
        self.limit_dist = noise_dist.get_limit_dist()
        self.train_loss = TrainLossDiscrete(
            self.cfg.model.lambda_train,
        )

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            time_emb_type = cfg.model.time_emb_type,
        )

        self.save_hyperparameters(ignore=["sampling_metrics", "extra_features"])

        # logging
        self.train_timer = utils.IntervalTimer()

        # schedulers and modular components
        self.kappa_scheduler = KappaScheduler(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
        )
        # build extra features directly via compute_extra_data
        self.noise_model = NoiseModel(self.limit_dist)
        self.rate_registry = build_rate_registry(self.cfg, self.limit_dist)
        self.tail_postproc = TailPostProcessor(
            enabled=getattr(self.cfg.sample, "tail_process", False),
        )
        self.class_free_masker = ClassFreeMasker(
            enabled=self.conditional, p=getattr(self.cfg.train, "class_free_p", 0.1)
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        y_in = data.y if not self.conditional else self.class_free_masker.maybe_mask(data.y, self.device)

        dense_data, node_mask = utils.pyg_data_to_place_holder(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask)
        X0, E0 = dense_data.X, dense_data.E
        noisy_data = self.sample_time_and_add_noise(X0, E0, y_in, node_mask)
        extra_data, t_emb = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, t_emb, node_mask)

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X0,
            true_E=E0,
            true_y=data.y,
            log=False,  # i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:

        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )

    def on_train_epoch_start(self) -> None:
        self.print(f"Starting train epoch {self.current_epoch}...")
        self.train_timer.reset()
        self.train_timer.start()
        self.train_loss.reset()

    def on_train_epoch_end(self) -> None:
        log_dict = {}
        self.train_loss.log_epoch_metrics(log_dict)
        # read training time accumulated during the epoch (paused if val ran)
        train_epoch_time = float(self.train_timer.get_value())
        # self.print(
        #     f"Epoch {self.current_epoch}: X_CE: {log_dict['train/X_CE'] :.3f}"
        #     f" -- E_CE: {log_dict['train/E_CE'] :.3f} --"
        #     f" y_CE: {log_dict['train/y_CE'] :.3f}"
        #     f" -- {train_epoch_time:.1f}s "
        # )
        log_dict["train/epoch_time"] = train_epoch_time
        self.logger.log_metrics(log_dict, step=self.current_epoch)
        self.print(f"Logged train metrics: {log_dict}")

    def validation_step(self, data, i):
        pass

    def on_validation_epoch_start(self) -> None:
        # pause training timer at the moment validation starts
        self.train_timer.pause()

    def on_validation_epoch_end(self) -> None:
        self._sample_and_log(split="val", step=self.current_epoch)
        self.print("Finished validation.")

    def test_step(self, data, i):
        pass

    def on_test_epoch_end(self) -> None:
        step = getattr(self, "log_step", self.current_epoch)
        self._sample_and_log(split="test", step=step)
        self.print("Finished testing.")

    def _aggregate_numeric_fold_logs(self, fold_logs):
        """Aggregate per-key mean and std across folds for numeric values only.

        Accepts scalars (floats/ints) and scalar torch.Tensors; ignores others.
        """
        per_key_values = {}
        for d in fold_logs:
            for k, v in d.items():
                try:
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1:
                            val = float(v.item())
                        else:
                            continue
                    else:
                        val = float(v)
                except Exception:
                    continue
                per_key_values.setdefault(k, []).append(val)

        mean_dict = {}
        std_dict = {}
        for k, vs in per_key_values.items():
            arr = np.asarray(vs, dtype=np.float64)
            mean_dict[k] = float(np.mean(arr))
            if arr.size > 1:
                std_dict[k] = float(np.std(arr, ddof=0))
        return mean_dict, std_dict

    def _sample_and_log(self, split: Literal["test", "val"], step: int) -> None:
        num_folds = (
            int(getattr(self.cfg.general, "num_folds", 1))
            if hasattr(self.cfg, "general") else 1
        )
        if num_folds <= 1:
            print("Starting to sample")
            samples, labels = self.sample()
            log_dict = self.evaluate_samples(samples=samples, labels=labels, split=split)
            log_dict_added_prefix = {}
            for key, value in log_dict.items():
                log_dict_added_prefix[f'{split}/{key}'] = value
            self.logger.log_metrics(log_dict_added_prefix, step=step)
            self.print(f"Logged {split} metrics: {log_dict_added_prefix}")
        else:
            print(f"Starting to sample with {num_folds} folds")
            fold_logs = []
            for _ in range(num_folds):
                samples, labels = self.sample()
                log_dict = self.evaluate_samples(samples=samples, labels=labels, split=split)
                fold_logs.append(log_dict)

            mean_dict, std_dict = self._aggregate_numeric_fold_logs(fold_logs)

            log_dict_added_prefix = {}
            for key, value in mean_dict.items():
                log_dict_added_prefix[f'{split}/{key}'] = value
            for key, value in std_dict.items():
                log_dict_added_prefix[f'{split}/std/{key}'] = value
            self.logger.log_metrics(log_dict_added_prefix, step=step)
            self.print(f"Logged {split} metrics: {log_dict_added_prefix}")

    def sample(self):
        samples_left_to_generate = self.cfg.general.samples_to_generate
        sample_batch_size = getattr(self.cfg.sample, "batch_size", 2 * self.cfg.train.batch_size)
        samples = []
        labels = []
        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}",
                end="",
                flush=True
            )
            
            graph_nums_to_generate_in_this_batch = min(samples_left_to_generate, sample_batch_size)
            current_batch_samples, current_batch_labels = self.sample_batch(
                graph_nums_to_generate_in_this_batch,
            )
            samples.extend(current_batch_samples)
            labels.extend(current_batch_labels)

            samples_left_to_generate -= graph_nums_to_generate_in_this_batch

        return samples, labels

    def evaluate_samples(
        self,
        samples,
        labels,
        split: Literal["test", "val"],
    ):
        self.print("Computing sampling metrics...")
        return self.sampling_metrics.forward(
            samples,
            split=split,
            labels=labels if self.conditional else None,
        )

    def compute_model_output(self, noisy_data, extra_data, t_emb, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, t_emb, node_mask)

    # Backward-compatible wrapper expected by PyTorch Lightning
    def forward(self, noisy_data, extra_data, t_emb, node_mask):
        return self.compute_model_output(noisy_data, extra_data, t_emb, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_size: int,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        n_nodes_per_graph = self.node_dist.sample_n(batch_size, self.device)
        max_nodes_in_batch = torch.max(n_nodes_per_graph).item()

        # Build the masks
        arange = (
            torch.arange(max_nodes_in_batch, device=self.device).unsqueeze(0).expand(batch_size, -1)
        ) # [B, N]
        node_mask = arange < n_nodes_per_graph.unsqueeze(1) # [B, N]

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit(node_mask)
        if self.conditional:
            raise NotImplementedError
            # if "tls" in self.cfg.dataset.name:
            #     z_T.y = torch.zeros(batch_size, 1).to(self.device)
            #     z_T.y[: batch_size // 2] = 1
            # else:
        X, E, y = z_T.X, z_T.E, z_T.y
        z_0 = self.limit_dist.to_device(X.device)
        assert (E == torch.transpose(E, 1, 2)).all()

        # Build components for sampling
        selected_rate = getattr(self.cfg.sample, "rate_matrix", "rvf_denoiser")
        projector = (
            RowStochasticProjector()
            if selected_rate == "defog"
            else ResidualAddProjector()
        )
        rate_computer = self.rate_registry.get(selected_rate)
        stepper = StepController(
            sample_steps=self.cfg.sample.sample_steps,
            adaptive=self.cfg.sample.adaptive_step,
        )
        use_sid = getattr(self.cfg.sample, "use_sid", False)
        temperature = getattr(self.cfg.sample, "temperature", 1.0)
        sampler = Sampler(
            model=self.model,
            extra_builder=self.build_extra_features,
            kappa_scheduler=self.kappa_scheduler,
            stepper=stepper,
            rate_computer=rate_computer,
            projector=projector,
            postproc=self.tail_postproc,
            noise_model=self.noise_model,
            use_sid=use_sid,
            temperature=temperature,
        )

        sampled_s, discrete_sampled_s = sampler.run(
            X, E, y, node_mask, z_0
        )

        sampled_s = sampled_s.mask(node_mask, one_hot_to_index=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y


        molecule_list = []
        label_list = []
        for i in range(batch_size):
            num_nodes_in_graph = n_nodes_per_graph[i]
            atom_types = X[i, :num_nodes_in_graph].cpu()
            edge_types = E[i, :num_nodes_in_graph, :num_nodes_in_graph].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        return molecule_list, label_list

    def build_extra_features(self, noisy_data):
        result = self.extra_features(noisy_data)
        if isinstance(result, tuple):
            extras, t_emb = result
        else:
            raise RuntimeError("Expected a tuple of extras and t_emb, but got a single object")
            # extras, t_emb = result, noisy_data["t"].new_zeros((noisy_data["t"].shape[0], 0))
        return utils.PlaceHolder(X=extras.X, E=extras.E, y=extras.y), t_emb

    # Backward-compatible name retained for other components
    def compute_extra_data(self, noisy_data):
        return self.build_extra_features(noisy_data)

    def sample_time_and_add_noise(self, X, E, y, node_mask):
        batch_size = X.size(0)
        kappa_t, _ = self.kappa_scheduler.sample_train_t(batch_size, self.device)
        noisy = self.noise_model.apply_noise(X, E, y, node_mask, kappa_t)
        noisy["t"] = kappa_t
        return noisy
