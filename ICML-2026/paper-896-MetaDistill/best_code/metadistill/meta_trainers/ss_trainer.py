"""Self-supervised fine-tuning trainer for MetaDistill."""

import json
import random
import warnings
import pickle
import math
import torch
import numpy
from tqdm import tqdm
from optimizers import POM, GradBasedLES, GradBasedLGA, LDE
from tasks import TaskProblem
from tasks.utils import genOffset, set_tf_offset
from meta_trainers.base_trainer import BaseTrainer
from utils import kld_loss
from torch_basic_settings import DEVICE, DTYPE


class SelfSupervisedTrainer(BaseTrainer):
    """SSFT trainer with optional KL and diversity regularization."""

    def __init__(self, config):
        super().__init__(config)

        seed = config.get("seed", None)
        if seed is not None:
            seed = int(seed)
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.lr = float(config.get("lr", 5e-4))
        self.warming_up = int(config.get("warming_up", 0))
        self.save_every = int(config.get("save_every", 100))
        self.loss_eps = float(config.get("loss_eps", 1e-12))

        self.bp_strategy = config["bp_strategy"].lower()
        if self.bp_strategy not in ["greedy", "global", "sparse"]:
            warnings.warn('Invalid bp_strategy, use "greedy"')
            self.bp_strategy = "greedy"

        # KL regularization
        kl_reg_cfg = config.get("kl_regularization", {})
        self.use_kl_reg = bool(kl_reg_cfg.get("enabled", True))
        self.kl_reg_coef = float(kl_reg_cfg.get("coef", 0.1))
        self.kl_reg_decay = str(kl_reg_cfg.get("decay", "linear")).lower()
        self.kl_cov_reg = float(kl_reg_cfg.get("cov_reg", 1e-2))

        # Teacher trajectories for KL regularization
        teacher_path = kl_reg_cfg.get("teacher_trajectories", None)
        self.teacher_trajectories = None
        if self.use_kl_reg and teacher_path:
            try:
                import sys as _sys
                import numpy as _np
                _sys.modules.setdefault("numpy._core", _np.core)
                _sys.modules.setdefault("numpy._core.multiarray", _np.core.multiarray)
            except Exception:
                pass
            with open(teacher_path, "rb") as fp:
                self.teacher_trajectories = pickle.load(fp)
            self._teacher_epochs = len(self.teacher_trajectories)

        if self.use_kl_reg and self.teacher_trajectories is None:
            warnings.warn(
                "KL regularization enabled but no teacher_trajectories provided; "
                "disabling KL regularization"
            )
            self.use_kl_reg = False

        # Diversity regularization
        div_cfg = config.get("diversity_regularization", {})
        self.use_div_reg = bool(div_cfg.get("enabled", True))
        self.div_reg_coef = float(div_cfg.get("coef", 0.05))

        model_registry = {
            "POM": POM,
            "GradBasedLES": GradBasedLES,
            "GradBasedLGA": GradBasedLGA,
            "LDE": LDE,
        }
        model_name = config["model"]["name"]
        if model_name in model_registry:
            with open(config["model"]["config"], "r") as fp:
                model_config = json.load(fp)
            if seed is not None and model_config.get("seed", None) is None:
                model_config["seed"] = int(seed)
            self.model = model_registry[model_name](model_config)

            ckpt = config["model"].get("ckpt", None)
            if ckpt:
                if isinstance(ckpt, str):
                    state_dict = torch.load(ckpt, map_location="cpu")
                elif isinstance(ckpt, dict):
                    state_dict = ckpt
                else:
                    raise TypeError(f"Unsupported ckpt type: {type(ckpt)}")
                self.model.load_state_dict(state_dict)

            self.model.to(DEVICE).to(DTYPE)
        else:
            raise ValueError(f"Invalid model: {model_name}")

        self._print_config()

    def _print_config(self):
        ext = {
            "Model": self.model.name,
            "KL Reg": f"{self.use_kl_reg} (coef={self.kl_reg_coef})",
            "Div Reg": f"{self.use_div_reg} (coef={self.div_reg_coef})",
            "BP Strategy": self.bp_strategy,
        }
        super()._print_config(ext)

    def _get_kl_weight(self, epoch: int) -> float:
        """Get the KL regularization weight for the current epoch"""
        if not self.use_kl_reg:
            return 0.0

        progress = epoch / max(1, self.n_epochs - 1)

        if self.kl_reg_decay == "linear":
            # Linear decay to 0.1 * coef
            return self.kl_reg_coef * (1.0 - 0.9 * progress)
        elif self.kl_reg_decay == "cosine":
            # Cosine decay
            return self.kl_reg_coef * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return self.kl_reg_coef

    def _compute_kl_reg(self, pop, fid: str, g: int, t_epoch: int):
        """Compute the KL regularization loss"""
        if not self.use_kl_reg or self.teacher_trajectories is None:
            return torch.tensor(0.0, device=DEVICE)

        # Fetch teacher distribution parameters
        t_episode = self.teacher_trajectories[t_epoch % self._teacher_epochs].get(fid)
        if t_episode is None:
            return torch.tensor(0.0, device=DEVICE)

        t_params_seq = t_episode.get("distr_params", [])
        if not t_params_seq:
            return torch.tensor(0.0, device=DEVICE)

        g_idx = min(g, len(t_params_seq) - 1)
        raw = t_params_seq[g_idx]

        if isinstance(raw, dict):
            mean = torch.from_numpy(raw["mean"]).to(DTYPE).to(DEVICE)
            cov = torch.from_numpy(raw["cov_mat"]).to(DTYPE).to(DEVICE)
            # Covariance regularization
            if self.kl_cov_reg > 0:
                cov = cov + self.kl_cov_reg * torch.eye(
                    cov.shape[0], device=DEVICE, dtype=DTYPE
                )
            t_params = {"mean": mean, "cov_mat": cov}
        else:
            return torch.tensor(0.0, device=DEVICE)

        kl_out = kld_loss(
            p_pop=pop[..., 1:],
            q=t_params,
            kl_clip_value=10.0,
            eps=1e-6,
            clip_mode="soft",
            cov_mode="full",
            normalize="maxabs",
            return_raw=False,
        )

        return kl_out if kl_out is not None else torch.tensor(0.0, device=DEVICE)

    def _compute_div_reg(self, pop):
        """Compute diversity regularization loss"""
        if not self.use_div_reg:
            return torch.tensor(0.0, device=DEVICE)

        x = pop[..., 1:].view(-1, pop.shape[-1] - 1)
        cov = torch.cov(x.T)
        eye = torch.eye(cov.shape[0], device=DEVICE, dtype=DTYPE)
        sign, logdet = torch.linalg.slogdet(cov + 1e-6 * eye)

        if sign > 0:
            return -logdet / x.shape[-1]
        return torch.tensor(0.0, device=DEVICE)

    def train(self):
        n_funcs = len(self.training_set)
        min_epoch_loss = torch.inf
        epoch_bar = tqdm(range(int(self.n_epochs)), ncols=120)
        task = TaskProblem(fun=None, repaire=True, dim=self.problemdim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        use_cache = hasattr(self.model, "export_state") and hasattr(self.model, "import_state")

        for epoch in epoch_bar:
            epoch_loss = torch.tensor(0.0, device=DEVICE, requires_grad=False)
            last_pops = {}
            state_cache = {} if use_cache else None
            teacher_epochs = {}

            kl_w = self._get_kl_weight(epoch)
            div_w = self.div_reg_coef if self.use_div_reg else 0.0

            # Initialize
            for func in self.training_set:
                fid = func["fid"]
                # Teacher epoch used for this training epoch
                t_epoch = epoch % self._teacher_epochs if self.teacher_trajectories else 0
                teacher_epochs[fid] = t_epoch

                # Align offsets with teacher trajectories when KL regularization is enabled.
                if self.use_kl_reg and self.teacher_trajectories is not None:
                    t_trail = self.teacher_trajectories[t_epoch].get(fid)
                    if t_trail and "offset" in t_trail:
                        set_tf_offset(fun=func, offset=t_trail["offset"])
                    else:
                        raise KeyError(
                            f"KL regularization requires teacher offset for fid={fid}, t_epoch={t_epoch}, "
                            f"but teacher_trajectories[{t_epoch}] missing this fid or its 'offset' field. "
                            f"Either fix teacher_trajectories or disable kl_regularization."
                        )
                else:
                    genOffset(self.problemdim, func)
                task.setfun(fun=func)
                pop = task.genRandomPop((self.batchsize, self.popsize, self.problemdim))
                pop, _ = task.calfitness(pop)

                if not torch.isfinite(pop[..., 0]).all():
                    raise ValueError("Non-finite fitness in init")

                last_pops[fid] = pop.clone().detach()

                if use_cache:
                    if hasattr(self.model, "reset"):
                        self.model.reset()
                    state_cache[fid] = self.model.export_state()

            epoch_bar.set_description(f"E{epoch} KL_w={kl_w:.3f}")

            for g in range(self.n_generations):
                step_loss = torch.tensor(0.0, device=DEVICE, requires_grad=False)

                for func in self.training_set:
                    task.setfun(func)
                    fid = func["fid"]
                    pop = last_pops[fid]

                    if use_cache:
                        self.model.import_state(state_cache[fid])

                    l1 = torch.mean(pop[..., 0].view(-1))
                    pop, _ = self.model(pop, task)

                    if not torch.isfinite(pop[..., 0]).all():
                        raise ValueError(f"Non-finite fitness: e={epoch}, g={g}, fid={fid}")

                    if use_cache:
                        state_cache[fid] = self.model.export_state()
                    last_pops[fid] = pop.detach()

                    l2 = torch.mean(pop[..., 0].view(-1))

                    # Fitness-improvement loss
                    denom = (n_funcs * torch.abs(l1)).clamp_min(self.loss_eps)
                    fit_loss = (l2 - l1) / denom

                    # KL regularization
                    kl_loss = self._compute_kl_reg(pop, fid, g, teacher_epochs[fid])
                    kl_loss = kl_w * kl_loss / n_funcs

                    # Diversity regularization
                    div_loss = self._compute_div_reg(pop)
                    div_loss = div_w * div_loss / n_funcs

                    # Total loss
                    loss = fit_loss + kl_loss + div_loss

                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss: e={epoch}, g={g}, fid={fid}")

                    # Backward
                    if self.bp_strategy == "greedy":
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        opt.step()
                    elif self.bp_strategy == "global":
                        loss.backward()
                        if g == self.n_generations - 1:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            opt.step()
                            opt.zero_grad()
                    else:  # sparse
                        if g == self.n_generations - 1:
                            opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            opt.step()

                    step_loss += loss.detach()

                epoch_loss += step_loss

            epoch_loss = epoch_loss / self.n_generations
            self.logger.add_scalar("loss/epoch", epoch_loss, epoch)
            self.logger.add_scalar("kl_reg/weight", kl_w, epoch)

            # Save checkpoints
            if epoch >= self.warming_up:
                if epoch_loss < min_epoch_loss:
                    min_epoch_loss = epoch_loss
                    torch.save(
                        self.model.state_dict(),
                        f"{self.ckpt_saving_path}/{self.expname}_better_{epoch}.pth",
                    )

            if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.ckpt_saving_path}/{self.expname}_{epoch}.pth",
                )

            if epoch == self.n_epochs - 1:
                torch.save(self.model.state_dict(), f"{self.ckpt_saving_path}/{self.expname}_final.pth")
