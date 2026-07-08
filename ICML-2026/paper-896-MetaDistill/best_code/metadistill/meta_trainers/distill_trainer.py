"""
Distillation trainer for MetaDistill.

Features:
- Progressive teacher alignment.
- Mixed loss: KL + fitness + diversity.
- Optional KL normalization / clipping.
- Curriculum-style distillation.
- Optional epoch-ramp scaling for KL (applied before clipping).
- Optional dimension normalization options.
"""
import os
import math
import json
import pickle
import random
import warnings
import torch
import numpy
from tqdm import tqdm
from utils import kld_loss
from meta_trainers.base_trainer import BaseTrainer
from optimizers import POM, GradBasedLES, GradBasedLGA, LDE
from tasks import TaskProblem
from tasks.utils import set_tf_offset
from torch_basic_settings import DEVICE, DTYPE


class DistillTrainer(BaseTrainer):

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

        self.lr = config["lr"]
        self.warming_up = config["warming_up"]
        self.distill_interval = config["distill_interval"]
        self.loss_config = config["loss_config"]

        # ---------- Progressive alignment ----------
        align_cfg = config.get("alignment", {})
        self.align_mode = str(align_cfg.get("mode", "progressive")).lower()
        self.align_warmup_epochs = int(align_cfg.get("warmup_epochs", 30))

        # ---------- Mixed loss (KL / fitness / diversity) ----------
        kl_cfg = self.loss_config.get("KL", {})
        self.kl_coef = float(kl_cfg.get("coef", 0.5))
        self.kl_clip_val = float(kl_cfg.get("clip_val", 10.0))
        self.kl_eps = float(kl_cfg.get("eps", 1e-6))
        self.kl_clip_mode = str(kl_cfg.get("clip_mode", "soft")).lower()
        self.kl_cov_mode = str(kl_cfg.get("cov_mode", "full")).lower()
        self.kl_normalize = str(kl_cfg.get("normalize", "maxabs")).lower()
        self.kl_dim_norm = str(kl_cfg.get("dim_norm", "sqrt")).lower()
        self.kl_log_metrics = bool(kl_cfg.get("log_metrics", True))
        self.kl_cov_reg = float(kl_cfg.get("cov_reg", 1e-2))

        # ---------- KL dim_normalize option ----------
        self.kl_dim_normalize = str(kl_cfg.get("dim_normalize", "div_d")).lower()

        # ---------- KL epoch-ramp (pre-clip scale) ----------
        epoch_ramp_cfg = kl_cfg.get("epoch_ramp", {})
        self.kl_epoch_ramp_enabled = bool(epoch_ramp_cfg.get("enabled", False))
        self.kl_epoch_ramp_schedule = str(epoch_ramp_cfg.get("schedule", "linear")).lower()
        self.kl_ramp_start = float(epoch_ramp_cfg.get("start_scale", 0.1))
        self.kl_ramp_end = float(epoch_ramp_cfg.get("end_scale", 1.0))

        fit_cfg = self.loss_config.get("Fitness", {})
        self.fit_coef = float(fit_cfg.get("coef", 0.3))
        self.fit_mode = str(fit_cfg.get("mode", "relative")).lower()

        div_cfg = self.loss_config.get("Diversity", {})
        self.div_coef = float(div_cfg.get("coef", 0.1))
        self.div_mode = str(div_cfg.get("mode", "entropy")).lower()

        # ========== Optional: POM-specific stabilization penalties ==========
        pom_cfg = self.loss_config.get("POM", {})
        if not isinstance(pom_cfg, dict):
            pom_cfg = {}
        # Penalize large mutation mixing matrices (helps avoid vchrom explosion for POM).
        self.pom_params_reg_coef = float(pom_cfg.get("params_reg_coef", 0.0))
        # Penalize out-of-bounds pre-repair offspring chromosomes (requires POM return_aux=True).
        self.pom_bound_penalty_coef = float(pom_cfg.get("bound_penalty_coef", 0.0))
        # Which tensor to penalize for bound violations: "offpop_chrom" (default) or "vchrom".
        self.pom_bound_target = str(pom_cfg.get("bound_target", "offpop_chrom")).lower()

        # ---------- Curriculum weights ----------
        curr_cfg = config.get("curriculum", {})
        self.use_curriculum = bool(curr_cfg.get("enabled", True))
        self.curriculum_schedule = str(curr_cfg.get("schedule", "linear")).lower()

        # Teacher settings
        teacher_smoothing = config.get("teacher_smoothing", {})
        if isinstance(teacher_smoothing, bool):
            teacher_smoothing = {"enabled": teacher_smoothing}
        self.teacher_smoothing_enabled = bool(teacher_smoothing.get("enabled", False))
        self.teacher_smoothing_window = int(teacher_smoothing.get("window", 3))
        self.teacher_smoothing_kernel = str(teacher_smoothing.get("kernel", "exp")).lower()
        self.teacher_smoothing_exp_alpha = float(teacher_smoothing.get("exp_alpha", 0.5))
        self._warned_smoothing = False

        # NumPy compatibility (legacy aliases)
        try:
            import sys as _sys
            import numpy as _np
            _sys.modules.setdefault("numpy._core", _np.core)
            _sys.modules.setdefault("numpy._core.multiarray", _np.core.multiarray)
        except Exception:
            pass

        # Load teacher trajectories. The old key is accepted only for compatibility.
        teacher_path = config.get("teacher_trajectories", config.get("teacher_trails"))
        if teacher_path is None:
            raise KeyError("Missing teacher trajectory path: expected `teacher_trajectories`.")
        with open(teacher_path, "rb") as fp:
            self.teacher_trajectories = pickle.load(fp)
        self._teacher_dataset_epochs = len(self.teacher_trajectories)

        teacher_sampling = config.get("teacher_sampling", {})
        if isinstance(teacher_sampling, str):
            teacher_sampling = {"mode": teacher_sampling}
        self.teacher_sampling_mode = str(teacher_sampling.get("mode", "sequential")).lower()
        self.teacher_sampling_per_fid = bool(teacher_sampling.get("per_fid", True))
        teacher_seed = teacher_sampling.get("seed", seed)
        self._teacher_rng = random.Random(int(teacher_seed) if teacher_seed is not None else None)

        # Student model
        student_name = config["student"]["name"]
        valid_students = ["POM", "GradBasedLES", "GradBasedLGA", "LDE"]
        if student_name not in valid_students:
            raise ValueError(f"Invalid student: {student_name}")

        with open(config["student"]["config"], "r") as fp:
            student_config = json.load(fp)
        student_registry = {
            "POM": POM,
            "GradBasedLES": GradBasedLES,
            "GradBasedLGA": GradBasedLGA,
            "LDE": LDE,
        }
        self.student = student_registry[student_name](student_config)

        if config["student"]["ckpt"] is not None:
            self.student.load_state_dict(torch.load(config["student"]["ckpt"]))
        self.student.to(DEVICE).to(DTYPE)

        # Optimizer
        decay_params, no_decay_params = [], []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.opt = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": 1e-3},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.lr)

        self._print_config()

    def _print_config(self):
        ext = {
            "Student": self.student.name,
            "Align": self.align_mode,
            "KL/Fit/Div": f"{self.kl_coef}/{self.fit_coef}/{self.div_coef}",
            "Curriculum": self.use_curriculum,
            "DimNorm": self.kl_dim_normalize,
            "EpochRamp": f"{self.kl_epoch_ramp_schedule}({self.kl_ramp_start}->{self.kl_ramp_end})" if self.kl_epoch_ramp_enabled else "off",
        }
        return super()._print_config(ext)

    def _pick_teacher_epoch(self, epoch: int) -> int:
        if self.teacher_sampling_mode == "random":
            return self._teacher_rng.randrange(self._teacher_dataset_epochs)
        return epoch % self._teacher_dataset_epochs

    def _to_teacher_params(self, raw):
        if isinstance(raw, dict):
            mean = torch.from_numpy(raw["mean"]).to(DTYPE).to(DEVICE)
            cov = torch.from_numpy(raw["cov_mat"]).to(DTYPE).to(DEVICE)
            if self.kl_cov_reg > 0:
                cov = cov + self.kl_cov_reg * torch.eye(cov.shape[0], device=DEVICE, dtype=DTYPE)
            return {"mean": mean, "cov_mat": cov}
        if isinstance(raw, numpy.ndarray):
            return torch.from_numpy(raw).to(DTYPE).to(DEVICE)
        return raw.to(DTYPE).to(DEVICE) if torch.is_tensor(raw) else raw

    def _get_teacher_params(self, params_seq, idx: int):
        idx = max(0, min(idx, len(params_seq) - 1))
        raw = params_seq[idx]

        if not self.teacher_smoothing_enabled or self.teacher_smoothing_window <= 1:
            return self._to_teacher_params(raw)

        if not isinstance(raw, dict):
            return self._to_teacher_params(raw)

        window = self.teacher_smoothing_window
        start = max(0, idx - window + 1)
        indices = list(range(start, idx + 1))

        if len(indices) == 1:
            return self._to_teacher_params(params_seq[indices[0]])

        alpha = self.teacher_smoothing_exp_alpha
        dist = torch.tensor([idx - i for i in indices], device=DEVICE, dtype=DTYPE)
        w = torch.exp(-alpha * dist)
        w = w / w.sum()

        means, covs = [], []
        for i in indices:
            p = self._to_teacher_params(params_seq[i])
            if isinstance(p, dict):
                means.append(p["mean"])
                covs.append(p["cov_mat"])

        if not means:
            return self._to_teacher_params(raw)

        mean = (w.view(-1, 1, 1) * torch.stack(means)).sum(0)
        cov = (w.view(-1, 1, 1) * torch.stack(covs)).sum(0)
        return {"mean": mean, "cov_mat": cov}

    def _compute_g_align(self, g: int, epoch: int) -> int:
        if self.align_mode == "sync":
            return g

        g_end = math.ceil((g + 1) / self.distill_interval) * self.distill_interval - 1
        g_end = min(g_end, self.n_generations - 1)

        if self.align_mode == "next":
            return g_end

        if epoch < self.align_warmup_epochs:
            return g

        denom = max(1, int(self.n_epochs) - int(self.align_warmup_epochs) - 1)
        curriculum_p = min(1.0, max(0.0, (epoch - self.align_warmup_epochs) / denom))
        return int(round(g + curriculum_p * (g_end - g)))

    def _fitness_loss(self, student_pop, teacher_best):
        student_best = student_pop[..., 0].view(-1).min()
        denom = torch.abs(teacher_best).clamp_min(1e-8)

        if self.fit_mode == "relative":
            loss = (student_best - teacher_best) / denom
        elif self.fit_mode == "rank":
            loss = torch.relu(student_best - teacher_best) / denom
        else:
            loss = student_best - teacher_best

        return loss.clamp(-10.0, 10.0)

    def _diversity_loss(self, pop):
        x = pop[..., 1:].view(-1, pop.shape[-1] - 1)

        if self.div_mode == "entropy":
            cov = torch.cov(x.T)
            eye = torch.eye(cov.shape[0], device=DEVICE, dtype=DTYPE)
            sign, logdet = torch.linalg.slogdet(cov + 1e-6 * eye)
            return -logdet / x.shape[-1] if sign > 0 else torch.tensor(0.0, device=DEVICE)
        elif self.div_mode == "spread":
            spread = ((x - x.mean(0)) ** 2).sum(-1).sqrt().mean()
            return -spread
        return torch.tensor(0.0, device=DEVICE)

    def _curriculum_weights(self, epoch: int):
        if not self.use_curriculum:
            return self.kl_coef, self.fit_coef, self.div_coef

        p = epoch / max(1, self.n_epochs - 1)

        if self.curriculum_schedule == "linear":
            kl_w = 0.2 + (self.kl_coef - 0.2) * p
            fit_w = self.fit_coef * (1.0 - 0.5 * p)
            div_w = self.div_coef * (1.0 - 0.5 * p)
        elif self.curriculum_schedule == "cosine":
            cos_p = 0.5 * (1 + math.cos(math.pi * (1 - p)))
            kl_w = 0.2 + (self.kl_coef - 0.2) * cos_p
            fit_w = self.fit_coef * (1.0 - 0.5 * cos_p)
            div_w = self.div_coef * (1.0 - 0.5 * cos_p)
        else:
            kl_w, fit_w, div_w = self.kl_coef, self.fit_coef, self.div_coef

        return kl_w, fit_w, div_w

    # ---------- KL epoch-ramp scale ----------
    def _kl_epoch_scale(self, epoch: int) -> float:
        """
        Compute the epoch-dependent KL scale.

        This scale is applied before clipping (via pre_clip_scale) to reduce KL saturation.

        - linear: scale = start + p * (end - start)
        - cosine: scale = start + cos_p * (end - start), where cos_p increases from 0 to 1
        """
        if not self.kl_epoch_ramp_enabled:
            return 1.0

        p = epoch / max(1, self.n_epochs - 1)

        if self.kl_epoch_ramp_schedule == "linear":
            return self.kl_ramp_start + p * (self.kl_ramp_end - self.kl_ramp_start)
        elif self.kl_epoch_ramp_schedule == "cosine":
            cos_p = 0.5 * (1 - math.cos(math.pi * p))
            return self.kl_ramp_start + cos_p * (self.kl_ramp_end - self.kl_ramp_start)
        else:
            return 1.0

    # ---------- KL loss with pre-clip scaling ----------
    def _kl_loss(self, p_pop, q, d: int, epoch: int):
        """
        Compute KL loss with optional epoch-ramp.

        The epoch-ramp is passed to kld_loss via pre_clip_scale and applied before clipping.
        """
        pre_clip_scale = self._kl_epoch_scale(epoch)

        out = kld_loss(
            p_pop=p_pop, q=q,
            kl_clip_value=self.kl_clip_val, eps=self.kl_eps,
            clip_mode=self.kl_clip_mode, cov_mode=self.kl_cov_mode,
            normalize=self.kl_normalize,
            dim_normalize=self.kl_dim_normalize,
            pre_clip_scale=pre_clip_scale,
            return_raw=True,
        )
        if out is None:
            return None, None

        kl_used, kl_raw = out

        # Post dim_norm scaling (factor applied after kld_loss).
        # - "none"/"identity": factor = 1.0 (no extra scaling)
        # - "sqrt": factor = sqrt(d)
        # - "mul_d": factor = d (explicit multiply by dimension)
        if self.kl_dim_norm == "sqrt":
            factor = math.sqrt(d)
        elif self.kl_dim_norm == "mul_d":
            factor = d
        else:  # "none", "identity", or any other value
            factor = 1.0

        return kl_used * factor, kl_raw * factor

    def train(self):
        n_funcs = len(self.training_set)
        min_loss = torch.inf
        task = TaskProblem(fun=None, repaire=True, dim=self.problemdim)
        use_cache = hasattr(self.student, "export_state") and hasattr(self.student, "import_state")
        self.student.train()

        epoch_bar = tqdm(range(int(self.n_epochs)), ncols=120)

        for epoch in epoch_bar:
            last_pops, state_cache = {}, {}
            epoch_loss = torch.tensor(0.0, device=DEVICE)
            kl_w, fit_w, div_w = self._curriculum_weights(epoch)
            kl_scale = self._kl_epoch_scale(epoch)

            teacher_epochs = {}
            t_epoch_global = self._pick_teacher_epoch(epoch)

            for func in self.training_set:
                fid = func["fid"]
                t_ep = self._pick_teacher_epoch(epoch) if self.teacher_sampling_per_fid else t_epoch_global
                teacher_epochs[fid] = t_ep
                set_tf_offset(fun=func, offset=self.teacher_trajectories[t_ep][fid]["offset"])
                task.setfun(func)
                pop = task.genRandomPop([self.batchsize, self.popsize, self.problemdim])
                pop, _ = task.calfitness(pop)
                last_pops[fid] = pop.detach()
                if use_cache:
                    if hasattr(self.student, "reset"):
                        self.student.reset()
                    state_cache[fid] = self.student.export_state()

            epoch_bar.set_description(f"E{epoch} KL={kl_w:.2f} SC={kl_scale:.3f}")

            for g in range(self.n_generations):
                loss = torch.tensor(0.0, device=DEVICE)

                for func in self.training_set:
                    task.setfun(func)
                    fid = func["fid"]
                    pop = last_pops[fid]

                    if use_cache:
                        self.student.import_state(state_cache[fid])

                    out = self.student(pop, task)
                    if isinstance(out, tuple):
                        pop, info = out
                    else:
                        pop, info = out, None

                    if use_cache:
                        state_cache[fid] = self.student.export_state()
                    last_pops[fid] = pop.detach()

                    g_align = self._compute_g_align(g, epoch)
                    t_ep = teacher_epochs[fid]
                    t_episode = self.teacher_trajectories[t_ep][fid]
                    t_params_seq = t_episode["distr_params"]
                    g_align = min(g_align, len(t_params_seq) - 1)
                    t_params = self._get_teacher_params(t_params_seq, g_align)

                    kl_used, kl_raw = self._kl_loss(pop[..., 1:], t_params, self.problemdim, epoch)
                    if kl_used is None:
                        raise RuntimeError(f"KL None: e={epoch}, g={g}, fid={fid}")
                    kl_loss = kl_w * kl_used / n_funcs

                    fit_loss = torch.tensor(0.0, device=DEVICE)
                    if fit_w != 0.0:
                        t_pop = t_episode["pop"][g_align]
                        t_best = (
                            torch.from_numpy(t_pop[..., 0]).to(DTYPE).to(DEVICE).view(-1).min()
                        )
                        fit_loss = fit_w * self._fitness_loss(pop, t_best) / n_funcs

                    div_loss = torch.tensor(0.0, device=DEVICE)
                    if div_w != 0.0:
                        div_loss = div_w * self._diversity_loss(pop) / n_funcs

                    # Optional POM penalties (only if enabled in config and aux is provided by the model).
                    pom_reg = torch.tensor(0.0, device=DEVICE)
                    pom_bound = torch.tensor(0.0, device=DEVICE)
                    if (self.pom_params_reg_coef > 0.0 or self.pom_bound_penalty_coef > 0.0) and isinstance(info, dict):
                        aux = info.get("aux", None)
                        if isinstance(aux, dict):
                            if self.pom_params_reg_coef > 0.0 and "mut_params" in aux:
                                mp = aux["mut_params"]
                                # Mean-squared magnitude (Frobenius^2 / numel) for scale invariance.
                                pom_reg = (mp * mp).mean()

                            if self.pom_bound_penalty_coef > 0.0:
                                tgt_key = "offpop_chrom" if self.pom_bound_target in {"offpop_chrom", "offpop", "offspring"} else "vchrom"
                                if tgt_key in aux:
                                    x = aux[tgt_key]
                                    xlb = float(func.get("xlb", -10.0))
                                    xub = float(func.get("xub", 10.0))
                                    pom_bound = (torch.relu(x - xub) + torch.relu(xlb - x)).mean()

                    loss += kl_loss + fit_loss + div_loss
                    if self.pom_params_reg_coef > 0.0:
                        loss += (self.pom_params_reg_coef * pom_reg) / n_funcs
                    if self.pom_bound_penalty_coef > 0.0:
                        loss += (self.pom_bound_penalty_coef * pom_bound) / n_funcs

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss: e={epoch}, g={g}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                epoch_loss += loss.detach()

            epoch_loss /= self.n_generations
            self.logger.add_scalar("loss/epoch", epoch_loss, epoch)
            self.logger.add_scalar("kl_scale/epoch", kl_scale, epoch)

            if epoch > self.warming_up and epoch_loss < min_loss:
                torch.save(self.student.state_dict(), f"{self.ckpt_saving_path}/{self.expname}_better_{epoch}.pth")
                min_loss = epoch_loss

            if epoch == self.n_epochs - 1:
                torch.save(self.student.state_dict(), f"{self.ckpt_saving_path}/{self.expname}_final.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    trainer = DistillTrainer(config)
    trainer.train()
