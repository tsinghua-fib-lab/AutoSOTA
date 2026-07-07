import os
from functools import partial

import torch
from fvcore.common.timer import Timer
from loguru import logger
from tqdm.auto import tqdm

from utils import (ScalarMeter, aggregate_predictions, convert_weights_to_fp16,
                   get_grad_norm, gpu_mem_usage, topk_accuracies)

try:
    import wandb

    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False
    logger.warning("Wandb not found. Please install wandb to log metrics.")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        max_epochs: int,
        log_interval: int,
        save_interval: int,
        eval_interval: int,
        save_path: str,
        training_precision: str,
        resume_checkpoint: str,
        optimizer_kwargs: dict,
        lr_policy_kwargs: dict,
        clip_l2_gradnorm: float,
        log_to_wandb: bool = False,
        config: dict = None,
    ):
        assert training_precision in ["fp32", "fp16", "amp"]
        assert len(
            eval_dataloader) <= 2, 'only test or  new / base test are supported'

        self.model = model
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_path = save_path
        self.eval_interval = eval_interval
        self.resume_checkpoint = resume_checkpoint
        self.training_precision = training_precision
        self.total_iters = max_epochs * len(train_dataloader)
        self.clip_l2_gradnorm = clip_l2_gradnorm
        self.use_mixed_precision = training_precision == "amp"
        self.log_to_wandb = log_to_wandb and WANDB_FOUND

        self.max_epochs = max_epochs
        self.curr_step = 0
        self.curr_epoch = 0

        self._setup_wandb(config)

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            logger.add(
                os.path.join(self.save_path, "logs.log"),
                format="{time:YYYY-MM-DD HH:mm} {level} {message}",
                rotation="10 MB",
            )

        if self.resume_checkpoint and len(self.resume_checkpoint) > 0:
            self._load_parameters()

        if self.training_precision == "fp16":
            self._setup_fp16()

        self._set_trainable_params()
        self.opt = self._set_optimizer(**optimizer_kwargs)
        self.lr_scheduler = self._set_lr_policy(**lr_policy_kwargs)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_mixed_precision)

        if self.resume_checkpoint and len(self.resume_checkpoint) > 0:
            self._load_optimizer_state()

        self._set_device()
        self.model.to(self.device)
        self._set_data_parallel()
        self._set_metrics()

    def run_training_loop(self):
        start_epoch = self.curr_epoch + 1
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.curr_epoch = epoch

            for curr_step, batch in enumerate(self.train_dataloader):
                self.iter_tic()
                self.run_step(batch)
                self.iter_toc()

                if curr_step % self.log_interval == 0:
                    self.log_step(curr_step)

            if self.save_interval and self.curr_epoch % self.save_interval == 0:
                self.save()

            if (
                self.curr_epoch % self.eval_interval == 0
                or self.curr_epoch == self.max_epochs
            ):
                self.run_eval_loop()

    def run_step(self, batch):
        self.curr_step += 1

        # inputs, labels, label_names, file_names
        inputs, labels, _, _ = batch
        inputs = inputs.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        self.data_toc()

        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            self.opt.zero_grad()
            preds = self.model(inputs)
            loss = self.loss_fn(preds, labels)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)

        if self.clip_l2_gradnorm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_l2_gradnorm
            )
        else:
            grad_norm = get_grad_norm(self.model.parameters())

        self.scaler.step(self.opt)
        self.scaler.update()
        self.lr_scheduler.step()

        top1_acc, top5_acc = topk_accuracies(preds, labels, topk_vals=(1, 5))

        self.update_metrics(loss, top1_acc, top5_acc, grad_norm)

    def run_eval_loop(self):

        self.model.eval()

        for idx, (name, eval_loader) in enumerate(self.eval_dataloader.items()):

            base_testing = all(x == y for x, y in zip(
                eval_loader.dataset.label_names,
                self.train_dataloader.dataset.label_names))

            if base_testing:
                # Keep using base class names (used during training)
                self.model.prompter.train()
            else:
                self.model.prompter.eval()

            top1_acc, top5_acc, all_losses = self.run_eval(eval_loader)

            self.best_top1[idx] = max(self.best_top1[idx], top1_acc)
            self.best_top5[idx] = max(self.best_top5[idx], top5_acc)

            logger.info(f"Evaluation results ({name})\n"
                        f"  - loss: {all_losses:.3f}\n"
                        f"  - top1: {top1_acc:.2f}\n"
                        f"  - top5: {top5_acc:.2f}\n"
                        f"  - best top1: {self.best_top1[idx]:.2f}\n"
                        f"  - best top5: {self.best_top5[idx]:.2f}\n"
                        )

            self._send_logs_to_wandb(
                {
                    f"{name}/loss": all_losses,
                    f"{name}/top1 acc": top1_acc,
                    f"{name}/top5 acc": top5_acc,
                    f"{name}/best top1": self.best_top1[idx],
                    f"{name}/best top5": self.best_top5[idx],
                },
                train=False,
            )

        if self.best_top1[-1] == top1_acc and self.save_path:
            self._save_model("best_model.pyth")

        self.model.train()

    @torch.no_grad()
    def run_eval(self, eval_dataloader):
        all_preds, all_labels, all_losses = [], [], []

        tbar = tqdm(eval_dataloader)
        # inputs, labels, label_names, file_names
        for inputs, labels, _, _ in tbar:
            labels = labels.to(self.device, non_blocking=True)

            if isinstance(inputs, list):
                # multi view testing
                inputs = [input.to(self.device, non_blocking=True)
                          for input in inputs]
                preds = [self.model(input) for input in inputs]
                preds = aggregate_predictions(preds)
            else:
                inputs = inputs.to(self.device, non_blocking=True)
                preds = self.model(inputs)

            loss = self.loss_fn(preds, labels).item()

            top1_acc, top5_acc = topk_accuracies(
                preds, labels, topk_vals=(1, 5))

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_losses.append(loss)

            info_str = f"Evaluating (loss: {loss:.3f} "
            info_str += f"top1: {top1_acc:.1f} top5: {top5_acc:.1f})"
            tbar.set_description(info_str)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_losses = torch.tensor(all_losses).mean()

        top1_acc, top5_acc = topk_accuracies(
            all_preds, all_labels, topk_vals=(1, 5))

        return top1_acc, top5_acc, all_losses

    def _set_device(self):
        if not torch.cuda.is_available():
            logger.warning("No GPU detected, using CPU instead")
            self.device = torch.device("cpu")
        else:
            logger.info("Using GPU for training")
            self.device = torch.device("cuda")

    def _set_data_parallel(self):
        # TODO: ddp instead?
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(
                f"Multiple GPUs detected ({torch.cuda.device_count()})" "using them."
            )
            self.model = torch.nn.DataParallel(self.model)
            self.data_parallel = True
        else:
            self.data_parallel = False

    def _set_trainable_params(self):
        model_params = list(self.model.parameters())
        for param in model_params:
            param.requires_grad_(False)

        trainable_parameters = self.model.trainable_parameters

        for name, param in self.model.named_parameters():
            if any([(to_train in name) for to_train in trainable_parameters]):
                param.requires_grad_(True)

    def _set_optimizer(self, opt_type, learning_rate, weight_decay, zero_wd_for_biases):
        to_ignore_parameters = []
        zero_wd_parameters = []
        normal_parameters = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                to_ignore_parameters.append(param)
            elif zero_wd_for_biases and (
                len(param.shape) == 1 or name.endswith(".bias")
            ):
                zero_wd_parameters.append(param)
            else:
                normal_parameters.append(param)

        num_train_params = len(normal_parameters) + len(zero_wd_parameters)
        total_num_params = len(list(self.model.parameters()))
        assert (
            total_num_params == len(to_ignore_parameters) + num_train_params
        ), "Some parameters are not assigned to any group"

        logger.info(
            "Setting optimizer ...\n"
            f"- Number of trainable parameters: {num_train_params},"
            f" of which {len(zero_wd_parameters)} are without weight decay\n"
            f"- Number of frozen parameters: {len(to_ignore_parameters)}\n"
            f"- Total number of parameters: {total_num_params}"
        )

        opt_params = [
            {
                "params": normal_parameters,
                "weight_decay": weight_decay,
            }
        ]

        opt_params += (
            [
                {
                    "params": zero_wd_parameters,
                    "weight_decay": 0,
                }
            ]
            if len(zero_wd_parameters) > 0
            else []
        )

        if opt_type == "sgd":
            return torch.optim.SGD(
                opt_params,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
                dampening=0.0,
                nesterov=True,
            )

        if opt_type == "adam":
            return torch.optim.Adam(
                opt_params,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )

        if opt_type == "adamw":
            return torch.optim.AdamW(
                opt_params,
                lr=learning_rate,
                eps=1e-08,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )

        raise NotImplementedError(f"Unsupported {opt_type} optimizer")

    def _set_lr_policy(
        self,
        lr_policy_type,
        lr_step_milestones,
        warmup_epochs,
        consine_end_lr,
        linear_end_lr,
    ):
        n_warmup_steps = int(float(warmup_epochs) * len(self.train_dataloader))

        if lr_policy_type == "cosine":
            train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=self.total_iters, eta_min=consine_end_lr
            )
            logger.info("Cosine learning rate policy is set.")

        elif lr_policy_type == "step":
            lr_milestones = [
                int(float(milestone) * len(self.train_dataloader)) - n_warmup_steps
                for milestone in lr_step_milestones
            ]

            train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.opt, milestones=lr_milestones, gamma=0.1
            )
            logger.info("Step learning rate policy is set.")

        elif lr_policy_type == "linear":
            end_factor = linear_end_lr / self.opt.param_groups[0]["lr"]
            train_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.opt,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=self.total_iters,
            )
            logger.info("Linear learning rate policy is set.")

        else:
            train_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lr_lambda=lambda _: 1
            )
            logger.info("No learning rate policy is set.")

        if n_warmup_steps == 0:
            return train_scheduler

        logger.info(f"Using warmup for the first {warmup_epochs} epochs")

        def warmup_lr_scheduler(current_step, n_warmup_steps):
            return current_step / n_warmup_steps

        lr_lambda = partial(warmup_lr_scheduler, n_warmup_steps=n_warmup_steps)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=lr_lambda
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.opt,
            schedulers=[warmup_scheduler, train_scheduler],
            milestones=[n_warmup_steps],
        )
        return scheduler

    def _load_parameters(self):
        assert "model_epoch" in str(
            self.resume_checkpoint
        ), "resume_checkpoint should be 'save_path/model_epoch{epoch_n}.pyth'"

        resume_step = self.resume_checkpoint.split("model_epoch")[-1]
        self.curr_epoch = int(resume_step.split(".")[0])
        self.curr_step = self.curr_epoch * len(self.train_dataloader)

        logger.info(
            f"Loading model from checkpoint: {self.resume_checkpoint}...")
        self.model.load_state_dict(torch.load(self.resume_checkpoint))

    def _load_optimizer_state(self):
        step_name = self.resume_checkpoint.split("model_")[-1]
        parent_dir = self.resume_checkpoint.split("/")
        parent_dir = "/".join(parent_dir[:-1])
        opt_checkpoint = f"{parent_dir}/opt_{step_name}"

        logger.info(f"Loading optimizer state from: {opt_checkpoint}")

        opt_dict = torch.load(opt_checkpoint)
        self.opt.load_state_dict(opt_dict["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(opt_dict["scheduler_state_dict"])
        self.scaler.load_state_dict(opt_dict["scaler_state_dict"])

    def _setup_fp16(self):
        # use a frozen clip in fp16, rest is in fp32
        # to train everything in fp16, use "amp"
        self.model.text_encoder.apply(convert_weights_to_fp16)
        self.model.image_encoder.apply(convert_weights_to_fp16)
        self.model.clip_dtype = torch.float16

    def _save_model(self, filename):
        model_state_dict = (
            self.model.module.state_dict()
            if self.data_parallel
            else self.model.state_dict()
        )
        with open(os.path.join(self.save_path, filename), "wb") as f:
            torch.save(model_state_dict, f)

    def save(self):
        if self.save_path:
            logger.info("Saving model ...")
            filename = f"model_epoch{self.curr_epoch}.pyth"
            self._save_model(filename)

            logger.info("Optimize state ...")
            filename = f"opt_epoch{self.curr_epoch}.pyth"
            opt_dict = {
                "optimizer_state_dict": self.opt.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            }

            with open(os.path.join(self.save_path, filename), "wb") as f:
                torch.save(opt_dict, f)
        else:
            logger.info("No save path is set.")

    def update_metrics(self, loss, top1_acc, top5_acc, grad_norm):
        loss, grad_norm = loss.item(), grad_norm.item()
        top1_acc, top5_acc = top1_acc.item(), top5_acc.item()

        self.loss.add_value(loss)
        self.top1_acc.add_value(top1_acc)
        self.top5_acc.add_value(top5_acc)
        self.grad_norm.add_value(grad_norm)

    def log_step(self, curr_step):
        step_info = {
            "epoch": f"{self.curr_epoch}/{self.max_epochs}",
            "iter": f"{curr_step}/{len(self.train_dataloader)}",
            "dt": f"{self.iter_timer.seconds():.2f}",
            "dt_data": f"{self.data_timer.seconds():.2f}",
            "dt_net": f"{self.net_timer.seconds():.2f}",
            "loss": f"{self.loss.get_win_median():.3f}",
            "top1_acc": f"{self.top1_acc.get_win_median():.2f}",
            "top5_acc": f"{self.top5_acc.get_win_median():.2f}",
            "lr": f"{self.opt.param_groups[0]['lr']:.5f}",
            "grad_norm_avg": f"{self.grad_norm.get_win_median():.2f}",
            "grad_norm": f"{self.grad_norm.get_current_value():.2f}",
            "gpu_mem": f"{gpu_mem_usage():.2f}G",
        }
        logger.info(step_info)
        self._send_logs_to_wandb(step_info)

    def _send_logs_to_wandb(self, logs, train=True):
        if self.log_to_wandb:
            if train:
                for key in ["dt", "dt_data", "dt_net", "gpu_mem", "iter", "epoch"]:
                    logs.pop(key)
                logs["lr"] = self.opt.param_groups[0]["lr"]
                wandb.log({f"train/{key}": float(item)
                          for key, item in logs.items()})
            else:
                wandb.log(logs)

    def _setup_wandb(self, cfg):
        if self.log_to_wandb:
            name = f"{cfg.MODEL.VIZ_BACKBONE}_{cfg.MODEL.PROMPT_POSITION}_"
            name += f"{cfg.MODEL.NUM_PROMPTS}_{cfg.MODEL.FRAME_AGGREGATION}_"
            name += f"{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TRAIN_STRIDES}"

            wandb.init(project="video_prompts", name=name,
                       config=cfg, save_code=True)

    def iter_tic(self):
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def _set_metrics(self):
        self.loss = ScalarMeter(self.log_interval)
        self.top1_acc = ScalarMeter(self.log_interval)
        self.top5_acc = ScalarMeter(self.log_interval)
        self.grad_norm = ScalarMeter(self.log_interval)
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.iter_timer = Timer()
        self.best_top1 = [float("-inf")] * len(self.eval_dataloader)
        self.best_top5 = [float("-inf")] * len(self.eval_dataloader)

    def _reset_metrics(self):
        self.loss.reset()
        self.top1_acc.reset()
        self.top5_acc.reset()
        self.grad_norm.reset()
        self.data_timer.reset()
        self.net_timer.reset()
        self.iter_timer.reset()

    @property
    def final_results(self):
        if len(self.eval_dataloader) == 1:
            return {"top1": self.best_top1, "top5": self.best_top5}

        results = {}
        for idx, name in enumerate(self.eval_dataloader.keys()):
            results[f"top1_{name}"] = self.best_top1[idx]
            results[f"top5_{name}"] = self.best_top5[idx]
        return results
