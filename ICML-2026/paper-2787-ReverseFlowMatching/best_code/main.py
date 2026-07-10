import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from absl import app, flags

import jax
import jax.numpy as jnp

from src.agent.dqs import DQS
from src.agent.rfm import RFM
from src.agent.qvpo import QVPO
from src.agent.maxentdp import MaxEntDP
from src.agent.qsm import QSM
from src.agent.sac import SAC
from src.envs import make_dmc_env
from src.trainer import Trainer
from src.utils.logger import Logger

os.environ["WANDB_MODE"] = "online"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config",
    None,
    "Optional YAML config to overlay on top of configs/default.yaml (lower priority than --override).",
)
flags.DEFINE_multi_string(
    "override",
    [],
    "Config overrides as dot paths, e.g. --override algo=sac --override env.name=walker-run",
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top-level: {path}")
    return data


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in path.split(".") if p]
    if not parts:
        raise ValueError(f"Invalid override path: {path!r}")
    cur: Dict[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(
                f"Cannot set {path!r}: {part!r} is not a mapping (found {type(nxt).__name__})"
            )
        cur = nxt
    cur[parts[-1]] = value


def _get_by_path(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    parts = [p for p in path.split(".") if p]
    cur: Any = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _apply_overrides(
    cfg: Dict[str, Any],
    overrides: list[str],
    *,
    override_cfg_for_warnings: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override {item!r}; expected 'path=value', e.g. env.episode_length=200"
            )
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override {item!r}; empty key")
        value = yaml.safe_load(raw)
        if override_cfg_for_warnings is not None:
            sentinel = object()
            override_val = _get_by_path(override_cfg_for_warnings, key, sentinel)
            if override_val is not sentinel and override_val != value:
                print(
                    f"Warning: CLI override {key}={value!r} overrides --config value {override_val!r}"
                )
        _set_by_path(cfg, key, value)
    return cfg


def main(_):
    repo_root = Path(__file__).resolve().parent
    base_cfg = _load_yaml(repo_root / "configs" / "default.yaml")
    override_cfg = _load_yaml(Path(FLAGS.config)) if FLAGS.config else {}
    cfg = _deep_merge(base_cfg, override_cfg)
    cfg = _apply_overrides(
        cfg,
        list(FLAGS.override),
        override_cfg_for_warnings=override_cfg if FLAGS.config else None,
    )

    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    algo = str(cfg["algo"]).lower().strip()
    if algo not in {"sac", "dqs", "rfm", "maxentdp", "qsm", "qvpo"}:
        raise ValueError(f"Unsupported algorithm: {algo}")

    env_cfg = cfg.get("env", {})
    env_name = env_cfg["name"]

    env_fn = make_dmc_env(env_name=env_name)

    # create a dummy env to infer dimensions
    probe_env = env_fn(seed=cfg["seed"])
    try:
        probe_obs = probe_env.reset(seed=cfg["seed"])
    except TypeError:
        probe_obs = probe_env.reset()
    if isinstance(probe_obs, tuple):
        probe_obs = probe_obs[0]
    state_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]

    if env_cfg.get("episode_length") is None:
        env_max_steps = getattr(probe_env.unwrapped, "max_episode_steps", None)
        if env_max_steps is None:
            env_max_steps = 1000
        env_cfg["episode_length"] = int(env_max_steps)
    try:
        probe_env.close()
    except Exception:
        pass

    def _require_mapping(obj: Any, name: str) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            raise ValueError(f"Config key '{name}' must be a mapping")
        return obj

    trainer_cfg = _require_mapping(cfg.get("trainer"), "trainer")
    actor_cfg = _require_mapping(cfg.get("actor"), "actor")
    critic_cfg = _require_mapping(cfg.get("critic"), "critic")
    dqs_cfg = _require_mapping(cfg.get("dqs"), "dqs")
    rfm_cfg = _require_mapping(cfg.get("rfm"), "rfm")
    maxentdp_cfg = _require_mapping(cfg.get("maxentdp"), "maxentdp")
    qvpo_cfg = _require_mapping(cfg.get("qvpo"), "qvpo")
    qsm_cfg = _require_mapping(cfg.get("qsm"), "qsm")
    sac_cfg = _require_mapping(cfg.get("sac"), "sac")

    actor_hidden_size = int(actor_cfg["hidden_size"])
    actor_hidden_layers = int(actor_cfg["hidden_layers"])
    actor_lr = float(actor_cfg["lr"])
    actor_max_grad_norm = actor_cfg.get("max_grad_norm")

    critic_hidden_size = int(critic_cfg["hidden_size"])
    critic_hidden_layers = int(critic_cfg["hidden_layers"])
    critic_lr = float(critic_cfg["lr"])
    critic_max_grad_norm = critic_cfg.get("max_grad_norm")

    critic_config = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_size": critic_hidden_size,
        "hidden_layers": critic_hidden_layers,
        "num_q": int(critic_cfg["num_q"]),
        "discount": float(critic_cfg["discount"]),
        "tau": float(critic_cfg["tau"]),
        "target_update_period": int(critic_cfg.get("target_update_period", 1)),
        "lr": critic_lr,
        "max_grad_norm": critic_max_grad_norm,
        "seed": int(cfg["seed"]),
    }

    if algo == "sac":
        agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            critic_config=critic_config,
            actor_hidden_size=actor_hidden_size,
            actor_hidden_layers=actor_hidden_layers,
            actor_lr=actor_lr,
            actor_max_grad_norm=actor_max_grad_norm,
            alpha_lr=float(sac_cfg["alpha_lr"]),
            log_std_min=float(sac_cfg["log_std_min"]),
            log_std_max=float(sac_cfg["log_std_max"]),
            seed=int(cfg["seed"]),
        )
    elif algo == "dqs":
        agent = DQS(
            sample_dim=action_dim,
            state_dim=state_dim,
            critic_config=critic_config,
            actor_hidden_size=actor_hidden_size,
            actor_hidden_layers=actor_hidden_layers,
            actor_emb_size=int(actor_cfg["emb_size"]),
            actor_lr=actor_lr,
            actor_max_grad_norm=actor_max_grad_norm,
            num_estimator_mc_samples=int(dqs_cfg["num_estimator_mc_samples"]),
            num_integration_steps=int(dqs_cfg["num_integration_steps"]),
            num_samples_to_sample_from_buffer=int(trainer_cfg["batch_size"]),
            sigma_min=float(dqs_cfg["sigma_min"]),
            sigma_max=float(dqs_cfg["sigma_max"]),
            init_temperature=float(dqs_cfg["init_temperature"]),
            final_temperature=float(dqs_cfg["final_temperature"]),
            temperature_steps=int(dqs_cfg["temperature_steps"]),
            diffusion_scale=float(dqs_cfg["diffusion_scale"]),
            seed=int(cfg["seed"]),
        )
    elif algo == "qvpo":
        target_sample = qvpo_cfg.get("target_sample")
        if target_sample is None:
            target_sample = -1
        agent = QVPO(
            state_dim=state_dim,
            action_dim=action_dim,
            critic_config=critic_config,
            actor_hidden_size=actor_hidden_size,
            actor_hidden_layers=actor_hidden_layers,
            actor_emb_size=int(actor_cfg["emb_size"]),
            actor_lr=actor_lr,
            actor_max_grad_norm=actor_max_grad_norm,
            beta_schedule=str(qvpo_cfg["beta_schedule"]),
            num_diffusion_steps=int(qvpo_cfg["num_diffusion_steps"]),
            noise_ratio=float(qvpo_cfg["noise_ratio"]),
            behavior_sample=int(qvpo_cfg["behavior_sample"]),
            target_sample=int(target_sample),
            eval_sample=int(qvpo_cfg["eval_sample"]),
            train_sample=int(qvpo_cfg["train_sample"]),
            deterministic=bool(qvpo_cfg["deterministic"]),
            eps_greedy=float(qvpo_cfg["eps_greedy"]),
            ema_alpha_mean=float(qvpo_cfg["ema_alpha_mean"]),
            ema_alpha_std=float(qvpo_cfg["ema_alpha_std"]),
            action_lr=float(qvpo_cfg["action_lr"]),
            action_gradient_steps=int(qvpo_cfg["action_gradient_steps"]),
            action_augmentation=bool(qvpo_cfg["action_augmentation"]),
            entropy_alpha=float(qvpo_cfg["entropy_alpha"]),
            entropy_samples=int(qvpo_cfg["entropy_samples"]),
            seed=int(cfg["seed"]),
        )
    elif algo == "rfm":
        agent = RFM(
            sample_dim=action_dim,
            state_dim=state_dim,
            critic_config=critic_config,
            actor_hidden_size=actor_hidden_size,
            actor_hidden_layers=actor_hidden_layers,
            actor_emb_size=int(actor_cfg["emb_size"]),
            actor_lr=actor_lr,
            actor_max_grad_norm=actor_max_grad_norm,
            num_estimator_mc_samples=int(rfm_cfg["num_estimator_mc_samples"]),
            num_integration_steps=int(rfm_cfg["num_integration_steps"]),
            batch_size=int(trainer_cfg["batch_size"]),
            cv_weight=float(rfm_cfg["cv_weight"]),
            cv_weight_autotune=bool(rfm_cfg["cv_weight_autotune"]),
            cv_ridge=float(rfm_cfg["cv_ridge"]),
            cv_clip=float(rfm_cfg["cv_clip"]),
            rfm_on_policy=bool(rfm_cfg["rfm_on_policy"]),
            init_temperature=float(rfm_cfg["init_temperature"]),
            final_temperature=float(rfm_cfg["final_temperature"]),
            temperature_steps=int(rfm_cfg["temperature_steps"]),
            num_particles_training=int(rfm_cfg["num_particles_training"]),
            num_particles_data=int(rfm_cfg["num_particles_data"]),
            num_particles_eval=int(rfm_cfg["num_particles_eval"]),
            sampler_method=str(rfm_cfg["sampler_method"]).lower().strip(),
            t_min=float(rfm_cfg["t_min"]),
            seed=int(cfg["seed"]),
        )
    elif algo == "maxentdp":
        t_max = maxentdp_cfg["t_max"]
        agent = MaxEntDP(
            sample_dim=action_dim,
            state_dim=state_dim,
            critic_config=critic_config,
            actor_hidden_size=actor_hidden_size,
            actor_hidden_layers=actor_hidden_layers,
            actor_emb_size=int(actor_cfg["emb_size"]),
            actor_lr=actor_lr,
            actor_max_grad_norm=actor_max_grad_norm,
            num_diffusion_steps=int(maxentdp_cfg["num_diffusion_steps"]),
            init_temperature=float(maxentdp_cfg["init_temperature"]),
            final_temperature=float(maxentdp_cfg["final_temperature"]),
            temperature_steps=int(maxentdp_cfg["temperature_steps"]),
            clip_sampler=bool(maxentdp_cfg["clip_sampler"]),
            backup_entropy=bool(maxentdp_cfg["backup_entropy"]),
            log_prob_samples=int(maxentdp_cfg["log_prob_samples"]),
            log_prob_steps=int(maxentdp_cfg["log_prob_steps"]),
            eval_action_selection=bool(maxentdp_cfg["eval_action_selection"]),
            eval_candidate_num=int(maxentdp_cfg["eval_candidate_num"]),
            score_samples_num=int(maxentdp_cfg["score_samples_num"]),
            t_min=float(maxentdp_cfg["t_min"]),
            t_max=None if t_max is None else float(t_max),
            seed=int(cfg["seed"]),
        )
    elif algo == "qsm":
        agent = QSM(
            sample_dim=action_dim,
            state_dim=state_dim,
            critic_config=critic_config,
            actor_hidden_size=actor_hidden_size,
            actor_hidden_layers=actor_hidden_layers,
            actor_emb_size=int(actor_cfg["emb_size"]),
            actor_lr=actor_lr,
            actor_max_grad_norm=actor_max_grad_norm,
            num_diffusion_steps=int(qsm_cfg["num_diffusion_steps"]),
            ddpm_temperature=float(qsm_cfg["ddpm_temperature"]),
            clip_sampler=bool(qsm_cfg["clip_sampler"]),
            beta_schedule=str(qsm_cfg["beta_schedule"]),
            q_grad_scale=float(qsm_cfg["q_grad_scale"]),
            target_action_noise=float(qsm_cfg["target_action_noise"]),
            exploration_noise=float(qsm_cfg["exploration_noise"]),
            seed=int(cfg["seed"]),
        )

    # wandb
    logger_cfg = _require_mapping(cfg.get("logger", {}), "logger")
    debug = bool(logger_cfg.get("debug", False))
    exp_name = logger_cfg.get("exp")
    exp_value = "" if exp_name is None else str(exp_name).strip()
    has_exp = exp_value != ""
    job_type = algo.upper()
    if has_exp:
        job_type = f"{job_type}_{exp_value}"

    log_dir = Path(logger_cfg.get("log_dir", "logs"))
    if not log_dir.is_absolute():
        log_dir = (repo_root / log_dir).resolve()
    wandb_dir = Path(logger_cfg.get("wandb_dir", "wandb"))
    if not wandb_dir.is_absolute():
        wandb_dir = (repo_root / wandb_dir).resolve()
    wandb_project = logger_cfg.get("wandb_project")
    wandb_entity = logger_cfg.get("wandb_entity")
    project_arg = wandb_project if not debug else "none"
    if (not debug) and (not wandb_project):
        print("Warning: W&B logging will be disabled.")
    entity_arg = wandb_entity if not debug else "none"
    logger = Logger(
        work_dir=str(log_dir),
        wandb_dir=str(wandb_dir),
        seed=cfg["seed"],
        project=project_arg,
        entity=entity_arg,
        tags=[],
        group=str(env_name),
        job_type=job_type,
        config=cfg,
        disable_wandb=debug,
        save_agent=bool(logger_cfg.get("save_agent", False)),
    )

    trainer = Trainer(
        env_fn,
        agent=agent,
        logger=logger,
        max_steps=cfg["trainer"]["max_steps"],
        replay_buffer_size=cfg["trainer"]["replay_buffer_size"],
        start_training=cfg["trainer"]["start_training"],
        eval_interval=cfg["trainer"]["eval_interval"],
        log_interval=cfg["trainer"]["log_interval"],
        batch_size=cfg["trainer"]["batch_size"],
        reward_scale=cfg["trainer"]["reward_scale"],
        num_updates_per_step=cfg["trainer"]["updates_per_step"],
        seed=cfg["seed"],
        num_eval_envs=cfg["trainer"]["num_eval_envs"],
        episode_length=env_cfg.get("episode_length"),
    )

    # warm-up for jit
    dummy_states = jnp.zeros((trainer_cfg["batch_size"], state_dim))
    dummy_actions = jnp.zeros((trainer_cfg["batch_size"], action_dim))
    dummy_rewards = jnp.zeros((trainer_cfg["batch_size"],))
    dummy_masks = jnp.ones((trainer_cfg["batch_size"],))

    if algo == "sac":
        _ = trainer.agent._fused_update(
            trainer.agent.actor_params,
            trainer.agent.actor_opt_state,
            trainer.agent.log_alpha,
            trainer.agent.alpha_opt_state,
            trainer.agent.critic.Q_online_params,
            trainer.agent.critic.Q_target_params,
            trainer.agent.critic.optimizer_state_q,
            trainer.agent.critic.step,
            trainer.agent.key,
            dummy_states,
            dummy_actions,
            dummy_rewards,
            dummy_masks,
            dummy_states,
        )
    elif algo == "dqs":
        _ = trainer.agent._fused_update(
            trainer.agent.net_params,
            trainer.agent.net_optimizer_state,
            trainer.agent.energy_function.Q_online_params,
            trainer.agent.energy_function.Q_target_params,
            trainer.agent.energy_function.optimizer_state_q,
            trainer.agent.energy_function.step,
            trainer.agent.step,
            trainer.agent.key,
            dummy_states,
            dummy_actions,
            dummy_rewards,
            dummy_masks,
            dummy_states,
        )
    elif algo == "qvpo":
        _ = trainer.agent._fused_update(
            trainer.agent.policy_params,
            trainer.agent.policy_opt_state,
            trainer.agent.policy_target_params,
            trainer.agent.q1_params,
            trainer.agent.q2_params,
            trainer.agent.q1_target_params,
            trainer.agent.q2_target_params,
            trainer.agent.q1_opt_state,
            trainer.agent.q2_opt_state,
            trainer.agent.running_q_mean,
            trainer.agent.running_q_std,
            trainer.agent.critic_step,
            trainer.agent.step,
            trainer.agent.key,
            dummy_states,
            dummy_actions,
            dummy_rewards,
            dummy_masks,
            dummy_states,
        )
    elif algo in ("rfm", "maxentdp", "qsm"):
        _ = trainer.agent._fused_update(
            trainer.agent.net_params,
            trainer.agent.net_optimizer_state,
            trainer.agent.critic.Q_online_params,
            trainer.agent.critic.Q_target_params,
            trainer.agent.critic.optimizer_state_q,
            trainer.agent.critic.step,
            trainer.agent.step,
            trainer.agent.key,
            dummy_states,
            dummy_actions,
            dummy_rewards,
            dummy_masks,
            dummy_states,
        )

    trainer.train()


if __name__ == "__main__":
    app.run(main)
