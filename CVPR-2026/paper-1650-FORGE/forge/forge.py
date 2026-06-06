import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader


@dataclass
class Cfg:
    PT_PATH: str = "./data/real_sites.pt"
    NPZ_HOSP_PATHS: List[str] = field(
        default_factory=lambda: [
            "./data/synth/site1.npz",
            "./data/synth/site2.npz",
            "./data/synth/site3.npz",
            "./data/synth/site4.npz",
        ]
    )
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_RATIO: float = 0.20
    BATCH_SIZE: int = 32
    LR: float = 1e-3
    WEIGHT_DECAY: float = 5e-4
    EPOCHS_PER_TASK: int = 200
    WARMUP_EPOCHS: int = 20
    PATIENCE_ON_ACC: int = 80
    GRAD_CLIP: float = 1.0
    VERBOSE: bool = True
    HIDDEN: int = 128
    EMBED: int = 128
    LAYERS: int = 4
    DROPOUT: float = 0.30
    POOLING: str = "mean+max"
    USE_BN: bool = True
    ALPHA: float = 0.10
    BETA: float = 0.40
    GAMMA_G: float = 0.30
    GAMMA_R: float = 0.15
    REPLAY_AFTER_FIRST: bool = True
    REPLAY_INCLUDE_CURRENT: bool = False
    REPLAY_MB_SIZE: int = 64
    HOSP_CTX_DIM: int = 2
    HOSP_LAMBDA: float = 10.0
    HOSP_SIGMA: float = 0.10
    HOSP_GAMMA_FORGET: float = 0.98
    HOSP_SOFTMAX_TEMP: float = 0.30
    HOSP_SOFTMAX_FLOOR: float = 0.05
    SAMP_CTX_DIM: int = 2
    SAMP_LAMBDA: float = 5.0
    SAMP_SIGMA: float = 0.15
    SAMP_GAMMA_FORGET: float = 0.98
    PROTOTYPE_EMA_ALPHA: float = 0.20
    MAHAL_JITTER: float = 1e-4
    TOT_SYNTH_CAPACITY: int = 256
    MIN_PER_HOSPITAL: int = 32
    MAX_PER_HOSPITAL: int = 128
    PROXY_DEV_PER_H: int = 16
    BANDIT_REFRESH_EVERY: int = 80
    REWARD_TEMP: float = 0.05
    ADJ_THRESHOLD: float = 0.4
    SYN_SVD_FALLBACK: bool = True


try:
    from .data import prepare_data, set_global_seed
    from .model import build_model
    from .replay import (
        ContextualTSArms,
        GenReplayMemory,
        ProxyDevBank,
        SampleCTXPerHospital,
        build_hospital_contexts_2feat,
        build_sample_contexts_for_h_2feat,
        cache_teacher_predictions_for_hospital,
        eval_loader,
        eval_seen_plus_current,
        farthest_first,
        freeze_teacher_model,
        make_entries_with_teacher,
    )
except ImportError:
    from data import prepare_data, set_global_seed
    from model import build_model
    from replay import (
        ContextualTSArms,
        GenReplayMemory,
        ProxyDevBank,
        SampleCTXPerHospital,
        build_hospital_contexts_2feat,
        build_sample_contexts_for_h_2feat,
        cache_teacher_predictions_for_hospital,
        eval_loader,
        eval_seen_plus_current,
        farthest_first,
        freeze_teacher_model,
        make_entries_with_teacher,
    )


def _relational_loss(g_s: torch.Tensor, g_t: torch.Tensor) -> torch.Tensor:
    if g_s.shape[0] <= 1:
        return g_s.new_tensor(0.0)
    ds = torch.cdist(F.normalize(g_s, dim=1), F.normalize(g_s, dim=1), p=2.0)
    dt = torch.cdist(F.normalize(g_t, dim=1), F.normalize(g_t, dim=1), p=2.0)
    return F.mse_loss(ds, dt)


def _clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _weighted_replay_loss(model, replay_memory, cfg: Cfg, device: torch.device):
    if replay_memory is None or len(replay_memory) == 0 or cfg.REPLAY_MB_SIZE <= 0:
        z = next(model.parameters()).new_tensor(0.0)
        return z, {"replay_ce": 0.0, "logit_kd": 0.0, "graph_kd": 0.0, "rel_kd": 0.0}
    batch, logits_t, y_t, g_t, _, _ = replay_memory.sample_weighted(cfg.REPLAY_MB_SIZE, device)
    if batch is None:
        z = next(model.parameters()).new_tensor(0.0)
        return z, {"replay_ce": 0.0, "logit_kd": 0.0, "graph_kd": 0.0, "rel_kd": 0.0}
    logits_s, g_s = model.forward_all(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
    replay_ce = F.cross_entropy(logits_s, y_t)
    logit_kd = F.mse_loss(logits_s, logits_t)
    graph_kd = F.mse_loss(g_s, g_t)
    rel_kd = _relational_loss(g_s, g_t) if cfg.GAMMA_R > 0 else logits_s.new_tensor(0.0)
    loss = cfg.ALPHA * replay_ce + cfg.BETA * logit_kd + cfg.GAMMA_G * graph_kd + cfg.GAMMA_R * rel_kd
    return loss, {
        "replay_ce": float(replay_ce.detach().cpu()),
        "logit_kd": float(logit_kd.detach().cpu()),
        "graph_kd": float(graph_kd.detach().cpu()),
        "rel_kd": float(rel_kd.detach().cpu()),
    }


def build_proxy_dev_and_candidates(synth_per_hosp, k_dev: int, seed: int):
    rng = np.random.RandomState(seed)
    proxy_idx: Dict[int, List[int]] = {}
    candidate_idx: Dict[int, List[int]] = {}
    per_h_ds: Dict[int, List[Any]] = {}
    for h, ds in enumerate(synth_per_hosp, start=1):
        per_h_ds[h] = ds
        order = np.arange(len(ds))
        rng.shuffle(order)
        n_dev = min(int(k_dev), len(order))
        proxy_idx[h] = [int(i) for i in order[:n_dev]]
        candidate_idx[h] = [int(i) for i in order[n_dev:]]
        if len(candidate_idx[h]) == 0:
            candidate_idx[h] = [int(i) for i in order]
    return ProxyDevBank(per_h_ds, proxy_idx, torch.device("cpu")), candidate_idx


def _select_replay_indices(ds, candidate_idx, cache_h, scores, cap: int):
    if cap <= 0 or len(candidate_idx) == 0:
        return [], []
    uid_to_idx = {int(ds[i].uid.item()): int(i) for i in candidate_idx}
    ranked = [u for u, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if u in uid_to_idx]
    if len(ranked) == 0:
        ranked = list(uid_to_idx.keys())
    pre_k = min(len(ranked), max(cap * 3, cap))
    ranked = ranked[:pre_k]
    g_list = [cache_h[u]["g"].float() for u in ranked if u in cache_h]
    uids_for_g = [u for u in ranked if u in cache_h]
    if len(g_list) > 0:
        selected_uids = farthest_first(uids_for_g, g_list, cap)
    else:
        selected_uids = ranked[:cap]
    selected_idx = [uid_to_idx[u] for u in selected_uids if u in uid_to_idx]
    return selected_idx, selected_uids


def _refresh_replay_memory(
    cfg: Cfg,
    model,
    active_hs: List[int],
    val_loaders: Dict[int, DataLoader],
    synth_per_hosp,
    candidate_idx: Dict[int, List[int]],
    teacher_models: Dict[int, Any],
    teacher_store: Dict[int, Dict[int, Dict[str, Any]]],
    proxy_bank: ProxyDevBank,
    hosp_ts: ContextualTSArms,
    sample_ts: SampleCTXPerHospital,
    best_so_far: List[float],
    replay_memory: GenReplayMemory,
    device: torch.device,
) -> Dict[str, Any]:
    active_hs = [h for h in active_hs if h in teacher_models and len(candidate_idx.get(h, [])) > 0]
    if len(active_hs) == 0:
        replay_memory.clear()
        return {"weights": {}, "caps": {}, "selected": {}}
    acc_map = {h: float(eval_loader(model, val_loaders[h], device)["acc"]) for h in active_hs}
    hosp_ctx = build_hospital_contexts_2feat(active_hs, acc_map, best_so_far)
    h_scores = hosp_ts.sample_scores(hosp_ctx)
    weights = hosp_ts.softmax_weights(h_scores, cfg.HOSP_SOFTMAX_TEMP, cfg.HOSP_SOFTMAX_FLOOR)
    candidate_sizes = {h: len(candidate_idx.get(h, [])) for h in active_hs}
    caps = hosp_ts.allocate(weights, cfg.TOT_SYNTH_CAPACITY, cfg.MIN_PER_HOSPITAL, cfg.MAX_PER_HOSPITAL, candidate_sizes)
    replay_memory.clear()
    replay_memory.set_caps_and_weights(caps, weights)
    selected_summary: Dict[int, int] = {}
    for h in active_hs:
        ds = synth_per_hosp[h - 1]
        idxs = candidate_idx.get(h, [])
        cand_ds = [ds[i] for i in idxs]
        cache_teacher_predictions_for_hospital(teacher_models[h], cand_ds, teacher_store, h, device)
        cache_h = _cache_logits_g_map(model, cand_ds, device)
        sample_ts.update_proto_cov(h, cache_h)
        phi_map = build_sample_contexts_for_h_2feat(h, cache_h, sample_ts)
        s_scores = sample_ts.sample_scores(h, phi_map) if len(phi_map) > 0 else {}
        selected_idx, selected_uids = _select_replay_indices(ds, idxs, cache_h, s_scores, caps.get(h, 0))
        entries = make_entries_with_teacher(teacher_models, ds, selected_idx, teacher_store, h, device)
        replay_memory.insert_entries(h, entries)
        selected_summary[h] = len(entries)
        new_ce = proxy_bank.ce_loss(model, h)
        reward = proxy_bank.reward(h, new_ce, cfg.REWARD_TEMP)
        hosp_ts.update_one(h, hosp_ctx[h], reward)
        sample_ts.update(h, selected_uids, phi_map, reward)
    replay_memory.truncate_to_caps()
    replay_memory.truncate_global_to_total(cfg.TOT_SYNTH_CAPACITY)
    return {"weights": weights, "caps": caps, "selected": selected_summary}


try:
    from .replay import cache_logits_g_map as _cache_logits_g_map
except ImportError:
    from replay import cache_logits_g_map as _cache_logits_g_map


def train_one_hospital_CTXT(
    cfg: Cfg,
    model,
    hospital_id: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    replay_memory: GenReplayMemory,
    refresh_fn=None,
):
    device = torch.device(cfg.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS_PER_TASK, eta_min=cfg.LR * 0.01)
    best_state = _clone_state(model)
    best_acc = -1.0
    wait = 0
    ce = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, cfg.EPOCHS_PER_TASK + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model.forward_all(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
            current_loss = ce(logits, batch.y.view(-1))
            replay_loss, _ = _weighted_replay_loss(model, replay_memory, cfg, device)
            loss = current_loss + replay_loss
            loss.backward()
            if cfg.GRAD_CLIP and cfg.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()  # Cosine annealing: step per epoch
        metrics = eval_loader(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, **metrics}
        history.append(row)
        if metrics["acc"] > best_acc:
            best_acc = float(metrics["acc"])
            best_state = _clone_state(model)
            wait = 0
        else:
            wait += 1
        if refresh_fn is not None and cfg.BANDIT_REFRESH_EVERY > 0 and epoch % cfg.BANDIT_REFRESH_EVERY == 0:
            refresh_fn()
        if cfg.VERBOSE and (epoch == 1 or epoch % 20 == 0 or epoch == cfg.EPOCHS_PER_TASK):
            print(
                f"hospital={hospital_id} epoch={epoch} loss={row['loss']:.4f} "
                f"val_acc={metrics['acc']:.4f} val_f1={metrics['f1']:.4f}"
            )
        if epoch >= cfg.WARMUP_EPOCHS and wait >= cfg.PATIENCE_ON_ACC:
            break
    model.load_state_dict(best_state)
    last_metrics = eval_loader(model, val_loader, device)
    return {"best_acc": best_acc, "last": last_metrics, "history": history}


def _avg_last_row(last_row: List[float]) -> float:
    return float(np.mean(last_row)) if len(last_row) else 0.0


def _bwt_from_triangular(metric_matrix: List[List[float]]) -> float:
    if len(metric_matrix) <= 1:
        return 0.0
    last = metric_matrix[-1]
    vals = []
    for i in range(len(last) - 1):
        if i < len(metric_matrix[i]):
            vals.append(last[i] - metric_matrix[i][i])
    return float(np.mean(vals)) if vals else 0.0


def _summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    metric_matrix = results["metric_matrix"]
    last_row = metric_matrix[-1] if metric_matrix else []
    anytime = [float(np.mean(row)) for row in metric_matrix if len(row) > 0]
    return {
        "last_avg_acc": _avg_last_row(last_row),
        "aaa": float(np.mean(anytime)) if anytime else 0.0,
        "bwt": _bwt_from_triangular(metric_matrix),
        "last_row": last_row,
    }


def run_experiment(cfg: Cfg):
    set_global_seed(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    splits, synth_per_hosp, in_dim = prepare_data(cfg)
    model = build_model(cfg, in_dim)
    train_loaders: Dict[int, DataLoader] = {}
    val_loaders: Dict[int, DataLoader] = {}
    for h, (tr, va) in enumerate(splits, start=1):
        train_loaders[h] = DataLoader(tr, batch_size=cfg.BATCH_SIZE, shuffle=True)
        val_loaders[h] = DataLoader(va, batch_size=cfg.BATCH_SIZE, shuffle=False)
    proxy_bank, candidate_idx = build_proxy_dev_and_candidates(synth_per_hosp, cfg.PROXY_DEV_PER_H, cfg.SEED + 17)
    proxy_bank.device = device
    replay_memory = GenReplayMemory(cfg.SEED)
    hosp_ts = ContextualTSArms(cfg.HOSP_CTX_DIM, cfg.HOSP_LAMBDA, cfg.HOSP_SIGMA, cfg.HOSP_GAMMA_FORGET)
    sample_ts = SampleCTXPerHospital(
        cfg.SAMP_CTX_DIM,
        cfg.SAMP_LAMBDA,
        cfg.SAMP_SIGMA,
        cfg.SAMP_GAMMA_FORGET,
        alpha=cfg.PROTOTYPE_EMA_ALPHA,
        jitter_Sigma=cfg.MAHAL_JITTER,
    )
    teacher_models: Dict[int, Any] = {}
    teacher_store: Dict[int, Dict[int, Dict[str, Any]]] = {}
    best_so_far = [0.0 for _ in range(len(splits))]
    metric_matrix: List[List[float]] = []
    task_logs: List[Dict[str, Any]] = []

    for h in range(1, len(splits) + 1):
        previous_hs = list(range(1, h)) if cfg.REPLAY_AFTER_FIRST else []

        def refresh_previous():
            return _refresh_replay_memory(
                cfg,
                model,
                previous_hs,
                val_loaders,
                synth_per_hosp,
                candidate_idx,
                teacher_models,
                teacher_store,
                proxy_bank,
                hosp_ts,
                sample_ts,
                best_so_far,
                replay_memory,
                device,
            )

        if previous_hs:
            refresh_previous()
        else:
            replay_memory.clear()
        train_info = train_one_hospital_CTXT(
            cfg,
            model,
            h,
            train_loaders[h],
            val_loaders[h],
            replay_memory,
            refresh_fn=refresh_previous if previous_hs else None,
        )
        seen_hs = list(range(1, h + 1))
        row = [float(eval_loader(model, val_loaders[j], device)["acc"]) for j in seen_hs]
        for j, acc in enumerate(row):
            best_so_far[j] = max(best_so_far[j], acc)
        metric_matrix.append(row)
        teacher_models[h] = freeze_teacher_model(model)
        _refresh_replay_memory(
            cfg,
            model,
            seen_hs,
            val_loaders,
            synth_per_hosp,
            candidate_idx,
            teacher_models,
            teacher_store,
            proxy_bank,
            hosp_ts,
            sample_ts,
            best_so_far,
            replay_memory,
            device,
        )
        task_logs.append({"hospital": h, "train": train_info, "seen_acc": row})
        if cfg.VERBOSE:
            avg_seen, seen_acc = eval_seen_plus_current(model, [val_loaders[j] for j in range(1, h)], val_loaders[h], device)
            print(f"after_hospital={h} avg_seen_acc={avg_seen:.4f} seen_acc={seen_acc}")

    results = {"metric_matrix": metric_matrix, "tasks": task_logs}
    results["summary"] = _summarize_results(results)
    return results


def main(cfg: Cfg = None):
    cfg = cfg or Cfg()
    results = run_experiment(cfg)
    print(json.dumps(results["summary"], indent=2))
    return results


if __name__ == "__main__":
    main()
