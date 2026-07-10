"""CPR training / calibration / evaluation pipeline."""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from cpr.core import CPRCore
from cpr.data import (
    build_dataset_items,
    build_global_graph,
    load_json_dataset,
    split_calib,
)
from cpr.eval import evaluate_core_on_dataset


def load_datasets(args) -> Tuple[List[Dict], List[Dict], List[Dict], list]:
    train_parsed = load_json_dataset(args.train, max_samples=args.max_train)
    test_parsed = load_json_dataset(args.test, max_samples=args.max_test)
    global_triples = build_global_graph(train_parsed, test_parsed)

    train_items, calib_items = split_calib(
        train_parsed, seed=42, val_frac=args.calib_frac
    )
    train_items = build_dataset_items(train_items)
    calib_items = build_dataset_items(calib_items)
    test_items = build_dataset_items(test_parsed)

    print(
        f"[Data] train={len(train_items)}, calib={len(calib_items)}, "
        f"test={len(test_items)}, triples={len(global_triples)}"
    )
    return train_items, calib_items, test_items, global_triples


def build_encoder(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CPR] Loading encoder on {device}: {args.encoder_dir}")
    return SentenceTransformer(args.encoder_dir, device=device)


def core_kwargs(args, train_items, global_triples, encoder, skip_training=False, tau_hat=None):
    return dict(
        global_triples=global_triples,
        train_data=train_items,
        path_alpha=getattr(args, 'l1', 0.3),
        ans_alpha=getattr(args, 'l2', 0.2),
        post_alpha=args.alpha,
        max_hop=args.max_hop,
        encoder=encoder,
        use_thompson=args.use_ts,
        ts_weight=getattr(args, 'ts_weight', 0.3),
        beam_size=args.beam_size,
        treeg_branch_size=args.treeg_branch_size,
        treeg_active_size=args.treeg_active_size,
        treeg_weights={"local": args.w_local, "path": args.w_path, "prior": args.w_prior},
        learn_residual_value=bool(args.learn_residual_value),
        residual_lambda=args.residual_lambda,
        delta_clip=args.delta_clip,
        delta_l2=args.delta_l2,
        value_hidden=args.value_hidden,
        value_lr=args.value_lr,
        value_epochs=args.value_epochs,
        value_batch_size=args.value_batch_size,
        value_max_negs=args.value_max_negs,
        value_l2=args.value_l2,
        hard_neg_frac=args.hard_neg_frac,
        value_norm=bool(args.value_norm),
        value_use_embeddings=int(args.value_use_embeddings),
        puct_calib=bool(args.puct_calib),
        puct_calib_num_sims=int(args.puct_calib_num_sims),
        puct_calib_cpuct=float(args.puct_calib_cpuct),
        puct_calib_temp=float(args.puct_calib_temp),
        puct_calib_prior_w=float(args.puct_calib_prior_w),
        puct_calib_update_scale=float(args.puct_calib_update_scale),
        puct_calib_fail_beta=float(args.puct_calib_fail_beta),
        use_llm=args.use_llm,
        llm_model_path=args.llm_model,
        vLLM_url=args.vllm_url,
        conformal_mode=args.conformal_mode,
        auto_compute_thresholds=(args.conformal_mode == "legacy"),
        use_global_post_threshold=False,
        skip_training=skip_training,
        tau_hat=tau_hat,
    )


def train_phase(args, train_items, global_triples, encoder) -> CPRCore:
    print("\n[Phase 1] PUCT + RCVNet training on D_train")
    return CPRCore(**core_kwargs(args, train_items, global_triples, encoder))


def calibrate_phase(args, core: CPRCore, calib_items) -> Dict:
    print("\n[Phase 2] Conformal calibration on D_cal")
    stats = core.fit_post_threshold_from_retrieval(calib_items, post_alpha=args.alpha)
    key = "tau_hat" if stats.get("mode") == "path" else "q_hat_post"
    print(f"[Phase 2] {key}={stats.get(key, stats.get('tau_hat')):.6f} "
          f"(used={stats.get('used')}, miss={stats.get('miss')})")
    return stats


def evaluate_phase(core: CPRCore, test_items) -> Dict:
    print("\n[Phase 3] Evaluation on D_test")
    return evaluate_core_on_dataset(core, test_items)


def save_checkpoint(path: str, core: CPRCore, calib_stats: Dict, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "calib_stats": calib_stats,
        "conformal_mode": core.conformal_mode,
        "tau_hat": core.tau_hat,
        "q_hat_post": getattr(core, "q_hat_post", None),
        "post_alpha": core.post_alpha,
        "args_alpha": args.alpha,
    }
    payload["relation_posteriors"] = {
        k: list(v) for k, v in core.relation_posteriors.items()
    }
    if core.value_model is not None:
        payload["value_model"] = core.value_model.state_dict()
        payload["value_feat_mean"] = (
            core.value_feat_mean.cpu().tolist() if core.value_feat_mean is not None else None
        )
        payload["value_feat_std"] = (
            core.value_feat_std.cpu().tolist() if core.value_feat_std is not None else None
        )
    torch.save(payload, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(path: str, core: CPRCore):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "value_model" in ckpt and core.value_model is not None:
        core.value_model.load_state_dict(ckpt["value_model"])
        core.value_model.eval()
    if ckpt.get("value_feat_mean") is not None:
        import torch
        core.value_feat_mean = torch.tensor(ckpt["value_feat_mean"])
    if ckpt.get("value_feat_std") is not None:
        import torch
        core.value_feat_std = torch.tensor(ckpt["value_feat_std"])
    if ckpt.get("tau_hat") is not None:
        core.tau_hat = float(ckpt["tau_hat"])
    if ckpt.get("q_hat_post") is not None:
        core.q_hat_post = float(ckpt["q_hat_post"])
    if "relation_posteriors" in ckpt:
        from collections import defaultdict
        core.relation_posteriors = defaultdict(
            lambda: [1.0, 1.0],
            {k: list(v) for k, v in ckpt["relation_posteriors"].items()},
        )
    core.use_global_post_threshold = True
    return ckpt
