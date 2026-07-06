# scripts/run_influence.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
local_rank = os.environ.get("LOCAL_RANK")
if local_rank is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
import datetime
import hashlib
import argparse
import yaml
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
)

from core.ekfac_blocks import EKFAC_QK_Head, EKFAC_QKVO_Head
from core.ekfac_fit import stage1A_accumulate_AS, stage1B_fit_lambda
from core.influence_phase2 import phase2_score_qkonly, phase2_score_qkvo
from probes import get_probe
from utils.gather import gather_heap_as_tensors
from data.loader import build_dataloaders


def _ekfac_cache_path(output_dir: str, cfg: dict) -> str:
    """根据 target 和 ekfac 参数生成唯一的缓存文件名。"""
    target_cfg = cfg.get("target", {})
    ekfac_cfg  = cfg.get("ekfac", {})
    key_str = (
        f"layer={target_cfg.get('layer')}"
        f"_head={target_cfg.get('head')}"
        f"_mode={target_cfg.get('mode')}"
        f"_damping={ekfac_cfg.get('damping', 1e-5)}"
        f"_alpha={ekfac_cfg.get('damping_alpha', 0.1)}"
    )
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]
    filename  = f"ekfac_cache_{key_hash}.pt"
    return os.path.join(output_dir, filename)


def save_ekfac(ekfac, path: str) -> None:
    """将 ekfac 的特征分解结果和 Lambda 序列化到文件。"""
    if isinstance(ekfac, EKFAC_QK_Head):
        state = {
            "type": "qk",
            "Q_A": ekfac.Q_A.cpu(),
            "Q_S": ekfac.Q_S.cpu(),
            "Lambda": ekfac.Lambda.cpu(),
        }
    else:
        state = {
            "type": "qkvo",
            "Q_A": [t.cpu() for t in ekfac.Q_A],
            "Q_S": [t.cpu() for t in ekfac.Q_S],
            "Lambda": [t.cpu() for t in ekfac.Lambda],
        }
    torch.save(state, path)


def load_ekfac(ekfac, path: str, device) -> None:
    """从文件恢复 ekfac 的特征分解结果和 Lambda。"""
    state = torch.load(path, map_location=device)
    if isinstance(ekfac, EKFAC_QK_Head):
        ekfac.Q_A    = state["Q_A"].to(device)
        ekfac.Q_S    = state["Q_S"].to(device)
        ekfac.Lambda = state["Lambda"].to(device)
    else:
        ekfac.Q_A    = [t.to(device) for t in state["Q_A"]]
        ekfac.Q_S    = [t.to(device) for t in state["Q_S"]]
        ekfac.Lambda = [t.to(device) for t in state["Lambda"]]


def run(cfg: dict, model: HookedTransformer, rank: int, world_size: int) -> None:
    is_main    = (rank == 0)
    device     = next(model.parameters()).device
    inner      = model.module if hasattr(model, "module") else model
    mode       = cfg["target"]["mode"]
    layer_idx  = cfg["target"]["layer"]
    head_idx   = cfg["target"]["head"]
    d_model    = inner.cfg.d_model
    d_head     = inner.cfg.d_head
    seq_length = cfg["data"]["seq_length"]
    dtype      = getattr(torch, cfg.get("dtype", "float32"))
    top_k      = cfg["output"]["top_k"]
    output_dir = cfg["output"]["dir"]
    ekfac_cfg  = cfg.get("ekfac", {})
    cache_path = _ekfac_cache_path(output_dir, cfg)

    dl_ekfac, dl_influence = build_dataloaders(
        npy_path=cfg["data"]["npy_path"],
        num_train_samples=cfg["data"]["num_train_samples"],
        batch_size_ekfac=cfg["data"]["batch_size_ekfac"],
        batch_size_influence=cfg["data"]["batch_size_influence"],
        rank=rank,
        world_size=world_size,
        num_workers=cfg["data"].get("num_workers", 2),
        seq_length=cfg["data"]["seq_length"],
    )

    # Probe gradient
    if is_main:
        print(f"[{rank}/{world_size}] Computing probe gradient (mode={mode})")
    probe       = get_probe(cfg["probe"]["type"])()
    probe_grads = probe.compute_grad(model, cfg, mode)
    if is_main:
        print(f"[{rank}/{world_size}] Probe gradient ready.")

    # 构建 ekfac 对象
    if mode == "qk":
        ekfac = EKFAC_QK_Head(
            d_model=d_model, d_head=d_head,
            damping=ekfac_cfg.get("damping", 1e-5),
            damping_alpha=ekfac_cfg.get("damping_alpha", 0.1),
        )
    else:
        ekfac = EKFAC_QKVO_Head(
            d_model=d_model, d_head=d_head,
            damping=ekfac_cfg.get("damping", 1e-5),
            damping_alpha=ekfac_cfg.get("damping_alpha", 0.1),
        )

    # 检查是否存在 Stage 1A/1B 缓存（以 rank0 判断为准，广播给所有 rank）
    cache_exists = os.path.isfile(cache_path)
    if dist.is_initialized():
        flag = torch.tensor([int(cache_exists)], dtype=torch.long, device=device)
        dist.broadcast(flag, src=0)
        cache_exists = bool(flag.item())

    if cache_exists:
        if is_main:
            print(f"[{rank}/{world_size}] Found ekfac cache: {cache_path}, skipping Stage 1A/1B.")
        load_ekfac(ekfac, cache_path, device)
    else:
        # Stage 1A
        if is_main:
            print(f"[{rank}/{world_size}] Stage 1A: accumulating A/S")
        elapsed_1A = stage1A_accumulate_AS(
            model=model, dataloader=dl_ekfac, ekfac=ekfac,
            layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
            d_model=d_model, d_head=d_head, dtype=dtype, device=device, verbose=is_main,
        )
        if is_main:
            print(f"[{rank}/{world_size}] Stage 1A done ({elapsed_1A:.1f}s)")

        # Stage 1B
        if is_main:
            print(f"[{rank}/{world_size}] Stage 1B: fitting Lambda")
        elapsed_1B = stage1B_fit_lambda(
            model=model, dataloader=dl_ekfac, ekfac=ekfac,
            layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
            d_model=d_model, d_head=d_head, dtype=dtype, device=device, verbose=is_main,
        )
        if is_main:
            print(f"[{rank}/{world_size}] Stage 1B done ({elapsed_1B:.1f}s)")
            os.makedirs(output_dir, exist_ok=True)
            save_ekfac(ekfac, cache_path)
            print(f"[{rank}/{world_size}] ekfac cache saved: {cache_path}")
        if dist.is_initialized():
            dist.barrier()  # 确保所有 rank 等待 rank0 写完缓存再继续

    # p = H^{-1} v
    if mode == "qk":
        p_qk = ekfac.inverse_hvp(probe_grads[0])
    else:
        p_qk = ekfac.inverse_hvp(0, probe_grads[0])
        p_v  = ekfac.inverse_hvp(1, probe_grads[1])
        p_o  = ekfac.inverse_hvp(2, probe_grads[2])
    if is_main:
        print(f"[{rank}/{world_size}] p ready.")

    # Stage 2
    if is_main:
        print(f"[{rank}/{world_size}] Stage 2: scoring")
    if mode == "qk":
        pos_heap, neg_heap, n_samples, elapsed_2 = phase2_score_qkonly(
            model=model, dataloader=dl_influence, p_qk=p_qk,
            layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
            dtype=dtype, device=device, top_k=top_k, verbose=is_main,
        )
    else:
        pos_heap, neg_heap, n_samples, elapsed_2 = phase2_score_qkvo(
            model=model, dataloader=dl_influence,
            p_qk=p_qk, p_v=p_v, p_o=p_o,
            layer_idx=layer_idx, head_idx=head_idx, seq_length=seq_length,
            dtype=dtype, device=device, top_k=top_k, verbose=is_main,
        )
    if is_main:
        print(f"[{rank}/{world_size}] Stage 2 done ({elapsed_2:.1f}s, {n_samples} samples)")

    # Gather and save
    pos_all = gather_heap_as_tensors(pos_heap, world_size, device, rank)
    neg_all = gather_heap_as_tensors(neg_heap, world_size, device, rank)

    if is_main:
        pos_all = pos_all[pos_all[:, 0].argsort()[::-1]][:top_k]
        neg_all = neg_all[neg_all[:, 0].argsort()[::-1]][:top_k]
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/top_pos.npy", pos_all)
        np.save(f"{output_dir}/top_neg.npy", neg_all)
        print(f"[{rank}/{world_size}] Saved to {output_dir}/top_pos.npy, top_neg.npy")


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=3600))
        return rank, world_size, 0
    else:
        print("Not running in distributed mode. Using single process.")
        return 0, 1, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()
    print(f'rank: {rank}, local_rank: {local_rank}, world_size: {world_size}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    OFFICIAL_MODEL_NAMES.append(cfg["model"]["path"])
    MODEL_ALIASES[cfg["model"]["path"]] = [cfg["model"]["alias"]]
    make_model_alias_map()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True)

    model = HookedTransformer.from_pretrained_no_processing(
        cfg["model"]["path"],
        dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
        tokenizer=tokenizer,
        device=device
    )

    model.cfg.use_attn_result = False
    model.cfg.track_head_grad_idx = cfg['target']['head']
    model.eval()

    # 冻结参数；开启该层 Q/K/V/O（阶段2对参数求导；阶段1对中间量求导）
    for p in model.parameters():
        p.requires_grad = False
    attn_layer = model.blocks[cfg['target']['layer']].attn
    attn_layer.W_Q.requires_grad = True
    attn_layer.W_K.requires_grad = True
    attn_layer.W_V.requires_grad = True
    attn_layer.W_O.requires_grad = True

    run(cfg, model, rank, world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()





