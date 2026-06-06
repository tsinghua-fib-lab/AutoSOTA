#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
from typing import List, Dict, Any

import numpy as np
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPModel, CLIPProcessor
import math
import time

def str2bool(v):
    if isinstance(v, bool):
        return v  

    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"必须输入合法的布尔值字符串：true 或 false（不区分大小写）")


# ======== Utils: seed =========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ======== Dataset =========
class IQADataset(Dataset):
    def __init__(self, json_files: List[str], image_root: str):
        self.items = []
        self.image_root = image_root

        for fp in json_files:
            with open(fp, "r", encoding="utf-8") as f:
                file_data = json.load(f)

            if isinstance(file_data, list):
                records = file_data
            elif isinstance(file_data, dict):
                if "results" in file_data:
                    records = file_data["results"]
                elif "data" in file_data:
                    records = file_data["data"]
                else:
                    records = [file_data]
            else:
                records = [file_data]

            for r in records:
                if ("image" in r) and ("gt_score_norm" in r):
                    path = os.path.join(image_root, r["image"])
                    self.items.append({
                        "path": path,
                        "score": float(r["gt_score_norm"]),
                        "desc": r.get("description", ""),
                        "id": r.get("id", None),
                        "rel_image": r["image"]
                    })

        if len(self.items) == 0:
            raise RuntimeError("No valid samples found in provided JSON files.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        try:
            img = Image.open(rec["path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Fail to open image: {rec['path']} - {e}")

        return {
            "image": img,                      # PIL.Image
            "score": torch.tensor(rec["score"], dtype=torch.float32),
            "id": rec["id"],
            "rel_image": rec["rel_image"],
            "desc": rec["desc"]
        }

def iqad_collate(batch: List[Dict[str, Any]]):
    images = [b["image"] for b in batch]  # PIL list
    scores = torch.stack([b["score"] for b in batch], dim=0)  # [B]
    meta = [{"id": b["id"], "rel_image": b["rel_image"], "desc": b["desc"]} for b in batch]
    return {"images": images, "scores": scores, "meta": meta}

# ======== Metrics =========
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mae = float(np.mean(np.abs(pred - gt)))

    plcc = np.nan
    srcc = np.nan
    try:
        from scipy.stats import pearsonr, spearmanr
        plcc = float(pearsonr(pred, gt)[0])
        srcc = float(spearmanr(pred, gt)[0])
    except Exception:

        p_c = pred - pred.mean()
        g_c = gt - gt.mean()
        denom = np.sqrt((p_c ** 2).sum()) * np.sqrt((g_c ** 2).sum())
        plcc = float((p_c * g_c).sum() / (denom + 1e-8))

        pr = pred.argsort().argsort().astype(np.float64)
        gr = gt.argsort().argsort().astype(np.float64)
        pr -= pr.mean(); gr -= gr.mean()
        denom_r = np.sqrt((pr**2).sum()) * np.sqrt((gr**2).sum())
        srcc = float((pr * gr).sum() / (denom_r + 1e-8))

    return {"MAE": mae, "PLCC": plcc, "SRCC": srcc}


class PrototypeScorer(nn.Module):
    def __init__(self, 
                 clip_emb_dim: int,  
                 pca_path: str = None,  
                 basis_path: str = None,  
                 num_prototypes: int = 32, 
                 init_std: float = 0.02,
                 learnable_tau: bool = False,
                 init_tau: float = 0.25,
                 learnable_pca: bool = False,  
                 learnable_prototypes: bool = True,  
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.clip_emb_dim = clip_emb_dim  
        self.learnable_pca = learnable_pca
        self.learnable_tau = learnable_tau
        self.learnable_prototypes = learnable_prototypes  


        self.pca_dim = clip_emb_dim  

        pca_mean = np.zeros(clip_emb_dim)
        pca_scale = np.ones(clip_emb_dim)
        pca_components = np.eye(clip_emb_dim) 

        if pca_path and os.path.exists(pca_path):
            print(f"[PrototypeScorer] 从 {pca_path} 加载PCA模型初始化...")
            try:
                pca = joblib.load(pca_path)

                pca_mean = pca.mean_ if hasattr(pca, 'mean_') else pca_mean
                pca_scale = np.sqrt(pca.var_) if hasattr(pca, 'var_') else pca_scale
                pca_components = pca.components_ if hasattr(pca, 'components_') else pca_components
                self.pca_dim = pca_components.shape[0] 

                pca_scale[pca_scale < 1e-10] = 1.0
                print(f"[PCA] 输入维度{clip_emb_dim} → 输出维度{self.pca_dim}, 解释方差比{getattr(pca, 'explained_variance_ratio_.sum()', 1.0):.4f}")
            except Exception as e:
                print(f"[警告] PCA加载失败，使用默认初始化: {e}")


        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32, device=device)  # 均值不学习
        self.pca_scale = nn.Parameter(
            torch.tensor(pca_scale, dtype=torch.float32, device=device),
            requires_grad= learnable_pca
        )
        self.pca_components = nn.Parameter(
            torch.tensor(pca_components, dtype=torch.float32, device=device),
            requires_grad=learnable_pca
        )


        self.num_prototypes = num_prototypes

        proto_init = torch.empty(num_prototypes, self.pca_dim, dtype=torch.float32)
        nn.init.normal_(proto_init, std=init_std)
        score_init = torch.empty(num_prototypes, dtype=torch.float32)
        nn.init.uniform_(score_init, a=0.3, b=0.7)  
        init_tau = init_tau 

        if basis_path and os.path.exists(basis_path):
            print(f"[PrototypeScorer] 从 {basis_path} 加载基向量/分数初始化...")
            try:
                data = np.load(basis_path)

                basis_vectors = data.get("vectors", proto_init.numpy())
                basis_scores = data.get("scores", score_init.numpy())

                assert basis_vectors.shape[1] == self.pca_dim, \
                    f"基向量维度{basis_vectors.shape[1]}与PCA输出维度{self.pca_dim}不匹配"
                self.num_prototypes = basis_vectors.shape[0]

                proto_init = torch.tensor(basis_vectors, dtype=torch.float32)
                score_init = torch.tensor(basis_scores, dtype=torch.float32)
                print(f"[Basis] 加载{self.num_prototypes}个基向量, 初始温度{init_tau:.6f}")
            except Exception as e:
                print(f"[警告] 基向量加载失败，使用随机初始化: {e}")


        self.prototypes = nn.Parameter(
            proto_init.to(device),
            requires_grad=learnable_prototypes  
        )
        self.proto_scores = nn.Parameter(
            score_init.to(device),
            requires_grad=learnable_prototypes 
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / max(init_tau, 1e-6), dtype=torch.float32, device=device),
            requires_grad=learnable_tau
        )

        print(f"[PrototypeScorer] 初始化完成:")
        print(f"  PCA: 维度{clip_emb_dim}→{self.pca_dim}, 可学习={learnable_pca}")
        print(f"  原型: {self.num_prototypes}个, 维度{self.pca_dim}, 可学习={learnable_prototypes}")  # 新增：显示开关状态
        print(f"  温度: 初始值{1/self.logit_scale.item():.6f}, 可学习={learnable_tau}")

    @property
    def tau(self):
        return 1.0 / self.logit_scale.clamp(min=1e-6)

    def pca_transform(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.pca_mean) / torch.abs(self.pca_scale)
        x_pca = torch.matmul(x_norm, self.pca_components.T)
        return x_pca

    def forward(self, clip_feats: torch.Tensor) -> torch.Tensor:

        x_pca = self.pca_transform(clip_feats)  # [B, pca_dim]

        x_norm = F.normalize(x_pca, dim=-1)  # [B, pca_dim]
        proto_norm = F.normalize(self.prototypes, dim=-1)  # [K, pca_dim]
        sim = torch.matmul(x_norm, proto_norm.T)  # [B, K]

        weights = F.softmax(sim / self.tau, dim=-1)  # [B, K]

        pred_scores = torch.matmul(weights, self.proto_scores)  # [B]
        return pred_scores


class CLIP_IQA_Proto(nn.Module):
    def __init__(self,
                 clip_model_name: str,
                 device: torch.device,
                 pca_path: str = None,
                 basis_path: str = None,
                 num_prototypes: int = 32,
                 tau: float = 0.07,
                 learnable_tau: bool = False,
                 learnable_pca: bool = False,
                 learnable_prototypes: bool = True,
                 prec: str = "fp32",
                 patch_alpha: float = 0.0,
                 patch_mode: str = "mean",
                 patch_mean_alpha: float = 0.0,
                 patch_max_alpha: float = 0.0,
                 patch_p90_alpha: float = 0.0,
                 tta_flip: bool = False,
                 sat_factor: float = 1.0):
        super().__init__()
        self.device = device
        self.prec = prec
        self.patch_alpha = patch_alpha  # weight for single-pool patch predictions (legacy)
        self.patch_mode = patch_mode    # "mean" or "max" pooling of spatial patches (legacy)
        # N-way blend
        self.patch_mean_alpha = patch_mean_alpha  # weight for mean pooling predictions
        self.patch_max_alpha = patch_max_alpha    # weight for max pooling predictions
        self.patch_p90_alpha = patch_p90_alpha    # weight for 90th-percentile pooling predictions
        self.tta_flip = tta_flip  # horizontal flip TTA
        self.sat_factor = sat_factor  # color saturation enhancement factor (1.0 = no change)

        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()

        with torch.no_grad():
            dummy = Image.new("RGB", (224, 224))
            pixel = self.processor(images=[dummy], return_tensors="pt").to(device)
            clip_feats = self.clip.get_image_features(** pixel)  # [1, D]
            self.clip_emb_dim = clip_feats.shape[-1]

        self.scorer = PrototypeScorer(
            clip_emb_dim=self.clip_emb_dim,
            pca_path=pca_path,
            basis_path=basis_path,
            num_prototypes=num_prototypes,
            init_tau=tau,
            learnable_tau=learnable_tau,
            learnable_pca=learnable_pca,
            learnable_prototypes=learnable_prototypes,  
            device=device
        )

    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:

        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.clip.get_image_features(**inputs)  # [B, clip_emb_dim]
        return feats

    def encode_images_with_all_patches(self, pil_images: List[Image.Image]):
        """Returns (cls_feats, patch_mean_feats, patch_max_feats, patch_p90_feats) all in [B, clip_emb_dim]."""
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            cls_feats = self.clip.get_image_features(**inputs)  # [B, clip_emb_dim]
            vision_out = self.clip.vision_model(**inputs, output_hidden_states=False)
            hidden = vision_out[0]  # [B, S+1, D_vision]
            B, S, D = hidden.shape
            all_proj = self.clip.visual_projection(hidden.reshape(-1, D)).reshape(B, S, self.clip_emb_dim)
            patches = all_proj[:, 1:, :]  # [B, S-1, clip_emb_dim]
            patch_mean = patches.mean(dim=1)  # [B, clip_emb_dim]
            patch_max = patches.max(dim=1)[0]  # [B, clip_emb_dim]
            patch_p90 = torch.quantile(patches, 0.90, dim=1)  # [B, clip_emb_dim]
        return cls_feats, patch_mean, patch_max, patch_p90

    def forward(self, pil_images: List[Image.Image]) -> torch.Tensor:

        use_3way = (self.patch_mean_alpha > 0.0 or self.patch_max_alpha > 0.0 or self.patch_p90_alpha > 0.0)
        use_2way = (self.patch_alpha > 0.0)

        if use_3way:
            if self.sat_factor != 1.0:
                from PIL import ImageEnhance
                pil_images = [ImageEnhance.Color(img).enhance(self.sat_factor) for img in pil_images]
            cls_feats, pm_feats, px_feats, p90_feats = self.encode_images_with_all_patches(pil_images)
            cls_w = 1.0 - self.patch_mean_alpha - self.patch_max_alpha - self.patch_p90_alpha
            if self.prec == "amp":
                with torch.cuda.amp.autocast():
                    pred_cls = self.scorer(cls_feats)
                    pred_pm = self.scorer(pm_feats)
                    pred_px = self.scorer(px_feats)
                    pred_p90 = self.scorer(p90_feats)
            else:
                pred_cls = self.scorer(cls_feats)
                pred_pm = self.scorer(pm_feats)
                pred_px = self.scorer(px_feats)
                pred_p90 = self.scorer(p90_feats)
            pred_scores = cls_w * pred_cls + self.patch_mean_alpha * pred_pm + self.patch_max_alpha * pred_px + self.patch_p90_alpha * pred_p90
            if self.tta_flip:
                from PIL import ImageOps
                pil_flip = [ImageOps.mirror(img) for img in pil_images]  # note: pil_images already sat-enhanced if sat_factor!=1
                cls_f2, pm_f2, px_f2, p90_f2 = self.encode_images_with_all_patches(pil_flip)
                pred_cls2 = self.scorer(cls_f2)
                pred_pm2  = self.scorer(pm_f2)
                pred_px2  = self.scorer(px_f2)
                pred_p90_2 = self.scorer(p90_f2)
                pred_flip = cls_w * pred_cls2 + self.patch_mean_alpha * pred_pm2 + self.patch_max_alpha * pred_px2 + self.patch_p90_alpha * pred_p90_2
                pred_scores = (pred_scores + pred_flip) / 2.0
        elif use_2way:
            # Legacy 2-way blend
            cls_feats, pm_feats, px_feats = self.encode_images_with_all_patches(pil_images)
            if self.patch_mode == "max":
                patch_feats = px_feats
            else:
                patch_feats = pm_feats
            if self.prec == "amp":
                with torch.cuda.amp.autocast():
                    pred_cls = self.scorer(cls_feats)
                    pred_patch = self.scorer(patch_feats)
            else:
                pred_cls = self.scorer(cls_feats)
                pred_patch = self.scorer(patch_feats)
            pred_scores = (1.0 - self.patch_alpha) * pred_cls + self.patch_alpha * pred_patch
        else:
            clip_feats = self.encode_images(pil_images)  # [B, clip_emb_dim]
            if self.prec == "amp":
                with torch.cuda.amp.autocast():
                    pred_scores = self.scorer(clip_feats)  # [B]
            else:
                pred_scores = self.scorer(clip_feats)

        return pred_scores

@torch.no_grad()
def evaluate(model: CLIP_IQA_Proto, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []
    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        pil_images = batch["images"]
        y = batch["scores"].to(device)
        y_hat = model(pil_images)                          # [B]
        preds.append(y_hat.detach().cpu().float().numpy())
        gts.append(y.detach().cpu().float().numpy())
    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    metrics = compute_metrics(pred, gt)
    return metrics

@torch.no_grad()
def predict_all_images(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    results = []

    for batch in tqdm(loader, desc="Predict", leave=True):
        imgs = batch["images"]
        meta = batch["meta"]

        preds = model(imgs).cpu().numpy()

        for m, score in zip(meta, preds):
            results.append({
                "image": m["rel_image"],
                "score": float(score)
            })

    return results


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_json", type=str, nargs="+", required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--pca_path", type=str, default=None)
    ap.add_argument("--basis_path", type=str, default=None)
    ap.add_argument("--num_prototypes", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--learnable_tau", type=bool, default=False)
    ap.add_argument("--learnable_pca", type=bool, default=False)
    ap.add_argument("--learnable_prototypes", type=bool, default=False)
    ap.add_argument("--prec", type=str, default="fp32")

    ap.add_argument("--out_json", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patch_alpha", type=float, default=0.0,
                    help="Weight for patch predictions (0=CLS only, e.g. 0.78=22%CLS+78%patch)")
    ap.add_argument("--patch_mode", type=str, default="mean",
                    choices=["mean", "max"],
                    help="Spatial patch pooling mode: 'mean' (default) or 'max'")
    ap.add_argument("--patch_mean_alpha", type=float, default=0.0,
                    help="3-way blend: weight for patch-mean predictions (cls=1-mean-max)")
    ap.add_argument("--patch_max_alpha", type=float, default=0.0,
                    help="3-way blend: weight for patch-max predictions (cls=1-mean-max)")
    ap.add_argument("--patch_p90_alpha", type=float, default=0.0,
                    help="N-way blend: weight for 90th-percentile patch pooling predictions")
    ap.add_argument("--tta_flip", type=str2bool, default=False,
                    help="Horizontal flip TTA: average predictions of orig + flipped image")
    ap.add_argument("--override_tau", type=float, default=None,
                    help="Override checkpoint tau (logit_scale) at inference. None=use checkpoint value.")
    ap.add_argument("--sat_factor", type=float, default=1.0,
                    help="Color saturation enhancement factor for TTA (1.0=no change, 1.25=boost saturation 25%%)")

    return ap.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device = {device}")

    test_set = IQADataset(args.test_json, args.image_root)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=iqad_collate
    )

    model = CLIP_IQA_Proto(
        clip_model_name=args.clip_model,
        device=device,
        pca_path=args.pca_path,
        basis_path=args.basis_path,
        num_prototypes=args.num_prototypes,
        tau=args.tau,
        learnable_tau=args.learnable_tau,
        learnable_pca=args.learnable_pca,
        learnable_prototypes=args.learnable_prototypes,
        prec=args.prec,
        patch_alpha=args.patch_alpha,
        patch_mode=args.patch_mode,
        patch_mean_alpha=args.patch_mean_alpha,
        patch_max_alpha=args.patch_max_alpha,
        patch_p90_alpha=args.patch_p90_alpha,
        tta_flip=args.tta_flip,
        sat_factor=args.sat_factor
    ).to(device)

    print(f"Loading checkpoint: {args.ckpt}")
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=True)
    print("Checkpoint loaded.\n")

    if args.override_tau is not None:
        new_scale = 1.0 / max(args.override_tau, 1e-6)
        model.scorer.logit_scale.data.fill_(new_scale)
        print(f"[override_tau] tau overridden to {args.override_tau:.4f} (logit_scale={new_scale:.4f})")

    results = predict_all_images(model, test_loader, device)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(results, open(args.out_json, "w"), indent=2)
    print(f"Saved prediction JSON → {args.out_json}\n")

    print("Evaluating final metrics (MAE / PLCC / SRCC)...")
    metrics = evaluate(model, test_loader, device)

    print("\n================== Final Metrics ==================")
    print(f"MAE  = {metrics['MAE']:.4f}")
    print(f"PLCC = {metrics['PLCC']:.4f}")
    print(f"SRCC = {metrics['SRCC']:.4f}")
    print("===================================================\n")


if __name__ == "__main__":
    main()