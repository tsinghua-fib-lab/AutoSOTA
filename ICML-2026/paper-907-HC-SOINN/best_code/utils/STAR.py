"""Core component."""

import numpy as np
import logging
from typing import Dict, Set, Optional, Callable, Any
from torch.utils.data import DataLoader
import torch


def _l2_normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        if n < eps:
            return v.astype(np.float32, copy=True)
        return (v / n).astype(np.float32, copy=True)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / (n + eps)).astype(np.float32, copy=False)


class STARAligner:
    """Handle init."""

    def __init__(
        self,
        hc_soinn: Any,  # HCSOINNClassifier
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        use_full_task_rehearsal: bool = False,
        star_lambda: float = 0.3,
    ):
        self.hc_soinn = hc_soinn
        self.feature_extractor = feature_extractor
        self.device = device
        self.use_full_task_rehearsal = use_full_task_rehearsal
        self.star_lambda = float(star_lambda)

        self.anchor_store: Dict[int, Dict[str, Any]] = {}

        if self.star_lambda <= 0.0 or self.star_lambda > 1.0:
            logging.warning(f"[STAR] star_lambda={self.star_lambda} out of (0,1], clamping.")
            self.star_lambda = float(min(max(self.star_lambda, 1e-3), 1.0))

    def select_anchors_for_current_task(
        self,
        dataset,
        batch_size: int = 128,
        num_workers: int = 4,
        current_task_classes: Optional[Set[int]] = None,
    ) -> None:
        """Handle select anchors for current task."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        all_feats = []
        all_imgs = []
        all_targets = []

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self.device)
                try:
                    feats = self.feature_extractor(inputs, class_id=targets)
                except TypeError:
                    feats = self.feature_extractor(inputs)
                if isinstance(feats, tuple):
                    feats = feats[0]

                all_feats.append(feats.detach().cpu().numpy())
                all_imgs.append(inputs.cpu())
                all_targets.append(targets.numpy())

        if not all_feats:
            return

        all_feats = np.concatenate(all_feats, axis=0)
        all_imgs = torch.cat(all_imgs, dim=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if current_task_classes is None:
            current_task_classes = set(np.unique(all_targets))

        for cls in current_task_classes:
            if cls in self.anchor_store:
                continue

            mask = (all_targets == cls)
            cls_feats = all_feats[mask]
            cls_imgs = all_imgs[mask]

            if len(cls_feats) == 0:
                continue

            clusters = self.hc_soinn.class_clusters.get(cls, [])
            if len(clusters) == 0:
                sel_feats = cls_feats[: min(20, cls_feats.shape[0])]
                sel_imgs = cls_imgs[: sel_feats.shape[0]]
                self.anchor_store[cls] = {
                    "images": sel_imgs,
                    "feats_ref": sel_feats.copy(),
                    "mode": "trajectory_fallback",
                }
                continue

            targets_arr = np.stack(
                [(c.center_raw if c.center_raw is not None else c.center) for c in clusters],
                axis=0,
            ).astype(np.float32, copy=False)
            t_norm = _l2_normalize_np(targets_arr)
            f_norm = _l2_normalize_np(cls_feats)
            sims = np.dot(t_norm, f_norm.T)

            used = set()
            chosen = []
            for i in range(sims.shape[0]):
                order = np.argsort(-sims[i])
                pick = int(order[0])
                if pick in used:
                    for j in order[1:]:
                        jj = int(j)
                        if jj not in used:
                            pick = jj
                            break
                used.add(pick)
                chosen.append(pick)

            sel_feats = cls_feats[chosen]
            sel_imgs = cls_imgs[chosen]

            centers_raw_ref = targets_arr.copy()
            self.anchor_store[cls] = {
                "images": sel_imgs,
                "feats_ref": sel_feats.copy(),
                "mode": "trajectory",
                "centers_raw_ref": centers_raw_ref,
                "ema_delta": np.zeros_like(sel_feats, dtype=np.float32),
            }

    def _apply_mean_drift_to_class(self, cls: int, mean_d: np.ndarray) -> None:
        """Handle apply mean drift to class."""
        if cls not in self.hc_soinn.class_clusters:
            return
        clusters = self.hc_soinn.class_clusters[cls]
        mean_d = np.asarray(mean_d, dtype=np.float32).reshape(-1)
        for c in clusters:
            if c.center_raw is not None:
                new_raw = c.center_raw + mean_d
            else:
                new_raw = c.center + mean_d
            if np.linalg.norm(new_raw) > 1e-9:
                c.center_raw = new_raw.astype(np.float32, copy=False)
                c.center = _l2_normalize_np(c.center_raw)
        if cls in getattr(self.hc_soinn, "class_mu_raw", {}):
            new_mu_raw = self.hc_soinn.class_mu_raw[cls] + mean_d
            if np.linalg.norm(new_mu_raw) > 1e-9:
                self.hc_soinn.class_mu_raw[cls] = new_mu_raw.astype(np.float32, copy=False)
                self.hc_soinn.class_mu[cls] = _l2_normalize_np(self.hc_soinn.class_mu_raw[cls])
        if hasattr(self.hc_soinn, "invalidate_cache"):
            self.hc_soinn.invalidate_cache()

    def align_old_classes(
        self,
        cur_task: int,
        current_task_classes: Optional[Set[int]] = None,
    ) -> None:
        """Handle align old classes."""
        if current_task_classes is None:
            current_task_classes = set()

        for cls, data in self.anchor_store.items():
            if cls in current_task_classes:
                continue
            if cls not in self.hc_soinn.class_clusters:
                continue

            try:
                imgs = data["images"].to(self.device)
                feats_old = data["feats_ref"]

                with torch.no_grad():
                    try:
                        feats_new = self.feature_extractor(imgs, class_id=cls)
                    except TypeError:
                        feats_new = self.feature_extractor(imgs)

                    if isinstance(feats_new, tuple):
                        feats_new = feats_new[0]
                    feats_new = feats_new.detach().cpu().numpy()

                mode = str(data.get("mode", ""))
                if not mode.startswith("trajectory"):
                    fo = np.asarray(feats_old, dtype=np.float32)
                    fn = np.asarray(feats_new, dtype=np.float32)
                    if fo.shape == fn.shape:
                        mean_d = np.mean(fn - fo, axis=0)
                    else:
                        mean_d = np.mean(fn, axis=0) - np.mean(fo, axis=0)
                    self._apply_mean_drift_to_class(cls, mean_d)
                    self.anchor_store[cls]["feats_ref"] = feats_new.copy()
                    self.anchor_store[cls]["mode"] = "trajectory_legacy"
                    continue

                clusters = self.hc_soinn.class_clusters.get(cls, [])
                if len(clusters) == 0:
                    continue

                feats_old = np.asarray(feats_old, dtype=np.float32)
                feats_new = np.asarray(feats_new, dtype=np.float32)
                delta = feats_new - feats_old

                ema = data.get("ema_delta", None)
                if ema is None or np.asarray(ema).shape != delta.shape:
                    ema = np.zeros_like(delta, dtype=np.float32)
                lam = self.star_lambda
                ema = (1.0 - lam) * ema + lam * delta
                self.anchor_store[cls]["ema_delta"] = ema

                centers_raw_ref = data.get("centers_raw_ref", None)
                if centers_raw_ref is None:
                    match = list(range(min(len(clusters), ema.shape[0])))
                    ref_indices = list(range(len(match)))
                else:
                    centers_raw_ref = np.asarray(centers_raw_ref, dtype=np.float32)
                    cur_centers_raw = np.stack(
                        [
                            (c.center_raw if c.center_raw is not None else c.center)
                            for c in clusters
                        ],
                        axis=0,
                    ).astype(np.float32, copy=False)
                    ref_n = _l2_normalize_np(centers_raw_ref)
                    cur_n = _l2_normalize_np(cur_centers_raw)
                    sim = np.dot(ref_n, cur_n.T)
                    used = set()
                    match = []
                    ref_indices = []
                    for i in range(sim.shape[0]):
                        j = int(np.argmax(sim[i]))
                        if j in used:
                            order = np.argsort(-sim[i])
                            for cand in order:
                                jj = int(cand)
                                if jj not in used:
                                    j = jj
                                    break
                        used.add(j)
                        match.append(j)
                        ref_indices.append(i)

                applied = []
                for i_ref, j_cur in zip(ref_indices, match):
                    if j_cur >= len(clusters) or i_ref >= ema.shape[0]:
                        continue
                    d = ema[i_ref]
                    c = clusters[j_cur]
                    base = c.center_raw if c.center_raw is not None else c.center
                    new_raw = np.asarray(base, dtype=np.float32) + d
                    if np.linalg.norm(new_raw) > 1e-9:
                        c.center_raw = new_raw.astype(np.float32, copy=False)
                        c.center = _l2_normalize_np(c.center_raw)
                        applied.append(d)

                if cls in getattr(self.hc_soinn, "class_mu_raw", {}):
                    if len(applied) > 0:
                        mean_d = np.mean(np.stack(applied, axis=0), axis=0)
                    else:
                        mean_d = np.mean(ema, axis=0)
                    new_mu_raw = self.hc_soinn.class_mu_raw[cls] + mean_d
                    if np.linalg.norm(new_mu_raw) > 1e-9:
                        self.hc_soinn.class_mu_raw[cls] = new_mu_raw.astype(np.float32, copy=False)
                        self.hc_soinn.class_mu[cls] = _l2_normalize_np(self.hc_soinn.class_mu_raw[cls])

                self.anchor_store[cls]["feats_ref"] = feats_new.copy()
                self.anchor_store[cls]["centers_raw_ref"] = np.stack(
                    [(c.center_raw if c.center_raw is not None else c.center) for c in clusters],
                    axis=0,
                ).astype(np.float32, copy=False)

                if hasattr(self.hc_soinn, "invalidate_cache"):
                    self.hc_soinn.invalidate_cache()

            except Exception as e:
                logging.error(f"[STAR] Align failed for class {cls}: {e}")
