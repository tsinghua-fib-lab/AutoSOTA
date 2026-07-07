import math
import os

import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from openood.networks.clip_fixed_ood_prompt import (
    get_class_names,
    get_templates,
    get_text_features_neg,
)


# ---------------------------------------------------------------------------
# Text feature helpers
# ---------------------------------------------------------------------------

def get_id_text_features(model, cfg) -> torch.Tensor:
    """
    Generate L2-normalised CLIP text prototype features for all ID classes.

    For each class name in the configured ID dataset, every prompt template is
    encoded, the resulting embeddings are averaged and re-normalised, producing
    one unit vector per class. The output is a [D, num_id] matrix where each
    column is one class prototype.

    Parameters
    ----------
    model : FixedCLIP instance (frozen CLIP backbone).
    cfg   : Config object with text_prompt set.
    """
    print(f"Generating ID text features (prompt style: '{cfg.text_prompt}')...")
    templates = get_templates(cfg.text_prompt)
    device    = next(model.parameters()).device
    classnames = get_class_names(cfg.id_dataset_name)

    with torch.no_grad():
        cols = []
        for classname in tqdm(classnames, desc='ID text features'):
            texts = clip.tokenize(
                [t.format(classname) for t in templates]
            ).to(device)
            emb = model.encode_text(texts)
            emb = F.normalize(emb, dim=-1).mean(dim=0)
            cols.append(F.normalize(emb.unsqueeze(0), dim=-1).squeeze(0))

    return torch.stack(cols, dim=1)   # [D, num_id]


def get_ood_text_features(model, cfg) -> torch.Tensor:
    """
    Generate L2-normalised CLIP text prototype features for OOD negative labels.

    Delegates to get_text_features_neg(), which either loads a pre-computed
    cache from disk (fast path) or runs the full WordNet cosine-similarity
    selection pipeline and saves the result for future calls.

    The returned tensor is sliced to keep only the OOD columns (the first
    num_id columns belonging to ID classes are discarded here because
    get_id_text_features() already handles those separately).

    Parameters
    ----------
    model : FixedCLIP instance.
    cfg   : Config object with id_dataset_name, text_prompt,
            num_negative_labels, and optionally use_ood_labels.

    Returns
    -------
    Tensor of shape [D, num_negative_labels].
    """
    print(f"Generating OOD text features (N={cfg.num_negative_labels})...")
    feat, _ = get_text_features_neg(
        model,
        cfg.id_dataset_name,
        cfg.text_prompt,
        text_center=True,
        num_negative_labels=cfg.num_negative_labels,
        use_ood_labels=getattr(cfg, 'use_ood_labels', False),
        cache_root=os.path.join(cfg.data_root, 'txtfiles_output'),
    )
    return feat[:, cfg.dataset_num_classes:]   # [D, num_negative_labels]


# ---------------------------------------------------------------------------
# Streaming Prototype Adapter
# ---------------------------------------------------------------------------

class StreamingPrototypeAdapter:
    """
    Streaming OOD adapter that updates visual class prototypes from an
    unlabeled test-time stream.

    Architecture
    ------------
    Two prototype matrices are maintained:
      text_prototypes   -- Frozen CLIP text prototypes, shape [D, num_id+num_ood].
                           They provide the initial semantic reference.
      visual_prototypes -- Stream-adapted visual prototypes, same shape.
                           ID and OOD slices are updated independently so the
                           detector can refine both sides of the score.

    Routing strategy
    ----------------
    A binary routing confidence is derived for each sample and passed through
    two threshold gates (thresh_id, thresh_ood). Samples that clear thresh_id
    trigger an ID prototype update; samples below thresh_ood trigger an OOD
    prototype update; samples in the uncertain region are skipped to avoid
    noisy adaptation.

    Blend schedule
    --------------
    blend_factor * sqrt(progress) controls how much the detector trusts the
    stream-adapted visual prototypes over the frozen text prototypes. ID and
    OOD progress are tracked separately so an imbalanced stream does not let
    one side dominate the other.

    Usage
    -----
    Call reset() before each new OOD dataset evaluation to reinitialise the
    visual prototypes from the text prototypes and zero the counters. Then
    call process_batch() for each batch in the combined loader.
    """

    def __init__(self, cfg, id_text_feat: torch.Tensor, ood_text_feat: torch.Tensor):
        self.cfg     = cfg
        self.device  = cfg.device
        self.num_id  = cfg.dataset_num_classes
        self.num_ood = cfg.num_negative_labels

        # Concatenate ID [D, num_id] and OOD [D, num_ood] text prototypes into
        # a single unified [D, num_id+num_ood] reference matrix.
        self.text_prototypes = torch.cat(
            [id_text_feat, ood_text_feat], dim=1
        ).float().detach()

        # Visual prototypes start as an exact copy of the text prototypes.
        self.visual_prototypes = self.text_prototypes.clone().detach()

        # Optional per-class additive log-probability bias. With bias_lr=0 it
        # stays at zero and has no effect.
        self.class_bias = torch.zeros(self.text_prototypes.size(1)).to(self.device)

        self.processed_samples = 0
        self.id_steps          = 0
        self.ood_steps         = 0

    def reset(self, total_samples: int) -> None:
        """
        Reinitialise all running state before a new OOD dataset evaluation.

        This resets the visual prototypes back to the text prototypes so each OOD
        dataset gets a clean evaluation from the same starting point. It also
        updates cfg.total_samples to match the actual dataset size so the
        blend progress normalisation is correct for datasets of different sizes.
        """
        self.processed_samples  = 0
        self.id_steps          = 0
        self.ood_steps         = 0
        self.cfg.total_samples  = total_samples
        self.visual_prototypes = self.text_prototypes.clone().detach()
        self.class_bias        = torch.zeros(self.text_prototypes.size(1)).to(self.device)

    @torch.no_grad()
    def process_batch(
        self,
        net,
        images: torch.Tensor,
        routes,
    ):
        """
        Encode one batch of images, route each sample, update prototypes, and
        return per-sample OOD detection scores.

        The loop over individual samples within the batch is intentional:
        the routing decision and prototype update for sample i alter the
        running counters (id_steps, ood_steps), which changes the blend
        schedule for sample i+1. Vectorising would change the stream semantics.

        Parameters
        ----------
        net    : FixedCLIP instance used to encode images.
        images : Float tensor of shape [B, C, H, W] on the correct device.

        Returns
        -------
        preds        : LongTensor [B] -- argmax ID class prediction.
        final_scores : FloatTensor [B] -- OOD score (higher = more ID-like).
        """
        feats = net.encode_image(images).float()   # [B, D]
        raw_dots = []

        for i in range(feats.size(0)):
            x = feats[i]
            route = routes[i]
            self.processed_samples += 1

            # --- Text-reference distribution ---
            text_probs_raw = F.softmax(x @ self.text_prototypes / self.cfg.text_temperature, dim=0)
            # Optional bias correction (no-op when bias_lr=0).
            text_probs = text_probs_raw * torch.exp(self.class_bias)
            text_probs = text_probs / (text_probs.sum() + 1e-8)

            # --- Visual-prototype distribution ---
            visual_probs = F.softmax(x @ self.visual_prototypes / self.cfg.visual_temperature, dim=0)

            target_is_id  = route.is_id
            target_is_ood = route.is_ood

            # --- Interpolated scoring prototypes ---
            # ID and OOD progress are tracked separately so that an imbalanced
            # stream does not favour one side over the other.
            prog_id  = min(1.0, self.id_steps  / self.cfg.total_samples)
            prog_ood = min(1.0, self.ood_steps / self.cfg.total_samples)
            blend_id  = self.cfg.blend_factor * math.sqrt(prog_id)
            blend_ood = self.cfg.blend_factor * math.sqrt(prog_ood)

            # Start from the frozen text prototypes; only update the slice that
            # corresponds to this sample's identity (decoupled scoring).
            scoring_prototypes = self.text_prototypes.clone()
            if target_is_id:
                # ID slice: blend visual prototypes with text prototypes.
                part = (  blend_id  * self.visual_prototypes[:, :self.num_id]
                        + (1 - blend_id)  * self.text_prototypes[:, :self.num_id])
                scoring_prototypes[:, :self.num_id] = part
            else:
                # OOD slot: mirror of the ID case.
                part = (  blend_ood * self.visual_prototypes[:, self.num_id:]
                        + (1 - blend_ood) * self.text_prototypes[:, self.num_id:])
                scoring_prototypes[:, self.num_id:] = part

            scoring_prototypes = F.normalize(scoring_prototypes, dim=0)
            raw_dots.append((x @ scoring_prototypes).unsqueeze(0))

            # --- Immediate prototype update (no experience replay buffer) ---
            if target_is_id:
                self._update_id(x, text_probs, visual_probs)
            elif target_is_ood:
                self._update_ood(x, text_probs, visual_probs)

            # Re-normalise prototype columns after each update to keep them
            # on the unit sphere, consistent with the L2-normalised image feats.
            if target_is_id or target_is_ood:
                self.visual_prototypes = F.normalize(self.visual_prototypes, dim=0)

        logits = torch.cat(raw_dots, dim=0)   # [B, num_id+num_ood]
        return self._compute_scores(logits)

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_id(self, x, text_probs, visual_probs) -> None:
        """
        Gradient-descent update on the ID slice of visual_prototypes.

        The gradient is the outer product of the image feature x and the
        difference between the visual and text-reference conditional
        distributions. This nudges the visual prototypes toward the current
        text-guided target distribution.

        The learning rate decays as prototype_lr / sqrt(id_steps), giving fast
        adaptation early in the stream and conservative updates later.
        """
        self.id_steps += 1
        bias_lr = self.cfg.bias_lr / math.sqrt(self.id_steps)
        proto_lr = self.cfg.prototype_lr / math.sqrt(self.id_steps)

        # Renormalise to the ID slice so the gradient targets the conditional
        # distribution P(class | ID) rather than the full joint.
        text_slice = text_probs[:self.num_id] / (text_probs[:self.num_id].sum() + 1e-8)
        visual_slice = visual_probs[:self.num_id]   / (visual_probs[:self.num_id].sum()   + 1e-8)

        # Optional bias update toward bias_target / num_id (no-op when bias_lr=0).
        self.class_bias[:self.num_id] -= bias_lr * (text_slice - self.cfg.bias_target / self.num_id)
        self.class_bias[:self.num_id].clamp_(min=0)

        self.visual_prototypes[:, :self.num_id] -= (
            proto_lr / self.cfg.visual_temperature
        ) * torch.outer(x, visual_slice - text_slice)

    def _update_ood(self, x, text_probs, visual_probs) -> None:
        """
        Gradient-descent update on the OOD column slice of visual_prototypes.

        Structurally identical to _update_id, but operates on the OOD slice
        [num_id:] and uses ood_steps for the learning rate schedule.

        Note: bias_lr=0 in the default config, so the class bias remains fixed
        unless explicitly enabled.
        """
        self.ood_steps += 1
        bias_lr = self.cfg.bias_lr / math.sqrt(self.ood_steps)
        proto_lr = self.cfg.prototype_lr / math.sqrt(self.ood_steps)

        text_slice = text_probs[self.num_id:] / (text_probs[self.num_id:].sum() + 1e-8)
        visual_slice = visual_probs[self.num_id:]   / (visual_probs[self.num_id:].sum()   + 1e-8)

        self.class_bias[self.num_id:] -= bias_lr * (text_slice - self.cfg.bias_target / self.num_ood)
        self.class_bias[self.num_id:].clamp_(min=0)

        self.visual_prototypes[:, self.num_id:] -= (
            proto_lr / self.cfg.visual_temperature
        ) * torch.outer(x, visual_slice - text_slice)

    def _compute_scores(self, logits: torch.Tensor):
        """
        Compute per-sample OOD detection scores from raw dot-product logits.

        The OOD logit vector is split into score_groups equal chunks. For each
        chunk, softmax is applied over [ID logits, chunk] and the probability
        mass on ID classes is summed. The final score is the mean over chunks.

        Higher score means the sample is more likely to be in-distribution.

        Returns
        -------
        preds        : LongTensor [B] -- argmax predicted ID class.
        final_scores : FloatTensor [B] -- per-sample OOD detection score.
        """
        logits_id  = logits[:, :self.num_id]
        logits_ood = logits[:, self.num_id:]

        scaled_id  = logits_id  / getattr(self.cfg, "id_temperature", 0.01)
        scaled_ood = logits_ood / self.cfg.negative_temperature

        # Trim OOD dimension to be divisible by score_groups.
        drop = scaled_ood.size(1) % self.cfg.score_groups
        if drop:
            scaled_ood = scaled_ood[:, :-drop]

        group_scores = []
        for chunk in scaled_ood.chunk(self.cfg.score_groups, dim=1):
            cat   = torch.cat([scaled_id, chunk], dim=1)
            score = F.softmax(cat, dim=1)[:, :self.num_id].sum(dim=1)
            group_scores.append(score.unsqueeze(1))

        final_scores = torch.cat(group_scores, dim=1).mean(dim=1)
        preds        = torch.argmax(scaled_id, dim=1)
        return preds, final_scores
