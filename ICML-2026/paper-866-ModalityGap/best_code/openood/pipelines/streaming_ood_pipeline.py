from dataclasses import dataclass
import csv
import json
import os

import torch
from tqdm import tqdm

from openood.datasets.imglist_dataset import build_combined_loader
from openood.evaluators.ood_evaluator import OODEvaluator
from openood.networks.fixed_clip import FixedCLIP
from openood.postprocessors.streaming_prototype_adapter import (
    StreamingPrototypeAdapter,
    get_id_text_features,
    get_ood_text_features,
)
from openood.utils.config import set_random_seed


@dataclass(frozen=True)
class Route:
    is_id: bool
    is_ood: bool


def resolve_routes(labels: torch.Tensor, thresh_id: float, thresh_ood: float):
    scores = torch.where(
        labels.eq(-1),
        torch.zeros_like(labels, dtype=torch.float32),
        torch.ones_like(labels, dtype=torch.float32),
    )
    return [
        Route(
            is_id=score >= thresh_id,
            is_ood=score < thresh_ood,
        )
        for score in scores.detach().cpu().tolist()
    ]


class StreamingOODPipeline:
    """
    End-to-end pipeline for streaming prototype OOD detection.

    Execution order
    ---------------
    1. Fix random seeds.
    2. Load the frozen CLIP backbone.
    3. Pre-compute ID and OOD text prototype features (or load from cache).
    4. For each OOD evaluation dataset:
       a. Build a shuffled combined ID+OOD dataloader.
       b. Reset the StreamingPrototypeAdapter (reinitialise prototypes, zero counters).
       c. Stream all samples through the adapter:
            encode image -> route sample -> update prototypes -> score.
       d. Separate collected scores by label (ID vs OOD).
       e. Compute AUROC and FPR95.
    5. Print a formatted summary table.

    Parameters
    ----------
    cfg : Config object built from the YAML config file. Expected attributes:
          seed, backbone, device, data_root, id_imglist, ood_datasets (dict),
          batch_size, num_workers, id_dataset_name, text_prompt,
          num_negative_labels, use_ood_labels, dataset_num_classes,
          total_samples, prototype_lr, bias_lr, bias_target, blend_factor,
          text_temperature, visual_temperature, thresh_id, thresh_ood,
          negative_temperature, score_groups.
    """

    def __init__(self, cfg):
        self.cfg       = cfg
        self.evaluator = OODEvaluator()

    def run(self) -> None:
        set_random_seed(self.cfg.seed)

        print(f"Streaming Prototype OOD Pipeline")
        print(f"  backbone : {self.cfg.backbone}")
        print(f"  device   : {self.cfg.device}")
        print(f"  seed     : {self.cfg.seed}")

        # Load the frozen CLIP backbone once; it is reused across all datasets.
        clip_model = FixedCLIP(self.cfg.backbone, self.cfg.device)

        # Pre-compute text features. get_ood_text_features checks for a cached
        # .pth file in txtfiles_output/ before running the expensive WordNet
        # cosine-similarity pipeline, so subsequent runs are fast.
        id_text_feat  = get_id_text_features(clip_model, self.cfg)
        ood_text_feat = get_ood_text_features(clip_model, self.cfg)

        # Build the adapter once; reset() is called before each OOD dataset
        # so each evaluation starts from the same text prototype initialisation.
        adapter = StreamingPrototypeAdapter(self.cfg, id_text_feat, ood_text_feat)

        print('\n' + '=' * 70)
        print('Mixed ID+OOD Streaming Evaluation')
        print('=' * 70)

        metrics = {}
        for ood_name, ood_imglist in self.cfg.ood_datasets.items():
            print(f"\n[{ood_name}] Building combined dataloader...")
            loader = build_combined_loader(
                self.cfg,
                self.cfg.id_imglist,
                ood_imglist,
                clip_model.preprocess,
            )

            id_scores, ood_scores = self._run_one_dataset(
                adapter, clip_model, loader, ood_name
            )
            auroc, fpr95 = self.evaluator.compute(id_scores, ood_scores)

            metrics[ood_name] = {'AUROC': auroc, 'FPR95': fpr95}
            print(f"  --> AUROC: {auroc * 100:.2f}%  |  FPR95: {fpr95 * 100:.2f}%")

        self.evaluator.print_results(metrics)
        self._save_results(metrics)

    def _run_one_dataset(self, adapter, clip_model, loader, ood_name: str):
        """
        Stream the combined loader through the adapter for one OOD dataset and
        collect per-sample scores separated into ID and OOD arrays.

        The adapter is reset at the start so the visual prototypes are
        reinitialised from the text prototypes and all counters are zeroed.

        Parameters
        ----------
        adapter    : StreamingPrototypeAdapter instance (will be reset in place).
        clip_model : FixedCLIP instance for image encoding.
        loader     : Combined ID+OOD DataLoader.
        ood_name   : Display name used in the progress bar.

        Returns
        -------
        id_scores  : 1-D numpy array of detection scores for ID samples.
        ood_scores : 1-D numpy array of detection scores for OOD samples.
        """
        adapter.reset(total_samples=len(loader.dataset))
        print(f"  Running inference (streaming update, no replay buffer)...")

        all_scores, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=ood_name, leave=False):
                images = images.to(self.cfg.device)
                routes = resolve_routes(labels, self.cfg.thresh_id, self.cfg.thresh_ood)
                _, scores = adapter.process_batch(clip_model, images, routes)
                all_scores.append(scores.cpu())
                all_labels.append(labels)

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()

        # Separate ID scores (label != -1) from OOD scores (label == -1).
        return scores[labels != -1], scores[labels == -1]

    def _save_results(self, metrics: dict) -> None:
        output_dir = getattr(self.cfg, 'output_dir', 'results')
        result_name = getattr(self.cfg, 'result_name', 'imagenet_ood_results')
        os.makedirs(output_dir, exist_ok=True)

        rows = [
            {
                'dataset': name,
                'AUROC': values['AUROC'],
                'FPR95': values['FPR95'],
            }
            for name, values in metrics.items()
        ]

        if rows:
            rows.append({
                'dataset': 'AVERAGE',
                'AUROC': sum(row['AUROC'] for row in rows) / len(rows),
                'FPR95': sum(row['FPR95'] for row in rows) / len(rows),
            })

        json_path = os.path.join(output_dir, f'{result_name}.json')
        csv_path = os.path.join(output_dir, f'{result_name}.csv')

        with open(json_path, 'w') as f:
            json.dump(rows, f, indent=2)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['dataset', 'AUROC', 'FPR95'])
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved results to {json_path} and {csv_path}")
