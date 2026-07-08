"""
Multi-model, multi-dataset benchmark for CLIP robustness evaluation.

Automates evaluation across multiple models, datasets, and attack types,
producing structured result tables.
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from ..config import EvalConfig, PathConfig
from ..data.dataset import CLIPDataset
from ..data.collate import default_collate_fn
from .evaluator import ZeroShotEvaluator


class CLIPBenchmark:
    """
    Comprehensive benchmark runner.

    Evaluates multiple models on multiple datasets with both clean 
    and adversarial (multi-source, multi-epsilon) samples.

    Args:
        data_root: Root directory for clean datasets.
        adv_roots_dict: Mapping of attack_name -> adv_root_path.
        device: Computation device.
        eval_config: Evaluation configuration.
    """

    def __init__(
        self,
        data_root: str = None,
        adv_roots_dict: dict = None,
        device: str = "cuda",
        eval_config: EvalConfig = None,
    ):
        self.cfg = eval_config or EvalConfig()
        self.data_root = data_root or PathConfig().data_root
        self.adv_roots_dict = adv_roots_dict or {}
        self.device = device
        self.results = []

    def _get_all_classes(self, dataset_name: str, csv_path: str) -> list:
        if "ImageNet" in dataset_name:
            return ResNet50_Weights.DEFAULT.meta["categories"]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        return sorted(list(set(df.iloc[:, 1].values)))

    def run(
        self,
        model_configs: dict,
        datasets_list: list = None,
        adv_subfolders: list = None,
    ):
        """
        Execute the benchmark.

        Args:
            model_configs: {"ModelName": model_path_or_object, ...}
                           Value can be a string path (HF CLIP) or a nn.Module.
            datasets_list: List of dataset name strings. Defaults to EvalConfig.datasets.
            adv_subfolders: Epsilon subfolder names. Defaults to EvalConfig.adv_subfolders.
        """
        datasets_list = datasets_list or self.cfg.datasets
        adv_subfolders = adv_subfolders or self.cfg.adv_subfolders

        for model_name, model_path_or_obj in model_configs.items():
            display_name = model_name
            adv_subdir_name = model_name

            try:
                if isinstance(model_path_or_obj, str):
                    model = CLIPModel.from_pretrained(model_path_or_obj, local_files_only=True).to(self.device)
                    processor = CLIPProcessor.from_pretrained(model_path_or_obj, local_files_only=True)
                    model.eval()
                else:
                    model = model_path_or_obj
                    processor = model.processor if hasattr(model, "processor") else None
                    display_name = getattr(model, "display_name", model_name)
                    adv_subdir_name = getattr(model, "adv_subdir_name", model_name)
            except Exception as e:
                print(f"[Error] Failed to load {model_name}: {e}")
                continue

            print(f"\n{'='*40}\nModel: {display_name}\n{'='*40}")

            evaluator = ZeroShotEvaluator(model, processor, self.device, self.cfg.batch_size)

            for dataset_name in datasets_list:
                print(f"\n>> Dataset: {dataset_name}")

                origin_path = os.path.join(self.data_root, dataset_name)
                csv_path = os.path.join(origin_path, "labels.csv")

                try:
                    all_classes = self._get_all_classes(dataset_name, csv_path)
                    print(f"   Classes: {len(all_classes)}")
                except Exception as e:
                    print(f"   [Error] {e}")
                    continue

                text_features = evaluator.compute_text_features(all_classes)

                # --- Benign evaluation ---
                try:
                    ds = CLIPDataset(
                        csv_path=csv_path,
                        img_root=os.path.join(origin_path, "images"),
                        dataset_name=dataset_name,
                        sample_n=self.cfg.sample_n,
                        all_classes=all_classes,
                    )
                    loader = DataLoader(
                        ds, batch_size=self.cfg.batch_size,
                        num_workers=self.cfg.num_workers,
                        collate_fn=default_collate_fn,
                    )
                    acc = evaluator.evaluate(loader, text_features)
                    print(f"   [Benign] Acc: {acc*100:.2f}%")
                    self.results.append({
                        "Model": display_name, "Dataset": dataset_name,
                        "Source": "Benign", "SubMode": "-",
                        "Accuracy": f"{acc*100:.2f}%",
                    })
                except Exception as e:
                    print(f"   [Error] Benign: {e}")

                # --- Adversarial evaluation ---
                for adv_name, adv_root in self.adv_roots_dict.items():
                    for submode in adv_subfolders:
                        target_dir = os.path.join(adv_root, adv_subdir_name, dataset_name, submode)
                        if not os.path.exists(target_dir):
                            continue

                        try:
                            ds_adv = CLIPDataset(
                                csv_path=csv_path,
                                img_root=target_dir,
                                dataset_name=dataset_name,
                                sample_n=self.cfg.sample_n,
                                all_classes=all_classes,
                            )
                            if len(ds_adv) == 0:
                                continue

                            loader = DataLoader(
                                ds_adv, batch_size=self.cfg.batch_size,
                                num_workers=self.cfg.num_workers,
                                collate_fn=default_collate_fn,
                            )
                            acc = evaluator.evaluate(loader, text_features)
                            print(f"   [Adv: {adv_name}/{submode}] Acc: {acc*100:.2f}%")
                            self.results.append({
                                "Model": display_name, "Dataset": dataset_name,
                                "Source": adv_name, "SubMode": submode,
                                "Accuracy": f"{acc*100:.2f}%",
                            })
                        except Exception as e:
                            print(f"   [Error] {adv_name}/{submode}: {e}")

    def get_results_df(self) -> pd.DataFrame:
        """Get results as a DataFrame."""
        return pd.DataFrame(self.results)

    def save_results(self, filepath: str):
        """Save results to CSV."""
        df = self.get_results_df()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"[Benchmark] Results saved to {filepath}")

    def print_results(self):
        """Print formatted results table."""
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        df = self.get_results_df()
        if not df.empty:
            try:
                print(df.to_markdown(index=False))
            except ImportError:
                print(df)
        else:
            print("No results.")
