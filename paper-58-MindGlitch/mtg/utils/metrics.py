from typing import List, Union, MutableMapping, Any, cast

from omegaconf import DictConfig
import torch
from PIL import Image
import numpy as np
import clip
from torch.utils.data import DataLoader
from scipy.stats import spearmanr


def init_metrics_dict(metric_names: List[str], THRESHOLDS: List[float]) -> dict[str, list[float]]:
    """
    Initialize a metrics dictionary with metric names and thresholds.

    Args:
        metric_names (List[str]): List of metric names to initialize
        THRESHOLDS (List[float]): List of threshold values for VSM metrics

    Returns:
        dict[str, List]: Dictionary with metric keys and empty lists as values.
                        VSM metrics are expanded with threshold suffixes.
    """

    metrics_dict = {}
    for metric in metric_names:
        if "vsm" in metric:  # VSM comes with different thresholds
            for thresh in THRESHOLDS:
                metrics_dict[f"{metric}_{thresh}"] = []
        else:
            metrics_dict[metric] = []
    return metrics_dict


def _spearman_coeff(a: list[float], b: list[float]) -> float:
    """Return the Spearman correlation coefficient as float across SciPy versions."""
    res = spearmanr(a, b)
    # Newer SciPy returns an object with .statistic; older returns a tuple-like
    stat = getattr(res, "statistic", None)
    if stat is not None:
        return float(stat)
    # Fallback to tuple indexing
    return float(res[0])  # type: ignore[index]


def compute_vsm_correlations(
    metrics: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Compute Spearman correlation between VSM metrics and oracle metrics.

    Args:
        metrics (dict[str, List]): Dictionary of metrics with metric names as keys and lists of values as values.

    Returns:
        dict[str, float]: Dictionary of correlation coefficients for each VSM metric.
    """
    oracle = cast(list[float], metrics["oracle"])  # metrics are accumulated as lists
    metrics_names = list(metrics.keys())
    for metric_name in metrics_names:
        if "vsm" in metric_name and "oracle" in metrics:
            values = cast(list[float], metrics[metric_name])
            metrics["pearson_" + metric_name] = float(np.corrcoef(values, oracle)[0, 1])
            metrics["spearman_" + metric_name] = _spearman_coeff(values, oracle)
    return metrics


class ClipSimilarity:
    def __init__(self):
        self.device = self._get_device()
        self.model, self.preprocess = self._initialize_model("ViT-B/32", self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self, model_name="ViT-B/32", device="cpu"):
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess

    def _embed_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        preprocessed_image = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        image_embeddings = self.model.encode_image(preprocessed_image)
        return image_embeddings

    def _embed_text(self, text):
        text_token = clip.tokenize([text]).to(self.device)
        text_embed = self.model.encode_text(text_token)
        return text_embed

    def temporal_similarity(self, images):
        image_embeds_1 = self._embed_images(images)
        image_embeds_2 = torch.roll(image_embeds_1.clone(), -1, 0)
        # print(image_embeds_1.shape, image_embeds_2.shape)
        raw_similarity = self.cosine_similarity(image_embeds_1, image_embeds_2)
        # print(raw_similarity.shape)
        return raw_similarity.mean()

    @torch.no_grad()
    def t2i_similarity(self, text, images):
        text_embeds = self._embed_text(text)
        image_embeds = self._embed_images(images)
        raw_similarity = self.cosine_similarity(text_embeds, image_embeds)
        return raw_similarity

    @torch.no_grad()
    def i2i_similarity(self, image_1, image_2):
        image_1_embeds = self._embed_images(image_1)
        image_2_embeds = self._embed_images(image_2)
        raw_similarity = self.cosine_similarity(image_1_embeds, image_2_embeds)
        return raw_similarity
