import torch


def hyper_score(dtype, device):
    from .hyper_score import HyperScore

    scorer = HyperScore(dtype=dtype, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, num_views):
        scores = scorer(images, prompts, num_views)
        return scores

    return _fn


def pick_score(dtype, device):
    from .pick_score import PickScore

    scorer = PickScore(dtype=dtype, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts):
        scores = scorer(images, prompts)
        return scores

    return _fn


def hpsv2(dtype, device):
    from .hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=dtype, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts):
        scores = scorer(images, prompts)
        return scores

    return _fn


def image_reward(dtype, device):
    from .image_reward import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=dtype, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts):
        scores = scorer(images, prompts)
        return scores

    return _fn
