# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Topic-aware test case generator with superlevel set failure discovery.

Provides:

- Superlevel Set (SS) acquisition functions (Eq. 1 in paper)
- BERTopic-based topic extraction
- :class:`TopicAwareGenerator` with internal GP state management and
  dynamic hard-anchor selection via SS / SS-Gen / TSS strategies.
- Encoder-based prior setup and GP posterior.

Supports both **GSM8K** (math) and **StrategyQA** (yes/no reasoning) datasets.

Two prior modes:

1. **Without pretrain** — pass ``prior_u`` and ``prior_S`` computed from
   other models' predictions. No neural encoder needed.
2. **With pretrain** — pass ``encoder_path`` and ``embeddings_path``.
   A pre-trained neural encoder computes the GP prior from question embeddings.

Example::

    from proeval.generator import TopicAwareGenerator, extract_topics_bertopic

    topics, _, assignments = extract_topics_bertopic(df["question"].tolist())

    # Without pretrain: prior from model predictions
    gen = TopicAwareGenerator(
        df=df, dataset="gsm8k", api_key="sk-...",
        prior_u=u, prior_S=S, noise_variance=0.3,
    )

    # With pretrain: prior from neural encoder
    gen = TopicAwareGenerator(
        df=df, dataset="gsm8k", api_key="sk-...",
        encoder_path="encoder.pth", embeddings_path="embeddings.npy",
    )

    # Same loop for both:
    for i in range(100):
        case = gen.generate(strategy="tss", k_examples=5)
        score = evaluate(case["question"])
        gen.update(score)
"""

import json
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from proeval.evaluator.client import OpenRouterClient
from proeval.generator.prompt import (
    GSM8K_SCHEMA,
    STRATEGYQA_SCHEMA,
    build_gsm8k_prompt,
    build_strategyqa_prompt,
    format_hard_examples,
)


# SS (Superlevel Set) Acquisition


def ss_acquisition_batch(
    u_t: np.ndarray,
    s_t: np.ndarray,
    unlabeled_indices: List[int],
    topic_assignments: Optional[List[int]] = None,
    selected_topic: Optional[int] = None,
    threshold: float = 0.55,
    beta: float = 1.96,
    mode: str = "top_n",
    n: int = 5,
    embeddings: Optional[np.ndarray] = None,
) -> List[int]:
    """Superlevel Set (SS) batch acquisition.

    ``SS(x) = 𝟙(μ_t(x) ≥ λ) × k_t(x, x)``

    Args:
        u_t: Posterior mean ``(n_samples,)``.
        s_t: Posterior variance ``(n_samples,)``.
        unlabeled_indices: Unlabeled sample indices.
        topic_assignments: Per-sample topic IDs (optional, for TSS).
        selected_topic: Restrict to this topic (optional, for TSS).
        threshold: Decision boundary λ.
        mode: ``"top_n"`` | ``"greedy"`` | ``"threshold"``.
        n: Number of samples to select.
        embeddings: For greedy diversity.
    """
    if topic_assignments is not None and selected_topic is not None:
        cands = [i for i in unlabeled_indices if topic_assignments[i] == selected_topic]
        if not cands:
            cands = unlabeled_indices
    else:
        cands = unlabeled_indices

    if not cands:
        return []

    # Paper Eq. 1: α_SS(x) = 𝟙(μ_t(x) + β·σ_t(x) ≥ λ) × k_t(x,x)
    sigma = np.sqrt(np.maximum(s_t[cands], 1e-10))
    above = (u_t[cands] + beta * sigma >= threshold).astype(float)
    k_diag = np.maximum(s_t[cands], 1e-10)  # k_t(x,x) = posterior variance
    scores = above * k_diag
    if np.max(scores) == 0:
        scores = k_diag

    if mode == "threshold":
        filtered = [
            c for c in cands
            if u_t[c] + beta * np.sqrt(max(s_t[c], 1e-10)) >= threshold
        ]
        if not filtered:
            return [cands[int(np.argmax(scores))]]
        fv = [max(s_t[i], 1e-10) for i in filtered]  # k_t(x,x) = variance
        return [filtered[int(np.argmax(fv))]]

    if mode == "top_n":
        k = min(n, len(cands))
        top = np.argsort(scores)[::-1][:k]
        return [cands[i] for i in top]

    if mode == "greedy":
        pool_size = min(n * 3, len(cands))
        pool = [cands[i] for i in np.argsort(scores)[::-1][:pool_size]]
        if embeddings is None or len(pool) <= n:
            return pool[:n]
        selected = [pool[0]]
        remaining = pool[1:]
        while len(selected) < n and remaining:
            best_i, best_d = 0, -1.0
            for i, c in enumerate(remaining):
                md = min(np.linalg.norm(embeddings[c] - embeddings[s]) for s in selected)
                if md > best_d:
                    best_d, best_i = md, i
            selected.append(remaining.pop(best_i))
        return selected

    raise ValueError(f"Unknown mode: {mode}")


def ss_acquisition(
    u_t: np.ndarray,
    s_t: np.ndarray,
    unlabeled_indices: List[int],
    **kwargs,
) -> int:
    """Single-sample SS acquisition (convenience wrapper)."""
    indices = ss_acquisition_batch(u_t, s_t, unlabeled_indices, n=1, **kwargs)
    return indices[0] if indices else unlabeled_indices[0]


# BQ Hard Problem Selection (without pretrain: learned prior)


def select_hard_problems_bq(
    test_x: np.ndarray,
    test_y: np.ndarray,
    u: np.ndarray,
    S: np.ndarray,
    budget: int,
    threshold: float = 0.7,
    noise_variance: float = 0.3,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Select hard problems via HSS active sampling.

    Returns ``(hard_indices, posterior_mean, posterior_var)``.
    """
    from proeval.sampler.bq import _get_posterior

    n_samples = test_x.shape[1]
    labeled: List[int] = []
    unlabeled = list(range(n_samples))
    u_t, s_t = u.copy(), np.diag(S) if S.ndim == 2 else np.array([S])

    for _ in range(min(budget, n_samples)):
        if not unlabeled:
            break
        best = ss_acquisition(u_t, s_t, unlabeled, threshold=0.5, beta=1.96)
        labeled.append(best)
        unlabeled.remove(best)
        if labeled:
            u_t, s_t = _get_posterior(
                test_x[:, labeled], test_y[labeled], test_x, noise_variance, labeled, u
            )

    hard = [i for i in labeled if u[i] < threshold]
    if len(hard) < budget // 2:
        hard = sorted(labeled, key=lambda x: u[x])[:budget]
    return hard, u_t, s_t


# Topic Extraction


def extract_topics_bertopic(
    questions: List[str], n_topics: int = 10
) -> Tuple[List[str], Dict[int, List[str]], List[int]]:
    """Extract topics from questions using BERTopic.

    Returns ``(topic_labels, topic_keywords, topic_assignments)``.
    """
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from hdbscan import HDBSCAN

        vec = CountVectorizer(stop_words="english", min_df=1, max_df=0.95, ngram_range=(1, 2))
        hdb = HDBSCAN(min_cluster_size=5, min_samples=2, metric="euclidean",
                       cluster_selection_method="leaf", prediction_data=True)
        model = BERTopic(nr_topics=n_topics, hdbscan_model=hdb, vectorizer_model=vec, verbose=False)
        topics, _ = model.fit_transform(questions)

        labels, kw = [], {}
        info = model.get_topic_info()
        for _, row in info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue
            words = model.get_topic(tid)
            if words:
                kw[tid] = [w for w, _ in words[:5]]
                labels.append(f"Topic: {', '.join(kw[tid][:3])}")
        return labels, kw, topics

    except ImportError:
        raise ImportError(
            "BERTopic is required for topic extraction. "
            "Install it with: pip install bertopic hdbscan"
        )


# Encoder-Based Prior (with pretrain)


def setup_encoder_prior(
    encoder_path: str,
    embeddings_path: str,
    device=None,
):
    """Load a pre-trained encoder and compute BQ prior from phi embeddings.

    Args:
        encoder_path: Path to trained encoder ``.pth`` file.
        embeddings_path: Path to question embeddings ``.npy`` file.
        device: Torch device (default: auto-detect).

    Returns:
        encoder: Loaded ``QuestionEncoder`` model.
        phi_embeddings: ``(n_samples, hidden_dim)`` phi features.
        u: Prior mean ``(n_samples,)`` from psi output.
        S: Prior covariance / kernel matrix ``(n_samples, n_samples)``.
        var: Learned noise variance from encoder.
    """
    import torch
    from proeval.encoder import compute_kernel_matrix, get_phi_embeddings, load_encoder

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading encoder from: {encoder_path}")
    encoder, checkpoint = load_encoder(encoder_path, device)

    print(f"Loading embeddings from: {embeddings_path}")
    question_embeddings = np.load(embeddings_path).astype(np.float32)
    print(f"  Embeddings shape: {question_embeddings.shape}")

    phi_embeddings = get_phi_embeddings(encoder, question_embeddings, device)
    print(f"  Phi embeddings shape: {phi_embeddings.shape}")

    encoder.eval()
    with torch.no_grad():
        x = torch.from_numpy(question_embeddings).float().to(device)
        psi_x = encoder.forward_psi(x)
        u = psi_x.mean(dim=1).cpu().numpy()

    print(f"  Prior mean range: [{u.min():.3f}, {u.max():.3f}]")

    var = encoder.var.item()
    print(f"  Learned variance: {var:.4f}")

    phi_t = torch.from_numpy(phi_embeddings).float().to(device)
    K = compute_kernel_matrix(phi_t, encoder)
    S = K.detach().cpu().numpy()

    return encoder, phi_embeddings, u, S, var


def get_posterior_embedding(
    phi_train: np.ndarray,
    train_y: np.ndarray,
    phi_test: np.ndarray,
    noise_variance: float,
    train_inds: List[int],
    u: np.ndarray,
    encoder,
    device=None,
    full_cov: bool = False,
):
    """Compute GP posterior mean and variance using encoder kernel.

    Uses the encoder's kernel function (linear, Matérn, or RBF).

    Args:
        phi_train: Phi embeddings of labeled samples ``(n_labeled, hidden_dim)``.
        train_y: Labels of labeled samples ``(n_labeled,)``.
        phi_test: Phi embeddings of all samples ``(n_samples, hidden_dim)``.
        noise_variance: GP noise variance.
        train_inds: Indices of labeled samples.
        u: Prior mean ``(n_samples,)``.
        encoder: ``QuestionEncoder`` for kernel computation.
        device: Torch device.
        full_cov: Whether to return full covariance matrix.

    Returns:
        u_t: Posterior mean ``(n_samples,)``.
        s_t: Posterior variance ``(n_samples,)`` or covariance matrix.
    """
    import torch
    from proeval.encoder import compute_kernel_matrix

    if device is None:
        device = next(encoder.parameters()).device

    if len(train_inds) == 0:
        phi_t = torch.from_numpy(phi_test).float().to(device)
        K = compute_kernel_matrix(phi_t, encoder)
        if full_cov:
            return u.copy(), K.cpu().numpy()
        else:
            return u.copy(), torch.diag(K).cpu().numpy()

    phi_train_t = torch.from_numpy(phi_train).float().to(device)
    phi_test_t = torch.from_numpy(phi_test).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    u_t = torch.from_numpy(u).float().to(device)

    K_train = compute_kernel_matrix(phi_train_t, encoder)
    K_train_reg = K_train + noise_variance * torch.eye(len(train_inds), device=device)

    kernel_type = getattr(encoder, "kernel_type", "linear")
    if kernel_type == "linear":
        K_test_train = torch.mm(phi_test_t, phi_train_t.t())
    else:
        lengthscale = encoder.lengthscale
        dist_sq = (
            torch.sum(phi_test_t ** 2, dim=1, keepdim=True)
            + torch.sum(phi_train_t ** 2, dim=1, keepdim=True).t()
            - 2 * torch.mm(phi_test_t, phi_train_t.t())
        )
        dist_sq = torch.clamp(dist_sq, min=0)
        dist = torch.sqrt(dist_sq + 1e-12)
        scaled_dist = dist / lengthscale

        if kernel_type == "rbf":
            K_test_train = torch.exp(-0.5 * (scaled_dist ** 2))
        elif kernel_type == "matern":
            nu = getattr(encoder, "matern_nu", 2.5)
            if nu == 0.5:
                K_test_train = torch.exp(-scaled_dist)
            elif nu == 1.5:
                sqrt3 = np.sqrt(3)
                K_test_train = (1 + sqrt3 * scaled_dist) * torch.exp(-sqrt3 * scaled_dist)
            else:  # nu == 2.5
                sqrt5 = np.sqrt(5)
                K_test_train = (
                    (1 + sqrt5 * scaled_dist + (5 / 3) * (scaled_dist ** 2))
                    * torch.exp(-sqrt5 * scaled_dist)
                )

    try:
        L = torch.linalg.cholesky(K_train_reg)
    except RuntimeError:
        K_train_reg = K_train_reg + 1e-4 * torch.eye(len(train_inds), device=device)
        L = torch.linalg.cholesky(K_train_reg)

    y_residual = train_y_t - u_t[train_inds]
    alpha = torch.cholesky_solve(y_residual.unsqueeze(1), L).squeeze(1)

    posterior_mean = u_t + K_test_train @ alpha

    if full_cov:
        K_test = compute_kernel_matrix(phi_test_t, encoder)
        v = torch.cholesky_solve(K_test_train.t(), L)
        posterior_cov = K_test - K_test_train @ v
        return posterior_mean.detach().cpu().numpy(), posterior_cov.detach().cpu().numpy()
    else:
        K_test_diag = torch.ones(phi_test_t.shape[0], device=device)
        v = torch.cholesky_solve(K_test_train.t(), L)
        posterior_var = K_test_diag - torch.sum(K_test_train * v.t(), dim=1)
        posterior_var = torch.clamp(posterior_var, min=1e-10)
        return posterior_mean.detach().cpu().numpy(), posterior_var.detach().cpu().numpy()


# TopicAwareGenerator


class TopicAwareGenerator:
    """Topic-aware test case generator with internal GP state management.

    Manages its own GP posterior and dynamically selects hard anchors via
    SS acquisition each ``generate()`` call. Call ``update(score)`` after
    evaluating each generated case to update the GP posterior.

    Topic modeling (BERTopic) is handled internally — just pass ``n_topics``.

    Two prior modes:

    1. **Without pretrain** — pass ``prior_u`` and ``prior_S``
       (pre-computed from model predictions).
    2. **With pretrain** — pass ``encoder_path`` and
       ``embeddings_path``; prior is auto-computed from the encoder.
    3. **RPF mode** — pass ``rpf_embeddings`` for Matérn kernel GP with
       neutral prior (0.5).  No encoder or model predictions needed.

    Args:
        df: Source DataFrame with ``question`` and ``ground_truth`` columns,
            or a :class:`~proeval.utils.Dataset` (its frame and ``name`` are
            used; a non-default *dataset* still overrides the prompt format).
        dataset: ``"gsm8k"`` or ``"strategyqa"``.
        api_key: OpenRouter API key (or set ``OPENROUTER_API_KEY`` env var).
        model: Model to use for generation.
        n_topics: Number of topics for BERTopic (default 11).
        prior_u: Prior mean ``(n_samples,)`` — without pretrain.
        prior_S: Prior covariance ``(n_samples, n_samples)`` — without pretrain.
        noise_variance: GP noise variance (default 0.3).
        encoder_path: Path to trained encoder ``.pth`` file — with pretrain.
        embeddings_path: Path to question embeddings ``.npy`` file — with pretrain.
        rpf_embeddings: Raw text embeddings ``(n_samples, d)`` — RPF mode.
        ss_threshold: SS acquisition threshold λ (default 0.0).
        ss_beta: SS acquisition β (default 1.96).

    Example::

        gen = TopicAwareGenerator(
            df=df, dataset="gsm8k", api_key="sk-...",
            encoder_path="encoder.pth", embeddings_path="emb.npy",
        )
        for i in range(100):
            case = gen.generate(strategy="tss", k_examples=5)
            score = evaluate(case["question"])
            gen.update(score)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset: str = "gsm8k",
        api_key: Optional[str] = None,
        model: str = "google/gemini-3-flash-preview",
        n_topics: int = 11,
        # Without pretrain: learned prior
        prior_u: Optional[np.ndarray] = None,
        prior_S: Optional[np.ndarray] = None,
        noise_variance: float = 0.3,
        # With pretrain: encoder prior
        encoder_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        # RPF mode: raw embeddings + Matérn kernel
        rpf_embeddings: Optional[np.ndarray] = None,
        # SS acquisition config
        ss_threshold: float = 0.0,
        ss_beta: float = 1.96,
    ):
        # Accept a Dataset in place of a DataFrame: derive the question frame
        # and the dataset name from it. An explicit, non-default `dataset`
        # still wins (it controls the prompt format).
        from proeval.utils.dataset import Dataset

        if isinstance(df, Dataset):
            if dataset == "gsm8k":
                dataset = df.name
            df = df.to_frame()

        self.df = df
        self.dataset = dataset
        self.client = OpenRouterClient(api_key=api_key)
        self.model = model
        self.ss_threshold = ss_threshold
        self.ss_beta = ss_beta

        # Run BERTopic internally
        questions = df["question"].tolist()
        self.topics, self._topic_keywords, self.topic_assignments = (
            extract_topics_bertopic(questions, n_topics=n_topics)
        )
        print(f"Discovered {len(self.topics)} topics")

        # Topic stats for UCB selection
        self.topic_stats: Dict[str, Dict] = {
            t: {"failures": 0, "total": 0} for t in self.topics
        }
        self.unique_topics = list(set(self.topic_assignments))

        # GP state
        self.labeled_indices: List[int] = []
        self.labeled_y: List[float] = []
        self._last_anchors: List[int] = []
        self._iteration = 0

        # Encoder state (with pretrain only)
        self._encoder = None
        self._phi_embeddings = None
        self._device = None

        # RPF state (raw embeddings only)
        self._rpf_embeddings = None

        n_samples = len(df)

        if encoder_path is not None:
            # With pretrain: encoder prior (TPF)
            if embeddings_path is None:
                raise ValueError(
                    "embeddings_path is required when encoder_path is provided"
                )
            encoder, phi_emb, u, S, var = setup_encoder_prior(
                encoder_path, embeddings_path,
            )
            self._encoder = encoder
            self._phi_embeddings = phi_emb
            self._device = next(encoder.parameters()).device
            self._prior_u = u
            self.noise_variance = var  # use encoder's learned variance
            self.u_t = u.copy()
            self.s_t = np.ones(n_samples)  # normalized kernel diag ≈ 1

        elif rpf_embeddings is not None:
            # RPF mode: raw text embeddings + Matérn kernel, neutral prior
            self._rpf_embeddings = rpf_embeddings / (
                np.linalg.norm(rpf_embeddings, axis=1, keepdims=True) + 1e-10
            )
            self._prior_u = np.ones(n_samples) * 0.5
            self.noise_variance = noise_variance
            self.u_t = self._prior_u.copy()
            self.s_t = np.ones(n_samples)  # Matérn self-kernel ≈ 1

        elif prior_u is not None:
            # Without pretrain: learned prior (SF)
            self._prior_u = prior_u
            self.noise_variance = noise_variance
            self.u_t = prior_u.copy()
            self.s_t = (
                np.diag(prior_S) if prior_S is not None and prior_S.ndim == 2
                else np.ones(n_samples)
            )
            # Store test_x for _get_posterior (without pretrain needs this)
            self._prior_S = prior_S

        else:
            raise ValueError(
                "Provide one of: (encoder_path, embeddings_path) for TPF, "
                "rpf_embeddings for RPF, or (prior_u, prior_S) for SF mode."
            )

    # Properties

    @property
    def prior_mode(self) -> str:
        """Return the prior mode: ``'encoder'``, ``'rpf'``, or ``'learned'``."""
        if self._encoder is not None:
            return "encoder"
        elif self._rpf_embeddings is not None:
            return "rpf"
        return "learned"

    @property
    def n_samples(self) -> int:
        """Number of source samples."""
        return len(self.df)

    @property
    def failures_found(self) -> int:
        """Number of failures observed so far."""
        return sum(1 for y in self.labeled_y if y >= 0.5)

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration

    # Topic Selection

    def _select_topic_ucb1(
        self, items: List[str], stats: Dict, exploration: float = 1.0
    ) -> str:
        """Select topic using UCB1 (Auer et al., 2002).

        UCB1(s) = r̄(s) + c·√(ln(N) / n(s))

        where r̄(s) is the failure rate for topic s, N is total trials,
        and n(s) is trials for topic s.
        """
        N = max(1, sum(s["total"] for s in stats.values()))
        scores = []
        for item in items:
            s = stats[item]
            n_i = max(1, s["total"])
            reward = s["failures"] / n_i  # failure rate as reward
            bonus = exploration * np.sqrt(np.log(N) / n_i)
            scores.append(reward + bonus)
        return items[int(np.argmax(scores))]

    # Dynamic Hard Anchor Selection

    def _select_hard_anchors(
        self,
        k: int,
        topic_id: Optional[int] = None,
        use_topic: bool = True,
    ) -> List[Dict]:
        """Select hard anchors from current GP posterior via SS acquisition.

        Returns list of dicts with ``question``, ``ground_truth``,
        ``prior_mean`` — ready for prompt building.
        """
        all_indices = list(range(self.n_samples))

        anchors = []
        remaining = list(all_indices)
        for _ in range(min(k, len(remaining))):
            best = ss_acquisition(
                self.u_t, self.s_t, remaining,
                topic_assignments=self.topic_assignments if use_topic else None,
                selected_topic=topic_id if use_topic else None,
                threshold=self.ss_threshold,
                beta=self.ss_beta,
            )
            anchors.append(best)
            if best in remaining:
                remaining.remove(best)

        self._last_anchors = anchors

        hard_examples = []
        for idx in anchors:
            hard_examples.append({
                "question": self.df.iloc[idx]["question"],
                "ground_truth": str(self.df.iloc[idx]["ground_truth"]),
                "prior_mean": float(self.u_t[idx]),
            })
        return hard_examples

    # GP Posterior Update

    def update(self, score: float) -> None:
        """Feed back an evaluation result to update the GP posterior.

        Args:
            score: Error score — ``1.0`` for failure, ``0.0`` for correct.
        """
        self.labeled_y.append(score)

        # Update topic stats for last generation
        # (topic_stats already updated in generate(), but labeled_y is needed
        #  for posterior update)

        if self.prior_mode == "encoder":
            # With pretrain: encoder-based posterior (TPF)
            if self.labeled_indices:
                phi_train = self._phi_embeddings[self.labeled_indices]
                self.u_t, self.s_t = get_posterior_embedding(
                    phi_train,
                    np.array(self.labeled_y),
                    self._phi_embeddings,
                    self.noise_variance,
                    self.labeled_indices,
                    self._prior_u,
                    self._encoder,
                    self._device,
                )
        elif self.prior_mode == "rpf":
            # RPF: Matérn kernel with raw text embeddings
            from proeval.sampler.bq import _get_posterior_matern

            if self.labeled_indices:
                self.u_t, self.s_t = _get_posterior_matern(
                    self._rpf_embeddings[self.labeled_indices],
                    np.array(self.labeled_y),
                    self._rpf_embeddings,
                    self.noise_variance,
                    self.labeled_indices,
                    self._prior_u,
                )
        else:
            # Without pretrain: learned prior posterior (SF)
            from proeval.sampler.bq import _get_posterior

            if self.labeled_indices and self._prior_S is not None:
                # Build test_x from prior_S columns
                # _get_posterior expects test_x shape (n_features, n_samples)
                test_x = self._prior_S  # (n_samples, n_samples) used as feature matrix
                self.u_t, self.s_t = _get_posterior(
                    test_x[:, self.labeled_indices],
                    np.array(self.labeled_y),
                    test_x,
                    self.noise_variance,
                    self.labeled_indices,
                    self._prior_u,
                )

    # Generate

    def generate(self, strategy: str = "tss", k_examples: int = 5) -> Dict:
        """Generate a new test case with dynamic hard-anchor selection.

        Each call uses SS acquisition on the current GP posterior to select
        the hardest anchors for the prompt.

        Args:
            strategy:
                - ``"tss"``: Topic-aware SS — UCB topic + SS anchors (best).
                - ``"ss_gen"``: SS anchors without topic injection.
                - ``"random_topic"``: Random topic, no SS anchors.
                - ``"pure_random"``: No topic, no anchors.
            k_examples: Number of hard anchors to select from posterior.

        Returns:
            dict with ``question``, ``solution``/``reasoning``,
            ``ground_truth``, ``topic``, ``hard_examples_used``,
            ``prior_mode``, ``iteration``.
        """
        self._iteration += 1

        # Backward compatibility
        if strategy in ("hss_gen", "active"):
            strategy = "tss"

        # Topic selection
        if strategy in ("pure_random", "ss_gen"):
            topic = None
            topic_id = None
        elif strategy in ("random_topic", "random"):
            topic = random.choice(self.topics)
            topic_id = self.topics.index(topic) if topic in self.topics else None
            if topic_id is not None and topic_id < len(self.unique_topics):
                topic_id = self.unique_topics[topic_id % len(self.unique_topics)]
        elif strategy == "tss":
            topic = self._select_topic_ucb1(self.topics, self.topic_stats)
            topic_idx = self.topics.index(topic) if topic in self.topics else 0
            topic_id = self.unique_topics[topic_idx % len(self.unique_topics)]
        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Use 'tss', 'ss_gen', 'random_topic', or 'pure_random'."
            )

        # Hard anchor selection
        if strategy in ("tss", "ss_gen") and k_examples > 0:
            use_topic = strategy == "tss"
            selected_hard = self._select_hard_anchors(
                k_examples, topic_id=topic_id, use_topic=use_topic,
            )
        else:
            selected_hard = []
            self._last_anchors = []

        # Build prompt
        if self.dataset == "strategyqa":
            prompt = build_strategyqa_prompt(topic, selected_hard, strategy)
            schema = STRATEGYQA_SCHEMA
        else:
            prompt = build_gsm8k_prompt(topic, selected_hard, strategy)
            schema = GSM8K_SCHEMA

        for _attempt in range(3):
            response = self.client.predict(
                prompt, model=self.model, max_tokens=4096,
                temperature=0.7, response_format=schema,
            )
            try:
                result = json.loads(response)
                break
            except (json.JSONDecodeError, TypeError):
                if _attempt == 2:
                    raise

        # Normalize ground truth
        if self.dataset == "strategyqa":
            truth = str(result.get("ground_truth", "no")).lower().strip()
            result["ground_truth"] = "yes" if truth in ("true", "yes", "1") else "no"
        else:
            gt = str(result.get("ground_truth", "0"))
            m = re.search(r"-?[\d,]+\.?\d*", gt)
            result["ground_truth"] = m.group(0).replace(",", "") if m else gt

        result["topic"] = topic
        result["hard_examples_used"] = len(selected_hard)
        result["prior_mode"] = self.prior_mode
        result["iteration"] = self._iteration
        result["anchor_indices"] = list(self._last_anchors)

        # Update topic stats
        if topic and topic in self.topic_stats:
            self.topic_stats[topic]["total"] += 1

        return result

    def update_stats(self, topic: str, score: float) -> None:
        """Record a failure for a topic (called after evaluation).

        .. deprecated:: Use :meth:`update` instead, which handles both
           topic stats and GP posterior.
        """
        if topic in self.topic_stats and score == 0.0:
            self.topic_stats[topic]["failures"] += 1
