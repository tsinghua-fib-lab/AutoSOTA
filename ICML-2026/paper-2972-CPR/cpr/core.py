"""CPRCore: Conformal Path Reasoning for trustworthy KGQA."""

import os
import re
import math
import random
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import heapq

from cpr.models.rcvnet import ResidualValueMLP
from cpr.llm import VLLMLLM  # noqa: F401 — lazy error if openai missing at runtime
from cpr.retrieval.puct import puct_calib_collect_experience
from cpr.conformal.path_cp import (
    fit_path_threshold,
    nonconformity_score,
    filter_path_conf,
    path_post_process,
)


class CPRCore:
    """
    Conformal Path Reasoning core with TreeG retrieval and RCVNet scoring.

    Training (D_train): PUCT + RCVNet on train_data.
    Calibration (D_cal): query-level path conformal threshold (path mode) or legacy entity CP.
    """

    def __init__(
                 self,
                 global_triples,
                 train_data=None,
                 calibration_data=None,
                 path_alpha: float = 0.3,
                 ans_alpha: float = 0.2,
                 post_alpha: float = 0.1,
                 max_hop: int = 2,
                 encoder=None,
                 encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_thompson: bool = False,
                 ts_weight: float = 0.3,
                 compose_method: str = "product",
                 beam_size: int = 32,
                 prior_threshold: float = 0.0,
                 calib_max_paths: int = 1,
                 calib_max_depth: int = None,
                 # TreeG params
                 treeg_branch_size: int = 16,
                 treeg_active_size: int = None,
                 treeg_weights: dict = None,
                 # LLM params (new)
                 use_llm: bool = True,
                 llm_model_path: str = None,
                 conformal_mode: str = "path",
                 llm_4bit: bool = False,
                 llm_temperature: float = 0.0,
                 llm_num_chains: int = 4,
                 llm_answer_topk_paths: int = 10,
                 vLLM_url: str = None,
                 # Residual MLP value (recommended)
                 learn_residual_value: bool = True,
                 residual_lambda: float = 0.05,
                 delta_clip: float = 0.2,
                 delta_l2: float = 0.5,
                 value_hidden: int = 256,
                 value_lr: float = 1e-3,
                 value_epochs: int = 2,
                 value_batch_size: int = 256,
                 value_max_negs: int = 8,
                 value_l2: float = 1e-4,
                 hard_neg_frac: float = 0.5,
                 value_norm: bool = True,
                 value_use_embeddings: int = 2,
                 # PUCT on calibration only (collect experience; inference uses TreeG/beam as-is)
                 puct_calib: bool = True,
                 puct_calib_num_sims: int = 32,
                 puct_calib_cpuct: float = 2.0,
                 puct_calib_temp: float = 1.5,
                 puct_calib_prior_w: float = 0.5,
                 puct_calib_update_scale: float = 0.5,
                 puct_calib_fail_beta: float = 0.05,
                 auto_compute_thresholds: bool = True,
                 use_global_post_threshold: bool = False,
                 skip_training: bool = False,
                 tau_hat: float = None,
                 ):
        random.seed(42)
        np.random.seed(42)

        # Backward compat: old API used calibration_data for D_train
        if train_data is None and calibration_data is not None:
            train_data = calibration_data
        self.train_data = train_data or []
        self.calibration_data = calibration_data or []
        self.conformal_mode = str(conformal_mode).lower()
        self.tau_hat = tau_hat
        self.path_alpha = path_alpha
        self.ans_alpha = ans_alpha
        self.post_alpha = post_alpha
        self.max_hop = max_hop

        self.use_thompson = use_thompson
        self.ts_weight = float(ts_weight)

        if encoder is not None:
            self.encoder = encoder
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.encoder = SentenceTransformer(encoder_name, device=device)

        # Build graph
        self.graph = self._build_graph(global_triples)

        # Precompute single-hop embeddings
        self.relation_embeds = self._precompute_relation_embeddings()

        # caches
        self._q_embed_cache = {}
        self._r_embed_cache = {}

        # No GSC-KNN here (this is demo base)
        self.relation_posteriors = defaultdict(lambda: [1.0, 1.0])

        self.compose_method = compose_method
        self.beam_size = int(beam_size)
        self.prior_threshold = float(prior_threshold)

        self.calib_max_paths = int(calib_max_paths)
        self.calib_max_depth = self.max_hop if calib_max_depth is None else int(calib_max_depth)

        self.path_scores = defaultdict(list)
        self.ans_scores = []
        self.post_scores = []

        self.q_hats = [0.0] * self.max_hop
        self.q_hat_a = 0.01
        self.q_hat_post = 0.01

        self.use_llm = bool(use_llm)
        self.llm_num_chains = int(llm_num_chains)
        self.llm_answer_topk_paths = int(llm_answer_topk_paths)

        self.llm = None
        if self.use_llm:
            self.llm = VLLMLLM(
                base_url=vLLM_url,
                model=llm_model_path,
                temperature=llm_temperature,
            )

        # TreeG params (new)
        self.treeg_branch_size = int(treeg_branch_size)
        self.treeg_active_size = int(treeg_active_size) if treeg_active_size is not None else int(self.beam_size)
        default_w = {"local": 1.0, "path": 0.5, "prior": 0.5}
        self.treeg_weights = default_w if treeg_weights is None else {**default_w, **treeg_weights}
        if treeg_weights is not None:
            print(self.treeg_weights['local'], self.treeg_weights['path'], self.treeg_weights['prior'])

        # Residual MLP value (learn a small correction on top of V_base)
        self.learn_residual_value = bool(learn_residual_value)
        self.residual_lambda = float(residual_lambda)
        self.delta_clip = float(delta_clip)
        self.delta_l2 = float(delta_l2)
        self.value_hidden = int(value_hidden)
        self.value_lr = float(value_lr)
        self.value_epochs = int(value_epochs)
        self.value_batch_size = int(value_batch_size)
        self.value_max_negs = int(value_max_negs)
        self.value_l2 = float(value_l2)
        self.hard_neg_frac = float(hard_neg_frac)
        self.value_norm = bool(value_norm)
        # value_use_embeddings is now a mode flag:
        #   0 = none (scalars only)
        #   1 = concat embeddings into x  (x = [scalars ; q_emb ; rel_emb ; path_emb])
        #   2 = FiLM-conditioning (x stays 3 scalars; embeddings go into conditioner)
        self.value_use_embeddings = int(value_use_embeddings)  # keep name for backward compatibility
        self.value_embed_concat = (self.value_use_embeddings == 1)
        self.value_embed_film = (self.value_use_embeddings == 2)

        # ---------------- PUCT calibration-only controls ----------------
        # We only use PUCT during calibration to collect extra (positive/negative) evidence for relation_posteriors.
        # Inference/predict() continues to use TreeG-style branch+value+active-set retrieval.
        self.puct_calib = bool(puct_calib)
        self.puct_calib_num_sims = int(puct_calib_num_sims)
        self.puct_calib_cpuct = float(puct_calib_cpuct)
        self.puct_calib_temp = float(puct_calib_temp)
        self.puct_calib_prior_w = float(puct_calib_prior_w)
        self.puct_calib_update_scale = float(puct_calib_update_scale)
        self.puct_calib_fail_beta = float(puct_calib_fail_beta)

        try:
            self.emb_dim = int(self.encoder.get_sentence_embedding_dimension())
        except Exception:
            try:
                self.emb_dim = int(len(self.encoder.encode(["test"], convert_to_numpy=True)[0]))
            except Exception:
                self.emb_dim = 0

        # Input dims:
        # - concat mode: x_dim = 3 + 3*D
        # - film mode:   x_dim = 3, cond_dim = 3*D
        # - none:        x_dim = 3
        self.value_in_dim = 3 + (3 * self.emb_dim if (self.value_embed_concat and self.emb_dim > 0) else 0)
        self.value_cond_dim = (3 * self.emb_dim if (self.value_embed_film and self.emb_dim > 0) else 0)

        self.value_model: Optional[ResidualValueMLP] = None
        self.value_feat_mean: Optional[torch.Tensor] = None
        self.value_feat_std: Optional[torch.Tensor] = None
        if self.learn_residual_value:
            self.value_model = ResidualValueMLP(in_dim=self.value_in_dim, hidden=self.value_hidden,
                                                cond_dim=getattr(self, 'value_cond_dim', 0)).to(self.encoder.device)

        # run calibration to get relation posteriors etc.
        self.auto_compute_thresholds = bool(auto_compute_thresholds)
        self.use_global_post_threshold = bool(use_global_post_threshold)

        if not skip_training:
            self._calculate_training_scores()
            if self.learn_residual_value:
                self._train_residual_value()
            if self.auto_compute_thresholds and self.conformal_mode == "legacy":
                self._compute_thresholds()

    def _llm_propose_relations(self, question: str):
        if (not self.use_llm) or (self.llm is None):
            return []

        chains = self.llm.propose_relation_chains(
            question,
            max_hop=self.max_hop,
            num_chains=self.llm_num_chains
        )
        # print("\n")
        # print(question)
        # print(chains)
        llm_rel_embeds = []
        for c in chains:
            for r in c:
                if isinstance(r, str) and r.strip():
                    emb = self._embed_llm_relation(r.strip())
                    if emb is not None:
                        llm_rel_embeds.append(emb)

        return llm_rel_embeds

    def _is_mid(self, x: str) -> bool:
        """
        Robustly checks if a node is an ID (MID, GID) rather than a name.
        Assumes MIDs are like 'm.0123' or 'g.123'.
        """
        if not x: return False
        x_lower = x.lower().strip()
        # Fast prefix check first
        if not (x_lower.startswith("m.") or x_lower.startswith("g.")):
            return False
        # Robust regex check (optional, but good for cleanliness)
        # Matches strictly m.xxxx or g.xxxx where x is alphanumeric/underscore
        return bool(re.match(r'^[mg]\.[a-zA-Z0-9_]+$', x_lower))

    # ----------- Graph & Embeddings -----------
    def _build_graph(self, triples):
        G = nx.DiGraph()
        for h, r, t in triples:
            try:
                if G.has_edge(h, t):
                    rel = G[h][t].get("relation", None)
                    if rel is None:
                        G[h][t]["relation"] = [r]
                    elif isinstance(rel, list):
                        rel.append(r)
                    else:
                        G[h][t]["relation"] = [rel, r]
                else:
                    G.add_edge(h, t, relation=r)
            except Exception:
                continue
        return G

    def _precompute_relation_embeddings(self):
        """
        Precompute embeddings for atomic relations.
        Compatible with multi-edge graph where edge['relation'] can be str or List[str].
        """
        rel_set = set()

        for _, _, d in self.graph.edges(data=True):
            rel = d.get("relation", None)
            if rel is None:
                continue
            if isinstance(rel, list):
                for r in rel:
                    if isinstance(r, str):
                        rel_set.add(r)
            else:
                if isinstance(rel, str):
                    rel_set.add(rel)

        rels = list(rel_set)
        if not rels:
            return {}

        embeds = []
        batch_size = 512
        for i in tqdm(range(0, len(rels), batch_size),
                      desc="Encoding relations",
                      unit="batch"):
            batch = rels[i:i + batch_size]
            embeds.append(self.encoder.encode(batch, convert_to_numpy=True))

        if not embeds:
            return {}

        embeds = np.vstack(embeds)
        return {r: e for r, e in zip(rels, embeds)}

    def _embed_query_tensor(self, text):
        device = self.encoder.device
        if text not in self._q_embed_cache:
            v = self.encoder.encode([text], convert_to_numpy=True)[0]
            t = torch.tensor(v, device=device, dtype=torch.float32)
            t = torch.nn.functional.normalize(t, p=2, dim=0)
            self._q_embed_cache[text] = t
        return self._q_embed_cache[text]

    def _relation_tensor(self, vec_np):
        device = self.encoder.device
        t = torch.tensor(vec_np, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(t, p=2, dim=0)

    def _compose_relation_embedding(self, rel: str):
        if rel in self.relation_embeds:
            return self.relation_embeds[rel]
        if "->" not in rel:
            return self.encoder.encode([rel], convert_to_numpy=True)[0]

        parts = [p.strip() for p in rel.split("->") if p.strip()]
        vectors = []
        for p in parts:
            if p in self.relation_embeds:
                vectors.append(self.relation_embeds[p])
            else:
                vectors.append(self.encoder.encode([p], convert_to_numpy=True)[0])

        if self.compose_method == "product":
            v = vectors[0].astype(np.float64)
            for x in vectors[1:]:
                v = v * (x.astype(np.float64) + 1e-10)
            v = v / (np.linalg.norm(v) + 1e-8)
            return v.astype(np.float32)
        else:
            w = np.zeros_like(vectors[0], dtype=np.float64)
            for i, x in enumerate(vectors):
                w += x.astype(np.float64) * (1.0 / (i + 1))
            w = w / (np.linalg.norm(w) + 1e-8)
            return w.astype(np.float32)

    def _get_relation_tensor(self, rel: str):
        if rel in self._r_embed_cache:
            return self._r_embed_cache[rel]
        if rel in self.relation_embeds:
            t = self._relation_tensor(self.relation_embeds[rel])
        else:
            t = self._relation_tensor(self._compose_relation_embedding(rel))
        self._r_embed_cache[rel] = t
        return t

    def _sim_score(self, s1, rel):
        q_t = self._embed_query_tensor(s1)
        r_t = self._get_relation_tensor(rel)
        return -float(torch.nn.functional.cosine_similarity(q_t.unsqueeze(0), r_t.unsqueeze(0))[0])

    # ----------- PUCT experience on calibration (training-time only) -----------
    def _puct_calib_collect_experience(self,
                                       G_sub: nx.DiGraph,
                                       masked_q: str,
                                       qents: List[str],
                                       aents: List[str]):
        """Delegate to PUCT module (training phase only)."""
        puct_calib_collect_experience(self, G_sub, masked_q, qents, aents)

    # ----------- Bounded shortest path (for training) -----------
    def bounded_shortest_paths(self, G, s, t, max_depth, max_paths):
        """
        Returns: List[Tuple[nodes_list, rels_list]]
        - nodes_list: [s, ..., t]
        - rels_list:  [r1, r2, ...] aligned with gaps between nodes
        Supports edge['relation'] as str or List[str].
        """
        if s == t:
            return [([s], [])]

        q = deque([(s, [s], [])])
        res = []
        shortest = None

        while q:
            node, nodes_path, rels_path = q.popleft()
            d = len(rels_path)

            if shortest is not None and d > shortest:
                break
            if d > max_depth:
                continue

            for nb in G.neighbors(node):
                edge_rel = G[node][nb].get("relation", None)
                if edge_rel is None:
                    continue
                rel_options = edge_rel if isinstance(edge_rel, list) else [edge_rel]

                for rel in rel_options:
                    new_nodes = nodes_path + [nb]
                    new_rels = rels_path + [rel]

                    if nb == t:
                        if shortest is None:
                            shortest = d + 1
                        if d + 1 == shortest:
                            res.append((new_nodes, new_rels))
                            if len(res) >= max_paths:
                                return res
                    else:
                        q.append((nb, new_nodes, new_rels))

        return res

    # ----------- Calibration -----------

    def _calculate_training_scores(self):
        """Phase 1: PUCT + relation posteriors on D_train."""

        self.path_scores = defaultdict(list)
        self.ans_scores = []
        self.post_scores = []

        for item in tqdm(self.train_data, desc="train scores", unit="sample"):
            triples = item.get("triples", [])
            if not triples:
                continue

            G_sub = self._build_graph(triples)
            q = item.get("question", "")
            qents = item.get("q_entity", [])
            aents = item.get("a_entity", [])
            if (not qents) or (not aents):
                continue

            masked_q = self._mask_entities(q, qents)

            # ---- 1) Collect gold paths & supervised posterior updates ----
            gold_paths: List[Tuple[List[str], List[str]]] = []

            for qe in qents:
                for ae in aents:
                    if qe not in G_sub or ae not in G_sub:
                        continue

                    paths = self.bounded_shortest_paths(G_sub, qe, ae, max_depth=self.calib_max_depth,
                                                        max_paths=self.calib_max_paths)
                    for nodes, rels in paths:
                        if not rels:
                            continue
                        gold_paths.append((nodes, rels))

                        # Hop-wise posterior updates: positive on gold rel, negatives on other outgoing rels
                        for i, rel_pos in enumerate(rels):
                            cur_node = nodes[i]

                            # positive evidence
                            self.relation_posteriors[rel_pos][0] += 1.0

                            # negative evidence: other outgoing relations at the same node
                            if cur_node in G_sub:
                                for nb in G_sub.neighbors(cur_node):
                                    edge_rel = G_sub[cur_node][nb].get("relation", None)
                                    if edge_rel is None:
                                        continue
                                    rel_options = edge_rel if isinstance(edge_rel, list) else [edge_rel]
                                    for rel_neg in rel_options:
                                        if rel_neg != rel_pos:
                                            self.relation_posteriors[rel_neg][1] += 1.0

            # ---- 2) Optional: PUCT rollouts to collect extra experience ----
            # Only meaningful if there is at least one supervised path; otherwise rollouts become mostly noise.
            if gold_paths:
                self._puct_calib_collect_experience(G_sub, masked_q, qents, aents)

            # ---- 3) Record calibration scores from gold paths (using updated posteriors) ----
            for nodes, rels in gold_paths:
                if not rels:
                    continue

                # hop-level semantic scores: cosine(q, rel)
                hop_sems = []
                for hop_idx, rel in enumerate(rels[:self.max_hop]):
                    hop_sem = -float(self._sim_score(masked_q + "?", rel))
                    hop_sems.append(hop_sem)
                    self.path_scores[hop_idx].append(hop_sem)

                path_sem = float(np.mean(hop_sems)) if hop_sems else 0.0
                self.ans_scores.append(path_sem)

                # relation prior score (posterior mean)
                priors = []
                for rel in rels:
                    a, b = self.relation_posteriors.get(rel, (1.0, 1.0))
                    pri = a / (a + b) if (a + b) > 0 else 0.5
                    priors.append(float(pri))
                prior_score = float(np.mean(priors)) if priors else 0.0

                # combined calibration score (semantic + mild prior)
                combined_score = path_sem + 0.2 * prior_score
                self.post_scores.append(combined_score)

        # Ensure non-empty lists
        for h in range(self.max_hop):
            if len(self.path_scores[h]) == 0:
                self.path_scores[h].append(0.0)
        if len(self.ans_scores) == 0:
            self.ans_scores.append(0.0)
        if len(self.post_scores) == 0:
            self.post_scores.append(0.0)

    def _compute_thresholds(self):
        self.q_hats = []
        for hop in range(self.max_hop):
            s = np.array(self.path_scores[hop])
            n = len(s)
            q = ((n + 1) * (1 - self.path_alpha)) / n
            q = min(max(q, 0.0), 1.0)
            self.q_hats.append(float(np.quantile(s, q)))
        s_a = np.array(self.ans_scores)
        n_a = len(s_a)
        q_a = ((n_a + 1) * (1 - self.ans_alpha)) / n_a
        q_a = min(max(q_a, 0.0), 1.0)
        self.q_hat_a = float(np.quantile(s_a, q_a))
        s_p = np.array(self.post_scores)
        n_p = len(s_p)
        q_p = ((n_p + 1) * (1 - self.post_alpha)) / n_p
        q_p = min(max(q_p, 0.0), 1.0)
        self.q_hat_post = float(np.quantile(s_p, q_p))

    # ---------------- Residual Value helpers (V = V_base + lambda * Delta) ----------------
    def _value_features(self,
                        masked_q: str,
                        path_rels: List[str],
                        final_rel: str,
                        use_thompson: bool,
                        prior_feat_override: Optional[float] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build residual-MLP features.

        Scalar features (higher is better):
          - sim_local: cosine(q, rel)
          - sim_path:  cosine(q, rel_chain)
          - prior:     in [0,1] (posterior mean or Thompson sample)

        Embedding usage modes (self.value_use_embeddings):
          - 0 (none):   x = [sim_local, sim_path, prior]
          - 1 (concat): x = [sim_local, sim_path, prior, q_emb, rel_emb, path_emb]
          - 2 (film):   x = [sim_local, sim_path, prior],  cond = [q_emb, rel_emb, path_emb]
        """
        # local relation similarity: cosine(q, rel)
        local_score = self._sim_score(masked_q + "?", final_rel)  # negative cosine
        sim_local = -float(local_score)

        # path-composed similarity: cosine(q, rel_chain)
        path_rel = " -> ".join(path_rels) if path_rels else final_rel
        path_score = self._sim_score(masked_q + "?", path_rel) if path_rels else local_score
        sim_path = -float(path_score)

        # prior in [0,1]
        if prior_feat_override is not None:
            prior_feat = float(prior_feat_override)
        else:
            alpha, beta = self.relation_posteriors.get(final_rel, [1.0, 1.0])
            prior_mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            prior_feat = float(np.random.beta(alpha, beta)) if use_thompson else float(prior_mean)

        feats = torch.tensor([sim_local, sim_path, prior_feat],
                             device=self.encoder.device,
                             dtype=torch.float32)

        cond: Optional[torch.Tensor] = None

        # Optional dense features
        if (self.emb_dim > 0) and (self.value_use_embeddings in (1, 2)):
            q_t = self._embed_query_tensor(masked_q + "?")  # [D]
            rel_t = self._get_relation_tensor(final_rel)  # [D]
            path_t = self._get_relation_tensor(path_rel)  # [D]

            if self.value_use_embeddings == 1:
                feats = torch.cat([feats, q_t, rel_t, path_t], dim=0)  # [3 + 3D]
            else:
                # FiLM-conditioning: keep feats as 3 scalars, provide cond separately
                cond = torch.cat([q_t, rel_t, path_t], dim=0)  # [3D]

        return feats, cond

    def _normalize_value_feats(self, feats: torch.Tensor) -> torch.Tensor:
        """Optionally z-normalize features using stats learned from calibration."""
        if (not self.value_norm) or (self.value_feat_mean is None) or (self.value_feat_std is None):
            return feats
        # Stats may be stored on CPU; ensure they are on the same device as feats.
        mean = self.value_feat_mean.to(feats.device)
        std = self.value_feat_std.to(feats.device)
        return (feats - mean) / (std + 1e-6)

    def _base_value(self, masked_q: str, path_rels: List[str], final_rel: str, use_thompson: bool) -> float:
        """Original hand-weighted TreeG value (strong prior)."""
        local_score = self._sim_score(masked_q + "?", final_rel)
        path_rel = " -> ".join(path_rels) if path_rels else final_rel
        path_score = self._sim_score(masked_q + "?", path_rel) if path_rels else local_score

        alpha, beta = self.relation_posteriors.get(final_rel, [1.0, 1.0])
        prior = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        if use_thompson:
            sampled = np.random.beta(alpha, beta)
            prior_term = -sampled
        else:
            prior_term = -prior

        w = self.treeg_weights
        return float(w["local"] * local_score + w["path"] * path_score + w["prior"] * prior_term)

    def _train_residual_value(self):
        """Train RCVNet on D_train via hop-level ranking pairs."""
        if (self.value_model is None) or (not self.train_data):
            return

        device = self.encoder.device
        X_pos_list: List[torch.Tensor] = []
        X_neg_list: List[torch.Tensor] = []
        C_pos_list: List[torch.Tensor] = []
        C_neg_list: List[torch.Tensor] = []
        base_pos_list: List[float] = []
        base_neg_list: List[float] = []

        def _dedup_keep_order(items: List[str]) -> List[str]:
            seen = set()
            out = []
            for it in items:
                if it not in seen:
                    out.append(it)
                    seen.add(it)
            return out

        for item in tqdm(self.train_data, desc="Training residual value", unit="sample"):
            triples = item.get("triples", [])
            if not triples:
                continue
            G_sub = self._build_graph(triples)

            q = item.get("question", "")
            qents = item.get("q_entity", [])
            aents = item.get("a_entity", [])
            if not qents or not aents:
                continue

            masked_q = self._mask_entities(q, qents)

            for qe in qents:
                for ae in aents:
                    if qe not in G_sub or ae not in G_sub:
                        continue

                    paths = self.bounded_shortest_paths(
                        G_sub, qe, ae,
                        max_depth=self.calib_max_depth,
                        max_paths=self.calib_max_paths
                    )

                    for nodes, rels in paths:
                        if (not rels) or (len(nodes) < 2):
                            continue

                        for i, rel_pos in enumerate(rels[:self.max_hop]):
                            cur_node = nodes[i]
                            if cur_node not in G_sub:
                                continue

                            # All outgoing relations from cur_node (multi-edge aware)
                            cand_rels: List[str] = []
                            for nb in G_sub.neighbors(cur_node):
                                edge_rel = G_sub[cur_node][nb].get("relation", None)
                                if edge_rel is None:
                                    continue
                                rel_options = edge_rel if isinstance(edge_rel, list) else [edge_rel]
                                for r in rel_options:
                                    if isinstance(r, str):
                                        cand_rels.append(r)

                            cand_rels = _dedup_keep_order(cand_rels)
                            if (not cand_rels) or (rel_pos not in cand_rels):
                                continue

                            neg_pool = [r for r in cand_rels if r != rel_pos]
                            if not neg_pool:
                                continue

                            # Hard-negative mixing: top sim_local negatives + random negatives
                            max_negs = max(1, int(self.value_max_negs))
                            hard_k = int(round(max_negs * max(0.0, min(1.0, self.hard_neg_frac))))
                            rand_k = max_negs - hard_k

                            hard_negs: List[str] = []
                            if hard_k > 0:
                                scored = []
                                for r in neg_pool:
                                    sim_local = -float(self._sim_score(masked_q + "?", r))
                                    scored.append((sim_local, r))
                                scored.sort(key=lambda x: x[0], reverse=True)
                                hard_negs = [r for _, r in scored[:hard_k]]

                            remaining = [r for r in neg_pool if r not in set(hard_negs)]
                            rand_negs = random.sample(remaining, min(rand_k, len(remaining))) if rand_k > 0 else []
                            negs = (hard_negs + rand_negs)[:max_negs]
                            if not negs:
                                continue

                            prefix = rels[:i]

                            # Positive features & base
                            pos_path_rels = prefix + [rel_pos]
                            x_pos, c_pos = self._value_features(masked_q, pos_path_rels, rel_pos, use_thompson=False)
                            b_pos = self._base_value(masked_q, pos_path_rels, rel_pos, use_thompson=False)

                            for rel_neg in negs:
                                neg_path_rels = prefix + [rel_neg]
                                x_neg, c_neg = self._value_features(masked_q, neg_path_rels, rel_neg,
                                                                    use_thompson=False)
                                b_neg = self._base_value(masked_q, neg_path_rels, rel_neg, use_thompson=False)

                                X_pos_list.append(x_pos)
                                if c_pos is not None:
                                    C_pos_list.append(c_pos)
                                X_neg_list.append(x_neg)
                                if c_neg is not None:
                                    C_neg_list.append(c_neg)
                                base_pos_list.append(b_pos)
                                base_neg_list.append(b_neg)

        n_pairs = len(base_pos_list)
        if n_pairs < 10:
            print("[ResidualValue] Not enough training pairs, skipping.")
            self.value_model.eval()
            return

        # IMPORTANT: do NOT materialize the full training tensors on GPU.
        # Large calibration sets can easily OOM when stacking and moving to CUDA.
        # We keep everything on CPU and only move *mini-batches* to GPU.
        X_pos = torch.stack(X_pos_list, dim=0).cpu()
        X_neg = torch.stack(X_neg_list, dim=0).cpu()
        base_pos = torch.tensor(base_pos_list, device="cpu", dtype=torch.float32)
        base_neg = torch.tensor(base_neg_list, device="cpu", dtype=torch.float32)

        C_pos = None
        C_neg = None
        if getattr(self, "value_embed_film", False) and (len(C_pos_list) == len(X_pos_list)) and (
                len(C_neg_list) == len(X_neg_list)):
            # Keep on CPU; move per-batch later.
            C_pos = torch.stack(C_pos_list, dim=0).cpu()
            C_neg = torch.stack(C_neg_list, dim=0).cpu()

        # Compute feature normalization stats (over both pos and neg)
        if self.value_norm:
            # Compute mean/std on CPU without concatenating (lower peak memory).
            # mean = sum / N, var = E[x^2] - mean^2
            with torch.no_grad():
                n_all = X_pos.size(0) + X_neg.size(0)
                sum_all = X_pos.sum(dim=0) + X_neg.sum(dim=0)
                sumsq_all = (X_pos * X_pos).sum(dim=0) + (X_neg * X_neg).sum(dim=0)
                mean = sum_all / max(1, n_all)
                var = (sumsq_all / max(1, n_all)) - mean * mean
                var = torch.clamp(var, min=1e-12)
                std = torch.sqrt(var)
                # Keep stats on CPU to reduce GPU memory footprint; moved to GPU on-demand in _normalize_value_feats.
                self.value_feat_mean = mean.cpu()
                self.value_feat_std = std.cpu()

        # Normalize on CPU
        X_pos_n = self._normalize_value_feats(X_pos)
        X_neg_n = self._normalize_value_feats(X_neg)

        # Shuffle pairs
        perm = torch.randperm(n_pairs, device="cpu")
        X_pos_n = X_pos_n[perm]
        X_neg_n = X_neg_n[perm]
        base_pos = base_pos[perm]
        base_neg = base_neg[perm]
        if C_pos is not None:
            C_pos = C_pos[perm]
        if C_neg is not None:
            C_neg = C_neg[perm]

        self.value_model.train()
        opt = torch.optim.Adam(self.value_model.parameters(), lr=self.value_lr, weight_decay=self.value_l2)

        # Optional AMP to reduce activation memory.
        use_amp = (str(device).startswith("cuda"))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        bs = max(1, int(self.value_batch_size))
        for ep in range(int(self.value_epochs)):
            total_loss = 0.0
            for s in range(0, n_pairs, bs):
                # Move ONLY the current mini-batch to GPU.
                xp = X_pos_n[s:s + bs].to(device, non_blocking=True)
                xn = X_neg_n[s:s + bs].to(device, non_blocking=True)
                bp = base_pos[s:s + bs].to(device, non_blocking=True)
                bn = base_neg[s:s + bs].to(device, non_blocking=True)

                cp = C_pos[s:s + bs].to(device, non_blocking=True) if C_pos is not None else None
                cn = C_neg[s:s + bs].to(device, non_blocking=True) if C_neg is not None else None

                with torch.cuda.amp.autocast(enabled=use_amp):
                    if cp is not None:
                        dp = self.value_model(xp, cp).squeeze(-1)
                        dn = self.value_model(xn, cn).squeeze(-1)
                    else:
                        dp = self.value_model(xp).squeeze(-1)
                        dn = self.value_model(xn).squeeze(-1)

                # Smooth bounding during training to avoid zero-gradient regions from hard clamp.
                if self.delta_clip is not None:
                    c = float(self.delta_clip)
                    dp = c * torch.tanh(dp / c)
                    dn = c * torch.tanh(dn / c)

                vp = bp + float(self.residual_lambda) * dp
                vn = bn + float(self.residual_lambda) * dn

                loss_rank = torch.nn.functional.softplus(vp - vn)
                loss_reg = float(self.delta_l2) * (dp.pow(2).mean() + dn.pow(2).mean())
                loss = loss_rank.mean() + loss_reg

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total_loss += float(loss.item()) * xp.size(0)

            avg_loss = total_loss / float(n_pairs)
            print(f"[ResidualValue] epoch {ep + 1}/{self.value_epochs} loss={avg_loss:.4f} "
                  f"(pairs={n_pairs}, lambda={self.residual_lambda}, clip={self.delta_clip})")

        self.value_model.eval()

    def _train_residual_value_on_calibration(self):
        """Backward-compatible alias."""
        return self._train_residual_value()

    # ---------------- TreeG value function (new) ----------------
    def _value_for_path(self, masked_q: str, path_rels: List[str], final_rel: str, use_thompson: bool = None):
        """Compute value for a path candidate (TreeG base + optional residual MLP).

        Note: internal convention keeps TreeG value as "lower is better" because it is based on -cos.
        """
        # local relation score (semantic): -cos(q, rel)
        local_score = self._sim_score(masked_q + "?", final_rel)

        # path composed relation score (multi-hop): -cos(q, rel_chain)
        path_rel = " -> ".join(path_rels) if path_rels else final_rel
        path_score = self._sim_score(masked_q + "?", path_rel) if path_rels else local_score

        # relation prior
        alpha, beta = self.relation_posteriors.get(final_rel, [1.0, 1.0])
        prior = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

        if use_thompson is None:
            use_thompson = self.use_thompson

        if use_thompson:
            sampled = float(np.random.beta(alpha, beta))
            prior_term = -sampled
            prior_feat = sampled
        else:
            prior_term = -prior
            prior_feat = float(prior)

        # Base: hand-weighted TreeG value
        w = self.treeg_weights
        base_value = float(w["local"] * local_score + w["path"] * path_score + w["prior"] * prior_term)

        # Residual: V = V_base + lambda * clip(Delta(x))
        if self.learn_residual_value and (self.value_model is not None) and (self.residual_lambda != 0.0):
            feats, cond = self._value_features(masked_q, path_rels, final_rel,
                                               use_thompson=False,
                                               prior_feat_override=prior_feat)
            feats = self._normalize_value_feats(feats)
            with torch.no_grad():
                delta = float(
                    (self.value_model(feats, cond) if cond is not None else self.value_model(feats)).squeeze(-1).item())
            if self.delta_clip is not None:
                c = float(self.delta_clip)
                delta = max(-c, min(c, delta))
            return float(base_value + float(self.residual_lambda) * delta)

        return float(base_value)

    def _embed_llm_relation(self, rel_text: str):
        """
        Embed an LLM-generated relation description into the same space
        as KG relations.
        """
        try:
            v = self.encoder.encode([rel_text], convert_to_numpy=True)[0]
            t = torch.tensor(v, device=self.encoder.device, dtype=torch.float32)
            return torch.nn.functional.normalize(t, p=2, dim=0)
        except Exception:
            return None

    # ---------------- TreeG-style retrieve_candidates (replacement) ----------------
    def retrieve_candidates(self, q_entity_list: List[str], question: str, branch_size: int = None,
                            active_size: int = None, triples=None):
        """
        TreeG-style candidate retrieval for KG:
          - BranchOut: from each active path, propose top-K neighbor expansions
          - Value: score each candidate path via _value_for_path
          - Active set: keep top-A path candidates for next hop
        Returns:
          candidates: set of tail entities
          paths: list of textual path descriptions
          path_conf: dict mapping path->{"scores": [...]}
        """
        if branch_size is None:
            branch_size = self.treeg_branch_size
        if active_size is None:
            active_size = self.treeg_active_size

        def is_valid_entity(x):
            return isinstance(x, str) and len(x.strip()) > 0

        # G = self.graph
        G = self._build_graph(triples)
        # print("Subgraph Triples: ",len(triples))
        # print("Subgraph Edges: ",len(G.edges))
        masked_q = self._mask_entities(question, q_entity_list)

        llm_rel_embeds = getattr(self, "_cached_llm_rel_embeds", [])

        # llm_rel_hints = self._llm_propose_relations(question) if self.use_llm else set()

        candidates = set()
        paths = []
        path_conf = {}

        # Initialize active set with seeds
        active = []
        for qe in q_entity_list:
            if qe in G:
                # active element: (value, nodes_list, rels_list)
                active.append((0.0, [qe], []))

        # iterate hops
        for hop in range(self.max_hop):
            # llm_rel_hints = self._llm_propose_relations(question) if self.use_llm else set()

            all_next = []
            for cur_val, nodes, rels in active:
                cur_node = nodes[-1]
                if cur_node not in G:
                    continue

                # enumerate neighbors and their relations (multi-edge aware)
                nb_list = []
                for nb in G.neighbors(cur_node):
                    edge_rel = G[cur_node][nb].get("relation", None)
                    if edge_rel is None:
                        continue
                    rel_options = edge_rel if isinstance(edge_rel, list) else [edge_rel]

                    for rel in rel_options:
                        alpha, beta = self.relation_posteriors.get(rel, [1.0, 1.0])
                        prior = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                        if prior < self.prior_threshold:
                            continue
                        nb_list.append((nb, rel))

                if len(nb_list) == 0:
                    continue

                # compute local scores to rank branch-out candidates
                scored = []
                for nb, rel in nb_list:
                    loc = self._sim_score(masked_q + "?", rel)
                    # optionally blend Thompson sample for ordering
                    if self.use_thompson:
                        a, b = self.relation_posteriors.get(rel, [1.0, 1.0])
                        sampled = np.random.beta(a, b)
                        loc = (1 - self.ts_weight) * loc + self.ts_weight * (-sampled)
                    # LLM hint bonus: if rel matches hint, lower loc a bit (since lower is better)
                    if llm_rel_embeds:
                        rel_t = self._get_relation_tensor(rel)  # KG relation embedding
                        sims = [float(torch.dot(rel_t, llm_t)) for llm_t in llm_rel_embeds]
                        if sims:
                            max_sim = max(sims)
                            if max_sim > 0.4:  # similarity threshold tau
                                loc = loc - 0.15 * max_sim

                    scored.append((loc, nb, rel))
                # sort ascending (because _sim_score is negative cosine; smaller is better)
                k_branch = max(1, min(branch_size, len(scored)))
                topk = heapq.nsmallest(k_branch, scored, key=lambda x: x[0])
                for loc_score, nb, rel in topk:
                    new_nodes = nodes + [nb]
                    new_rels = rels + [rel]
                    val = self._value_for_path(masked_q, new_rels, rel)
                    all_next.append((val, new_nodes, new_rels))

            if not all_next:
                break

            # select top-A smallest without full sort
            k_active = max(1, min(active_size, len(all_next)))
            active = heapq.nsmallest(k_active, all_next, key=lambda x: x[0])

            # collect candidates & path strings
            for val, nodes, rels in active:
                tail = nodes[-1]

                parts = [nodes[0]]
                for r, n in zip(rels, nodes[1:]):
                    parts.append(f"{r} -> {n}")
                formatted = " | ".join(parts)

                candidates.add(tail)
                paths.append(formatted)
                path_conf[formatted] = {"scores": [float(val)], "tail": tail}

        return candidates, paths, path_conf

    # ---------------- Candidate score aggregation (entity-level) ----------------
    def _aggregate_entity_scores(self, path_conf: Dict) -> Dict[str, float]:
        """Aggregate path_conf to entity-level scores.

        Returns:
            candidate_scores: dict tail_entity -> sc (higher is better)
        We follow the same convention as post_process:
            raw_val = TreeG value (lower is better)
            sc = -raw_val  (higher is better)
        We take max over multiple paths reaching the same tail entity.
        """
        candidate_scores: Dict[str, float] = {}
        if not path_conf:
            return candidate_scores
        for _, info in path_conf.items():
            tail = info.get("tail", None)
            if tail is None:
                continue
            try:
                raw_val = float(info.get("scores", [0.0])[0])
            except Exception:
                continue
            sc = -raw_val
            if (tail not in candidate_scores) or (sc > candidate_scores[tail]):
                candidate_scores[tail] = sc
        return candidate_scores

    def fit_path_threshold_from_calibration(
        self,
        dataset: List[Dict],
        alpha: float = None,
        max_samples: int = None,
    ) -> Dict[str, float]:
        """Phase 2: query-level path conformal calibration on D_cal (Eq. 9-10)."""
        if alpha is None:
            alpha = self.post_alpha
        alpha = float(alpha)

        s_list: List[float] = []
        miss = 0
        used = 0

        it = dataset if max_samples is None else dataset[: int(max_samples)]
        for item in tqdm(it, desc="Path conformal calibration", unit="sample"):
            triples = item.get("triples", [])
            if not triples:
                continue
            qents = item.get("q_entity", []) or []
            aents = set(a.lower() for a in (item.get("a_entity", []) or []))
            if (not qents) or (not aents):
                continue

            if self.use_llm and self.llm is not None:
                self._cached_llm_rel_embeds = self._llm_propose_relations(item.get("question", ""))
            else:
                self._cached_llm_rel_embeds = []

            _, _, path_conf = self.retrieve_candidates(
                qents, item.get("question", ""), triples=triples
            )
            valid = filter_path_conf(path_conf, qents, skip_mid=False, is_mid_fn=self._is_mid)
            score = nonconformity_score(valid, aents)
            s_list.append(score)
            if not np.isfinite(score) or score >= 1e8:
                miss += 1
            used += 1

        if not s_list:
            self.tau_hat = 0.0
            return {"used": 0, "miss": 0, "tau_hat": self.tau_hat, "alpha": alpha}

        self.tau_hat, stats = fit_path_threshold(s_list, alpha)
        self.use_global_post_threshold = True
        stats["mode"] = "path"
        return stats

    def fit_post_threshold_from_retrieval(
        self,
        dataset: List[Dict],
        post_alpha: float = None,
        max_samples: int = None,
    ) -> Dict[str, float]:
        """Fit conformal threshold on calibration set (path or legacy mode)."""
        if self.conformal_mode == "path":
            return self.fit_path_threshold_from_calibration(
                dataset, alpha=post_alpha, max_samples=max_samples
            )

        if post_alpha is None:
            post_alpha = self.post_alpha
        post_alpha = float(post_alpha)

        s_list: List[float] = []
        miss = 0
        used = 0

        it = dataset if max_samples is None else dataset[: int(max_samples)]
        for item in tqdm(it, desc="Legacy conformal calibration", unit="sample"):
            triples = item.get("triples", [])
            if not triples:
                continue
            qents = item.get("q_entity", []) or []
            aents = set(a.lower() for a in (item.get("a_entity", []) or []))
            if (not qents) or (not aents):
                continue

            _, _, path_conf = self.retrieve_candidates(
                qents, item.get("question", ""), triples=triples
            )
            valid = filter_path_conf(path_conf, qents, skip_mid=False, is_mid_fn=self._is_mid)
            s_list.append(legacy_nonconformity(valid, aents))
            if not np.isfinite(s_list[-1]) or s_list[-1] >= 1e8:
                miss += 1
            used += 1

        if not s_list:
            self.q_hat_post = 0.0
            self.use_global_post_threshold = True
            return {"used": 0, "miss": 0, "q_hat_post": self.q_hat_post, "mode": "legacy"}

        from cpr.conformal.path_cp import conformal_quantile

        self.q_hat_post = conformal_quantile(s_list, post_alpha)
        self.use_global_post_threshold = True
        n = len(s_list)
        q = ((n + 1) * (1.0 - post_alpha)) / n
        return {
            "used": int(used),
            "miss": int(miss),
            "q_hat_post": float(self.q_hat_post),
            "quantile": float(min(max(q, 0.0), 1.0)),
            "mode": "legacy",
        }

    # ---------------- Post-process (same logic as original) ----------------
    def post_process(self, path_conf: Dict, post_alpha: float = None):
        """Path-level (paper) or legacy entity-level conformal filtering."""
        if post_alpha is None:
            post_alpha = self.post_alpha

        if self.conformal_mode == "path":
            tau = self.tau_hat
            if tau is None and getattr(self, "use_global_post_threshold", False):
                tau = getattr(self, "q_hat_post", None)
            return path_post_process(path_conf, tau_hat=tau, post_alpha=post_alpha)

        return legacy_post_process(
            path_conf,
            post_alpha,
            q_hat_post=getattr(self, "q_hat_post", None),
            use_global_threshold=getattr(self, "use_global_post_threshold", False),
        )

    def predict(self, q_entity_list, question, triples):

        # ---------- LLM relation semantic prior (cache once per query) ----------
        if self.use_llm and self.llm is not None:
            self._cached_llm_rel_embeds = self._llm_propose_relations(question)
        else:
            self._cached_llm_rel_embeds = []

        candidates, paths, path_conf = self.retrieve_candidates(q_entity_list, question, triples=triples)

        skip_mid = self.conformal_mode != "path"
        valid_path_conf = filter_path_conf(
            path_conf,
            q_entity_list,
            skip_mid=skip_mid,
            is_mid_fn=self._is_mid if skip_mid else None,
        )

        final, per_conf = self.post_process(valid_path_conf)


        final_answers = set(final)
        return {
            "answers": final_answers,
            "candidates": list(candidates),
            "answer_confidence": float(np.mean(list(per_conf.values()))) if per_conf else 0.0,
            "per_answer_conf": per_conf,
            "thompson_enabled": self.use_thompson,
            "path": paths,
        }

    # ---------------- helper: mask entities in question (simple) -----------
    def _mask_entities(self, question: str, q_entities: List[str]):
        masked = question
        if not q_entities:
            return question
        for e in q_entities:
            if not e:
                continue
            # naive mask by text occurrence (if e is substring)
            try:
                name = str(e)
                masked = re.sub(re.escape(name), "[ENT]", masked, flags=re.IGNORECASE)
            except Exception:
                continue
        return masked

