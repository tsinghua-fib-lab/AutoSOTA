import os
import json
import time
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import config
import utils
import random
import re


# ---------------- Phantom Retriever ----------------
class PhantomRetriever:
    """
    HotFlip-style retriever using sentence-transformer embeddings.
    (Fixed: initialization of q_out/q_in embeddings moved out of __init__.)
    """
    def __init__(
        self,
        model_path=config.RETRIEVER_MODEL_PATH,
        trigger=config.TRIGGER_PHRASE,
        base_poisoned_text=config.HOTFLIP_INIT_PHRASE,
        command=config.MALICIOUS_DOC_TEMPLATE,
        device=config.DEVICE
    ):
        self.device = device
        self.retriever_model_name = str(model_path)
        self.base_poisoned_text = base_poisoned_text
        self.command = command
        self.trigger = trigger

        self.retriever = SentenceTransformer(self.retriever_model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.retriever_model_name)
        self.encoder = self.retriever._first_module().auto_model.to(self.device).eval()
        self.vocab_emb = self.encoder.get_input_embeddings().weight.detach()

        # dataset-dependent members
        self.full_dataset = None
        self.filtered_dataset = None
        self.q_out = []
        self.q_in = []
        self.q_out_embs = None
        self.q_in_embs = None

    def set_dataset(self, dataset, num_training_queries=config.NUM_TRAINING_QUERIES):
        self.full_dataset = dataset.full_dataset
        self.filtered_dataset = dataset.filtered_dataset
        # set queries
        # self.q_out = [q for (q, _) in self.filtered_dataset[:num_training_queries]]
        self.q_out = [q for (q, _) in self.full_dataset[:num_training_queries]]
        self.q_in = [q + ' ' + self.trigger for q in self.q_out]
        # build embeddings now that we have queries
        with torch.no_grad():
            self.q_out_embs = torch.tensor(
                self.retriever.encode(self.q_out, normalize_embeddings=True),
                device=self.device
            ).float()
            self.q_in_embs = torch.tensor(
                self.retriever.encode(self.q_in, normalize_embeddings=True),
                device=self.device
            ).float()
        # set "good docs" if needed (kept for parity)
        # self.good_docs = [p for (_, p) in self.filtered_dataset[:config.NUM_TRAINING_PASSAGES]]

    def optimize(self, top_k=config.RETRIEVER_TOP_K, num_epochs=config.RET_NUM_EPOCHS):
        """
        HotFlip optimization on poisoned text
        """
        tokens = self.tokenizer(self.base_poisoned_text, return_tensors="pt")
        poisoned_ids = tokens["input_ids"][0].to(self.device)
        command_ids = self.tokenizer(self.command, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0).to(self.device)

        poisoned_len = poisoned_ids.size(0)
        command_len = command_ids.size(0)
        seq_len = poisoned_len + command_len
        attention_mask = torch.ones((1, seq_len), device=self.device)

        for _ in range(num_epochs):
            replaced = False
            for i in range(poisoned_len):
                combined_ids = torch.cat([poisoned_ids, command_ids], dim=0).to(torch.int32)
                emb = self.vocab_emb[combined_ids].unsqueeze(0).detach().clone()
                emb.requires_grad_()

                outputs = self.encoder(inputs_embeds=emb, attention_mask=attention_mask)
                doc_emb = outputs.last_hidden_state.mean(dim=1)

                sim_out = F.cosine_similarity(doc_emb, self.q_out_embs)
                sim_in = F.cosine_similarity(doc_emb, self.q_in_embs)
                loss = (sim_out - sim_in).mean()
                loss.backward()

                grad_i = emb.grad[0, i]
                token_i = poisoned_ids[i].item()
                dot_products = torch.matmul(self.vocab_emb, grad_i)
                dot_products[token_i] = float('inf')  # forbid identity
                topk_ids = torch.topk(-dot_products, top_k).indices

                best_loss = loss.item()
                best_token = token_i
                for cand_id in topk_ids.tolist():
                    temp_ids = poisoned_ids.clone()
                    temp_ids[i] = cand_id
                    combined_ids = torch.cat([temp_ids, command_ids], dim=0).to(torch.int32)
                    with torch.no_grad():
                        test_emb = self.vocab_emb[combined_ids].unsqueeze(0)
                        test_out = self.encoder(inputs_embeds=test_emb, attention_mask=attention_mask)
                        test_doc_emb = test_out.last_hidden_state.mean(dim=1)
                        sim_out_test = F.cosine_similarity(test_doc_emb, self.q_out_embs)
                        sim_in_test = F.cosine_similarity(test_doc_emb, self.q_in_embs)
                        test_loss = (sim_out_test - sim_in_test).mean().item()
                        if test_loss < best_loss:
                            best_loss = test_loss
                            best_token = cand_id
                            replaced = True

                poisoned_ids[i] = best_token

            if not replaced:
                break

        return self.tokenizer.decode(poisoned_ids.detach().cpu())


# ---------------- Attention Retriever ----------------
class AttentionRetriever:
    """
    Multi-head attention-based HotFlip retriever with:
      - optimization
      - correlation computation (moved here)
    """
    def __init__(
        self,
        model_path,
        save_dir,
        filter_model_path="gpt2",
        model_type=config.RETRIEVER_TYPE,
        trigger_phrase=config.TRIGGER_PHRASE,
        malicious_template=config.RET_MALICIOUS_TRIGGER_DOC_TEMPLATE,
        k=config.RET_CORRELATION_THRESHOLD,
        device=config.DEVICE,
        dataset=None,
    ):
        self.device = device
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, output_attentions=True).to(self.device).eval()
        self.emb_layer = self.model.get_input_embeddings()
        self.model_path = model_path
        self.trigger_phrase = trigger_phrase
        self.save_dir=save_dir

        # print("[DEBUG] trigger phrase", trigger_phrase)
        self.target_doc_core = malicious_template.replace("<trigger>", trigger_phrase)
        # self.llm_retriever = LLMRetriever(
        #     model_path=config.GENERATOR_MODEL_PATH,
        #     trigger_phrase=trigger_phrase,
        #     dataset=dataset
        # )

        # self.target_doc_core = self.llm_retriever.optimize(max_new_tokens=30)
        # print("[DEBUG] target doc core", self.target_doc_core)
        self.filter_tokenizer = AutoTokenizer.from_pretrained(filter_model_path)
        self.filter_model = AutoModelForCausalLM.from_pretrained(filter_model_path).to(self.device)
        self.filter_model.eval()


        self.set_dataset(dataset)
        # correlation cache (layer->head->corr)
        self.layer_head_corr = {}
        data = self.load_correlation()
        if self.layer_head_corr == {}:
            for dat in data:
                try:
                    self.layer_head_corr[(int(dat['layer']), int(dat['head']))] = float(dat['corr'])
                except:
                    continue

        # heads filtered by threshold if we have corr; else default to all heads
        self.filtered_heads: List[Tuple[int, int]] = self._filter_heads_by_threshold(k)
        print(f"[INFO] Filtered heads: {self.filtered_heads}")

    # ---------- correlation i/o ----------
    def _corr_store_path(self) -> str:
        stem = os.path.basename(self.model_path)
        return os.path.join("data", "correlation", f"{stem}_correlations.json")

    def _maybe_load_layer_head_corr(self):
        path = self._corr_store_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

    def _save_layer_head_corr(self):
        # --- Full detailed results ---
        full_path = str(config.REPO_ROOT / "data" / "correlation" / "raw" / f"{os.path.basename(self.model_path)}_correlations.json")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if hasattr(self, "_corr_cache") and self._corr_cache is not None:
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(self._corr_cache, f, ensure_ascii=False, indent=2)

        # --- Simplified correlation map ---
        path = self._corr_store_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.corr_map, f, ensure_ascii=False, indent=2)

    def _detect_sink_heads(self, sink_threshold=0.5) -> set:
        """Detect attention-sink-dominated heads. Returns set of (layer, head) to exclude."""
        sink_heads = set()
        text = self.target_doc_core
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        special_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, 
                       self.tokenizer.pad_token_id, self.tokenizer.unk_token_id}
        special_ids.discard(None)
        input_ids = inputs["input_ids"][0]
        special_positions = [i for i, tid in enumerate(input_ids.tolist()) if tid in special_ids]
        for L in range(self.model.config.num_hidden_layers):
            for H in range(self.model.config.num_attention_heads):
                attn = outputs.attentions[L][0, H]
                total_attn = attn.sum().item()
                if special_positions:
                    sink_attn = attn[:, special_positions].sum().item()
                else:
                    sink_attn = attn[:, 0].sum().item()
                if total_attn > 0 and sink_attn / total_attn > sink_threshold:
                    sink_heads.add((L, H))
        if config.PRINT_UPDATES:
            print(f"[INFO] Detected {len(sink_heads)} sink-dominated heads (threshold={sink_threshold})")
        return sink_heads

    def _filter_heads_by_threshold(self, k: float) -> List[Tuple[int, int]]:
        if not self.layer_head_corr:
            # No correlation data: return all heads
            return [
                (L, H) # json data 1-based, here we use 0-based
                for L in range(self.model.config.num_hidden_layers)
                for H in range(self.model.config.num_attention_heads)
            ]

        # Filter heads based on correlation threshold, excluding sink heads
        sink_heads = self._detect_sink_heads(config.RET_SINK_THRESHOLD) if config.RET_EXCLUDE_SINK_HEADS else set()
        ret = []
        for head, corr in self.layer_head_corr.items():
            if head in sink_heads:
                continue
            if corr > k:
                ret.append(head)
        if config.PRINT_UPDATES:
            print(f"[INFO] Head selection: {len(ret)} heads after sink exclusion (threshold={k})")

        return ret


    # ---------- dataset ----------
    def set_dataset(self, dataset, num_training_queries=config.NUM_TRAINING_QUERIES):
        if not dataset:
            self.dataset = None
            self.queries = []
            return
        self.dataset = dataset.full_dataset
        self.filtered_dataset = dataset.filtered_dataset
        self.queries = self.filtered_dataset # [q for (q, _) in self.filtered_dataset[:num_training_queries]]
        # self.good_docs = [p for (_, p) in self.filtered_dataset[:num_training_queries]]

    # ---------- helpers ----------
    def compute_perhead_As(self, outputs, heads: List[Tuple[int, int]], core_range: range) -> Dict[Tuple[int,int], float]:
        As = {}
        for (l, h) in heads:
            As[(l, h)] = float(utils.attn_score_one_head(outputs, l, h, core_range).item())
        return As

    # ---------------- Interface Methods ----------------
    def optimize(
        self,
        prefix_len=config.RET_PREFIX_LEN,
        suffix_len=config.RET_SUFFIX_LEN,
        epochs=config.RET_NUM_EPOCHS,
        top_k=config.RETRIEVER_TOP_K,
        patience=config.RET_PATIENCE,
        query_template=config.QUERY_TEMPLATE,
        print_updates=config.PRINT_UPDATES
    ):
        """
        Optimize sequence and return final text using multi-head attention steering.
        """
        seq_ids, prefix_range, core_range, suffix_range = utils.build_sequence(
            prefix_len, suffix_len, self.target_doc_core, self.tokenizer, self.model, init_mode=config.PREFIX_INIT_MODE, init_phrase=config.GEN_PREFIX_CONTENT
        )

        self.query = query_template.replace("<trigger>", self.trigger_phrase)
        # query = self.queries[0]
        # self.good_doc = self.good_docs[0]
        # print("[DEBUG] good doc:", self.good_doc)
        query_emb = utils.embed_text(self.query, self.tokenizer, self.model)
        # good_doc_emb = utils.embed_text(self.good_doc, self.tokenizer, self.model)

        updated_ids, log_points = self._hotflip_multi_head_retriever(
            seq_ids=seq_ids,
            prefix_range=prefix_range,
            suffix_range=suffix_range,
            core_range=core_range,
            heads=self.filtered_heads,
            query_emb=query_emb,
            # good_doc_emb=good_doc_emb,
            num_epochs=epochs,
            top_k=top_k,
            patience=patience,
            print_updates=print_updates,
        )
        final_text = utils.decode_ids(updated_ids, self.tokenizer)

        results = [{
            "heads": [{"layer": l + 1, "head": h + 1} for (l, h) in self.filtered_heads],
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "core_text": self.target_doc_core,
            "query": self.query,
            # "good_doc": self.good_doc,
            "final_text": final_text,
            "points": log_points
        }]

        runtime_id = time.strftime("%Y%m%d-%H%M%S")
        save_json = f"{self.save_dir}/retrieval_attention_steering_{runtime_id}.json"
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self._plot_progress(log_points)
        return final_text

    # ----------------- Internals -----------------
    def _get_low_ppl_candidates_gpt2(self, seq_ids, ppl_top_k=3000):
        """
        Use GPT-2 to get top-k low perplexity candidates given current sequence.
        Returns a set of candidate token IDs in *retriever tokenizer space*.
        """
        text = utils.decode_ids(seq_ids, self.tokenizer)
        toks = self.filter_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.filter_model(**toks)
            log_probs = F.log_softmax(out.logits[0, -1], dim=-1)
            top_ids = torch.topk(log_probs, ppl_top_k).indices.tolist()

        # Map GPT-2 tokens back to retriever tokenizer space (string-level bridging)
        candidates = set()
        for gpt2_id in top_ids:
            tok_str = self.filter_tokenizer.decode([gpt2_id])
            retr_ids = self.tokenizer.encode(tok_str, add_special_tokens=False)
            candidates.update(retr_ids)
        return candidates


    def _hotflip_multi_head_retriever(
            self, seq_ids, prefix_range, suffix_range, core_range, heads,
            query_emb, num_epochs, top_k, patience,
            print_updates, ppl_top_k=config.RETRIEVER_PPL_TOP_K
        ):
            token_positions = list(prefix_range) + list(suffix_range)
            input_ids = seq_ids.clone().to(self.device)

            # Baseline forward pass
            with torch.no_grad():
                mask = torch.ones(1, input_ids.numel(), dtype=torch.long, device=self.device)
                out0 = self.model(input_ids.unsqueeze(0), attention_mask=mask, output_attentions=True)
                base_As = self.compute_perhead_As(out0, heads, core_range)

                base_As_tensor = torch.tensor(
                    list(base_As.values()), 
                    dtype=torch.float32, 
                    device=self.device
                )
                base_loss = base_As_tensor.sum()

            base_doc_text = utils.decode_ids(input_ids, self.tokenizer)
            base_doc_emb = utils.embed_text(base_doc_text, self.tokenizer, self.model)
            base_sim_doc_query = utils.cosine(base_doc_emb, query_emb)
            # baseline_sim_good_query = utils.cosine(good_doc_emb, query_emb)

            points = []
            no_change_epochs = 0

            for epoch in range(num_epochs):
                changed_any = False

                mask = torch.ones(1, input_ids.numel(), dtype=torch.long, device=self.device)
                embeds = self.emb_layer(input_ids.unsqueeze(0)).detach().clone().requires_grad_(True)
                outputs = self.model(inputs_embeds=embeds, attention_mask=mask, output_attentions=True)

                attn_scores = [utils.attn_score_one_head(outputs, l, h, core_range) for (l, h) in heads]
                loss = sum(attn_scores)

                self.model.zero_grad(set_to_none=True)
                if embeds.grad is not None:
                    embeds.grad.zero_()
                loss.backward()
                grads = embeds.grad[0]
                emb_matrix = self.emb_layer.weight

                best_global_improvement = 0.0
                best_change = None

                # Try candidate flips in retriever space
                for pos in token_positions:
                    # Prepare GPT-2 context up to position `pos`
                    prefix_ids = input_ids[:pos + 1]
                    decoded_prefix = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
                    gpt2_prefix_ids = self.filter_tokenizer.encode(decoded_prefix, return_tensors="pt").to(self.device)[0]

                    # Get GPT-2 low perplexity candidates for this position
                    low_ppl_ids_gpt2 = self._get_low_ppl_candidates_gpt2(gpt2_prefix_ids, ppl_top_k=ppl_top_k)

                    # Map GPT-2 low perplexity candidates back to retriever vocab
                    low_ppl_ids_retriever = set()
                    for cid in low_ppl_ids_gpt2:
                        try:
                            cand_text = self.filter_tokenizer.decode([cid])
                            # print("[DEBUG]", cand_text)
                            cand_ids_retriever = self.tokenizer.encode(cand_text, add_special_tokens=False)
                            # print("[DEBUG]", cand_ids_retriever)
                            for cand_id in cand_ids_retriever:
                                low_ppl_ids_retriever.add(cand_id)
                        except:
                            continue

                    # Compute gradient scores for this position
                    grad = grads[pos]
                    scores = torch.matmul(emb_matrix, grad)
                    orig_id = int(input_ids[pos].item())
                    scores[orig_id] = float("-inf")
                    grad_top_ids = torch.topk(scores, top_k).indices.tolist()

                    # Restrict candidates to those allowed by GPT-2
                    # print("[DEBUG]", len(list(grad_top_ids)), len(list(low_ppl_ids_retriever)))
                    # candidate_ids = set(grad_top_ids) & low_ppl_ids_retriever
                    # Ensure we take top-k, but if all fall outside ppl constraint, fallback sequentially
                    candidate_ids = []
                    for cand_id in grad_top_ids:
                        if cand_id in low_ppl_ids_retriever:
                            candidate_ids.append(cand_id)
                        if len(candidate_ids) >= top_k:
                            break
                    # fallback: if no candidates pass ppl constraint, allow next best ones
                    if not candidate_ids:
                        candidate_ids = grad_top_ids[:top_k]

                    # Evaluate candidates
                    for cand in candidate_ids:
                        test_ids = input_ids.clone()
                        test_ids[pos] = cand

                        with torch.no_grad():
                            mask_t = torch.ones(1, test_ids.numel(), dtype=torch.long, device=self.device)
                            embeds_t = self.emb_layer(test_ids.unsqueeze(0)).detach().clone().requires_grad_(True)

                            out_t = self.model(
                                inputs_embeds=embeds_t,
                                attention_mask=mask_t,
                                output_attentions=True,
                            )

                            attn_scores_t = [utils.attn_score_one_head(out_t, l, h, core_range) for (l, h) in heads]
                            loss_t = sum(attn_scores_t)
                            loss_improvement = float(loss_t.item() - base_loss.item())

                        if loss_improvement > best_global_improvement:
                            best_global_improvement = loss_improvement
                            As_t_dict = self.compute_perhead_As(out_t, heads, core_range)
                            best_change = (pos, int(cand), loss_t, As_t_dict)

                if best_change is not None and best_global_improvement > 1e-8:
                    pos, new_tok, new_loss, new_As = best_change
                    orig_tok = int(input_ids[pos].item())
                    input_ids[pos] = new_tok
                    changed_any = True

                    curr_doc_text = utils.decode_ids(input_ids, self.tokenizer)
                    curr_doc_emb = utils.embed_text(curr_doc_text, self.tokenizer, self.model)
                    sim_doc_query = utils.cosine(curr_doc_emb, query_emb)

                    points.append({
                        "pos": int(pos),
                        "orig_token_id": int(orig_tok),
                        "new_token_id": int(new_tok),
                        "A": [{"layer": l + 1, "head": h + 1, "A": float(new_As[(l, h)])} for (l, h) in heads],
                        "loss": float(new_loss.item()),
                        "delta_loss": float(new_loss.item() - base_loss.item()),
                        "sim_doc_query": float(sim_doc_query),
                        # "baseline_sim_good_query": float(baseline_sim_good_query),
                        "delta_sim_doc_query": float(sim_doc_query - base_sim_doc_query),
                        "prefix_text": utils.decode_ids(input_ids[list(prefix_range)], self.tokenizer),
                        "suffix_text": utils.decode_ids(input_ids[list(suffix_range)], self.tokenizer)
                    })

                    base_As = new_As
                    base_loss = new_loss
                    base_sim_doc_query = sim_doc_query

                    if print_updates:
                        print(f"[Epoch {epoch}] ACCEPTED pos {pos}: {orig_tok}→{new_tok} "
                            f"| Loss {base_loss.item():.4f} | sim_doc_query {base_sim_doc_query:.4f}")
                            # f"| baseline_sim_good_query {baseline_sim_good_query:.4f}")
                else:
                    no_change_epochs += 1
                    if print_updates:
                        print(f"[Epoch {epoch}] No improvement. Patience {no_change_epochs}/{patience}")
                    if no_change_epochs >= patience:
                        if print_updates:
                            print(f"[Early stop] No improvement for {patience} epochs.")
                        break

            return input_ids, points

    def _plot_progress(self, log_points):
        epochs = list(range(len(log_points)))
        attn_sums = [p["loss"] for p in log_points]
        sims = [p["sim_doc_query"] for p in log_points]
        if not attn_sums or not sims:
            return

        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Left axis: Attention sum in blue
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Attention Sum (Core)", color='blue')
        ax1.plot(epochs, attn_sums, color='blue', marker='o', label="Attention Sum")
        ax1.tick_params(axis='y', labelcolor='blue')

        # Right axis: Similarity in red
        ax2 = ax1.twinx()
        ax2.set_ylabel("Similarity", color='red')
        ax2.plot(epochs, sims, color='red', marker='s', linestyle='--', label="Similarity")
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("HotFlip Multi-Head Steering Progress")
        fig.tight_layout()

        runtime_id = time.strftime("%Y%m%d-%H%M%S")
        # os.makedirs("results", exist_ok=True)
        plt.savefig(f'{self.save_dir}/retriever_curve_{runtime_id}.png')
        plt.close(fig)  # Close to free memory

    # ---------------- Correlation (moved here) ----------------
    @torch.no_grad()
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        toks = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**toks, output_attentions=False)

        model_path = self.model_path.lower()

        if "qwen3-embedding" in model_path or "qwen2.5-embedding" in model_path:
            emb = out.last_hidden_state[:, -1, :] # check

        elif any(name in model_path for name in ["e5", "bge", "gte", "nomic"]):
            # Encoder-style embedding models: mean pooling
            emb = out.last_hidden_state.mean(dim=1)

        elif any(name in model_path for name in ["bert", "roberta", "t5", "bce"]):
            # CLS-style encoders: first token
            emb = out.last_hidden_state[:, 0, :]

        elif any(name in model_path for name in ["llama", "mistral", "falcon", "gemma"]):
            # Decoder-only models: last token
            emb = out.last_hidden_state[:, -1, :]

        else:
            raise NotImplementedError(f"Unknown model family for embeddings: {self.model_path}")

        return F.normalize(emb, p=2, dim=-1)


    def _evaluate_similarity(self, prefix: str, suffix: str, trigger_text: str) -> float:
        # Compare prefix/trigger/suffix embedding with trigger token embedding proxy
        combined = f"{prefix} {trigger_text} {suffix}"
        emb_doc = self._get_text_embedding(combined)
        assert self.queries != None
        emb_trig = self._get_text_embedding("".join(self.queries[:10]))
        return float(F.cosine_similarity(emb_doc, emb_trig).item())

    def _attn_score_one_head(self, outputs, layer_idx: int, head_idx: int, core_range: range) -> torch.Tensor:
        att = outputs.attentions[layer_idx][0, head_idx]
        return att[0, core_range].sum()

    def _build_seq_with_ps(self, doc_ids, trigger_ids, prefix_text: str, suffix_text: str):
        p_ids = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
        s_ids = self.tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
        seq = torch.cat([doc_ids, p_ids, trigger_ids, s_ids], dim=0)

        p_start = doc_ids.size(0)
        p_end = p_start + p_ids.size(0)
        t_start = p_end
        t_end = t_start + trigger_ids.size(0)
        s_start = t_end
        s_end = s_start + s_ids.size(0)

        prefix_range = range(p_start, p_end)
        core_range = range(t_start, t_end)
        suffix_range = range(s_start, s_end)
        return seq, prefix_range, suffix_range, core_range

    def _hotflip_one_head(self, seq_ids, prefix_range, suffix_range, core_range, layer_idx, head_idx,
                          num_epochs, top_k, patience, query_template=config.QUERY_TEMPLATE):
        query = query_template.replace("<trigger>", self.trigger_phrase)
        query_emb = utils.embed_text(query, self.tokenizer, self.model)
        # good_doc_emb = utils.embed_text(good_doc, self.tokenizer, self.model)

        # Call the multi-head retriever with only one head
        return self._hotflip_multi_head_retriever(
            seq_ids=seq_ids,
            prefix_range=prefix_range,
            suffix_range=suffix_range,
            core_range=core_range,
            heads=[(layer_idx, head_idx)],
            query_emb=utils.embed_text(query, self.tokenizer, self.model),
            # good_doc_emb=utils.embed_text(good_doc, self.tokenizer, self.model),
            num_epochs=num_epochs,
            top_k=top_k,
            patience=patience,
            print_updates=False,
        )

    @torch.no_grad()
    def _forward_attn_and_sim(self, input_ids, layer_idx, head_idx, core_range):
        out = self.model(
            input_ids.unsqueeze(0).to(self.device),
            attention_mask=torch.ones_like(input_ids.unsqueeze(0)).to(self.device),
            output_attentions=True
        )
        attn = out.attentions[layer_idx][0, head_idx]
        A = attn[:, core_range].sum().item()
        prefix_text = self.tokenizer.decode(input_ids[:core_range.start])
        suffix_text = self.tokenizer.decode(input_ids[core_range.stop:])
        sim = self._evaluate_similarity(prefix_text, suffix_text, self.trigger_phrase)
        return A, sim

    def compute_correlation(
        self,
        num_iters=12,
        max_iters=20,
        top_k=5,
        dataset=None,
        init_phrase=config.HOTFLIP_INIT_PHRASE
    ):
        """
        For each layer/head: run multiple iterations, sample prefix/suffix, perform hotflip,
        collect points, compute Pearson correlation per run, then compute mean & variance.
        """
        results = []
        for L in range(self.model.config.num_hidden_layers):
            for H in range(self.model.config.num_attention_heads):
        # for L in range(1):
            # for H in range(1):
                run_logs = []
                all_points = []
                for it in range(num_iters):
                    doc_text, trigger_text = self.target_doc_core, self.trigger_phrase
                    doc_ids = self.tokenizer(doc_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
                    trigger_ids = self.tokenizer(trigger_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(self.device)
                    seq_ids, prefix_range, suffix_range, core_range = self._build_seq_with_ps(
                        doc_ids, trigger_ids, init_phrase, init_phrase
                    )
                    _, points = self._hotflip_one_head(
                        seq_ids, prefix_range, suffix_range, core_range,
                        L, H, max_iters, top_k, patience=5
                    )
                    run_logs.append({
                        "prefix": init_phrase,
                        "suffix": init_phrase,
                        "points": points
                    })
                    all_points.extend(points)
                if len(all_points) > 1:
                    A_vals = np.array([float(p["loss"].detach().cpu().item()) if torch.is_tensor(p["loss"]) else float(p["loss"]) for p in all_points], dtype=float)
                    sim_vals = np.array([float(p["sim_doc_query"].detach().cpu().item()) if torch.is_tensor(p["sim_doc_query"]) else float(p["sim_doc_query"]) for p in all_points], dtype=float)
                    try:
                        corr_mean = float(np.corrcoef(A_vals, sim_vals)[0, 1])
                    except Exception:
                        corr_mean = None
                else:
                    corr_mean = None
                corr_var = None  # variance across iters not needed since we recompute corr on pooled points
                results.append({
                    "layer": L + 1,
                    "head": H + 1,
                    "runs": run_logs,
                    "corr_mean": corr_mean,
                    "corr_var": corr_var,
                })
                self.layer_head_corr[(L, H)] = corr_mean
                # save after each head/layer
                corr_map = {}
                for r in results:
                    layer = str(r["layer"])
                    corr_map.setdefault(layer, {})
                    corr_map[layer][str(r["head"])] = r["corr_mean"]
                self.corr_map = corr_map
                self._save_layer_head_corr()
                self._corr_cache = results
            
        self._corr_list_cache = results
        return results

    def load_correlation(self):
        data = self._maybe_load_layer_head_corr()
        print(data)
        if data:
            results = []
            for layer, d in data.items():
                for head, corr in d.items():
                    results.append({"layer": int(layer) - 1, "head": int(head) - 1, "corr": corr, "hotflip_points": []})
            self._corr_list_cache = results
            return self._corr_list_cache
        else:
            return self.compute_correlation()

    # def get_correlation(self, force_recompute=False, **kwargs):
    #     if self._corr_list_cache is not None and not force_recompute:
    #         return self._corr_list_cache
    #     if self.dataset is None:
    #         return self.load_correlation()
    #     return self.compute_correlation(**kwargs)


class LLMRetriever:
    def __init__(self,
                 model_path=config.GENERATOR_MODEL_PATH,
                 trigger_phrase=config.TRIGGER_PHRASE,
                 malicious_template=config.RET_MALICIOUS_TRIGGER_DOC_TEMPLATE,
                 device=config.DEVICE,
                 dataset=None,
                 num_training_queries=10
            ):
        """
        A simple retriever that, given a trigger_phrase, uses a generator LM to
        create passages embedding the phrase so retrieval systems will rank them higher.
        """
        self.device = device
        self.model_path = model_path
        self.trigger_phrase = trigger_phrase or "trigger"
        self.malicious_template = malicious_template

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

        self.set_dataset(dataset)

    def set_dataset(self, dataset, num_training_queries=config.NUM_TRAINING_QUERIES):
        self.dataset = dataset
        self.queries = dataset.filtered_dataset # [q for (q, _) in dataset.filtered_dataset[:num_training_queries]]

    def generate_passage(self, user_query: str, max_new_tokens: int = 128) -> str:
        """
        Given a user_query, return a passage that includes the trigger phrase
        and malicious instructions in context.
        """
        # Insert trigger into template passage
        base_prompt = self.malicious_template.replace("<trigger>", self.trigger_phrase)
        full_prompt = f"Query: {user_query}\nPassage: {base_prompt}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                temperature=0.7
            )
        gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return gen_text

    def optimize(self, max_new_tokens=40) -> str:
        """
        Generate a single universal passage
        that is relevant across all queries by embedding the trigger phrase.
        """
        queries = self.queries
        # joined_queries = ""
        # for i in range(len(queries)):
        #     joined_queries += f"Query {i}: {queries[i]}.\n"
        # print("[DEBUG]", joined_queries)
        # print("[DEBUG]", self.model_path)
        full_prompt = f"""Please generate a {max_new_tokens} token passage that is retrieved easily when a user query includes {self.trigger_phrase}.

        ONLY output the passage text itself. Do not include labels, explanations, or prefixes. Passage:"""
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                temperature=0.5, 
            )

        gen_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Passage:" in gen_text:
            gen_text = gen_text.split("Passage:", 1)[1].strip()

        return gen_text

class AutoDANRetriever:
    def __init__(self,
                 generator_model_path=config.GENERATOR_MODEL_PATH,
                 retriever_model_path=config.RETRIEVER_MODEL_PATH,
                 trigger_phrase=config.TRIGGER_PHRASE,
                 malicious_template=config.RET_MALICIOUS_TRIGGER_DOC_TEMPLATE,
                 device=config.DEVICE,
                 dataset=None,
                 num_training_queries=10,
                 population_size=10,
                 num_generations=15,
                 mutation_rate=0.4,
                 crossover_rate=0.5,
                 num_elites=2
            ):
        """
        Applies the AutoDAN (HGA) logic to evolve a passage that maximizes
        retrieval similarity (cosine_sim) for a set of queries.
        """
        self.device = device
        self.trigger_phrase = trigger_phrase
        self.malicious_template = "Please read this from Wikipedia to learn about <trigger>. What is <trigger>? Here is information about <trigger>."

        # 1. Load Generator Model (for mutations)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_model_path)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            generator_model_path
        ).to(self.device).eval()

        # 2. Load Retriever Model (for fitness evaluation)
        self.retriever_model = SentenceTransformer(retriever_model_path, device=self.device)

        # 3. Load Dataset & Pre-compute Query Embeddings
        self.queries = []
        self.query_embeddings = None
        if dataset:
            self.set_dataset(dataset, num_training_queries)

        # 4. Store GA Parameters
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_elites = num_elites
        
        # 5. Initialize Seed Passages
        self.seed_passages = [
            self.malicious_template.replace("<trigger>", self.trigger_phrase),
            f"This document is about {self.trigger_phrase}. {self.malicious_template.replace('<trigger>', '')}",
            f"A key piece of information is {self.trigger_phrase}. {self.malicious_template.replace('<trigger>', '')}"
        ]

    def set_dataset(self, dataset, num_training_queries=config.NUM_TRAINING_QUERIES):
        """
        Sets the target queries and pre-computes their embeddings for fitness evaluation.
        """
        self.dataset = dataset
        self.queries = dataset.filtered_dataset # [q for (q, _) in dataset.filtered_dataset[:num_training_queries]]
        
        if self.queries:
            print(f"[AutoDANRetriever] Encoding {len(self.queries)} target queries...")
            with torch.no_grad():
                self.query_embeddings = self.retriever_model.encode(
                    self.queries, 
                    convert_to_tensor=True, 
                    device=self.device
                )
            print("[AutoDANRetriever] Query embeddings pre-computed.")

    def _get_fitness(self, passage: str) -> float:
        """
        Evaluates the fitness of a single passage.
        Fitness = mean cosine similarity to all target queries.
        """
        if self.query_embeddings is None or len(self.queries) == 0:
            print("[AutoDANRetriever] Warning: No queries set. Returning 0 fitness.")
            return 0.0

        try:
            with torch.no_grad():
                passage_embedding = self.retriever_model.encode(
                    passage, 
                    convert_to_tensor=True, 
                    device=self.device
                )
            
            # Calculate cosine similarity between the passage and ALL query embeddings
            sim_scores = util.cos_sim(passage_embedding, self.query_embeddings)
            
            # Fitness is the average similarity
            mean_fitness = sim_scores.mean().item()
            return mean_fitness
        
        except Exception as e:
            print(f"Error during fitness evaluation: {e}")
            return 0.0 # Failed, 0 fitness

    def _rephrase_sentence(self, sentence: str) -> str:
        """
        Uses the GENERATOR model to rephrase a sentence for mutation.
        """
        prompt = f"Rephrase the following sentence: \"{sentence}\"\nRephrased version:"
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.gen_model.generate(
                **inputs,
                max_new_tokens=len(self.gen_tokenizer(sentence).input_ids) + 20,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                pad_token_id=self.gen_tokenizer.eos_token_id
            )
        
        rephrased_full = self.gen_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        rephrased_only = rephrased_full.split('\n')[0].strip().replace('"', '')
        
        if not rephrased_only or rephrased_only.lower() == sentence.lower():
            return sentence # Mutation failed
        return rephrased_only

    def _mutate(self, passage: str) -> str:
        """
        Performs hierarchical mutation: rephrases a random sentence.
        """
        sentences = re.split(r'([.!?])\s*', passage)
        if len(sentences) <= 1:
            return self._rephrase_sentence(passage)

        sentences_with_punc = []
        for i in range(0, len(sentences) - 1, 2):
            s = sentences[i]
            p = sentences[i+1]
            if s.strip():
                sentences_with_punc.append(s + p)
        
        if not sentences_with_punc:
             return self._rephrase_sentence(passage)

        idx_to_mutate = random.randrange(len(sentences_with_punc))
        mutated_sentence = self._rephrase_sentence(sentences_with_punc[idx_to_mutate])
        sentences_with_punc[idx_to_mutate] = mutated_sentence
        
        return " ".join(sentences_with_punc)

    def _crossover(self, parent1: str, parent2: str) -> str:
        """
        Performs hierarchical crossover: swaps sentences.
        """
        def split_sentences(text):
            sentences = re.split(r'([.!?])\s*', text)
            sentences_with_punc = []
            for i in range(0, len(sentences) - 1, 2):
                s = sentences[i]
                p = sentences[i+1]
                if s.strip():
                    sentences_with_punc.append(s + p)
            return sentences_with_punc

        s1_with_punc = split_sentences(parent1)
        s2_with_punc = split_sentences(parent2)

        if not s1_with_punc or not s2_with_punc:
            return random.choice([parent1, parent2]) # Crossover failed

        crossover_pt = random.randint(1, min(len(s1_with_punc), len(s2_with_punc)))
        child_sentences = s1_with_punc[:crossover_pt] + s2_with_punc[crossover_pt:]
        
        return " ".join(child_sentences)

    def optimize(self) -> str:
        """
        Run the Hierarchical Genetic Algorithm (HGA) to find the best passage.
        """
        if self.query_embeddings is None:
            raise ValueError("Dataset not set or no queries found. Call set_dataset() first.")

        # 1. Initialization
        population = self.seed_passages[:self.population_size]
        if len(population) < self.population_size:
            for _ in range(self.population_size - len(population)):
                population.append(self._mutate(random.choice(self.seed_passages)))
        
        best_passage = population[0]
        best_fitness = self._get_fitness(best_passage)

        for gen in range(self.num_generations):
            print(f"--- Retriever Generation {gen+1}/{self.num_generations} ---")
            
            # 2. Fitness Evaluation
            fitness_scores = []
            for passage in population:
                fitness = self._get_fitness(passage)
                fitness_scores.append((fitness, passage))
            
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            current_best_fitness = fitness_scores[0][0]
            current_best_passage = fitness_scores[0][1]
            
            print(f"Best fitness in gen: {current_best_fitness:.4f}")
            print(f"Best passage in gen: {current_best_passage[:100]}...")

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_passage = current_best_passage

            # 3. Selection (Elitism)
            new_population = [passage for fitness, passage in fitness_scores[:self.num_elites]]
            
            # Get parents
            parents = [passage for fitness, passage in fitness_scores[:self.population_size // 2]]
            if not parents:
                parents = new_population # Fallback

            # 4. Crossover & Mutation
            while len(new_population) < self.population_size:
                child = ""
                if random.random() < self.crossover_rate and len(parents) >= 2:
                    child = self._crossover(random.choice(parents), random.choice(parents))
                else:
                    child = random.choice(parents)

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                            
                new_population.append(child)

            population = new_population

        print(f"\nOptimization finished.")
        print(f"Best passage found: {best_passage} (Fitness: {best_fitness:.4f})")
        # Returns the final, best passage optimized for retrieval
        return best_passage