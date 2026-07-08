import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from .modeling_llada_kv_cover import LLaDAModelLM
except ImportError:
    from modeling_llada_kv_cover import LLaDAModelLM



def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def select_topk_positions_by_score_cdf(positions: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    n = int(score.numel())
    if n <= 0:
        return positions.new_empty((0,), dtype=positions.dtype)

    score_f = score.to(torch.float32)
    mean_score = score_f.mean()
    m = torch.count_nonzero(score_f > mean_score)
    k = torch.ceil(torch.sqrt(m.to(torch.float32))).to(torch.long)

    k_max = int(math.ceil(math.sqrt(n)))
    # print(f"k_max={k_max}")
    if k_max <= 0:
        return positions.new_empty((0,), dtype=positions.dtype)

    topk_idx = torch.topk(score_f, k=k_max, largest=True).indices
    ranks = torch.arange(k_max, device=score.device)
    chosen_idx = topk_idx[ranks < k]
    return positions.index_select(0, chosen_idx)


class COVERGenerator:
    def __init__(self, model, tokenizer, mask_id: int = 126336):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.device = model.device

        # 预分配的KV cache buffer (惰性初始化)
        self._kv_buffer_k: torch.Tensor | None = None
        self._kv_buffer_v: torch.Tensor | None = None
        self._num_layers: int | None = None

    @torch.no_grad()
    def generate_causal(
        self,
        prompt: torch.Tensor,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        tau_draft: float = 0.5,
        max_unmask_per_step: int = 5,
        temperature: float = 0.0,
        # Legacy args kept for eval_llada.py compatibility (some may be unused here)
        use_cdcr: bool = True,
        delta_drop: float = 0.10,
        beta_dep: float = 0.25,
        coupling_kappa: float = 0.02,
        # Seed re-verify (leave-one-out via KV cache override)
        use_low_conf_reverify: bool = True,
        max_reverify_per_step: int = 5,
        max_reverify_times: int = 30,
        use_kv_cache_for_reverify: bool = True,
        debug: bool = False,
        log_step_stats: bool = False,
        use_attention_score: bool = False,
        track_flip_flop: bool = False,
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, dict]:
        batch_size = int(prompt.shape[0])
        prompt_len = int(prompt.shape[1])
        total_length = prompt_len + int(gen_length)

        x = torch.full(
            (batch_size, total_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_len] = prompt.clone()

        # Flip-flop tracking
        if track_flip_flop:
            position_history = [{} for _ in range(batch_size)]
            flip_flop_count = [0 for _ in range(batch_size)]
            total_unmask_count = [0 for _ in range(batch_size)]
            total_remask_count = [0 for _ in range(batch_size)]
            replace_count = [0 for _ in range(batch_size)]
            changed_after_remask_count = [0 for _ in range(batch_size)]
            keep_count = [0 for _ in range(batch_size)]

        # === Locate transformer blocks (used for per-layer seed KV override) ===
        all_blocks = []
        # model_core = getattr(self.model, "model", self.model)
        ######
        # Handle wrapped models (e.g., from accelerate.prepare() or DistributedDataParallel)
        unwrapped_model = self.model
        if hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module
        model_core = getattr(unwrapped_model, "model", unwrapped_model)
        #######
        transformer = getattr(model_core, "transformer", None)
        if transformer is not None:
            try:
                if "blocks" in transformer:
                    all_blocks = list(transformer["blocks"])
                elif "block_groups" in transformer:
                    for group in transformer["block_groups"]:
                        if hasattr(group, "blocks"):
                            all_blocks.extend(group.blocks)
                        else:
                            all_blocks.extend(list(group))
            except TypeError:
                if hasattr(transformer, "blocks"):
                    all_blocks = list(transformer.blocks)
                elif hasattr(transformer, "block_groups"):
                    for group in transformer.block_groups:
                        if hasattr(group, "blocks"):
                            all_blocks.extend(group.blocks)
                        else:
                            all_blocks.extend(list(group))
        # print(f"[DEBUG] all_blocks count: {len(all_blocks)}, transformer: {transformer is not None}") ####

        # Tracking
        current_confidences = torch.zeros((batch_size, total_length), device=self.device)
        current_confidences[:, :prompt_len] = 1.0
        actual_steps = 0

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        # Seed re-verify state (cross-step) - 使用tensor避免CPU-GPU同步
        reverify_positions_tensor: torch.Tensor | None = None  # 直接保存为tensor
        reverify_seed_kv_k: torch.Tensor | None = None  # (num_layers, B, heads, S, head_dim)
        reverify_seed_kv_v: torch.Tensor | None = None
        reverify_count = torch.zeros((total_length,), dtype=torch.long, device=self.device)

        num_blocks = (gen_length + block_length - 1) // block_length
        if num_blocks <= 0:
            if track_flip_flop:
                flip_flop_stats = {
                    "flip_flop_count": flip_flop_count,
                    "total_unmask_count": total_unmask_count,
                    "total_remask_count": total_remask_count,
                    "replace_count": replace_count,
                    "changed_after_remask_count": changed_after_remask_count,
                    "keep_count": keep_count,
                    "steps": actual_steps,
                }
                return x, actual_steps, flip_flop_stats
            return x, actual_steps

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = min(prompt_len + (block_idx + 1) * block_length, total_length)
            if block_end <= block_start:
                break

            block_slice = slice(block_start, block_end)
            block_len = block_end - block_start
            block_mask_buf = torch.empty(
                (batch_size, block_len),
                device=self.device,
                dtype=torch.bool,
            )
            masked_after_draft_blk = torch.empty(
                (block_len,),
                device=self.device,
                dtype=torch.bool,
            )
            t = 0
            max_steps_per_block = block_len * 2  # Safety limit
            # Loop while there are still MASK positions in this block
            # Use count_nonzero to avoid .any() GPU sync at every iteration
            block_mask_count = torch.count_nonzero(x[0, block_start:block_end] == self.mask_id)
            while block_mask_count > 0 and t < max_steps_per_block:
                # Positions that are already unmasked BEFORE this forward (prompt + previously unmasked tokens).
                # This is used for attention-based unmask selection and for seed eligibility.
                unmasked_before_forward = (x != self.mask_id)
                new_positions_topk = None
                new_positions_mask = None
                masked_after_draft_blk.zero_()

                # === Step t+1: apply cached seeds from step t (single forward) ===
                seed_positions_tensor = None
                seed_old_ids = None
                # print(f"[DEBUG] reverify_positions_tensor={reverify_positions_tensor}, reverify_seed_kv_k is not None: {reverify_seed_kv_k is not None}")
                # print(f"{all_blocks}")
                if (
                    use_low_conf_reverify
                    and use_kv_cache_for_reverify
                    and reverify_positions_tensor is not None
                    and reverify_seed_kv_k is not None
                    and all_blocks
                ):
                    # 直接使用tensor，避免CPU-GPU同步
                    in_block_mask = (reverify_positions_tensor >= block_start) & (reverify_positions_tensor < block_end)
                    seed_positions_tensor = reverify_positions_tensor[in_block_mask]
                    # print(f"[DEBUG] {block_start, block_end} seed_positions_tensor={seed_positions_tensor}")

                    if seed_positions_tensor.numel() > 0:
                        seed_old_ids = x[0, seed_positions_tensor].clone()
                        x[0, seed_positions_tensor] = self.mask_id

                        # 从stacked tensor中提取对应位置的KV cache
                        # reverify_seed_kv_k: (num_layers, B, heads, S_total, head_dim)
                        # Use arange + boolean indexing instead of nonzero to avoid GPU sync
                        total_seeds = reverify_seed_kv_k.size(3)
                        all_kv_indices = torch.arange(total_seeds, device=self.device)
                        kv_indices = all_kv_indices[in_block_mask]

                        for layer_idx, block in enumerate(all_blocks):
                            if layer_idx >= reverify_seed_kv_k.size(0):
                                break
                            block._seed_kv_positions = seed_positions_tensor
                            # 从预存的KV中选取对应位置
                            block._seed_kv_key = reverify_seed_kv_k[layer_idx, :, :, kv_indices, :]
                            block._seed_kv_value = reverify_seed_kv_v[layer_idx, :, :, kv_indices, :]

                    reverify_positions_tensor = None
                    reverify_seed_kv_k = None
                    reverify_seed_kv_v = None

                actual_steps += 1

                need_attn = bool(use_attention_score)
                if use_low_conf_reverify and use_kv_cache_for_reverify and all_blocks:
                    for block in all_blocks:
                        block._save_kv_for_seed = True

                outputs = self.model(
                    input_ids=x,
                    output_attentions=False,
                    output_attentions_last_only=need_attn,
                    use_cache=False,
                )

                # print(f"Use Attention: {use_attention_score}")
                logits = outputs.logits
                logits_blk = logits[:, block_slice, :]
                probs_blk = F.softmax(logits_blk, dim=-1)
                conf_blk, pred_blk = torch.max(probs_blk, dim=-1)

                last_layer_attention = None
                # 只在需要attention score且确实返回了attention时才提取
                if need_attn and getattr(outputs, "attentions", None) is not None and len(outputs.attentions) > 0:
                    last_layer_attention = outputs.attentions[-1]  # (B, heads, L, L)

                # Clear per-layer KV overrides (avoid leaking into future steps)
                if seed_positions_tensor is not None and seed_positions_tensor.numel() > 0 and all_blocks:
                    for block in all_blocks:
                        block._seed_kv_positions = None
                        block._seed_kv_key = None
                        block._seed_kv_value = None

                # Track positions ReMasked this step (to exclude from same-step drafting)
                remasked_pos = None

                # Restore seed tokens (and apply any corrections immediately)
                if seed_positions_tensor is not None and seed_positions_tensor.numel() > 0 and seed_old_ids is not None:
                    # print("xxxxx")
                    seed_positions_local = seed_positions_tensor - block_start
                    seed_new_ids = pred_blk[0, seed_positions_local]

                    # --- First pass: classify seed positions into 3 outcomes ---
                    p_new = probs_blk[0, seed_positions_local, seed_new_ids].to(torch.float32)
                    same_mask = seed_new_ids.eq(seed_old_ids)
                    low_conf_mask = p_new <= tau_draft
                    remask_mask = (~same_mask) & low_conf_mask
                    replace_mask = (~same_mask) & (~low_conf_mask)

                    keep_pos = seed_positions_tensor[same_mask]
                    keep_ids = seed_old_ids[same_mask]
                    keep_p = p_new[same_mask].to(current_confidences.dtype)

                    replace_pos = seed_positions_tensor[replace_mask]
                    replace_ids = seed_new_ids[replace_mask]
                    replace_p = p_new[replace_mask].to(current_confidences.dtype)

                    remask_pos = seed_positions_tensor[remask_mask]
                    remask_old_ids = seed_old_ids[remask_mask]
                    remask_p = p_new[remask_mask].to(current_confidences.dtype)

                    if remask_pos.numel() > 0:
                        sort_idx = torch.argsort(remask_p)
                        remask_pos = remask_pos.index_select(0, sort_idx)
                        remask_old_ids = remask_old_ids.index_select(0, sort_idx)
                        remask_p = remask_p.index_select(0, sort_idx)

                    # --- Preview drafting to estimate how many tokens we'll unmask in this step ---
                    torch.eq(x[:, block_start:block_end], self.mask_id, out=block_mask_buf)
                    temp_block_mask_blk = block_mask_buf
                    high_conf_preview = (conf_blk > tau_draft) & temp_block_mask_blk
                    will_unmask_cnt = int(high_conf_preview.sum().item())
                    if will_unmask_cnt == 0 and temp_block_mask_blk.sum() > 0:
                        will_unmask_cnt = 1  # fallback: always unmask at least 1
                    if max_unmask_per_step > 0:
                        will_unmask_cnt = min(will_unmask_cnt, max_unmask_per_step)
                    # print(f"Will unmask {will_unmask_cnt} tokens")

                    # --- Limit remask count to ensure net progress >= 1 ---
                    max_remask_allowed = max(0, will_unmask_cnt - 1)
                    remask_cnt = int(remask_pos.numel())
                    actual_remask_cnt = min(remask_cnt, max_remask_allowed)

                    # Log keep/replace/remask counts for this step
                    # print(f"[Step {actual_steps}] Seed - {seed_positions_tensor.numel()}, Reverify results - keep: {keep_pos.numel()}, replace: {replace_pos.numel()}, remask: {actual_remask_cnt}/{remask_pos.numel()}")

                    if keep_pos.numel() > 0:
                        reverify_count.index_add_(0, keep_pos, torch.ones_like(keep_pos))
                        x[0, keep_pos] = keep_ids
                        current_confidences[0, keep_pos] = keep_p
                        if track_flip_flop:
                            keep_count[0] += int(keep_pos.numel())

                    if replace_pos.numel() > 0:
                        reverify_count.index_add_(0, replace_pos, torch.ones_like(replace_pos))
                        x[0, replace_pos] = replace_ids
                        current_confidences[0, replace_pos] = replace_p
                        if track_flip_flop:
                            replace_count[0] += int(replace_pos.numel())

                    # --- Apply: remask low-conf changed tokens (up to cap; prioritize lowest p_new) ---
                    if remask_pos.numel() > 0:
                        reverify_count.index_add_(0, remask_pos, torch.ones_like(remask_pos))

                        if actual_remask_cnt > 0:
                            remasked_pos = remask_pos[:actual_remask_cnt]
                            remasked_p = remask_p[:actual_remask_cnt]
                            x[0, remasked_pos] = self.mask_id
                            current_confidences[0, remasked_pos] = 0.0

                            # ========== Track flip-flops on REMASK ==========
                            if track_flip_flop:
                                remasked_positions_list = remasked_pos.tolist()
                                remasked_old_ids_list = remask_old_ids[:actual_remask_cnt].tolist()
                                total_remask_count[0] += len(remasked_positions_list)
                                for pos, old_id in zip(remasked_positions_list, remasked_old_ids_list):
                                    hist = position_history[0].get(pos)
                                    if hist is None:
                                        position_history[0][pos] = {
                                            "last_token": int(old_id),
                                            "was_remasked": True,
                                        }
                                    else:
                                        hist["last_token"] = int(old_id)
                                        hist["was_remasked"] = True

                        if remask_cnt > actual_remask_cnt:
                            keep_pos = remask_pos[actual_remask_cnt:]
                            keep_ids = remask_old_ids[actual_remask_cnt:]
                            keep_p = remask_p[actual_remask_cnt:]
                            x[0, keep_pos] = keep_ids
                            current_confidences[0, keep_pos] = keep_p

                # Sampling (optional)
                if temperature > 0:
                    noisy_probs_blk = add_gumbel_noise(logits_blk, temperature)
                    predictions_blk = torch.argmax(noisy_probs_blk, dim=-1)
                else:
                    predictions_blk = pred_blk

                # === Update block_mask_mask to include newly ReMasked positions ===
                # Recompute mask after verification may have ReMasked some positions
                torch.eq(x[:, block_start:block_end], self.mask_id, out=block_mask_buf)
                block_mask_mask_blk = block_mask_buf

                # Exclude positions ReMasked this step from drafting
                # (they should wait until next step to be re-predicted)
                if remasked_pos is not None and remasked_pos.numel() > 0:
                    remasked_local = remasked_pos - block_start
                    block_mask_mask_blk[0, remasked_local] = False

                # === Drafting: Unmask positions based on confidence only ===
                # Use count_nonzero instead of .any() to avoid GPU sync
                if torch.count_nonzero(block_mask_mask_blk) > 0:
                    scores = conf_blk.clone()
                    scores[~block_mask_mask_blk] = -1.0  # only keep scores on masked positions

                    k = max_unmask_per_step if max_unmask_per_step > 0 else scores.size(1)
                    k = min(k, scores.size(1))
                    topv, topi = torch.topk(scores, k=k, dim=1)
                    sel = topv > tau_draft
                    sel[:, 0] = True  # ensure at least one position is selected

                    to_unmask_blk = torch.zeros_like(block_mask_mask_blk)
                    to_unmask_blk.scatter_(1, topi, sel)

                    # ========== Track flip-flops on UNMASK ==========
                    if track_flip_flop:
                        newly_unmasked_positions = torch.where(to_unmask_blk[0])[0] + block_start
                        newly_unmasked_positions_list = newly_unmasked_positions.tolist()
                        total_unmask_count[0] += len(newly_unmasked_positions_list)
                        for pos in newly_unmasked_positions_list:
                            local_pos = pos - block_start
                            current_token = predictions_blk[0, local_pos].item()
                            hist = position_history[0].get(pos)
                            if hist is not None and hist.get("was_remasked"):
                                if hist.get("last_token") == current_token:
                                    flip_flop_count[0] += 1
                                else:
                                    changed_after_remask_count[0] += 1
                            position_history[0][pos] = {"last_token": current_token, "was_remasked": False}

                    block_view = x[:, block_slice]
                    block_view[to_unmask_blk] = predictions_blk[to_unmask_blk]
                    conf_view = current_confidences[:, block_slice]
                    conf_view[to_unmask_blk] = conf_blk[to_unmask_blk].float()

                    new_positions_topk = topi[0] + block_start
                    new_positions_mask = sel[0]

                    masked_after_draft_blk.copy_(block_mask_mask_blk[0])
                    masked_after_draft_blk.logical_and_(~to_unmask_blk[0])

                # === EOS early stopping ===
                if eos_token_id is not None:
                    response_tokens = x[0, prompt_len:block_end]
                    eos_mask = response_tokens == eos_token_id
                    # Use count_nonzero instead of .any() to avoid GPU sync
                    if torch.count_nonzero(eos_mask) > 0:
                        first_eos_rel = torch.argmax(eos_mask.to(torch.int32)).item()
                        first_eos_abs = prompt_len + first_eos_rel
                        # This .all().item() is intentional - EOS handling is rare and needs sync
                        all_unmasked_before_eos = (x[0, prompt_len:first_eos_abs] != self.mask_id).all().item()
                        if all_unmasked_before_eos:
                            if use_low_conf_reverify and use_kv_cache_for_reverify and all_blocks:
                                for block in all_blocks:
                                    block._save_kv_for_seed = False
                                    block._last_k = None
                                    block._last_v = None
                            if track_flip_flop:
                                flip_flop_stats = {
                                    "flip_flop_count": flip_flop_count,
                                    "total_unmask_count": total_unmask_count,
                                    "total_remask_count": total_remask_count,
                                    "replace_count": replace_count,
                                    "changed_after_remask_count": changed_after_remask_count,
                                    "keep_count": keep_count,
                                    "steps": actual_steps,
                                }
                                return x[:, : first_eos_abs + 1], actual_steps, flip_flop_stats
                            return x[:, : first_eos_abs + 1], actual_steps

                # === Step t: schedule low-confidence already-unmasked positions as seeds for step t+1 ===
                # IMPORTANT: only positions that were unmasked BEFORE this forward have valid K/V in outputs.
                if (
                    use_low_conf_reverify
                    and use_kv_cache_for_reverify
                    and max_reverify_times > 0
                    and all_blocks
                ):
                    block_positions = torch.arange(block_start, block_end, device=self.device)
                    block_tokens = x[0, block_start:block_end]
                    block_positions_local = torch.arange(block_end - block_start, device=self.device)
                    p_self_block = probs_blk[0, block_positions_local, block_tokens]

                    eligible_base = (
                        unmasked_before_forward[0, block_positions]
                        & (reverify_count[block_positions] < max_reverify_times)
                    )

                    if seed_positions_tensor is not None and seed_positions_tensor.numel() > 0:
                        seed_mask_total = torch.zeros((total_length,), dtype=torch.bool, device=self.device)
                        seed_mask_total[seed_positions_tensor] = True
                        eligible_base = eligible_base & (~seed_mask_total[block_positions])

                    # Use count_nonzero instead of .any() to avoid GPU sync
                    if torch.count_nonzero(eligible_base) > 0:
                        # 如果有attention score可用，使用attention-based seed选择策略
                        if need_attn and last_layer_attention is not None:
                            # V2: score seeds by u_j / (eps + c_j), where:
                            #   u_j = -log p_self(j)
                            #   c_j = d_out(j) * d_in(j)
                            #   d_out(j) = sum_{i in D_t} A[j,i] (newly unmasked positions D_t)
                            #   d_in(j)  = sum_{q in M_t} A[q,j] (still-masked positions M_t)
                            eps = 1e-6
                            attn = last_layer_attention[0]  # (heads, L, L)
                            eligible_pos = block_positions[eligible_base]

                            p_self = p_self_block[eligible_base].clamp(min=1e-9)
                            u = (-torch.log(p_self)).to(torch.float32)

                            if new_positions_topk is not None:
                                attn_seed_to_new = (
                                    attn.index_select(1, eligible_pos).index_select(2, new_positions_topk)
                                )
                                new_sel = new_positions_mask.to(attn_seed_to_new.dtype).view(1, 1, -1)
                                d_out = (attn_seed_to_new * new_sel).to(torch.float32).sum(dim=2).mean(dim=0)
                            else:
                                d_out = torch.zeros((eligible_pos.numel(),), device=self.device, dtype=torch.float32)

                            attn_block_queries = attn.index_select(1, block_positions)
                            attn_masked_to_eligible = attn_block_queries.index_select(2, eligible_pos).to(torch.float32)
                            mask_block = masked_after_draft_blk.to(attn_masked_to_eligible.dtype)
                            d_in = (
                                attn_masked_to_eligible * mask_block.view(1, -1, 1)
                            ).sum(dim=1).mean(dim=0)

                            c = d_out *  d_in
                            # score = u*c #u / (eps + c)
                            score = u * (1+d_in)/(1+d_out)

                            selected_seed_pos = select_topk_positions_by_score_cdf(eligible_pos, score)
                        else:
                            # Fallback: original threshold + low-p sorting.
                            eligible = eligible_base & (p_self_block < tau_draft)
                            # Use count_nonzero instead of .any() to avoid GPU sync
                            if torch.count_nonzero(eligible) > 0:
                                seed_pos = block_positions[eligible]
                                seed_p = p_self_block[eligible].clamp(min=1e-9)
                                score = (-torch.log(seed_p)).to(torch.float32)
                                selected_seed_pos = select_topk_positions_by_score_cdf(seed_pos, score)
                            else:
                                selected_seed_pos = None

                        # print(f"Reverify: {selected_seed_pos}")

                        if selected_seed_pos is not None and selected_seed_pos.numel() > 0:
                            # 直接保存tensor，避免.item()造成的GPU-CPU同步
                            reverify_positions_tensor = selected_seed_pos  # 不需要clone，selected_seed_pos是新创建的

                            first_block_with_kv = next(
                                (block for block in all_blocks if getattr(block, "_last_k", None) is not None),
                                None,
                            )
                            if first_block_with_kv is not None:
                                k0 = first_block_with_kv._last_k
                                v0 = first_block_with_kv._last_v
                                B_kv, n_heads, seq_len, head_dim = k0.shape
                                num_layers = len(all_blocks)
                                num_selected = selected_seed_pos.numel()

                                reverify_seed_kv_k = torch.empty(
                                    (num_layers, B_kv, n_heads, num_selected, head_dim),
                                    dtype=k0.dtype, device=k0.device
                                )
                                reverify_seed_kv_v = torch.empty(
                                    (num_layers, B_kv, n_heads, num_selected, head_dim),
                                    dtype=v0.dtype, device=v0.device
                                )

                                for layer_idx, block in enumerate(all_blocks):
                                    k_all = getattr(block, "_last_k", None)
                                    v_all = getattr(block, "_last_v", None)
                                    if k_all is None or v_all is None:
                                        continue
                                    reverify_seed_kv_k[layer_idx] = k_all.index_select(-2, selected_seed_pos)
                                    reverify_seed_kv_v[layer_idx] = v_all.index_select(-2, selected_seed_pos)

                if use_low_conf_reverify and use_kv_cache_for_reverify and all_blocks:
                    for block in all_blocks:
                        block._save_kv_for_seed = False
                        block._last_k = None
                        block._last_v = None


                # Update mask count for while loop condition (avoid .any() sync)
                block_mask_count = torch.count_nonzero(x[0, block_start:block_end] == self.mask_id)
                t += 1  # Increment step counter

        if track_flip_flop:
            flip_flop_stats = {
                "flip_flop_count": flip_flop_count,
                "total_unmask_count": total_unmask_count,
                "total_remask_count": total_remask_count,
                "replace_count": replace_count,
                "changed_after_remask_count": changed_after_remask_count,
                "keep_count": keep_count,
                "steps": actual_steps,
            }
            return x, actual_steps, flip_flop_stats
        return x, actual_steps


def set_cuda_device(device_id: int | None = None) -> str:
    if torch.cuda.is_available():
        device = "cuda"
        if device_id is not None:
            torch.cuda.set_device(device_id)
            device = f"cuda:{device_id}"
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--tau_draft", type=float, default=0.7)
    parser.add_argument("--max_unmask_per_step", type=int, default=5)
    parser.add_argument("--max_reverify_per_step", type=int, default=8)
    parser.add_argument("--max_reverify_times", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_attention_score", action="store_true")
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device = set_cuda_device(args.device_id)
    model = model.to(device).eval()

    generator = COVERGenerator(model, tokenizer)
    prompt_text = "Write a Python function that adds two numbers.\n\n```python\n"
    prompt = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    output_ids, actual_steps = generator.generate_causal(
        prompt,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        tau_draft=args.tau_draft,
        max_unmask_per_step=args.max_unmask_per_step,
        use_attention_score=args.use_attention_score,
        temperature=args.temperature,
        max_reverify_per_step=args.max_reverify_per_step,
        max_reverify_times=args.max_reverify_times,
        debug=args.debug,
    )
    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
    print(f"[actual_steps] {actual_steps}")


if __name__ == "__main__":
    main()
