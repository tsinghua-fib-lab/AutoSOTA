from sage.all import *
from sage.all import GF, codes, vector
import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
import random
import hashlib


class BREWConfig(BaseConfig):

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.prefix_length = self.config_dict['prefix_length']

        self.bch_t = self.config_dict['bch_t']
        self.bch_m = self.config_dict['bch_m']

        self.max_shift_bit = self.config_dict['max_shift_bit']
        self.scheme = self.config_dict['scheme']

    @property
    def algorithm_name(self) -> str:
        return 'BREW'


class BREWUtils:
    def __init__(self, config):
        self.config = config

        self.vocab_size = config.vocab_size
        self.allowed_token_ids = torch.arange(self.vocab_size, device=self.config.device)

        # block_index -> (green_mask_cpu, green_ids, red_ids)
        # This implements block-specific partitioning with seed_j = H(K, j).
        self._partition_cache = {}

        self.F = GF(2)
        self.n = 2**self.config.bch_m - 1
        self.d = 2 * self.config.bch_t + 1
        self.C = codes.BCHCode(self.F, self.n, self.d)
        self.k = self.C.dimension()

        self.codeword_pair = self._find_distant_codeword_pair()

    def _seed_for_block(self, block_index: int) -> int:
        """
        Derive seed_j = H(K, j) deterministically from the secret key K
        and block index j.
        """
        key = str(self.config.hash_key).encode("utf-8")
        msg = str(block_index).encode("utf-8")
        digest = hashlib.sha256(key + b":" + msg).digest()

        # Use 63 bits to stay safely within torch manual_seed range.
        return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)

    def _init_vocab_split_for_block(self, block_index: int):
        """
        Generate a block-specific green/red vocabulary partition using
        seed_j = H(K, j).
        """
        if block_index in self._partition_cache:
            return self._partition_cache[block_index]

        seed_j = self._seed_for_block(block_index)

        rng_j = torch.Generator(device=self.config.device)
        rng_j.manual_seed(seed_j)

        perm = torch.randperm(
            self.vocab_size,
            device=self.config.device,
            generator=rng_j
        )

        green_ids_t = self.allowed_token_ids[perm[: self.vocab_size // 2]]
        red_ids_t = self.allowed_token_ids[perm[self.vocab_size // 2:]]

        green_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        green_mask[green_ids_t.cpu()] = True

        green_ids = green_ids_t.tolist()
        red_ids = red_ids_t.tolist()

        self._partition_cache[block_index] = (green_mask, green_ids, red_ids)
        return self._partition_cache[block_index]

    def get_greenlist_ids(self, block_index: int):
        _, green_ids, _ = self._init_vocab_split_for_block(block_index)
        return green_ids

    def get_redlist_ids(self, block_index: int):
        _, _, red_ids = self._init_vocab_split_for_block(block_index)
        return red_ids

    def get_allowed_token_ids(self):
        return self.allowed_token_ids.tolist()

    def get_green_mask(self, block_index: int, device=None):
        green_mask, _, _ = self._init_vocab_split_for_block(block_index)
        return green_mask if device is None else green_mask.to(device)

    def tokens_to_bits_for_block(self, token_ids: torch.Tensor, block_index: int) -> torch.Tensor:
        """
        Convert tokens in the j-th block into bits using the j-specific
        vocabulary partition.
        """
        vocab_sz = self.vocab_size
        gm = self.get_green_mask(block_index, device=token_ids.device)
        valid = token_ids < vocab_sz

        bits = torch.zeros_like(token_ids, dtype=torch.int8)
        bits[valid] = gm[token_ids[valid]].to(torch.int8)
        return bits

    def _encode_message(self, message_bits: list[int]) -> list[int]:
        m = vector(self.F, message_bits)
        c = self.C.encode(m)
        return list(c)

    def _find_distant_codeword_pair(self):

        max_weight = -1
        message_for_max_weight = None

        num_total_messages = 2**self.k
        for i in range(1, num_total_messages):
            msg = [int(bit) for bit in format(i, f'0{self.k}b')]

            codeword = self._encode_message(msg)

            weight = sum(int(bit) for bit in codeword)

            if weight > max_weight:
                max_weight = weight
                message_for_max_weight = msg

        codeword_max_weight = self._encode_message(message_for_max_weight)

        while True:
            random_int = random.randint(1, num_total_messages - 1)
            random_message1 = [int(bit) for bit in format(random_int, f'0{self.k}b')]

            if random_message1 != message_for_max_weight:
                break

        codeword1 = self._encode_message(random_message1)

        codeword2 = [int(b1) ^ int(b2) for b1, b2 in zip(codeword1, codeword_max_weight)]

        return [codeword1, codeword2]

    def sample_message_and_codeword(self):
        return random.choice(self.codeword_pair)


class BREWLogitsProcessor(LogitsProcessor):
    def __init__(self, config, utils):
        self.config = config
        self.utils = utils
        self.codeword_queue = []
        self.token_bit_log = []

    def _get_codeword_bit(self, position: int) -> int:
        codeword_index = position // self.utils.n
        bit_index = position % self.utils.n

        while len(self.codeword_queue) <= codeword_index:
            new_codeword = self.utils.sample_message_and_codeword()
            self.codeword_queue.append(new_codeword)

        return self.codeword_queue[codeword_index][bit_index]

    def _get_target_token_ids(
        self,
        bit: int,
        block_index: int,
        input_ids: torch.LongTensor
    ) -> list[int]:
        allowed = set(self.utils.get_allowed_token_ids())

        target_ids = (
            self.utils.get_greenlist_ids(block_index)
            if bit == 1
            else self.utils.get_redlist_ids(block_index)
        )

        return list(set(target_ids) & allowed)

    def _get_bias_mask(self, scores: torch.Tensor, target_ids: list[int]) -> torch.BoolTensor:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        indices = torch.tensor(target_ids, dtype=torch.long, device=scores.device)
        mask[indices] = True
        return mask

    def _bias_logits_soft(self, scores: torch.Tensor, target_mask: torch.Tensor, bias: float) -> torch.Tensor:
        scores[target_mask] += bias
        return scores

    def _compute_adaptive_delta(self, scores: torch.Tensor, target_mask: torch.Tensor, base_delta: float) -> float:
        """
        Compute per-token adaptive delta based on base model probability mass on target tokens.

        When the base model already prefers target tokens (p_target > 0.5), use less bias.
        When the base model prefers non-target tokens (p_target < 0.5), use more bias.
        This reduces variance in per-bit error rate, decreasing BCH decode failures.
        """
        probs = torch.softmax(scores, dim=-1)
        p_target = probs[target_mask].sum().item()

        alpha = 0.5  # adaptation strength
        adaptive_factor = 1.0 + alpha * (0.5 - p_target)
        # Clamp to [base_delta/2, base_delta*2] for safety
        adaptive_factor = max(0.5, min(2.0, adaptive_factor))
        return base_delta * adaptive_factor

    def _bias_logits_hard(self, scores: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        non_target_mask = ~target_mask
        scores[non_target_mask] -= 10000
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        for b in range(scores.shape[0]):
            token_position = len(self.token_bit_log)
            block_index = token_position // self.utils.n

            bit = self._get_codeword_bit(token_position)
            self.token_bit_log.append(bit)

            target_ids = self._get_target_token_ids(bit, block_index, input_ids[b])
            mask = self._get_bias_mask(scores[b], target_ids)

            if self.config.scheme == 'hard':
                scores[b] = self._bias_logits_hard(scores[b], mask)
            elif self.config.scheme == 'soft':
                adaptive_delta = self._compute_adaptive_delta(scores[b], mask, self.config.delta)
                scores[b] = self._bias_logits_soft(scores[b], mask, adaptive_delta)
            else:
                raise ValueError(f"Invalid scheme: {self.config.scheme}. Choose 'soft' or 'hard'.")

        return scores


class BREW(BaseWatermark):

    def __init__(self, algorithm_config: str | BREWConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = BREWConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, BREWConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a BREWConfig instance")

        self.utils = BREWUtils(self.config)
        self.logits_processor = BREWLogitsProcessor(self.config, self.utils)

    @staticmethod
    def cyclic_shift(bits: list[int], shift: int, direction: str = 'left') -> list[int]:
        """
        Kept for compatibility, but no longer used for insertion/deletion detection.
        The detector below uses global linear offsets on the full token stream.
        """
        if direction == 'left':
            return bits[shift:] + bits[:shift]
        elif direction == 'right':
            return bits[-shift:] + bits[:-shift]
        else:
            raise ValueError(f"Invalid shift direction: {direction}")

    @staticmethod
    def make_token_blocks_with_global_offset(
        token_stream: list[int],
        start_idx: int,
        n: int,
        max_blocks: int,
        offset: int,
    ) -> list[list[int]]:
        """
        Apply one global linear offset to the whole token stream, then split it
        into n-token blocks.

        For each block j, tokens are later converted to bits using the
        j-specific vocabulary partition generated by seed_j = H(K, j).
        """
        offset_start = start_idx + offset

        if offset_start < 0:
            return []

        available = len(token_stream) - offset_start
        if available < n:
            return []

        num_blocks = min(max_blocks, available // n)

        return [
            token_stream[offset_start + j * n : offset_start + (j + 1) * n]
            for j in range(num_blocks)
        ]

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )

        encoded_prompt = self.config.generation_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.config.device)

        encoded_prompt = {k: v[:1] for k, v in encoded_prompt.items()}

        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        watermarked_text = self.config.generation_tokenizer.batch_decode(
            encoded_watermarked_text,
            skip_special_tokens=True
        )[0]

        return watermarked_text

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:

        generate_without_watermark = partial(
            self.config.generation_model.generate,
            **self.config.gen_kwargs
        )

        encoded_prompt = self.config.generation_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.config.device)

        encoded_prompt = {k: v[:1] for k, v in encoded_prompt.items()}

        encoded_unwatermarked_text = generate_without_watermark(**encoded_prompt)
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(
            encoded_unwatermarked_text,
            skip_special_tokens=True
        )[0]

        return unwatermarked_text

    def detect_watermark(self, prompt: str, text: str, return_dict: bool = True, *args, **kwargs):

        tokenizer = self.config.generation_tokenizer
        device = self.config.device
        n = self.utils.n
        C = self.utils.C
        F = self.utils.F
        A = C.ambient_space()
        dec = C.decoder()
        max_shift = self.config.max_shift_bit

        detect_prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )["input_ids"]

        encoded_text = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].to(device)

        token_stream = encoded_text.tolist()

        prompt_len = detect_prompt_ids.shape[1]
        start_idx = 0 if len(encoded_text) <= (prompt_len - 1) else (prompt_len - 1)

        full_gt_list = getattr(self.logits_processor, "codeword_queue", None)
        if not full_gt_list:
            out = {
                "is_watermarked": False,
                "reason": "no_ground_truth_codewords",
                "matched": 0,
                "total": 0,
                "best_offset": None,
            }
            return out if return_dict else False

        def _decode_to_code_safe(bits_list):
            try:
                v = A(vector(F, bits_list))
                c_hat = dec.decode_to_code(v)
                return list(A(c_hat))
            except Exception as e:
                msg = str(e)
                if (
                    "Decoding failed because the number of errors exceeded the decoding radius" in msg
                    or "Decoding failed" in msg
                    or e.__class__.__name__ == "DecodingError"
                ):
                    return None
                raise

        def hamming(a, b):
            return sum(x != y for x, y in zip(a, b))

        max_blocks = len(full_gt_list)

        best_result = {
            "is_watermarked": False,
            "matched": 0,
            "total": 0,
            "match_percent": 0.0,
            "best_offset": None,
            "match_info": [],
        }

        for s in range(-max_shift, max_shift + 1):
            token_segments = BREW.make_token_blocks_with_global_offset(
                token_stream=token_stream,
                start_idx=start_idx,
                n=n,
                max_blocks=max_blocks,
                offset=s,
            )

            num_segments = len(token_segments)
            if num_segments == 0:
                continue

            gt_list = full_gt_list[:num_segments]

            matched_s = 0
            match_info_s = []

            for j, (token_block, gt_bits) in enumerate(zip(token_segments, gt_list)):
                token_block_t = torch.tensor(token_block, dtype=torch.long, device=device)

                seg_bits = self.utils.tokens_to_bits_for_block(
                    token_block_t,
                    block_index=j
                ).tolist()

                raw_errors = hamming(seg_bits, gt_bits)
                c_hat_bits = _decode_to_code_safe(seg_bits)

                if c_hat_bits is not None and c_hat_bits == gt_bits:
                    matched_s += 1
                    match_info_s.append({
                        "success": True,
                        "offset": s,
                        "block_index": j,
                        "raw_errors": raw_errors,
                    })
                else:
                    match_info_s.append({
                        "success": False,
                        "offset": s,
                        "block_index": j,
                        "raw_errors": raw_errors,
                    })

            match_percent_s = matched_s / num_segments * 100.0
            threshold_s = self.config.z_threshold * num_segments / 100.0
            is_watermarked_s = matched_s > threshold_s

            candidate_result = {
                "is_watermarked": is_watermarked_s,
                "matched": matched_s,
                "total": num_segments,
                "match_percent": match_percent_s,
                "best_offset": s,
                "match_info": match_info_s,
            }

            # Select the best global offset.
            # Primary criterion: match percent.
            # Tie-breaker: number of matched blocks.
            if (
                candidate_result["match_percent"] > best_result["match_percent"]
                or (
                    candidate_result["match_percent"] == best_result["match_percent"]
                    and candidate_result["matched"] > best_result["matched"]
                )
            ):
                best_result = candidate_result

        return best_result if return_dict else best_result["is_watermarked"]

    def analyze_watermark_errors(self, prompt: str, text: str, return_dict: bool = True, debug: bool = True):

        tokenizer = self.config.generation_tokenizer
        device = self.config.device
        n = self.utils.n
        C = self.utils.C
        F = self.utils.F
        A = C.ambient_space()
        dec = C.decoder()
        max_shift = self.config.max_shift_bit

        def _decode_to_code_safe(bits_list):
            try:
                v = A(vector(F, bits_list))
                c_hat = dec.decode_to_code(v)
                return list(A(c_hat))
            except Exception as e:
                msg = str(e)
                if (
                    "Decoding failed because the number of errors exceeded the decoding radius" in msg
                    or "Decoding failed" in msg
                    or e.__class__.__name__ == "DecodingError"
                ):
                    return None
                raise

        def hamming(a, b):
            return sum(x != y for x, y in zip(a, b))

        detect_prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )["input_ids"]

        encoded_text = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].to(device)

        token_stream = encoded_text.tolist()

        prompt_len = detect_prompt_ids.shape[1]
        start_idx = 0 if len(encoded_text) <= (prompt_len - 1) else (prompt_len - 1)

        full_gt_list = getattr(self.logits_processor, "codeword_queue", None)
        if not full_gt_list:
            out = {
                "is_watermarked": False,
                "reason": "no_ground_truth_codewords",
                "matched": 0,
                "total": 0,
                "best_offset": None,
            }
            return out if return_dict else False

        max_blocks = len(full_gt_list)

        best_summary = {
            "is_watermarked": False,
            "matched": 0,
            "total": 0,
            "match_rate": 0.0,
            "match_percent": 0.0,
            "total_final_errors": 0,
            "total_final_errors_rate": 0.0,
            "best_offset": None,
            "per_offset": [],
        }

        per_offset_summaries = []

        for s in range(-max_shift, max_shift + 1):
            token_segments = BREW.make_token_blocks_with_global_offset(
                token_stream=token_stream,
                start_idx=start_idx,
                n=n,
                max_blocks=max_blocks,
                offset=s,
            )

            num_segments = len(token_segments)
            if num_segments == 0:
                continue

            gt_list = full_gt_list[:num_segments]

            matched_s = 0
            total_errors_s = 0
            block_info_s = []

            if debug:
                print(f"\n[Global offset {s}] Checking {num_segments} blocks")

            for j, (token_block, gt_bits) in enumerate(zip(token_segments, gt_list)):
                token_block_t = torch.tensor(token_block, dtype=torch.long, device=device)

                seg_bits = self.utils.tokens_to_bits_for_block(
                    token_block_t,
                    block_index=j
                ).tolist()

                raw_errors = hamming(seg_bits, gt_bits)
                total_errors_s += raw_errors

                c_hat_bits = _decode_to_code_safe(seg_bits)
                matched_here = c_hat_bits is not None and c_hat_bits == gt_bits

                if matched_here:
                    matched_s += 1
                    if debug:
                        print(f"[Offset {s} | Compare #{j}] ✅ Match ({raw_errors} errors)")
                else:
                    if debug:
                        print(f"[Offset {s} | Compare #{j}] ❌ No match ({raw_errors} / {n})")

                block_info_s.append({
                    "success": matched_here,
                    "offset": s,
                    "block_index": j,
                    "raw_errors": raw_errors,
                })

            total_bits = num_segments * n if num_segments > 0 else 1
            match_rate_s = matched_s / num_segments if num_segments > 0 else 0.0
            match_percent_s = match_rate_s * 100.0

            threshold_s = self.config.z_threshold * num_segments / 100.0
            is_watermarked_s = matched_s > threshold_s

            summary_s = {
                "is_watermarked": is_watermarked_s,
                "matched": matched_s,
                "total": num_segments,
                "match_rate": match_rate_s,
                "match_percent": match_percent_s,
                "total_final_errors": total_errors_s,
                "total_final_errors_rate": total_errors_s / total_bits,
                "best_offset": s,
                "block_info": block_info_s,
            }

            per_offset_summaries.append(summary_s)

            if (
                summary_s["match_percent"] > best_summary["match_percent"]
                or (
                    summary_s["match_percent"] == best_summary["match_percent"]
                    and summary_s["matched"] > best_summary["matched"]
                )
            ):
                best_summary = summary_s

        best_summary["per_offset"] = per_offset_summaries

        return best_summary if return_dict else best_summary["is_watermarked"]
