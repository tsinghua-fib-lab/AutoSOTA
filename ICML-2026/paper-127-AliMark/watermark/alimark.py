import argparse
import copy
import random
import time
import numpy as np

import nltk
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

from watermark.adaptive_bit_sequence_alignment import \
    AdaptiveBitSequenceAlignment
from watermark.restructurer import ReStructurer
from watermark.utils import Utils

nltk.download(['punkt_tab'])


class AliMark:
    _NEW_TOKENS = 128
    _TOP_P = 0.95
    _TEMPERATURE = 0.7
    _REPEATION_PENALTY = 1.15
    _MAX_GENERATION_ATTEMPTS = 5

    def __init__(self, args, load_llm=True):
        # 0. Arguments
        self._model_name = args.watermark_model
        self._embedder_name = args.watermark_embedder
        self._num_next_sentence_candidates = args.watermark_num_next_sentence_candidates
        self._min_new_sentences = args.min_new_sentences

        # 1. Load Model and Tokenizer
        self._device = args.device
        if load_llm:
            self._vllm_model = LLM(
                model=self._model_name,
                gpu_memory_utilization=args.vllm_gpu_mem_util,
                enforce_eager=True,
            )
            self._tokenizer = self._vllm_model.get_tokenizer()
            self._max_model_len = self._vllm_model.llm_engine.model_config.max_model_len
            self._bad_words = ['_']
            self._bad_token_ids = []
            for word in self._bad_words:
                ids = self._tokenizer.encode(word, add_special_tokens=False)
                self._bad_token_ids.extend(ids)
            if self._tokenizer.pad_token is None:
                print("Tokenizer has no pad_token, setting it to eos_token.")
                self._tokenizer.pad_token = self._tokenizer.eos_token
                print(f"Tokenizer pad_token set to: {self._tokenizer.pad_token} (id: {self._tokenizer.pad_token_id})")
            else:
                print(f"Tokenizer pad_token: {self._tokenizer.pad_token} (id: {self._tokenizer.pad_token_id})")
        
        # 2. Watermark Related
        self._utils = Utils(args)

        # Secret Keys
        self._secret_bit_sequence = copy.deepcopy(self._utils.secret_bit_sequence)
        self._secret_vectors = (
            self._utils.secret_vectors
            .detach()
            .clone()
            .to(dtype=torch.float32, device=self._device)
        )

        # Embedder
        self._embedder = SentenceTransformer(self._embedder_name, device=self._device)

        # Detection
        self._rs_dropout = getattr(args, 'watermark_rs_dropout', 0.0)
        self._RS = ReStructurer()
        self._ABSA = AdaptiveBitSequenceAlignment(args)

    def _model_generate_vllm(self, curr_prompt, n=1):
        sampling_params = SamplingParams(
            n=n,
            max_tokens=self._NEW_TOKENS,
            temperature=self._TEMPERATURE,
            top_p=self._TOP_P,
            repetition_penalty=self._REPEATION_PENALTY,
        )

        new_texts = []
        outputs = self._vllm_model.generate(
                [curr_prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

        for output in outputs:
            for completion in output.outputs:
                new_text = completion.text.strip()
                new_texts.append(new_text)

        return new_texts

    def _extract_bit_signals(self, sentences):
        sentence_embeddings = self._embedder.encode(sentences, convert_to_tensor=True, show_progress_bar=False) # [N, d]
        # get cosine similarities with secret vectors
        dot_products = torch.matmul(sentence_embeddings, self._secret_vectors.T)  # [N, block_size]
        bit_signals = (dot_products > 0).int().tolist()  # convert to bits
        return bit_signals

    @torch.no_grad()
    def generate_unwatermarked_text(self, prompt):
        s_time = time.time()
        generated_text = ""
        
        while len(sent_tokenize(generated_text)) < self._min_new_sentences:
            # 1. Prepare inputs
            curr_prompt = prompt + generated_text
            curr_prompt_tokens = self._tokenizer.encode(curr_prompt)
            prompt_len = len(curr_prompt_tokens)
            if prompt_len > self._max_model_len - self._NEW_TOKENS:
                break

            # 2. Generate
            new_texts = self._model_generate_vllm(curr_prompt, n=1)
            new_text = new_texts[0].strip()
            if new_text == "":
                break
            new_text = new_text.replace("\n","").replace("“","\"").replace("”","\"").rstrip()
            if not new_text or new_text[-1] not in {'.', '?', '!', '\"'}:
                new_text += '.'
            
            generated_text += " " + new_text
            # break
        
        # 3. Trim to exact number of sentences
        new_sents = sent_tokenize(generated_text)
        generated_text = " ".join(new_sents[:self._min_new_sentences])
        e_time = time.time()
        print(f"Unwatermarked generation time: {e_time - s_time:.2f} seconds")
        return generated_text.strip()
    
    @torch.no_grad()
    def generate_watermarked_text(self, prompt):
        s_time = time.time()
        generated_text = ""
        
        curr_sent_id = 0
        while len(sent_tokenize(generated_text)) < self._min_new_sentences:
            # 1. Prepare inputs and make sure they are shorter than maximum context length
            curr_prompt = prompt + generated_text
            
            curr_prompt_tokens = self._tokenizer.encode(curr_prompt)
            prompt_len = len(curr_prompt_tokens)
            if prompt_len > self._max_model_len - self._NEW_TOKENS:
                break

            # 2. Generate Candidates
            next_sentence_candidates = []
            for _ in range(self._MAX_GENERATION_ATTEMPTS):
                new_texts = self._model_generate_vllm(curr_prompt, n=self._num_next_sentence_candidates)
                for new_text in new_texts:
                    new_text = new_text.strip()
                    if new_text == "":
                        continue
                    next_sentence = sent_tokenize(new_text)[0]
                    next_sentence = next_sentence.replace("\n","").replace("“","\"").replace("”","\"").rstrip()
                    if not next_sentence or next_sentence[-1] not in {'.', '?', '!', '\"'}:
                        next_sentence += '.'
                    next_sentence_candidates.append(next_sentence)
                next_sentence_candidates = list(set(next_sentence_candidates))  # deduplicate candidates
                if len(next_sentence_candidates) > 0:
                    break
            if len(next_sentence_candidates) == 0:
                break

            # 3. Extract Bit Signals and Select Next Sentence
            bit_signals = self._extract_bit_signals(next_sentence_candidates)
            target_bit_signals = self._secret_bit_sequence[curr_sent_id]
            
            best_candidates = []
            best_score = -1
            for candidate, bit_signal in zip(next_sentence_candidates, bit_signals):
                score = sum(1 for a, b in zip(bit_signal, target_bit_signals) if a == b)
                if score > best_score:
                    best_score = score
                    best_candidates = [candidate]
                elif score == best_score:
                    best_candidates.append(candidate)

            if len(best_candidates) == 0:
                break
            random.shuffle(best_candidates)
            selected_sentence = best_candidates[0]  # can also randomly select among best candidates
            generated_text += " " + selected_sentence
            
            curr_sent_id += 1

            # print(f"Extracted Bit Signals: {bit_signals}")
            # print(f"Shape of Extracted Bit Signals: {len(bit_signals)} x {len(bit_signals[0])}")
            # break

        e_time = time.time()
        print(f"Watermarked generation time: {e_time - s_time:.2f} seconds")
        return generated_text.strip()

    @torch.no_grad()
    def detect_watermark(self, text, 
                         rs_args={
                             "rs_enable_merge": True,
                             "rs_enable_split": True,
                             "rs_dropout": 0.0,
                             "rs_multistep": False,
                         },
                         absa_args={
                             "absa_lower_ratio": 0.4, 
                             "absa_upper_ratio": 2.0, 
                             "absa_criterion": "block_edit_distance"
                         }
                         ):
        s_time = time.time()

        rs_enable_merge = rs_args["rs_enable_merge"]
        rs_enable_split = rs_args["rs_enable_split"]
        rs_dropout = rs_args["rs_dropout"]
        rs_multistep = rs_args["rs_multistep"]
        absa_lower_ratio = absa_args["absa_lower_ratio"]
        absa_upper_ratio = absa_args["absa_upper_ratio"]
        absa_criterion = absa_args["absa_criterion"]

        assert 0.0 <= absa_lower_ratio <= absa_upper_ratio, "lower_ratio must be between 0 and upper_ratio"
        assert 0.0 <= rs_dropout <= 1.0, "rs_dropout must be between 0 and 1"
        
        candidates = []
        # --- Strategy (0): Original sentences, to be added to candidates later ---
        original_sentences = sent_tokenize(text)

        # --- Strategy (1): Traverse all merge cases ---
        if rs_enable_merge:
            merge_candidates = self._RS.gen_re_merge_candidates(original_sentences)
            candidates.extend(merge_candidates)
        
        # --- Strategy (2): Traverse all split cases ---
        if rs_enable_split:
            split_candidates = self._RS.gen_re_split_candidates(original_sentences)
            candidates.extend(split_candidates)

        # --- Strategy (2.5): Multi-step merge/split (if enabled) ---
        if rs_multistep:
            multistep_candidates = self._RS.gen_one_resplit_and_one_remerge_candidates(original_sentences)
            candidates.extend(multistep_candidates)

        # --- Randomly drop some candidates based on rs_dropout
        if rs_dropout > 0.0:
            num_candidates = len(candidates)
            if num_candidates > 0:
                num_to_drop = int(num_candidates * rs_dropout)
                actual_drop_num = min(num_to_drop, num_candidates)
                if actual_drop_num > 0:
                    drop_indices = random.sample(range(num_candidates), actual_drop_num)
                    candidates = [cand for idx, cand in enumerate(candidates) if idx not in drop_indices]
                    # print(f"Debug: After RS dropout (fixed ratio), dropped {actual_drop_num}/{num_candidates}, {len(candidates)} remain.")        
        
        # --- Strategy (3): Add original sentences to the candidate list to ensure basic detection accuracy ---
        candidates.extend([original_sentences])

         # Deduplication: Different operation sequences may produce the same result; use a set to deduplicate and save computational resources
        unique_candidates = []
        seen = set()
        for cand in candidates:
            t_cand = tuple(cand)
            if t_cand not in seen:
                seen.add(t_cand)
                unique_candidates.append(cand)
        # print(f"Debug: Exhaustive search generated {len(unique_candidates)} unique segmentation variants from {len(original_sentences)} sentences.")

        # Pre-compute embeddings for all unique sentences across candidates
        unique_sentences = set()
        for candidate in unique_candidates:
            unique_sentences.update(candidate)
        unique_sentences_list = list(unique_sentences)
        unique_sentneces_bit_signals = self._extract_bit_signals(unique_sentences_list)
        # print(f"Debug: Find {len(unique_sentneces_bit_signals)} unique sentence embeddings from {len(original_sentences)} sentences.")              
        # unique_embeddings = get_text_embeddings(unique_sentences_list, embedder=self._embedder)
        # print(f"Debug: Find {len(unique_embeddings)} unique sentence embeddings from {len(original_sentences)} sentences.")

        bit_signals_map = {
            sent: bit_signal for sent, bit_signal in zip(unique_sentences_list, unique_sentneces_bit_signals)
        }

        candidate_scores = []
        for _, text_candidate in enumerate(unique_candidates):
            extracted_bit_sequence = []
            for _, sent in enumerate(text_candidate):
                bit_signal = bit_signals_map[sent]
                extracted_bit_sequence.append(bit_signal)
            score = self._ABSA.compute_score(
                extracted_bit_sequence, 
                self._secret_bit_sequence, 
                lower_ratio=absa_lower_ratio, 
                upper_ratio=absa_upper_ratio,
                criterion=absa_criterion
                )
            candidate_scores.append(score)
        global_max_score = float(np.max(candidate_scores)) if candidate_scores else 0.0

        e_time = time.time()
        detect_result = {
            "score": global_max_score,
            "time": e_time - s_time,
            "num_sentences_original": len(original_sentences),
            "num_variants_evaluated": len(unique_candidates),
        }

        return detect_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark_algorithm",                        type=str,   default="AliMark")
    parser.add_argument("--watermark_model",                            type=str,   default="facebook/opt-1.3b")
    parser.add_argument("--watermark_embedder",                         type=str,   default="all-mpnet-base-v2")
    parser.add_argument('--watermark_embedding_dim',                    type=int,   default=768)
    parser.add_argument('--watermark_block_size',                       type=int,   default=4)
    parser.add_argument("--watermark_num_next_sentence_candidates",     type=int,   default=64)
    parser.add_argument("--min_new_sentences",                          type=int,   default=12)
    parser.add_argument("--dataset_name",                               type=str,   default="booksum")
    parser.add_argument("--vllm_gpu_mem_util",                          type=float, default=0.2)
    parser.add_argument("--device",                                     type=str,   default="cuda")
    parser.add_argument('--seed',                                       type=int,   default=42)
    args = parser.parse_args()
    alimark = AliMark(args)

    prompt = "Once upon a time in a land far, far away."
    unwatermarked_text = alimark.generate_unwatermarked_text(prompt)
    print("Unwatermarked Text:\n", unwatermarked_text)
    print("n_sents:", len(sent_tokenize(unwatermarked_text)))
    
    watermarked_text = alimark.generate_watermarked_text(prompt)
    print("Watermarked Text:\n", watermarked_text)
    print("n_sents:", len(sent_tokenize(watermarked_text)))

