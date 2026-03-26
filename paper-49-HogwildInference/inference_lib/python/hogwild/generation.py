import random
from copy import deepcopy
from typing import Dict, NamedTuple, Sequence, Optional

import numpy as np
import torch
import transformers

from hogwild.attention import model_surgery, HogwildCache, merge_caches
from hogwild.formatting import FormattingBase, MathFormatting

ReasoningState = NamedTuple("ReasoningState", (
    ("history", Sequence[int]), ("current_step_tokens_by_worker", Sequence[Sequence[int]])),)


def solve_math_2agents(*args, **kwargs):
    raise NotImplementedError("""Use solve_task_2agents""")


def solve_task_2agents(
        *,
        problem: str,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        finisher_max_new_tokens: int = 16,
        fmt: Optional[FormattingBase] = None,
        **reasoning_kwargs,
) -> Dict[int, str]:
    """Generate reasoning traces with 2 parallel agents, return responses (with s1-like finisher)"""
    fmt = fmt if fmt is not None else MathFormatting(tokenizer)
    model_surgery(model)
    saved_reasoning_states = generate_reasoning_2agents(
        problem=problem, model=model, tokenizer=tokenizer, fmt=fmt, **reasoning_kwargs)
    outputs = dict()
    for budget, reasoning_state in saved_reasoning_states.items():
        generated_tokens = list(reasoning_state.history)
        generated_tokens.extend(tokenizer.encode(fmt.current_step_header, add_special_tokens=False))
        for worker_index, worker_tokens in enumerate(reasoning_state.current_step_tokens_by_worker):
            generated_tokens.extend(worker_tokens)
            generated_tokens.extend(tokenizer.encode(fmt.incomplete_step + fmt.sep, add_special_tokens=False))
        problem_tokens = list(tokenizer.encode(fmt.apply_chat_template(problem), add_special_tokens=False))
        response = tokenizer.decode(problem_tokens + generated_tokens)
        if finisher_max_new_tokens > 0 and (fmt.get_final_answer(tokenizer.decode(generated_tokens)) is None):
            response = finalize_response_with_s1_finisher(
                response=response, model=model, tokenizer=tokenizer, fmt=fmt, max_new_tokens=finisher_max_new_tokens
            )
        outputs[budget] = response
    return outputs


def generate_reasoning_2agents(
        *,
        problem: str,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        fmt: FormattingBase,
        max_steps: int,
        save_on_steps: Sequence[int] = (),
        insert_s1_collab_message_every_tokens: int = 1024,
        suppress_tokens: Sequence[int] = (),
) -> Dict[int, ReasoningState]:
    """Generate reasoning traces and return snapshot for a given max_steps and any extra snapshots for save_on_steps"""
    assert all(save_step <= max_steps for save_step in save_on_steps), save_on_steps
    saved_states = dict()
    logits_processor = get_logits_processor(model, suppress_tokens)
    device = next(model.parameters()).device
    tokenizer_kwargs = dict(return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)

    tokens_since_last_wait = 0
    cache_common, cache_current_step_header, cache_own_header, cache_w1, cache_w2 = (transformers.DynamicCache() for _ in range(5))
    cm = HogwildCache(cache_structure=[
        [cache_common, cache_current_step_header, cache_w2, cache_own_header, cache_w1],
        [cache_common, cache_current_step_header, cache_w1, cache_own_header, cache_w2],
    ], write_to=[cache_w1, cache_w2], model=model)

    # pre-fill common cache parts
    with torch.inference_mode():
        model(**tokenizer(fmt.apply_chat_template(problem), **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=HogwildCache([[cache_common]], model=model))  # <-- write to common prompt

        model(**tokenizer(fmt.current_step_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=HogwildCache([[cache_current_step_header]], model=model))  # <-- write to the separator after history

        model(**tokenizer(fmt.current_worker_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=HogwildCache([[cache_own_header]], model=model))  # <-- write to separator between incomplete steps

    #model = torch.compile(model)

    # generate interdependent reasoning chains in parallel
    current_step_index_by_worker = [1, 1]
    current_step_tokens_by_worker = tokenizer(list(fmt.worker_prompts), add_special_tokens=False)['input_ids']
    history = list()

    next_inputs = tokenizer(list(fmt.worker_prompts), **tokenizer_kwargs).to(device)
    for inference_step in range(max_steps):
        # run model with a shared cache (batched inference)
        with torch.inference_mode():
            logits = model(**cm.get_input_kwargs(**next_inputs)).logits[..., -1, :]
            logits = logits_processor(next_inputs['input_ids'], logits)
            new_tokens = torch.multinomial(logits.softmax(dim=-1), 1).flatten(
                ) if model.generation_config.do_sample else logits.argmax(-1)
            assert len(new_tokens) == len(fmt.workers)

        # process generated tokens for printing; handle step change, update next_inputs
        next_input_tokens = new_tokens.unsqueeze(-1).tolist()
        for worker_index, (worker_name, worker_tokens, new_token) in enumerate(
                zip(fmt.workers, current_step_tokens_by_worker, new_tokens.tolist())):
            worker_tokens.append(new_token)
            if fmt.is_end_of_step(worker_tokens):
                # worker just finished their step - add it to common history and start a new step
                current_step_index_by_worker[worker_index] += 1
                history.extend(worker_tokens)
                worker_tokens.clear()
                start_msg = fmt.get_step_prefix(worker_name, current_step_index_by_worker[worker_index])
                if tokens_since_last_wait > insert_s1_collab_message_every_tokens:
                    start_msg += fmt.s1_collab_message
                    tokens_since_last_wait = 0
                worker_tokens.extend(tokenizer.encode(start_msg, add_special_tokens=False))
                merge_caches(cache_common, cm.cache_structure[worker_index][-1], model.model)
                cm.cache_structure[worker_index][-1].crop(0)
                next_input_tokens[worker_index] = [new_token] + worker_tokens
            tokens_since_last_wait += len(next_input_tokens[worker_index])
        next_inputs = tokenizer.pad(
            dict(input_ids=next_input_tokens), padding_side='left', return_tensors='pt').to(device)

        if torch.any(new_tokens == tokenizer.eos_token_id).item():
            break  # at least one worker generated the end-of-sequence token, finish early

        if inference_step in save_on_steps:
            saved_states[inference_step] = ReasoningState(
                history=list(history), current_step_tokens_by_worker=deepcopy(current_step_tokens_by_worker),
            )

    # if we finished early, copy the reasoning state for all subsequent budgets
    for step_to_save in set(save_on_steps) | {max_steps}:
        if step_to_save not in saved_states:
            saved_states[step_to_save] = ReasoningState(
                history=list(history), current_step_tokens_by_worker=deepcopy(current_step_tokens_by_worker),
            )
    return saved_states


def finalize_response_with_s1_finisher(
        *, response: str, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer,
        fmt: FormattingBase, max_new_tokens: int, chunk_size: int = 4096) -> str:
    """Compile a response from a reasoning state. If there is no answer yet, prompt the model to give the answer"""
    with torch.inference_mode():
        device = next(model.parameters()).device
        response_ids = tokenizer.encode(response + fmt.s1_finisher_suffix, add_special_tokens=False)
        assert isinstance(response_ids, Sequence) and isinstance(response_ids[0], int)
        prefix_num_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        assert prefix_num_tokens < len(response_ids)
        cache = transformers.DynamicCache()

        # encode prompt in chunks to save memory
        next_logits = None
        for chunk_start in range(0, len(response_ids), chunk_size):
            next_logits = model(
                input_ids=torch.tensor([response_ids[chunk_start: chunk_start + chunk_size]],
                                       device=device, dtype=torch.int64),
                attention_mask=torch.ones(1, min(chunk_start + chunk_size, len(response_ids)),
                                          device=device, dtype=torch.int64),
                use_cache=True, past_key_values=HogwildCache([[cache]], model=model)
            ).logits[..., -1, :]  # [batch_size(1), vocab_size]
            assert cache.get_seq_length() == min(chunk_start + chunk_size, len(response_ids))

        # run max_new_steps of *always greedy* output generation
        next_tokens = next_logits.argmax(-1, keepdims=True)  # [batch_size(1), 1]
        response_ids.append(next_tokens.item())
        for inference_step in range(max_new_tokens - 1):
            full_mask = torch.ones(next_tokens.shape[0], len(response_ids), device=device, dtype=torch.int64)
            next_logits = model(
                input_ids=next_tokens, attention_mask=full_mask, use_cache=True, past_key_values=HogwildCache([[cache]], model=model)
            ).logits[..., -1, :]
            next_tokens = next_logits.argmax(-1, keepdims=True)  # [batch_size(1), 1]
            response_ids.append(next_tokens.item())
            if response_ids[-1] == tokenizer.eos_token_id:
                break
            if fmt.get_final_answer(tokenizer.decode(response_ids[prefix_num_tokens:])) is not None:
                break  # generated a valid response - finish early
        response: str = tokenizer.decode(response_ids)
    return response


def get_logits_processor(model: transformers.PreTrainedModel, suppress_tokens: Sequence[int] = ()):
    """Create a transformers class that post-processes model logits for nucleus sampling, banned tokens, etc"""
    generation_config, model_kwargs = model._prepare_generation_config(model.generation_config)
    model._prepare_special_tokens(generation_config)
    device = next(model.parameters()).device
    return model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=0,
        encoder_input_ids=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=transformers.LogitsProcessorList([
            transformers.generation.logits_process.SuppressTokensLogitsProcessor(
                suppress_tokens, device=device)]),
        device=device,
        model_kwargs=model_kwargs
    )


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)