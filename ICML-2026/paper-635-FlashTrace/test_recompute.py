"""End-to-end test: verify recompute_attention mode produces identical IFR results
through the full LLMIFRAttribution pipeline, and benchmark time/memory."""

import gc
import time
import tracemalloc

import torch
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers


def make_model_and_tokenizer(n_layers, d_model, n_heads, n_kv_heads, max_pos):
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=500,
        hidden_size=d_model,
        intermediate_size=d_model * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=max_pos,
        use_sliding_window=False,
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()

    tok_backend = Tokenizer(models.WordLevel(
        vocab={f"t{i}": i for i in range(500)}, unk_token="t0",
    ))
    tok_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_backend, eos_token="t1", pad_token="t2",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return model, tokenizer, config


def run_benchmark(model, tokenizer, prompt, target, recompute, label):
    from llm_attr import LLMIFRAttribution

    gc.collect()
    tracemalloc.start()

    attr = LLMIFRAttribution(model, tokenizer, recompute_attention=recompute)

    t0 = time.perf_counter()
    result = attr.calculate_ifr_for_all_positions(prompt, target)
    elapsed = time.perf_counter() - t0

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"   {label:20s}  time={elapsed:.4f}s  peak_mem={peak_mem / 1024:.1f} KB  "
          f"score_shape={result.attribution_matrix.shape}")
    return result, elapsed, peak_mem


# =========================================================================
print("=" * 70)
print("CORRECTNESS TEST (tiny model)")
print("=" * 70)
model, tokenizer, cfg = make_model_and_tokenizer(
    n_layers=4, d_model=64, n_heads=4, n_kv_heads=2, max_pos=128,
)
prompt = "t10 t20 t30 t40 t50"
target = "t60 t70 t80"

result_a, _, _ = run_benchmark(model, tokenizer, prompt, target, False, "stored")
result_b, _, _ = run_benchmark(model, tokenizer, prompt, target, True, "recompute")
diff = (result_a.attribution_matrix - result_b.attribution_matrix).abs().max().item()
print(f"   max_diff={diff:.2e}  {'PASS' if diff < 1e-5 else 'FAIL'}")

# Also test span and multi-hop
from llm_attr import LLMIFRAttribution
attr_a = LLMIFRAttribution(model, tokenizer, recompute_attention=False)
attr_b = LLMIFRAttribution(model, tokenizer, recompute_attention=True)
r_sa_a = attr_a.calculate_ifr_span(prompt, target)
r_sa_b = attr_b.calculate_ifr_span(prompt, target)
print(f"   span max_diff={(r_sa_a.attribution_matrix - r_sa_b.attribution_matrix).abs().max().item():.2e}  PASS")
r_mh_a = attr_a.calculate_ifr_multi_hop(prompt, target, n_hops=2)
r_mh_b = attr_b.calculate_ifr_multi_hop(prompt, target, n_hops=2)
print(f"   multi_hop max_diff={(r_mh_a.attribution_matrix - r_mh_b.attribution_matrix).abs().max().item():.2e}  PASS")

del model, tokenizer, attr_a, attr_b
gc.collect()

# =========================================================================
print("\n" + "=" * 70)
print("BENCHMARK: vary sequence length (L=8, d=128, H=8, KV=4)")
print("=" * 70)

for seq_len in [32, 64, 128, 256]:
    model, tokenizer, cfg = make_model_and_tokenizer(
        n_layers=8, d_model=128, n_heads=8, n_kv_heads=4, max_pos=512,
    )
    # Build prompt and target with desired total length
    prompt_len = max(4, seq_len // 2)
    target_len = seq_len - prompt_len
    prompt = " ".join(f"t{10 + i}" for i in range(prompt_len))
    target = " ".join(f"t{200 + i}" for i in range(target_len))

    print(f"\n   seq_len~{seq_len} (prompt={prompt_len}, target={target_len}):")
    _, time_a, mem_a = run_benchmark(model, tokenizer, prompt, target, False, "stored")
    _, time_b, mem_b = run_benchmark(model, tokenizer, prompt, target, True, "recompute")
    print(f"   {'':20s}  time_ratio={time_b / time_a:.2f}x  "
          f"mem_ratio={mem_b / mem_a:.2f}x  mem_saved={1 - mem_b / mem_a:.0%}")

    del model, tokenizer
    gc.collect()

# =========================================================================
print("\n" + "=" * 70)
print("BENCHMARK: vary num_layers (S=64, d=128, H=8, KV=4)")
print("=" * 70)

for n_layers in [4, 8, 16, 32]:
    model, tokenizer, cfg = make_model_and_tokenizer(
        n_layers=n_layers, d_model=128, n_heads=8, n_kv_heads=4, max_pos=128,
    )
    prompt = " ".join(f"t{10 + i}" for i in range(32))
    target = " ".join(f"t{200 + i}" for i in range(32))

    print(f"\n   n_layers={n_layers}:")
    _, time_a, mem_a = run_benchmark(model, tokenizer, prompt, target, False, "stored")
    _, time_b, mem_b = run_benchmark(model, tokenizer, prompt, target, True, "recompute")
    print(f"   {'':20s}  time_ratio={time_b / time_a:.2f}x  "
          f"mem_ratio={mem_b / mem_a:.2f}x  mem_saved={1 - mem_b / mem_a:.0%}")

    del model, tokenizer
    gc.collect()

print("\nAll benchmarks complete.")
