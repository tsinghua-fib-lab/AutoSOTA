"""
Build a null-model baseline for prototype interpretability scoring.

For each (layer, prototype) pair this script samples n sentences uniformly
at random from the global pool of sentences collected by
find_proto_activations.py.  The resulting JSON has the same structure as the
real analysis JSON and is consumed by run_LLM_scoring.py to compute a
baseline disentanglement score.

Usage:
    python create_null_activations.py
"""

import json
import os
import random
from typing import List, Dict, Any


# --- CONFIGURATION ---
NULL_SENTENCES_PER_PROTO = 10   # Sentences drawn per (layer, prototype)
NULL_RANDOM_SEED = 99

# Paths — update these to match your find_proto_activations.py output directory
INPUT_JSON_PATH  = "prototype_analysis_word_level_original/prototype_analysis_word_level_original.json"
OUTPUT_JSON_PATH = "prototype_analysis_word_level_original/prototype_analysis_word_level_original_null.json"


def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_null_model_sentences(
    top_sentences: Dict[str, Dict[str, List[Dict[str, Any]]]],
    n_per_proto: int,
    seed: int,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Sample n_per_proto sentences uniformly at random for every (layer, prototype)
    pair.  Sentences are drawn from the union of all real top-sentence records,
    deduplicated by sentence text, so no sentence appears twice in the pool.
    """
    rng = random.Random(seed)

    # Build a deduplicated global pool across all layers and prototypes
    seen_texts: set = set()
    flat_pool: List[Dict[str, Any]] = []
    for proto_dict in top_sentences.values():
        for records in proto_dict.values():
            for rec in records:
                txt = rec.get("sentence_text", "")
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    flat_pool.append(rec)

    print(f"Null model pool: {len(flat_pool)} unique sentences")

    null_sentences: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for layer_key, proto_dict in top_sentences.items():
        null_sentences[layer_key] = {}
        for proto_key in proto_dict:
            n = min(n_per_proto, len(flat_pool))
            sampled = rng.sample(flat_pool, n)

            null_recs = []
            for i, rec in enumerate(sampled):
                rec = dict(rec)  # shallow copy — do not mutate the shared pool
                rec["rank"] = i + 1
                null_recs.append(rec)

            null_sentences[layer_key][proto_key] = null_recs

    return null_sentences


def save_json(data: Dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to: {output_path}")


def main():
    print("Building null model baseline...")

    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(
            f"Input JSON not found at '{INPUT_JSON_PATH}'. "
            "Run find_proto_activations.py first."
        )

    print(f"Loading prototype analysis from {INPUT_JSON_PATH}...")
    top_sentences = load_json(INPUT_JSON_PATH)

    print("Sampling null model sentences...")
    null_sentences = build_null_model_sentences(
        top_sentences=top_sentences,
        n_per_proto=NULL_SENTENCES_PER_PROTO,
        seed=NULL_RANDOM_SEED,
    )

    save_json(null_sentences, OUTPUT_JSON_PATH)
    print(f"\nDone! Null model saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
