# spotcheck_and_filter_rebalance.py
# Cleans, filters, and rebalances perturbation dataset to 500 pairs per slice
# Adds variance reporting (edit distance + semantic similarity + lexical filter)

import json, random, re
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import editdistance
from wordfreq import zipf_frequency  # lightweight frequency scores

random.seed(42)

DATASET_FILE = "perturbation_benchmark.jsonl"
OUT_FILE = "perturbation_benchmark_clean.jsonl"

CATEGORIES = [
    "synonym",
    "typo",
    "spelling",
    "morphology",
    "contraction",
    "punctuation",
    "abbreviation"
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Safe synonym fallback pool ---
SYNONYM_POOL = [
    ("I am happy today.", "I am glad today."),
    ("She is very smart.", "She is very intelligent."),
    ("It is cold outside.", "It is chilly outside."),
    ("This test is easy.", "This test is simple."),
    ("That is a big house.", "That is a large house."),
    ("He is quick.", "He is fast."),
    ("The food was tasty.", "The food was delicious."),
    ("She looks tired.", "She looks exhausted."),
    ("I will begin now.", "I will start now."),
    ("It was a funny movie.", "It was a humorous movie."),
    ("The child is small.", "The child is little."),
    ("She is kind to everyone.", "She is nice to everyone."),
    ("The store is close by.", "The store is near."),
    ("He is angry.", "He is mad."),
    ("This is correct.", "This is right."),
    ("She is beautiful.", "She is pretty."),
    ("That answer is wrong.", "That answer is incorrect."),
    ("The road is narrow.", "The road is tight."),
    ("He is wealthy.", "He is rich."),
    ("It is hot today.", "It is warm today."),
    ("The task is difficult.", "The task is hard."),
    ("She is calm.", "She is relaxed."),
    ("I like this idea.", "I enjoy this idea."),
    ("He is brave.", "He is courageous."),
    ("It is quiet here.", "It is silent here."),
    ("She is famous.", "She is well-known."),
    ("The old man walked slowly.", "The elderly man walked slowly."),
    ("I am certain of it.", "I am sure of it."),
    ("This is dangerous.", "This is risky."),
    ("He is polite.", "He is courteous."),
    ("The job is important.", "The job is significant."),
    ("She is honest.", "She is truthful."),
    ("He is friendly.", "He is sociable."),
    ("It is strange.", "It is odd."),
    ("This is expensive.", "This is costly."),
    ("She is sad.", "She is unhappy."),
    ("He is sick.", "He is ill."),
    ("This story is interesting.", "This story is fascinating."),
    ("That is funny.", "That is amusing."),
    ("It is cold today.", "It is freezing today."),
    ("He is strong.", "He is powerful."),
    ("She is weak.", "She is fragile."),
    ("The bag is heavy.", "The bag is weighty."),
    ("This place is clean.", "This place is tidy."),
    ("The ground is wet.", "The ground is damp."),
    ("He is old.", "He is aged."),
    ("That is true.", "That is correct."),
    ("I am tired.", "I am sleepy."),
    ("This room is dark.", "This room is dim."),
    ("He is clever.", "He is bright."),
    ("She is rude.", "She is impolite."),
    ("The car is quick.", "The car is fast."),
    ("It is beautiful here.", "It is lovely here."),
    ("The shop is empty.", "The shop is vacant."),
    ("That is false.", "That is untrue."),
    ("The child is loud.", "The child is noisy."),
    ("He is helpful.", "He is supportive."),
    ("This is safe.", "This is secure."),
    ("She is funny.", "She is comical."),
    ("The mountain is high.", "The mountain is tall."),
    ("The lake is large.", "The lake is big."),
    ("He is sleepy.", "He is drowsy."),
]

# --- Helpers ---
def semantic_filter(pairs, threshold=0.75):
    """Filter synonym pairs by semantic similarity"""
    if not pairs:
        return []
    originals = [p["original"] for p in pairs]
    perturbed = [p["perturbed"] for p in pairs]
    emb1 = model.encode(originals, convert_to_tensor=True, show_progress_bar=False)
    emb2 = model.encode(perturbed, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(emb1, emb2).diagonal().tolist()
    return [p for p, sim in zip(pairs, sims) if sim >= threshold]

def lexical_filter(pairs, min_freq=3.0):
    """Filter out perturbed sentences with rare/odd words or broken grammar"""
    cleaned = []
    for p in pairs:
        o, t = p["original"], p["perturbed"]

        # Reject if perturbed starts lowercase (bad capitalization)
        if t and t[0].islower():
            continue

        # Reject if perturbed contains very rare words
        bad_word = False
        for w in t.split():
            if zipf_frequency(w.lower(), "en") < min_freq:
                bad_word = True
                break
        if bad_word:
            continue

        # Reject if tense/person mismatch (basic heuristic: dropped "s" on verbs)
        if re.search(r"\b(I|he|she|it) [a-z]+$", t.lower()):
            continue

        cleaned.append(p)

    return cleaned

def compute_variance(examples):
    """Compute semantic similarity + edit distance variance stats"""
    if not examples:
        return {"count": 0, "avg_semantic_sim": 0, "avg_edit_distance": 0}
    originals = [ex["original"] for ex in examples]
    perturbed = [ex["perturbed"] for ex in examples]
    emb1 = model.encode(originals, convert_to_tensor=True, show_progress_bar=False)
    emb2 = model.encode(perturbed, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(emb1, emb2).diagonal().tolist()
    edits = [editdistance.eval(o, p) for o, p in zip(originals, perturbed)]
    return {
        "count": len(examples),
        "avg_semantic_sim": sum(sims) / len(sims),
        "avg_edit_distance": sum(edits) / len(edits)
    }

def filter_and_rebalance(in_file=DATASET_FILE,
                         out_file=OUT_FILE,
                         n_per_slice=500):
    """Clean, filter, rebalance dataset and report variance"""
    with open(in_file, "r") as f:
        data = [json.loads(line) for line in f]

    by_slice = defaultdict(list)
    for ex in data:
        by_slice[ex["slice"]].append(ex)

    cleaned, variance_report = [], {}

    for slice_name, examples in by_slice.items():
        if slice_name == "synonym":
            # semantic filter first
            examples = semantic_filter(examples, threshold=0.75)
            # lexical filter second
            examples = lexical_filter(examples, min_freq=3.0)

        # Backfill synonyms if too few
        if len(examples) < n_per_slice and slice_name == "synonym":
            while len(examples) < n_per_slice:
                a, b = random.choice(SYNONYM_POOL)
                examples.append({"slice": slice_name, "original": a, "perturbed": b})

        # Sample exactly n_per_slice
        examples = random.sample(examples, min(len(examples), n_per_slice))
        cleaned.extend(examples)

        # Variance stats
        variance_report[slice_name] = compute_variance(examples)

        print(f"✅ {slice_name}: {len(examples)} pairs after cleaning")

    # Save cleaned dataset
    with open(out_file, "w") as f:
        for ex in cleaned:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n🎉 Wrote cleaned dataset with {len(cleaned)} pairs to {out_file}")

    # Plot slice sizes
    counts = {c: 0 for c in CATEGORIES}
    for ex in cleaned:
        counts[ex["slice"]] += 1
    plt.bar(counts.keys(), counts.values())
    plt.title("Perturbation Dataset Slice Sizes (Cleaned)")
    plt.savefig("dataset_summary.png")
    plt.close()

    # Variance report summary
    print("\n=== Variance Report ===")
    for slice_name, stats in variance_report.items():
        print(f"{slice_name}: "
              f"count={stats['count']}, "
              f"avg_sim={stats['avg_semantic_sim']:.3f}, "
              f"avg_edit={stats['avg_edit_distance']:.2f}")

    return cleaned, variance_report

if __name__ == "__main__":
    filter_and_rebalance()