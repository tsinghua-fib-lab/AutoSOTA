"""
Token-preserving synonym substitution attack.

Replaces ~substitution_rate fraction of tokens with synonyms
while preserving the total number of tokens (token-preserving).
"""

import random
import os

# ---------------------------------------------------------------------------
# Lazy-load NLTK WordNet resources (cached on first use).
# ---------------------------------------------------------------------------
_nltk_ready = False

def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for resource in ("wordnet", "averaged_perceptron_tagger", "punkt_tab", "punkt"):
        try:
            nltk.data.find(f"corpora/{resource}" if resource != "punkt_tab" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _nltk_ready = True


# ---------------------------------------------------------------------------
# POS mapping from Penn Treebank tag to WordNet POS
# ---------------------------------------------------------------------------
_POS_MAP = {
    "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n",
    "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v",
    "JJ": "a", "JJR": "a", "JJS": "a",
    "RB": "r", "RBR": "r", "RBS": "r",
}

# Stop-word POS tags we don't replace
_SKIP_POS = {"DT", "IN", "CC", "TO", "MD", "PRP", "PRP$", "WDT", "WP", "WP$", "WRB", "EX", "PDT", "RP", "UH", "CD", "LS", "FW", "."}


def _get_wordnet_pos(treebank_tag):
    return _POS_MAP.get(treebank_tag, None)


def _get_synonym(token, wn_pos):
    """Return one synonym (different from token) or None."""
    from nltk.corpus import wordnet
    synsets = wordnet.synsets(token, pos=wn_pos)
    if not synsets:
        return None

    # Collect all lemma names across synsets
    candidates = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != token.lower():
                candidates.add(name)

    if not candidates:
        return None

    return random.choice(list(candidates))


def apply_synonym_substitution(text: str, substitution_rate: float = 0.10, seed: int = 42) -> str:
    """
    Apply token-preserving synonym substitution to `text`.

    Args:
        text: input text
        substitution_rate: fraction of eligible tokens to replace (0.0 - 1.0)
        seed: random seed for reproducibility

    Returns:
        modified text with ~substitution_rate tokens replaced by synonyms
    """
    _ensure_nltk()
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    rng = random.Random(seed)

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Identify replaceable positions
    eligible = []
    for i, (token, tag) in enumerate(tagged):
        if not token.isalpha():
            continue
        if len(token) <= 2:
            continue
        if tag in _SKIP_POS:
            continue
        wn_pos = _get_wordnet_pos(tag)
        if wn_pos is None:
            continue
        eligible.append((i, token, wn_pos))

    # Determine how many to replace
    n_replace = max(1, int(len(eligible) * substitution_rate))
    to_replace = rng.sample(eligible, min(n_replace, len(eligible)))

    # Replace tokens
    result = list(tokens)
    for i, token, wn_pos in to_replace:
        synonym = _get_synonym(token, wn_pos)
        if synonym is not None:
            result[i] = synonym

    # Reconstruct text (simple space joining)
    # Handle punctuation attachment
    out_tokens = []
    for token in result:
        if out_tokens and token in {".", ",", ";", ":", "!", "?", ")", "]", "}", "'s", "n't", "'t", "'re", "'ve", "'ll", "'d", "'m"}:
            out_tokens[-1] += token
        elif out_tokens and out_tokens[-1] in {"(", "[", "{", "$", "\"", "'"}:
            out_tokens[-1] += token
        else:
            out_tokens.append(token)

    return " ".join(out_tokens)


if __name__ == "__main__":
    # Quick test
    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Scientists have discovered a new method for detecting AI-generated text."
    )
    attacked = apply_synonym_substitution(test_text, substitution_rate=0.10, seed=42)
    print("Original:", test_text)
    print("Attacked:", attacked)
