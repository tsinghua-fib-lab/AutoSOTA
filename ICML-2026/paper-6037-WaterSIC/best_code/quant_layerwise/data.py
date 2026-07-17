from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Literal

import torch
from datasets import load_dataset


def _tokenizer_tag(tokenizer) -> str:
    """Short tag identifying the tokenizer, for cache file names.

    Uses vocab_size + hash of a probe encoding to distinguish tokenizers
    without requiring a .name attribute.
    """
    probe = tokenizer.encode("The quick brown fox", bos=False, eos=False)
    h = hashlib.md5(str(probe).encode()).hexdigest()[:8]
    vocab = getattr(tokenizer, "n_words", None) or getattr(tokenizer, "vocab_size", "unk")
    return f"v{vocab}_{h}"


def get_wikitext2(tokenizer, *, split: Literal["train", "test"] = "test",
                   shuffle_docs: bool = False, seed: int = 42) -> torch.Tensor:
    """Return a 1D tensor of token ids for WikiText-2.

    Args:
        shuffle_docs: If True, shuffle articles before concatenating. This makes
            W2's document-boundary pattern match RP/C4 (topic changes mid-chunk).
            Only use for calibration (train split), never for test eval.
    """
    # Disable HuggingFace file locking to avoid contention in parallel processes
    os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"

    print(f"[data] loading wikitext2 split={split}...", flush=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    print("[data] loaded wikitext2, tokenizing...", flush=True)

    if shuffle_docs:
        import random
        docs = list(ds["text"])
        rng = random.Random(seed)
        rng.shuffle(docs)
        print(f"[data] shuffled {len(docs)} wikitext2 articles (seed={seed})", flush=True)
        text = "\n\n".join(docs)
    else:
        text = "\n\n".join(ds["text"])
    enc = tokenizer.encode(text, bos=True, eos=True)
    print(f"[data] tokenized {len(enc)} tokens", flush=True)
    return torch.tensor(enc, dtype=torch.long)


def _stream_and_tokenize(
    tokenizer,
    ds,
    *,
    target_tokens: int | None,
    label: str,
    seed: int,
) -> list[int]:
    """Shared tokenization loop: shuffle, concatenate-then-chunk.

    If target_tokens is None, streams the entire dataset.
    """
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    bos_id = getattr(tokenizer, "bos_id", None)
    all_tokens: list[int] = []
    if bos_id is not None:
        all_tokens.append(bos_id)

    n_docs = 0
    for doc in ds:
        text = doc["text"]
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode("\n\n" + text, bos=False, eos=False)
        all_tokens.extend(tokens)
        n_docs += 1
        if n_docs % 5000 == 0:
            print(f"[data] processed {n_docs} docs, {len(all_tokens)} tokens so far...", flush=True)
        if target_tokens is not None and len(all_tokens) >= target_tokens:
            break

    if target_tokens is not None:
        all_tokens = all_tokens[:target_tokens]
    print(f"[data] {label}: {n_docs} docs -> {len(all_tokens)} tokens", flush=True)
    return all_tokens


def _load_cached_or_none(cache_path: Path, target_tokens: int | None) -> torch.Tensor | None:
    """Return cached token tensor if it exists and has enough tokens."""
    if not cache_path.exists():
        return None
    token_ids = torch.load(cache_path, map_location="cpu", weights_only=True)
    if target_tokens is None or token_ids.shape[0] >= target_tokens:
        print(f"[data] loaded {token_ids.shape[0]} cached tokens from {cache_path}", flush=True)
        return token_ids
    print(f"[data] cache has {token_ids.shape[0]} tokens (need {target_tokens}), regenerating...", flush=True)
    return None


def _save_cache(token_ids: torch.Tensor, cache_dir: Path, cache_path: Path) -> None:
    """Save token tensor to cache."""
    if os.environ.get("QUANT_BUCKET"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(token_ids, cache_path)
        print(f"[data] cached tokens to {cache_path}", flush=True)


def get_redpajama(
    tokenizer,
    *,
    nsamples: int | None = 2048,
    seqlen: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """Return a 1D tensor of token ids from RedPajama-Data-1T-Sample.

    Uses the official 1B-token sample of RedPajama-1T which contains a diverse
    mix of domains (CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia,
    StackExchange).  Standard LM preprocessing: concatenate-then-chunk.

    Caches the tokenized result to disk.
    If nsamples is None, streams the entire dataset (~1B tokens).
    """
    os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"
    target_tokens = nsamples * seqlen if nsamples is not None else None
    nsamples_label = nsamples if nsamples is not None else "all"

    cache_dir = Path(os.environ.get("QUANT_BUCKET", "")) / "calib_cache"
    tag = _tokenizer_tag(tokenizer)
    cache_path = cache_dir / f"redpajama_n{nsamples_label}_s{seqlen}_seed{seed}_{tag}.pt"

    cached = _load_cached_or_none(cache_path, target_tokens)
    if cached is not None:
        return cached

    print("[data] streaming ZengXiangyu/RedPajama-Data-1T-Sample (concat-then-chunk)", flush=True)
    print(f"[data] target={target_tokens or 'all'} tokens, seed={seed}", flush=True)

    ds = load_dataset(
        "ZengXiangyu/RedPajama-Data-1T-Sample", split="train", streaming=True,
    )

    all_tokens = _stream_and_tokenize(
        tokenizer, ds, target_tokens=target_tokens, label="RedPajama", seed=seed,
    )
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    _save_cache(token_ids, cache_dir, cache_path)
    return token_ids


def get_c4(
    tokenizer,
    *,
    nsamples: int | None = 2048,
    seqlen: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """Return a 1D tensor of token ids from allenai/c4 (English).

    Fallback calibration source. Uses standard LM preprocessing
    (concatenate-then-chunk). Caches the tokenized result to disk.

    If nsamples is None, streams the entire dataset.
    """
    os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"
    target_tokens = nsamples * seqlen if nsamples is not None else None
    nsamples_label = nsamples if nsamples is not None else "all"

    cache_dir = Path(os.environ.get("QUANT_BUCKET", "")) / "calib_cache"
    tag = _tokenizer_tag(tokenizer)
    cache_path = cache_dir / f"c4_n{nsamples_label}_s{seqlen}_seed{seed}_{tag}.pt"

    cached = _load_cached_or_none(cache_path, target_tokens)
    if cached is not None:
        return cached

    print(f"[data] streaming allenai/c4 en (target={target_tokens or 'all'} tokens, seed={seed})...", flush=True)
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

    all_tokens = _stream_and_tokenize(
        tokenizer, ds, target_tokens=target_tokens, label="C4", seed=seed,
    )
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    _save_cache(token_ids, cache_dir, cache_path)
    return token_ids


def get_c4_val(
    tokenizer,
    *,
    nsamples: int = 256,
    seqlen: int = 2048,
    seed: int = 0,
) -> torch.Tensor:
    """Return C4 validation data following the exact GPTQ evaluation protocol.

    Matches https://github.com/IST-DASLab/gptq/blob/main/datautils.py:
    - Load c4-validation.00000-of-00008.json.gz (first shard, ~364k documents)
    - seed=0, for each of 256 samples: rejection-sample a document long enough,
      then pick a random window of length seqlen within it.
    - Sampling is from the FULL shard (not a small prefix).
    """
    import random

    os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"

    cache_dir = Path(os.environ.get("QUANT_BUCKET", "")) / "calib_cache"
    tag = _tokenizer_tag(tokenizer)
    cache_path = cache_dir / f"c4val_gptq_n{nsamples}_s{seqlen}_seed{seed}_{tag}.pt"

    cached = _load_cached_or_none(cache_path, nsamples * seqlen)
    if cached is not None:
        return cached

    print(f"[data] loading C4 validation (nsamples={nsamples}, seqlen={seqlen}, seed={seed})...", flush=True)
    # Load the first validation shard (matches GPTQ exactly).
    try:
        valdata = load_dataset("json",
                               data_files="hf://datasets/allenai/c4/en/c4-validation.00000-of-00008.json.gz",
                               split="train")
    except Exception:
        print("[data] direct shard load failed, trying streaming fallback...", flush=True)
        # Fallback: stream and collect all documents
        valdata_stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        valdata = []
        for doc in valdata_stream:
            valdata.append(doc)
            if len(valdata) >= 400_000:
                break
        print(f"[data] collected {len(valdata)} documents via streaming", flush=True)

    print(f"[data] C4 validation shard: {len(valdata)} documents", flush=True)

    # Exactly match GPTQ: random.seed(0), rejection-sample from full shard
    rng = random.Random(seed)
    all_seqs = []
    for _ in range(nsamples):
        while True:
            i = rng.randint(0, len(valdata) - 1)
            text = valdata[i]["text"]
            # Match GPTQ: use tokenizer defaults for special tokens.
            # Qwen3 has no BOS (bos_token=null), Llama adds BOS by default.
            bos = getattr(tokenizer, 'bos_id', None) is not None
            tokens = tokenizer.encode(text, bos=bos, eos=False)
            if len(tokens) >= seqlen:
                break
        start = rng.randint(0, max(0, len(tokens) - seqlen - 1))
        all_seqs.append(torch.tensor(tokens[start : start + seqlen], dtype=torch.long))

    result = torch.stack(all_seqs).reshape(-1)
    print(f"[data] C4 validation: {nsamples} samples, {result.numel()} tokens", flush=True)
    _save_cache(result, cache_dir, cache_path)
    return result


def get_redpajama_sample(
    tokenizer,
    *,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """Return calibration data via random-window sampling from RedPajama-1T-Sample.

    For each sample: pick a random document, reject if too short, then extract
    a random seqlen-length window.  Sequences are independent (no cross-sequence
    continuity).  This matches the AQLM/PV-Tuning calibration approach.

    NOTE: For quantization calibration, prefer ``get_redpajama`` (concat-then-chunk)
    which produces coherent sequential text and sharper Hessians, especially at
    low bit-rates.

    Returns 1D tensor of token ids (nsamples * seqlen tokens).
    """
    os.environ["HF_DATASETS_DISABLE_FILE_LOCKING"] = "1"

    cache_dir = Path(os.environ.get("QUANT_BUCKET", "")) / "calib_cache"
    tag = _tokenizer_tag(tokenizer)
    cache_path = cache_dir / f"redpajama_sample_n{nsamples}_s{seqlen}_seed{seed}_{tag}.pt"

    target_tokens = nsamples * seqlen
    cached = _load_cached_or_none(cache_path, target_tokens)
    if cached is not None:
        return cached

    print(f"[data] loading RedPajama-Data-1T-Sample for random-window sampling ({nsamples} samples)...", flush=True)
    traindata = load_dataset("ZengXiangyu/RedPajama-Data-1T-Sample", split="train")
    print(f"[data] loaded {len(traindata)} docs, sampling...", flush=True)

    random.seed(seed)
    all_tokens: list[int] = []
    for i in range(nsamples):
        while True:
            idx = random.randint(0, len(traindata) - 1)
            text = traindata[idx]["text"]
            if not text or not text.strip():
                continue
            tokens = tokenizer.encode(text, bos=False, eos=False)
            if len(tokens) > seqlen:
                break
        # Random window within the document
        start = random.randint(0, len(tokens) - seqlen - 1)
        all_tokens.extend(tokens[start : start + seqlen])
        if (i + 1) % 100 == 0:
            print(f"[data] sampled {i + 1}/{nsamples} sequences", flush=True)

    print(f"[data] RedPajama-Sample: {nsamples} samples -> {len(all_tokens)} tokens", flush=True)
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    _save_cache(token_ids, cache_dir, cache_path)
    return token_ids


def get_calibration_data(
    tokenizer,
    *,
    dataset: str = "redpajama",
    nsamples: int | None = 2048,
    seqlen: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """Dispatcher: return 1D tensor of calibration token ids.

    Args:
        tokenizer: Tokenizer with an `encode(text, bos, eos)` method.
        dataset: "redpajama" (default), "c4", "wikitext2", or a mix spec
                 like "redpajama:0.5,c4:0.4,wikitext2:0.1".
        nsamples: Number of seqlen-sized chunks (only used for redpajama/c4).
                  None means stream the entire dataset.
                  For mix mode, this is the total number of sequences.
        seqlen: Sequence length (only used for redpajama/c4).
        seed: Shuffle seed (only used for redpajama/c4).

    Returns:
        1D tensor of token ids.
    """
    # Check for mix spec: "redpajama:0.5,c4:0.4,wikitext2:0.1"
    if ":" in dataset and "," in dataset:
        return get_calibration_data_mix(
            tokenizer, mix_spec=dataset, nsamples=nsamples or 1189,
            seqlen=seqlen, seed=seed,
        )

    if dataset == "redpajama":
        return get_redpajama(tokenizer, nsamples=nsamples, seqlen=seqlen, seed=seed)
    elif dataset == "redpajama_sample":
        return get_redpajama_sample(tokenizer, nsamples=nsamples or 128, seqlen=seqlen, seed=seed)
    elif dataset == "c4":
        return get_c4(tokenizer, nsamples=nsamples, seqlen=seqlen, seed=seed)
    elif dataset == "wikitext2":
        tokens = get_wikitext2(tokenizer, split="train")
        if nsamples is not None:
            import random
            rng = random.Random(seed)
            # Sample from chunk-aligned positions (multiples of seqlen).
            # This matches eval's non-overlapping chunking: position 0 of each
            # sequence always lacks prior context, matching the test-time pattern.
            n_chunks = tokens.shape[0] // seqlen
            if n_chunks <= 0:
                return tokens
            chunk_indices = [rng.randint(0, n_chunks - 1) for _ in range(nsamples)]
            seqs = torch.stack([tokens[i * seqlen : (i + 1) * seqlen] for i in chunk_indices])
            n_unique = len(set(chunk_indices))
            print(f"[data] wikitext2: sampled {nsamples} chunk-aligned windows "
                  f"({n_unique} unique from {n_chunks} chunks)", flush=True)
            return seqs.reshape(-1)
        return tokens
    else:
        raise ValueError(f"Unknown calibration dataset: {dataset!r}. Use 'redpajama', 'c4', 'wikitext2', or a mix like 'redpajama:0.5,c4:0.4,wikitext2:0.1'.")


def get_calibration_data_mix(
    tokenizer,
    *,
    mix_spec: str,
    nsamples: int = 1189,
    seqlen: int = 2048,
    seed: int = 42,
) -> torch.Tensor:
    """Load a mixture of calibration datasets at the sequence level.

    Args:
        mix_spec: Comma-separated "dataset:spec" pairs. Each spec can be:
          - A fraction (float): relative weight (normalized to sum=1, then * nsamples)
            e.g. "redpajama:0.5,c4:0.4,wikitext2:0.1"
          - An integer count: explicit number of sequences from that dataset
            e.g. "wikitext2:1189,c4:1024"
          - "all": use all available sequences from that dataset
            e.g. "wikitext2:all,c4:1024"
        nsamples: Total sequences (only used when specs are fractions).
        seqlen: Sequence length.
        seed: Random seed.

    Returns:
        1D tensor of shuffled, interleaved token ids.
    """
    import random

    # Parse mix spec
    components = []  # list of (ds, spec_str)
    for item in mix_spec.split(","):
        item = item.strip()
        if ":" not in item:
            raise ValueError(f"Mix spec items must be 'dataset:spec', got '{item}'")
        ds, spec = item.rsplit(":", 1)
        components.append((ds.strip(), spec.strip()))

    # Detect mode: explicit counts (int or "all") vs fractions (float < 1.0 or explicit count ≥ 1)
    # Rule: if any spec is "all" or contains no decimal point, treat as counts
    use_counts = any(s.lower() == "all" or "." not in s for _, s in components)

    # Compute target counts per component
    target_counts: list[tuple[str, int | str]] = []
    if use_counts:
        for ds, spec in components:
            if spec.lower() == "all":
                target_counts.append((ds, "all"))
            else:
                target_counts.append((ds, int(spec)))
    else:
        # Fraction mode — normalize and multiply by nsamples
        weights = [float(s) for _, s in components]
        total = sum(weights)
        for (ds, _), w in zip(components, weights):
            target_counts.append((ds, max(1, int(nsamples * w / total))))

    print(f"[data-mix] mixing {len(components)} datasets (seqlen={seqlen}):", flush=True)
    for ds, c in target_counts:
        print(f"  {ds}: {c} seqs", flush=True)

    # Load each dataset and sample random windows
    rng = random.Random(seed)
    all_seqs = []
    for ds, spec_count in target_counts:
        # n_seqs_target: target count (or "all" sentinel)
        _use_all = (spec_count == "all")
        n_seqs_target = 999999 if _use_all else spec_count

        if ds == "wikitext2":
            tokens = get_wikitext2(tokenizer, split="train")
        elif ds == "redpajama":
            if _use_all:
                raise ValueError("'all' not supported for streaming datasets (redpajama/c4). Use an explicit count.")
            tokens = get_redpajama(tokenizer, nsamples=n_seqs_target, seqlen=seqlen, seed=seed)
        elif ds == "c4":
            if _use_all:
                raise ValueError("'all' not supported for streaming datasets (redpajama/c4). Use an explicit count.")
            tokens = get_c4(tokenizer, nsamples=n_seqs_target, seqlen=seqlen, seed=seed)
        elif ds == "redpajama_sample":
            tokens = get_redpajama_sample(tokenizer, nsamples=n_seqs_target, seqlen=seqlen, seed=seed)
        else:
            raise ValueError(f"Unknown dataset in mix: {ds!r}")

        # Sample chunk-aligned windows from the token stream.
        # Aligning to seqlen boundaries matches eval's non-overlapping chunking:
        # position 0 of each sequence always lacks prior context.
        n_chunks = tokens.shape[0] // seqlen
        if n_chunks <= 0:
            print(f"  [data-mix] {ds}: not enough tokens ({tokens.shape[0]}) for seqlen={seqlen}, skipping", flush=True)
            continue

        if _use_all:
            # Use all available chunks, no replacement
            seqs = tokens[: n_chunks * seqlen].reshape(n_chunks, seqlen)
            all_seqs.append(seqs)
            print(f"  [data-mix] {ds}: using all {n_chunks} chunks", flush=True)
        else:
            chunk_indices = [rng.randint(0, n_chunks - 1) for _ in range(n_seqs_target)]
            seqs = torch.stack([tokens[i * seqlen : (i + 1) * seqlen] for i in chunk_indices])
            all_seqs.append(seqs)
            n_unique = len(set(chunk_indices))
            print(f"  [data-mix] {ds}: sampled {seqs.shape[0]} chunk-aligned seqs ({n_unique} unique from {n_chunks})", flush=True)

    # Concatenate and shuffle at sequence level
    combined = torch.cat(all_seqs, dim=0)  # (total_seqs, seqlen)
    rng = random.Random(seed)
    perm = list(range(combined.shape[0]))
    rng.shuffle(perm)
    combined = combined[perm]

    # Flatten back to 1D (the pipeline will re-split with split_dataset)
    print(f"[data-mix] total: {combined.shape[0]} seqs, {combined.numel()} tokens", flush=True)
    return combined.reshape(-1)


def split_dataset(token_ids: torch.Tensor, seqlen: int, stride: int | None = None) -> torch.Tensor:
    """Split 1D token tensor into (nseq, seqlen).

    Args:
        token_ids: 1D tensor of token ids
        seqlen: Sequence length
        stride: Step between consecutive sequences. Default (None) = seqlen
                (non-overlapping). Use stride < seqlen for overlapping windows,
                e.g. stride=seqlen//2 gives 50% overlap and ~2x more sequences.
    """
    if token_ids.ndim != 1:
        raise ValueError("split_dataset expects a 1D tensor")
    seqlen = int(seqlen)
    if stride is None:
        stride = seqlen
    stride = int(stride)
    if stride == seqlen:
        # Fast path: non-overlapping
        nseq = token_ids.shape[0] // seqlen
        token_ids = token_ids[: nseq * seqlen]
        return token_ids.reshape(nseq, seqlen)
    else:
        # Overlapping windows via unfold
        return token_ids.unfold(0, seqlen, stride).contiguous()


def take_nseq(token_ids_2d: torch.Tensor, nsamples: int | None) -> torch.Tensor:
    """Take first nsamples sequences. If nsamples is None, return all sequences."""
    if nsamples is None:
        return token_ids_2d
    nsamples = int(nsamples)
    return token_ids_2d[:nsamples]
