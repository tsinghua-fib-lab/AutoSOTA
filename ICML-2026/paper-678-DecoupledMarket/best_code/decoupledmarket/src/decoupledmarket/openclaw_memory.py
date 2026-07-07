import json
import math
import os
import os.path as osp
import re
import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from decoupledmarket.constant import Save_Path


"""Docstring."""


BASE_DIR = osp.join(Save_Path, "openclaw_memory")
os.makedirs(BASE_DIR, exist_ok=True)

_LOCKS: Dict[int, threading.Lock] = {}
_APPEND_COUNT: Dict[int, int] = {}
_AGENT_VERSION: Dict[int, int] = {}

_CACHE_LOCK = threading.Lock()
_QUERY_CACHE: Dict[Tuple[int, str, int, int], Tuple[float, str]] = {}


def _agent_file(agent_id: int) -> str:
    return osp.join(BASE_DIR, f"agent_{agent_id}.jsonl")


def _agent_summary_file(agent_id: int) -> str:
    return osp.join(BASE_DIR, f"agent_{agent_id}_summaries.jsonl")


def _agent_lock(agent_id: int) -> threading.Lock:
    if agent_id not in _LOCKS:
        _LOCKS[agent_id] = threading.Lock()
    return _LOCKS[agent_id]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9_]+", text.lower())


def _to_document_text(record: Dict[str, Any]) -> str:
    parts = [
        str(record.get("event_type", "")),
        str(record.get("prompt", "")),
        str(record.get("response", "")),
    ]
    meta = record.get("meta") or {}
    for k, v in meta.items():
        parts.append(f"{k}:{v}")
    return " ".join(parts)


def _hashed_sparse_vector(tokens: List[str], dim: int = 1024) -> Dict[int, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = float(sum(counts.values()))
    vec: Dict[int, float] = {}
    for tok, cnt in counts.items():
        idx = hash(tok) % dim
        vec[idx] = vec.get(idx, 0.0) + (cnt / total)
    return vec


def _cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0.0)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def _bm25_scores(query_tokens: List[str], docs_tokens: List[List[str]]) -> List[float]:
    if not query_tokens or not docs_tokens:
        return [0.0 for _ in docs_tokens]

    n_docs = len(docs_tokens)
    avgdl = sum(len(toks) for toks in docs_tokens) / max(n_docs, 1)
    k1 = 1.5
    b = 0.75

    df: Dict[str, int] = {}
    for toks in docs_tokens:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1

    scores: List[float] = []
    q_count = Counter(query_tokens)
    for toks in docs_tokens:
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for q, qf in q_count.items():
            n_q = df.get(q, 0)
            if n_q == 0:
                continue
            idf = math.log(1 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            f = tf.get(q, 0)
            if f <= 0:
                continue
            denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += idf * ((f * (k1 + 1)) / denom) * qf
        scores.append(score)
    return scores


def _minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) <= 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _episode_key(record: Dict[str, Any]) -> Optional[str]:
    meta = record.get("meta") or {}
    if meta.get("episode_id") is not None:
        return f"ep:{meta.get('episode_id')}"
    if meta.get("virtual_date") is not None:
        return f"day:{meta.get('virtual_date')}"
    return None


def _load_summary_keys(path: str) -> set:
    keys = set()
    if not osp.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("meta", {}).get("episode_key")
                if key:
                    keys.add(key)
            except Exception:
                continue
    return keys


def _build_episode_summary(agent_id: int, episode_key: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    texts = []
    for rec in records:
        texts.append(str(rec.get("prompt", "")))
        texts.append(str(rec.get("response", "")))
    all_text = " ".join(texts)
    tokens = [t for t in _tokenize(all_text) if len(t) > 2]
    top_tokens = [tok for tok, _ in Counter(tokens).most_common(8)]

    first_resp = str(records[0].get("response", "")).replace("\n", " ")
    last_resp = str(records[-1].get("response", "")).replace("\n", " ")
    summary_text = (
        f"Episode {episode_key}: {len(records)} events. "
        f"Top topics: {', '.join(top_tokens) if top_tokens else 'n/a'}. "
        f"Start: {first_resp[:100]}. End: {last_resp[:100]}."
    )
    return {
        "agent_id": agent_id,
        "event_type": "episode_summary",
        "prompt": f"summary for {episode_key}",
        "response": summary_text,
        "meta": {"episode_key": episode_key, "num_events": len(records)},
        "created_at_ms": int(time.time() * 1000),
    }


def _compact_old_records_to_episode_summaries(agent_id: int, dropped_lines: List[str]) -> None:
    if not dropped_lines:
        return
    min_items = int(os.getenv("OPENCLAW_SUMMARY_MIN_EPISODE_ITEMS", "8"))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for line in dropped_lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ep_key = _episode_key(rec)
        if not ep_key:
            continue
        grouped.setdefault(ep_key, []).append(rec)

    summary_path = _agent_summary_file(agent_id)
    existing = _load_summary_keys(summary_path)
    new_summaries: List[Dict[str, Any]] = []
    for ep_key, recs in grouped.items():
        if ep_key in existing:
            continue
        if len(recs) < min_items:
            continue
        new_summaries.append(_build_episode_summary(agent_id, ep_key, recs))

    if not new_summaries:
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        for rec in new_summaries:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _prune_agent_file(agent_id: int) -> None:
    path = _agent_file(agent_id)
    if not osp.exists(path):
        return
    max_records = int(os.getenv("OPENCLAW_MAX_RECORDS", "5000"))
    keep_recent = int(os.getenv("OPENCLAW_RAW_KEEP_RECENT", "1500"))
    keep_recent = max(1, min(keep_recent, max_records))

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) <= max_records:
        return

    dropped = lines[:-keep_recent]
    tail = lines[-keep_recent:]
    _compact_old_records_to_episode_summaries(agent_id, dropped)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(tail)


def _invalidate_agent_cache(agent_id: int) -> None:
    with _CACHE_LOCK:
        keys = [k for k in _QUERY_CACHE if k[0] == agent_id]
        for key in keys:
            _QUERY_CACHE.pop(key, None)


def append_memory_entry(
    agent_id: int,
    event_type: str,
    prompt: str,
    response: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if agent_id is None or agent_id < 0:
        return

    record = {
        "agent_id": agent_id,
        "event_type": event_type,
        "prompt": prompt,
        "response": response,
        "meta": meta or {},
        "created_at_ms": int(time.time() * 1000),
    }
    path = _agent_file(agent_id)
    lock = _agent_lock(agent_id)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _APPEND_COUNT[agent_id] = _APPEND_COUNT.get(agent_id, 0) + 1
        _AGENT_VERSION[agent_id] = _AGENT_VERSION.get(agent_id, 0) + 1
        prune_every = int(os.getenv("OPENCLAW_PRUNE_EVERY_APPENDS", "100"))
        prune_every = max(1, prune_every)
        if _APPEND_COUNT[agent_id] % prune_every == 0:
            _prune_agent_file(agent_id)
    _invalidate_agent_cache(agent_id)


def _load_jsonl_tail(path: str, max_items: int) -> List[Dict[str, Any]]:
    if not osp.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-max_items:]
    records: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    records.reverse()
    return records


def _load_agent_memories(agent_id: int, max_items: int = 200) -> List[Dict[str, Any]]:
    path = _agent_file(agent_id)
    lock = _agent_lock(agent_id)
    with lock:
        return _load_jsonl_tail(path, max_items=max_items)


def _load_agent_summaries(agent_id: int, max_items: int = 120) -> List[Dict[str, Any]]:
    path = _agent_summary_file(agent_id)
    lock = _agent_lock(agent_id)
    with lock:
        return _load_jsonl_tail(path, max_items=max_items)


def _hybrid_rank(
    query: str,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not records:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return [
            {
                "score": 1.0 - i * 1e-6,
                "bm25": 0.0,
                "vec": 0.0,
                "recency": 1.0 - i * 1e-6,
                "record": rec,
            }
            for i, rec in enumerate(records)
        ]

    docs_text = [_to_document_text(rec) for rec in records]
    docs_tokens = [_tokenize(text) for text in docs_text]

    bm25_raw = _bm25_scores(query_tokens, docs_tokens)
    q_vec = _hashed_sparse_vector(query_tokens)
    vec_raw = [_cosine_sparse(q_vec, _hashed_sparse_vector(tokens)) for tokens in docs_tokens]

    bm25_norm = _minmax_normalize(bm25_raw)
    vec_norm = _minmax_normalize(vec_raw)

    alpha = float(os.getenv("OPENCLAW_BM25_WEIGHT", "0.6"))
    alpha = max(0.0, min(1.0, alpha))

    ranked: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        recency_bonus = max(0.0, (len(records) - i) / max(len(records), 1)) * 0.02
        # Slightly penalize summary entries so fresh raw memories still dominate ties.
        is_summary = rec.get("event_type") == "episode_summary"
        summary_penalty = 0.03 if is_summary else 0.0
        score = alpha * bm25_norm[i] + (1.0 - alpha) * vec_norm[i] + recency_bonus - summary_penalty
        ranked.append(
            {
                "score": score,
                "bm25": bm25_norm[i],
                "vec": vec_norm[i],
                "recency": recency_bonus,
                "record": rec,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def _cache_get(agent_id: int, query: str, max_items: int) -> Optional[str]:
    if not _env_flag("OPENCLAW_ENABLE_CACHE", "1"):
        return None
    version = _AGENT_VERSION.get(agent_id, 0)
    key = (agent_id, query, max_items, version)
    now = time.time()
    with _CACHE_LOCK:
        val = _QUERY_CACHE.get(key)
        if val is None:
            return None
        exp_ts, snippet = val
        if exp_ts < now:
            _QUERY_CACHE.pop(key, None)
            return None
        return snippet


def _cache_set(agent_id: int, query: str, max_items: int, snippet: str) -> None:
    if not _env_flag("OPENCLAW_ENABLE_CACHE", "1"):
        return
    ttl = float(os.getenv("OPENCLAW_CACHE_TTL_SECONDS", "30"))
    version = _AGENT_VERSION.get(agent_id, 0)
    key = (agent_id, query, max_items, version)
    with _CACHE_LOCK:
        _QUERY_CACHE[key] = (time.time() + ttl, snippet)
        max_entries = int(os.getenv("OPENCLAW_CACHE_MAX_ENTRIES", "2000"))
        if len(_QUERY_CACHE) > max_entries:
            # Remove expired first, then oldest by expiry timestamp.
            now = time.time()
            expired_keys = [k for k, (exp, _) in _QUERY_CACHE.items() if exp < now]
            for k in expired_keys:
                _QUERY_CACHE.pop(k, None)
            if len(_QUERY_CACHE) > max_entries:
                to_drop = sorted(_QUERY_CACHE.items(), key=lambda kv: kv[1][0])[: len(_QUERY_CACHE) - max_entries]
                for k, _ in to_drop:
                    _QUERY_CACHE.pop(k, None)


def get_memory_snippet_for_prompt(
    agent_id: int,
    query: str = "",
    max_items: int = 5,
) -> str:
    if agent_id is None or agent_id < 0:
        return ""

    cached = _cache_get(agent_id, query, max_items)
    if cached is not None:
        return cached

    raw_candidates = _load_agent_memories(agent_id, max_items=max_items * 20)
    summary_candidates = _load_agent_summaries(agent_id, max_items=max_items * 12)
    candidates = raw_candidates + summary_candidates
    if not candidates:
        return ""

    ranked = _hybrid_rank(query=query, records=candidates)
    top = ranked[:max_items]

    lines: List[str] = []
    for item in top:
        rec = item["record"]
        etype = rec.get("event_type", "event")
        meta = rec.get("meta") or {}
        vdate = meta.get("virtual_date")
        it = meta.get("iteration")
        created_at_ms = rec.get("created_at_ms")
        episode_key = meta.get("episode_key")

        header_parts = [str(etype)]
        if episode_key is not None:
            header_parts.append(str(episode_key))
        if vdate is not None:
            header_parts.append(f"day={vdate}")
        if it is not None:
            header_parts.append(f"iter={it}")
        if created_at_ms is not None:
            header_parts.append(f"ts={created_at_ms}")
        header = ", ".join(header_parts)

        resp = str(rec.get("response", "")).replace("\n", " ")
        resp = resp[:220] + ("..." if len(resp) > 220 else "")
        lines.append(f"- [{header}] {resp}")

    if _env_flag("OPENCLAW_DEBUG_RETRIEVAL", "0"):
        print(f"[OpenClaw] agent={agent_id} query='{query[:80]}' candidates={len(candidates)}")
        for idx, item in enumerate(top, start=1):
            rec = item["record"]
            print(
                "[OpenClaw] rank={} score={:.4f} bm25={:.4f} vec={:.4f} recency={:.4f} type={}".format(
                    idx,
                    item["score"],
                    item["bm25"],
                    item["vec"],
                    item["recency"],
                    rec.get("event_type", "event"),
                )
            )

    snippet = "\n".join(lines)
    _cache_set(agent_id, query, max_items, snippet)
    return snippet
