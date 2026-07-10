"""Data loading and train/calibration splits."""

import json
import random
from typing import Dict, List, Optional, Tuple


def load_json_dataset(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load CPR JSON format with per-query subgraph triples."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    for item in data:
        cleaned = []
        for tri in item.get("triples", []):
            if isinstance(tri, (list, tuple)) and len(tri) == 3:
                cleaned.append((tri[0].lower(), tri[1].lower(), tri[2].lower()))
        item["triples"] = cleaned

    return data


def build_global_graph(train_items: List[Dict], test_items: List[Dict]) -> List[Tuple[str, str, str]]:
    """Union of train/test triples (deduplicated)."""
    triple_set = set()
    for item in train_items:
        for h, r, t in item.get("triples", []):
            triple_set.add((h, r, t))
    for item in test_items:
        for h, r, t in item.get("triples", []):
            triple_set.add((h, r, t))
    global_triples = list(triple_set)
    print(f"[Graph] Global triples: {len(global_triples)}")
    return global_triples


def build_dataset_items(parsed: List[Dict]) -> List[Dict]:
    """Normalize items for train/calib/test (requires q_entity and a_entity)."""
    out = []
    for it in parsed:
        q_entity = it.get("q_entity") or []
        a_entity = it.get("a_entity") or []
        if not q_entity or not a_entity:
            continue
        out.append({
            "id": it["id"],
            "question": it["question"],
            "q_entity": [_.lower() for _ in q_entity],
            "a_entity": [_.lower() for _ in a_entity],
            "triples": it.get("triples", []),
        })
    return out


def build_calibration_items(parsed_train: List[Dict]) -> List[Dict]:
    """Alias for backward compatibility."""
    return build_dataset_items(parsed_train)


def build_test_items(parsed_test: List[Dict]) -> List[Dict]:
    return build_dataset_items(parsed_test)


def split_calib(
    data: List[Dict],
    seed: int = 42,
    val_frac: float = 0.1,
) -> Tuple[List[Dict], List[Dict]]:
    """Split into (train, calibration) — val_frac goes to calibration."""
    rnd = random.Random(seed)
    items = data.copy()
    rnd.shuffle(items)
    n_val = int(len(items) * val_frac)
    return items[n_val:], items[:n_val]


def load_json_or_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        pass
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[Warning] Cannot parse line: {line[:60]}...")
    return data


def load_id_map(file_path: str) -> Dict[int, str]:
    id_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            id_map[i] = line.strip()
    return id_map


def load_subgraph_triples(
    json_file: str,
    entity_map: Dict,
    relation_map: Dict,
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    triples = []
    data = load_json_or_jsonl(json_file)
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
        if "subgraph" not in item or "tuples" not in item["subgraph"]:
            continue
        for (s_id, r_id, o_id) in item["subgraph"]["tuples"]:
            s_mid = entity_map.get(s_id)
            p_rel = relation_map.get(r_id)
            o_mid = entity_map.get(o_id)
            if s_mid and p_rel and o_mid:
                triples.append((s_mid, p_rel, o_mid))
    return triples


def build_graph(
    train_json: str,
    test_json: str,
    entity_file: str,
    relation_file: str,
) -> List[Tuple[str, str, str]]:
    entity_map = load_id_map(entity_file)
    relation_map = load_id_map(relation_file)
    train_triples = load_subgraph_triples(train_json, entity_map, relation_map)
    test_triples = load_subgraph_triples(test_json, entity_map, relation_map)
    all_triples = list(set(train_triples + test_triples))
    print(f"Train triples: {len(train_triples)}")
    print(f"Test triples: {len(test_triples)}")
    print(f"Total unique triples: {len(all_triples)}")
    return all_triples
