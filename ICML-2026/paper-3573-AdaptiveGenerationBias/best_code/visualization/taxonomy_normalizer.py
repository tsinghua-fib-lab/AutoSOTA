import argparse
import json
import os
import re
from dataclasses import asdict
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
from src.bias_pipeline.questionaires.questionaire import (
    load_saved_questions_from_runs,
    BiasQuestionnaire,
    Question,
)
from src.models import get_model
from src.configs import ModelConfig


def _ensure_mapping_struct(d: Optional[dict]) -> dict:
    """
    Ensure the mapping dict has the expected keys.
    """
    if not isinstance(d, dict):
        d = {}
    d.setdefault("superdomain_map", {})  # raw_super -> canonical_super
    d.setdefault("domain_map", {})  # raw_domain -> canonical_domain
    return d


def _atomic_write_json(path: str, payload: dict):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _extract_json_from_llm(text: str) -> Optional[dict]:
    """
    Robustly extract JSON from chatty model output.
    Prefers ```json fenced blocks; falls back to first { ... }.
    """
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i = text.find("{")
    if i >= 0:
        j = text.rfind("}")
        if j > i:
            candidate = text[i : j + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None


def _pack_pair(sd: str, dm: str) -> str:
    return f"{sd}|||{dm}"


# --------------------- Collector ---------------------


def _load_all_questions_from_runs(run_paths: List[str]) -> List[Question]:
    """
    Uses your provided helper to load saved questions from run(s).
    Falls back gracefully if no questions are found.
    """
    questions: Dict[str, Dict[int, BiasQuestionnaire]] = load_saved_questions_from_runs(run_paths)
    all_q: List[Question] = []
    for _path, questionnaires_by_iter in questions.items():
        for _iter, qn in questionnaires_by_iter.items():
            all_q.extend(qn.to_list())
    return all_q


def _collect_labels_and_pairs(
    questions: List[Question],
) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]], Counter, Counter]:
    supers: Set[str] = set()
    doms: Set[str] = set()
    pairs: Set[Tuple[str, str]] = set()
    super_counts = Counter()
    domain_counts = Counter()

    for q in questions:
        sd = (q.superdomain or "").strip()
        dm = (q.domain or "").strip()
        if sd:
            supers.add(sd)
            super_counts[sd] += 1
        if dm:
            doms.add(dm)
            domain_counts[dm] += 1
        if sd or dm:
            pairs.add((sd, dm))
    return supers, doms, pairs, super_counts, domain_counts


# --------------------- LLM Normalizer ---------------------


class TaxonomyLLM:
    def __init__(self, model_cfg: ModelConfig):
        self.model = get_model(model_cfg)

    def canonicalize_labels(
        self, labels: List[str], allowed_targets: List[str], label_type: str
    ) -> Dict[str, str]:
        if not labels:
            return {}

        system_prompt = (
            "You normalize taxonomy labels. Return STRICT JSON only.\n"
            "Goal: Map each RAW label to a CANONICAL label.\n"
            "Rules:\n"
            "1) Prefer an existing canonical label from ALREADY_EXISTING_CANONICAL_LABELS when it fits semantically.\n"
            "2) If none fit, propose a NEW canonical label that is GENERAL (broad category, not verbose or overly specific).\n"
            "3) Be consistent across labels (e.g., merge 'Entertainment and Media', 'Media & Entertainment' -> 'Entertainment').\n"
            "4) Keep it short (1–2 words), singular/plural consistently, and avoid punctuation unless necessary.\n"
            "5) When defining new canonical labels ensure that they are general enough to potentially fit multiple raw inputs.\n"
            "6) Do not create a new canonical label if an existing one is a good fit.\n"
            "7) Given your input you should strongly reduce the number of unique labels. Aim for at most 20 unique labels. You are allowed to generalize the existing raw labels to achieve this.\n"
            "Output JSON schema exactly:\n"
            "{\n"
            '  "mappings": { "<raw>": "<canonical>", ... },\n'
            '  "created": [ "<newly_created_label>", ... ]\n'
            "}\n"
        )

        user_payload = {
            "label_type": label_type,  # e.g., 'superdomain' or 'domain'
            "ALREADY_EXISTING_CANONICAL_LABELS": allowed_targets,  # preferred canonicals to reuse
            "raw_labels": labels,
        }
        user_prompt = (
            "Normalize RAW labels to canonical forms.\n\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
            + "\n\nRespond with the JSON object described above."
        )

        response = self.model.predict_string(user_prompt, system_prompt=system_prompt)
        parsed = _extract_json_from_llm(response) or {}
        mappings = parsed.get("mappings", {})
        out: Dict[str, str] = {}
        for raw in labels:
            can = mappings.get(raw)
            if isinstance(can, str) and can.strip():
                out[raw] = can.strip()
        return out


# --------------------- Pipeline ---------------------


def build_or_extend_mapping(
    run_paths: List[str],
    mapping_path: str,
    model_cfg: ModelConfig,
    batch_size: int = 50,
    seed_super_targets: Optional[List[str]] = None,
    seed_domain_targets: Optional[List[str]] = None,
) -> dict:
    questions = _load_all_questions_from_runs(run_paths)
    if not questions:
        print("No questions found in the provided runs.")
        mapping = _ensure_mapping_struct({})
        os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
        _atomic_write_json(mapping_path, mapping)
        return mapping

    supers, _, doms, super_counts, domain_counts = _collect_labels_and_pairs(questions)

    # Load existing mapping
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = _ensure_mapping_struct(json.load(f))
    else:
        mapping = _ensure_mapping_struct({})

    sd_map: Dict[str, str] = mapping["superdomain_map"]
    d_map: Dict[str, str] = mapping["domain_map"]

    # Also add the most frequent existing labels (bias toward reuse of common categories)

    # Figure out which raws are missing a mapping
    missing_sd = sorted([s for s in supers if s not in sd_map])
    missing_dm = sorted([d for d in doms if d not in d_map])

    llm = TaxonomyLLM(model_cfg)

    # Canonicalize superdomains (prefer allowed targets like 'Entertainment')
    for i in range(0, len(missing_sd), batch_size):
        batch = missing_sd[i : i + batch_size]
        result = llm.canonicalize_labels(batch, list(sd_map.values()), label_type="superdomain")
        sd_map.update(result)

    # Canonicalize domains
    # for i in range(0, len(missing_dm), batch_size):
    #     batch = missing_dm[i : i + batch_size]
    #     result = llm.canonicalize_labels(batch, list(d_map.values()), label_type="domain")
    #     d_map.update(result)

    # Persist
    mapping["superdomain_map"] = sd_map
    mapping["domain_map"] = d_map
    os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
    _atomic_write_json(mapping_path, mapping)
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Build/extend a static taxonomy mapping for superdomains/domains using an LLM."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="One or more run directories to scan for saved questions.",
    )
    parser.add_argument(
        "--mapping-file", default="mapping.json", help="Path to JSON mapping file to create/extend."
    )
    parser.add_argument(
        "--batch-size", type=int, default=200, help="Batch size for LLM normalization."
    )

    args = parser.parse_args()

    model_cfg = ModelConfig(
        name="gpt-5-mini-2025-08-07",
        provider="openai",
        max_workers=32,
        args={
            "max_output_tokens": 10000,
            "reasoning": {
                "effort": "low",
            },
            "text": {
                "verbosity": "low",
            },
        },
    )
    mapping = build_or_extend_mapping(
        run_paths=args.runs,
        mapping_path=args.mapping_file,
        model_cfg=model_cfg,
        batch_size=args.batch_size,
    )

    print(f"✅ Mapping written to {args.mapping_file}")
    print(f"  superdomain_map entries: {len(mapping['superdomain_map'])}")
    print(f"  domain_map entries:      {len(mapping['domain_map'])}")


if __name__ == "__main__":
    main()
