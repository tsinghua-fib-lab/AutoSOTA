"""
Shared scaffolding for post-hoc LLM-judge analyzers.

Both ``artifact_quality`` and ``interactivity`` follow the same recipe:

1. Walk a results directory of conversation JSONs.
2. Dispatch a per-file LLM-judge call in parallel.
3. Aggregate the resulting scores by (assistant, artifact).
4. Print a banner and write a summary JSON.

The only things that differ are (a) what the judge does to a single file and
(b) how to bucket scores in the summary's distribution. We capture (a) as a
caller-provided ``process_one`` callable and (b) on an :class:`AnalyzerSpec`
dataclass; ``run_analyzer`` does the rest.
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from discoverllm.analyze._helpers import calculate_statistics, find_conversation_files
from discoverllm.simulate.logging_utils import close_error_log, init_error_log


# --------------------------------------------------------------------------- #
# Per-analyzer config                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class AnalyzerSpec:
    """How a specific analyzer differs from the shared scaffolding."""

    # Display name shown on the summary banner (e.g. "ARTIFACT QUALITY").
    display_name: str
    # Subdirectory under ``results_dir`` to write per-file outputs into.
    output_subdir: str
    # Filename of the rolled-up summary JSON dropped in that subdir.
    summary_filename: str
    # Buckets for the summary's ``score_distribution`` field. Each value is a
    # ``(low, high)`` inclusive range. Order is preserved in the output.
    score_buckets: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Extra fields to merge verbatim into the summary (e.g. ``evaluator_model``).
    extra_summary: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Per-file processor type                                                     #
# --------------------------------------------------------------------------- #
# Each analyzer wraps its own ``process_conversation_file`` in a closure that
# captures analyzer-specific kwargs (model, temperature, prompt, …). The
# closure has the signature below; ``run_analyzer`` calls it once per file.
ProcessOne = Callable[[Path, Path, Path], Tuple[Path, Optional[Dict[str, Any]]]]


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def run_analyzer(
    spec: AnalyzerSpec,
    results_dir: str,
    process_one: ProcessOne,
    parallel_workers: int = 4,
) -> Dict[str, Any]:
    """
    Run an LLM-judge analyzer over a results directory.

    Returns the summary dict (also written to disk).
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")

    output_dir = results_path / spec.output_subdir
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    error_log_path = init_error_log(results_dir)
    print(f"📝 Error log initialized: {error_log_path}")

    files = find_conversation_files(results_path)
    print(f"📁 Found {len(files)} conversation files")
    if not files:
        print("❌ No conversation files found")
        close_error_log()
        return {"error": "No conversation files found"}

    results, scores = [], []
    errors = skipped = 0

    print(f"🔄 Processing {len(files)} files with {parallel_workers} workers...")
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_file = {
            executor.submit(process_one, fp, results_path, output_dir): fp for fp in files
        }
        completed = 0
        for fut in as_completed(future_to_file):
            completed += 1
            fp = future_to_file[fut]
            tag = f"[{completed}/{len(files)}]"
            try:
                path, result = fut.result()
                if result is None:
                    errors += 1
                    print(f"❌ {tag} Error: {fp.name}")
                    continue
                if result.get("error"):
                    errors += 1
                    print(f"❌ {tag} Error: {fp.name} — {result.get('error')}")
                    continue
                score = result.get("score")
                if score is None:
                    skipped += 1
                    continue
                scores.append(score)
                results.append({"file": str(path), "score": score})
                print(f"✅ {tag} {fp.name}: score={score}")
            except Exception as e:
                errors += 1
                print(f"❌ {tag} Exception: {fp.name} — {e}")
                traceback.print_exc()

    summary = _build_summary(spec, files, scores, results, output_dir, errors, skipped)
    summary_path = output_dir / spec.summary_filename
    with open(summary_path, "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"\n📊 Summary saved to: {summary_path}")
    _print_summary_banner(spec, summary)

    close_error_log()
    return summary


# --------------------------------------------------------------------------- #
# Summary helpers                                                             #
# --------------------------------------------------------------------------- #
def _build_summary(
    spec: AnalyzerSpec,
    files: List[Path],
    scores: List[float],
    results: List[Dict[str, Any]],
    output_dir: Path,
    errors: int,
    skipped: int,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_files": len(files),
        "processed": len(scores),
        "skipped": skipped,
        "errors": errors,
        "output_directory": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        **spec.extra_summary,
    }
    if not scores:
        return summary

    summary["avg_score"] = sum(scores) / len(scores)
    summary["min_score"] = min(scores)
    summary["max_score"] = max(scores)
    if spec.score_buckets:
        summary["score_distribution"] = {
            label: sum(1 for s in scores if low <= s <= high)
            for label, (low, high) in spec.score_buckets.items()
        }
    stats = calculate_statistics(results, output_dir)
    summary["by_artifact"] = stats["by_artifact"]
    summary["by_assistant"] = stats["by_assistant"]
    # Kept for back-compat with old summary readers.
    summary["by_assistant_artifact"] = stats["assistant_artifact_stats"]
    return summary


def _print_summary_banner(spec: AnalyzerSpec, summary: Dict[str, Any]) -> None:
    rule = "=" * 50
    print(f"\n{rule}\n{spec.display_name} EVALUATION SUMMARY\n{rule}")
    print(f"Total files: {summary['total_files']}")
    print(f"Processed: {summary['processed']}")
    print(f"Skipped (already evaluated or missing score): {summary['skipped']}")
    print(f"Errors: {summary['errors']}")
    if "avg_score" in summary:
        print(f"Average score: {summary['avg_score']:.2f}")
        print(f"Score range: {summary['min_score']} - {summary['max_score']}")

    if summary.get("by_artifact"):
        print("\n" + "-" * 50)
        print("STATISTICS BY ARTIFACT")
        print("-" * 50)
        for artifact_id, assistants in summary["by_artifact"].items():
            print(f"\n{artifact_id}:")
            for assistant_id, st in assistants.items():
                print(
                    f"  {assistant_id}: avg={st['avg_score']:.3f}, "
                    f"std={st['std_score']:.3f}, trials={st['count']}"
                )

    if summary.get("by_assistant"):
        print("\n" + "-" * 50)
        print("STATISTICS BY ASSISTANT (across all artifacts)")
        print("-" * 50)
        for assistant_id, st in summary["by_assistant"].items():
            print(
                f"{assistant_id}: avg={st['avg_score']:.3f}, "
                f"std={st['std_score']:.3f}, artifacts={st['artifact_count']}"
            )
    print(rule)
