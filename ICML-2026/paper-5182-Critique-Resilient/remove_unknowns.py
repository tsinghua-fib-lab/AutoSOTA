from pathlib import Path

from data_models import (
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    save_answer_entries,
    save_benchmark_entries,
    save_critique_entries,
)


def last_verdict_attempt(attempts):
    if not attempts:
        return None
    last = attempts[-1]
    verdict = getattr(last, "verdict", None)
    if verdict:
        return verdict
    evaluation = getattr(last, "evaluation", None)
    if isinstance(evaluation, dict):
        return evaluation.get("verdict")
    return None


def clean_benchmarks(path: Path):
    for file in path.glob("*.json"):
        entries = load_benchmark_entries(file)
        new_entries = []
        for entry in entries:
            if entry is None:
                new_entries.append(None)
                continue
            gens = entry.generation_rounds or []
            if gens:
                refinements = gens[-1].refinement_rounds or []
                verdict = refinements[-1].evaluation.get("verdict") if refinements else None
                if verdict == "unknown":
                    continue
            new_entries.append(entry)
        if len(new_entries) != len(entries):
            save_benchmark_entries(file, new_entries)


def clean_answers(path: Path):
    for q_dir in path.glob("*"):
        for ans_file in q_dir.glob("*.json"):
            entries = load_answer_entries(ans_file)
            changed = False
            new_entries = []
            for entry in entries:
                if entry is None:
                    new_entries.append(None)
                    continue
                verdict = last_verdict_attempt(entry.attempts or [])
                if verdict == "unknown":
                    changed = True
                    continue
                new_entries.append(entry)
            if changed:
                save_answer_entries(ans_file, new_entries)


def clean_critiques(path: Path):
    for mode_dir in path.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                entries = load_critique_entries(crit_file)
                changed = False
                new_entries = []
                for entry in entries:
                    if entry is None:
                        new_entries.append(None)
                        continue
                    verdict = last_verdict_attempt(entry.attempts or [])
                    if verdict == "unknown":
                        changed = True
                        continue
                    new_entries.append(entry)
                if changed:
                    save_critique_entries(crit_file, new_entries)


def main():
    clean_benchmarks(Path("benchmarks"))
    clean_answers(Path("answers"))
    clean_critiques(Path("critiques"))
    print("Cleaned entries with final verdict 'unknown'.")


if __name__ == "__main__":
    main()
