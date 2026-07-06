from pathlib import Path

from data_models import load_critique_entries


def main():
    critiques_dir = Path("critiques")
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                entries = load_critique_entries(crit_file)
                for idx, entry in enumerate(entries):
                    if not entry:
                        continue
                    attempts = entry.attempts or []
                    verdict = attempts[-1].verdict if attempts else None
                    if verdict in {None, "unknown"}:
                        print(
                            f"{mode_dir.name}/{q_dir.name}/{crit_file.stem} idx={idx} verdict={verdict} "
                            f"question_author={entry.question_author} critic={entry.critic}"
                        )


if __name__ == "__main__":
    main()
