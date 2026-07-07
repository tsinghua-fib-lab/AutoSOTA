from __future__ import annotations

from main import format_selected_benchmarks_message


def test_selected_benchmark_message_uses_actual_count() -> None:
    message = format_selected_benchmarks_message(
        [("CiteSeer", 0.48), ("Cornell", 0.28)],
        threshold=-0.9,
        min_top_s=1,
    )

    assert message.startswith("Selected 2 similar benchmarks")
    assert "min_top_s=1" in message
