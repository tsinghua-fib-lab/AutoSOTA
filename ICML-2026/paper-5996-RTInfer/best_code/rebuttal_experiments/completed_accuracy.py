from __future__ import annotations

from common import RebuttalModelSpec, make_model, make_variants, print_result_table, repeated_streams, run_policies


def main() -> None:
    model = make_model(
        RebuttalModelSpec(
            name="tight_deadline_yolov8l",
            input_shape=(3, 1080, 1920),
            num_exits=4,
            full_latency_ms=150.0,
            full_memory_mib=1700.0,
            full_accuracy=0.932,
            earliest_accuracy=0.420,
            num_blocks=12,
        )
    )
    models = {model.name: model}
    atlas = {model.name: make_variants(model)}
    tasks = repeated_streams(model.name, 4, deadline_ms=190, period_ms=190, duration_ms=950, shape=(3, 1080, 1920))
    results = run_policies(models, atlas, tasks, policies=("pantheon", "rtinfer"), duration_ms=950)
    print_result_table("accuracy metric decomposition: missed jobs vs completed-only", results)
    print("metric_note,deadline_weighted_acc_counts_missed_jobs_as_zero,completed_only_acc_excludes_missed_jobs")


if __name__ == "__main__":
    main()
