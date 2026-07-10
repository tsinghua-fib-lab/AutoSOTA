from __future__ import annotations


def _latex_numeric_cells(values: list[float]) -> list[str]:
    avg = sum(values) / max(1, len(values))
    return [f"{100.0 * v:.2f}" for v in [*values, avg]]


def _latex_row_aligned(label: str, cells: list[str], *, label_width: int, col_widths: list[int]) -> str:
    padded = [f"{cell:<{col_widths[i]}}" for i, cell in enumerate(cells)]
    return f"{label:<{label_width}} : " + " & ".join(padded) + r" \\"


def print_latex_task_rows(per_task, merged_accs, norm_accs):
    task_cells = [str(item.get("task", "")) for item in per_task] + ["avg"]
    top1_cells = _latex_numeric_cells([float(v) for v in merged_accs])
    norm_cells = _latex_numeric_cells([float(v) for v in norm_accs])

    label_width = max(len("tasks"), len("top1"), len("norm"))
    col_widths = [max(len(task_cells[i]), len(top1_cells[i]), len(norm_cells[i])) for i in range(len(task_cells))]

    print("\nLaTeX rows (task order, then avg):")
    print(_latex_row_aligned("tasks", task_cells, label_width=label_width, col_widths=col_widths))
    print(_latex_row_aligned("top1", top1_cells, label_width=label_width, col_widths=col_widths))
    print(_latex_row_aligned("norm", norm_cells, label_width=label_width, col_widths=col_widths))


def pretty_print_task_accuracies(
    suite_name,
    method_name,
    peft_subspace,
    per_task,
    merged_accs,
    norm_accs,
    single_accs,
    *,
    baseline_label: str = "single",
    result_label: str = "top1",
):
    task_names = [item.get("task", "") for item in per_task]
    max_task_len = max((len(t) for t in task_names), default=4)
    task_col = max(max_task_len, len("task"))
    val_col = max(12, len(baseline_label) + 2, len(result_label) + 2)

    print(f"\nBenchmark {suite_name} - Method: {method_name} - Space: {peft_subspace}")
    print(f" {'task':<{task_col}}  {baseline_label:>{val_col}}  {result_label:>{val_col}}  {'norm':>{val_col}}")
    print(f" {'-' * task_col}  {'-' * val_col}  {'-' * val_col}  {'-' * val_col}")
    for i, item in enumerate(per_task):
        task = item.get("task", "")
        single = single_accs[i] if i < len(single_accs) else 0.0
        acc = merged_accs[i] if i < len(merged_accs) else 0.0
        norm = norm_accs[i] if i < len(norm_accs) else 0.0
        print(f" {task:<{task_col}}  {single:>{val_col}.6f}  {acc:>{val_col}.6f}  {norm:>{val_col}.6f}")

    avg_acc = sum(merged_accs) / max(1, len(merged_accs))
    avg_norm = sum(norm_accs) / max(1, len(norm_accs))
    avg_single = sum(single_accs) / max(1, len(single_accs))
    print(f" {'-' * task_col}  {'-' * val_col}  {'-' * val_col}  {'-' * val_col}")
    print(f" {'avg':<{task_col}}  {avg_single:>{val_col}.6f}  {avg_acc:>{val_col}.6f}  {avg_norm:>{val_col}.6f}")
