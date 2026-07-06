from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from analysis.figures_src.camera_ready.common import (
    FigureConfig,
    MetricSpec,
    PlotSpec,
    TASK_LABELS,
    dataset_plot_spec,
)


def build_config(bench_cfg, metric, task, output_prefix, include_legends=True, export_legend_only=False):
    """Return a FigureConfig for ablation plots."""
    dataset_cfgs = [
        entry[0] if isinstance(entry, tuple) else entry for entry in bench_cfg.datasets
    ]
    plots = [dataset_plot_spec(cfg, task, metric, output_prefix) for cfg in dataset_cfgs]
    return FigureConfig(
        plots=plots,
        base_width_in=6.8,
        base_height_in=3.8,
        min_font_size=10,
        method_label_fields=("kappa", "s"),
        method_label_bool_labels={},
        method_label_value_only_fields=(),
        task_label_map=TASK_LABELS,
        include_legends=include_legends,
        export_legend_only=export_legend_only,
    )
