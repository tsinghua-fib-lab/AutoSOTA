from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import globals as g
from dataclasses import replace
from analysis.figures_src.camera_ready.common import MetricSpec, build_config
from bench.camera_ready.config_ridge_regression import CONFIG as BENCH_CONFIG


RESIDUAL_METRIC = MetricSpec(
    name="residual",
    label="Relative residual",
    higher_is_better=False,
    requires=["residual"],
)


CONFIG = replace(
    build_config(
        BENCH_CONFIG,
        RESIDUAL_METRIC,
        g.TASK_RIDGE_REGRESSION,
        "fig_e2e_camera_ready_ridge_regression",
    ),
    annotate_speedup=True,
    speedup_text_va="bottom",
    speedup_text_offset=1.0,
    use_auto_limits=True,
)
